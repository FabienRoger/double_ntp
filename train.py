from itertools import islice
import json
import os
from pathlib import Path
import time
import traceback
from typing import Callable, Generator, Literal, Optional, TypeVar
import attrs
from datasets import load_dataset
import random

import ray
from tqdm import tqdm
from ray.experimental.tqdm_ray import tqdm as ray_tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StableLmForCausalLM, GPTNeoXForCausalLM

from model import GPTNeoXForDoubleCausalLM
from lion_pytorch import Lion

VERSION = "v1.4"

# Note: I assume datasets are shuffled

T = TypeVar("T")


def cache_pure_gen(get_gen: Callable[[], Generator[T, None, None]]) -> Callable[[], Generator[T, None, None]]:
    def new_get_gen():
        name = get_gen.__name__
        cache_file = f".cache/{name}.jsonl"

        if not os.path.exists(cache_file):
            os.makedirs(".cache", exist_ok=True)
            Path(cache_file).write_text("")
            print(f"Cache file {cache_file} created")

        lines_loaded = 0
        lineset = set()
        with open(cache_file) as f:
            for line in f:
                if line and line not in lineset:
                    lineset.add(line)
                    yield json.loads(line)
                    lines_loaded += 1

        print(f"Loaded {lines_loaded} lines from cache file {cache_file}")

        gen = get_gen()
        for _ in range(lines_loaded):
            next(gen)

        with open(cache_file, "a") as f:
            for item in gen:
                f.write(json.dumps(item) + "\n")
                yield item

    return new_get_gen


@cache_pure_gen
def get_text_gen() -> Generator[str, None, None]:
    print("Loading text data")
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)

    for i, row in enumerate(data):
        if i % 100 == 0:
            print(f"Text {i}")
        yield row["text"]

    raise StopIteration()


@cache_pure_gen
def get_code_gen() -> Generator[str, None, None]:
    print("Loading code data")
    data = load_dataset("codeparrot/github-code", split="train", streaming=True)

    for i, row in enumerate(data):
        if i % 100 == 0:
            print(f"Code {i}")
        yield row["code"]

    raise StopIteration()


def get_gen_mix(
    a_gen: Generator[T, None, None], b_gen: Generator[T, None, None], p_b: float
) -> Generator[T, None, None]:
    """Return a balanced mixed. c4 text is ~3x smaller than codeparrot code"""
    rng = random.Random(0)

    while True:
        take_b = rng.random() < p_b
        if take_b:
            yield next(b_gen)
        else:
            yield next(a_gen)


def get_tokens(
    s_gen: Generator[str, None, None],
    tokenizer: AutoTokenizer,
    seq_len: int,
    tokenization_batch_size: int = 64,
) -> Generator[list[int], None, None]:
    eos_token = tokenizer.eos_token_id
    current = []
    # for s in s_gen:
    #     current += tokenizer.encode(s, add_special_tokens=False) + [eos_token]
    #     if len(current) >= seq_len:
    #         yield current[:seq_len]
    #         current = current[seq_len : 2 * seq_len]  # intentionally don't keep more than 2 fragments
    # raise StopIteration()
    while True:
        new_sequences = []
        for _ in range(tokenization_batch_size):
            new_sequences.append(next(s_gen))
        tokenized = tokenizer(new_sequences, add_special_tokens=False)["input_ids"]
        for toks in tokenized:
            toks.append(eos_token)
            current += toks
            if len(current) >= seq_len:
                yield current[:seq_len]
                current = current[seq_len : 2 * seq_len]


def get_eos_tokens(tokenizer: AutoTokenizer, seq_len: int) -> Generator[list[int], None, None]:
    eos_token = tokenizer.eos_token_id
    while True:
        yield [eos_token] * seq_len


def get_batched(
    tok_gen: Generator[list[int], None, None],
    batch_size: int,
    buffer_size: int = 10_000,
) -> Generator[list[list[int]], None, None]:
    rng = random.Random(0)

    toks = []
    for t in tok_gen:
        toks.append(t)
        if len(toks) == buffer_size:
            sampled_idxs = set(rng.sample(range(len(toks)), batch_size))
            yield [toks[i] for i in sampled_idxs]
            toks = [t for i, t in enumerate(toks) if i not in sampled_idxs]

    raise StopIteration()


data_kinds = ["text", "code", "pad"]
DataKind = Literal["text", "code", "pad"]


def to_torch(tokens: list[list[int]]) -> torch.Tensor:
    return torch.tensor(tokens, dtype=torch.long).cuda()


log_file = "log.txt"
Path(log_file).write_text("")


@ray.remote(num_gpus=1)
def train(
    run_name: str,
    train_batches: int = 20_000,
    train_batch_size: int = 100,
    val_batch_size: int = 100,
    val_batches: int = 1,
    seq_len: int = 512,
    lr: float = 1e-4,
    eval_every: int = 20,
    right_frac: float = 0.1,
    left_frac: float = 0.9,
    model_name: str = "EleutherAI/pythia-70m",
    warmup_steps: int = 200,
    version: str = VERSION,
):
    config = locals().copy()

    double_ntp = GPTNeoXForDoubleCausalLM.from_name(model_name).cuda()
    optimizer = Lion(double_ntp.parameters(), lr=lr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_gen = get_text_gen()
    code_gen = get_code_gen()
    eos_token_gen = get_eos_tokens(tokenizer, seq_len)

    val_tokens = {
        "text": list(islice(get_tokens(text_gen, tokenizer, seq_len), val_batch_size * val_batches)),
        "code": list(islice(get_tokens(code_gen, tokenizer, seq_len), val_batch_size * val_batches)),
        "pad": list(islice(eos_token_gen, val_batch_size * val_batches)),
    }

    @torch.no_grad
    def get_val_ntps(left_kind: DataKind, right_kind: DataKind) -> torch.Tensor:
        val_tokens_left = val_tokens[left_kind]
        val_tokens_right = val_tokens[right_kind]
        left_losses = []
        right_losses = []
        for i in range(val_batches):
            left_batch = val_tokens_left[i * val_batch_size : (i + 1) * val_batch_size]
            right_batch = val_tokens_right[i * val_batch_size : (i + 1) * val_batch_size]
            loss_left, loss_right = double_ntp.get_ntp_losses(to_torch(left_batch), to_torch(right_batch))
            right_losses.append(loss_right.item())
            left_losses.append(loss_left.item())
        return sum(left_losses) / val_batches, sum(right_losses) / val_batches

    mix_gen = get_gen_mix(text_gen, code_gen, p_b=0.25)  # code is 3x larger than text

    # separate get_tokens since there is overlap between tokens
    left_gen = get_gen_mix(eos_token_gen, get_tokens(mix_gen, tokenizer, seq_len), p_b=left_frac)
    right_gen = get_gen_mix(eos_token_gen, get_tokens(mix_gen, tokenizer, seq_len), p_b=right_frac)
    train_gen = zip(get_batched(left_gen, train_batch_size), get_batched(right_gen, train_batch_size))

    import wandb

    mode = "online"
    # mode = "offline"
    wandb.init(project="double_ntp", name=run_name, config=config, mode=mode)

    st = time.time()
    for i, (batch_left, batch_right) in enumerate(ray_tqdm(train_gen, total=train_batches)):
        try:
            data_time = time.time() - st
            st = time.time()

            # adjust lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * min(1, (i + 1) / warmup_steps)

            stats = {"t": i, "lr": optimizer.param_groups[0]["lr"]}

            if i % eval_every == 0:
                all_losses_left = []
                all_losses_right = []
                for left_kind in data_kinds:
                    for right_kind in data_kinds:
                        if left_kind == right_kind == "pad":
                            continue

                        left_loss, right_loss = get_val_ntps(left_kind, right_kind)

                        stats[f"right_{left_kind}_{right_kind}"] = right_loss
                        stats[f"left_{left_kind}_{right_kind}"] = left_loss

                        if left_kind != "pad":
                            all_losses_left.append(left_loss)
                        if right_kind != "pad":
                            all_losses_right.append(right_loss)
                stats["right_avg"] = sum(all_losses_right) / len(all_losses_right)
                stats["left_avg"] = sum(all_losses_left) / len(all_losses_left)
                stats["avg"] = (stats["right_avg"] + stats["left_avg"]) / 2

            eval_time = time.time() - st
            st = time.time()

            optimizer.zero_grad()

            left_loss, right_loss = double_ntp.get_ntp_losses(to_torch(batch_left), to_torch(batch_right))
            loss = left_loss + right_loss
            loss.backward()
            optimizer.step()

            train_time = time.time() - st
            st = time.time()

            stats["train_loss"] = loss.item()
            stats["train_left_loss"] = left_loss.item()
            stats["train_right_loss"] = right_loss.item()
            stats["data_time"] = data_time
            stats["eval_time"] = eval_time
            stats["train_time"] = train_time

            wandb.log(stats)

        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"Exception: {e}\n")
                f.write(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # train("test")
    ray.init()

    lrs = [6e-5]
    # right_fracs = [0.01, 0.1, 1, 0]
    right_fracs = [0, 0.1, 0.9]

    deps = [
        train.options(name=f"l{lr}_f{right_frac}_{VERSION}").remote(
            f"l{lr}_f{right_frac}_{VERSION}", lr=lr, right_frac=right_frac
        )
        for lr in lrs
        for right_frac in right_fracs
    ]

    for dep in deps:
        ray.get(dep)
