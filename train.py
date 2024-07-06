from itertools import islice
import itertools
import json
from math import ceil
import os
from pathlib import Path
import time
import traceback
from typing import Callable, Generator, Literal, Optional, TypeVar
import attrs
from datasets import load_dataset
import random

import numpy as np
import ray
from tqdm import tqdm
from ray.experimental.tqdm_ray import tqdm as ray_tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StableLmForCausalLM, GPTNeoXForCausalLM

from model import GPTNeoXForDoubleCausalLM
from lion_pytorch import Lion

VERSION = "v2.3"

# Note: I assume datasets are shuffled

T = TypeVar("T")


def get_text_gen() -> Generator[str, None, None]:
    print("Loading text data")
    data = load_dataset("allenai/c4", "en", split="train", streaming=True)

    for i, row in enumerate(data):
        if i % 100 == 0:
            print(f"Text {i}")
        yield row["text"]


def get_code_gen() -> Generator[str, None, None]:
    print("Loading code data")
    data = load_dataset("codeparrot/github-code", split="train", streaming=True)

    for i, row in enumerate(data):
        if i % 100 == 0:
            print(f"Code {i}")
        yield row["code"]


def get_gen_mix(
    a_gen: Generator[T, None, None], b_gen: Generator[T, None, None], p_b: float
) -> Generator[T, None, None]:
    rng = random.Random(0)

    while True:
        take_b = rng.random() < p_b
        if take_b:
            yield next(b_gen)
        else:
            yield next(a_gen)


def get_gen_multi_mix(
    a_gen_left: Generator[T, None, None],
    b_gen_left: Generator[T, None, None],
    a_gen_right: Generator[T, None, None],
    b_gen_right: Generator[T, None, None],
    p_aa: float,
    p_ab: float,
    p_ba: float,
    p_bb: float,
) -> Generator[tuple[T, T], None, None]:
    rng = random.Random(0)

    assert p_aa + p_ab + p_ba + p_bb == 1

    while True:
        try:
            u = rng.random()
            if u < p_aa:
                yield next(a_gen_left), next(a_gen_right)
            elif u < p_aa + p_ab:
                yield next(a_gen_left), next(b_gen_right)
            elif u < p_aa + p_ab + p_ba:
                yield next(b_gen_left), next(a_gen_right)
            else:
                yield next(b_gen_left), next(b_gen_right)
        except StopIteration:
            return


def get_tokens(
    s_gen: Generator[str, None, None],
    tokenizer: AutoTokenizer,
    seq_len: int,
    tokenization_batch_size: int = 64,
) -> Generator[list[int], None, None]:
    eos_token = tokenizer.eos_token_id
    current = []

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
    tok_gen: Generator[T, None, None],
    batch_size: int,
    buffer_size: int = 10_000,
) -> Generator[list[T], None, None]:
    rng = random.Random(0)

    toks = []
    for t in tok_gen:
        toks.append(t)
        if len(toks) == buffer_size:
            sampled_idxs = set(rng.sample(range(len(toks)), batch_size))
            yield [toks[i] for i in sampled_idxs]
            toks = [t for i, t in enumerate(toks) if i not in sampled_idxs]


data_kinds = ["text", "code", "pad"]
DataKind = Literal["text", "code", "pad"]


def to_torch(tokens: list[list[int]]) -> torch.Tensor:
    return torch.tensor(np.array(tokens), dtype=torch.long).cuda()


log_file = "log.txt"
Path(log_file).write_text("")


def get_cached_toks(
    gen: Generator[T, None, None], n: int, name: str, entries_per_file: int = 10_000
) -> Generator[T, None, None]:
    save_folder = f".cache/{name}_{n}"

    if os.path.exists(save_folder):
        for i in itertools.count():
            if not os.path.exists(f"{save_folder}/{i}.npy"):
                return
            print(f"Loading {name} {i}")
            array = np.load(f"{save_folder}/{i}.npy")
            for item in array:
                yield item
        return

    print(f"Cache {name} not found, generating")
    os.makedirs(save_folder, exist_ok=True)
    c = 0
    items = []
    for i in range(n):
        try:
            item = next(gen)
        except StopIteration:
            raise ValueError(f"Not enough items {i} < {n} for {name} generator")

        items.append(item)
        if len(items) == entries_per_file:
            np.save(f"{save_folder}/{c}.npy", np.array(items))
            c += 1
            items = []
        yield item
    if items:
        np.save(f"{save_folder}/{c}.npy", np.array(items))


@ray.remote(num_gpus=1)
def train(
    run_name: str,
    train_seqs: int = 4_000_000,
    val_seqs: int = 256,
    train_batch_size: int = 100,
    val_batch_size: int = 100,
    seq_len: int = 512,
    lr: float = 1e-4,
    eval_ratio: int = 20,  # how many batches of eval per batch of train
    right_frac: float = 0.1,
    no_left_frac: float = 1 / 3,
    model_name: str = "EleutherAI/pythia-70m",
    warmup_steps: int = 200,
    max_norm: float = 1.0,
    only_toks: bool = False,
    dry_batches: bool = False,
    version: str = VERSION,
):
    config = locals().copy()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_gen = get_text_gen()
    code_gen = get_code_gen()
    eos_token_gen = get_eos_tokens(tokenizer, seq_len)

    val_tokens = {
        "text": list(get_cached_toks(get_tokens(text_gen, tokenizer, seq_len), val_seqs, "val_text")),
        "code": list(get_cached_toks(get_tokens(code_gen, tokenizer, seq_len), val_seqs, "val_code")),
        "pad": list(islice(eos_token_gen, val_seqs)),
    }

    mix_gen = get_gen_mix(text_gen, code_gen, p_b=0.25)  # code is 3x larger than text

    if only_toks:
        for _ in ray_tqdm(
            get_cached_toks(get_tokens(mix_gen, tokenizer, seq_len), train_seqs, "train_left"), total=train_seqs
        ):
            pass
        for _ in ray_tqdm(
            get_cached_toks(get_tokens(mix_gen, tokenizer, seq_len), train_seqs, "train_right"), total=train_seqs
        ):
            pass
        return

    # separate get_tokens since there is overlap between tokens
    train_gen = get_gen_multi_mix(
        get_cached_toks(get_tokens(mix_gen, tokenizer, seq_len), train_seqs, "train_left"),
        eos_token_gen,
        get_cached_toks(get_tokens(mix_gen, tokenizer, seq_len), train_seqs, "train_right"),
        eos_token_gen,
        p_aa=right_frac * (1 - no_left_frac),
        p_ab=1 - right_frac,
        p_ba=right_frac * no_left_frac,
        p_bb=0,
    )
    batched = get_batched(train_gen, train_batch_size)

    if dry_batches:
        for name, toks in val_tokens.items():
            print(f"Val {name}: {len(toks)}")
        c = 0
        for _ in ray_tqdm(batched, total=train_seqs // train_batch_size):
            c += 1
        print(f"Generated {c} batches")
        return

    double_ntp = GPTNeoXForDoubleCausalLM.from_name(model_name).cuda()
    torch.set_float32_matmul_precision("high")
    double_ntp = torch.compile(double_ntp)
    print("Model loaded")

    optimizer = Lion(double_ntp.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    eval_every = ceil(eval_ratio * val_seqs / val_batch_size)

    @torch.no_grad
    def get_val_ntps(left_kind: DataKind, right_kind: DataKind) -> torch.Tensor:
        val_tokens_left = val_tokens[left_kind]
        val_tokens_right = val_tokens[right_kind]
        left_losses = []
        right_losses = []
        val_batches = ceil(len(val_tokens_left) / val_batch_size)
        for i in range(val_batches):
            left_batch = val_tokens_left[i * val_batch_size : (i + 1) * val_batch_size]
            right_batch = val_tokens_right[i * val_batch_size : (i + 1) * val_batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_left, loss_right = double_ntp(to_torch(left_batch), to_torch(right_batch))
            right_losses.append(loss_right.item() * len(right_batch))
            left_losses.append(loss_left.item() * len(left_batch))
        return sum(left_losses) / len(val_tokens_left), sum(right_losses) / len(val_tokens_right)

    import wandb

    mode = "online"
    # mode = "offline"
    wandb.init(project="double_ntp", name=run_name, config=config, mode=mode)

    st = time.time()
    pbar = ray_tqdm(batched, total=train_seqs // train_batch_size)
    for i, batch in enumerate(pbar):
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

                        if left_kind != "pad":
                            stats[f"left_{left_kind}_{right_kind}"] = left_loss
                            all_losses_left.append(left_loss)
                        if right_kind != "pad":
                            stats[f"right_{left_kind}_{right_kind}"] = right_loss
                            all_losses_right.append(right_loss)

                stats["right_avg"] = sum(all_losses_right) / len(all_losses_right)
                stats["left_avg"] = sum(all_losses_left) / len(all_losses_left)
                stats["avg"] = (stats["right_avg"] + stats["left_avg"]) / 2

            eval_time = time.time() - st
            st = time.time()

            optimizer.zero_grad()

            batch_left, right_batch = zip(*batch)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                left_loss, right_loss = double_ntp(to_torch(batch_left), to_torch(right_batch))
                loss = left_loss + right_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(double_ntp.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            train_time = time.time() - st
            st = time.time()

            stats["train_loss"] = loss.item()
            stats["train_left_loss"] = left_loss.item()
            stats["train_right_loss"] = right_loss.item()
            stats["data_time"] = data_time
            stats["eval_time"] = eval_time
            stats["train_time"] = train_time

            pbar.set_description(f"loss={loss.item():.3f}")

            # wandb.finish()
            # return

            wandb.log(stats)

        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"Exception: {e}\n")
                f.write(traceback.format_exc())
            raise e


if __name__ == "__main__":
    # train("test", only_toks=True, train_seqs=512)
    # train("test", dry_batches=True, train_seqs=512)
    # train("test", only_toks=True)
    # train("test", dry_batches=True)
    # train("test", model_name="EleutherAI/pythia-70m", train_batch_size=110)
    # train("test", model_name="EleutherAI/pythia-160m", train_batch_size=70)
    # train("test", model_name="EleutherAI/pythia-410m", train_batch_size=40)
    # train("test", dry_batches=True)
    ray.init()

    lrs = [6e-5, 3e-5]
    # right_fracs = [0.01, 0.1, 1, 0]
    right_fracs = [0.1, 0, 0.01, 0.9]
    models = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
    model_shorthands = [m.split("-")[-1] for m in models]
    batch_sizes = [110, 70, 40]

    deps = [
        train.options(name=f"{short}_l{lr}_f{right_frac}_{VERSION}").remote(
            f"{short}_l{lr}_f{right_frac}_{VERSION}",
            lr=lr,
            right_frac=right_frac,
            model_name=model_name,
            train_batch_size=batch_size,
            val_batch_size=batch_size,
        )
        for lr in lrs
        for right_frac in right_fracs
        for short, model_name, batch_size in zip(model_shorthands, models, batch_sizes)
    ]

    for dep in deps:
        ray.get(dep)
