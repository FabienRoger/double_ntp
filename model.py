import attrs
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch import nn


def get_identity_init_linear(dim: int):
    l = nn.Linear(dim, dim)
    l.weight.data.copy_(torch.eye(dim))
    l.bias.data.zero_()
    return l


def get_loss(lm_logits: torch.Tensor, labels: torch.Tensor):
    labels = labels.to(lm_logits.device)
    # we are doing next-token prediction; shift prediction scores and input ids by one
    shift_logits = lm_logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))


class GPTNeoXForDoubleCausalLM(torch.nn.Module):
    backbone: GPTNeoXForCausalLM
    embed_adapter: nn.Linear
    unembed_adapter: nn.Linear

    def __init__(self, backbone, embed_adapter, unembed_adapter):
        super().__init__()
        self.backbone = backbone
        self.embed_adapter = embed_adapter
        self.unembed_adapter = unembed_adapter

    def forward(self, input_ids_left: torch.Tensor, input_ids_right: torch.Tensor):
        embeds_left = self.backbone.gpt_neox.embed_in(input_ids_left)
        embeds_right = self.embed_adapter(self.backbone.gpt_neox.embed_in(input_ids_right))

        inputs_embeds = embeds_left + embeds_right
        outputs = self.backbone.gpt_neox(inputs_embeds=inputs_embeds)[0]

        loss_left = get_loss(self.backbone.embed_out(outputs), input_ids_left)
        loss_right = get_loss(self.backbone.embed_out(self.unembed_adapter(outputs)), input_ids_right)

        return loss_left, loss_right

    @classmethod
    def from_model(cls, model: GPTNeoXForCausalLM):
        return cls(
            backbone=model,
            embed_adapter=get_identity_init_linear(model.config.hidden_size),
            unembed_adapter=get_identity_init_linear(model.config.hidden_size),
        )

    @classmethod
    def from_name(cls, name: str):
        model = GPTNeoXForCausalLM.from_pretrained(name, revision="step0")
        return cls.from_model(model)


if __name__ == "__main__":
    from lion_pytorch import Lion

    model = GPTNeoXForDoubleCausalLM.from_name("EleutherAI/pythia-70m")
    # print(list(model.named_parameters()))
    print(*[(n, p.dtype) for n, p in model.named_parameters()], sep="\n")
