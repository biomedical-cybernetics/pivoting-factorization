import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
from utils.linear_pivot import PivotLinear
from tqdm import tqdm

@torch.no_grad()
def replace_with_compress(model):
    device = torch.device("cuda:0")
    named_modules = list(model.named_modules())
    for name, module in tqdm(named_modules):
        if isinstance(module, SVDLinear):
            print(f"SVDLinear: {name}")
            module.to(device)
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            child_name = name.split(".")[-1]
            module.float()
            new_module = PivotLinear.from_svd_linear(module)
            new_module.half()
            new_module.cpu()
            setattr(parent, child_name, new_module)
            module.cpu()
            del module


class SVDLinear(nn.Module):
    def __init__(self, U, V, bias=None) -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = U.size(1)
        self.ALinear.weight.data = U
        self.BLinear.weight.data = V

    def forward(self, inp):
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y

@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i:i+1,:-1].to(model.device)
        labels = dataset[i:i+1,1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()

