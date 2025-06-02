import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
import math

FACTOR = 16  # Round up the shape to the nearest multiple of 16

class PivotLinear(nn.Module):
    def __init__(self, B, A, index_array, d_out, bias=None, inplace=False) -> None:
        super().__init__()
        self.ALinear = nn.Linear(A.size(1), A.size(0), bias=False)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)
        self.BLinear = nn.Linear(B.size(1), B.size(0), bias=False)
        self.ALinear.weight.data = A.contiguous()
        self.BLinear.weight.data = B.contiguous()
        index_tensor = torch.tensor(index_array, dtype=torch.long)
        self.d_out = d_out
        mask = torch.ones(self.d_out, dtype=torch.bool)
        mask[index_tensor] = False
        self.register_buffer('true_idx', torch.nonzero(mask, as_tuple=False).squeeze(-1))
        self.register_buffer('false_idx', torch.nonzero(~mask, as_tuple=False).squeeze(-1))
        self.inplace = inplace


    @staticmethod
    def factorization(weight, rank):
        # Perform QR decomposition with column pivoting
        Q, R, P = scipy.linalg.qr(weight.cpu().detach().numpy().astype(np.float32), pivoting=True)
        pivot_columns = P[:rank]
        pivot_columns = np.sort(pivot_columns)
        left = weight[:, pivot_columns]
        right = torch.linalg.lstsq(left, weight).solution
        return pivot_columns, left, right

    @staticmethod
    def from_svd_linear(svdLinear):
        rank = svdLinear.ALinear.weight.shape[1]
        weight = svdLinear.ALinear.weight @ svdLinear.BLinear.weight
        bias = svdLinear.ALinear.bias
        rank = math.ceil(rank / FACTOR) * FACTOR
        return PivotLinear.from_linear(weight, bias, rank)

    @staticmethod
    def from_linear(
            weight: torch.Tensor,
            bias: torch.Tensor,
            rank,
    ):
        # weight_np = weight.cpu().detach().numpy().astype(np.float32)
        d_out, d_in = weight.shape
        pivot_rows, left, right = PivotLinear.factorization(weight.T, rank)
        A = right.T
        B = left.T
        mask = np.ones(d_out, dtype=bool)
        mask[pivot_rows] = False
        A = A[mask]
        new_linear = PivotLinear(B.to(weight.dtype), A.to(weight.dtype), pivot_rows, d_out, bias)

        return new_linear

    def forward(self, inp):
        if self.inplace and self.d_out == inp.shape[-1]:  # square
            output = inp
        else:
            output = torch.empty([*inp.shape[:-1], self.d_out], dtype=inp.dtype, device=inp.device)
        y = self.BLinear(inp)
        output.index_copy_(-1, self.false_idx, y)
        y = self.ALinear(y)
        output.index_copy_(-1, self.true_idx, y)
        return output