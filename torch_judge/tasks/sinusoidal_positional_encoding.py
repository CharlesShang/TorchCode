"""Sinusoidal positional encoding task."""

TASK = {
    "title": "Sinusoidal Positional Encoding",
    "difficulty": "Medium",
    "function_name": "sinusoidal_positional_encoding",
    "hint": "For even columns use sin(pos / base^(i / dim)); for odd columns use cos(pos / base^(i / dim)). Build position as (seq_len, 1) and frequencies as (ceil(dim/2),).",
    "tests": [
        {
            "name": "Shape, dtype, and first row",
            "code": """
import torch
pe = {fn}(seq_len=5, dim=7, dtype=torch.float64)
assert pe.shape == (5, 7), f'Unexpected shape: {pe.shape}'
assert pe.dtype == torch.float64, f'Unexpected dtype: {pe.dtype}'
assert torch.allclose(pe[0, 0::2], torch.zeros_like(pe[0, 0::2])), 'sin columns at position 0 should be 0'
assert torch.allclose(pe[0, 1::2], torch.ones_like(pe[0, 1::2])), 'cos columns at position 0 should be 1'
""",
        },
        {
            "name": "Matches reference formula",
            "code": """
import torch
import math
seq_len, dim = 4, 6
pe = {fn}(seq_len, dim, base=10000.0)
position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
expected = torch.zeros(seq_len, dim)
expected[:, 0::2] = torch.sin(position * div_term)
expected[:, 1::2] = torch.cos(position * div_term)
assert torch.allclose(pe, expected, atol=1e-6), 'Values do not match sinusoidal formula'
""",
        },
        {
            "name": "Odd dimensions are supported",
            "code": """
import torch
pe = {fn}(3, 5)
assert pe.shape == (3, 5), f'Unexpected shape: {pe.shape}'
assert torch.isfinite(pe).all(), 'Encoding contains NaN or Inf'
assert not torch.allclose(pe[:, -1], torch.zeros_like(pe[:, -1])), 'Last odd column should be populated'
""",
        },
        {
            "name": "Deterministic and non-trainable",
            "code": """
import torch
pe1 = {fn}(8, 8)
pe2 = {fn}(8, 8)
assert torch.equal(pe1, pe2), 'Encoding should be deterministic'
assert not pe1.requires_grad, 'Positional encoding should not require grad by default'
""",
        },
    ],
}
