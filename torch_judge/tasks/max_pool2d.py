"""MaxPool2d task."""

TASK = {
    "title": "MaxPool2d",
    "difficulty": "Medium",
    "function_name": "my_max_pool2d",
    "hint": "Convert kernel_size/stride/padding to pairs. Pad with -inf, extract sliding windows with unfold or loops, then take max over each kH x kW window.",
    "tests": [
        {
            "name": "Matches F.max_pool2d on a simple tensor",
            "code": """
import torch
import torch.nn.functional as F
x = torch.arange(1.0, 17.0).view(1, 1, 4, 4)
out = {fn}(x, kernel_size=2)
expected = F.max_pool2d(x, kernel_size=2)
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.equal(out, expected), f'{out} vs {expected}'
""",
        },
        {
            "name": "Supports stride and padding",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
x = torch.randn(2, 3, 5, 6)
out = {fn}(x, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
expected = F.max_pool2d(x, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
assert out.shape == expected.shape, f'Shape mismatch: {out.shape} vs {expected.shape}'
assert torch.allclose(out, expected, atol=1e-6), 'Values differ from F.max_pool2d'
""",
        },
        {
            "name": "Handles multi-channel batches",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(1)
x = torch.randn(4, 5, 8, 7)
out = {fn}(x, kernel_size=3, stride=2, padding=1)
expected = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
assert out.shape == (4, 5, 4, 4), f'Unexpected output shape: {out.shape}'
assert torch.allclose(out, expected, atol=1e-6), 'Batch/channel pooling is incorrect'
""",
        },
        {
            "name": "Gradient flows to selected maxima",
            "code": """
import torch
x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
out = {fn}(x, kernel_size=2)
out.sum().backward()
expected_grad = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]])
assert x.grad is not None, 'x.grad is None'
assert torch.equal(x.grad, expected_grad), f'{x.grad} vs {expected_grad}'
""",
        },
    ],
}
