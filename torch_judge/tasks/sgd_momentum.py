"""SGD with momentum task."""

TASK = {
    "title": "SGD with Momentum",
    "difficulty": "Medium",
    "function_name": "MySGD",
    "hint": "For each parameter, start from grad. Add weight_decay * p if requested. Update the momentum buffer as v = momentum * v + grad, then subtract lr * v from p inside torch.no_grad().",
    "tests": [
        {
            "name": "Single step without momentum",
            "code": """
import torch
p = torch.tensor([1.0, -2.0], requires_grad=True)
opt = {fn}([p], lr=0.1)
(p.pow(2).sum()).backward()
opt.step()
expected = torch.tensor([0.8, -1.6])
assert torch.allclose(p.detach(), expected, atol=1e-6), f'{p.detach()} vs {expected}'
""",
        },
        {
            "name": "Matches torch.optim.SGD with momentum and weight decay",
            "code": """
import torch
torch.manual_seed(0)
p = torch.randn(5, requires_grad=True)
p_ref = p.detach().clone().requires_grad_(True)
opt = {fn}([p], lr=0.05, momentum=0.9, weight_decay=0.01)
opt_ref = torch.optim.SGD([p_ref], lr=0.05, momentum=0.9, weight_decay=0.01)
for _ in range(5):
    loss = (p.pow(2).sum() + 0.1 * (p * torch.roll(p, shifts=1)).sum())
    loss_ref = (p_ref.pow(2).sum() + 0.1 * (p_ref * torch.roll(p_ref, shifts=1)).sum())
    loss.backward()
    loss_ref.backward()
    opt.step()
    opt_ref.step()
    opt.zero_grad()
    opt_ref.zero_grad()
assert torch.allclose(p.detach(), p_ref.detach(), atol=1e-6), f'{p.detach()} vs {p_ref.detach()}'
""",
        },
        {
            "name": "zero_grad clears gradients",
            "code": """
import torch
p = torch.randn(3, requires_grad=True)
opt = {fn}([p], lr=0.1, momentum=0.9)
p.sum().backward()
assert p.grad is not None, 'grad should exist before zero_grad'
opt.zero_grad()
assert p.grad is not None, 'zero_grad may zero in place, but should leave a tensor'
assert torch.equal(p.grad, torch.zeros_like(p)), 'grad should be all zeros after zero_grad'
""",
        },
        {
            "name": "Skips parameters with no grad",
            "code": """
import torch
p = torch.tensor([1.0], requires_grad=True)
q = torch.tensor([5.0], requires_grad=True)
opt = {fn}([p, q], lr=0.1, momentum=0.9)
(p * 2).sum().backward()
opt.step()
assert torch.allclose(q.detach(), torch.tensor([5.0])), 'Parameter with no grad should not change'
""",
        },
    ],
}
