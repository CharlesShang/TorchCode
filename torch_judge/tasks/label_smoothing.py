"""Label smoothing cross-entropy task."""

TASK = {
    "title": "Label Smoothing Loss",
    "difficulty": "Medium",
    "function_name": "label_smoothing_loss",
    "hint": "Compute log_probs with logsumexp. The smoothed loss is (1 - eps) * NLL + eps * mean(-log_probs over classes), then apply the requested reduction.",
    "tests": [
        {
            "name": "smoothing=0 matches cross_entropy",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
logits = torch.randn(6, 10)
targets = torch.randint(0, 10, (6,))
out = {fn}(logits, targets, smoothing=0.0)
expected = F.cross_entropy(logits, targets)
assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'
""",
        },
        {
            "name": "Matches manual label smoothing formula",
            "code": """
import torch
logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]])
targets = torch.tensor([0, 2])
eps = 0.2
log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
nll = -log_probs[torch.arange(targets.numel()), targets]
smooth = -log_probs.mean(dim=-1)
expected = ((1 - eps) * nll + eps * smooth).mean()
out = {fn}(logits, targets, smoothing=eps)
assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'
""",
        },
        {
            "name": "Supports none and sum reductions",
            "code": """
import torch
torch.manual_seed(1)
logits = torch.randn(4, 5)
targets = torch.tensor([0, 1, 3, 4])
losses = {fn}(logits, targets, smoothing=0.1, reduction='none')
summed = {fn}(logits, targets, smoothing=0.1, reduction='sum')
assert losses.shape == (4,), f'Expected per-example losses, got {losses.shape}'
assert torch.allclose(losses.sum(), summed, atol=1e-6), 'sum reduction must equal losses.sum()'
""",
        },
        {
            "name": "Numerically stable and differentiable",
            "code": """
import torch
logits = torch.tensor([[1000.0, 999.0, -1000.0]], requires_grad=True)
targets = torch.tensor([1])
loss = {fn}(logits, targets, smoothing=0.1)
assert torch.isfinite(loss), 'Loss should be finite for large logits'
loss.backward()
assert logits.grad is not None, 'logits.grad is None'
assert torch.isfinite(logits.grad).all(), 'Gradient contains NaN or Inf'
""",
        },
    ],
}
