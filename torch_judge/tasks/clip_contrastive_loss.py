"""CLIP-style contrastive loss task."""

TASK = {
    "title": "CLIP Contrastive Loss",
    "difficulty": "Medium",
    "function_name": "clip_contrastive_loss",
    "hint": "L2-normalize both embedding matrices, build logits = image @ text.T / temperature, then average cross-entropy in both directions using labels arange(batch_size).",
    "tests": [
        {
            "name": "Matches PyTorch reference",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(0)
image = torch.randn(6, 8)
text = torch.randn(6, 8)
temperature = 0.2
image_n = F.normalize(image, dim=-1)
text_n = F.normalize(text, dim=-1)
logits = image_n @ text_n.T / temperature
labels = torch.arange(image.shape[0])
expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
out = {fn}(image, text, temperature=temperature)
assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'
""",
        },
        {
            "name": "Matched pairs beat shuffled pairs",
            "code": """
import torch
emb = torch.eye(5)
good = {fn}(emb, emb, temperature=0.1)
bad = {fn}(emb, torch.roll(emb, shifts=1, dims=0), temperature=0.1)
assert good < bad, f'Matched loss {good} should be lower than shuffled loss {bad}'
""",
        },
        {
            "name": "Gradient flows to both towers",
            "code": """
import torch
torch.manual_seed(1)
image = torch.randn(4, 6, requires_grad=True)
text = torch.randn(4, 6, requires_grad=True)
loss = {fn}(image, text)
loss.backward()
assert image.grad is not None and text.grad is not None, 'Both inputs need gradients'
assert torch.isfinite(image.grad).all() and torch.isfinite(text.grad).all(), 'Gradients contain NaN or Inf'
""",
        },
        {
            "name": "Rejects mismatched batches",
            "code": """
import torch
try:
    {fn}(torch.randn(3, 4), torch.randn(2, 4))
except (AssertionError, ValueError):
    pass
else:
    raise AssertionError('Should reject different batch sizes')
""",
        },
    ],
}
