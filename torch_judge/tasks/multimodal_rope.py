"""Multimodal Rotary Position Embedding (M-RoPE) task."""

TASK = {
    "title": "Multimodal RoPE (M-RoPE)",
    "difficulty": "Hard",
    "function_name": "apply_multimodal_rope",
    "hint": (
        "Treat adjacent channels as rotary pairs. Build one inverse frequency per pair, "
        "assign consecutive pair sections to temporal, height, and width positions, then "
        "rotate Q and K. When all three position IDs are equal, the result must reduce to "
        "ordinary 1D RoPE."
    ),
    "tests": [
        {
            "name": "Shapes, dtypes, and norms",
            "code": """
import torch
torch.manual_seed(0)
q = torch.randn(2, 4, 7, 12)
k = torch.randn(2, 2, 7, 12)
position_ids = torch.randint(0, 5, (3, 2, 7))
q_rot, k_rot = {fn}(q, k, position_ids, (2, 2, 2))
assert q_rot.shape == q.shape and k_rot.shape == k.shape
assert q_rot.dtype == q.dtype and k_rot.dtype == k.dtype
assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5)
""",
        },
        {
            "name": "Text positions reduce to ordinary 1D RoPE",
            "code": """
import torch
torch.manual_seed(1)
B, H, N, D = 1, 2, 5, 12
q = torch.randn(B, H, N, D)
k = torch.randn(B, H, N, D)
text_pos = torch.arange(N).view(1, 1, N).expand(3, B, N)
q_rot, k_rot = {fn}(q, k, text_pos, (1, 2, 3))

def rope_1d_reference(x, positions, base=10000.0):
    pairs = x.reshape(B, H, N, D // 2, 2)
    inv_freq = base ** (-2 * torch.arange(D // 2, dtype=x.dtype) / D)
    angles = positions.to(x.dtype).unsqueeze(-1) * inv_freq
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

expected_q = rope_1d_reference(q, text_pos[0])
expected_k = rope_1d_reference(k, text_pos[0])
assert torch.allclose(q_rot, expected_q, atol=1e-5)
assert torch.allclose(k_rot, expected_k, atol=1e-5)
""",
        },
        {
            "name": "Temporal coordinates affect only temporal pairs",
            "code": """
import torch
q = torch.ones(1, 1, 2, 12)
k = q.clone()
position_ids = torch.zeros(3, 1, 2)
position_ids[0, 0, 1] = 1
q_rot, _ = {fn}(q, k, position_ids, (2, 2, 2))
assert torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-6)
assert not torch.allclose(q_rot[..., 1, :4], q[..., 1, :4], atol=1e-6)
assert torch.allclose(q_rot[..., 1, 4:], q[..., 1, 4:], atol=1e-6)
""",
        },
        {
            "name": "Per-axis global shifts preserve attention scores",
            "code": """
import torch
torch.manual_seed(2)
q = torch.randn(1, 2, 6, 12)
k = torch.randn(1, 2, 6, 12)
position_ids = torch.randint(0, 4, (3, 1, 6))
shift = torch.tensor([3, 5, 7]).view(3, 1, 1)
q1, k1 = {fn}(q, k, position_ids, (2, 2, 2))
q2, k2 = {fn}(q, k, position_ids + shift, (2, 2, 2))
scores1 = torch.matmul(q1, k1.transpose(-1, -2))
scores2 = torch.matmul(q2, k2.transpose(-1, -2))
assert torch.allclose(scores1, scores2, atol=2e-5), 'RoPE should depend on relative coordinates'
""",
        },
        {
            "name": "Validation and gradient flow",
            "code": """
import torch
q = torch.randn(1, 2, 4, 12, requires_grad=True)
k = torch.randn(1, 1, 4, 12, requires_grad=True)
position_ids = torch.zeros(3, 1, 4)
q_rot, k_rot = {fn}(q, k, position_ids, (2, 2, 2))
(q_rot.sum() + k_rot.sum()).backward()
assert q.grad is not None and k.grad is not None

for bad_positions, bad_sections in [
    (torch.zeros(2, 1, 4), (2, 2, 2)),
    (position_ids, (1, 1, 1)),
]:
    try:
        {fn}(q.detach(), k.detach(), bad_positions, bad_sections)
    except (AssertionError, TypeError, ValueError):
        pass
    else:
        raise AssertionError('invalid position_ids or sections should be rejected')
""",
        },
    ],
}
