"""Fused RMSNorm + Residual task."""

TASK = {'title': 'Fused RMSNorm + Residual',
 'difficulty': 'Medium',
 'function_name': 'fused_rmsnorm_residual',
 'hint': 'First compute updated_residual = x + residual, optionally in fp32. Then RMS-normalize '
         'updated_residual and multiply by weight. Return both normalized output and updated '
         'residual.',
 'tests': [{'name': 'Matches manual RMSNorm',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 8)\n'
                    'r = torch.randn(2, 3, 8)\n'
                    'w = torch.randn(8)\n'
                    'y, updated = {fn}(x, r, w, eps=1e-5)\n'
                    'manual_updated = x.float() + r.float()\n'
                    'manual = manual_updated * torch.rsqrt(manual_updated.pow(2).mean(dim=-1, '
                    'keepdim=True) + 1e-5) * w\n'
                    "assert torch.allclose(updated, manual_updated), 'updated residual mismatch'\n"
                    "assert torch.allclose(y, manual.to(x.dtype), atol=1e-6), 'RMSNorm output "
                    "mismatch'\n"},
           {'name': 'fp32 residual option',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 4, dtype=torch.float16)\n'
                    'r = torch.randn(2, 4, dtype=torch.float16)\n'
                    'w = torch.ones(4, dtype=torch.float16)\n'
                    'y, updated = {fn}(x, r, w, residual_in_fp32=True)\n'
                    "assert y.dtype == torch.float16, f'output dtype should match x, got "
                    "{y.dtype}'\n"
                    "assert updated.dtype == torch.float32, f'updated residual should be fp32, got "
                    "{updated.dtype}'\n"},
           {'name': 'No fp32 residual option',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 4, dtype=torch.float16)\n'
                    'r = torch.randn(2, 4, dtype=torch.float16)\n'
                    'w = torch.ones(4, dtype=torch.float16)\n'
                    'y, updated = {fn}(x, r, w, residual_in_fp32=False)\n'
                    "assert updated.dtype == torch.float16, f'{updated.dtype}'\n"},
           {'name': 'Gradient flow',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 5, requires_grad=True)\n'
                    'r = torch.randn(2, 3, 5, requires_grad=True)\n'
                    'w = torch.randn(5, requires_grad=True)\n'
                    'y, updated = {fn}(x, r, w)\n'
                    '(y.sum() + updated.sum()).backward()\n'
                    'assert x.grad is not None and r.grad is not None and w.grad is not None, '
                    "'Missing gradients'\n"}]}
