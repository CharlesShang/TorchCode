"""QK Norm Attention task."""

TASK = {'title': 'QK Norm Attention',
 'difficulty': 'Medium',
 'function_name': 'qk_norm_attention',
 'hint': 'Normalize q and k with their L2 norm over the head dimension before computing attention '
         'scores. Then apply mask, softmax, and multiply by v.',
 'tests': [{'name': 'Matches manual reference',
            'code': '\n'
                    'import torch, math\n'
                    'torch.manual_seed(0)\n'
                    'q = torch.randn(2, 3, 4)\n'
                    'k = torch.randn(2, 5, 4)\n'
                    'v = torch.randn(2, 5, 6)\n'
                    'qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)\n'
                    'kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)\n'
                    'expected = torch.softmax((qn @ kn.transpose(-2, -1)) * math.sqrt(4), dim=-1) '
                    '@ v\n'
                    'out = {fn}(q, k, v)\n'
                    "assert torch.allclose(out, expected, atol=1e-6), 'Manual reference "
                    "mismatch'\n"},
           {'name': 'Invariant to positive q/k scaling',
            'code': '\n'
                    'import torch\n'
                    'q = torch.randn(1, 3, 8)\n'
                    'k = torch.randn(1, 4, 8)\n'
                    'v = torch.randn(1, 4, 5)\n'
                    'out1 = {fn}(q, k, v)\n'
                    'out2 = {fn}(q * 10.0, k * 0.1, v)\n'
                    "assert torch.allclose(out1, out2, atol=1e-5), 'QK norm should remove "
                    "magnitude scaling'\n"},
           {'name': 'Mask hides invalid keys',
            'code': '\n'
                    'import torch\n'
                    'q = torch.randn(1, 1, 4)\n'
                    'k = torch.randn(1, 3, 4)\n'
                    'v = torch.tensor([[[1.0], [10.0], [1000.0]]])\n'
                    'mask = torch.tensor([[True, True, False]])\n'
                    'out = {fn}(q, k, v, mask=mask)\n'
                    "assert out.item() < 10.1, f'Masked large value leaked into output: {out}'\n"}]}
