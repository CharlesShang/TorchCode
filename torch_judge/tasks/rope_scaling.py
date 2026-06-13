"""RoPE Scaling task."""

TASK = {'title': 'RoPE Scaling',
 'difficulty': 'Medium',
 'function_name': 'apply_scaled_rope',
 'hint': 'Linear RoPE scaling divides positions by scaling_factor before computing the rotary '
         'angles. scaling_factor=1 should be ordinary RoPE.',
 'tests': [{'name': 'Shape and first position unchanged',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 5, 8)\n'
                    'out = {fn}(x)\n'
                    "assert out.shape == x.shape, f'{out.shape}'\n"
                    "assert torch.allclose(out[:, 0], x[:, 0], atol=1e-6), 'position 0 should be "
                    "unchanged'\n"},
           {'name': 'Norm preserved',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(3, 7, 12)\n'
                    'out = {fn}(x, scaling_factor=4.0)\n'
                    "assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5), 'RoPE "
                    "should preserve norms'\n"},
           {'name': 'Scaling changes nonzero positions',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(1, 6, 8)\n'
                    'out1 = {fn}(x, scaling_factor=1.0)\n'
                    'out2 = {fn}(x, scaling_factor=2.0)\n'
                    "assert not torch.allclose(out1[:, 1:], out2[:, 1:]), 'scaling_factor should "
                    "affect positions > 0'\n"},
           {'name': 'Custom positions',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(1, 3, 8)\n'
                    'pos = torch.tensor([0.0, 10.0, 20.0])\n'
                    'out = {fn}(x, positions=pos)\n'
                    "assert out.shape == x.shape and torch.isfinite(out).all(), 'Custom positions "
                    "should work'\n"}]}
