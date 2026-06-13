"""DiT Block task."""

TASK = {'title': 'DiT Block',
 'difficulty': 'Hard',
 'function_name': 'DiTBlock',
 'hint': 'Use two affine-free LayerNorms, one zero-initialized modulation linear that returns 6 '
         'chunks, self-attention, and an MLP. Apply each residual branch as x = x + gate * branch.',
 'tests': [{'name': 'Forward shape',
            'code': '\n'
                    'import torch\n'
                    'block = {fn}(dim=16, num_heads=4, cond_dim=8)\n'
                    'x = torch.randn(2, 5, 16)\n'
                    'cond = torch.randn(2, 8)\n'
                    'out = block(x, cond)\n'
                    "assert out.shape == x.shape, f'Unexpected shape: {out.shape}'\n"},
           {'name': 'Zero modulation starts as identity',
            'code': '\n'
                    'import torch\n'
                    'block = {fn}(16, 4, 8)\n'
                    'x = torch.randn(2, 5, 16)\n'
                    'cond = torch.randn(2, 8)\n'
                    'out = block(x, cond)\n'
                    "assert torch.allclose(out, x, atol=1e-6), 'AdaLN-Zero gates should make block "
                    "identity at init'\n"},
           {'name': 'Nonzero gates activate residual branches',
            'code': '\n'
                    'import torch\n'
                    'block = {fn}(16, 4, 8)\n'
                    'with torch.no_grad():\n'
                    '    block.modulation.bias[2*16:3*16].fill_(1.0)\n'
                    '    block.modulation.bias[5*16:6*16].fill_(1.0)\n'
                    'x = torch.randn(2, 5, 16)\n'
                    'cond = torch.randn(2, 8)\n'
                    'out = block(x, cond)\n'
                    "assert not torch.allclose(out, x), 'Nonzero gates should change the residual "
                    "output'\n"},
           {'name': 'Gradients flow',
            'code': '\n'
                    'import torch\n'
                    'block = {fn}(8, 2, 4)\n'
                    'with torch.no_grad():\n'
                    '    block.modulation.bias[2*8:3*8].fill_(1.0)\n'
                    '    block.modulation.bias[5*8:6*8].fill_(1.0)\n'
                    'x = torch.randn(2, 3, 8, requires_grad=True)\n'
                    'cond = torch.randn(2, 4, requires_grad=True)\n'
                    'block(x, cond).sum().backward()\n'
                    "assert x.grad is not None and cond.grad is not None, 'Missing gradients'\n"}]}
