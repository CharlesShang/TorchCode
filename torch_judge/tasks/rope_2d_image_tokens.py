"""2D RoPE for Image Tokens task."""

TASK = {'title': '2D RoPE for Image Tokens',
 'difficulty': 'Hard',
 'function_name': 'apply_2d_rope',
 'hint': 'Split the channel dim into y-half and x-half. Apply ordinary 1D RoPE to the y-half using '
         'row positions and to the x-half using column positions.',
 'tests': [{'name': 'Shape and origin unchanged',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 4, 8)\n'
                    'out = {fn}(x, height=2, width=2)\n'
                    "assert out.shape == x.shape, f'{out.shape}'\n"
                    "assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6), 'Position (0,0) "
                    "should be unchanged'\n"},
           {'name': 'Preserves norms',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(1, 2, 12, 16)\n'
                    'out = {fn}(x, 3, 4)\n'
                    "assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5), 'RoPE "
                    "should preserve per-token norms'\n"},
           {'name': 'Rejects invalid head_dim',
            'code': '\n'
                    'import torch\n'
                    'try:\n'
                    '    {fn}(torch.randn(1, 1, 4, 6), 2, 2)\n'
                    'except (AssertionError, ValueError):\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('head_dim not divisible by 4 should be rejected')\n"},
           {'name': 'Rejects wrong sequence length',
            'code': '\n'
                    'import torch\n'
                    'try:\n'
                    '    {fn}(torch.randn(1, 1, 5, 8), 2, 2)\n'
                    'except (AssertionError, ValueError):\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('N != height * width should be rejected')\n"}]}
