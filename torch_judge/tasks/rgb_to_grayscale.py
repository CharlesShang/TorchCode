"""RGB to Grayscale task."""

TASK = {'title': 'RGB to Grayscale',
 'difficulty': 'Easy',
 'function_name': 'rgb_to_grayscale',
 'hint': 'Use luma weights [0.299, 0.587, 0.114]. Reshape them to broadcast over (..., 3, H, W), '
         'multiply, and sum over the channel axis.',
 'tests': [{'name': 'Matches luma formula',
            'code': '\n'
                    'import torch\n'
                    'x = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], '
                    '[11., 12.]]])\n'
                    'out = {fn}(x, keepdim=False)\n'
                    'expected = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]\n'
                    "assert torch.allclose(out, expected), f'{out} vs {expected}'\n"},
           {'name': 'Batched keepdim shape',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(4, 3, 8, 8)\n'
                    'out = {fn}(x)\n'
                    "assert out.shape == (4, 1, 8, 8), f'{out.shape}'\n"},
           {'name': 'Custom weights',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 4, 5)\n'
                    'w = torch.tensor([1.0, 0.0, 0.0])\n'
                    'out = {fn}(x, weights=w, keepdim=False)\n'
                    "assert torch.allclose(out, x[:, 0]), 'Custom weights should select red "
                    "channel'\n"},
           {'name': 'Rejects non-RGB',
            'code': '\n'
                    'import torch\n'
                    'try:\n'
                    '    {fn}(torch.randn(1, 4, 8, 8))\n'
                    'except (AssertionError, ValueError):\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('Should reject channel dimension != 3')\n"}]}
