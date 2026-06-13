"""Bilinear Resize NCHW task."""

TASK = {'title': 'Bilinear Resize NCHW',
 'difficulty': 'Medium',
 'function_name': 'bilinear_resize',
 'hint': 'Build floating source coordinates for output rows/cols, gather the four neighboring '
         "pixels, then blend with bilinear weights. Match PyTorch's align_corners coordinate "
         'formulas.',
 'tests': [{'name': 'Matches F.interpolate align_corners=False',
            'code': '\n'
                    'import torch\n'
                    'import torch.nn.functional as F\n'
                    'torch.manual_seed(0)\n'
                    'x = torch.randn(2, 3, 5, 7)\n'
                    'out = {fn}(x, 9, 4, align_corners=False)\n'
                    "expected = F.interpolate(x, size=(9, 4), mode='bilinear', "
                    'align_corners=False)\n'
                    "assert out.shape == expected.shape, f'{out.shape}'\n"
                    "assert torch.allclose(out, expected, atol=1e-5), f'Max diff "
                    "{(out-expected).abs().max()}'\n"},
           {'name': 'Matches F.interpolate align_corners=True',
            'code': '\n'
                    'import torch\n'
                    'import torch.nn.functional as F\n'
                    'torch.manual_seed(1)\n'
                    'x = torch.randn(1, 2, 4, 4)\n'
                    'out = {fn}(x, 6, 6, align_corners=True)\n'
                    "expected = F.interpolate(x, size=(6, 6), mode='bilinear', "
                    'align_corners=True)\n'
                    "assert torch.allclose(out, expected, atol=1e-5), f'Max diff "
                    "{(out-expected).abs().max()}'\n"},
           {'name': 'Identity size',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 4, 5)\n'
                    'out = {fn}(x, 4, 5)\n'
                    "assert torch.allclose(out, x, atol=1e-6), 'Resizing to same size should be "
                    "identity'\n"}]}
