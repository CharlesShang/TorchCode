"""Sobel Edge Magnitude task."""

TASK = {'title': 'Sobel Edge Magnitude',
 'difficulty': 'Medium',
 'function_name': 'sobel_edges',
 'hint': 'Use the standard Sobel kernels for x/y gradients. Apply them depthwise to every channel '
         'with padding=1, then compute sqrt(gx^2 + gy^2 + eps).',
 'tests': [{'name': 'Shape and nonnegative',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 8, 8)\n'
                    'out = {fn}(x)\n'
                    "assert out.shape == x.shape, f'{out.shape}'\n"
                    "assert (out >= 0).all(), 'Magnitude should be nonnegative'\n"},
           {'name': 'Matches manual depthwise conv',
            'code': '\n'
                    'import torch\n'
                    'import torch.nn.functional as F\n'
                    'x = torch.randn(1, 2, 5, 5)\n'
                    'kx = '
                    'torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).repeat(2,1,1,1)\n'
                    'ky = '
                    'torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]).view(1,1,3,3).repeat(2,1,1,1)\n'
                    'expected = torch.sqrt(F.conv2d(x,kx,padding=1,groups=2).pow(2) + '
                    'F.conv2d(x,ky,padding=1,groups=2).pow(2) + 1e-6)\n'
                    'out = {fn}(x)\n'
                    "assert torch.allclose(out, expected, atol=1e-6), 'Sobel result mismatch'\n"},
           {'name': 'Gradient flow',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(1, 1, 4, 4, requires_grad=True)\n'
                    '{fn}(x).sum().backward()\n'
                    "assert x.grad is not None, 'Missing gradient'\n"}]}
