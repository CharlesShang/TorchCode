"""Temporal Average Pooling for Video task."""

TASK = {'title': 'Temporal Average Pooling for Video',
 'difficulty': 'Medium',
 'function_name': 'temporal_avg_pool',
 'hint': 'Pad the temporal dimension if requested, unfold over dimension 2 of B,C,T,H,W, then '
         'average over the kernel dimension.',
 'tests': [{'name': 'Matches avg_pool3d',
            'code': '\n'
                    'import torch\n'
                    'import torch.nn.functional as F\n'
                    'video = torch.randn(2, 3, 8, 4, 5)\n'
                    'out = {fn}(video, kernel_size=3, stride=2, padding=1)\n'
                    'expected = F.avg_pool3d(video, kernel_size=(3,1,1), stride=(2,1,1), '
                    'padding=(1,0,0))\n'
                    "assert out.shape == expected.shape, f'{out.shape} vs {expected.shape}'\n"
                    "assert torch.allclose(out, expected, atol=1e-6), 'Temporal avg pool "
                    "mismatch'\n"},
           {'name': 'Default stride',
            'code': '\n'
                    'import torch\n'
                    'video = torch.arange(1., 7.).view(1, 1, 6, 1, 1)\n'
                    'out = {fn}(video, 2)\n'
                    'assert torch.equal(out.flatten(), torch.tensor([1.5, 3.5, 5.5])), '
                    "f'{out.flatten()}'\n"},
           {'name': 'Gradient flow',
            'code': '\n'
                    'import torch\n'
                    'video = torch.randn(1, 1, 4, 2, 2, requires_grad=True)\n'
                    '{fn}(video, 2).sum().backward()\n'
                    "assert video.grad is not None, 'Missing gradient'\n"}]}
