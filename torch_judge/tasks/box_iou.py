"""Bounding Box IoU task."""

TASK = {'title': 'Bounding Box IoU',
 'difficulty': 'Easy',
 'function_name': 'box_iou',
 'hint': 'Use broadcasting. Intersection top-left is max of x1/y1, bottom-right is min of x2/y2. '
         'Clamp width/height at zero, then IoU = inter / union.',
 'tests': [{'name': 'Known values',
            'code': '\n'
                    'import torch\n'
                    'b1 = torch.tensor([[0.,0.,2.,2.], [0.,0.,1.,1.]])\n'
                    'b2 = torch.tensor([[1.,1.,3.,3.], [0.,0.,2.,2.]])\n'
                    'out = {fn}(b1, b2)\n'
                    'expected = torch.tensor([[1/7, 1.0], [0.0, 0.25]])\n'
                    "assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'\n"},
           {'name': 'Pairwise shape',
            'code': '\n'
                    'import torch\n'
                    'out = {fn}(torch.randn(5, 4), torch.randn(7, 4))\n'
                    "assert out.shape == (5, 7), f'{out.shape}'\n"},
           {'name': 'No overlap is zero',
            'code': '\n'
                    'import torch\n'
                    'b1 = torch.tensor([[0.,0.,1.,1.]])\n'
                    'b2 = torch.tensor([[2.,2.,3.,3.]])\n'
                    "assert torch.equal({fn}(b1, b2), torch.zeros(1, 1)), 'No overlap should have "
                    "IoU 0'\n"}]}
