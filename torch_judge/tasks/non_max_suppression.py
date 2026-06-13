"""Non-Max Suppression task."""

TASK = {'title': 'Non-Max Suppression',
 'difficulty': 'Medium',
 'function_name': 'non_max_suppression',
 'hint': 'Sort boxes by descending score. Repeatedly keep the highest-scoring box, remove '
         'remaining boxes whose IoU with it exceeds the threshold, and continue.',
 'tests': [{'name': 'Suppresses overlapping lower score',
            'code': '\n'
                    'import torch\n'
                    'boxes = torch.tensor([[0.,0.,2.,2.], [0.1,0.1,2.1,2.1], [5.,5.,6.,6.]])\n'
                    'scores = torch.tensor([0.9, 0.8, 0.7])\n'
                    'keep = {fn}(boxes, scores, 0.5)\n'
                    "assert torch.equal(keep, torch.tensor([0, 2])), f'{keep}'\n"},
           {'name': 'Keeps descending score order',
            'code': '\n'
                    'import torch\n'
                    'boxes = torch.tensor([[0.,0.,1.,1.], [2.,2.,3.,3.], [4.,4.,5.,5.]])\n'
                    'scores = torch.tensor([0.2, 0.9, 0.5])\n'
                    'keep = {fn}(boxes, scores, 0.1)\n'
                    "assert torch.equal(keep, torch.tensor([1, 2, 0])), f'{keep}'\n"},
           {'name': 'Empty input',
            'code': '\n'
                    'import torch\n'
                    'keep = {fn}(torch.empty(0, 4), torch.empty(0), 0.5)\n'
                    "assert keep.shape == (0,) and keep.dtype == torch.long, f'{keep}'\n"}]}
