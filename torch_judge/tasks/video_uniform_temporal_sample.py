"""Video Uniform Temporal Sampling task."""

TASK = {'title': 'Video Uniform Temporal Sampling',
 'difficulty': 'Medium',
 'function_name': 'uniform_temporal_sample',
 'hint': 'Use torch.linspace(0, T-1, num_frames).round().long() to select evenly spaced frame '
         'indices along the temporal dimension.',
 'tests': [{'name': 'Known unbatched indices',
            'code': '\n'
                    'import torch\n'
                    'video = torch.arange(10).view(10, 1, 1, 1)\n'
                    'out = {fn}(video, 4)\n'
                    'assert torch.equal(out.flatten(), torch.tensor([0, 3, 6, 9])), '
                    "f'{out.flatten()}'\n"},
           {'name': 'Batched temporal dim',
            'code': '\n'
                    'import torch\n'
                    'video = torch.arange(2 * 8 * 1 * 1 * 1).view(2, 8, 1, 1, 1)\n'
                    'out = {fn}(video, 3, temporal_dim=1)\n'
                    "assert out.shape == (2, 3, 1, 1, 1), f'{out.shape}'\n"
                    "assert torch.equal(out[:, :, 0, 0, 0], video[:, [0, 4, 7], 0, 0, 0]), 'Wrong "
                    "sampled frames'\n"},
           {'name': 'Can oversample with repeated frames',
            'code': '\n'
                    'import torch\n'
                    'video = torch.arange(3).view(3, 1, 1, 1)\n'
                    'out = {fn}(video, 5)\n'
                    "assert out.shape[0] == 5, f'{out.shape}'\n"
                    "assert out[0].item() == 0 and out[-1].item() == 2, 'Should include "
                    "endpoints'\n"}]}
