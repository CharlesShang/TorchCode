"""Video Tubelet Patchify task."""

TASK = {'title': 'Video Tubelet Patchify',
 'difficulty': 'Hard',
 'function_name': 'VideoTubeletPatcher',
 'hint': 'Reshape B,T,C,H,W into blocks over time/height/width, permute block axes before '
         'C/tubelet axes, flatten each tubelet, then reverse for unpatchify.',
 'tests': [{'name': 'Patchify shape',
            'code': '\n'
                    'import torch\n'
                    'video = torch.randn(2, 4, 3, 8, 8)\n'
                    'patcher = {fn}(tubelet_size=2, patch_size=4)\n'
                    'tokens = patcher.patchify(video)\n'
                    "assert tokens.shape == (2, 8, 96), f'{tokens.shape}'\n"},
           {'name': 'Roundtrip exact',
            'code': '\n'
                    'import torch\n'
                    'video = torch.randn(1, 4, 2, 6, 8)\n'
                    'patcher = {fn}(2, 2)\n'
                    'tokens = patcher.patchify(video)\n'
                    'recon = patcher.unpatchify(tokens, tuple(video.shape))\n'
                    "assert torch.equal(recon, video), 'unpatchify(patchify(video)) should "
                    "reconstruct exactly'\n"},
           {'name': 'First tubelet order',
            'code': '\n'
                    'import torch\n'
                    'video = torch.arange(1., 1 + 1*2*1*2*2).view(1, 2, 1, 2, 2)\n'
                    'patcher = {fn}(2, 2)\n'
                    'tokens = patcher.patchify(video)\n'
                    "assert torch.equal(tokens[0, 0], video.flatten()), f'{tokens[0,0]} vs "
                    "{video.flatten()}'\n"},
           {'name': 'Rejects incompatible shape',
            'code': '\n'
                    'import torch\n'
                    'patcher = {fn}(2, 3)\n'
                    'try:\n'
                    '    patcher.patchify(torch.randn(1, 3, 1, 6, 6))\n'
                    'except (AssertionError, ValueError):\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('Should reject T not divisible by tubelet_size')\n"}]}
