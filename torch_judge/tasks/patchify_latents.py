"""Patchify / Unpatchify Latents task."""

TASK = {'title': 'Patchify / Unpatchify Latents',
 'difficulty': 'Medium',
 'function_name': 'PatchifyLatents',
 'hint': 'Patchify with reshape + permute: (B,C,H,W) -> (B,H/pH,W/pW,C,pH,pW) -> (B,N,C*pH*pW). '
         'Reverse the same axes for unpatchify.',
 'tests': [{'name': 'Patchify shape',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 3, 8, 8)\n'
                    'patcher = {fn}(2)\n'
                    'tokens = patcher.patchify(x)\n'
                    "assert tokens.shape == (2, 16, 12), f'Unexpected token shape: "
                    "{tokens.shape}'\n"},
           {'name': 'Round trip reconstruction',
            'code': '\n'
                    'import torch\n'
                    'x = torch.randn(2, 4, 6, 10)\n'
                    'patcher = {fn}((3, 2))\n'
                    'tokens = patcher.patchify(x)\n'
                    'recon = patcher.unpatchify(tokens, image_size=(6, 10), channels=4)\n'
                    "assert torch.equal(recon, x), 'unpatchify(patchify(x)) must exactly "
                    "reconstruct x'\n"},
           {'name': 'First patch flatten order',
            'code': '\n'
                    'import torch\n'
                    'x = torch.arange(1.0, 17.0).view(1, 1, 4, 4)\n'
                    'patcher = {fn}(2)\n'
                    'tokens = patcher.patchify(x)\n'
                    'expected_first = torch.tensor([1.0, 2.0, 5.0, 6.0])\n'
                    "assert torch.equal(tokens[0, 0], expected_first), f'{tokens[0, 0]} vs "
                    "{expected_first}'\n"},
           {'name': 'Rejects incompatible size',
            'code': '\n'
                    'import torch\n'
                    'patcher = {fn}(3)\n'
                    'try:\n'
                    '    patcher.patchify(torch.randn(1, 1, 5, 6))\n'
                    'except (AssertionError, ValueError):\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('Should reject H/W not divisible by patch size')\n"}]}
