"""Classifier-Free Guidance task."""

TASK = {'title': 'Classifier-Free Guidance',
 'difficulty': 'Medium',
 'function_name': 'classifier_free_guidance',
 'hint': 'The core formula is uncond + guidance_scale * (cond - uncond). Optional rescale matches '
         'the guided standard deviation to the conditional prediction and linearly mixes it back.',
 'tests': [{'name': 'Core formula',
            'code': '\n'
                    'import torch\n'
                    'uncond = torch.tensor([1.0, 2.0, 3.0])\n'
                    'cond = torch.tensor([2.0, 4.0, 8.0])\n'
                    'out = {fn}(uncond, cond, guidance_scale=2.5)\n'
                    'expected = uncond + 2.5 * (cond - uncond)\n'
                    "assert torch.allclose(out, expected), f'{out} vs {expected}'\n"},
           {'name': 'Scale endpoints',
            'code': '\n'
                    'import torch\n'
                    'u = torch.randn(2, 3)\n'
                    'c = torch.randn(2, 3)\n'
                    "assert torch.allclose({fn}(u, c, guidance_scale=0.0), u), 'scale=0 should "
                    "return uncond'\n"
                    "assert torch.allclose({fn}(u, c, guidance_scale=1.0), c), 'scale=1 should "
                    "return cond'\n"},
           {'name': 'Rescale keeps finite shape',
            'code': '\n'
                    'import torch\n'
                    'torch.manual_seed(0)\n'
                    'u = torch.randn(2, 3, 4, 4)\n'
                    'c = torch.randn(2, 3, 4, 4)\n'
                    'out = {fn}(u, c, guidance_scale=7.5, rescale=0.7)\n'
                    "assert out.shape == u.shape, 'Shape mismatch'\n"
                    "assert torch.isfinite(out).all(), 'Output contains NaN or Inf'\n"}]}
