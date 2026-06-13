"""Logit-Normal Timestep Sampling task."""

TASK = {'title': 'Logit-Normal Timestep Sampling',
 'difficulty': 'Medium',
 'function_name': 'sample_logit_normal_timesteps',
 'hint': 'Draw z from Normal(mean, std), then map to (0, 1) with sigmoid(z). Accept a '
         'torch.Generator so tests and users can reproduce samples.',
 'tests': [{'name': 'Shape, range, dtype',
            'code': '\n'
                    'import torch\n'
                    't = {fn}(128, dtype=torch.float64)\n'
                    "assert t.shape == (128,), f'{t.shape}'\n"
                    "assert t.dtype == torch.float64, f'{t.dtype}'\n"
                    "assert (t > 0).all() and (t < 1).all(), 'All samples should be in (0, 1)'\n"},
           {'name': 'Generator makes samples reproducible',
            'code': '\n'
                    'import torch\n'
                    'g1 = torch.Generator().manual_seed(123)\n'
                    'g2 = torch.Generator().manual_seed(123)\n'
                    't1 = {fn}(16, generator=g1)\n'
                    't2 = {fn}(16, generator=g2)\n'
                    "assert torch.equal(t1, t2), 'Same generator seed should produce same "
                    "samples'\n"},
           {'name': 'Mean shifts distribution',
            'code': '\n'
                    'import torch\n'
                    'g1 = torch.Generator().manual_seed(0)\n'
                    'g2 = torch.Generator().manual_seed(0)\n'
                    'low = {fn}(10000, mean=-2.0, generator=g1).mean()\n'
                    'high = {fn}(10000, mean=2.0, generator=g2).mean()\n'
                    "assert high > low + 0.3, f'mean parameter should shift samples: {low} vs "
                    "{high}'\n"}]}
