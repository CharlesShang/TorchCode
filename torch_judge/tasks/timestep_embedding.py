"""Diffusion Timestep Embedding task."""

TASK = {'title': 'Diffusion Timestep Embedding',
 'difficulty': 'Easy',
 'function_name': 'timestep_embedding',
 'hint': 'Use half_dim frequencies: exp(-log(max_period) * arange(half) / half). Concatenate cos(t '
         '* freq) and sin(t * freq), then pad one zero column for odd dimensions.',
 'tests': [{'name': 'Shape and first timestep',
            'code': '\n'
                    'import torch\n'
                    't = torch.tensor([0, 1, 10])\n'
                    'out = {fn}(t, dim=7)\n'
                    "assert out.shape == (3, 7), f'Unexpected shape: {out.shape}'\n"
                    "assert torch.allclose(out[0, :3], torch.ones(3)), 'cos columns at t=0 should "
                    "be 1'\n"
                    "assert torch.allclose(out[0, 3:], torch.zeros(4)), 'sin/pad columns at t=0 "
                    "should be 0'\n"},
           {'name': 'Matches reference formula',
            'code': '\n'
                    'import torch, math\n'
                    't = torch.tensor([1.0, 2.0])\n'
                    'dim = 6\n'
                    'half = dim // 2\n'
                    'freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)\n'
                    'expected = torch.cat([torch.cos(t[:, None] * freqs[None]), torch.sin(t[:, '
                    'None] * freqs[None])], dim=-1)\n'
                    'out = {fn}(t, dim)\n'
                    "assert torch.allclose(out, expected, atol=1e-6), 'Embedding formula "
                    "mismatch'\n"},
           {'name': 'Float timesteps and gradients',
            'code': '\n'
                    'import torch\n'
                    't = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)\n'
                    'out = {fn}(t, 8)\n'
                    'out.sum().backward()\n'
                    "assert t.grad is not None, 'timesteps should be differentiable when "
                    "float'\n"}]}
