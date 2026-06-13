"""Euler Flow Sampler task."""

TASK = {'title': 'Euler Flow Sampler',
 'difficulty': 'Medium',
 'function_name': 'euler_flow_sample',
 'hint': 'Start from x0 and integrate dx/dt = model(x, t) using dt = 1 / steps. Pass a '
         'batch-shaped t tensor at every step.',
 'tests': [{'name': 'Constant velocity integrates exactly',
            'code': '\n'
                    'import torch\n'
                    'class ConstantVelocity:\n'
                    '    def __call__(self, x, t): return torch.ones_like(x) * 3\n'
                    'x0 = torch.zeros(2, 4)\n'
                    'out = {fn}(ConstantVelocity(), x0, steps=6)\n'
                    "assert torch.allclose(out, torch.ones_like(x0) * 3, atol=1e-6), f'{out}'\n"},
           {'name': 'Trajectory length',
            'code': '\n'
                    'import torch\n'
                    'class ZeroVelocity:\n'
                    '    def __call__(self, x, t): return torch.zeros_like(x)\n'
                    'traj = {fn}(ZeroVelocity(), torch.randn(1, 2), steps=5, '
                    'return_trajectory=True)\n'
                    "assert isinstance(traj, list), 'trajectory should be a list'\n"
                    "assert len(traj) == 6, f'Expected steps+1 states, got {len(traj)}'\n"},
           {'name': 'Passes batch-shaped timesteps',
            'code': '\n'
                    'import torch\n'
                    'class Recorder:\n'
                    '    def __init__(self): self.ts = []\n'
                    '    def __call__(self, x, t):\n'
                    '        self.ts.append(t.detach().clone()); return torch.zeros_like(x)\n'
                    'model = Recorder()\n'
                    '{fn}(model, torch.randn(3, 2), steps=4)\n'
                    "assert len(model.ts) == 4, 'Model should be called once per step'\n"
                    "assert all(t.shape == (3,) for t in model.ts), f'Timestep shapes: {[t.shape "
                    "for t in model.ts]}'\n"
                    "assert torch.allclose(model.ts[-1], torch.full((3,), 0.75)), f'Last t: "
                    "{model.ts[-1]}'\n"}]}
