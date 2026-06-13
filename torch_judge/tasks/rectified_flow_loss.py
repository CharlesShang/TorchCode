"""Rectified Flow Loss task."""

TASK = {'title': 'Rectified Flow Loss',
 'difficulty': 'Medium',
 'function_name': 'rectified_flow_loss',
 'hint': 'Sample or accept t with shape (B,). Interpolate x_t = (1-t) x0 + t x1. The target '
         'velocity is x1 - x0. Train model(x_t, t) with MSE.',
 'tests': [{'name': 'Zero model has target MSE',
            'code': '\n'
                    'import torch\n'
                    'class ZeroModel:\n'
                    '    def __call__(self, x, t): return torch.zeros_like(x)\n'
                    'x0 = torch.zeros(2, 3)\n'
                    'x1 = torch.ones(2, 3) * 2\n'
                    'loss = {fn}(ZeroModel(), x0, x1, t=torch.tensor([0.2, 0.8]))\n'
                    "assert torch.allclose(loss, torch.tensor(4.0)), f'{loss}'\n"},
           {'name': 'Perfect model has zero loss',
            'code': '\n'
                    'import torch\n'
                    'class PerfectModel:\n'
                    '    def __init__(self, target): self.target = target\n'
                    '    def __call__(self, x, t): return self.target\n'
                    'x0 = torch.randn(2, 3, 4)\n'
                    'x1 = torch.randn(2, 3, 4)\n'
                    'loss = {fn}(PerfectModel(x1 - x0), x0, x1, t=torch.tensor([0.1, 0.9]))\n'
                    "assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7), f'{loss}'\n"},
           {'name': 'Calls model with interpolated x_t',
            'code': '\n'
                    'import torch\n'
                    'class Recorder:\n'
                    '    def __init__(self): self.x = None; self.t = None\n'
                    '    def __call__(self, x, t):\n'
                    '        self.x = x.detach().clone(); self.t = t.detach().clone(); return '
                    'torch.zeros_like(x)\n'
                    'model = Recorder()\n'
                    'x0 = torch.zeros(2, 1)\n'
                    'x1 = torch.ones(2, 1) * 10\n'
                    't = torch.tensor([0.25, 0.75])\n'
                    "{fn}(model, x0, x1, t=t, reduction='sum')\n"
                    'assert torch.allclose(model.x.squeeze(), torch.tensor([2.5, 7.5])), '
                    "f'{model.x}'\n"
                    "assert torch.equal(model.t, t), 't should be passed unchanged to model'\n"},
           {'name': 'None reduction shape',
            'code': '\n'
                    'import torch\n'
                    'class ZeroModel:\n'
                    '    def __call__(self, x, t): return torch.zeros_like(x)\n'
                    'x0 = torch.zeros(2, 3)\n'
                    'x1 = torch.ones(2, 3)\n'
                    'loss = {fn}(ZeroModel(), x0, x1, t=torch.tensor([0.5, 0.5]), '
                    "reduction='none')\n"
                    "assert loss.shape == x0.shape, f'{loss.shape}'\n"}]}
