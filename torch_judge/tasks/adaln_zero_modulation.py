"""AdaLN-Zero Modulation task."""

TASK = {'title': 'AdaLN-Zero Modulation',
 'difficulty': 'Medium',
 'function_name': 'AdaLNZero',
 'hint': 'Use LayerNorm without affine params. A linear layer maps cond -> 6 * dim. '
         'Zero-initialize that layer so gates start at zero and the block can begin as an identity '
         'when used in a residual block.',
 'tests': [{'name': 'Zero initialization',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(dim=8, cond_dim=4)\n'
                    'assert torch.equal(m.modulation.weight, '
                    "torch.zeros_like(m.modulation.weight)), 'weight should be zero-initialized'\n"
                    'assert torch.equal(m.modulation.bias, torch.zeros_like(m.modulation.bias)), '
                    "'bias should be zero-initialized'\n"},
           {'name': 'Initial output and gates',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(8, 4)\n'
                    'x = torch.randn(2, 3, 8)\n'
                    'cond = torch.randn(2, 4)\n'
                    'x_msa, gate_msa, x_mlp, gate_mlp = m(x, cond)\n'
                    'expected = m.norm(x)\n'
                    "assert torch.allclose(x_msa, expected, atol=1e-6), 'zero init should return "
                    "normalized x for MSA path'\n"
                    "assert torch.allclose(x_mlp, expected, atol=1e-6), 'zero init should return "
                    "normalized x for MLP path'\n"
                    "assert torch.equal(gate_msa, torch.zeros_like(gate_msa)), 'MSA gate should "
                    "start at zero'\n"
                    "assert torch.equal(gate_mlp, torch.zeros_like(gate_mlp)), 'MLP gate should "
                    "start at zero'\n"},
           {'name': 'Bias modulation formula',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(4, 2)\n'
                    'with torch.no_grad():\n'
                    '    m.modulation.bias.copy_(torch.tensor([1., 1., 1., 1., 0.5, 0.5, 0.5, 0.5, '
                    '2., 2., 2., 2., -1., -1., -1., -1., 1., 1., 1., 1., -2., -2., -2., -2.]))\n'
                    'x = torch.randn(1, 2, 4)\n'
                    'cond = torch.randn(1, 2)\n'
                    'x_msa, gate_msa, x_mlp, gate_mlp = m(x, cond)\n'
                    'norm = m.norm(x)\n'
                    "assert torch.allclose(x_msa, norm * 1.5 + 1.0, atol=1e-6), 'MSA shift/scale "
                    "formula mismatch'\n"
                    "assert torch.allclose(x_mlp, norm * 2.0 - 1.0, atol=1e-6), 'MLP shift/scale "
                    "formula mismatch'\n"
                    "assert torch.allclose(gate_msa, torch.full_like(gate_msa, 2.0)), 'MSA gate "
                    "mismatch'\n"
                    "assert torch.allclose(gate_mlp, torch.full_like(gate_mlp, -2.0)), 'MLP gate "
                    "mismatch'\n"}]}
