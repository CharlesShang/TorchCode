"""Multi-Token Prediction Loss task."""

TASK = {'title': 'Multi-Token Prediction Loss',
 'difficulty': 'Medium',
 'function_name': 'multi_token_prediction_loss',
 'hint': 'For head k, logits[:, :, k] predicts targets shifted by k+1. Flatten all valid '
         'head/position examples and apply stable cross-entropy with logsumexp.',
 'tests': [{'name': 'Matches manual CE',
            'code': '\n'
                    'import torch\n'
                    'import torch.nn.functional as F\n'
                    'torch.manual_seed(0)\n'
                    'logits = torch.randn(2, 3, 2, 5)\n'
                    'targets = torch.randint(0, 5, (2, 5))\n'
                    'parts = []\n'
                    'for k in range(2):\n'
                    '    parts.append(F.cross_entropy(logits[:, :, k, :].reshape(-1, 5), '
                    "targets[:, k+1:k+1+3].reshape(-1), reduction='none'))\n"
                    'expected = torch.cat(parts).mean()\n'
                    'out = {fn}(logits, targets)\n'
                    "assert torch.allclose(out, expected, atol=1e-6), f'{out} vs {expected}'\n"},
           {'name': 'Ignore index',
            'code': '\n'
                    'import torch\n'
                    'logits = torch.randn(1, 2, 2, 4)\n'
                    'targets = torch.tensor([[1, -100, 2, 3]])\n'
                    'out = {fn}(logits, targets, ignore_index=-100)\n'
                    "assert out.dim() == 0 and torch.isfinite(out), 'Loss should be finite scalar "
                    "with ignored positions'\n"},
           {'name': 'All ignored returns differentiable zero',
            'code': '\n'
                    'import torch\n'
                    'logits = torch.randn(1, 2, 2, 4, requires_grad=True)\n'
                    'targets = torch.full((1, 4), -100)\n'
                    'loss = {fn}(logits, targets)\n'
                    'loss.backward()\n'
                    "assert loss.item() == 0.0, 'All ignored should return zero loss'\n"
                    "assert logits.grad is not None, 'Zero loss should remain connected to "
                    "logits'\n"},
           {'name': 'Gradient flow',
            'code': '\n'
                    'import torch\n'
                    'logits = torch.randn(2, 3, 2, 5, requires_grad=True)\n'
                    'targets = torch.randint(0, 5, (2, 5))\n'
                    '{fn}(logits, targets).backward()\n'
                    'assert logits.grad is not None and torch.isfinite(logits.grad).all(), '
                    "'Missing or invalid gradients'\n"}]}
