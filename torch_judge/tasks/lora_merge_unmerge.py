"""LoRA Merge / Unmerge task."""

TASK = {'title': 'LoRA Merge / Unmerge',
 'difficulty': 'Medium',
 'function_name': 'MergeableLoRALinear',
 'hint': 'The LoRA update is (alpha / r) * (B @ A). In unmerged mode add x @ update.T at forward '
         'time. merge() adds update to the base weight once; unmerge() subtracts it.',
 'tests': [{'name': 'Merge preserves output',
            'code': '\n'
                    'import torch\n'
                    'layer = {fn}(8, 4, r=2, alpha=4)\n'
                    'with torch.no_grad():\n'
                    '    layer.lora_B.normal_(0, 0.02)\n'
                    'x = torch.randn(5, 8)\n'
                    'y_before = layer(x)\n'
                    'layer.merge()\n'
                    'y_after = layer(x)\n'
                    "assert layer.merged, 'Layer should be marked merged'\n"
                    "assert torch.allclose(y_before, y_after, atol=1e-6), 'merge should preserve "
                    "output'\n"},
           {'name': 'Unmerge restores weight',
            'code': '\n'
                    'import torch\n'
                    'layer = {fn}(8, 4, r=2, alpha=2)\n'
                    'with torch.no_grad(): layer.lora_B.normal_(0, 0.01)\n'
                    'w0 = layer.weight.detach().clone()\n'
                    'layer.merge(); layer.unmerge()\n'
                    "assert not layer.merged, 'Layer should be unmerged'\n"
                    "assert torch.allclose(layer.weight, w0, atol=1e-7), 'unmerge should restore "
                    "base weight'\n"},
           {'name': 'Base frozen and LoRA trainable',
            'code': '\n'
                    'layer = {fn}(8, 4)\n'
                    "assert not layer.weight.requires_grad, 'base weight should be frozen'\n"
                    "assert layer.lora_A.requires_grad and layer.lora_B.requires_grad, 'LoRA "
                    "params should be trainable'\n"},
           {'name': 'No double merge',
            'code': '\n'
                    'import torch\n'
                    'layer = {fn}(4, 4, r=2)\n'
                    'with torch.no_grad(): layer.lora_B.fill_(0.1)\n'
                    'w0 = layer.weight.detach().clone()\n'
                    'update = layer._update().detach().clone()\n'
                    'layer.merge(); layer.merge()\n'
                    "assert torch.allclose(layer.weight, w0 + update, atol=1e-7), 'merge should be "
                    "idempotent'\n"}]}
