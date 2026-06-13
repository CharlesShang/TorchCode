"""MMDiT Joint Attention task."""

TASK = {'title': 'MMDiT Joint Attention',
 'difficulty': 'Hard',
 'function_name': 'MMDiTJointAttention',
 'hint': 'Project image and text tokens with separate qkv layers, concatenate q/k/v along sequence '
         'length, run one attention operation, then split and apply separate output projections.',
 'tests': [{'name': 'Output shapes',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(16, 4)\n'
                    'image = torch.randn(2, 6, 16)\n'
                    'text = torch.randn(2, 3, 16)\n'
                    'image_out, text_out = m(image, text)\n'
                    "assert image_out.shape == image.shape, f'Image shape mismatch: "
                    "{image_out.shape}'\n"
                    "assert text_out.shape == text.shape, f'Text shape mismatch: "
                    "{text_out.shape}'\n"},
           {'name': 'Has separate modality projections',
            'code': '\n'
                    'm = {fn}(8, 2)\n'
                    "for name in ['qkv_img', 'qkv_txt', 'proj_img', 'proj_txt']:\n"
                    "    assert hasattr(m, name), f'Missing {name}'\n"
                    "assert m.qkv_img is not m.qkv_txt, 'image/text qkv projections should be "
                    "separate modules'\n"},
           {'name': 'Text tokens influence image output',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(8, 2)\n'
                    'image = torch.randn(1, 2, 8)\n'
                    'text1 = torch.zeros(1, 2, 8)\n'
                    'text2 = torch.randn(1, 2, 8) * 5\n'
                    'out1, _ = m(image, text1)\n'
                    'out2, _ = m(image, text2)\n'
                    "assert not torch.allclose(out1, out2), 'Image output should attend to text "
                    "tokens'\n"},
           {'name': 'Gradients flow to both modalities',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(8, 2)\n'
                    'image = torch.randn(1, 2, 8, requires_grad=True)\n'
                    'text = torch.randn(1, 3, 8, requires_grad=True)\n'
                    'image_out, text_out = m(image, text)\n'
                    '(image_out.sum() + text_out.sum()).backward()\n'
                    "assert image.grad is not None and text.grad is not None, 'Missing input "
                    "gradients'\n"}]}
