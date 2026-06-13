"""QLoRA NF4 Quantization task."""

TASK = {'title': 'QLoRA NF4 Quantization',
 'difficulty': 'Hard',
 'function_name': 'NF4Quantizer',
 'hint': 'Normalize each block by its max absolute value, then choose the nearest entry in a fixed '
         '16-value NF4 codebook. Store uint8 codes plus one scale per block.',
 'tests': [{'name': 'Codes and scales',
            'code': '\n'
                    'import torch\n'
                    'q = {fn}(block_size=8)\n'
                    'w = torch.randn(3, 5)\n'
                    'codes, scales, shape = q.quantize(w)\n'
                    "assert codes.dtype == torch.uint8, f'{codes.dtype}'\n"
                    "assert int(codes.min()) >= 0 and int(codes.max()) <= 15, 'codes must be in "
                    "[0, 15]'\n"
                    "assert scales.shape == (2,), f'{scales.shape}'\n"
                    "assert shape == tuple(w.shape), f'{shape}'\n"},
           {'name': 'Roundtrip shape and finite error',
            'code': '\n'
                    'import torch\n'
                    'q = {fn}(block_size=16)\n'
                    'w = torch.randn(7, 9) * 0.5\n'
                    'codes, scales, shape = q.quantize(w)\n'
                    'recon = q.dequantize(codes, scales, shape)\n'
                    "assert recon.shape == w.shape, f'{recon.shape}'\n"
                    "assert torch.isfinite(recon).all(), 'reconstruction contains NaN/Inf'\n"
                    "assert (w - recon).abs().mean() < 0.08, 'NF4 reconstruction error is too "
                    "high'\n"},
           {'name': 'Per-block scale is max abs',
            'code': '\n'
                    'import torch\n'
                    'q = {fn}(block_size=4)\n'
                    'w = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 2.0, 0.0])\n'
                    '_, scales, _ = q.quantize(w)\n'
                    "assert torch.allclose(scales, torch.tensor([2.0, 3.0])), f'{scales}'\n"}]}
