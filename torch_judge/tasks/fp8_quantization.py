"""FP8 Quantization Basics task."""

TASK = {'title': 'FP8 Quantization Basics',
 'difficulty': 'Medium',
 'function_name': 'FP8Quantizer',
 'hint': 'Use one dynamic scale: scale = amax(abs(x)) / max_fp8. Quantized values are '
         'rounded/clamped integer codes that dequantize with code * scale.',
 'tests': [{'name': 'Code range and scalar scale',
            'code': '\n'
                    'import torch\n'
                    'qz = {fn}()\n'
                    'x = torch.randn(100)\n'
                    'q, scale = qz.quantize(x)\n'
                    "assert q.dtype == torch.int16, f'{q.dtype}'\n"
                    "assert int(q.min()) >= -448 and int(q.max()) <= 448, 'codes outside FP8-style "
                    "range'\n"
                    "assert scale.dim() == 0, 'scale should be scalar'\n"},
           {'name': 'Roundtrip is close',
            'code': '\n'
                    'import torch\n'
                    'qz = {fn}()\n'
                    'x = torch.linspace(-2, 2, 100)\n'
                    'q, scale = qz.quantize(x)\n'
                    'recon = qz.dequantize(q, scale)\n'
                    "assert recon.shape == x.shape, f'{recon.shape}'\n"
                    "assert (x - recon).abs().mean() < 0.01, 'Dynamic quantization error too "
                    "high'\n"},
           {'name': 'All-zero tensor',
            'code': '\n'
                    'import torch\n'
                    'qz = {fn}()\n'
                    'x = torch.zeros(8)\n'
                    'q, scale = qz.quantize(x)\n'
                    'recon = qz.dequantize(q, scale)\n'
                    "assert torch.equal(q, torch.zeros_like(q)), 'zero should quantize to zero'\n"
                    "assert torch.equal(recon, x), 'zero should dequantize to zero'\n"}]}
