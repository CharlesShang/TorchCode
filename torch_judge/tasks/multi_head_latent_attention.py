"""Multi-Head Latent Attention task."""

TASK = {'title': 'Multi-Head Latent Attention',
 'difficulty': 'Hard',
 'function_name': 'MultiHeadLatentAttention',
 'hint': 'Project queries normally. Compress x to latent_dim for the cache, optionally append to '
         'latent_cache, then up-project cached latents into keys and values for attention.',
 'tests': [{'name': 'Output and cache shapes',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(dim=16, num_heads=4, latent_dim=6)\n'
                    'x = torch.randn(2, 5, 16)\n'
                    'out, cache = m(x)\n'
                    "assert out.shape == x.shape, f'Output shape: {out.shape}'\n"
                    "assert cache.shape == (2, 5, 6), f'Cache shape: {cache.shape}'\n"},
           {'name': 'Appends latent cache',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(16, 4, 6)\n'
                    'x1 = torch.randn(2, 3, 16)\n'
                    'x2 = torch.randn(2, 2, 16)\n'
                    '_, c1 = m(x1)\n'
                    '_, c2 = m(x2, c1)\n'
                    'expected = torch.cat([m.kv_down(x1), m.kv_down(x2)], dim=1)\n'
                    "assert c2.shape == (2, 5, 6), f'{c2.shape}'\n"
                    "assert torch.allclose(c2, expected, atol=1e-6), 'Cache should store appended "
                    "compressed latents'\n"},
           {'name': 'Latent cache is smaller than raw KV',
            'code': '\n'
                    'm = {fn}(dim=32, num_heads=4, latent_dim=8)\n'
                    "assert m.latent_dim < 2 * m.dim, 'latent cache should be smaller than storing "
                    "K and V directly'\n"},
           {'name': 'Gradients flow',
            'code': '\n'
                    'import torch\n'
                    'm = {fn}(8, 2, 4)\n'
                    'x = torch.randn(2, 3, 8, requires_grad=True)\n'
                    'out, cache = m(x)\n'
                    '(out.sum() + cache.sum()).backward()\n'
                    "assert x.grad is not None, 'Missing input gradients'\n"}]}
