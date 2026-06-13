"""Paged KV Cache task."""

TASK = {'title': 'Paged KV Cache',
 'difficulty': 'Hard',
 'function_name': 'PagedKVCache',
 'hint': 'Keep K/V block tensors plus a page table mapping seq_id to block ids. Append token '
         'chunks into the last block, allocate new blocks as needed, and gather blocks on get().',
 'tests': [{'name': 'Append and retrieve across block boundary',
            'code': '\n'
                    'import torch\n'
                    'cache = {fn}(1, 2, 4, block_size=3, max_blocks=4)\n'
                    'k = torch.randn(5, 2, 4)\n'
                    'v = torch.randn(5, 2, 4)\n'
                    "cache.append(0, 'a', k, v)\n"
                    "got_k, got_v = cache.get(0, 'a')\n"
                    "assert torch.equal(got_k, k), 'Retrieved keys differ from appended keys'\n"
                    "assert torch.equal(got_v, v), 'Retrieved values differ from appended "
                    "values'\n"},
           {'name': 'Separate sequences stay independent',
            'code': '\n'
                    'import torch\n'
                    'cache = {fn}(1, 1, 2, block_size=2, max_blocks=4)\n'
                    'k1, v1 = torch.ones(3, 1, 2), torch.ones(3, 1, 2) * 10\n'
                    'k2, v2 = torch.ones(1, 1, 2) * 2, torch.ones(1, 1, 2) * 20\n'
                    "cache.append(0, 'a', k1, v1)\n"
                    "cache.append(0, 'b', k2, v2)\n"
                    "assert torch.equal(cache.get(0, 'a')[0], k1), 'seq a corrupted'\n"
                    "assert torch.equal(cache.get(0, 'b')[1], v2), 'seq b corrupted'\n"},
           {'name': 'Free releases blocks',
            'code': '\n'
                    'import torch\n'
                    'cache = {fn}(1, 1, 1, block_size=2, max_blocks=2)\n'
                    "cache.append(0, 'a', torch.ones(4, 1, 1), torch.ones(4, 1, 1))\n"
                    "cache.free('a')\n"
                    "cache.append(0, 'b', torch.ones(4, 1, 1) * 2, torch.ones(4, 1, 1) * 3)\n"
                    "assert torch.equal(cache.get(0, 'b')[0], torch.ones(4, 1, 1) * 2), 'Blocks "
                    "should be reusable after free'\n"},
           {'name': 'Capacity error',
            'code': '\n'
                    'import torch\n'
                    'cache = {fn}(1, 1, 1, block_size=2, max_blocks=1)\n'
                    'try:\n'
                    "    cache.append(0, 'a', torch.ones(3, 1, 1), torch.ones(3, 1, 1))\n"
                    'except RuntimeError:\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('Should run out of blocks')\n"}]}
