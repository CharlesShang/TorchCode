"""Prefix Cache task."""

TASK = {'title': 'Prefix Cache',
 'difficulty': 'Medium',
 'function_name': 'PrefixCache',
 'hint': 'Store cached prompts as tuples of token ids. For lookup, scan cached prompts and choose '
         'the longest one that equals the beginning of the requested prompt.',
 'tests': [{'name': 'Exact hit',
            'code': '\n'
                    'import torch\n'
                    'c = {fn}()\n'
                    'k = torch.randn(3, 2)\n'
                    'v = torch.randn(3, 2)\n'
                    'c.add([1, 2, 3], k, v)\n'
                    'matched, got_k, got_v = c.lookup([1, 2, 3])\n'
                    "assert matched == 3, f'{matched}'\n"
                    "assert torch.equal(got_k, k) and torch.equal(got_v, v), 'KV mismatch'\n"},
           {'name': 'Longest prefix wins',
            'code': '\n'
                    'import torch\n'
                    'c = {fn}()\n'
                    'c.add([1, 2], torch.ones(2, 1), torch.ones(2, 1))\n'
                    'c.add([1, 2, 3], torch.ones(3, 1) * 3, torch.ones(3, 1) * 4)\n'
                    'matched, k, v = c.lookup([1, 2, 3, 9])\n'
                    "assert matched == 3, f'{matched}'\n"
                    "assert torch.equal(k, torch.ones(3, 1) * 3), 'Should return longest matching "
                    "prefix'\n"},
           {'name': 'Miss',
            'code': '\n'
                    'import torch\n'
                    'c = {fn}()\n'
                    'c.add([1, 2], torch.ones(2, 1), torch.ones(2, 1))\n'
                    'matched, k, v = c.lookup([2, 1])\n'
                    "assert matched == 0 and k is None and v is None, 'Expected cache miss'\n"},
           {'name': 'Lookup returns clones',
            'code': '\n'
                    'import torch\n'
                    'c = {fn}()\n'
                    'c.add([1], torch.ones(1, 1), torch.ones(1, 1))\n'
                    '_, k, _ = c.lookup([1, 2])\n'
                    'k.fill_(99)\n'
                    '_, k2, _ = c.lookup([1, 2])\n'
                    "assert torch.equal(k2, torch.ones(1, 1)), 'Caller mutation should not alter "
                    "cache'\n"}]}
