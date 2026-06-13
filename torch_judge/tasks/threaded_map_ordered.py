"""Ordered Threaded Map task."""

TASK = {'title': 'Ordered Threaded Map',
 'difficulty': 'Medium',
 'function_name': 'threaded_map',
 'hint': 'Use ThreadPoolExecutor.map for the simplest ordered implementation; it preserves input '
         'order and propagates exceptions when consumed.',
 'tests': [{'name': 'Preserves order',
            'code': '\n'
                    'import time\n'
                    'def fn(x):\n'
                    '    time.sleep(0.01 * (5 - x))\n'
                    '    return x * x\n'
                    'out = {fn}(fn, list(range(6)), max_workers=3)\n'
                    'assert out == [0, 1, 4, 9, 16, 25], out\n'},
           {'name': 'Exception propagates',
            'code': '\n'
                    'def bad(x):\n'
                    '    if x == 2:\n'
                    "        raise ValueError('boom')\n"
                    '    return x\n'
                    'try:\n'
                    '    {fn}(bad, [1, 2, 3], max_workers=2)\n'
                    'except ValueError as e:\n'
                    "    assert 'boom' in str(e)\n"
                    'else:\n'
                    "    raise AssertionError('Worker exception should propagate')\n"},
           {'name': 'Empty input',
            'code': '\nassert {fn}(lambda x: x, [], max_workers=2) == []\n'}]}
