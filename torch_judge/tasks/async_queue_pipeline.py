"""Async Queue Pipeline task."""

TASK = {'title': 'Async Queue Pipeline',
 'difficulty': 'Hard',
 'function_name': 'async_queue_pipeline',
 'hint': 'Put (index, item) pairs into an asyncio.Queue, add one sentinel per worker, and have '
         'workers write results into a preallocated list by index.',
 'tests': [{'name': 'Preserves order',
            'code': '\n'
                    'def run_async(coro):\n'
                    '    import asyncio\n'
                    '    import threading\n'
                    '    box = {}\n'
                    '    def target():\n'
                    '        try:\n'
                    "            box['value'] = asyncio.run(coro)\n"
                    '        except BaseException as e:\n'
                    "            box['error'] = e\n"
                    '    t = threading.Thread(target=target)\n'
                    '    t.start()\n'
                    '    t.join(timeout=5)\n'
                    '    if t.is_alive():\n'
                    "        raise TimeoutError('async test timed out')\n"
                    "    if 'error' in box:\n"
                    "        raise box['error']\n"
                    "    return box.get('value')\n"
                    '\n'
                    'import asyncio\n'
                    'async def work(x):\n'
                    '    await asyncio.sleep(0.01 * (5 - x))\n'
                    '    return x * x\n'
                    'out = run_async({fn}(range(6), work, num_workers=3))\n'
                    'assert out == [0, 1, 4, 9, 16, 25], out\n'},
           {'name': 'Uses multiple workers',
            'code': '\n'
                    'def run_async(coro):\n'
                    '    import asyncio\n'
                    '    import threading\n'
                    '    box = {}\n'
                    '    def target():\n'
                    '        try:\n'
                    "            box['value'] = asyncio.run(coro)\n"
                    '        except BaseException as e:\n'
                    "            box['error'] = e\n"
                    '    t = threading.Thread(target=target)\n'
                    '    t.start()\n'
                    '    t.join(timeout=5)\n'
                    '    if t.is_alive():\n'
                    "        raise TimeoutError('async test timed out')\n"
                    "    if 'error' in box:\n"
                    "        raise box['error']\n"
                    "    return box.get('value')\n"
                    '\n'
                    'import asyncio\n'
                    "state = {'active': 0, 'max_active': 0}\n"
                    'async def work(x):\n'
                    "    state['active'] += 1\n"
                    "    state['max_active'] = max(state['max_active'], state['active'])\n"
                    '    await asyncio.sleep(0.02)\n'
                    "    state['active'] -= 1\n"
                    '    return x\n'
                    'out = run_async({fn}(range(8), work, num_workers=4))\n'
                    'assert out == list(range(8))\n'
                    "assert state['max_active'] > 1, state\n"},
           {'name': 'Exception propagates',
            'code': '\n'
                    'def run_async(coro):\n'
                    '    import asyncio\n'
                    '    import threading\n'
                    '    box = {}\n'
                    '    def target():\n'
                    '        try:\n'
                    "            box['value'] = asyncio.run(coro)\n"
                    '        except BaseException as e:\n'
                    "            box['error'] = e\n"
                    '    t = threading.Thread(target=target)\n'
                    '    t.start()\n'
                    '    t.join(timeout=5)\n'
                    '    if t.is_alive():\n'
                    "        raise TimeoutError('async test timed out')\n"
                    "    if 'error' in box:\n"
                    "        raise box['error']\n"
                    "    return box.get('value')\n"
                    '\n'
                    'async def bad(x):\n'
                    '    if x == 2:\n'
                    "        raise RuntimeError('boom')\n"
                    '    return x\n'
                    'try:\n'
                    '    run_async({fn}(range(4), bad, num_workers=2))\n'
                    'except RuntimeError as e:\n'
                    "    assert 'boom' in str(e)\n"
                    'else:\n'
                    "    raise AssertionError('Worker exception should propagate')\n"},
           {'name': 'Empty input',
            'code': '\n'
                    'def run_async(coro):\n'
                    '    import asyncio\n'
                    '    import threading\n'
                    '    box = {}\n'
                    '    def target():\n'
                    '        try:\n'
                    "            box['value'] = asyncio.run(coro)\n"
                    '        except BaseException as e:\n'
                    "            box['error'] = e\n"
                    '    t = threading.Thread(target=target)\n'
                    '    t.start()\n'
                    '    t.join(timeout=5)\n'
                    '    if t.is_alive():\n'
                    "        raise TimeoutError('async test timed out')\n"
                    "    if 'error' in box:\n"
                    "        raise box['error']\n"
                    "    return box.get('value')\n"
                    '\n'
                    'async def work(x): return x\n'
                    'out = run_async({fn}([], work, num_workers=2))\n'
                    'assert out == [], out\n'}]}
