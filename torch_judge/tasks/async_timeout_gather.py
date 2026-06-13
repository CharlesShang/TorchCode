"""Async Gather with Timeout Defaults task."""

TASK = {'title': 'Async Gather with Timeout Defaults',
 'difficulty': 'Medium',
 'function_name': 'gather_with_timeout',
 'hint': 'Wrap every coroutine in asyncio.wait_for. Catch asyncio.TimeoutError and return the '
         'default value for that specific coroutine; gather the wrappers to preserve order.',
 'tests': [{'name': 'Timeout returns default',
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
                    'async def work(delay, value):\n'
                    '    await asyncio.sleep(delay)\n'
                    '    return value\n'
                    "out = run_async({fn}([work(0.01, 'a'), work(0.1, 'b')], timeout=0.03, "
                    "default='timeout'))\n"
                    "assert out == ['a', 'timeout'], out\n"},
           {'name': 'Preserves order',
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
                    '    await asyncio.sleep(0.01 * (3 - x))\n'
                    '    return x\n'
                    'out = run_async({fn}([work(0), work(1), work(2)], timeout=0.1))\n'
                    'assert out == [0, 1, 2], out\n'},
           {'name': 'Non-timeout exception propagates',
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
                    'async def bad():\n'
                    "    raise ValueError('bad')\n"
                    'try:\n'
                    '    run_async({fn}([bad()], timeout=0.1))\n'
                    'except ValueError as e:\n'
                    "    assert 'bad' in str(e)\n"
                    'else:\n'
                    "    raise AssertionError('Non-timeout exception should propagate')\n"}]}
