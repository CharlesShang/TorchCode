"""Async Retry Helper task."""

TASK = {'title': 'Async Retry Helper',
 'difficulty': 'Medium',
 'function_name': 'async_retry',
 'hint': 'Call the zero-argument async function. If it raises one of the selected exceptions, '
         'sleep and retry until attempts are exhausted, then re-raise the last error.',
 'tests': [{'name': 'Eventually succeeds',
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
                    "state = {'n': 0}\n"
                    'async def flaky():\n'
                    "    state['n'] += 1\n"
                    "    if state['n'] < 3:\n"
                    "        raise RuntimeError('again')\n"
                    '    return 42\n'
                    'out = run_async({fn}(flaky, retries=3))\n'
                    "assert out == 42 and state['n'] == 3, (out, state)\n"},
           {'name': 'Reraises after retries',
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
                    "state = {'n': 0}\n"
                    'async def always_bad():\n'
                    "    state['n'] += 1\n"
                    "    raise RuntimeError('nope')\n"
                    'try:\n'
                    '    run_async({fn}(always_bad, retries=2))\n'
                    'except RuntimeError as e:\n'
                    "    assert 'nope' in str(e)\n"
                    'else:\n'
                    "    raise AssertionError('Should re-raise final exception')\n"
                    "assert state['n'] == 3, state\n"},
           {'name': 'Does not retry excluded exception',
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
                    "state = {'n': 0}\n"
                    'async def bad_value():\n'
                    "    state['n'] += 1\n"
                    "    raise ValueError('stop')\n"
                    'try:\n'
                    '    run_async({fn}(bad_value, retries=5, exceptions=(RuntimeError,)))\n'
                    'except ValueError:\n'
                    '    pass\n'
                    'else:\n'
                    "    raise AssertionError('Excluded exception should propagate immediately')\n"
                    "assert state['n'] == 1, state\n"}]}
