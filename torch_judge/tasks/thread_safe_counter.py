"""Thread-Safe Counter task."""

TASK = {'title': 'Thread-Safe Counter',
 'difficulty': 'Easy',
 'function_name': 'ThreadSafeCounter',
 'hint': 'Keep the integer private and protect every read/write with threading.Lock. increment '
         'should hold the lock around value += n and return the new value.',
 'tests': [{'name': 'Sequential increments',
            'code': '\n'
                    'c = {fn}(10)\n'
                    'assert c.increment() == 11\n'
                    'assert c.increment(4) == 15\n'
                    'assert c.value() == 15\n'},
           {'name': 'Concurrent increments',
            'code': '\n'
                    'import threading\n'
                    'c = {fn}()\n'
                    'def worker():\n'
                    '    for _ in range(1000):\n'
                    '        c.increment()\n'
                    'threads = [threading.Thread(target=worker) for _ in range(8)]\n'
                    'for t in threads: t.start()\n'
                    'for t in threads: t.join()\n'
                    "assert c.value() == 8000, f'Race condition: {c.value()}'\n"},
           {'name': 'Has lock',
            'code': '\n'
                    'import threading\n'
                    'c = {fn}()\n'
                    'assert any(isinstance(v, type(threading.Lock())) for v in '
                    "c.__dict__.values()), 'Expected a threading.Lock instance'\n"}]}
