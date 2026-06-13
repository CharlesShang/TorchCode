"""Bounded Blocking Queue task."""

TASK = {'title': 'Bounded Blocking Queue',
 'difficulty': 'Medium',
 'function_name': 'BoundedBlockingQueue',
 'hint': 'Use threading.Condition around a list/deque. put waits while full; get waits while '
         'empty; notify waiting threads after every state change.',
 'tests': [{'name': 'FIFO sequential',
            'code': '\n'
                    'q = {fn}(2)\n'
                    "q.put('a')\n"
                    "q.put('b')\n"
                    'assert q.qsize() == 2\n'
                    "assert q.get() == 'a'\n"
                    "assert q.get() == 'b'\n"
                    'assert q.qsize() == 0\n'},
           {'name': 'Producer consumer threads',
            'code': '\n'
                    'import threading\n'
                    'q = {fn}(3)\n'
                    'produced = list(range(20))\n'
                    'consumed = []\n'
                    'def producer():\n'
                    '    for x in produced:\n'
                    '        q.put(x)\n'
                    'def consumer():\n'
                    '    for _ in produced:\n'
                    '        consumed.append(q.get())\n'
                    't1 = threading.Thread(target=producer)\n'
                    't2 = threading.Thread(target=consumer)\n'
                    't1.start(); t2.start(); t1.join(2); t2.join(2)\n'
                    "assert not t1.is_alive() and not t2.is_alive(), 'Threads deadlocked'\n"
                    "assert consumed == produced, f'{consumed}'\n"},
           {'name': 'get blocks until item',
            'code': '\n'
                    'import threading, time\n'
                    'q = {fn}(1)\n'
                    'box = []\n'
                    'def consumer():\n'
                    '    box.append(q.get())\n'
                    't = threading.Thread(target=consumer)\n'
                    't.start()\n'
                    'time.sleep(0.05)\n'
                    "assert box == [], 'get should block while empty'\n"
                    'q.put(42)\n'
                    't.join(1)\n'
                    'assert box == [42], box\n'}]}
