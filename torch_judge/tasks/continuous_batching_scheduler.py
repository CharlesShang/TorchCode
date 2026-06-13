"""Continuous Batching Scheduler task."""

TASK = {'title': 'Continuous Batching Scheduler',
 'difficulty': 'Hard',
 'function_name': 'ContinuousBatchingScheduler',
 'hint': 'Track requests as waiting_prefill or decoding. Each step schedules up to max_batch_size '
         'requests, prefilling waiting requests first, then decoding active ones. Decode '
         'increments generated count and finishes at max_new_tokens.',
 'tests': [{'name': 'Prefill before decode',
            'code': '\n'
                    's = {fn}(max_batch_size=2)\n'
                    "s.add_request('a', [1, 2], 2)\n"
                    "s.add_request('b', [3], 1)\n"
                    'step1 = s.step()\n'
                    "assert step1 == {'prefill': ['a', 'b'], 'decode': [], 'finished': []}, step1\n"
                    'step2 = s.step()\n'
                    "assert step2['decode'] == ['a', 'b'] and step2['finished'] == ['b'], step2\n"},
           {'name': 'Max batch size is respected',
            'code': '\n'
                    's = {fn}(max_batch_size=1)\n'
                    "s.add_request('a', [1], 1)\n"
                    "s.add_request('b', [2], 1)\n"
                    "assert s.step()['prefill'] == ['a'], 'Only one request should be scheduled'\n"
                    "assert s.step()['prefill'] == ['b'], 'Second request should prefill next'\n"},
           {'name': 'New request joins while old decodes',
            'code': '\n'
                    's = {fn}(max_batch_size=2)\n'
                    "s.add_request('a', [1], 3)\n"
                    's.step()  # prefill a\n'
                    "s.add_request('b', [2], 1)\n"
                    'step = s.step()\n'
                    "assert step['prefill'] == ['b'], step\n"
                    "assert step['decode'] == ['a'], step\n"},
           {'name': 'Finishes all requests',
            'code': '\n'
                    's = {fn}(max_batch_size=4)\n'
                    "s.add_request('a', [1], 2)\n"
                    'seen_finished = []\n'
                    'for _ in range(5):\n'
                    '    out = s.step()\n'
                    "    seen_finished.extend(out['finished'])\n"
                    '    if not s.has_pending():\n'
                    '        break\n'
                    "assert seen_finished == ['a'], seen_finished\n"
                    "assert not s.has_pending(), 'Scheduler should be empty'\n"}]}
