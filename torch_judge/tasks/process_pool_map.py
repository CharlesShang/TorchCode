"""Process Pool Map task."""

TASK = {'title': 'Process Pool Map',
 'difficulty': 'Medium',
 'function_name': 'process_pool_map',
 'hint': 'Use ProcessPoolExecutor and return list(executor.map(...)). In notebook/macOS-style '
         'environments, an explicit fork context can make simple picklable functions easier to '
         'test.',
 'tests': [{'name': 'Built-in function ordered',
            'code': '\n'
                    'out = {fn}(abs, [-3, 2, -1, 0], max_workers=2)\n'
                    'assert out == [3, 2, 1, 0], out\n'},
           {'name': 'Math function',
            'code': '\n'
                    'import math\n'
                    'out = {fn}(math.sqrt, [1.0, 4.0, 9.0], max_workers=2)\n'
                    'assert out == [1.0, 2.0, 3.0], out\n'},
           {'name': 'Empty input', 'code': '\nassert {fn}(abs, [], max_workers=2) == []\n'}]}
