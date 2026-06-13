"""Speculative Decoding with Verification task."""

TASK = {'title': 'Speculative Decoding with Verification',
 'difficulty': 'Hard',
 'function_name': 'speculative_decode_verify',
 'hint': 'Generate draft_steps greedy draft tokens. Then replay from the original prefix using the '
         'target model. Accept matching draft tokens; on the first mismatch append the target '
         'token and restart.',
 'tests': [{'name': 'Perfect draft matches target',
            'code': '\n'
                    'import torch\n'
                    'class Inc:\n'
                    '    def __call__(self, tokens):\n'
                    '        logits = torch.zeros(10); logits[(tokens[-1] + 1) % 10] = 1; return '
                    'logits\n'
                    'out = {fn}(Inc(), Inc(), [0], max_new_tokens=5, draft_steps=3)\n'
                    'assert out == [0, 1, 2, 3, 4, 5], out\n'},
           {'name': 'Mismatch falls back to target token',
            'code': '\n'
                    'import torch\n'
                    'class Target:\n'
                    '    def __call__(self, tokens):\n'
                    '        logits = torch.zeros(10); logits[(tokens[-1] + 1) % 10] = 1; return '
                    'logits\n'
                    'class BadDraft:\n'
                    '    def __call__(self, tokens):\n'
                    '        logits = torch.zeros(10); logits[9] = 1; return logits\n'
                    'out = {fn}(Target(), BadDraft(), [0], max_new_tokens=3, draft_steps=2)\n'
                    'assert out == [0, 1, 2, 3], out\n'},
           {'name': 'Stops at EOS',
            'code': '\n'
                    'import torch\n'
                    'class EOSModel:\n'
                    '    def __call__(self, tokens):\n'
                    '        logits = torch.zeros(5); logits[2 if len(tokens) < 3 else 4] = 1; '
                    'return logits\n'
                    'out = {fn}(EOSModel(), EOSModel(), [0], max_new_tokens=10, draft_steps=4, '
                    'eos_token_id=4)\n'
                    'assert out[-1] == 4 and len(out) == 4, out\n'}]}
