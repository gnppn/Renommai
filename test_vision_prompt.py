#!/usr/bin/env python3
"""
Simple local test to validate parsing of vision model responses.
It loads the prompt, prints it, then simulates a valid and an invalid model response
and checks the parsing logic used by the simplified script (tag-based).
"""

import re
from pathlib import Path

PROMPT_PATH = Path('prompts/vision_prompt_simplifi\u00e9.txt')


def parse_vision_response(full_response):
    """Return a dict with 'text' and 'structured' extracted from response.
    If [TEXT] tags exist extract between them and the rest as structured.
    Otherwise return the whole response as text and try to heuristically
    extract lines starting with Institution/Object/Date as structured.
    """
    if '[TEXT]' in full_response and '[/TEXT]' in full_response:
        text_start = full_response.find('[TEXT]') + len('[TEXT]')
        text_end = full_response.find('[/TEXT]')
        text_content = full_response[text_start:text_end].strip()
        structured = full_response[text_end + len('[/TEXT]'):].strip()
        return {'text': text_content, 'structured': structured}

    # fallback heuristics
    text_content = full_response.strip()
    lines = [l.strip() for l in full_response.splitlines() if l.strip()]
    struct_lines = [l for l in lines if re.match(r'^(Institution|Object|Objet|Date)\b', l, flags=re.IGNORECASE)]
    structured = '\n'.join(struct_lines)
    return {'text': text_content, 'structured': structured}


if __name__ == '__main__':
    print('--- Prompt preview ---')
    try:
        print(PROMPT_PATH.read_text(encoding='utf-8')[:1000])
    except Exception as e:
        print('Could not read prompt:', e)

    print('\n--- Running simulated responses ---\n')

    valid = """[TEXT]
FACTURE EDF
Date d'émission: 15 mars 2024
Montant TTC: 127,84€
Référence client: 123456789
[/TEXT]

Institution 1: Edf
Institution 2: Edf Particuliers
Object 1: Facture Electricite
Object 2: Facture Energie
Date: 2024-03
"""

    invalid = """This is a free-form text output without the tags.
Some header lines and content.
Institution: Edf
Object: Facture
Date: 07/06/13
"""

    for name, resp in [('VALID', valid), ('INVALID', invalid)]:
        print(f'--- {name} RESPONSE ---')
        result = parse_vision_response(resp)
        print('Text extracted:')
        print(result['text'])
        print('\nStructured extracted:')
        print(result['structured'] or '<empty>')
        print('\n')

    print('Test complete.')
