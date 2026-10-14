"""Korean display metadata; execution IDs, input keys and saved enums stay stable."""
import json
from pathlib import Path

DEFINITIONS = json.loads((Path(__file__).parent/'locales/ko/nodeDefs.json').read_text(encoding='utf-8'))
CATEGORIES = {'Koharu/Input':'Koharu/입력','Koharu/Output':'Koharu/출력',
              'Koharu/Detection':'Koharu/감지','Koharu/Model Loaders':'Koharu/모델 로더',
              'Koharu/OCR':'Koharu/OCR','Koharu/Mask':'Koharu/마스크','Koharu/Image':'Koharu/이미지'}


def configure(registry):
    for name, cls in registry.items():
        definition = DEFINITIONS[name]
        cls.DESCRIPTION = definition['description']
        cls.CATEGORY = CATEGORIES.get(cls.CATEGORY,cls.CATEGORY)
        original = cls.INPUT_TYPES.__func__
        def schema(node_cls, _original=original, _definition=definition):
            values = _original(node_cls)
            translated = dict(values)
            for section in ('required','optional'):
                if section not in values:
                    continue
                fields = {}
                for key, spec in values[section].items():
                    metadata = _definition['inputs'].get(key,{})
                    options = dict(spec[1]) if len(spec)>1 else {}
                    if metadata.get('tooltip'):
                        options['tooltip'] = metadata['tooltip']
                    fields[key] = (spec[0],options,*spec[2:])
                translated[section] = fields
            return translated
        cls.INPUT_TYPES = classmethod(schema)
    return {name:DEFINITIONS[name]['display_name'] for name in registry}
