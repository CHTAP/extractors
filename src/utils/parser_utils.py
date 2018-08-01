"""A simple alternative tokenizer which parses text by splitting on whitespace."""
from builtins import object

import numpy as np

from snorkel.models import construct_stable_id
from snorkel.parser import Parser, ParserConnection

class SimpleTokenizer(Parser):
    """Tokenizes text on whitespace only using split()."""

    def __init__(self, delim="<NB>"):
        self.delim = delim
        super(SimpleTokenizer, self).__init__(name="simple")

    def parse(self, document, contents):
        """Parse the document.
        :param document: The Document context of the data model.
        :param contents: The text contents of the document.
        :rtype: a *generator* of tokenized text.
        """
        i = 0
        offset = 0
        for text in contents.split(self.delim):
            if not len(text.strip()):
                continue
            words = text.split()
            char_offsets = [0] + [
                int(_) for _ in np.cumsum([len(x) + 1 for x in words])[:-1]
            ]
            abs_char_offsets = [idx + offset for idx in char_offsets]

            i += 1
            offset += len(text)

            # Assign the stable id as document's stable id plus absolute
            # character offset
            abs_sent_offset = abs_char_offsets[0]
            abs_sent_offset_end = abs_sent_offset + char_offsets[-1] + len(words[-1])

            text = " ".join(words)
            stable_id = construct_stable_id(document, "sentence", abs_sent_offset, abs_sent_offset_end)
            yield {
                "text": text,
                "words": words,
                "pos_tags": [""] * len(words),
                "ner_tags": [""] * len(words),
                "lemmas": [""] * len(words),
                "dep_parents": [0] * len(words),
                "dep_labels": [""] * len(words),
                "char_offsets": char_offsets,
                "abs_char_offsets": char_offsets,
                "stable_id": stable_id,
                "position": i,
                "document": document,
            }

    def connect(self):
        return ParserConnection(self)
