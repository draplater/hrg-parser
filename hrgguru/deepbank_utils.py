import re
from delphin.derivation import Derivation


def parse_spans(span_lines, derivation_str):
    regex = re.compile(r"\((\d+), \d+, \d+, <(\d+):(\d+)>")
    c_spans = {}
    for line in span_lines.split("\n"):
        m = regex.search(line)
        if not m:
            continue
        key, start, end = m.groups()
        c_spans[int(key)] = (int(start), int(end))

    derivation = Derivation.from_string(derivation_str)  # type: Derivation
    # return [c_spans[j.id] for i in derivation.terminals()
    #         for j in i.tokens]
    return [(c_spans[i.tokens[0].id][0], c_spans[i.tokens[-1].id][1])
            for i in derivation.terminals()]
