from collections import Counter
from typing import NamedTuple, Mapping, Dict, List
from hrgguru.hrg import CFGRule
from hrgguru.hyper_graph import strip_category
from vocab_utils import Dictionary


def encode_nonterminal(edge):
    return "{}#{}".format(edge.label, len(edge.nodes))


class HRGStatistics(NamedTuple("HRGStatistics",
                               [("edge_names", Mapping[str, int]),
                                ("structural_edges", Dictionary),
                                ("categories", Dictionary),
                                ("nonterminals", Dictionary)
                                ])):
    @classmethod
    def from_derivations(cls,
                         derivations  # type: Dict[str, List[CFGRule]]
                         ):
        edge_counter = Counter()
        structural_edges = Dictionary(initial=())
        categories_dict = Dictionary(initial=("*UNK*"))
        nonterminals_dict = Dictionary(initial=())
        for derivation in derivations.values():
            for rule in derivation:
                if rule.hrg is not None:
                    for edge in rule.hrg.rhs.edges:
                        edge_counter.update([edge.label])
                        if edge.is_terminal and len(edge.nodes) == 2:
                            structural_edges.update([edge.label])
                        if edge.is_terminal and len(edge.nodes) == 1:
                            categories_dict.update([strip_category(edge.label)])
                        if not edge.is_terminal:
                            nonterminals_dict.update([encode_nonterminal(edge)])
        return cls(edge_counter, structural_edges, categories_dict, nonterminals_dict)

    def __str__(self):
        return "{} names, {} structual edges, {} categories, {} nonterminals".format(
            len(self.edge_names), len(self.structural_edges),
            len(self.categories), len(self.nonterminals))
