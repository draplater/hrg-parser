import pickle
from collections import Counter, defaultdict
from nltk.stem import WordNetLemmatizer

from hrgguru.hrg import HRGRule

wordnet_lemma = WordNetLemmatizer()


from hrgguru.const_tree import Lexicon
from hrgguru.hyper_graph import HyperEdge, HyperGraph

name = "deepbank1.1-lite-all-lexicalizedg-fix"
with open("deepbank-preprocessed/cfg_hrg_mapping-{}.pickle".format(name), "rb") as f:
    cfg_hrg_mapping = pickle.load(f)

unlexicalized_rules = defaultdict(Counter)


def transform_edge(edge, lexicon):
    assert edge.is_terminal
    if not edge.label.startswith("_"):
        return edge
    label = edge.label
    lemma_start = label.find("_") + 1
    lemma_end = label.find("_", lemma_start)
    tag_end = label.find("_", lemma_end + 1)
    lemma_end_slash = label.rfind("/", lemma_start, lemma_end)
    if lemma_end_slash != -1:
        lemma_end = lemma_end_slash
    old_lemma = label[lemma_start:lemma_end]
    pos = label[lemma_end+1:tag_end]
    if tag_end != -1 and pos in ("n", "v", "a"):
        pred_lemma = wordnet_lemma.lemmatize(lexicon, pos)
        if old_lemma != pred_lemma:
            raise ArithmeticError("Unmatch lemma {} {}".format(old_lemma, pred_lemma))
    else:
        if lexicon != old_lemma:
            raise ArithmeticError("{} {} {}".format(lexicon, old_lemma, label))
    transformed_label = label[:lemma_start] + "{NEWLEMMA}" + label[lemma_end:]
    new_edge = HyperEdge(edge.nodes, transformed_label,
                         edge.is_terminal, edge.span)
    return new_edge


for (key_lhs, key_rhs), sync_rules in cfg_hrg_mapping.items():
    for sync_rule, count in sync_rules.items():
        if isinstance(key_rhs[0], Lexicon) and sync_rule.hrg is not None:
            assert len(key_rhs) == 1
            word = key_rhs[0].string
            postag = key_lhs
            origin_lhs = sync_rule.hrg.lhs
            sub_graph = sync_rule.hrg.rhs
            try:
                new_subgraph = HyperGraph(sub_graph.nodes,
                                          frozenset(transform_edge(edge, word)
                                                    for edge in sub_graph.edges)
                                          )
                standard_new_subgraph, node_map = new_subgraph.to_standardized_node_names(True)
                new_lhs = HyperEdge([node_map[i] for i in origin_lhs.nodes],
                                    origin_lhs.label, origin_lhs.is_terminal,
                                    origin_lhs.span)
                new_rule = HRGRule(new_lhs, standard_new_subgraph)
                unlexicalized_rules[postag].update([new_rule])
            except ArithmeticError as e:
                print(e)

for k, v in unlexicalized_rules.items():
    unlexicalized_rules[k] = dict(v.most_common(20))

with open("deepbank-preprocessed/unlexicalized-{}.pickle".format(name), "wb") as f:
    pickle.dump(unlexicalized_rules, f)
