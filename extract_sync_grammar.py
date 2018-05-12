from __future__ import unicode_literals

import traceback
from io import open

import pickle
from collections import Counter
import gzip

import os
from itertools import zip_longest

from multiprocessing import Pool

import re
from random import Random

import six

from delphin.mrs import eds, simplemrs
from derivation_analysis import SampleCounter
from hrgguru.const_tree import ConstTree, Lexicon
from hrgguru.hrg import CFGRule, HRGDerivation
from hrgguru.hyper_graph import HyperGraph, HyperEdge, GraphNode, strip_category

deepbank_export_path = "../large-data/hrg/deepbank_export_1.1/"
main_dir_base = "../large-data/hrg/"

DONT_STRIP = 0
STRIP_ALL_LABELS = 1
STRIP_TO_UNLABEL = 2
FUZZY_TREE = 3
STRIP_INTERNAL_LABELS = 5


def fix_punct_hyphen(tree):
    span = tree.span
    words = list(tree.generate_words())
    if all(i.span == span or i.span[1] - i.span[0] == 0
           for i in words) and sum(1 for i in words if i.span == span) >= 2:
        lexicon = Lexicon("".join(i.string for i in words))
        lexicon.span = span
        tree.child = [lexicon]
    elif isinstance(tree.child[0], ConstTree):
        for i in tree.child:
            assert isinstance(i, ConstTree)
            fix_punct_hyphen(i)


def strip_label(tree):
    if isinstance(tree, ConstTree):
        tree.tag = tree.tag.split("_")[0]
        for i in tree.child:
            strip_label(i)


def strip_label_internal(tree):
    if isinstance(tree, ConstTree) and isinstance(tree.child[0], ConstTree):
        tree.tag = tree.tag.split("_")[0]
        for i in tree.child:
            strip_label_internal(i)


def strip_unary(node):
    while len(node.child) == 1 and \
            isinstance(node.child[0], ConstTree) and node.tag == node.child[0].tag:
        node.child = node.child[0].child
    for sub_tree in node.child:
        if isinstance(sub_tree, ConstTree):
            strip_unary(sub_tree)


def strip_to_unlabel(node):
    while len(node.child) == 1 and isinstance(node.child[0], ConstTree):
        node.child = node.child[0].child
    node.tag = "X"
    for sub_tree in node.child:
        if isinstance(sub_tree, ConstTree):
            strip_to_unlabel(sub_tree)


def extract_features(hg,  # type: HyperGraph
                     cfg,  # type: ConstTree
                     log_func=print
                     ):
    delphin_span_to_word_span = {}
    for idx, node in enumerate(cfg.generate_words()):
        node.word_span = delphin_span_to_word_span[node.span] = (idx, idx + 1)
    for idx, node in enumerate(cfg.generate_rules()):
        node.word_span = delphin_span_to_word_span[node.span] = (
            node.child[0].word_span[0], node.child[-1].word_span[1])

    node_mapping = {}  # node -> pred edge
    real_edges = []
    for edge in hg.edges:  # type: HyperEdge
        if len(edge.nodes) == 1:
            main_node = edge.nodes[0]  # type: GraphNode
            if node_mapping.get(main_node) is not None:
                log_func("Dumplicate node name {} and {}!".format(
                    node_mapping[main_node],
                    edge.label
                ))
                continue
            if not edge.is_terminal:
                log_func("non-terminal edge {} found.".format(edge.label))
            node_mapping[main_node] = edge
        elif len(edge.nodes) == 2:
            real_edges.append(edge)
        else:
            log_func("Invalid hyperedge with node count {}".format(len(edge.nodes)))

    names = []
    args = []
    for node, pred_edge in node_mapping.items():
        assert pred_edge.span is not None
        names.append((delphin_span_to_word_span[pred_edge.span], strip_category(pred_edge.label)))

    for edge in real_edges:
        pred_edges = [node_mapping.get(i) for i in edge.nodes]
        if any(i is None for i in pred_edges):
            log_func("No span for edge {}, nodes {}!".format(edge, pred_edges))
            continue
        args.append((delphin_span_to_word_span[pred_edges[0].span], strip_category(pred_edges[0].label),
                     delphin_span_to_word_span[pred_edges[1].span], strip_category(pred_edges[1].label),
                     edge.label))
    return set(names), set(args)


def span_overlap(a, b):
    return a != b and a[0] >= b[0] and a[1] <= b[1]


def fuzzy_cfg(cfg, names):
    random_obj = Random(45)
    spans = {i[0] for i in names}
    words = list(cfg.generate_words())

    def wrap_word(span):
        ret = ConstTree("X")
        ret.word_span = span
        ret.child.append(words[span[0]])
        return ret

    def make_sub_tree(span):
        ret = ConstTree("X")
        ret.word_span = span
        if span[1] - span[0] == 1:
            return wrap_word(span)
        else:
            return ret

    sub_trees = [make_sub_tree(i) for i in spans]
    sub_trees.sort(key=lambda x: x.word_span[1] - x.word_span[0], reverse=True)

    top_trees = []
    while len(sub_trees) > 1:
        this_tree = sub_trees[-1]
        parent_tree = None
        for other_tree in sub_trees[:-1]:
            if span_overlap(this_tree.word_span, other_tree.word_span):
                if parent_tree is None or span_overlap(other_tree.word_span, parent_tree.word_span):
                    parent_tree = other_tree
        if parent_tree is None:
            top_trees.append(this_tree)
        else:
            parent_tree.child.append(this_tree)
        sub_trees.pop()

    if len(sub_trees) == 0:
        root = sub_trees[0]
        if root.word_span[1] - root.word_span[0] != len(words):
            new_root = ConstTree("X")
            new_root.child.append(root)
            root = new_root
    else:
        root = ConstTree("X")
        root.word_span = (0, len(words))
        root.child = sub_trees

    def sort_and_fill_blank(node):
        if not node.child:
            node.child = [wrap_word((i, i + 1)) for i in range(*node.word_span)]
        elif isinstance(node.child[0], ConstTree):
            node.child.sort(key=lambda x: x.word_span)
            new_child_list = []
            for i in range(node.word_span[0], node.child[0].word_span[0]):
                new_child_list.append(wrap_word((i, i + 1)))
            for child_node, next_child_node in zip_longest(node.child, node.child[1:]):
                new_child_list.append(child_node)
                end = next_child_node.word_span[0] if next_child_node is not None else node.word_span[1]
                for i in range(child_node.word_span[1], end):
                    new_child_list.append(wrap_word((i, i + 1)))
            origin_children = node.child
            node.child = new_child_list
            for child in origin_children:
                sort_and_fill_blank(child)

    sort_and_fill_blank(root)

    def random_merge(node):
        children = node.child
        for child_node in children:
            if isinstance(child_node, ConstTree):
                random_merge(child_node)
            else:
                assert len(children) == 1
        while len(children) > 2:
            idx = random_obj.randint(0, len(children) - 2)
            tree_a = children[idx]
            tree_b = children[idx + 1]
            new_tree = ConstTree("X")
            new_tree.word_span = (tree_a.word_span[0], tree_b.word_span[1])
            new_tree.child = [tree_a, tree_b]
            children[idx] = new_tree
            children.pop(idx + 1)

    random_merge(root)
    root.populate_spans_internal()
    return root


def mapper(options):
    main_dir, bank, strip_tree, is_train, graph_type, detect_func_name = options
    detect_func = {"small": HRGDerivation.detect_small,
                   "large": HRGDerivation.detect_large,
                   "lexicalized": HRGDerivation.detect_lexicalized}[detect_func_name]
    result = []
    with open(main_dir + bank, encoding="utf-8") as f:
        if bank.startswith("."):
            return
        while True:
            sent_id = f.readline().strip()
            if not sent_id:
                break
            assert sent_id.startswith("#")
            sent_id = sent_id[1:]
            tree_literal = f.readline().strip()
            try:
                with gzip.open(deepbank_export_path + bank + "/" + sent_id + ".gz",
                               "rb") as f_gz:
                    contents = f_gz.read().decode("utf-8")
                cfg = ConstTree.from_java_code_deepbank_1_1(tree_literal, contents)

                # strip labels
                if strip_tree == STRIP_ALL_LABELS or strip_tree == STRIP_INTERNAL_LABELS:
                    if strip_tree == STRIP_ALL_LABELS:
                        strip_label(cfg)
                    elif strip_tree == STRIP_INTERNAL_LABELS:
                        strip_label_internal(cfg)
                    strip_unary(cfg)
                elif strip_tree == STRIP_TO_UNLABEL or strip_tree == FUZZY_TREE:
                    strip_to_unlabel(cfg)

                cfg = cfg.condensed_unary_chain()
                cfg.populate_spans_internal()
                fix_punct_hyphen(cfg)
                fields = contents.strip().split("\n\n")
                if graph_type == "eds":
                    eds_literal = fields[-2]
                    eds_literal = re.sub("\{.*\}", "", eds_literal)
                    e = eds.loads_one(eds_literal)
                    hg = HyperGraph.from_eds(e)
                elif graph_type == "dmrs":
                    mrs_literal = fields[-3]
                    mrs_obj = simplemrs.loads_one(mrs_literal)
                    hg = HyperGraph.from_mrs(mrs_obj)
                else:
                    raise Exception("Invalid graph type!")
                names, args = extract_features(hg, cfg)
                if strip_tree == 3:
                    cfg = fuzzy_cfg(cfg, names)
                derivations = CFGRule.extract(hg, cfg,
                                              # draw=True,
                                              sent_id=sent_id,
                                              detect_func=detect_func)
                sent_id_info = "# ID: " + sent_id + "\n"
                span_info = "# DelphinSpans: " + repr(
                    [i.span for i in cfg.generate_words()]) + "\n"
                args_info = "# Args: " + repr(list(args)) + "\n"
                names_info = "# Names: " + repr(list(names)) + "\n"
                header = sent_id_info + span_info + args_info + names_info
                original_cfg = cfg.to_string(with_comma=False).replace("+++", "+!+")
                rules = list(cfg.generate_rules())
                assert len(derivations) == len(rules)
                for syn_rule, cfg_rule in zip(derivations, rules):
                    assert cfg_rule.tag == syn_rule.lhs
                    new_name = "{}#{}".format(cfg_rule.tag,
                                              len(syn_rule.hrg.lhs.nodes) \
                                                  if syn_rule.hrg is not None else 0)
                    cfg_rule.tag = new_name
                additional_cfg = cfg.to_string(with_comma=False).replace("+++", "+!+")
                if any(rule
                       for rule in cfg.generate_rules() if len(rule.child) > 2):
                    if is_train:
                        print("{} Not binary tree!".format(sent_id))
                    else:
                        raise Exception("Not binary tree!")
                result.append((sent_id, derivations, header + original_cfg,
                               header + additional_cfg))
            except Exception as e:
                print(sent_id)
                print(e.__class__.__name__)
                result.append((sent_id, None, None, None))
                traceback.print_exc()
    return bank, result


def extract_sync_grammer(java_out_dir, output_file, output_fulllabel_file,
                         name, strip_tree=DONT_STRIP, limit=None,
                         graph_type="eds",
                         detect_func="small"):
    derivations = {}
    banks = sorted(os.listdir(java_out_dir))
    if limit is not None:
        banks = banks[:limit]
    pool = Pool(processes=8)

    all_rules = {}

    def convert_derivation(derivation):
        """ convert multiple object of the same rule into one object"""
        for rule in derivation:
            standard_rule = all_rules.get(rule)
            if standard_rule is None:
                standard_rule = all_rules[rule] = rule
            yield standard_rule

    is_train = "train" in java_out_dir
    all_options = [(java_out_dir, i, strip_tree, is_train,
                    graph_type, detect_func) for i in banks]

    results_list = list(pool.imap_unordered(mapper, all_options))
    pool.terminate()
    results_list.sort(key=lambda x: x[0])

    with open(output_file, "w") as f_1, \
            open(output_fulllabel_file, "w") as f_2:
        for bank, results in results_list:
            print("Processing " + bank)
            for result in results:
                sent_id, derivation, original_cfg, additional_cfg = result
                if derivation is not None:
                    derivations[sent_id] = list(convert_derivation(derivation))
                    f_1.write(original_cfg + "\n")
                    f_2.write(additional_cfg + "\n")

    if not is_train:
        return

    # write derivations
    with open("deepbank-preprocessed/derivations-{}.pickle".format(name), "wb") as f:
        pickle.dump(derivations, f)

    # write count
    rule_count = SampleCounter()
    for file_name, steps in derivations.items():
        for step, rule in enumerate(steps):
            rule_count.add(rule, (file_name, step))

    with open("deepbank-preprocessed/count-{}.pickle".format(name), "wb") as f:
        pickle.dump(rule_count, f)

    # write SHRG
    from collections import defaultdict
    counter = defaultdict(Counter)
    for rule, (count, example) in rule_count.items():
        # if all(i.is_terminal for i in rule.hrg.rhs.edges):
        #     continue
        if rule.rhs is None:
            continue

        def format_tag_and_edge(tag, edge):
            if isinstance(tag, Lexicon):
                return tag
            assert isinstance(tag, six.string_types)
            return "{}#{}".format(tag.replace("+++", "+!+"),
                                  0 if edge is None else len(edge.nodes))

        external_point_count = len(rule.hrg.lhs.nodes) if rule.hrg is not None else 0
        cfg = ("{}#{}".format(rule.lhs.replace("+++", "+!+"), external_point_count),
               tuple(format_tag_and_edge(tag, edge) for tag, edge in rule.rhs))
        counter[cfg][rule] = count

    with open("deepbank-preprocessed/cfg_hrg_mapping-{}.pickle".format(name), "wb") as f:
        pickle.dump(counter, f)

    sorted_counter = sorted(((k, len(v)) for k, v in counter.items()), key=lambda x: x[1])

    X = []
    Y = []
    for idx, (hrg, count) in enumerate(reversed(sorted_counter)):
        X.append(idx)
        Y.append(count)


def extract_banks(prefix, strip_tree=DONT_STRIP, graph_type="eds",
                  limit=None, detect_func="small"):
    for mode in ("train", "dev", "test"):
        main_dir = main_dir_base + "/java_out_" + mode + "/"
        output_file = "./deepbank-preprocessed/" + prefix + "." + mode
        output_fulllabel_file = "./deepbank-preprocessed/" + prefix + ".fulllabel." + mode
        extract_sync_grammer(main_dir, output_file, output_fulllabel_file,
                             prefix, strip_tree=strip_tree, graph_type=graph_type,
                             limit=limit, detect_func=detect_func)


if __name__ == '__main__':
    extract_banks("deepbank1.1-lite-test", STRIP_ALL_LABELS)
    # extract_banks("deepbank1.1-lite-all-lexicalizedg-fix", STRIP_ALL_LABELS, detect_func="lexicalized")
    #extract_banks("deepbank1.1-lite-internal", STRIP_INTERNAL_LABELS)
    # extract_banks("deepbank1.1-dmrs-unlabeled", 2, graph_type="dmrs")
    # extract_banks("deepbank1.1-unlabeled", 2, graph_type="eds")
    # extract_banks("deepbank1.1-fuzzy-3", FUZZY_TREE, graph_type="eds")
