import gzip
import pickle
import random
import traceback
import typing
import weakref
from argparse import ArgumentParser
from ast import literal_eval
from itertools import chain, product, zip_longest
from operator import itemgetter
from pprint import pformat, pprint
from typing import List, Mapping, Tuple, Optional, Dict, Union, Iterable

import dynet as dn
import sys

import re

import os

import time
import attr

from nltk.stem import WordNetLemmatizer

import nn
from beam import Beam
from common_utils import AttrDict, set_proc_name, ensure_dir, smart_open, split_to_batches, IdentityDict
from delphin.mrs import eds, simplemrs
from delphin.mrs.components import nodes as mrs_nodes, links as mrs_links
from hrgguru.eds import ScorerResult, EDSScorer
from hrgguru.hrg import CFGRule, HRGRule
from hrgguru.const_tree import Lexicon as HLexicon
from hrgguru.hyper_graph import GraphNode, HyperEdge, HyperGraph, strip_category
from hrgguru.sub_graph import SubGraph
from logger import logger, log_to_file
from parser_base import DependencyParserBase
from span.const_tree import ConstTree, Lexicon
from collections import Counter, deque, defaultdict

from span.embedding_based_scorer import EmbeddingHRGScorer
from span.feature_based_scorer import StructuredPeceptronHRGScorer
from span.hrg_statistics import HRGStatistics
from tagger_base.network import POSTagClassification


@attr.s(slots=True, cmp=False)
class BeamItem(object):
    node_info_ref = attr.ib()  # type: weakref
    sync_rule = attr.ib(type=CFGRule)
    sub_graph = attr.ib(type=SubGraph)
    score = attr.ib(type=float)
    left = attr.ib()  # type: Optional[BeamItem]
    right = attr.ib()  # type: Optional[BeamItem]
    is_gold = attr.ib()
    own_features = attr.ib(type=frozenset)
    total_features = attr.ib(type=frozenset)

    def __lt__(self, other):
        return self.score < other.score


@attr.s(slots=True)
class TraversalSequenceItem(object):
    __weakref__ = attr.ib(init=False, hash=False, repr=False, cmp=False)
    cfg_node = attr.ib(type=ConstTree)
    gold_rule = attr.ib(default=None)
    rule_getter = attr.ib(default=None)
    beam = attr.ib(type=Iterable[BeamItem], default=None)
    correspondents = attr.ib(type=Dict[CFGRule, dn.Expression], default=None)
    left = attr.ib(default=None)
    right = attr.ib(default=None)
    gold_item = attr.ib(default=None)
    early_updated = attr.ib(default=False)
    is_preterminal = attr.ib(type=bool, init=False)

    def __attrs_post_init__(self):
        self.is_preterminal = isinstance(self.cfg_node.children[0], Lexicon)


def output_hg(sent_id, hg):
    # draw eds
    node_mapping = {}
    real_edges = []
    nodes = []
    edges = []
    for edge in hg.edges:  # type: HyperEdge
        if len(edge.nodes) == 1:
            main_node = edge.nodes[0]  # type: GraphNode
            if node_mapping.get(main_node) is None:
                node_mapping[main_node] = edge
            else:
                print("Dumplicate node name {} and {}!".format(
                    node_mapping[main_node],
                    edge.label
                ))
        elif len(edge.nodes) == 2:
            real_edges.append(edge)
        else:
            print("Invalid hyperedge with node count {}".format(len(edge.nodes)))

    for node, pred_edge in node_mapping.items():
        assert pred_edge.span is not None
        nodes.append((node.name, pred_edge.label))

    for edge in real_edges:
        node_1, node_2 = edge.nodes
        pred_edges = [node_mapping.get(i) for i in edge.nodes]
        if any(i is None for i in pred_edges):
            print("No span for edge {}, nodes {}!".format(edge, pred_edges))
            continue
        edges.append((node_1.name, node_2.name, edge.label))

    return "#{}\n{}\n".format(sent_id, len(nodes)) + \
           "\n".join(" ".join(i) for i in nodes) + "\n" + \
           "{}\n".format(len(edges)) + \
           "\n".join(" ".join(i) for i in edges) + "\n\n"


def eds_for_smatch(sent_id, e):
    nodes = [(node.nodeid, str(node.pred)) for node in e.nodes()]

    edges = [(node.nodeid, target, label)
             for node in e.nodes()
             for label, target in e.edges(node.nodeid).items()]

    return "#{}\n{}\n".format(sent_id, len(nodes)) + \
           "\n".join(" ".join(i) for i in nodes) + "\n" + \
           "{}\n".format(len(edges)) + \
           "\n".join(" ".join(i) for i in edges) + "\n\n"


def mrs_for_smatch(sent_id, m):
    nodes = [(str(node.nodeid), str(node.pred)) for node in m.eps()]

    edges = [(str(start), str(end), rargname + "/" + post)
             for start, end, rargname, post in mrs_links(m)
             if start != 0]

    return "#{}\n{}\n".format(sent_id, len(nodes)) + \
           "\n".join(" ".join(i) for i in nodes) + "\n" + \
           "{}\n".format(len(edges)) + \
           "\n".join(" ".join(i) for i in edges) + "\n\n"


@attr.s(slots=True)
class PrintLogger(object):
    total_count = attr.ib(default=sys.float_info.epsilon)
    correct_count = attr.ib(default=0)
    total_loss = attr.ib(default=0)
    total_predict_score = attr.ib(default=0)
    total_gold_score = attr.ib(default=0)

    def print(self, idx):
        logger.info("Sent {}, Correctness: {:.2f}, Pred: {:.2f}, Gold: {:.2f},"
                    "loss: {:.2f}".format(
            idx + 1,
            self.correct_count / self.total_count * 100,
            self.total_predict_score,
            self.total_gold_score,
            self.total_loss))
        self.total_count = sys.float_info.epsilon
        self.correct_count = 0
        self.total_loss = 0
        self.total_predict_score = 0
        self.total_gold_score = 0


empty_beam_item = BeamItem(None, None, None, 0.0, None, None, True, frozenset(), frozenset())


class EdgeEvaluation(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--edge-bilinear-dim", type=int, dest="edge_bilinear_dim",
                           default=100)
        group.add_argument("--edge-mlp-dims", dest="edge_mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[])

    def __init__(self, model, tag_dict, options):
        super().__init__(model)
        self.activation = nn.activations[options.activation]
        self.bilinear_layer = nn.BiLinear(self,
                                          options.lstm_dims * 2 * options.span_lstm_layers + options.category_embedding,
                                          options.edge_bilinear_dim)
        dense_dims = [options.edge_bilinear_dim] + options.mlp_dims + \
                     [len(tag_dict)]
        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation)

    def __call__(self, source_vec, target_vec):
        return self.dense_layer(self.bilinear_layer(source_vec, target_vec))

    def restore_components(self, restored_params):
        self.bilinear_layer, self.dense_layer = restored_params


class UdefQParser(DependencyParserBase):
    DataType = ConstTree
    scorers = {"feature": StructuredPeceptronHRGScorer, "embedding": EmbeddingHRGScorer}

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        super(UdefQParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--span-model",
                           dest="span_model_prefix",
                           required=True)
        group.add_argument("--span-model-format", dest="span_model_format",
                           choices=nn.model_formats, default=None)
        group.add_argument("--derivations", dest="derivations", required=True)
        group.add_argument("--grammar", dest="grammar", required=True)
        group.add_argument("--scorer", dest="scorer", choices=cls.scorers.keys(), default="feature")
        group.add_argument("--batch-size", dest="batch_size", type=int, default=10)
        group.add_argument("--beam-size", dest="beam_size", type=int, default=16)
        group.add_argument("--category-embedding", type=int, default=100)
        group.add_argument("--use-graph-embedding", action="store_true", default=False)
        group.add_argument("--no-backprop", action="store_true", default=False)
        group.add_argument("--greedy-at-leaf", action="store_true", default=False)
        group.add_argument("--model-format", dest="model_format", choices=nn.model_formats, default="pickle")

        EdgeEvaluation.add_parser_arguments(arg_parser)
        POSTagClassification.add_parser_arguments(arg_parser)

    @classmethod
    def add_predict_arguments(cls, arg_parser):
        super(UdefQParser, cls).add_predict_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--model-format", dest="model_format", choices=nn.model_formats, default=None)

    @classmethod
    def add_common_arguments(cls, arg_parser):
        super(UdefQParser, cls).add_common_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__ + "(common)")
        group.add_argument("--deepbank-dir", default="./")
        group.add_argument("--graph-type", choices=["eds", "dmrs"], default="eds")
        group.add_argument("--unlexicalized-rules")

    @classmethod
    def get_next_arg_parser(cls, stage, options):
        if options.mode != "train":
            return None
        if stage == 1:
            arg_parser = ArgumentParser(sys.argv[0])
            scorer_class = cls.scorers[options.scorer]
            scorer_class.add_parser_arguments(arg_parser)
            return arg_parser
        else:
            return None

    def __init__(self, options, data_train):
        self.model = dn.Model()
        (span_model_options, statistics), self.container = nn.model_load_helper(
            options.span_model_format, options.span_model_prefix, self.model)

        logger.info(statistics)
        self.statistics = statistics
        self.options = options
        self.options.__dict__ = AttrDict(chain(span_model_options.__dict__.items(),
                                               options.__dict__.items()))
        logger.info(pformat(self.options.__dict__))

        self.optimizer = nn.get_optimizer(self.model, options)

        with open(options.derivations, "rb") as f:
            self.derivations = pickle.load(f, encoding="latin1")
        self.hrg_statistics = HRGStatistics.from_derivations(self.derivations)
        print(self.hrg_statistics)

        self.span_ebd_network, self.span_eval_network, self.label_eval_network = self.container.components
        self.scorer_network = self.scorers[options.scorer](self.container, self.hrg_statistics, self.options)
        self.edge_eval = EdgeEvaluation(self.container, self.hrg_statistics.structural_edges, options)
        self.category_embedding = nn.EmbeddingFromDictionary(self.container, self.hrg_statistics.categories,
                                                             options.category_embedding)
        self.category_scorer = POSTagClassification(self.container, self.hrg_statistics.categories, options,
                                                    options.lstm_dims * options.span_lstm_layers
                                                    )

        with open(options.grammar, "rb") as f:
            self.grammar = pickle.load(f, encoding="latin1")  # type: Mapping[str, Mapping[CFGRule, int]]
        self.terminal_mapping = defaultdict(Counter)  # type: Mapping[str, typing.Counter]
        for (cfg_lhs, cfg_rhs_list), value in self.grammar.items():
            if all(isinstance(i, HLexicon) for i in cfg_rhs_list):
                self.terminal_mapping[cfg_lhs] += value

        self.lexicon_mapping = defaultdict(Counter)  # type: Mapping[Tuple[HLexicon, int], typing.Counter]
        for (cfg_lhs, cfg_rhs_list), value in self.grammar.items():
            rule_name, main_node_count = cfg_lhs.rsplit("#", 1)
            main_node_count = int(main_node_count)
            if all(isinstance(i, HLexicon) for i in cfg_rhs_list):
                lexicon = cfg_rhs_list[0]
                self.lexicon_mapping[lexicon, main_node_count] += value

        self.lemmatizer = WordNetLemmatizer()

        if options.unlexicalized_rules is not None:
            with open(options.unlexicalized_rules, "rb") as f:
                self.unlexicalized_rules = pickle.load(f)

    def __getstate__(self):
        odict = dict(self.__dict__)
        for i in ("model", "optimizer", "container", "span_ebd_network",
                  "scorer_network", "lemmatizer"):
            del odict[i]
        return odict

    @classmethod
    def load(cls, prefix, new_options=None):
        model = dn.Model()
        model_format = new_options.model_format if new_options is not None else None
        ret, savable = nn.model_load_helper(model_format, prefix, model)
        ret.model = model
        ret.optimizer = nn.trainers[ret.options.optimizer](
            *((ret.model, ret.options.learning_rate)
              if ret.options.learning_rate is not None else (ret.model,)))
        ret.container = savable
        ret.span_ebd_network, ret.span_eval_network, ret.label_eval_network, \
        ret.scorer_network, ret.edge_eval, ret.category_embedding, ret.category_scorer = ret.container.components
        ret.lemmatizer = WordNetLemmatizer()
        if getattr(new_options, "unlexicalized_rules", None) is not None:
            with open(new_options.unlexicalized_rules, "rb") as f:
                ret.unlexicalized_rules = pickle.load(f)
        return ret

    def save(self, prefix):
        nn.model_save_helper(self.options.model_format, prefix, self.container, self)

    def training_session(self, tree, print_logger, derivations=()):
        is_train = bool(derivations)
        sentence_interface = tree.to_sentence()
        self.populate_delphin_spans(tree, is_train)
        self.span_ebd_network.init_special()
        span_features = self.span_ebd_network.get_span_features(sentence_interface)

        # each cfg tree node is assigned a beam and a list of sync rule correspondents
        traversal_sequence = []
        cfg_node_to_item = {}
        exprs = []

        tree_nodes = list(tree.generate_rules())

        delphin_span_to_word_span = {tree_node.extra["DelphinSpan"]: tree_node.span
                                     for tree_node in tree.root_first()}

        span_exprs_cache = {}
        feature_exprs_cache = {}
        total_loss = dn.scalarInput(0.0)

        def get_edges(graph):
            real_edges, real_nodes = self.get_real_edges(graph)
            edge_tuples = [(delphin_span_to_word_span[source.span], strip_category(source.label),
                            delphin_span_to_word_span[target.span], strip_category(target.label), label)
                           for source, label, target in real_edges]
            nodes_tuples = [(delphin_span_to_word_span[node.span], strip_category(node.label))
                            for node in real_nodes]
            features = frozenset(edge_tuples + nodes_tuples)
            return features

        def get_span_expr(start, end):
            ret = span_exprs_cache.get((start, end))
            if not ret:
                ret = span_features[start, end]
                if getattr(self.options, "no_backprop", False):
                    ret = dn.nobackprop(ret)
                span_exprs_cache[start, end] = ret
            return ret

        def edge_feature_to_scores(edge_features, return_expr=False):
            feature_scores = 0.0 if not return_expr else dn.scalarInput(0.0)
            for e in edge_features:
                if len(e) == 5:  # edge feature
                    start_span, start_category, end_span, end_category, label = e
                    exprs = feature_exprs_cache.get(e)
                    if exprs is None:
                        exprs = feature_exprs_cache[e] = self.edge_eval(
                            dn.concatenate([get_span_expr(*start_span),
                                            self.category_embedding(start_category)]),
                            dn.concatenate([get_span_expr(*end_span),
                                            self.category_embedding(end_category)]),
                        )[self.hrg_statistics.structural_edges.word_to_int[label]]
                    feature_scores += (exprs if return_expr else exprs.value()) - (
                        1.0 if is_train and e in tree.extra["DelphinArgsSet"] else 0.0)
                else:
                    span, name = e
                    exprs = feature_exprs_cache.get(e)
                    if exprs is None:
                        exprs = feature_exprs_cache[e] = self.category_scorer.mlp(
                            get_span_expr(*span)
                        )[self.hrg_statistics.categories.word_to_int[name]]
                    feature_scores += (exprs if return_expr else exprs.value()) - (
                        1.0 if is_train and e in tree.extra["DelphinNamesSet"] else 0.0)
            return feature_scores

        def rule_loss_collector(
                beam_item  # type: BeamItem
        ):
            if beam_item.sync_rule is None:
                yield dn.scalarInput(0.0)
                raise StopIteration
            node_info = beam_item.node_info_ref()
            if node_info.early_updated:
                yield dn.scalarInput(0.0)
                raise StopIteration

            if self.options.use_graph_embedding:
                # calculate rule loss
                correspondents = node_info.correspondents
                pred_expr = correspondents[beam_item.sync_rule]
                gold_expr = correspondents[node_info.gold_rule]
                loss = (pred_expr - gold_expr) if pred_expr is not gold_expr else dn.scalarInput(0.0)
            else:
                loss = dn.scalarInput(0.0)

            # add rule correctness statistics
            print_logger.total_count += 1
            if beam_item.sync_rule == node_info.gold_rule:
                print_logger.correct_count += 1

            # calculate edge loss
            gold_item = node_info.gold_item
            if gold_item is not beam_item:
                predict_edge_scores = edge_feature_to_scores(
                    beam_item.own_features - gold_item.own_features,
                    return_expr=True)  # ignore common features
                gold_edge_scores = edge_feature_to_scores(
                    gold_item.own_features - beam_item.own_features, True)
                print_logger.total_gold_score += gold_item.score
                print_logger.total_predict_score += beam_item.score

                struct_loss = predict_edge_scores - gold_edge_scores
                loss += struct_loss
            yield loss

            # loss of children node
            if beam_item.left is not None:
                for i in rule_loss_collector(beam_item.left):
                    yield i
            if beam_item.right is not None:
                for i in rule_loss_collector(beam_item.right):
                    yield i
            node_info.early_updated = 1

        def get_loss(final_beam_item, final_gold_item):
            rule_losses = list(rule_loss_collector(final_beam_item))
            loss = sum(rule_losses)
            # if loss.value() < 0:
            #     loss = dn.scalarInput(0.0)
            return loss

        # generate expressions
        if derivations:
            assert len(derivations) == len(tree_nodes)
        for gold_rule, tree_node in zip_longest(derivations, tree_nodes):
            if tree_node.tag.endswith("#0"):
                node_info = TraversalSequenceItem(tree_node)
                cfg_node_to_item[tree_node] = node_info
                traversal_sequence.append(node_info)
                continue
            try:
                rules_dict = self.rule_lookup(tree_node, is_train)
                correspondents = set(rules_dict.items())
            except ValueError as e:
                traceback.print_exc()
                yield [dn.scalarInput(0.0)]
                yield dn.scalarInput(0.0) if derivations is not None else None
                raise StopIteration
            rule_getter = self.scorer_network.get_rules(
                get_span_expr(*tree_node.span),
                correspondents,
                gold_rule, self.options.use_graph_embedding)
            node_info = TraversalSequenceItem(tree_node, gold_rule, rule_getter)
            traversal_sequence.append(node_info)
            cfg_node_to_item[tree_node] = node_info
            if isinstance(tree_node.children[0], ConstTree):
                node_info.left = cfg_node_to_item[tree_node.children[0]]
            if len(tree_node.children) == 2 and isinstance(tree_node.children[1], ConstTree):
                node_info.right = cfg_node_to_item[tree_node.children[1]]
            exprs.extend(next(rule_getter))
        yield exprs

        # fill rules and scores
        for node_info in traversal_sequence:
            if node_info.rule_getter is not None:
                node_info.correspondents = next(node_info.rule_getter)

        # do tree beam search
        for node_info in traversal_sequence:
            if node_info.correspondents is None:
                # deal with semantic null
                node_info.beam = Beam(self.options.beam_size)
                node_info.beam.push(empty_beam_item)
                node_info.gold_item = empty_beam_item if derivations else None
                continue
            else:
                sync_rules = sorted(node_info.correspondents.items(),
                                    key=lambda x: x[1].value(),
                                    reverse=True
                                    )
            if node_info.is_preterminal:
                # deal with pre-terminal nodes
                # create gold item
                if is_train:
                    sub_graph = SubGraph.create_leaf_graph(node_info.cfg_node, node_info.gold_rule)
                    edge_features = get_edges(sub_graph.graph)
                    total_score = node_info.correspondents[node_info.gold_rule].value()
                    total_score += edge_feature_to_scores(edge_features)
                    node_info.gold_item = BeamItem(
                        weakref.ref(node_info), node_info.gold_rule,
                        sub_graph, total_score, None, None, True, edge_features, edge_features)
                # do beam search
                node_info.beam = Beam(self.options.beam_size)
                # top_rules = sync_rules[:self.options.beam_size]  # type: List[Tuple[CFGRule, dn.Expression]]
                top_rules = sync_rules
                for sync_rule, score_expr in top_rules:
                    score = score_expr.value()
                    is_gold = sync_rule == node_info.gold_rule
                    if is_gold:
                        beam_item = node_info.gold_item
                    else:
                        sub_graph = SubGraph.create_leaf_graph(node_info.cfg_node, sync_rule)
                        edge_features = get_edges(sub_graph.graph)
                        total_score = score
                        total_score += edge_feature_to_scores(edge_features)
                        beam_item = BeamItem(weakref.ref(node_info), sync_rule,
                                             sub_graph, total_score, None, None,
                                             False, edge_features, edge_features
                                             )
                    node_info.beam.push(beam_item)
                if is_train:
                    has_gold = any(item for item in node_info.beam if item.is_gold)
                    if self.options.greedy_at_leaf or not has_gold:
                        # early update
                        best_item = node_info.beam.best_item()
                        if best_item is node_info.gold_item:
                            continue
                        total_loss += get_loss(best_item, node_info.gold_item)
                        node_info.beam.clear()
                        node_info.beam.push(node_info.gold_item)

            else:  # aka: node is not preterminal
                # create gold item
                if is_train:
                    left_item = node_info.left.gold_item
                    right_item = node_info.right.gold_item
                    gold_graph = SubGraph.merge(
                        node_info.cfg_node, node_info.gold_rule,
                        left_item.sub_graph, right_item.sub_graph)
                    edge_features = get_edges(gold_graph.graph)
                    own_features = edge_features - left_item.total_features - right_item.total_features
                    feature_scores = edge_feature_to_scores(own_features)
                    total_score = left_item.score + right_item.score + \
                                  node_info.correspondents[node_info.gold_rule].value() + feature_scores
                    node_info.gold_item = BeamItem(
                        weakref.ref(node_info), node_info.gold_rule,
                        gold_graph, total_score, left_item, right_item, True, own_features, edge_features)

                node_info.beam = Beam(self.options.beam_size)
                # deal with non-leaf nodes
                for sync_rule, score_expr in sync_rules:
                    score = score_expr.value()
                    for left_item, right_item in product(node_info.left.beam,
                                                         node_info.right.beam):
                        is_gold = left_item.is_gold and right_item.is_gold and sync_rule == node_info.gold_rule
                        if is_gold:
                            beam_item = node_info.gold_item
                        else:
                            total_score = left_item.score + right_item.score + score
                            sub_graph = SubGraph.merge(
                                node_info.cfg_node, sync_rule,
                                left_item.sub_graph, right_item.sub_graph)

                            edge_tuples = [(edge.nodes[0], edge.label)
                                           for edge in sub_graph.graph.edges
                                           if len(edge.nodes) == 2 and edge.span is None]
                            edge_features = get_edges(sub_graph.graph)
                            own_features = edge_features - left_item.total_features - right_item.total_features
                            feature_scores = edge_feature_to_scores(own_features)
                            total_score += feature_scores

                            if not is_train:
                                # check consistency
                                if len(edge_tuples) != len(set(edge_tuples)):
                                    total_score -= 10

                            beam_item = BeamItem(weakref.ref(node_info), sync_rule,
                                                 sub_graph, total_score, left_item, right_item,
                                                 False, own_features, edge_features)
                        node_info.beam.push(beam_item)
                if is_train:
                    has_gold = any(item for item in node_info.beam if item.is_gold)
                    if not has_gold:
                        # early update
                        best_item = node_info.beam.best_item()
                        total_loss += get_loss(best_item, node_info.gold_item)
                        node_info.beam.clear()
                        node_info.beam.push(node_info.gold_item)

        final_beam_item = traversal_sequence[-1].beam.best_item()
        if not is_train:
            yield final_beam_item
            raise StopIteration

        final_gold_item = traversal_sequence[-1].gold_item

        total_loss += get_loss(final_beam_item, final_gold_item)
        yield total_loss

    def train(self, trees):
        print_logger = PrintLogger()
        print_per = (100 // self.options.batch_size + 1) * self.options.batch_size
        for sent_idx, batch_idx, batch_trees in split_to_batches(
                trees, self.options.batch_size):
            if sent_idx % print_per == 0 and sent_idx != 0:
                print_logger.print(sent_idx)
            sessions = [self.training_session(tree, print_logger,
                                              self.derivations[tree.extra["ID"]])
                        for tree in batch_trees]
            exprs = [expr for session in sessions for expr in next(session)]
            dn.forward(exprs)
            loss = sum(next(session) for session in sessions) / len(sessions)
            print_logger.total_loss += loss.value()
            loss.backward()
            self.optimizer.update()
            dn.renew_cg()

    @staticmethod
    def get_real_edges(hg):
        node_mapping = {}  # node -> pred edge
        real_edges = []
        ret_edges = []
        for edge in hg.edges:  # type: HyperEdge
            if len(edge.nodes) == 1:
                main_node = edge.nodes[0]  # type: GraphNode
                if node_mapping.get(main_node) is not None:
                    continue
                if not edge.is_terminal:
                    raise Exception("Non-terminal edge should not exist there {}".format(edge))
                node_mapping[main_node] = edge
            elif len(edge.nodes) == 2:
                real_edges.append(edge)
            else:
                raise Exception("Hyperedge should not exist there")

        for edge in real_edges:
            pred_edges = [node_mapping.get(i) for i in edge.nodes]
            if pred_edges[0] is not None and pred_edges[1] is not None:
                ret_edges.append((pred_edges[0], edge.label, pred_edges[1]))

        return ret_edges, list(node_mapping.values())

    def construct_derivation(self, beam_item):
        if beam_item.sub_graph is not None:
            yield beam_item.sub_graph.graph, beam_item.sync_rule
        else:
            yield None, None
        if beam_item.left is not None:
            yield from self.construct_derivation(beam_item.left)
        if beam_item.right is not None:
            yield from self.construct_derivation(beam_item.right)

    def predict(self, trees, return_derivation=False):
        print_logger = PrintLogger()
        for sent_idx, batch_idx, batch_trees in split_to_batches(
                trees, self.options.batch_size):
            sessions = [self.training_session(tree, print_logger)
                        for tree in batch_trees]
            exprs = [expr for session in sessions for expr in next(session)]
            dn.forward(exprs)
            for tree, session in zip(batch_trees, sessions):
                final_beam_item = next(session)
                graph = final_beam_item.sub_graph.graph
                if return_derivation:
                    yield tree.extra["ID"], graph, list(self.construct_derivation(final_beam_item))
                else:
                    yield tree.extra["ID"], graph
            dn.renew_cg()

    def rule_lookup(self, tree_node, is_train):
        keyword = (tree_node.tag,
                   tuple(i.tag if isinstance(i, ConstTree) else i
                         for i in tree_node.children))
        result = self.grammar.get(keyword)
        if result is None and all(isinstance(i, Lexicon) for i in tree_node.children):
            return self.sync_grammar_fallback(tree_node)
        elif is_train and isinstance(tree_node.children[0], Lexicon):
            result = result | self.sync_grammar_fallback(tree_node, False)
        if result is None:
            raise ValueError(str(tree_node))
        return result

    @staticmethod
    def populate_delphin_spans(tree, is_train):
        preterminals = list(tree.generate_preterminals())
        spans = literal_eval(tree.extra["DelphinSpans"])
        assert len(preterminals) == len(spans)

        for span, preterminal in zip(spans, preterminals):
            preterminal.extra["DelphinSpan"] = span

        for rule in tree.generate_rules():
            if isinstance(rule.children[0], ConstTree):
                rule.extra["DelphinSpan"] = (rule.children[0].extra["DelphinSpan"][0],
                                             rule.children[-1].extra["DelphinSpan"][1])

        # args and names
        if is_train:
            args_tuples = literal_eval(tree.extra["Args"])
            tree.extra["DelphinArgsSet"] = frozenset(args_tuples)
            names_tuples = literal_eval(tree.extra["Names"])
            tree.extra["DelphinNamesSet"] = frozenset(names_tuples)

    pattern_number = re.compile(r"^[0-9.,]+$")

    def transform_edge(self, edge, lexicon):
        if "NEWLEMMA" in edge.label:
            word = lexicon.string.replace("_", "+")
            if "_u_unknown" in edge.label:
                item = word
            else:
                pos = edge.label[edge.label.find("NEWLEMMA") + 10]
                if pos in ("n", "v", "a"):
                    item = self.lemmatizer.lemmatize(word, pos)
                else:
                    item = self.lemmatizer.lemmatize(lexicon.string.replace("_", "+"))
            new_label = edge.label.format(NEWLEMMA=item)
            # print(edge.label, lexicon, item, new_label)
            return HyperEdge(edge.nodes, new_label,
                             edge.is_terminal, edge.span)
        return edge

    def recover_rule(self, rule, lexicon, tag):
        return CFGRule(lhs=tag,
                       rhs=((lexicon, None),),
                       hrg=HRGRule(
                           lhs=rule.lhs,
                           rhs=HyperGraph(
                               nodes=rule.rhs.nodes,
                               edges=frozenset(self.transform_edge(edge, lexicon)
                                               for edge in rule.rhs.edges)
                           )
                       ))

    def sync_grammar_fallback(self, tree_node, need_recover=False):
        tag = tree_node.tag
        lexicon = tree_node.children[0]
        if hasattr(self, "unlexicalized_rules"):
            results = Counter({
                self.recover_rule(rule, lexicon, tag): count
                for rule, count in self.unlexicalized_rules[tag].items()})
            if results:
                return results
        return self.sync_grammar_fallback_2(tree_node)

    def sync_grammar_fallback_2(self, tree_node):
        rule_name, main_node_count = tree_node.tag.rsplit("#", 1)
        word = tree_node.children[0].string
        main_node_count = int(main_node_count)
        if main_node_count == 1:
            main_node = GraphNode("0")
            surface = tree_node.children[0].string

            if self.pattern_number.match(surface):
                label = "card"
            elif rule_name.find("generic_proper") >= 0:
                label = "named"
            else:
                lemma = self.lemmatizer.lemmatize(word)
                if rule_name.find("n_-_c-pl-unk_le") >= 0:
                    label = "_{}/nns_u_unknown".format(lemma)
                elif rule_name.find("n_-_mc_le") >= 0 or rule_name.find("n_-_c_le") >= 0:
                    label = "_{}_n_1".format(lemma)  # more number is used
                elif rule_name.find("generic_mass_count_noun") >= 0:
                    label = "_{}/nn_u_unknown".format(lemma)  # more number is used
                else:
                    candidates = self.lexicon_mapping[HLexicon(word), main_node_count]
                    if candidates:
                        return candidates
                    else:
                        label = "named"

            old_edge = HyperEdge(
                nodes=[main_node],
                label=rule_name,
                is_terminal=False
            )

            main_edge = HyperEdge(
                nodes=[main_node],
                label=label,
                is_terminal=True
            )

            fallback = CFGRule(lhs=rule_name,
                               rhs=((tree_node.children[0], None),),
                               hrg=HRGRule(
                                   lhs=old_edge,
                                   rhs=HyperGraph(
                                       nodes=frozenset([main_node]),
                                       edges=frozenset({main_edge})
                                   )
                               ))
        else:
            ret1 = self.terminal_mapping.get(tree_node.tag)
            if ret1:
                return Counter([ret1.most_common(1)[0][0]])
            connected_nodes = [GraphNode(str(i)) for i in range(main_node_count)]
            centural_node = GraphNode(str(main_node_count + 1))
            old_edge = HyperEdge(
                nodes=connected_nodes,
                label=rule_name,
                is_terminal=False
            )
            main_edges = [HyperEdge(
                nodes=[centural_node, i],
                label="???",
                is_terminal=True
            ) for i in connected_nodes]
            fallback = CFGRule(lhs=rule_name,
                               rhs=((tree_node.children[0], None),),
                               hrg=HRGRule(
                                   lhs=old_edge,
                                   rhs=HyperGraph(
                                       nodes=frozenset(connected_nodes + [centural_node]),
                                       edges=frozenset(main_edges)
                                   )
                               ))
        return Counter([fallback])

    @classmethod
    def predict_and_output(cls, parser, options, sentences, output_file):
        total_result = ScorerResult.zero()
        with open(output_file + ".txt", "w") as f, \
                open(output_file + ".graph", "w") as f_graph, \
                open(output_file + ".gold_graph", "w") as f_gold_graph:
            for sent_id, graph in parser.predict(sentences):
                with gzip.open(options.deepbank_dir + "/" + sent_id + ".gz",
                               "rb") as f_gz:
                    contents = f_gz.read().decode("utf-8")
                fields = contents.strip().split("\n\n")
                if options.graph_type == "eds":
                    eds_literal = fields[-2]
                    eds_literal = re.sub(r"\{.*\}", "", eds_literal)
                    e = eds.loads_one(eds_literal)
                    gold_graph = EDSScorer.from_eds(e, sent_id)
                    f_gold_graph.write(eds_for_smatch(sent_id, e))
                else:
                    assert options.graph_type == "dmrs"
                    m = simplemrs.loads_one(fields[-3])
                    gold_graph = EDSScorer.from_mrs(m)
                    f_gold_graph.write(mrs_for_smatch(sent_id, m))
                result = EDSScorer.from_hypergraph(graph).compare_with(
                    gold_graph,
                    True, log_func=lambda x: print(x, file=f))
                f.write(str(result))
                f_graph.write(output_hg(sent_id, graph))
                total_result += result
            print("Total:")
            print(total_result)
            f.write(str(total_result))
        current_path = os.path.dirname(__file__)
        os.system('{}/../utils/smatch_1 {} {} > {}.smatch'.format(
            current_path, output_file + ".graph", output_file + ".gold_graph", output_file))
        os.system('cat {}.smatch'.format(output_file))

    @classmethod
    def train_parser(cls, options, data_train=None, data_dev=None, data_test=None):
        set_proc_name(options.title)
        ensure_dir(options.output)
        path = os.path.join(options.output, "{}_{}_train.log".format(options.title,
                                                                     int(time.time())))
        log_to_file(path)
        logger.name = options.title

        logger.info('Options:\n%s', pformat(options.__dict__))
        if data_train is None:
            data_train = cls.DataType.from_file(options.conll_train)

        if data_dev is None:
            data_dev = {i: cls.DataType.from_file(i, False) for i in options.conll_dev}

        try:
            os.makedirs(options.output)
        except OSError:
            pass

        parser = cls(options, data_train)
        random_obj = random.Random(1)

        def do_predict(epoch):
            for file_name, dev_sentences in data_dev.items():
                try:
                    prefix, suffix = os.path.basename(file_name).rsplit(".", 1)
                except ValueError:
                    prefix = file_name
                    suffix = ""

                dev_output = os.path.join(options.output, '{}_epoch_{}.{}'.format(prefix, epoch, suffix))
                cls.predict_and_output(parser, options, dev_sentences, dev_output)

        if options.epochs == 0:
            print("Predict directly.")
            do_predict(0)

        for epoch in range(options.epochs):
            logger.info('Starting epoch %d', epoch)
            random_obj.shuffle(data_train)
            parser.train(data_train)

            # save model and delete old model
            for i in range(0, epoch - options.max_save):
                path = os.path.join(options.output, os.path.basename(options.model)) + str(i + 1)
                if os.path.exists(path):
                    os.remove(path)
            path = os.path.join(options.output, os.path.basename(options.model)) + str(epoch + 1)
            parser.save(path)
            do_predict(epoch)

    @classmethod
    def predict_with_parser(cls, options):
        if options.input_format == "standard":
            data_test = cls.DataType.from_file(options.conll_test, False)
        elif options.input_format == "space":
            with smart_open(options.conll_test) as f:
                data_test = [cls.DataType.from_words_and_postags([(word, "X") for word in line.strip().split(" ")])
                             for line in f]
        elif options.input_format == "english":
            from nltk import download, sent_tokenize
            from nltk.tokenize import TreebankWordTokenizer
            download("punkt")
            with smart_open(options.conll_test) as f:
                raw_sents = sent_tokenize(f.read().strip())
                tokenized_sents = TreebankWordTokenizer().tokenize_sents(raw_sents)
                data_test = [cls.DataType.from_words_and_postags([(token, "X") for token in sent])
                             for sent in tokenized_sents]
        elif options.input_format == "tokenlist":
            with smart_open(options.conll_test) as f:
                items = eval(f.read())
            data_test = cls.DataType.from_words_and_postags(items)
        else:
            raise ValueError("invalid format option")

        logger.info('Initializing...')
        parser = cls.load(options.model, options)

        ts = time.time()
        cls.predict_and_output(parser, options, data_test, options.out_file)
        te = time.time()
        logger.info('Finished predicting and writing test. %.2f seconds.', te - ts)

        # if options.evaluate:
        #     cls.DataType.evaluate_with_external_program(options.conll_test,
        #                                                 options.out_file)
