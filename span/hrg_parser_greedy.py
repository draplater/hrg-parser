import gzip
import pickle
import random
import typing
from argparse import ArgumentParser
from ast import literal_eval
from itertools import chain
from pprint import pformat
from typing import List, Mapping, Tuple

import dynet as dn
import sys

import re

import os

import time

from nltk import WordNetLemmatizer

import nn
from common_utils import AttrDict, set_proc_name, ensure_dir, smart_open
from delphin.mrs import eds
from hrgguru.eds import ScorerResult, EDSScorer
from hrgguru.hrg import CFGRule, HRGRule
from hrgguru.const_tree import Lexicon as HLexicon
from hrgguru.hyper_graph import GraphNode, HyperEdge, HyperGraph
from logger import logger, log_to_file
from parser_base import DependencyParserBase
from span.const_tree import ConstTree, Lexicon
from collections import Counter, deque, defaultdict

from span.count_based_scorer import CountBasedHRGScorer
from span.embedding_based_scorer import EmbeddingHRGScorer
from span.feature_based_scorer import StructuredPeceptronHRGScorer
from span.hrg_statistics import HRGStatistics
from span.hrg_parser import UdefQParser as UdefQParserBeamSearch


class UdefQParser(DependencyParserBase):
    DataType = ConstTree
    scorers = {"feature": StructuredPeceptronHRGScorer, "embedding": EmbeddingHRGScorer, "count": CountBasedHRGScorer}

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
        group.add_argument("--model-format", dest="model_format", choices=nn.model_formats, default="pickle")

    @classmethod
    def add_predict_arguments(cls, arg_parser):
        super(UdefQParser, cls).add_predict_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--model-format", dest="model_format", choices=nn.model_formats, default=None)

    @classmethod
    def add_common_arguments(cls, arg_parser):
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

        self.optimizer = nn.trainers[options.optimizer](
            *((self.model, options.learning_rate)
            if options.learning_rate is not None else (self.model,)))

        with open(options.derivations, "rb") as f:
            self.derivations = pickle.load(f, encoding="latin1")
        self.hrg_statistics = HRGStatistics.from_derivations(self.derivations)

        self.span_ebd_network, self.span_eval_network, self.label_eval_network = self.container.components
        self.scorer_network = self.scorers[options.scorer](self.container, self.hrg_statistics, self.options)

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
        ret.scorer_network = ret.container.components
        ret.lemmatizer = WordNetLemmatizer()
        return ret

    def save(self, prefix):
        nn.model_save_helper(self.options.model_format, prefix, self.container, self)

    def train(self, trees):
        if isinstance(self.scorer_network, CountBasedHRGScorer):
            logger.info("No need to train a count-based scorer.")
            return

        total_count = sys.float_info.epsilon
        correct_count = 0
        pending = []
        total_loss = 0
        for tree_idx, tree in enumerate(trees):
            # if tree_idx == 5000:
            #     return
            if (tree_idx + 1) % 100 == 0:
                logger.info("Sent {}, Correctness: {:.2f}, loss: {:.2f}".format(
                    tree_idx + 1,
                    correct_count / total_count * 100,
                    total_loss))
                total_count = sys.float_info.epsilon
                correct_count = 0
                total_loss = 0
            sent_id = tree.extra["ID"]
            derivations = self.derivations[sent_id]  # type: List[CFGRule]

            sentence_interface = tree.to_sentence()
            self.span_ebd_network.init_special()
            span_features = self.span_ebd_network.get_span_features(sentence_interface)

            cfg_nodes = list(tree.generate_rules())  # type: List[ConstTree]
            assert len(derivations) == len(cfg_nodes)

            for gold_rule, tree_node in zip(derivations, cfg_nodes):
                if tree_node.tag.endswith("#0"):
                    continue
                try:
                    correspondents = set(self.rule_lookup(tree_node, True).items())
                except ValueError as e:
                    print(e)
                    continue

                # print(span_features[tree_node.span].npvalue().shape)
                pending.append((self.scorer_network.get_best_rule(
                    span_features[tree_node.span],
                    correspondents,
                    gold_rule), gold_rule))

            if tree_idx % self.options.batch_size == 0 or tree_idx == len(trees) - 1:
                # generate expressions
                exprs = []
                for item, gold_rule in pending:
                    exprs.extend(next(item))

                # do batch calculation
                if exprs:
                    dn.forward(exprs)

                # calculate loss
                loss = dn.scalarInput(0.0)
                for item, gold_rule in pending:
                    best_rule, this_loss, real_best_rule = next(item)
                    if this_loss is not None:
                        total_count += 1
                        loss += this_loss
                        if real_best_rule == gold_rule:
                            correct_count += 1
                loss.forward()
                total_loss += loss.scalar_value()
                loss.backward()
                self.optimizer.update()
                dn.renew_cg()
                pending = []

    def predict(self, trees, return_derivations=False):
        derivations = []
        for idx, tree in enumerate(trees):
            sentence_interface = tree.to_sentence()
            self.populate_delphin_spans(tree)
            self.span_ebd_network.init_special()
            span_features = self.span_ebd_network.get_span_features(sentence_interface)
            r = [i for i in tree.root_first()]

            syn_rules = []
            for i in tree.root_first():
                correspondents = set(self.rule_lookup(i, False).items())
                best_rule_getter = self.scorer_network.get_best_rule(
                    span_features[i.span],
                    correspondents,
                    None)
                exprs = next(best_rule_getter)
                best_rule, this_loss, real_best_rule = next(best_rule_getter)
                syn_rules.append(best_rule)

            rule_mapping = dict(zip(r, syn_rules))

            def transform_edge(mapping, edge, span):
                return HyperEdge((mapping[i] for i in edge.nodes),
                                 edge.label,
                                 edge.is_terminal,
                                 span)

            # deal wth root rule

            # create nodes in working graph
            nodes_mapping = {i: GraphNode() for i in syn_rules[0].hrg.rhs.nodes}

            # edge -> span
            span_mapping = {}
            for cfg_subnode, (name, edge) in zip(r[0], syn_rules[0].rhs):
                if edge is not None:
                    span_mapping[edge] = cfg_subnode.extra["DelphinSpan"]

            # create edges in working graph
            new_edges = frozenset(transform_edge(nodes_mapping, edge, span_mapping.get(edge))
                                  for edge in syn_rules[0].hrg.rhs.edges)

            for new_edge in new_edges:
                if len(new_edge.nodes) == 1 and new_edge.span is None:
                    new_edge.span = r[0].extra["DelphinSpan"]

            step = 0
            working_graph = HyperGraph(frozenset(nodes_mapping.values()),
                                       new_edges)
            derivations.append((working_graph, syn_rules[0]))

            queue = deque()

            if isinstance(tree.children[0], ConstTree):
                # add children nodes to queue
                for i, (_, j) in zip(tree.children, syn_rules[0].rhs):
                    if j is not None:
                        queue.append((i, rule_mapping[i], transform_edge(nodes_mapping, j, span_mapping.get(j))))

            while queue:
                # each step substitute one nonteminal edge into subgraph,
                # and append child substitution into queue
                target_cfg_rule, target_sync_rule, target_edge = queue.popleft()
                assert target_edge in working_graph.edges
                target_nodes_mapping = dict(zip(target_sync_rule.hrg.lhs.nodes, target_edge.nodes))
                for node in target_sync_rule.hrg.rhs.nodes:
                    if node not in target_nodes_mapping.keys():
                        target_nodes_mapping[node] = GraphNode()

                # edge -> span
                span_mapping = {}
                for cfg_subnode, (name, edge) in zip(target_cfg_rule, target_sync_rule.rhs):
                    if edge is not None:
                        span_mapping[edge] = cfg_subnode.extra["DelphinSpan"]

                new_nodes = working_graph.nodes | frozenset(target_nodes_mapping.values())
                new_edges_this_step = frozenset(
                    transform_edge(target_nodes_mapping, edge, span_mapping.get(edge))
                    for edge in target_sync_rule.hrg.rhs.edges)
                new_edges = (working_graph.edges - {target_edge}) | new_edges_this_step

                for new_edge in new_edges_this_step:
                    if len(new_edge.nodes) == 1 and new_edge.span is None:
                        new_edge.span = target_cfg_rule.extra["DelphinSpan"]

                step += 1
                working_graph = HyperGraph(new_nodes, new_edges)
                derivations.append((working_graph, target_sync_rule))

                for i, (_, j) in zip(target_cfg_rule.children, target_sync_rule.rhs):
                    if j is not None:
                        queue.append((i, rule_mapping[i],
                                      transform_edge(target_nodes_mapping, j, span_mapping.get(j))))
            if not return_derivations:
                yield tree.extra["ID"], working_graph
            else:
                yield tree.extra["ID"], working_graph, derivations
            dn.renew_cg()

    @staticmethod
    def populate_delphin_spans(tree):
        preterminals = list(tree.generate_preterminals())
        spans = literal_eval(tree.extra["DelphinSpans"])
        assert len(preterminals) == len(spans)

        for span, preterminal in zip(spans, preterminals):
            preterminal.extra["DelphinSpan"] = span

        for rule in tree.generate_rules():
            if isinstance(rule.children[0], ConstTree):
                rule.extra["DelphinSpan"] = (rule.children[0].extra["DelphinSpan"][0],
                                             rule.children[-1].extra["DelphinSpan"][1])

    pattern_number = re.compile(r"^[0-9.,]+$")

    train_parser = classmethod(UdefQParserBeamSearch.train_parser.__func__)
    predict_with_parser = classmethod(UdefQParserBeamSearch.predict_with_parser.__func__)
    rule_lookup = UdefQParserBeamSearch.rule_lookup
    sync_grammar_fallback = UdefQParserBeamSearch.sync_grammar_fallback
    sync_grammar_fallback_2 = UdefQParserBeamSearch.sync_grammar_fallback_2
    recover_rule = UdefQParserBeamSearch.recover_rule
    transform_edge = UdefQParserBeamSearch.transform_edge
    predict_and_output = UdefQParserBeamSearch.predict_and_output
