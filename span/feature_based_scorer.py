from typing import Optional, Iterable, Tuple

import nn
import numpy as np
import dynet as dn

from common_utils import ensure_dir
from hrgguru.hrg import CFGRule
from hrgguru.hyper_graph import strip_category
from logger import logger
from span.hrg_statistics import HRGStatistics, encode_nonterminal


class StructuredPeceptronHRGScorer(nn.DynetSaveable):
    def extract_features(self,
                         rule,  # type: Optional[CFGRule]
                         count
                         ):
        if not self.options.use_count:
            count = 1
        result = [0 for _ in range(len(self.possible_features) + 1)]
        result[-1] = count
        if rule.hrg is not None:
            if len(rule.rhs) == 2:
                left_info, right_info = rule.rhs
                if len(left_info) == 2:
                    left_label, left_edge = left_info
                    if left_edge is not None:
                        if left_edge.nodes == rule.hrg.lhs.nodes:
                            result[self.feature_index["head_left"]] = 1
                        # elif len(left_edge.nodes) == 2:
                        #     if left_edge.nodes[0] == rule.hrg.lhs.nodes[0]:
                        #         result[self.feature_index["head_left_1/2"]] = 1
                        #     elif len(rule.hrg.lhs.nodes) >= 2 and left_edge.nodes[1] == rule.hrg.lhs.nodes[1]:
                        #         result[self.feature_index["head_left_2/2"]] = 1
                if len(right_info) == 2:
                    right_label, right_edge = right_info
                    if right_edge is not None:
                        if right_edge.nodes == rule.hrg.lhs.nodes:
                            result[self.feature_index["head_right"]] = 1
                        # elif len(rule.hrg.lhs.nodes) >= 2 and len(right_edge.nodes) == 2:
                        #     if right_edge.nodes[0] == rule.hrg.lhs.nodes[0]:
                        #         result[self.feature_index["head_right_1/2"]] = 1
                        #     elif right_edge.nodes[1] == rule.hrg.lhs.nodes[1]:
                        #         result[self.feature_index["head_right_2/2"]] = 1
            for edge in rule.hrg.rhs.edges:
                if edge.is_terminal and len(edge.nodes) == 2:
                    label = edge.label
                elif edge.is_terminal and len(edge.nodes) == 1:
                    label = strip_category(edge.label)
                elif not edge.is_terminal:
                    label = encode_nonterminal(edge)
                else:
                    label = "INVALID"
                if label in self.edge_labels:
                    feature = ("Edge", label)
                    result[self.feature_index[feature]] += 1
        return tuple(result), count

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--hrg-mlp-dims", dest="hrg_mlp_dims",
                           type=int, nargs="*",
                           help="MLP Layers", default=[100])
        group.add_argument("--hrg-loss-margin", dest="hrg_loss_margin",
                           type=int, default=0.1)
        group.add_argument("--conflict-output-dir", dest="conflict_output_dir",
                           default=None)
        group.add_argument("--use-count", action="store_true", dest="use_count", default=False)

    def __init__(self, model,
                 hrg_statistics,  # type: HRGStatistics
                 options):
        super(StructuredPeceptronHRGScorer, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]

        self.edge_labels = list(
            word for word, count in hrg_statistics.nonterminals.most_common(300)) + \
                           list(hrg_statistics.structural_edges) + \
                           list(hrg_statistics.categories)

        self.possible_features = [("Edge", k) for k in self.edge_labels]
        logger.info("Consider {} features as graph embedding".format(
            len(self.possible_features)))
        self.possible_features.append("head_left")
        self.possible_features.append("head_right")
        # self.possible_features.append("head_left_1/2")
        # self.possible_features.append("head_left_2/2")
        # self.possible_features.append("head_right_1/2")
        # self.possible_features.append("head_right_2/2")
        self.feature_index = {i: idx for idx, i in enumerate(self.possible_features)}

        dense_dims = [options.lstm_dims * 2 * options.span_lstm_layers + len(self.possible_features) + 1] + \
                     options.hrg_mlp_dims + [1]
        # don't use bias in last transform
        use_bias = [True] * (len(dense_dims) - 2) + [False]

        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation, use_bias)
        self.count_scale = self.add_parameters((1,))
        self.count_scale_2 = self.add_parameters((1,))

        if self.options.conflict_output_dir:
            ensure_dir(self.options.conflict_output_dir)

    def get_rules(self,
                  span_feature,
                  rules_and_counts,  # type: Iterable[Tuple[CFGRule, int]]
                  gold=None,
                  need_score=True
                  ):
        feature_map = {rule: self.extract_features(rule, count)
                       for rule, count in rules_and_counts}  # rule -> feature
        features = frozenset(feature_map.values())
        if len(features) == 1:
            yield []
            yield {rule: dn.scalarInput(0.0) for rule, count in rules_and_counts}
            raise StopIteration

        def calculate_score(feature, count):
            feature_score = self.dense_layer(
                dn.concatenate([span_feature, dn.inputTensor(np.array(feature))]))
            return feature_score

        features_to_exprs = {(feature, count): calculate_score(feature, count)
                             for feature, count in features}  # feature -> score

        if gold is not None:
            gold_feature = feature_map[gold]
        else:
            gold_feature = None

        # waiting for other batch
        yield features_to_exprs.values()

        def get_score(rule):
            feature = feature_map[rule]
            score = features_to_exprs[feature]
            if gold is not None and feature == gold_feature:
                score -= self.options.hrg_loss_margin
            return score

        rules_and_scores = {rule: get_score(rule)
                            for rule, count in rules_and_counts
                            }
        yield rules_and_scores

    def get_best_rule(self,
                      span_feature,
                      rules_and_counts,  # type: Iterable[Tuple[CFGRule, int]]
                      gold=None
                      ):
        feature_map = {rule: self.extract_features(rule, count)
                       for rule, count in rules_and_counts}  # rule -> feature
        features = frozenset(feature_map.values())
        if len(features) == 1:
            best_rule = sorted(((count, rule) for rule, count in rules_and_counts),
                               key=lambda x: x[0],
                               reverse=True)[0][1]
            yield []
            yield best_rule, None, best_rule
            raise StopIteration

        def calculate_score(feature, count):
            feature_score = self.dense_layer(
                dn.concatenate([span_feature, dn.inputTensor(np.array(feature))]))
            if self.options.use_count:
                count_score = self.count_scale_2.expr() * dn.log(dn.abs(self.count_scale.expr() * count + 1))
            else:
                count_score = 0
            return feature_score + count_score

        features_to_exprs = {(feature, count): calculate_score(feature, count)
                             for feature, count in features}  # feature -> score

        if gold is not None:
            gold_feature = feature_map[gold]
        else:
            gold_feature = None

        def get_score_count_rule(rule, count):
            feature = feature_map[rule]
            score = features_to_exprs[feature].value()
            if gold is not None and feature == gold_feature:
                score -= self.options.hrg_loss_margin
            return score, count, rule

        # waiting for other batch
        yield features_to_exprs.values()

        score_count_rule = [get_score_count_rule(rule, count)
                            for rule, count in rules_and_counts]  # tuple (score, count, rule)
        # chooce the highest score, if not unique, choose the most frequent
        best_score, best_count, best_rule = sorted(score_count_rule,
                                                   key=lambda x: (x[0], x[1]),
                                                   reverse=True)[0]
        best_feature = feature_map[best_rule]
        best_expr = features_to_exprs[best_feature]

        if gold is not None:
            _, _, real_best_rule = \
                sorted(
                    ((score if feature_map[rule] != gold_feature else score + self.options.hrg_loss_margin, count, rule)
                     for score, count, rule in score_count_rule),
                    key=lambda x: (x[0], x[1]),
                    reverse=True)[0]
            assert gold in [i[0] for i in rules_and_counts]
            gold_expr = features_to_exprs[gold_feature]
            # output conflict
            if self.options.conflict_output_dir:
                if real_best_rule != gold and feature_map[real_best_rule] == gold_feature:
                    base_name = "{}_{}".format(hash(real_best_rule), hash(gold))
                    real_best_rule.hrg.draw(self.options.conflict_output_dir + "/" + base_name + "_real",
                                            draw_format="png")
                    gold.hrg.draw(self.options.conflict_output_dir + "/" + base_name + "_gold",
                                  draw_format="png")
                    # with open(self.options.conflict_output, "a") as f:
                    #     print("Conflict:\n real: {}\n gold: {}".format(
                    #         real_best_rule, gold), file=f)
            if best_feature == gold_feature:
                loss = dn.scalarInput(0.0)
            else:
                loss = best_expr - gold_expr + 1
            yield best_rule, loss, real_best_rule
            raise StopIteration
        yield best_rule, None, best_rule
        raise StopIteration

    def restore_components(self, components):
        self.dense_layer, self.count_scale, self.count_scale_2 = components
