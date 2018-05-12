from typing import Optional, List, Sequence, Tuple

import dynet as dn

import nn
from hrgguru.hrg import HRGRule, CFGRule
from hrgguru.hyper_graph import HyperEdge
from span.hrg_statistics import HRGStatistics


class EmbeddingHRGScorer(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--edge-embedding-dim", dest="edge_embedding_dim",
                           type=int, default=300)
        group.add_argument("--attention-dim", dest="attention_dim",
                           type=int, default=300)
        group.add_argument("--edge-count", dest="edge_count",
                           type=int, default=297)
        group.add_argument("--dag-lstm-dim", dest="dag_lstm_dim",
                           type=int, default=400)
        group.add_argument("--hrg-loss-margin", dest="hrg_loss_margin",
                           type=int, default=0.1)
        group.add_argument("--use-attention", action="store_true", dest="attention", default=False)
        group.add_argument("--hrg-mlp-dims", dest="hrg_mlp_dims",
                           type=int, nargs="*",
                           help="MLP Layers", default=[100])

    def __init__(self, model,
                 hrg_statistics,  # type: HRGStatistics
                 options):
        super(EmbeddingHRGScorer, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]

        self.freq_edges = [edge for edge, count in hrg_statistics.edge_names.most_common(self.options.edge_count)]
        self.edge_embedding = nn.Embedding(self, list(self.freq_edges),
                                           options.edge_embedding_dim,
                                           init=dn.IdentityInitializer())

        dense_dims = [options.lstm_dims * 2 * options.lstm_layers + options.edge_embedding_dim] + options.hrg_mlp_dims + \
                     [1]
        # don't use bias in last transform
        use_bias = [True] * (len(dense_dims) - 2) + [False]

        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation, use_bias)
        self.attention_w1 = self.add_parameters((options.attention_dim,
                                                 options.edge_embedding_dim))
        self.attention_w2 = self.add_parameters((options.attention_dim,
                                                 options.lstm_dims * 2 * options.lstm_layers))
        self.attention_v = self.add_parameters((1, options.attention_dim))

        # self.rnn_w = self.add_parameters((hdim, 2*hdim))

    def graph_embedding(self,
                        span_feature,
                        rule  # type: Optional[HRGRule]
                        ):
        if rule is None:
            return self.edge_embedding("*PAD*")
        edges = list(rule.rhs.edges)  # type: List[HyperEdge]
        if self.options.attention:
            edge_features = dn.concatenate_cols([self.edge_embedding(edge.label) for edge in edges])
            att_weights = dn.softmax(dn.transpose(self.attention_v.expr() * dn.tanh(
                dn.colwise_add(self.attention_w1.expr() * edge_features, self.attention_w2.expr() * span_feature))))
            context = dn.reshape(edge_features * att_weights, (self.options.edge_embedding_dim,))
        else:
            edge_features = [self.edge_embedding(edge.label) for edge in edges]
            context = dn.reshape(dn.esum(edge_features), (self.options.edge_embedding_dim,))
        return context

    def get_best_rule(self,
                      span_feature,
                      rules_and_counts,  # type: Sequence[Tuple[CFGRule, int]]
                      gold=None
                      ):
        if len(rules_and_counts) == 1:
            best_rule = list(rules_and_counts)[0][0]
            return best_rule, None, best_rule

        graph_exprs = {rule: self.dense_layer(dn.concatenate([span_feature,
                                                              self.graph_embedding(span_feature, rule.hrg)]))
                       for rule, count in rules_and_counts}

        def get_score_count_rule(rule, count):
            score = graph_exprs[rule].value()
            if gold is not None and rule == gold:
                score -= self.options.hrg_loss_margin
            return score, count, rule

        score_count_rule = [get_score_count_rule(rule, count)
                            for rule, count in rules_and_counts]  # tuple (score, count, rule)
        best_score, best_count, best_rule = sorted(score_count_rule,
                                                   key=lambda x: (x[0], x[1]),
                                                   reverse=True)[0]
        _, _, real_best_rule = sorted(((score if rule != gold else score + self.options.hrg_loss_margin, count, rule)
                                       for score, count, rule in score_count_rule),
                                      key=lambda x: (x[0], x[1]),
                                      reverse=True)[0]
        best_expr = graph_exprs[best_rule]

        if gold is not None:
            if best_rule == gold:
                loss = dn.scalarInput(0.0)
            else:
                assert gold in [i[0] for i in rules_and_counts]
                gold_expr = graph_exprs[gold]
                loss = best_expr - gold_expr + self.options.hrg_loss_margin
            return best_rule, loss, real_best_rule
        return best_rule, None, real_best_rule

    def restore_components(self, components):
        self.edge_embedding, self.dense_layer, self.attention_w1, \
        self.attention_w2, self.attention_v = components