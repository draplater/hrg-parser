from collections import namedtuple

import nn
import numpy as np
import dynet as dn
from logger import logger
from nn import DynetSaveable


class SpanFeature(object):
    def __init__(self, fwd, bwd):
        self.fwd = fwd
        self.bwd = bwd

    def __getitem__(self, item):
        left, right = item
        return dn.concatenate([self.fwd[right] - self.fwd[left],
                               self.bwd[left+1] - self.bwd[right+1]])

    def __len__(self):
        return len(self.fwd) - 2


class BiLSTMMinus(DynetSaveable):
    LSTMLayer = namedtuple("LSTMLayer", ["fwd", "back"])
    def __init__(self, model, dims, builder=dn.VanillaLSTMBuilder):
        super(BiLSTMMinus, self).__init__(model)
        self.rnns = []
        for i in range(len(dims) - 1):
            self.rnns.append(self.LSTMLayer(
                self.add_lstm_builder(builder, 1, dims[i], dims[i+1] / 2),
                self.add_lstm_builder(builder, 1, dims[i], dims[i+1] / 2)
            ))

    def __call__(self, word_embeddings):
        input_tensors = word_embeddings
        layers_fwd_out = []
        layers_back_out = []

        for forward_cell, backward_cell in self.rnns:
            forward_results = forward_cell.initial_state().transduce(input_tensors)
            layers_fwd_out.append(forward_results)

            input_tensors.reverse()
            backward_results = backward_cell.initial_state().transduce(input_tensors)
            backward_results.reverse()
            layers_back_out.append(backward_results)

            input_tensors = [dn.concatenate([forward, backward])
                             for forward, backward in zip(forward_results,
                                                          backward_results)]

        fwd_out = [dn.concatenate(list(items)) for items in zip(*layers_fwd_out)]
        back_out = [dn.concatenate(list(items)) for items in zip(*layers_back_out)]

        return SpanFeature(fwd_out, back_out)

    def __getstate__(self):
        return {"params": self.params}

    def set_dropout(self, dropout):
        for fwd, back in self.rnns:
            fwd.set_dropout(dropout)
            back.set_dropout(dropout)

    def disable_dropout(self):
        for fwd, back in self.rnns:
            fwd.disable_dropout()
            back.disable_dropout()

    def restore_components(self, components):
        self.rnns = []
        for i in range(0, len(components), 2):
            self.rnns.append(self.LSTMLayer(
                components[i],
                components[i+1]
            ))


class SpanEmbeddings(DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--wembedding", type=int, dest="wembedding_dims", default=50)
        group.add_argument("--pembedding", type=int, dest="pembedding_dims", default=20)
        group.add_argument("--cembedding", type=int, dest="cembedding_dims", default=50)
        group.add_argument("--extrn", dest="ext_embedding", help="External embeddings", metavar="FILE")
        group.add_argument("--word-threshold", type=int, dest="word_threshold", default=0)
        group.add_argument("--clstmlayers", type=int, dest="clstm_layers", default=2)
        group.add_argument("--lstmlayers", type=int, dest="lstm_layers", default=2)
        group.add_argument("--span-lstm-layers", type=int, dest="span_lstm_layers", default=2)
        group.add_argument("--lstmdims", type=int, dest="lstm_dims", default=200)
        group.add_argument("--lstm-dropout", type=float, dest="lstm_dropout", default=0.0)

    def __init__(self, model, statistics, options):
        super(SpanEmbeddings, self).__init__(model)
        self.options = options

        self.ldims = options.lstm_dims

        if options.ext_embedding is not None:
            self.ext_embedding = nn.ExternalEmbedding(self, options.ext_embedding,
                                                      extra=("*EMPTY*", "*START*", "*END*"))
            e_dim = self.ext_embedding.dim
            logger.info('Load external embedding. Vector dimensions %d', self.ext_embedding.dim)
        else:
            self.ext_embedding = None
            e_dim = 0

        self.total_dims = options.wembedding_dims + options.pembedding_dims + e_dim
        self.rnn = BiLSTMMinus(self, [self.total_dims] + [self.ldims * 2] * options.span_lstm_layers)

        if options.cembedding_dims > 0 and options.word_threshold > 1:
            self.char_embedding = nn.Embedding(self,
                                               list(statistics.characters),
                                               options.cembedding_dims)
            self.c_lstm = nn.BiLSTM(self, [options.cembedding_dims] + [options.wembedding_dims] * options.span_lstm_layers)
            self.freq_words = set(word for word, count in statistics.words.items()
                                  if count >= options.word_threshold)
            logger.info("Word embedding size: {}".format(len(self.freq_words)))
            self.word_embedding = nn.Embedding(self,
                                               self.freq_words, options.wembedding_dims,
                                               extra=("*EMPTY*", "*START*", "*END*"))
        else:
            self.word_embedding = nn.Embedding(self, list(statistics.words), options.wembedding_dims)

        if options.pembedding_dims > 0:
            self.pos_embedding = nn.Embedding(self, list(statistics.postags),
                                              options.pembedding_dims,
                                              extra=("*EMPTY*", "*START*", "*END*"))
        self.init_special()

    def __getstate__(self):
        self.start_feature = None
        self.end_feature = None
        return super(SpanEmbeddings, self).__getstate__()

    def init_special(self):
        start_embeddings = [self.word_embedding("*START*")]
        if self.options.pembedding_dims > 0:
            start_embeddings.append(self.pos_embedding("*START*"))
        if self.ext_embedding is not None:
            start_embeddings.append(self.ext_embedding("*START*"))
        self.start_feature = dn.concatenate(start_embeddings)

        end_embeddings = [self.word_embedding("*END*")]
        if self.options.pembedding_dims > 0:
            end_embeddings.append(self.pos_embedding("*END*"))
        if self.ext_embedding is not None:
            end_embeddings.append(self.ext_embedding("*END*"))
        self.end_feature = dn.concatenate(end_embeddings)

    def restore_components(self, components):
        if self.options.ext_embedding is not None:
            self.ext_embedding = components.pop(0)
        self.rnn = components.pop(0)
        if self.options.cembedding_dims > 0 and self.options.word_threshold > 1:
            self.char_embedding = components.pop(0)
            self.c_lstm = components.pop(0)
        self.word_embedding = components.pop(0)
        if self.options.pembedding_dims > 0:
            self.pos_embedding = components.pop(0)
        assert not components

    def get_vecs(self, node):
        #  word -> input vector of LSTM
        if self.options.cembedding_dims == 0 or self.options.word_threshold <= 1 \
                or node.norm in self.freq_words:
            word_vec = self.word_embedding(node.norm)
        else:
            char_vecs = [self.char_embedding(i) for i in node.norm]
            char_vecs_o = self.c_lstm(char_vecs)
            word_vec = (char_vecs_o[0] + char_vecs_o[-1]) / 2
        vecs = [word_vec]

        if self.options.pembedding_dims > 0:
            pos_vec = self.pos_embedding(node.postag)
            vecs.append(pos_vec)

        if self.ext_embedding is not None:
            ext_vec = self.ext_embedding(node.form, (node.norm,))
            vecs.append(ext_vec)

        return dn.concatenate(vecs)

    def get_span_features(self, sentence):
        return self.rnn([self.start_feature] + [self.get_vecs(i) for i in sentence] + [self.end_feature])


class SpanEvaluation(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--activation", type=str, dest="activation", default="relu")
        group.add_argument("--mlp-dims", dest="mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[100])

    def __init__(self, model, options):
        super(SpanEvaluation, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]
        self.ldims = options.lstm_dims

        dense_dims = [options.lstm_dims * 2 * options.span_lstm_layers] + options.mlp_dims + [1]
        # don't use bias in last transform
        use_bias = [True] * (len(dense_dims) - 2) + [False]

        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation, use_bias)

    def get_span_scores(self, spans):
        length = len(spans)
        result = [[None for i in range(length+1)]
                  for j in range(length+1)
                  ]
        for i in range(length):
            for j in range(i+1, length+1):
                result[i][j] = self.dense_layer(spans[i,j])

        return result

    def get_span_scores_values(self, result):
        length = len(result) - 1
        scores = np.zeros((length, length+1), dtype=np.float64)

        active_exprs = []
        for i in range(length):
            for j in range(i+1, length+1):
                active_exprs.append(result[i][j])

        values = dn.npvalues(active_exprs)
        idx = 0
        for i in range(length):
            for j in range(i+1, length+1):
                scores[i, j] = values[idx]
                idx += 1

        assert idx == len(values)
        return scores

    def restore_components(self, components):
        self.dense_layer, = components


class LabelEvaluation(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--label-mlp-dims", dest="label_mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[100])

    def __init__(self, model, tag_dict, options):
        super(LabelEvaluation, self).__init__(model)
        self.options = options
        self.activation = nn.activations[options.activation]
        self.label_count = len(tag_dict)

        dense_dims = [options.lstm_dims * 2 * options.span_lstm_layers] + options.label_mlp_dims + [len(tag_dict)]
        # don't use bias in last transform
        use_bias = [True] * (len(dense_dims) - 2) + [False]

        self.dense_layer = nn.DenseLayers(self, dense_dims, self.activation, use_bias)

    def get_span_scores(self, spans):
        length = len(spans)
        result = [[None for i in range(length+1)]
                  for j in range(length+1)
                  ]
        for i in range(length):
            for j in range(i+1, length+1):
                result[i][j] = self.dense_layer(spans[i,j])

        return result

    def get_span_scores_values(self, results):
        length = len(results) - 1
        active_exprs = []
        for i in range(length):
            for j in range(i+1, length+1):
                active_exprs.append(results[i][j])

        scores = np.zeros((length, length+1, self.label_count), dtype=np.float64)
        values = dn.npvalues(active_exprs)
        idx = 0
        for i in range(length):
            for j in range(i+1, length+1):
                scores[i, j] = values[idx]
                idx += 1
        assert idx == len(active_exprs)
        return scores

    def restore_components(self, components):
        self.dense_layer, = components
