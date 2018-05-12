import time
import sys

from os.path import dirname

import math

from common_utils import split_to_batches
from logger import logger
import nn
import dynet as dn

from parser_base import DependencyParserBase
from span.leaftagger import POSTagParser
from span.const_tree import ConstTree, ConstTreeStatistics
from span.network import SpanEvaluation, SpanEmbeddings, LabelEvaluation
from multiprocessing.dummy import Pool as ThreadPool

import pyximport

pyximport.install(build_dir=dirname(__file__) + "/.cache/")
from span.cdecoder import CKYBinaryRuleDecoder, CKYRuleFreeDecoder


class PrintLogger(object):
    def reset(self):
        self.total_loss = 0.0
        self.total_gold_span = sys.float_info.epsilon
        self.total_predict_span = sys.float_info.epsilon
        self.correct_predict_span = 0.0
        self.start_time = time.time()
        self.loss = dn.scalarInput(0.0)

    def __init__(self):
        self.reset()

    def print(self, sent_idx):
        logger.info(
            'Processing sentence number: %d, Loss: %.2f, '
            'Accuracy: %.2f, Recall: %.2f, '
            'Time: %.2f',
            sent_idx, self.total_loss,
            self.correct_predict_span / self.total_predict_span * 100,
            self.correct_predict_span / self.total_gold_span * 100,
            time.time() - self.start_time
        )
        self.reset()


decoder_types = {"binary": CKYBinaryRuleDecoder, "rulefree": CKYRuleFreeDecoder}


class SpanParser(DependencyParserBase):
    DataType = ConstTree

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        super(SpanParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--optimizer", type=str, dest="optimizer", default="adam", choices=nn.trainers.keys())
        group.add_argument("--batch-size", type=int, dest="batch_size", default=32)
        group.add_argument("--decoder", dest="decoder", choices=decoder_types, default="rulefree")
        group.add_argument("--model-format", dest="model_format", choices=nn.model_formats, default="pickle")

        SpanEmbeddings.add_parser_arguments(arg_parser)
        SpanEvaluation.add_parser_arguments(arg_parser)
        LabelEvaluation.add_parser_arguments(arg_parser)

    @classmethod
    def add_predict_arguments(cls, arg_parser):
        super(SpanParser, cls).add_predict_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--model-format", dest="model_format", choices=nn.model_formats, default=None)

    @classmethod
    def add_common_arguments(cls, arg_parser):
        super(SpanParser, cls).add_common_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__ + "(common)")
        group.add_argument("--leaftag-model", dest="leaftag_model")
        group.add_argument("--concurrent-count", dest="concurrent_count", type=int, default=2)
        group.add_argument("--max-sentence-length", type=int, dest="max_sentence_length", default=100)

    def __init__(self, options, train_trees=None, restored_model_and_network=None):
        self.options = options
        self.decode_type = decoder_types[self.options.decoder]
        if "func" in options:
            del options.func

        if restored_model_and_network:
            self.model, self.container = restored_model_and_network
            self.span_ebd_network, self.span_eval_network, self.label_eval_network = self.container.components
        else:
            self.model = dn.Model()
            self.statistics = statistics = ConstTreeStatistics.from_sentences(train_trees)
            logger.info(statistics)
            self.int_to_label = statistics.labels.int_to_word
            self.label_to_int = statistics.labels.word_to_int
            self.container = nn.Container(self.model)
            self.span_ebd_network = SpanEmbeddings(self.container, statistics, options)
            self.span_eval_network = SpanEvaluation(self.container, self.options)
            self.label_eval_network = LabelEvaluation(self.container, statistics.labels, self.options)
            self.create_decoders()

        if train_trees is not None:
            # when training, pre-convert tree to sentence to construct LSTM input
            for tree in train_trees:
                tree.extra["Sentence"] = tree.to_sentence()

        self.optimizer = nn.trainers[options.optimizer](self.model)

        if self.options.leaftag_model is not None:
            self.leaf_tagger = POSTagParser.load(self.options.leaftag_model, None)
            self.tagger_trained = True
        else:
            self.leaf_tagger = None

    def create_decoders(self):
        # pre allocate memory for decoders
        self.decoders = [self.decode_type(max(self.options.max_sentence_length,
                                              self.statistics.max_sentence_length + 10),
                                          len(self.statistics.labels))
                         for _ in range(int(self.options.batch_size / 2 + 1))]

    def spawn_decoder(self, pool, span_scores, label_scores, leaftag_scores, decoder):
        return pool.apply_async(decoder,
                                (self.statistics.rules,
                                 span_scores, label_scores,
                                 leaftag_scores,
                                 self.statistics.leaftag_to_label,
                                 self.statistics.internal_labels))

    def predict_session(self, tree, pool, decoder):
        # stage1: generate all expressions
        sentence_interface = tree.to_sentence()
        span_features = self.span_ebd_network.get_span_features(sentence_interface)
        span_exprs = self.span_eval_network.get_span_scores(span_features)
        label_exprs = self.label_eval_network.get_span_scores(span_features)
        if self.leaf_tagger is not None:
            sent_embeddings = self.leaf_tagger.sent_embeddings.get_lstm_output(sentence_interface)
            tag_exprs = self.leaf_tagger.tag_classification(sent_embeddings)
            coarse_tag_exprs = self.leaf_tagger.coarsetag_classification(sent_embeddings)
        else:
            tag_exprs = coarse_tag_exprs = []

        yield [j for i in span_exprs for j in i if j is not None] + \
              [j for i in label_exprs for j in i if j is not None] + \
              tag_exprs + coarse_tag_exprs

        # stage2: generate all scores and spawn decoder
        span_scores = self.span_eval_network.get_span_scores_values(span_exprs)
        label_scores = self.label_eval_network.get_span_scores_values(label_exprs)
        if self.leaf_tagger:
            leaftag_scores = self.leaf_tagger.exprs_to_scores(tag_exprs, coarse_tag_exprs)
        else:
            leaftag_scores = None
        decoded_future = self.spawn_decoder(pool, span_scores,
                                            label_scores, leaftag_scores, decoder)
        yield decoded_future

        # stage3: generate result
        try:
            decoded = decoded_future.get()
            ret = decoded.to_const_tree(self.int_to_label,
                                        list(tree.generate_words())
                                        ).expanded_unary_chain()
        except ArithmeticError:
            logger.info("Can not decode sentence {}".format(tree.extra.get("ID")))
            ret = ConstTree.from_words_and_postags(
                [(word, "__ERROR__")
                 for word in tree.generate_words()])
        ret.extra = tree.extra
        yield ret

    def predict(self, trees):
        self.span_ebd_network.rnn.disable_dropout()
        pool = ThreadPool(self.options.concurrent_count)
        for sentence_idx, batch_idx, batch_trees in split_to_batches(
                trees, len(self.decoders)):
            self.span_ebd_network.init_special()
            sessions = [self.predict_session(tree, pool, decoder)
                        for tree, decoder in zip(batch_trees, self.decoders)]
            # stage1: generate all expressions and forward
            expressions = [j for i in sessions for j in next(i)]
            dn.forward(expressions)
            # stage2: spawn all decoders
            for session in sessions:
                next(session)
            # stage3: get all results
            for session in sessions:
                yield next(session)
            dn.renew_cg()

    def train_session(self, tree, print_logger, pool, decoder):
        loss = dn.scalarInput(0.0)

        # stage1: generate all expressions
        tree_t = tree.condensed_unary_chain()
        sentence_interface = tree.extra["Sentence"]
        span_features = self.span_ebd_network.get_span_features(sentence_interface)
        span_exprs = self.span_eval_network.get_span_scores(span_features)
        label_exprs = self.label_eval_network.get_span_scores(span_features)
        if self.leaf_tagger is not None:
            sent_embeddings = self.leaf_tagger.sent_embeddings.get_lstm_output(sentence_interface)
            tag_exprs = self.leaf_tagger.tag_classification(sent_embeddings)
            coarse_tag_exprs = self.leaf_tagger.coarsetag_classification(sent_embeddings)
        else:
            tag_exprs = coarse_tag_exprs = []

        yield [j for i in span_exprs for j in i if j is not None] + \
              [j for i in label_exprs for j in i if j is not None] + \
              tag_exprs + coarse_tag_exprs

        gold_labels = set(tree_t.generate_scoreable_spans())
        gold_spans = set(i[0:2] for i in gold_labels)
        span_scores = self.span_eval_network.get_span_scores_values(span_exprs)
        label_scores = self.label_eval_network.get_span_scores_values(label_exprs)
        if self.leaf_tagger is not None:
            leaftag_scores = self.leaf_tagger.exprs_to_scores(tag_exprs, coarse_tag_exprs)
        else:
            leaftag_scores = None

        for span in gold_spans:
            span_scores[span] -= 1

        for start, end, label in gold_labels:
            if end - start == 1:
                if self.leaf_tagger is None:
                    label_scores[start, end, self.label_to_int[label]] -= 1
                else:
                    leaftag_scores[start,
                                   self.statistics.leaftags.word_to_int[label]] -= 1
            else:
                label_scores[start, end, self.label_to_int[label]] -= 1

        decoded_future = self.spawn_decoder(pool, span_scores,
                                            label_scores, leaftag_scores, decoder)
        yield decoded_future

        try:
            decoded = decoded_future.get()
        except ArithmeticError:
            logger.info("Can not decode sentence {}".format(tree.extra.get("ID")))
            yield dn.scalarInput(0.0)
            raise StopIteration

        predicted_labels = set(decoded.generate_scoreable_spans(
            self.int_to_label))

        # stage3: calculate loss
        for span in gold_labels | predicted_labels:
            start, end, label = span
            predicted_exist = span in predicted_labels
            gold_exist = span in gold_labels
            if gold_exist and predicted_exist:
                print_logger.total_gold_span += 1
                print_logger.total_predict_span += 1
                print_logger.correct_predict_span += 1
            elif gold_exist and not predicted_exist:
                print_logger.total_gold_span += 1
                if end - start == 1:
                    if self.leaf_tagger is None:
                        loss -= label_exprs[start][end][self.label_to_int[label]] - 1
                else:
                    loss -= span_exprs[start][end] - 1
                    loss -= label_exprs[start][end][self.label_to_int[label]] - 1
            else:
                assert not gold_exist and predicted_exist
                print_logger.total_predict_span += 1
                if end - start == 1:
                    if self.leaf_tagger is None:
                        loss += label_exprs[start][end][self.label_to_int[label]]
                else:
                    try:
                        loss += span_exprs[start][end]
                    except NotImplementedError:
                        print(loss, span_exprs[start][end])
                    loss += label_exprs[start][end][self.label_to_int[label]]
        yield loss

    def train(self, sentences):
        self.span_ebd_network.rnn.set_dropout(self.options.lstm_dropout)
        self.span_ebd_network.init_special()
        pool = ThreadPool(self.options.concurrent_count)
        print_logger = PrintLogger()
        print_per = (100 // self.options.batch_size + 1) * self.options.batch_size

        for sentence_idx, batch_idx, batch_trees in split_to_batches(
                sentences, self.options.batch_size):
            if sentence_idx != 0 and sentence_idx % print_per == 0:
                print_logger.print(sentence_idx)

            self.span_ebd_network.init_special()
            sessions = [self.train_session(tree, print_logger, pool, decoder)
                        for tree, decoder in zip(batch_trees, self.decoders)]

            batch_size_2 = int(math.ceil(len(sessions) / 2) + 0.5)
            assert batch_size_2 * 2 >= len(sessions)
            for _, _, sub_sessions in split_to_batches(
                    sessions, batch_size_2):
                # stage1: generate all expressions and forward
                expressions = [j for i in sub_sessions for j in next(i)]
                dn.forward(expressions)
                # stage2: spawn all decoders
                for session in sub_sessions:
                    next(session)
            # stage3: get all losses
            loss = dn.esum([next(session) for session in sessions])
            loss /= len(sessions)
            print_logger.total_loss += loss.value()
            loss.backward()
            self.optimizer.update()
            dn.renew_cg()
            self.span_ebd_network.init_special()

    def save(self, prefix):
        nn.model_save_helper(self.options.model_format, prefix,
                             self.container, [self.options, self.statistics])

    @classmethod
    def load(cls, prefix, new_options=None):
        """
        :param prefix: model file name prefix
        :type prefix: str
        :rtype: MaxSubGraphParser
        """
        model = dn.Model()
        model_format = new_options.model_format if new_options is not None else None
        (options, statistics), savable = nn.model_load_helper(
            model_format, prefix, model)
        logger.info(statistics)
        if new_options:
            options.__dict__.update(new_options.__dict__)
        ret = cls(options, None, (model, savable))
        ret.statistics = statistics
        ret.label_to_int = statistics.labels.word_to_int
        ret.int_to_label = statistics.labels.int_to_word
        ret.create_decoders()
        return ret
