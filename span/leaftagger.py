import sys

import nn
import numpy as np
import dynet as dn

from common_utils import split_to_batches
from edge_eval_network import SentenceEmbeddings
from logger import logger
from parser_base import DependencyParserBase
from span.const_tree import ConstTree, ConstTreeStatistics
from tagger_base.viterbi_crf import ViterbiDecoder, greedy_decode
from tagger_base.network import POSTagClassification


class PrintLogger(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_count = sys.float_info.epsilon
        self.correct_count = 0
        self.total_loss = 0

    def print(self, idx):
        logger.info("Sentence %d, Correctness: %.2f, Loss: %.2f",
                    idx, self.correct_count / self.total_count * 100,
                    self.total_loss)
        self.reset()


class POSTagParser(DependencyParserBase):
    DataType = ConstTree

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(POSTagParser, cls).add_parser_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--batch-size", dest="batch_size", type=int, default=32)
        group.add_argument("--optimizer", type=str, dest="optimizer", default="adam", choices=nn.trainers.keys())
        group.add_argument("--disable-viterbi", action="store_true", dest="disable_viterbi", default=False)
        SentenceEmbeddings.add_parser_arguments(arg_parser)
        POSTagClassification.add_parser_arguments(arg_parser)
        ViterbiDecoder.add_parser_arguments(arg_parser)

    @classmethod
    def options_hook(cls, options):
        # disable postag embedding
        options.pembedding_dims = 0
        super(POSTagParser, cls).options_hook(options)

    def __init__(self, options, data_train):
        self.model = dn.Model()
        self.optimizer = nn.trainers[options.optimizer](
            *((self.model, options.learning_rate)
            if options.learning_rate is not None else (self.model,)))
        self.options = options

        self.statistics = ConstTreeStatistics.from_sentences(data_train)
        logger.info(self.statistics)
        self.container = nn.Container(self.model)
        self.sent_embeddings = SentenceEmbeddings(self.container, self.statistics, options)
        self.tag_classification = POSTagClassification(self.container, self.statistics.leaftags, options)
        self.coarsetag_classification = POSTagClassification(self.container, self.statistics.coarsetags, options)
        self.tag_dict = self.statistics.leaftags
        self.coarse_tag_dict = self.statistics.coarsetags
        self.viterbi_decoder = ViterbiDecoder(self.container, self.statistics.leaftags, self.options)

    def __getstate__(self):
        return {i: getattr(self, i)
                for i in ("options", "statistics")
                }

    def exprs_to_scores(self, tag_exprs, coarse_tag_exprs,
                        gold_tags=None, gold_coarsetags=None):
        # calculate score
        scores = np.stack([i.npvalue() for i in tag_exprs])
        coarse_scores = np.stack([i.npvalue() for i in coarse_tag_exprs])

        if gold_tags is not None and gold_coarsetags is not None:
            for idx, (gold_tag, gold_coarsetag) in enumerate(zip(gold_tags, gold_coarsetags)):
                scores[idx, gold_tag] -= 1
                coarse_scores[idx, gold_coarsetag] -= 1

        coarse_scores_expanded = np.zeros(scores.shape, scores.dtype)
        for leaftag_idx, coarsetag_idx in enumerate(self.statistics.leaftag_to_coarsetag):
            coarse_scores_expanded[:, leaftag_idx] = coarse_scores[:, coarsetag_idx]
        return scores + coarse_scores_expanded

    def decode(self, scores):
        if self.options.disable_viterbi:
            return greedy_decode(scores)
        else:
            return self.viterbi_decoder(scores)

    def training_session(self, tree, print_logger):
        tree_t = tree.condensed_unary_chain()
        sentence = tree.to_sentence()
        sent_embeddings = self.sent_embeddings.get_lstm_output(sentence)
        tag_exprs = self.tag_classification(sent_embeddings)
        coarse_tag_exprs = self.coarsetag_classification(sent_embeddings)
        yield tag_exprs + coarse_tag_exprs

        gold_tags = [self.tag_dict.word_to_int[postag]
                     for word, postag in tree_t.generate_word_and_postag()]
        gold_coarsetags = [self.coarse_tag_dict.word_to_int[
                               ConstTreeStatistics.tag_to_coursetag(postag)]
                           for word, postag in tree_t.generate_word_and_postag()]

        # decode
        pred_tags = self.decode(
            self.exprs_to_scores(tag_exprs, coarse_tag_exprs,
                                 gold_tags, gold_coarsetags))  # type: list

        pred_coarsetags = [self.statistics.leaftag_to_coarsetag[leaftag_idx]
                           for leaftag_idx in pred_tags]

        gold_expr = dn.esum([tag_exprs[idx][tag] + coarse_tag_exprs[idx][coarse_tag]
                             for idx, (tag, coarse_tag) in
                             enumerate(zip(gold_tags, gold_coarsetags))])

        # calculate loss
        pred_expr = dn.esum([tag_exprs[idx][tag] + coarse_tag_exprs[idx][coarse_tag]
                             for idx, (tag, coarse_tag) in
                             enumerate(zip(pred_tags, pred_coarsetags))])

        loss_shift = len(pred_tags)
        # calculate correctness
        for pred_tag, gold_tag in zip(pred_tags, gold_tags):
            print_logger.total_count += 1
            if pred_tag == gold_tag:
                print_logger.correct_count += 1
                loss_shift -= 1

        loss = pred_expr - gold_expr + loss_shift
        yield loss

    def predict_session(self, tree):
        sentence = tree.to_sentence()
        sent_embeddings = self.sent_embeddings.get_lstm_output(sentence)
        tag_exprs = self.tag_classification(sent_embeddings)
        coarse_tag_exprs = self.coarsetag_classification(sent_embeddings)
        yield tag_exprs + coarse_tag_exprs

        # decode
        pred_tags = self.decode(
            self.exprs_to_scores(tag_exprs, coarse_tag_exprs))  # type: list
        # pred_coarsetags = [self.statistics.leaftag_to_coarsetag[leaftag_idx]
        #                     for leaftag_idx in pred_tags]

        words_and_postags = [(word.string,
                              self.tag_dict.int_to_word[tag_int])
                             for word, tag_int in zip(tree.generate_words(), pred_tags)]
        new_tree = ConstTree.from_words_and_postags(
            words_and_postags, False).expanded_unary_chain()
        yield new_tree

    def train(self, trees, data_train=None):
        print_logger = PrintLogger()
        print_per = (100 // self.options.batch_size + 1) * self.options.batch_size
        self.sent_embeddings.rnn.set_dropout(self.options.lstm_dropout)
        for sentence_idx, batch_idx, batch_trees in split_to_batches(
                trees, self.options.batch_size):
            if sentence_idx % print_per == 0 and sentence_idx != 0:
                print_logger.print(sentence_idx)
            sessions = [self.training_session(tree, print_logger)
                        for tree in batch_trees]
            exprs = [i for session in sessions for i in next(session)]
            dn.forward(exprs)
            loss = dn.esum([next(session) for session in sessions]) / len(sessions)

            # update
            print_logger.total_loss += loss.scalar_value()
            loss.backward()
            self.optimizer.update()
            dn.renew_cg()

    def predict(self, trees):
        self.sent_embeddings.rnn.disable_dropout()
        for sentence_idx, batch_idx, batch_trees in split_to_batches(
                trees, self.options.batch_size):
            sessions = [self.predict_session(tree)
                        for tree in batch_trees]
            exprs = [i for session in sessions for i in next(session)]
            dn.forward(exprs)
            for session in sessions:
                yield next(session)
            dn.renew_cg()

    def save(self, prefix):
        nn.model_save_helper("pickle", prefix, self.container, self)

    @classmethod
    def load(cls, prefix, new_options=None):
        model = dn.Model()
        ret, savable = nn.model_load_helper("pickle", prefix, model)
        ret.model = model
        ret.optimizer = nn.trainers[ret.options.optimizer](
            *((ret.model, ret.options.learning_rate)
            if ret.options.learning_rate is not None else (ret.model,)))
        ret.container = savable
        ret.sent_embeddings, ret.tag_classification, ret.coarsetag_classification, ret.viterbi_decoder = ret.container.components
        ret.tag_dict = ret.statistics.leaftags
        ret.coarse_tag_dict = ret.statistics.coarsetags
        return ret
