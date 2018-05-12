import sys
import math
from multiprocessing.dummy import Pool

import attr
import dynet as dn
import numpy as np

import nn
from common_utils import split_to_batches
from dynet_parser_base import DynetParserBase
from edge_eval_network import SentenceEmbeddings
from logger import logger
from supertagger.data_reader import Graph2015ForSuperTag
from tagger_base.network import POSTagClassification
from tagger_base.viterbi_crf import ViterbiDecoder, greedy_decode
from vocab_utils import Statistics


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


class SuperTagger(DynetParserBase):
    available_data_formats = {"sdp2015": Graph2015ForSuperTag}
    default_data_format_name = "sdp2015"

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(SuperTagger, cls).add_parser_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--concurrent-count", dest="concurrent_count", type=int, default=2)
        group.add_argument("--disable-viterbi", action="store_true", dest="disable_viterbi", default=False)
        SentenceEmbeddings.add_parser_arguments(arg_parser)
        POSTagClassification.add_parser_arguments(arg_parser)
        ViterbiDecoder.add_parser_arguments(arg_parser)

    @classmethod
    def options_hook(cls, options):
        # disable supertag embedding
        options.supertag_embedding = 0
        super(SuperTagger, cls).options_hook(options)

    def __init__(self, options, data_train):
        self.model = dn.Model()
        self.optimizer = nn.trainers[options.optimizer](
            *((self.model, options.learning_rate)
            if options.learning_rate is not None else (self.model,)))
        self.options = options

        self.statistics = Statistics.from_sentences(data_train)
        logger.info(str(self.statistics))
        self.container = nn.Container(self.model)
        self.sent_embeddings = SentenceEmbeddings(self.container, self.statistics, options)
        self.tag_classification = POSTagClassification(self.container, self.statistics.supertags, options)
        self.tag_dict = self.statistics.supertags
        self.viterbi_decoder = ViterbiDecoder(self.container, self.statistics.supertags, self.options)

    def __getstate__(self):
        return {i: getattr(self, i)
                for i in ("options", "statistics")
                }

    def exprs_to_scores(self, tag_exprs, gold_tags=None):
        # calculate score
        scores = np.stack([i.npvalue() for i in tag_exprs])
        if gold_tags is not None:
            for idx, gold_tag in enumerate(gold_tags):
                scores[idx, gold_tag] -= 1
        return scores

    def training_session(self, sentence, print_logger, pool):
        if sentence[1].supertag == "__NotForTraining__":
            yield []
            yield None
            yield dn.scalarInput(0.0)
            raise StopIteration

        sent_embeddings = self.sent_embeddings.get_lstm_output(sentence)
        tag_exprs = self.tag_classification(sent_embeddings)
        yield tag_exprs

        gold_tags = [self.tag_dict.word_to_int[node.supertag]
                     for node in sentence]

        # decode
        scores = self.exprs_to_scores(tag_exprs, gold_tags)
        if self.options.disable_viterbi:
            pred_tags = greedy_decode(scores)
            yield None
        else:
            pred_tags_future = pool.apply_async(self.viterbi_decoder, (scores,))
            yield pred_tags_future
            pred_tags = pred_tags_future.get()

        if self.options.disable_viterbi:
            gold_expr = dn.esum([tag_exprs[idx][tag]
                                 for idx, tag in enumerate(gold_tags)])
            pred_expr = dn.esum([tag_exprs[idx][tag]
                                 for idx, tag in enumerate(pred_tags)])
        else:
            gold_expr = self.viterbi_decoder.get_expr(tag_exprs, gold_tags)
            pred_expr = self.viterbi_decoder.get_expr(tag_exprs, pred_tags)
        loss_shift = len(pred_tags)
        # calculate correctness
        for pred_tag, gold_tag in zip(pred_tags, gold_tags):
            print_logger.total_count += 1
            if pred_tag == gold_tag:
                print_logger.correct_count += 1
                loss_shift -= 1

        loss = pred_expr - gold_expr + loss_shift
        yield loss

    def predict_session(self, sentence, pool):
        sent_embeddings = self.sent_embeddings.get_lstm_output(sentence)
        tag_exprs = self.tag_classification(sent_embeddings)
        yield tag_exprs

        # decode
        scores = self.exprs_to_scores(tag_exprs)
        if self.options.disable_viterbi:
            pred_tags = greedy_decode(scores)
            yield None
        else:
            pred_tags_future = pool.apply_async(self.viterbi_decoder, (scores,))
            yield pred_tags_future
            pred_tags = pred_tags_future.get()

        DataFormatClass = self.get_data_formats()[self.options.data_format]
        new_sentence = DataFormatClass()
        new_sentence.comment = sentence.comment
        for node, tag_int in zip(sentence, pred_tags):
            new_sentence.append(attr.evolve(
                node, sense=self.tag_dict.int_to_word[tag_int]))
        yield new_sentence

    def train(self, trees, data_train=None):
        print_logger = PrintLogger()
        pool = Pool(self.options.concurrent_count)
        print_per = (100 // self.options.batch_size + 1) * self.options.batch_size
        self.sent_embeddings.rnn.set_dropout(self.options.lstm_dropout)
        for sentence_idx, batch_idx, batch_trees in split_to_batches(
                trees, self.options.batch_size):
            if sentence_idx % print_per == 0 and sentence_idx != 0:
                print_logger.print(sentence_idx)
            sessions = [self.training_session(tree, print_logger, pool)
                        for tree in batch_trees]

            batch_size_2 = int(math.ceil(len(sessions) / 2) + 0.5)
            assert batch_size_2 * 2 >= len(sessions)
            for _, _, sub_sessions in split_to_batches(
                    sessions, batch_size_2):
                exprs = [i for session in sub_sessions for i in next(session)]
                if exprs:
                    dn.forward(exprs)
                futures = [next(session) for session in sub_sessions]
            loss = dn.esum([next(session) for session in sessions]) / len(sessions)

            # update
            print_logger.total_loss += loss.scalar_value()
            loss.backward()
            self.optimizer.update()
            dn.renew_cg()

    def predict(self, trees):
        self.sent_embeddings.rnn.disable_dropout()
        pool = Pool(self.options.concurrent_count)
        for sentence_idx, batch_idx, batch_trees in split_to_batches(
                trees, self.options.batch_size):
            sessions = [self.predict_session(tree, pool)
                        for tree in batch_trees]
            exprs = [i for session in sessions for i in next(session)]
            dn.forward(exprs)
            futures = [next(session) for session in sessions]
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
        ret.container = savable
        ret.sent_embeddings, ret.tag_classification, ret.viterbi_decoder = ret.container.components
        ret.tag_dict = ret.statistics.supertags
        ret.optimizer = nn.trainers[ret.options.optimizer](
            *((ret.model, ret.options.learning_rate)
            if ret.options.learning_rate is not None else (ret.model,)))
        logger.info("Learning rate: {}".format(ret.optimizer.learning_rate))
        ret.options.__dict__.update(new_options.__dict__)
        return ret
