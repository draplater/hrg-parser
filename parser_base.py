import random

import six
from io import open

from argparse import ArgumentParser
from pprint import pformat

import os
import sys
from abc import ABCMeta, abstractmethod, abstractproperty

import time

import graph_utils
import tree_utils
from common_utils import set_proc_name, ensure_dir, smart_open
from logger import logger, log_to_file
from training_scheduler import TrainingScheduler


@six.add_metaclass(ABCMeta)
class DependencyParserBase(object):
    DataType = None
    available_data_formats = {}
    default_data_format_name = "default"

    @classmethod
    def get_data_formats(cls):
        """ for old class which has "DataType" but not "available_data_formats" """
        if not cls.available_data_formats:
            return {"default": cls.DataType}
        else:
            return cls.available_data_formats

    @abstractmethod
    def train(self, graphs):
        pass

    @abstractmethod
    def predict(self, graphs):
        """:rtype: list[self.DataType]"""
        pass

    @abstractmethod
    def save(self, prefix):
        pass

    @classmethod
    @abstractmethod
    def load(cls, prefix, new_options=None):
        pass

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(DependencyParserBase.__name__)
        group.add_argument("--title", type=str, dest="title", default="default")
        group.add_argument("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                           required=True)
        group.add_argument("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", nargs="+",
                           required=True)
        group.add_argument("--outdir", type=str, dest="output", required=True)
        group.add_argument("--max-save", type=int, dest="max_save", default=100)
        group.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model.")
        group.add_argument("--epochs", type=int, dest="epochs", default=30)
        group.add_argument("--lr", type=float, dest="learning_rate", default=None)

    @classmethod
    def add_predict_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(DependencyParserBase.__name__)
        group.add_argument("--output", dest="out_file", help="Output file", metavar="FILE", required=True)
        group.add_argument("--model", dest="model", help="Load/Save model file", metavar="FILE", required=True)
        group.add_argument("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", required=True)
        group.add_argument("--eval", action="store_true", dest="evaluate", default=False)
        group.add_argument("--format", dest="input_format", choices=["standard", "tokenlist",
                                                                     "space", "english", "english-line"],
                           help='Input format. (default)"standard": use the same format of treebank;\n'
                                'tokenlist: like [[(sent_1_word1, sent_1_pos1), ...], [...]];\n'
                                'space: sentence is separated by newlines, and words are separated by space;'
                                'no POSTag info will be used. \n'
                                'english: raw english sentence that will be processed by NLTK tokenizer, '
                                'no POSTag info will be used.',
                           default="standard"
                           )

    @classmethod
    def add_common_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(DependencyParserBase.__name__ + "(train and test)")
        group.add_argument("--dynet-seed", type=int, dest="seed", default=0)
        group.add_argument("--dynet-autobatch", type=int, dest="autobatch", default=0)
        group.add_argument("--dynet-mem", dest="mem", default=0)
        group.add_argument("--dynet-gpus", type=int, dest="mem", default=0)
        group.add_argument("--dynet-l2", type=float, dest="l2", default=0.0)
        group.add_argument("--dynet-weight-decay", type=float, dest="weight_decay", default=0.0)
        group.add_argument("--output-scores", action="store_true", dest="output_scores", default=False)
        group.add_argument("--data-format", dest="data_format",
                           choices=cls.get_data_formats(),
                           default=cls.default_data_format_name)

    @classmethod
    def options_hook(cls, options):
        logger.info('Options:\n%s', pformat(options.__dict__))

    @classmethod
    def train_parser(cls, options, data_train=None, data_dev=None, data_test=None):
        set_proc_name(options.title)
        ensure_dir(options.output)
        path = os.path.join(options.output, "{}_{}_train.log".format(options.title,
                                                                     int(time.time())))
        log_to_file(path)
        logger.name = options.title
        cls.options_hook(options)
        DataFormatClass = cls.get_data_formats()[options.data_format]

        if data_train is None:
            data_train = DataFormatClass.from_file(options.conll_train)

        if data_dev is None:
            data_dev = {i: DataFormatClass.from_file(i, False) for i in options.conll_dev}

        try:
            os.makedirs(options.output)
        except OSError:
            pass

        parser = cls(options, data_train)
        random_obj = random.Random(1)
        for epoch in range(options.epochs):
            logger.info('Starting epoch %d', epoch)
            random_obj.shuffle(data_train)
            options.is_train = True
            parser.train(data_train)

            # save model and delete old model
            for i in range(0, epoch - options.max_save):
                path = os.path.join(options.output, os.path.basename(options.model)) + str(i + 1)
                if os.path.exists(path):
                    os.remove(path)
            path = os.path.join(options.output, os.path.basename(options.model)) + str(epoch + 1)
            parser.save(path)

            def predict(sentences, gold_file, output_file):
                options.is_train = False
                with open(output_file, "w") as f_output:
                    if hasattr(DataFormatClass, "file_header"):
                        f_output.write(DataFormatClass.file_header + "\n")
                    for i in parser.predict(sentences):
                        f_output.write(i.to_string())
                # script_path = os.path.join(os.path.dirname(__file__), "main.py")
                # p = subprocess.Popen([sys.executable, script_path, "mst+empty", "predict", "--model", path,
                #                       "--test", gold_file,
                #                       "--output", output_file], stdout=sys.stdout)
                # p.wait()
                DataFormatClass.evaluate_with_external_program(gold_file, output_file)

            for file_name, file_content in data_dev.items():
                try:
                    prefix, suffix = os.path.basename(file_name).rsplit(".", 1)
                except ValueError:
                    prefix = os.path.basename(file_name)
                    suffix = ""

                dev_output = os.path.join(options.output, '{}_epoch_{}.{}'.format(prefix, epoch + 1, suffix))
                predict(file_content, file_name, dev_output)

    @classmethod
    def predict_with_parser(cls, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        if options.input_format == "standard":
            data_test = DataFormatClass.from_file(options.conll_test, False)
        elif options.input_format == "space":
            with smart_open(options.conll_test) as f:
                data_test = [DataFormatClass.from_words_and_postags([(word, "X") for word in line.strip().split(" ")])
                             for line in f]
        elif options.input_format.startswith("english"):
            from nltk import download, sent_tokenize
            from nltk.tokenize import TreebankWordTokenizer
            download("punkt")
            with smart_open(options.conll_test) as f:
                raw_sents = []
                for line in f:
                    if options.input_format == "english-line":
                        raw_sents.append(line.strip())
                    else:
                        this_line_sents = sent_tokenize(line.strip())
                        raw_sents.extend(this_line_sents)
                tokenized_sents = TreebankWordTokenizer().tokenize_sents(raw_sents)
                data_test = [DataFormatClass.from_words_and_postags([(token, "X") for token in sent])
                             for sent in tokenized_sents]
        elif options.input_format == "tokenlist":
            with smart_open(options.conll_test) as f:
                items = eval(f.read())
            data_test = DataFormatClass.from_words_and_postags(items)
        else:
            raise ValueError("invalid format option")

        logger.info('Loading Model...')
        options.is_train = False
        parser = cls.load(options.model, options)
        logger.info('Model loaded')

        ts = time.time()
        with smart_open(options.out_file, "w") as f_output:
            if hasattr(DataFormatClass, "file_header"):
                f_output.write(DataFormatClass.file_header + "\n")
            for i in parser.predict(data_test):
                f_output.write(i.to_string())
        te = time.time()
        logger.info('Finished predicting and writing test. %.2f seconds.', te - ts)

        if options.evaluate:
            DataFormatClass.evaluate_with_external_program(options.conll_test,
                                                           options.out_file)

    @classmethod
    def get_arg_parser(cls):
        parser = ArgumentParser(sys.argv[0])
        cls.fill_arg_parser(parser)
        return parser

    @classmethod
    def fill_arg_parser(cls, parser):
        sub_parsers = parser.add_subparsers()
        sub_parsers.required = True
        sub_parsers.dest = 'mode'

        # Train
        train_subparser = sub_parsers.add_parser("train")
        cls.add_parser_arguments(train_subparser)
        cls.add_common_arguments(train_subparser)
        train_subparser.set_defaults(func=cls.train_parser)

        # Predict
        predict_subparser = sub_parsers.add_parser("predict")
        cls.add_predict_arguments(predict_subparser)
        cls.add_common_arguments(predict_subparser)
        predict_subparser.set_defaults(func=cls.predict_with_parser)

        eval_subparser = sub_parsers.add_parser("eval")
        eval_subparser.add_argument("--data-format", dest="data_format",
                                    choices=cls.get_data_formats(),
                                    default=cls.default_data_format_name)
        eval_subparser.add_argument("gold")
        eval_subparser.add_argument("system")
        eval_subparser.set_defaults(func=cls.eval_only)

    @classmethod
    def get_training_scheduler(cls, train=None, dev=None, test=None):
        return TrainingScheduler(cls.train_parser, cls, train, dev, test)

    @classmethod
    def eval_only(cls, options):
        DataFormatClass = cls.get_data_formats()[options.data_format]
        DataFormatClass.evaluate_with_external_program(options.gold, options.system)

    @classmethod
    def get_next_arg_parser(cls, stage, options):
        return None


@six.add_metaclass(ABCMeta)
class GraphParserBase(DependencyParserBase):
    available_data_formats = {"sdp2014": graph_utils.Graph, "sdp2015": graph_utils.Graph2015}
    default_data_format_name = "sdp2014"


@six.add_metaclass(ABCMeta)
class TreeParserBase(DependencyParserBase):
    available_data_formats = {"conllu": tree_utils.Sentence}
    default_data_format_name = "conllu"
