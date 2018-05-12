from six.moves import zip
import pickle
import random

import dynet as dn

import graph_utils
import nn
import tree_utils
from vocab_utils import Statistics
from max_sub_graph.model import MaxSubGraphParser
from max_sub_tree.mstlstm import MaxSubTreeParser
from .network import EdgeDoubleEvaluationNetwork
from parser_base import GraphParserBase, DependencyParserBase, TreeParserBase
from max_sub_graph import cost_augments, graph_decoders
from max_sub_tree.decoder import decoders as tree_decoders


class Proxy(object):
    def __init__(self, subject, attrs=None):
        self._subject = subject
        if attrs is not None:
            for k, v in attrs.items():
                setattr(self, k, v)

    def __getattr__(self, attr):
        return getattr(self._subject, attr)

    def __call__(self, *args, **kwargs):
        return self._subject(*args, **kwargs)


class CouplingTreeAndGraphParser(DependencyParserBase):
    DataType = None

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        super(CouplingTreeAndGraphParser, cls).add_parser_arguments(arg_parser)

        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--trainer", type=str, dest="trainer", default="adam", choices=nn.trainers.keys())
        group.add_argument("--default-state", type=str, dest="state", required=True)
        group.add_argument("--cost-augment", type=str, dest="cost_augment", default="simple")
        group.add_argument("--decoder", type=str, dest="decoder", default="arcfactor", choices=graph_decoders.keys())
        group.add_argument("--predict-decoder", type=str, dest="test_decoder", default=None)
        group.add_argument("--hamming-a", type=float, dest="hamming_a", default=0.4)
        group.add_argument("--hamming-b", type=float, dest="hamming_b", default=0.6)
        group.add_argument("--vine-arc-length", type=int, dest="vine_arc_length", default=20)
        group.add_argument("--basic-costaug-decrease", type=int, dest="basic_costaug_decrease", default=1)
        group.add_argument("--tree-decoder", type=str, dest="tree_decoder", default="eisner",
                           choices=tree_decoders.keys())
        group.add_argument("--tree-cost-augment", action="store_true", dest="tree_cost_augment", default=True)

        EdgeDoubleEvaluationNetwork.add_parser_arguments(arg_parser)

    def __init__(self, options, train_data=None, restore_file=None):
        self.model = dn.Model()
        random.seed(1)
        self.trainer = nn.trainers[options.trainer](self.model)

        self.graph_decoder = graph_decoders[options.decoder](options)
        self.graph_test_decoder = graph_decoders[options.test_decoder](options) \
            if options.test_decoder is not None \
            else self.graph_decoder
        self.tree_decoder = tree_decoders[options.tree_decoder]
        self.graph_cost_augment = cost_augments[options.cost_augment]
        self.costaugFlag = options.tree_cost_augment

        self.options = options

        if "func" in options:
            del options.func

        self.labelsFlag = options.labelsFlag

        if restore_file:
            self.both_network, = self.model.load(restore_file)
        else:
            statistics_graph = Statistics.from_sentences(train_data[0])
            statistics_tree = Statistics.from_sentences(train_data[1])
            self.both_network = EdgeDoubleEvaluationNetwork(self.model, (statistics_graph, statistics_tree), options)

        self.state = options.state

        self.graph_parser_view = Proxy(self)
        self.graph_parser_view.network = self.both_network.graph_network
        self.graph_parser_view.decoder = self.graph_decoder
        self.graph_parser_view.cost_augment = self.graph_cost_augment

        self.tree_parser_view = Proxy(self)
        self.tree_parser_view.network = self.both_network.tree_network
        self.tree_parser_view.decoder = self.tree_decoder

    # noinspection PyCallByClass,PyTypeChecker,PyAttributeOutsideInit
    def predict(self, data):
        if self.state == "graph":
            self.network = self.both_network.graph_network
            self.test_decoder = self.graph_test_decoder
            for i in MaxSubGraphParser.predict.im_func(self, data):
                yield i
            del self.network, self.test_decoder
        elif self.state == "tree":
            self.network = self.both_network.tree_network
            self.decoder = self.tree_decoder
            for i in MaxSubTreeParser.predict.im_func(self, data):
                yield i
            del self.network, self.decoder
        else:
            raise TypeError

    # noinspection PyCallByClass,PyTypeChecker,PyAttributeOutsideInit
    def train(self, data):
        if self.state == "graph":
            return MaxSubGraphParser.train.im_func(self.graph_parser_view, data)
        elif self.state == "tree":
            return MaxSubTreeParser.train.im_func(self.tree_parser_view, data)
        else:
            raise TypeError

    def train_alternative(self, graphs, trees):
        for graph_loss, tree_loss in zip(MaxSubGraphParser.train_gen.im_func(self.graph_parser_view, graphs, False),
                                         MaxSubTreeParser.train_gen.im_func(self.tree_parser_view, trees, False)):
            total_loss = graph_loss + tree_loss
            total_loss.backward()
            self.trainer.update()
            dn.renew_cg()
        self.trainer.update_epoch()

    @classmethod
    def train_parser(cls, options, train_data=None, test_data=None, dev_data=None):
        if options.state == "graph":
            GraphParserBase.train_parser.im_func(GraphParserClassView, options, train_data, test_data, dev_data)
        elif options.state == "tree":
            TreeParserBase.train_parser.im_func(TreeParserClassView, options, train_data, test_data, dev_data)

    @classmethod
    def predict_with_parser(cls, options):
        raise NotImplementedError

    def save(self, prefix):
        with open(prefix + ".options", "wb") as f:
            pickle.dump(self.options, f)
        # noinspection PyArgumentList
        self.model.save(prefix, [self.both_network])

    @classmethod
    def load(cls, prefix):
        """
        :param prefix: model file name prefix
        :type prefix: str
        :rtype: MaxSubGraphParser
        """
        with open(prefix + ".options") as f:
            options = pickle.load(f)
        ret = cls(options, None, prefix)
        return ret


GraphParserClassView = Proxy(CouplingTreeAndGraphParser, {"DataType": graph_utils.Graph})
TreeParserClassView = Proxy(CouplingTreeAndGraphParser, {"DataType": tree_utils.Sentence})
