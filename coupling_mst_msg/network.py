from edge_eval_network import EdgeEvaluationNetwork
from nn import Merge, BiLSTM, DynetSaveable


class EdgeDoubleEvaluationNetwork(DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        EdgeEvaluationNetwork.add_parser_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--not-share-embedding", action="store_false", dest="share_embedding", default=True)
        group.add_argument("--not-share-lstm", action="store_false", dest="share_lstm", default=True)

    def __init__(self, model, statistics, options):
        super(EdgeDoubleEvaluationNetwork, self).__init__(model)
        self.share_embedding = options.share_embedding
        self.share_lstm = options.share_lstm
        self.graph_network = EdgeEvaluationNetwork(model, statistics[0], options)
        self.tree_network = EdgeEvaluationNetwork(model, statistics[1], options)
        self.set_share_parameter(model)

    def set_share_parameter(self, model):
        if self.share_embedding:
            self.graph_network.word_embedding = self.tree_network.word_embedding
            self.graph_network.pos_embedding = self.tree_network.pos_embedding
            if hasattr(self.graph_network, "ext_embedding"):
                self.graph_network.ext_embedding = self.tree_network.ext_embedding
        if self.share_lstm:
            share_rnn = BiLSTM(model, self.graph_network.rnn.dims)
            self.graph_network.rnn = Merge([self.graph_network.rnn, share_rnn])
            self.tree_network.rnn = Merge([self.tree_network.rnn, share_rnn])

    def get_components(self):
        return self.tree_network, self.graph_network

    def restore_components(self, components):
        self.tree_network, self.graph_network = components
        self.set_share_parameter() # TODO: restore share LSTM
