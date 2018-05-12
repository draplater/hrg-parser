import nn


class POSTagClassification(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        """:type arg_parser: argparse.ArgumentParser"""
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--tagger-mlp-dims", dest="tagger_mlp_dims", type=int, nargs="*",
                           help="MLP Layers", default=[100])
        group.add_argument("--activation", dest="activation", type=str, help="Activation", default="tanh",
                           choices=nn.activations.keys())

    def __init__(self, model, tag_dict, options, lstm_dims=None):
        """:type tag_dict: vocab_utils.Dictionary"""
        super(POSTagClassification, self).__init__(model)
        mlp_dims = [(lstm_dims or options.lstm_dims) * 2] + \
                   [len(tag_dict) if i == -1 else i
                    for i in options.tagger_mlp_dims] + \
                   [len(tag_dict)]
        self.mlp = nn.DenseLayers(self, mlp_dims, nn.activations[options.activation])
        self.options = options

    def __call__(self, sentence_embeddings):
        return [self.mlp(i) for idx, i in enumerate(sentence_embeddings)]

    def restore_components(self, components):
        self.mlp, = components

