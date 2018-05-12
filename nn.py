from __future__ import unicode_literals

import abc
import gzip
import pickle
from io import open

import os
import six
import sys
from six.moves import range

import dynet as dn
import numpy as np

from vocab_utils import Dictionary

alpha = 1.6732632423543772848170429916717
scale = 1.0507009873554804934193349852946
epsilon = sys.float_info.epsilon


def selu(x):
    """ :type x: dn.Expression
        :rtype: dn.Expression """
    positive = dn.rectify(x)
    positive_indicator = dn.rectify(dn.cdiv(positive, positive + epsilon))
    negative = -dn.rectify(-x)
    exp_negative = dn.exp(negative) - positive_indicator
    exp_negative_minus_alpha = exp_negative * alpha - alpha + positive_indicator * alpha
    # x>0: x=x * scale; x<0: x = (alpha * exp(x) - alpha) * scale
    ret = (positive + exp_negative_minus_alpha) * scale
    return ret


def leaky_relu(x):
    """:type x: dn.Expression
    :rtype: dn.Expression"""
    positive = dn.rectify(x)
    negative = dn.rectify(-x) * -0.01
    ret = positive + negative
    return ret


activations = {'tanh': dn.tanh, 'sigmoid': dn.logistic, 'relu': dn.rectify,
               'tanh3': (lambda x: dn.tanh(dn.cwise_multiply(
                   dn.cwise_multiply(x, x), x))),
               'selu': selu, "leaky-relu": leaky_relu}
trainers = {"adam": dn.AdamTrainer, "sgd": dn.SimpleSGDTrainer,
            "momentum": dn.MomentumSGDTrainer,
            "rmsprop": dn.RMSPropTrainer}

recurrent_builders = {"lstm": dn.VanillaLSTMBuilder, "gru": dn.GRUBuilder}


def recurrent_factory_factory(builder):
    # use closure to hold "builder"
    return lambda model, dims: BiLSTM(model, dims, builder)


recurrents = {name: recurrent_factory_factory(builder)
              for name, builder in recurrent_builders.items()}


# the follow code doesn't work because the builder isn't in closure
# recurrents = {name: (lambda model, dims: BiLSTM(model, dims, builder))
#               for name, builder in recurrent_builders.items()}


def get_optimizer(model, options):
    # backward compatity
    if not hasattr(options, "optimizer"):
        return dn.AdamTrainer(model)

    return trainers[options.optimizer](
        *((model, options.learning_rate)
          if options.learning_rate is not None else (model,)))


@six.add_metaclass(abc.ABCMeta)
class DynetSaveable(object):
    def __init__(self, parent_saveable):
        if isinstance(parent_saveable, dn.Model):
            self.model = parent_saveable
        elif isinstance(parent_saveable, DynetSaveable):
            self.model = parent_saveable.model.add_subcollection()
            parent_saveable.params.append((DynetSaveable, self))
        else:
            raise TypeError

        self.params = []

    def add_parameters(self, dim, init=None, name=None):
        if name is None:
            ret = self.model.add_parameters(dim, init)
        else:
            ret = self.model.add_parameters(dim, init, name)
        self.params.append((dn.Parameters, (dim, None, name)))
        return ret

    def add_lookup_parameters(self, dim, init=None, name=None):
        if name is None:
            ret = self.model.add_lookup_parameters(dim, init)
        else:
            ret = self.model.add_lookup_parameters(dim, init, name)
        self.params.append((dn.LookupParameters, (dim, init, name)))
        return ret

    def add_lstm_builder(self, builder, layers, input_dim, hidden_dim):
        ret = builder(layers, input_dim, hidden_dim, self.model)
        self.params.append((builder, (layers, input_dim, hidden_dim)))
        return ret

    def get_picklable_obj(self):
        """ Faster save/load interface"""
        return (self,
                [i.as_array() for i in self.model.parameters_list()],
                [i.as_array() for i in self.model.lookup_parameters_list()]
                )

    @classmethod
    def from_picklable_obj(cls, obj, model, is_root=True):
        self, param_list, lookup_param_list = obj
        self.model = model
        restored_params = []
        for param_type, args in self.params:
            if param_type is dn.Parameters:
                restored_params.append(model.parameters_from_numpy(param_list.pop(0)))
            elif param_type is dn.LookupParameters:
                restored_params.append(model.lookup_parameters_from_numpy(lookup_param_list.pop(0)))
            elif param_type is DynetSaveable:
                sub_component = args
                cls.from_picklable_obj((sub_component, param_list, lookup_param_list),
                                       model.add_subcollection(), False)
                restored_params.append(sub_component)
            elif issubclass(param_type, dn.VanillaLSTMBuilder.__bases__[0]):
                builder = param_type(*(args + (model,)))
                for rnn_param in builder.param_collection().parameters_list():
                    rnn_param.set_value(param_list.pop(0))
                for rnn_param in builder.param_collection().lookup_parameters_list():
                    rnn_param.set_value(lookup_param_list.pop(0))
                restored_params.append(builder)
            else:
                raise TypeError(param_type)

        if is_root and (param_list or lookup_param_list):
            raise TypeError("Unmatched model!")
        self.restore_components(restored_params)
        return self

    @property
    def spec(self):
        return self

    def param_collection(self):
        return self.model

    @staticmethod
    def from_spec(spec, model):
        spec.model = model
        restored_params = []
        for param_type, args in spec.params:
            if param_type is dn.Parameters:
                args = tuple(i for i in args if i is not None)
                restored_params.append(spec.model.add_parameters(*args))
            elif param_type is dn.LookupParameters:
                args = tuple(i for i in args if i is not None)
                restored_params.append(spec.model.add_lookup_parameters(*args))
            elif issubclass(param_type, dn.VanillaLSTMBuilder.__bases__[0]):
                restored_params.append(param_type(*(args + (spec.model,))))
            elif param_type is DynetSaveable:
                sub_component = args
                sub_component.__class__.from_spec(sub_component, spec.model.add_subcollection())
                restored_params.append(sub_component)
            else:
                raise TypeError(param_type)
        spec.restore_components(restored_params)
        return spec

    def __getstate__(self):
        result = {k: v for k, v in self.__dict__.items()
                  if k != "model" and
                  not isinstance(v, dn.LookupParameters) and
                  not isinstance(v, dn.Parameters)
                  }
        return result

    @abc.abstractmethod
    def restore_components(self, restored_params):
        pass


model_formats = ["dynet", "pickle", "pickle-gzip"]


def detect_saved_model_type(prefix):
    if os.path.exists(prefix):
        return "pickle"
    elif os.path.exists(prefix + ".data"):
        return "dynet"
    elif os.path.exists(prefix + ".gz"):
        return "pickle-gzip"
    else:
        raise FileNotFoundError("Model {} not found!".format(prefix))


def model_load_helper(mode, prefix, model):
    """
    Save/Load helper for backward compatibly.
    It save/load options and model.
    """
    if mode is None:
        mode = detect_saved_model_type(prefix)

    if mode == "dynet":
        with open(prefix + ".options", "rb") as f:
            options = pickle.load(f)
        return options, dn.load(prefix, model)[0]
    elif mode == "pickle":
        with open(prefix, "rb") as f:
            options, picklable = pickle.load(f)
        return options, DynetSaveable.from_picklable_obj(picklable, model)
    elif mode == "pickle-gzip":
        with open(prefix + ".gz", "rb") as f:
            options, picklable = pickle.load(f)
        return options, DynetSaveable.from_picklable_obj(picklable, model)
    else:
        raise TypeError("Invalid model format.")


def model_save_helper(mode, prefix, savable, options):
    if mode == "dynet":
        # noinspection PyArgumentList
        dn.save(prefix, [savable])
        with open(prefix + ".options", "wb") as f:
            pickle.dump(options, f)
    elif mode == "pickle":
        picklable = savable.get_picklable_obj()
        with open(prefix, "wb") as f:
            pickle.dump((options, picklable), f)
    elif mode == "pickle-gzip":
        picklable = savable.get_picklable_obj()
        with gzip.open(prefix, "wb") as f:
            pickle.dump((options, picklable), f)
    else:
        raise TypeError("Invalid model format.")


class DenseLayers(DynetSaveable):
    def __init__(self, model, dims, activation, use_bias=None):
        """
        :type model: Union[dn.Model, Saveable]
        :type dims: [int]
        :param model:
        :param dims:
        """
        if use_bias is None:
            self.use_bias = [True] * (len(dims) - 1)
        else:
            assert len(use_bias) == len(dims) - 1
            self.use_bias = use_bias

        super(DenseLayers, self).__init__(model)
        self.activation = activation
        self.layer_count = len(dims) - 1
        self.weights = []  # type: [dn.Expression]
        self.biases = []  # type: [dn.Expression]
        for i in range(len(dims) - 1):
            input_dim = dims[i]
            output_dim = dims[i + 1]
            self.weights.append(self.add_parameters((output_dim, input_dim)))
            if self.use_bias[i]:
                if isinstance(self.use_bias[i], dn.PyInitializer):
                    bias = self.add_parameters(output_dim, init=self.use_bias[i])
                else:
                    bias = self.add_parameters(output_dim)
            else:
                bias = None
            self.biases.append(bias)

    def __call__(self, input_tensor):
        tensor = input_tensor
        for idx, w_d in enumerate(zip(self.weights, self.biases)):
            weight, bias = w_d
            if bias is not None:
                tensor = dn.affine_transform([bias.expr(), weight.expr(), tensor])
            else:
                tensor = weight.expr() * tensor
            if idx != len(self.weights) - 1:
                tensor = self.activation(tensor)
        return tensor

    def __getstate__(self):
        state_members = ["layer_count", "use_bias", "activation", "params"]
        return {i: getattr(self, i) for i in state_members}

    def restore_components(self, components):
        assert len(components) == self.layer_count + sum(i for i in self.use_bias)
        self.weights = []
        self.biases = []
        pointer = 0
        for i in range(self.layer_count):
            self.weights.append(components[pointer])
            pointer += 1
            if self.use_bias[i]:
                self.biases.append(components[pointer])
                pointer += 1
            else:
                self.biases.append(None)
        assert pointer == len(components)


class BiLSTM(DynetSaveable):
    def __init__(self, model, dims, builder=dn.VanillaLSTMBuilder):
        super(BiLSTM, self).__init__(model)
        self.dims = dims
        self.forward_cells = []
        self.backward_cells = []
        for i in range(len(dims) - 1):
            input_dim = dims[i]
            output_dim = dims[i + 1]
            self.forward_cells.append(self.add_lstm_builder(builder, 1, input_dim, output_dim / 2))
            self.backward_cells.append(self.add_lstm_builder(builder, 1, input_dim, output_dim / 2))

    def __call__(self, word_embeddings):
        input_tensors = word_embeddings
        for layer, (forward_cell, backward_cell) in enumerate(
                zip(self.forward_cells, self.backward_cells), 1):
            forward_results = forward_cell.initial_state().transduce(input_tensors)

            input_tensors.reverse()
            backward_results = backward_cell.initial_state().transduce(input_tensors)
            backward_results.reverse()

            input_tensors = [dn.concatenate([forward, backward])
                             for forward, backward in zip(forward_results,
                                                          backward_results)]
        return input_tensors

    def get_layers_output(self, word_embeddings, layers):
        input_tensors = word_embeddings
        output_tensors = [input_tensors]
        for layer, (forward_cell, backward_cell) in enumerate(
                zip(self.forward_cells, self.backward_cells), 1):
            forward_results = forward_cell.initial_state().transduce(input_tensors)

            input_tensors.reverse()
            backward_results = backward_cell.initial_state().transduce(input_tensors)
            backward_results.reverse()

            input_tensors = [dn.concatenate([forward, backward])
                             for forward, backward in zip(forward_results,
                                                          backward_results)]
            output_tensors.append(input_tensors)
        return [output_tensors[layer] for layer in layers]

    def __getstate__(self):
        return {"dims": self.dims, "params": self.params}

    def set_dropout(self, dropout):
        for cell in self.forward_cells + self.backward_cells:
            cell.set_dropout(dropout)

    def disable_dropout(self):
        for cell in self.forward_cells + self.backward_cells:
            cell.disable_dropout()

    def restore_components(self, components):
        self.forward_cells = components[0::2]
        self.backward_cells = components[1::2]


class HighWayRecurrentWrapper(DynetSaveable):
    def __init__(self, model, dims, highway_count, builder=dn.VanillaLSTMBuilder):
        super(HighWayRecurrentWrapper, self).__init__(model)
        self.dims = dims
        self.birnn_layers = []
        self.highway_i_factors = [None]
        self.highway_o_factors = [None]
        self.highway_biases = [None]
        for i in range(highway_count):
            self.birnn_layers.append(BiLSTM(self, dims, builder))
            if i != 0:
                self.highway_i_factors.append(self.add_parameters((dims[-1], dims[-1])))
                self.highway_o_factors.append(self.add_parameters((dims[-1], dims[-1])))
                self.highway_biases.append(self.add_parameters((dims[-1],), dn.ConstInitializer(-3.0)))

    def __call__(self, word_embeddings):
        highway_memories = word_embeddings
        for birnn_layer, highway_i_factor, \
            highway_o_factor, highway_bias in zip(self.birnn_layers,
                                                  self.highway_i_factors, self.highway_o_factors,
                                                  self.highway_biases):
            output_tensors = birnn_layer(highway_memories)

            if highway_memories is word_embeddings:
                highway_memories = output_tensors
            else:
                new_highway_memories = []
                for memory_vector, output_vector in zip(highway_memories, output_tensors):
                    highway_bias_expr = highway_bias.expr()
                    highway_i_factor_expr = highway_i_factor.expr()
                    highway_o_factor_expr = highway_o_factor.expr()
                    transform_rate = dn.logistic(
                        dn.affine_transform([highway_bias_expr, highway_i_factor_expr,
                                             memory_vector, highway_o_factor_expr, output_vector]))

                    keep_rate = 1 - transform_rate
                    new_highway_memories.append(
                        dn.cmult(keep_rate, memory_vector) +
                        dn.cmult(transform_rate, output_vector))
                highway_memories = new_highway_memories
        return highway_memories

    def __getstate__(self):
        return {"dims": self.dims, "params": self.params}

    def set_dropout(self, dropout):
        for cell in self.birnn_layers:
            cell.set_dropout(dropout)

    def disable_dropout(self):
        for cell in self.birnn_layers:
            cell.disable_dropout()

    def restore_components(self, components):
        self.forward_cells = components[0::2]
        self.backward_cells = components[1::2]


class BiLinear(DynetSaveable):
    def __init__(self, model, input_dim, output_dim):
        super(BiLinear, self).__init__(model)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.w1 = self.add_parameters((output_dim, input_dim))
        self.w2 = self.add_parameters((output_dim, input_dim))
        self.bias = self.add_parameters(output_dim)

    def __call__(self, input_1, input_2):
        return self.w1.expr() * input_1 + self.w2.expr() * input_2 + self.bias.expr()

    def restore_components(self, components):
        self.w1, self.w2, self.bias = components


class Biaffine(object):
    def __init__(self, model, input_dim, activation):
        self.input_dim = input_dim
        self.activation = activation
        self.w1 = model.add_parameters((input_dim, input_dim))
        self.w2 = model.add_parameters(input_dim)

    def __call__(self, head, dep):
        return head * self.w1.expr() * dep + head * self.w2.expr()


def read_embedding(embedding_filename, encoding):
    if embedding_filename.endswith(".gz"):
        external_embedding_fp = gzip.open(embedding_filename, 'rb')
    else:
        external_embedding_fp = open(embedding_filename, 'rb')

    def embedding_gen():
        for line in external_embedding_fp:
            fields = line.decode(encoding).strip().split(' ')
            if len(fields) <= 2:
                continue
            token = fields[0]
            vector = [float(i) for i in fields[1:]]
            yield token, vector

    external_embedding = dict(embedding_gen())
    external_embedding_fp.close()
    return external_embedding


def get_external_embedding(model, embedding_filename, encoding="utf-8",
                           extra=("*EMPTY*", "*PAD*", "*INITIAL*")):
    external_embedding = read_embedding(embedding_filename, encoding)
    dim = len(next(iter(six.itervalues(external_embedding))))

    extrn_dict = {word: i for i, word in enumerate(external_embedding, len(extra))}

    elookup = model.add_lookup_parameters((len(external_embedding) + len(extra), dim))
    for word, i in six.iteritems(extrn_dict):
        embedding = external_embedding[word]
        assert len(embedding) == dim
        elookup.init_row(i, external_embedding[word])

    for idx, word in enumerate(extra):
        extrn_dict[word] = idx

    return extrn_dict, elookup, dim


class EmbeddingBase(DynetSaveable):
    def __call__(self, word, alternative=None, const=False):
        idx = self.vocab.get(word, 0)
        if idx == 0 and alternative is not None:
            for word_i in alternative:
                idx = self.vocab.get(word_i, 0)
                if idx != 0:
                    break
        return self.lookup[idx] if not const else dn.transpose(dn.const_parameter(self.lookup))[idx]

    def restore_components(self, components):
        self.lookup, = components


class Embedding(EmbeddingBase):
    def __init__(self, model, vocab, dim, extra=("*EMPTY*", "*PAD*", "*INITIAL*"), init=None):
        super(Embedding, self).__init__(model)
        self.vocab = {word: idx for idx, word in enumerate(vocab, len(extra))}
        for idx, word in enumerate(extra):
            self.vocab[word] = idx
        self.lookup = self.add_lookup_parameters((len(vocab) + len(extra), dim), init)
        self.dim = dim


class EmbeddingFromDictionary(EmbeddingBase):
    def __init__(self,
                 model,
                 dictionary,  # type: Dictionary
                 dim,  # type: int
                 init=None,
                 external_init=None,
                 external_encoding="utf-8"
                 ):
        super(EmbeddingFromDictionary, self).__init__(model)
        self.vocab = dictionary.word_to_int
        self.lookup = self.add_lookup_parameters((len(self.vocab), dim), init)
        self.dim = dim
        if external_init is not None:
            ext_embedding = read_embedding(external_init, external_encoding)
            for word, idx in self.vocab.items():
                ebd = ext_embedding.get(word)
                if ebd is not None:
                    self.lookup.init_row(idx, ebd)


class ExternalEmbedding(EmbeddingBase):
    def __init__(self, model, embedding_filename, encoding="utf-8", extra=("*EMPTY*", "*PAD*", "*INITIAL*")):
        super(ExternalEmbedding, self).__init__(model)
        self.vocab, self.lookup, self.dim = get_external_embedding(
            self, embedding_filename, encoding, extra)


class Container(DynetSaveable):
    def __init__(self, model):
        super(Container, self).__init__(model)
        self.components = []

    def __getstate__(self):
        ret = super(Container, self).__getstate__()
        del ret["components"]
        return ret

    def restore_components(self, restored_params):
        self.components = restored_params


class Merge(DynetSaveable):
    def __init__(self, blocks, mode=dn.esum):
        super(Merge, self).__init__(model)  # TODO: ???
        self.blocks = blocks
        self.mode = mode

    def __call__(self, *args, **kwargs):
        outputs = [i(*args, **kwargs) for i in self.blocks]
        if isinstance(outputs[0], list):
            return [self.mode(list(i)) for i in zip(*outputs)]
        return self.mode(outputs)

    def __getstate__(self):
        return {"mode": self.mode}

    def get_components(self):
        return self.blocks

    def restore_components(self, components):
        self.blocks = components


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc


def test_activation():
    import matplotlib.pyplot as plt
    x = np.arange(-5, 5, 0.1)
    y = [selu(dn.scalarInput(i)).value() for i in x]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    test_activation()
