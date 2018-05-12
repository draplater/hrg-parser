from abc import ABCMeta
import six

import nn
from parser_base import DependencyParserBase


@six.add_metaclass(ABCMeta)
class DynetParserBase(DependencyParserBase):
    default_batch_size = 32
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        super(DynetParserBase, cls).add_parser_arguments(arg_parser)
        group = arg_parser.add_argument_group(DependencyParserBase.__name__)
        group.add_argument("--optimizer", type=str, dest="optimizer", default="adam", choices=nn.trainers.keys())

    @classmethod
    def add_common_arguments(cls, arg_parser):
        super(DynetParserBase, cls).add_common_arguments(arg_parser)
        group = arg_parser.add_argument_group(cls.__name__ + "(common)")
        group.add_argument("--batch-size", type=int, dest="batch_size", default=cls.default_batch_size)
