import importlib
from argparse import ArgumentParser

import sys

# lazy load modules
from training_scheduler import parse_cmd_multistage

parsers = {
    "span": ("span.model", "SpanParser"),
    "hrg": ("span.hrg_parser", "UdefQParser")
}


if __name__ == '__main__':
    parser_name_arg_parser = ArgumentParser(sys.argv[0])
    parser_name_arg_parser.add_argument("parser",
                                        help="Parser you want to use",
                                        choices=parsers.keys())
    parser_name_args = parser_name_arg_parser.parse_args(sys.argv[1:2])
    parser_name = parser_name_args.parser

    module_name, class_name = parsers[parser_name]
    ParserClass = getattr(importlib.import_module(module_name), class_name)
    args = parse_cmd_multistage(ParserClass, sys.argv[2:])
    args.func(args)
