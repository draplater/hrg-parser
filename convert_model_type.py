from argparse import ArgumentParser
import dynet as dn
import nn


def main():
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--input-type", dest="input_type", choices=nn.model_formats, default=None)
    parser.add_argument("--output-type", dest="output_type", choices=nn.model_formats, default="pickle")
    args = parser.parse_args()
    m = dn.Model()
    options, savable = nn.model_load_helper(args.input_type, args.input, m)
    nn.model_save_helper(args.output_type, args.output, savable, options)


if __name__ == '__main__':
    main()