from argparse import ArgumentParser
from pprint import pprint

from graph_utils import Graph, Graph2015
from span.const_tree import ConstTree
from srltagger.data_reader import OneLineTaggedWords
from supertagger.data_reader import Graph2015ForSuperTag
from tree_utils import Sentence

formats = {"tree": Sentence, "graph": Graph, "sdp2015": Graph2015,
           "sdp2015supertag": Graph2015ForSuperTag,
           "span": ConstTree, "srl": OneLineTaggedWords}


def select_best(format_, inputs, key, k=1):
    results = {i: formats[format_].extract_performance(i)
               for i in inputs}
    sorted_results = sorted(((v[key], v["epoch"])
                             for file_name, v in results.items()),
                            reverse=True)
    if k == 1:
        return sorted_results[0]
    else:
        return sorted_results[:k]


def select_best_k(format_, inputs, key, k=1):
    results = {i: formats[format_].extract_performance(i)
               for i in inputs}
    sorted_results = sorted(results.items(),
                            key=lambda x: x[1].get(key, -1),
                            reverse=True)
    return sorted_results[:k]


def main():
    parser = ArgumentParser()
    parser.add_argument("--inputs", nargs="+", dest="inputs", required=True)
    parser.add_argument("--key", dest="key", required=True)
    parser.add_argument("--k", dest="k", type=int, default=1)

    parser.add_argument("--format", dest="format", choices=list(formats.keys()),
                        required=True)
    parser.add_argument("--print-format", dest="print_format",
                        help="Output formatter. Example: \"{UP} {UR} {UF}\"")

    args = parser.parse_args()
    for best_filename, best_dict in select_best_k(
            args.format, args.inputs, args.key, args.k):
        if args.print_format is None:
            print("Epoch {} has best score {}".format(best_dict["epoch"],
                                                      best_dict[args.key]))
        else:
            print(args.print_format.format(**best_dict))


if __name__ == '__main__':
    main()
