import re
from argparse import ArgumentParser

import os

from select_best import select_best_k, formats


def main():
    parser = ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("key")
    parser.add_argument("--k", dest="k", type=int, default=1)
    parser.add_argument("--dev-keyword", dest="dev_keyword", default="dev")
    parser.add_argument("--format", dest="format", choices=list(formats.keys()),
                        required=True)
    parser.add_argument("--print-format", dest="print_format",
                        help="Output formatter. Example: \"{UP} {UR} {UF}\"")
    parser.add_argument("--delete-poor-model", "-x",
                        action="store_true", dest="delete_poor_model", default=False)
    parser.add_argument("--delete-poor-output", "-y",
                        action="store_true", dest="delete_poor_output", default=False)

    args = parser.parse_args()
    score_files = []
    for j in os.listdir(args.input_dir):
        full_path = os.path.join(args.input_dir, j)
        # if original extension is ".txt", select ".txt.txt" instead
        if args.dev_keyword in j and full_path.endswith(".txt") \
                and not os.path.exists(full_path + ".txt"):
            score_files.append(full_path)

    best_k = select_best_k(
        args.format, score_files, args.key, args.k)

    if not best_k:
        exit(0)

    for best_filename, best_dict in best_k:
        best_dict["name"] = args.input_dir
        if args.print_format is None:
            print("Epoch {} has best score {}".format(best_dict["epoch"],
                                                      best_dict.get(args.key, -1)))
        else:
            print(args.print_format.format(**best_dict))

    if args.delete_poor_model:
        keep_model_files = {"model.{}".format(v[1]["epoch"]) for v in best_k}
        for j in os.listdir(args.input_dir):
            if j.startswith("model.") and j not in keep_model_files:
                os.remove(os.path.join(args.input_dir, j))

    if args.delete_poor_output:
        if len(best_k) > 0:
            prefix = os.path.basename(re.search(r"^(.*_epoch_)", best_k[0][0]).group(1))
            keep_prefix = {prefix + str(v[1]["epoch"]) for v in best_k}
            for j in os.listdir(args.input_dir):
                full_path = os.path.join(args.input_dir, j)
                if j.startswith(prefix) and not full_path in score_files and \
                        not any(j.startswith(k) for k in keep_prefix):
                    os.remove(full_path)


if __name__ == '__main__':
    main()
