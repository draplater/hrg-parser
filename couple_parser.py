from coupling_mst_msg.model import CouplingTreeAndGraphTrainer

if __name__ == '__main__':
    args = CouplingTreeAndGraphTrainer.get_arg_parser().parse_args()
    args.func(args)
