import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dynet_config
dynet_config.set(random_seed=42, mem=4096, autobatch=True, requested_gpus=0)
import dynet

from span.leaftagger import POSTagParser

home = os.path.expanduser("~")
data_dir = f"{home}/Development/large-data/cfg/"

op = {
    "lstmlayers": 2,
    "disable-viterbi": True,
    "word-threshold": 50,
    "max-save": 50,
    "wembedding": 200,
    "lstm-dropout": 0.35,
    "extrn": data_dir + "../sskip.100.vectors",
}

outdir_prefix = home + "/work/leaftagger/"
scheduler = POSTagParser.get_training_scheduler()

name = "lite"
op["train"] = home + f"/Development/HRGGuru/deepbank-preprocessed/deepbank1.1-{name}.fulllabel.train"
op["dev"] =  home + f"/Development/HRGGuru/deepbank-preprocessed/deepbank1.1-{name}.fulllabel.dev"
scheduler.add_options(f"deepbank1.1-{name}", op, outdir_prefix)

scheduler.run_parallel()
