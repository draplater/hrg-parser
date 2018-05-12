import dynet_config
dynet_config.set(random_seed=42, mem="2048,2048,512,512", autobatch=True)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from span.hrg_parser import UdefQParser

home = os.path.expanduser("~")

op = {
    "span-model": home + "/work/span/model-deepbank1.1-lite/model.10",
    "epochs": 20,
    "scorer": "feature",
    "beam-size": 4,
    "hrg-loss-margin": 1,
    "use-graph-embedding": True,
    "deepbank-dir": home + "/Development/deepbank1.1",
    "greedy-at-leaf": True,
}

op.update({
    "derivations": home + "/Development/HRGGuru/deepbank-preprocessed/derivations-deepbank1.1-lite.pickle",
    "grammar": home + "/Development/HRGGuru/deepbank-preprocessed/cfg_hrg_mapping-deepbank1.1-lite.pickle",
    "train": home + "/Development/HRGGuru/deepbank-preprocessed/deepbank1.1-lite.fulllabel.train",
    "dev": home + "/work/span/model-deepbank1.1-lite/deepbank1.1-lite.fulllabel_epoch_10.dev",
    "deepbank-dir": home + "/Development/deepbank1.1"
})

outdir_prefix = home + "/work/hrg/"

trainer = UdefQParser.get_training_scheduler()

trainer.add_options(f"deepbank1.1-lite-beam4-delexicalized7", op, outdir_prefix)

trainer.run_parallel()
