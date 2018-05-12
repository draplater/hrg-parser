import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training_scheduler import LazyLoadTrainingScheduler


def initialize_dynet(title):
    import dynet_config
    dynet_config.set(random_seed=42, mem="1536,1536,512,512",
                     autobatch=True, requested_gpus=1)
    import dynet
    time.sleep(2)


home = os.path.expanduser("~")
data_dir = f"{home}/Development/large-data/cfg/"
op = {
    "word-threshold": 20,
    "pembedding": 0,
    "wembedding": 150,
    "decoder": "binary",
    "lstmdims": 250,
    "mlp-dims": [250],
    "label-mlp-dims": [250],
    "concurrent-count": 20,
    "extrn": data_dir + "../sskip.100.vectors",
}

outdir_prefix = home + "/work/span/"
trainer = LazyLoadTrainingScheduler("span.model", "SpanParser", initializer=initialize_dynet)

op["train"] = home + "/Development/HRGGuru/deepbank-preprocessed/deepbank1.1-lite.fulllabel.train"
op["dev"] =  home + "/Development/HRGGuru/deepbank-preprocessed/deepbank1.1-lite.fulllabel.dev"
trainer.add_options("deepbank1.1-lite-internal", op, outdir_prefix)

trainer.run_parallel()
