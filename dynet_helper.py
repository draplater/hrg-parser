import sys

try:
    assert '--dynet-viz' not in sys.argv and \
           '--dynet-gpu' not in sys.argv and \
           not ('--dynet-gpus' in sys.argv or '--dynet-gpu-ids' in sys.argv)
    from _dynet import *
except AssertionError:
    if '--dynet-viz' in sys.argv:
        sys.argv.remove('--dynet-viz')
        from dynet_viz import *
    elif '--dynet-gpu' in sys.argv:  # the python gpu switch.
        sys.argv.remove('--dynet-gpu')
        def print_graphviz(**kwarge):
            print("Run with --dynet-viz to get the visualization behavior.")
        from _gdynet import *
    elif '--dynet-gpus' in sys.argv or '--dynet-gpu-ids' in sys.argv: # but using the c++ gpu switches suffices to trigger gpu.
        def print_graphviz(**kwarge):
            print("Run with --dynet-viz to get the visualization behavior.")
        from _gdynet import *

__version__ = 2.0

params = DynetParams()
params.set_random_seed(42)
params.set_weight_decay(1e-6)
init_from_params(params)
