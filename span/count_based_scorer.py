from typing import Iterable, Tuple

import nn
from hrgguru.hrg import CFGRule
from span.hrg_statistics import HRGStatistics


class CountBasedHRGScorer(nn.DynetSaveable):
    def __init__(self, model,
             hrg_statistics,  # type: HRGStatistics
             options):
        super(CountBasedHRGScorer, self).__init__(model)

    @classmethod
    def add_parser_arguments(cls, arg_parser):
        pass

    def get_best_rule(self,
                      span_feature,
                      rules_and_counts,  # type: Iterable[Tuple[CFGRule, int]]
                      gold=None
                      ):
            yield []
            best_rule = sorted(((count, rule) for rule, count in rules_and_counts),
                               key=lambda x: x[0],
                               reverse=True)[0][1]
            yield best_rule, None, best_rule

    def restore_components(self, restored_params):
        pass
