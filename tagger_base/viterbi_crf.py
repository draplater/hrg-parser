from operator import itemgetter

import dynet as dn
import numpy as np
from os.path import dirname

import nn

import pyximport

pyximport.install(build_dir=dirname(__file__) + "/.cache/")
from tagger_base.cviterbi import viterbi


class ViterbiDecoder(nn.DynetSaveable):
    @classmethod
    def add_parser_arguments(cls, arg_parser):
        group = arg_parser.add_argument_group(cls.__name__)
        group.add_argument("--beam-size", dest="beam_size", type=int, default=None)

    def __init__(self, model, tag_dict, options):
        super(ViterbiDecoder, self).__init__(model)
        self.index_to_tag = tag_dict.int_to_word
        self.tag_to_index = tag_dict.word_to_int
        self.tag_count = len(self.index_to_tag)
        self.virtual_start = self.tag_count
        self.virtual_end = self.tag_count + 1

        # the two matrices are stored in transposed style: x[next_tag, previous_tag]
        self.transition_params = self.add_parameters((self.tag_count + 2,
                                                      self.tag_count + 2),
                                                     dn.ConstInitializer(0.0))
        self.valid_transition_map = np.ones((self.tag_count + 2, self.tag_count + 2),
                                            dtype=np.int8)
        self.beam_size = options.beam_size

    def __call__(self, tag_scores):
        n = tag_scores.shape[0]
        if n == 0:
            return []
        transition_scores = self.transition_params.expr().value()
        return viterbi(tag_scores, transition_scores,
                       self.valid_transition_map, self.beam_size)

    def python_viterbi(self, prob_matrix, exprs):
        n = prob_matrix.shape[0]
        transition_scores = self.transition_params.expr().value()
        if n == 0:
            return []
        dp = np.zeros((n, self.tag_count))
        backpoint = np.full((n, self.tag_count), -1)
        for tag in range(self.tag_count):
            if self.valid_transition_map[tag][self.virtual_start]:
                dp[0, tag] = prob_matrix[0, tag] + transition_scores[tag, self.virtual_start]
            else:
                dp[0, tag] = -float("inf")

        for i in range(1, n):
            for tag_i in range(self.tag_count):
                max_prob = -float("inf")
                max_tag_previous = -2
                for tag_previous in range(self.tag_count):
                    if self.valid_transition_map[tag_i][tag_previous]:
                        continue
                    new_value = prob_matrix[i, tag_i] + \
                                transition_scores[tag_i, tag_previous] + dp[i - 1, tag_previous]
                    if new_value > max_prob:
                        max_prob = new_value
                        max_tag_previous = tag_previous
                dp[i, tag_i] = max_prob
                backpoint[i, tag_i] = max_tag_previous

        max_last_label, max_last_score = max(((
            i,
            dp[n - 1, i] + transition_scores[self.virtual_end, i])
            for i in range(self.tag_count)
            if self.valid_transition_map[self.virtual_end][i]),
            key=itemgetter(1))

        path = [max_last_label]
        while True:
            i = n - len(path)
            tag_back = path[-1]
            tag_previous = int(backpoint[i, tag_back])
            if tag_previous != -1:
                path.append(tag_previous)
            else:
                break
        return path[::-1]

    def crf_forward(self, observations):
        init_alphas = [-1e10] * (self.tag_count + 2)
        init_alphas[self.virtual_start] = 0
        for_expr = dn.inputVector(init_alphas)
        transition_param_expr = self.transition_params.expr()

        for obs in observations:
            obs_broadcast_2d = dn.transpose(dn.concatenate([obs] * (self.tag_count + 2), 1))
            last_for_expr_broadcast_2d = dn.concatenate([for_expr] * (self.tag_count + 2), 1)
            total = last_for_expr_broadcast_2d + \
                    transition_param_expr + obs_broadcast_2d
            for_expr = dn.logsumexp_dim(total, 0)
        terminal_expr = for_expr + transition_param_expr[self.virtual_end]
        alpha = dn.logsumexp_dim(terminal_expr, 0)
        return alpha

    def get_expr(self, obs_exprs, tags):
        if len(tags) == 0:
            return dn.scalarInput(0.0)

        transition_param_expr = self.transition_params.expr()
        ret = transition_param_expr[tags[0]][self.virtual_start]
        for idx in range(len(tags)):
            this_tag = int(tags[idx])
            ret += obs_exprs[idx][this_tag]
            if idx > 0:
                previous_tag = int(tags[idx - 1])
                ret += self.transition_params.expr()[this_tag][previous_tag]
        ret += transition_param_expr[self.virtual_end][tags[-1]]
        return ret

    def restore_components(self, restored_params):
        self.transition_params, = restored_params


def greedy_decode(scores):
    return np.argmax(scores, 1).tolist()
