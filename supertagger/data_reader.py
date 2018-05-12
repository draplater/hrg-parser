import sys
import traceback

from graph_utils import Graph2015


class Graph2015ForSuperTag(Graph2015):
    @classmethod
    def evaluate_with_external_program(cls, gold_file, output_file, limit=None):
        total_count = sys.float_info.epsilon
        correct_count = 0
        try:
            gold_sents = cls.from_file(gold_file)
            output_sents = cls.from_file(output_file)
            assert len(gold_sents) == len(output_sents)
            for gold_sent, output_sent in zip(gold_sents, output_sents):
                assert len(gold_sent) == len(output_sent)
                for gold_node, output_node in zip(gold_sent[1:], output_sent[1:]):
                    total_count += 1
                    if gold_node.supertag == output_node.supertag:
                        correct_count += 1
            output_msg = "Accuracy: {}".format(correct_count / total_count * 100)
            with open(output_file + ".txt", "w") as f:
                f.write(output_msg + "\n")
            print(output_msg)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            with open(output_file + ".txt", "w") as f:
                f.write(traceback.format_exc() + "\n")

    @classmethod
    def from_file(cls, file_name, use_edge=True):
        ret = super(Graph2015ForSuperTag, cls).from_file(file_name, True)
        if not use_edge:
            for sentence in ret:
                for node in sentence:
                    node.sense = None
        return ret