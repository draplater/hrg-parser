# encoding: utf-8
from __future__ import unicode_literals, division

import tempfile
from collections import namedtuple

import os
from multiprocessing.pool import Pool

import numpy as np
import re
import six
import attr

from vocab_utils import Dictionary
from common_utils import smart_open
from span.pegre_main import Peg, nonterminal, sequence, regex, literal, choice, one_or_more, Ignore
from tree_utils import Sentence, SentenceNode


class ConstTreeParserError(Exception):
    def __init__(self, message):
        self.value = message

    def __str__(self):
        return self.value


@six.python_2_unicode_compatible
class Lexicon(object):
    def __init__(self, string, span=None):
        self.string = string
        self.span = span

    def __str__(self):
        return u"Lexicon <{}>".format(self.string)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.string) + 2

    def __eq__(self, other):
        return self.string == other.string


def make_parser():
    TREE = nonterminal('TREE')
    LHS = regex(r"[^\s]+")
    LEAF = regex(r'[^()]+')
    Spacing = regex(r'\s+', value=Ignore)
    return Peg(
        grammar=dict(
            start=TREE,
            TREE=choice(
                sequence(literal("("), LHS, Spacing, LEAF, literal(")"),
                         value=lambda x: ConstTree.make_leaf_node(x[1], x[2])),
                sequence(literal("("), LHS, Spacing,
                         one_or_more(TREE, delimiter=Spacing), literal(")"),
                         value=lambda x: ConstTree.make_internal_node(x[1], x[2]))
            )
        )
    )


@six.python_2_unicode_compatible
class ConstTree(object):
    """
    c-structure of LFG.
    """

    tree_literal_parser = make_parser()

    def __init__(self, tag, span=None, extra_info=None):
        self.children = []
        self.tag = tag
        self.span = span
        self.extra = extra_info or {}

    def __hash__(self):
        return id(self)

    def __str__(self):
        """
        print tree in the console
        :return: tree
        """
        child_string = " + ".join(i.string if isinstance(i, Lexicon) else i.tag
                                  for i in self.children)
        return "{} -> {}".format(self.tag, child_string)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self.tag != other.tag:
            return False
        if len(self.children) != len(other.children):
            return False
        for i, j in zip(self.children, other.children):
            if i != j:
                return False
        return True

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            for i in self.children:
                if isinstance(i, ConstTree) and i.tag.upper() == item.upper():
                    return i
        if isinstance(item, int):
            return self.children[item]
        raise KeyError

    @classmethod
    def make_leaf_node(cls, tag, lexicon):
        ret = cls(tag)
        ret.children.append(Lexicon(lexicon))
        return ret

    @classmethod
    def make_internal_node(cls, tag, sub_trees):
        ret = cls(tag)
        ret.children.extend(sub_trees)
        return ret

    @classmethod
    def from_string_2(cls, s):
        """
        construct ConstTree from parenthesis representation
        :param s: unicode string of parenthesis representation
        :return: ConstTree root
        :rtype: ConstTree
        """
        ret = cls.tree_literal_parser.parse(s)
        ret.populate_spans_terminal()
        ret.populate_spans_leaf()
        return ret

    @classmethod
    def from_string(cls, s):
        tokens = s.replace("(", " ( ").replace(")", " ) ").split()

        def read_tree(index):
            """:return: (tree, new_index) """
            assert index < len(tokens) and tokens[index] == "("
            #  read lpar
            index += 1
            paren_count = 1

            # read label
            label = tokens[index]
            index += 1
            tree = cls(label)

            while True:
                if tokens[index] == "(":
                    children, index = read_tree(index)
                    tree.children.append(children)
                elif tokens[index] == ")":
                    index += 1
                    break
                else:
                    word = tokens[index]
                    index += 1
                    tree.children.append(Lexicon(word))
            return tree, index

        tree, index = read_tree(0)
        assert index == len(tokens)
        tree.populate_spans_terminal()
        tree.populate_spans_leaf()
        return tree

    def populate_spans_terminal(self):
        for idx, i in enumerate(self.generate_words()):
            i.span = (idx, idx + 1)

    def populate_spans_leaf(self):
        for i in self.children:
            if isinstance(i, ConstTree):
                i.populate_spans_leaf()
        self.span = (self.children[0].span[0], self.children[-1].span[1])

    def generate_words(self):
        for i in self.children:
            if isinstance(i, ConstTree):
                for j in i.generate_words():
                    yield j

        for i in self.children:
            if isinstance(i, Lexicon):
                yield i

    def generate_preterminals(self):
        for i in self.children:
            if isinstance(i, ConstTree):
                for j in i.generate_preterminals():
                    yield j

        for i in self.children:
            if isinstance(i, Lexicon):
                yield self

    def generate_word_and_postag(self):
        if len(self.children) == 1 and isinstance(self.children[0], Lexicon):
            yield (self.children[0].string, self.tag)
        else:
            for i in self.children:
                for j in i.generate_word_and_postag():
                    yield j

    def generate_rules(self):
        for i in self.children:
            if isinstance(i, ConstTree):
                for j in i.generate_rules():
                    yield j
        yield self

    def root_first(self):
        yield self
        for i in self.children:
            if isinstance(i, ConstTree):
                for j in i.root_first():
                    yield j

    def generate_spans(self):
        for i in self.children:
            if isinstance(i, ConstTree):
                for j in i.generate_spans():
                    yield j
            else:
                assert isinstance(i, Lexicon)
                yield i.span
        yield self.span

    def expanded_unary_chain(self):
        tags = self.tag.split("+++")
        root_node = last_node = ConstTree(tags[0], self.span)
        for tag in tags[1:]:
            current_node = ConstTree(tag, self.span)
            last_node.children = [current_node]
            last_node = current_node

        if isinstance(self.children[0], Lexicon):
            last_node.children = list(self.children)
        else:
            last_node.children = list(i.expanded_unary_chain()
                                      for i in self.children)
        return root_node

    def condensed_unary_chain(self):
        if len(self.children) == 1:
            if isinstance(self.children[0], Lexicon):
                ret = ConstTree(self.tag, self.span)
                ret.children = list(self.children)
                return ret
            else:
                assert isinstance(self.children[0], ConstTree)
                new_tag = self.tag + "+++" + self.children[0].tag
                tail = self.children[0]

                while len(tail.children) == 1 and isinstance(tail.children[0], ConstTree):
                    tail = tail.children[0]
                    new_tag += "+++" + tail.tag

                ret = ConstTree(new_tag, self.span)
                if len(tail.children) == 1:
                    ret.children = list(tail.children)
                else:
                    ret.children = list(i.condensed_unary_chain()
                                        for i in tail.children)
                return ret
        else:
            ret = ConstTree(self.tag, self.span)
            ret.children = list(i.condensed_unary_chain()
                                for i in self.children)
            return ret

    def generate_scoreable_spans(self):
        if self.children and isinstance(self.children[0], ConstTree):
            for j in self.children[0].generate_scoreable_spans():
                yield j
            for i in range(1, len(self.children) - 1):
                for j in self.children[i].generate_scoreable_spans():
                    yield j
                yield (self.children[0].span[0], self.children[i].span[1], "___EMPTY___")
            for j in self.children[-1].generate_scoreable_spans():
                yield j
        yield self.span + (self.tag,)

    def to_parathesis(self, suffix="\n"):
        return "({} {}){}".format(self.tag, " ".join([i.to_parathesis("") if isinstance(i, ConstTree)
                                                      else i.string for i in self.children]),
                                  suffix)

    def to_string(self):
        return "\n".join("# {}: {}".format(k, v) for k, v in self.extra.items()) + \
               ("\n" if self.extra else "") + \
               self.to_parathesis()

    @classmethod
    def from_file(cls, file_name, use_edge=None, limit=float("inf")):
        result = []
        with smart_open(file_name) as f:
            extra_info = {}
            count = 0
            for line in f:
                if count >= limit:
                    break
                line_s = line.strip()
                if not line_s:
                    continue
                if line_s.startswith("#"):
                    key, _, value = line_s[1:].partition(":")
                    if value:
                        extra_info[key.strip()] = value.strip()
                    continue
                result.append((line_s, extra_info))
                extra_info = {}
                count += 1
        with Pool(processes=8) as pool:
            trees = list(pool.imap_unordered(cls.line_mapper,
                                             list(enumerate(result)),
                                             chunksize=400
                                             ))
        trees = [tree for idx, tree in sorted(trees)]
        return trees

    @classmethod
    def line_mapper(cls, args):
        idx, (line_s, extra_info) = args
        tree = cls.from_string(line_s)
        tree.extra = extra_info
        return idx, tree

    def to_sentence(self):
        result = Sentence()
        for idx, (word, postag) in enumerate(self.generate_word_and_postag(), 1):
            result.append(SentenceNode(idx, word, word, postag, postag, None, None, None))
        return result

    @classmethod
    def evaluate_with_external_program(cls, gold_file, output_file):
        with open(gold_file) as f_gold, \
                tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", delete=False) as gold_tmp:
            for line in f_gold:
                line_s = line.strip()
                if line_s and not line_s.startswith("#"):
                    gold_tmp.write(line)

        with open(output_file) as f_output, \
                tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", delete=False) as output_tmp:
            for line in f_output:
                line_s = line.strip()
                if line_s and not line_s.startswith("#"):
                    output_tmp.write(line)

        current_path = os.path.dirname(__file__)
        os.system('{}/../utils/evalb {} {} > {}.txt'.format(
            current_path, gold_tmp.name, output_tmp.name, output_file))
        os.system('cat %s.txt | awk \'/Summary/,EOF { print $0 }\'' % output_file)
        os.remove(gold_tmp.name)
        os.remove(output_tmp.name)

    performance_block_pattern = re.compile(r"-- All --(.*?)\n\n", re.DOTALL)
    performance_pattern = re.compile(r"^(.*?) +=. +([0-9.]+)", re.MULTILINE)

    @classmethod
    def extract_performance(cls, perf_file_name):
        with open(perf_file_name) as f:
            content = f.read()
            content_block = cls.performance_block_pattern.findall(content)[0]

        def generate_items():
            for k, v in cls.performance_pattern.findall(content_block):
                yield k, float(v)

        result = dict(generate_items())
        epoch = re.findall(r"epoch_(\d+)[_.]", perf_file_name)[0]
        result["epoch"] = int(epoch)
        return result

    @classmethod
    def from_words_and_postags(cls, items, escape=True):
        def item_to_preterminal(item):
            word, postag = item
            if escape:
                word = re.sub("[([{]", "-LRB-", word)
                word = re.sub("[)\]}]", "-RRB-", word)
                word = word.replace('"', '``')
            ret = cls(postag)
            ret.children = [Lexicon(word)]
            return ret

        tree = cls("TOP")
        tree.children = [item_to_preterminal(item) for item in items]
        return tree


@attr.s(slots=True)
class ConstTreeStatistics(object):
    words = attr.ib()  # type: Dictionary
    postags = attr.ib()  # type: Dictionary
    leaftags = attr.ib()  # type: Dictionary
    coarsetags = attr.ib()  # type: Dictionary
    labels = attr.ib()  # type: Dictionary
    characters = attr.ib()  # type: Dictionary
    rules = attr.ib()  # type: np.array
    leaftag_to_label = attr.ib()  # type: np.array
    leaftag_to_coarsetag = attr.ib()  # type: np.array
    internal_labels = attr.ib()  # type: np.array
    max_sentence_length = attr.ib()  # type: np.array

    @classmethod
    def tag_to_coursetag(cls, tag):
        return tag.rsplit("+!+", 1)[-1].rsplit("+++", 1)[-1].rsplit("#", 1)[0]

    @classmethod
    def from_sentences(cls, sentences):
        words = Dictionary()
        postags = Dictionary(initial=())
        leaftags = Dictionary(initial=())
        coarsetags = Dictionary(initial=())
        labels = Dictionary(initial=("___EMPTY___",))
        characters = Dictionary()
        rules = []
        internal_labels = {0}
        max_sentence_length = 0
        for sentence in sentences:
            tree_t = sentence.condensed_unary_chain()
            words_and_postags = list(sentence.generate_word_and_postag())
            if len(words_and_postags) > max_sentence_length:
                max_sentence_length = len(words_and_postags)
            for word, postag in words_and_postags:
                words.update([word])
                characters.update(word)
                postags.update([postag])
                coarsetags.update([cls.tag_to_coursetag(postag)])
            for word, leaftag in words_and_postags:
                leaftags.update([leaftag])
            for rule in tree_t.generate_rules():
                labels.update([rule.tag])
                if isinstance(rule.children[0], ConstTree):
                    internal_labels.add(labels.word_to_int[rule.tag])
            for rule in tree_t.generate_rules():
                if isinstance(rule.children[0], ConstTree) and len(rule.children) == 2:
                    rules.append((labels.word_to_int[rule.tag],) +
                                 tuple(labels.word_to_int[i.tag]
                                       for i in rule.children))
        leaftag_to_label = [labels.word_to_int[tag_name]
                            for tag_name in leaftags.int_to_word]
        leaftag_to_coarsetag = [coarsetags.word_to_int[cls.tag_to_coursetag(tag_name)]
                                for tag_name in leaftags.int_to_word]
        # noinspection PyArgumentList
        ret = cls(words, postags, leaftags, coarsetags, labels, characters,
                  np.array(list(set(rules)), dtype=np.int32),
                  np.array(leaftag_to_label, dtype=np.int32),
                  leaftag_to_coarsetag,
                  np.array(sorted(internal_labels), dtype=np.int32),
                  max_sentence_length
                  )
        return ret

    def __str__(self):
        return "{} words, {} postags, {} leaftags, {} coarsetags," \
               "{} labels, {} characters, {} rules, {} internal labels, " \
               "longest sentence has {} words".format(
            len(self.words), len(self.postags), len(self.leaftags), len(self.coarsetags), len(self.labels),
            len(self.characters), len(self.rules), len(self.internal_labels), self.max_sentence_length
        )


if __name__ == '__main__':
    a = ConstTree.from_string(u"""(S (NP (N (N (NP (N (N "pierre"))) (N (N (N "Vinken,")))) (AP (ADV (N (ADJ "61") (N (N "years")))) (AP (AP "old,")))))
 (VP (V "will")
  (VP (VP (V (V "join")) (NP (DET "the") (N (N (N "board")))))
   (PP (PP (P "as") (NP (DET "a") (N (AP "nonexecutive") (N (N (N "director")))))) (PP (NP (NP (DET (N (N "nov."))) (N (N (N "29."))))))))))""")
    print(set(a.generate_spans()))
