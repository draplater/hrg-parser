# encoding: utf-8
from __future__ import unicode_literals

import gzip
import re

import six

from hrgguru.pegre_main import Peg, nonterminal, sequence, regex, literal, choice, one_or_more, Ignore
from typing import Generator


class ConstTreeParserError(Exception):
    def __init__(self, message):
        self.value = message

    def __str__(self):
        return self.value


@six.python_2_unicode_compatible
class Lexicon(object):
    def __init__(self, string):
        self.string = string

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
    LEAF = regex(r'"[^"]+"', value=lambda x: x[1:-1])
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


def make_parser_std():
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
    tree_literal_parser_std = make_parser_std()

    def __init__(self, tag):
        self.child = []
        self.tag = tag
        self.span = None
        self.word_span = None

    def __str__(self):
        """
        print tree in the console
        :return: tree
        """
        child_string = " + ".join(i.string if isinstance(i, Lexicon) else i.tag
                                  for i in self.child)
        return "{} {} -> {}".format(self.word_span, self.tag, child_string)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            for i in self.child:
                if isinstance(i, ConstTree) and i.tag.upper() == item.upper():
                    return i
        if isinstance(item, int):
            return self.child[item]
        raise KeyError

    @classmethod
    def make_leaf_node(cls, tag, lexicon):
        ret = cls(tag)
        ret.child.append(Lexicon(lexicon))
        return ret

    @classmethod
    def make_internal_node(cls, tag, sub_trees):
        ret = cls(tag)
        ret.child.extend(sub_trees)
        return ret

    @classmethod
    def from_string(cls, s):
        """
        construct ConstTree from parenthesis representation
        :param s: unicode string of parenthesis representation
        :return: ConstTree root
        :rtype: ConstTree
        """
        return cls.tree_literal_parser.parse(s)

    @staticmethod
    def from_string_0(s):
        """
        construct ConstTree from parenthesis representation
        :param s: unicode string of parenthesis representation
        :return: ConstTree root
        """
        assert isinstance(s, six.text_type)
        # always wrap with __root__
        if not s.startswith(u"( "):
            s = u"( {})".format(s)
        pos = 0
        stack = []
        while pos < len(s):
            if s[pos] == ')':
                pattern_match = re.search("\)+", s[pos:])
                match_string = pattern_match.group(0)
                for i in range(match_string.count(")")):
                    if not stack:
                        raise ConstTreeParserError(
                            'redundant ")" at pos {}.'.format(pos + i))
                    node = stack.pop()
                    if node.tag != '__root__':
                        if not stack:
                            raise ConstTreeParserError("???")
                        stack[-1].child.append(node)
                pos += pattern_match.end(0)
                continue
            pattern_match = re.search("[^\s)]+", s[pos:])
            match_string = pattern_match.group(0)
            if match_string == '(':
                root = ConstTree("__root__")
                stack.append(root)
            elif match_string.startswith('('):
                tag = match_string[1:]
                if not re.match("^[\w/?-]+$", tag):
                    raise ConstTreeParserError(
                        'Invalid tag "{}" at pos {}.'.format(tag, pos))
                node = ConstTree(tag)
                stack.append(node)
            else:
                stack[-1].child.append(Lexicon(match_string))
            pos += pattern_match.end(0)
        if len(stack) != 0:
            raise ConstTreeParserError('missing ")".')
        return root.child[0]

    @classmethod
    def from_java_code(cls, tree_string):
        tree = cls.tree_literal_parser_std.parse(tree_string)
        for word in tree.generate_words():
            spans = re.findall(r"#__#\[(\d+),(\d+)\]", word.string)
            start = min(int(i[0]) for i in spans)
            end = max(int(i[1]) for i in spans)
            word.span = (start, end)
            word.string = re.sub(r"#__#\[(\d+),(\d+)\]", "", word.string)
        return tree

    @classmethod
    def from_java_code_deepbank_1_1(cls, tree_string, sent_contents):
        tree = cls.tree_literal_parser_std.parse(tree_string)
        if "NotExist" in tree_string:
            prev_span_map = {}
            fields = sent_contents.split("\n\n")
            regex = re.compile(r"\(\d+, \d+, \d+, <(\d+):(\d+)>")
            spans = [(int(i[0]), int(i[1])) for i in regex.findall(fields[3])]
            spans = list(set(spans))
            spans.sort()
            for idx in range(len(spans)):
                prev_span_map[spans[idx]] = spans[idx - 1] if idx > 0 else None
            words = list(tree.generate_words())
            for idx, word in enumerate(words):
                if "NotExist" in word.string:
                    right_span = re.findall(r"#__#\[(\d+),(\d+)\]", word.string)[0]
                    prev_span = prev_span_map[int(right_span[0]), int(right_span[1])]
                    word.string = word.string.replace("NotExist", "{},{}".format(
                        prev_span[0], prev_span[1]))
                    words[idx - 1].string = re.sub(r"#__#\[(\d+),(\d+)\]",
                                                   "#__#[{},{}]".format(
                                                       prev_span[0], prev_span[0]),
                                                   words[idx - 1].string
                                                   )

        for word in tree.generate_words():
            spans = re.findall(r"#__#\[(\d+),(\d+)\]", word.string)
            start = min(int(i[0]) for i in spans)
            end = max(int(i[1]) for i in spans)
            word.span = (start, end)
            word.string = re.sub(r"#__#\[(\d+),(\d+)\]", "", word.string)
        return tree

    def generate_words(self):
        """
        :rtype: Generator[Lexicon, None, None]
        """
        for i in self.child:
            if isinstance(i, ConstTree):
                for j in i.generate_words():
                    yield j

        for i in self.child:
            if isinstance(i, Lexicon):
                yield i

    def generate_rules(self):
        for i in self.child:
            if isinstance(i, ConstTree):
                for j in i.generate_rules():
                    yield j
        yield self

    def root_first(self):
        yield self
        for i in self.child:
            if isinstance(i, ConstTree):
                for j in i.root_first():
                    yield j

    def generate_preterminals(self):
        for i in self.child:
            if isinstance(i, ConstTree):
                for j in i.generate_preterminals():
                    yield j

        for i in self.child:
            if isinstance(i, Lexicon):
                yield self

    def is_binary(self):
        if isinstance(self.child[0], Lexicon):
            return True
        return len(self.child) <= 2 and all(i.is_binary() for i in self.child)

    def to_string(self, with_comma=True):
        return "({} {})".format(self.tag, " ".join([i.to_string(with_comma)
                                                    if isinstance(i, ConstTree)
                                                    else ("\"{}\"" if with_comma else "{}").format(
            i.string.strip()) for i in self.child]))

    def condensed_unary_chain(self):
        if len(self.child) == 1:
            if isinstance(self.child[0], Lexicon):
                ret = ConstTree(self.tag)
                ret.span = self.span
                ret.child = list(self.child)
                return ret
            else:
                assert isinstance(self.child[0], ConstTree)
                new_tag = self.tag + "+++" + self.child[0].tag
                tail = self.child[0]

                while len(tail.child) == 1 and isinstance(tail.child[0], ConstTree):
                    tail = tail.child[0]
                    new_tag += "+++" + tail.tag

                ret = ConstTree(new_tag)
                ret.span = self.span
                if len(tail.child) == 1:
                    ret.child = list(tail.child)
                else:
                    ret.child = list(i.condensed_unary_chain()
                                     for i in tail.child)
                return ret
        else:
            ret = ConstTree(self.tag)
            ret.span = self.span
            ret.child = list(i.condensed_unary_chain()
                             for i in self.child)
            return ret

    def expanded_unary_chain(self):
        tags = self.tag.split("+++")
        root_node = last_node = ConstTree(tags[0])
        root_node.span = self.span
        for tag in tags[1:]:
            current_node = ConstTree(tag)
            current_node.span = self.span
            last_node.child = [current_node]
            last_node = current_node

        if isinstance(self.child[0], Lexicon):
            last_node.child = list(self.child)
        else:
            last_node.child = list(i.expanded_unary_chain()
                                   for i in self.child)
        return root_node

    def populate_spans_internal(self):
        for i in self.child:
            if isinstance(i, ConstTree):
                i.populate_spans_internal()
        self.span = (self.child[0].span[0], self.child[-1].span[1])


if __name__ == '__main__':
    tree = ConstTree.from_java_code_deepbank_1_1(
        "(ROOT (root_informal (flr-hd_nwh_c (flr-hd_nwh-nc-np_c (sp-hd_n_c (d_-_sg-nmd_le (punct_ldq ?#__#[6,6]) (d_-_sg-nmd_le what#__#[NotExist]_a#__#[6,7])) (aj-hdn_norm_c (j_att_dlr (aj_pp_i-er_le good#__#[8,12])) (hdn_optcmp_c (n_ms-cnt_ilr (n_pp_mc-of_le feeling#__#[13,20]))))) (sb-hd_nmc_c (hdn_bnp-qnt_c (n_-_pr-it_le it#__#[21,23])) (hd-cmp_u_c (v_vp_mdl-p-unsp_le would#__#[24,29]) (hd-aj_scp_c (hd-aj_int-unsl_c (hd_xcmp_c (v_np_be_le be#__#[30,32])) (hd-cmp_u_c (p_np_i_le for#__#[33,36]) (hdn_bnp-qnt_c (n_-_pr-me_le me#__#[37,39])))) (hd-cmp_u_c (p_vp_bse_le to#__#[40,42]) (hd-cmp_u_c (v_n3s-bse_ilr (v_np_le do#__#[43,45])) (hdn_bnp-qnt_c (hdn_optcmp_c (n_-_pr-dei-sg_le (n_-_pr-dei-sg_le (n_-_pr-dei-sg_le that#__#[46,52]) (punct_comma ,#__#[52,52])) (punct_rdq ?#__#[52,52])))))))))) (sb-hd_nmc_c (hdn_bnp-qnt_c (n_-_pr-he_le he#__#[53,55])) (hd-cmp_u_c (hd_optcmp_c (v_3s-fin_olr (v_pp*-cp_fin-imp_le says#__#[56,60]))) (sb-hd_nmc_c (hdn_bnp-qnt_c (n_-_pr-he_le he#__#[61,63])) (hd_xcmp_c (hd_optcmp_c (v_pst_olr (v_pst_olr (v_pp*-cp_le thought#__#[64,72])) (punct_period .#__#[72,72]))))))))))",
        "/home/chenyufei/Development/large-data/deepbank1.1/export/wsj07c/20758069.gz")

    print(tree)
