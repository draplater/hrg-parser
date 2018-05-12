from __future__ import unicode_literals

import codecs
from collections import namedtuple

import sys

import six

from typing import Union, Optional, List

from namedlist import namedlist


class CoNLLNode(object):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        try:
            return cls(*fields)
        except TypeError:
            print("Line: " + line)
            raise

    def to_line(self, sep='\t'):
        return sep.join(six.text_type(i) for i in self)


class CoNLLUNode(CoNLLNode,
                 namedlist("_", ["id_", "form", "lemma", "cpostag", "postag", "feats",
                                 "head", "deprel", "deps", "misc"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        if len(fields) < len(cls._fields):
            fields.extend(["_" for _ in range(len(cls._fields) - len(fields))])
        try:
            return cls(*fields)
        except TypeError:
            print("Line: " + line)
            raise


class CoNLL09Node(namedtuple("_", ["id_", "form", "lemma", "plemma", "postag", "ppostag", "feats",
                                   "pfeats", "head", "phead", "deprel", "pdeprel", "fillpred", "pred", "arg"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        if len(fields) < 13:
            raise AttributeError("too few fields: {}".format(line))
        if len(fields) < 14:
            fields.extend(["-"] * (14 - len(fields)))
        cols = fields[0:14]
        cols.append(fields[14:])
        return cls(*cols)

    def to_line(self, sep="\t"):
        return sep.join(list(self[0:-1]) + self[-1])


class CoNLL08Node(namedtuple("_", ["id_", "form", "lemma", "postag", "ppostag",
                                   "split_form", "split_lemma", "ppostags",
                                   "head", "deprel", "pred", "arg"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        cols = fields[0:11]
        cols.append(fields[11:])
        try:
            return cls(*cols)
        except TypeError:
            print("Line: " + line)
            raise

    def to_line(self, sep="\t"):
        return sep.join(six.text_type(i) for i in list(self[0:-1]) + self[-1])

    def __hash__(self):
        return int(self.id_)


class SDPNode(namedlist("_", ["id_", "form", "lemma", "postag",
                              "top", "pred", "sense", "arg"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        cols = fields[0:7]
        cols.append(fields[7:])
        try:
            return cls(*cols)
        except TypeError:
            print("Line: " + line)
            raise

    def to_line(self, sep="\t"):
        return sep.join(six.text_type(i) for i in list(self[0:-1]) + self[-1])

    def __hash__(self):
        return int(self.id_)


class OldSDPNode(namedlist("_", ["id_", "form", "lemma", "postag",
                                 "top", "pred", "arg"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        cols = fields[0:6]
        if len(cols) == 4:
            cols.extend(["-", "-"])
        cols.append(fields[6:])
        try:
            return cls(*cols)
        except TypeError:
            print("Line: " + line)
            raise

    def to_line(self, sep="\t"):
        return sep.join(six.text_type(i) for i in list(self[0:-1]) + self[-1])

    def __hash__(self):
        return int(self.id_)


class TTNode(namedtuple("_", ["start", "end", "form", "lemma", "postag"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        return cls(*fields)

    def to_line(self):
        raise NotImplementedError


class SimpleNode(namedlist("_", ["word", "postag", "head", "deprel"])):
    @classmethod
    def from_line(cls, line):
        fields = line.split()
        return cls(*fields)

    def to_line(self, sep='\t'):
        return sep.join(six.text_type(i) for i in self)


class CoNLL06Node(CoNLLNode,
                  namedtuple("_", ["id_", "form", "lemma", "cpostag", "postag",
                                   "feats", "head", "deprel", "phead", "pdeprel"])):
    pass


class BaseConLLSentence(list):
    NodeType = None

    def __init__(self, *args):
        super(BaseConLLSentence, self).__init__(*args)
        self.comment = None  # type: Optional[Union[str, List[str]]]

    @classmethod
    def get_sentence(cls, file_object):
        sentence = cls()
        start = True
        for i in file_object:
            line = i.strip()
            if not line:
                break
            if start:
                if line.startswith("#") and line[1] != "\t":
                    comment = line[1:].strip()
                    if sentence.comment is None:
                        sentence.comment = comment
                    elif isinstance(sentence.comment, six.string_types):
                        sentence.comment = [sentence.comment, comment]
                    else:
                        assert isinstance(sentence.comment, list)
                        sentence.comment.append(comment)
                    continue
            sentence.append(cls.NodeType.from_line(line))
            start = False
        return sentence

    @classmethod
    def get_all_sentences(cls, file_object, limit=None):
        result = []
        while True:
            sentence = cls.get_sentence(file_object)
            if not sentence:
                break
            result.append(sentence)
            if limit is not None and len(result) == limit:
                break
        return result

    def get_comment_line(self):
        if self.comment is None:
            return ""
        elif isinstance(self.comment, six.string_types):
            return "# {}".format(self.comment.strip()).replace("\n", " ") + "\n"
        else:
            assert isinstance(self.comment, (list, tuple))
            return "\n".join("# {}".format(i.strip()).replace("\n", " ") for i in self.comment) + "\n"

    def to_string(self, sep="\t"):
        return self.get_comment_line() + "\n".join(i.to_line(sep) for i in self) + "\n\n"


class CoNLL09Sentence(BaseConLLSentence):
    NodeType = CoNLL09Node


class CoNLLUSentence(BaseConLLSentence):
    NodeType = CoNLLUNode


class TTSentence(BaseConLLSentence):
    NodeType = TTNode


class CoNLL08Sentence(BaseConLLSentence):
    NodeType = CoNLL08Node


class SimpleSentence(BaseConLLSentence):
    NodeType = SimpleNode


class CoNLL06Sentence(BaseConLLSentence):
    NodeType = CoNLL06Node


class SDPSentence(BaseConLLSentence):
    NodeType = SDPNode

    @classmethod
    def get_sentence(cls, file_object):
        sentence = cls()
        for i in file_object:
            line = i.strip()
            if line.startswith("#"):
                sentence.comment = line[1:].strip()
                continue
            if line == "null":
                continue
            if not line:
                break
            sentence.append(cls.NodeType.from_line(line))
        return sentence


class OldSDPSentence(BaseConLLSentence):
    NodeType = OldSDPNode

    @classmethod
    def get_sentence(cls, file_object):
        sentence = cls()
        for i in file_object:
            line = i.strip()
            if line.startswith("#"):
                sentence.comment = line[1:].strip()
                continue
            if line == "null":
                continue
            if not line:
                break
            sentence.append(cls.NodeType.from_line(line))
        return sentence


def sent_convert(sent):
    result = CoNLLUSentence()
    for i in sent:
        # noinspection PyArgumentList
        result.append(
            CoNLLUNode(i.id_, i.form, i.lemma, '_', i.postag, i.feats, i.head, i.deprel, '_', '_', []))
    return result


def main():
    with codecs.open(sys.argv[1]) as f:
        sents = CoNLL09Sentence.get_all_sentences(f)
        with codecs.open(sys.argv[2], "w") as f_w:
            for i in sents:
                f_w.write(sent_convert(i).to_string())


def make_converter(SourceSentence, TargetSentence, node_converter, sep="\t"):
    def converter(source_file, output_file):
        with codecs.open(source_file) as f:
            sents = SourceSentence.get_all_sentences(f)
            with codecs.open(output_file, "w") as f_w:
                for source_sent in sents:
                    converted_sent = TargetSentence(node_converter(i) for i in source_sent)
                    f_w.write(converted_sent.to_string(sep))

    return converter


if __name__ == '__main__':
    main()
