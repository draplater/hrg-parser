from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memset
import cython
cimport cython
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

from span.const_tree import ConstTree, Lexicon
ctypedef unsigned char index_t

cdef struct Item:
    index_t start
    index_t end
    short label
    float score
    long left  # cython doesn't allow pointer in memorview, so define it as long
    long right
    bint initialized

cdef inline void item_assign(Item* item, index_t start, index_t end, short label,
                      double score, Item* left, Item* right) nogil:
    item.start = start
    item.end = end
    item.label = label
    item.score = score
    item.left = <long> left
    item.right = <long> right
    item.initialized = True

cdef class ItemWrapped(object):
    cdef public:
        object start, end, label, score, left, right

    @staticmethod
    cdef create(Item* item):
        cdef ItemWrapped self = ItemWrapped()
        self.start = item.start
        self.end = item.end
        self.label = item.label
        self.score = item.score
        self.left = ItemWrapped.create(<Item *> item.left) if item.left != 0 else None
        self.right = ItemWrapped.create(<Item *> item.right) if item.right != 0 else None
        return self

    def generate_scoreable_spans(self, label_map):
        yield (self.start, self.end, label_map[self.label])
        if self.left is not None:
            for i in self.left.generate_scoreable_spans(label_map):
                yield i

        if self.right is not None:
            for i in self.right.generate_scoreable_spans(label_map):
                yield i

    def flat_children(self):
        if self.left is None and self.right is None:
            return []
        return ([self.left] if self.left.label != 0
            else self.left.flat_children()) + [self.right]

    def to_const_tree(self, label_map, words):
        ret = ConstTree(label_map[self.label], (self.start, self.end))
        if self.end - self.start == 1:
            word = words[self.start]
            assert isinstance(word, Lexicon)
            ret.children = [word]
        else:
            ret.children = [i.to_const_tree(label_map, words) for i in self.flat_children()]
        return ret

    def __str__(self):
        return "[{}, {}, {}]".format(self.start, self.end, self.label)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Item* cky_binary_rule_decoder(int[:, :] rules,
                            np.float64_t[:, :] span_scores,
                            np.float64_t[:, :, :] label_scores,
                            np.float64_t[:, :] leaftag_scores,
                            np.int32_t[:] leaftag_to_label,
                            Item[:,:,:] table
                           ) nogil:
    cdef int sent_length =  span_scores.shape[0]
    cdef int label_count = label_scores.shape[2]
    cdef int leaftag_count = leaftag_to_label.shape[0]
    cdef int rule_count = rules.shape[0]
    cdef bint use_leaftag = leaftag_scores is not None
    cdef double score
    cdef index_t i, k, l, start, end
    cdef short label_idx, lhs, rhs1, rhs2
    cdef int rule, leaftag_idx
    cdef Item *left_item
    cdef Item *right_item
    cdef Item *i_item
    cdef Item *current_item

    with nogil:
        for i in range(sent_length):
            for leaftag_idx in range(leaftag_count):
                label_idx = leaftag_to_label[leaftag_idx]
                if use_leaftag:
                    item_assign(&table[i,i+1,label_idx], i, i+1, label_idx,
                                # leaftag_scores[i, leaftag_idx] + label_scores[i, i+1, label_idx],
                                leaftag_scores[i, leaftag_idx],
                                NULL, NULL)
                else:
                    item_assign(&table[i,i+1,label_idx], i, i+1, label_idx,
                                label_scores[i, i+1, label_idx], NULL, NULL)

        for l in range(2, sent_length + 1):
             # for start in prange(sent_length - l + 1,
             #                     num_threads=10, schedule="dynamic", chunksize=1):
             for start in range(sent_length - l + 1):
                 end = start + l
                 for label_idx in range(label_count):
                     current_item = &table[start, end, label_idx]
                     current_item.score = -INFINITY

                 for rule in range(rule_count):
                     lhs = rules[rule,0]
                     rhs1 = rules[rule,1]
                     rhs2 = rules[rule,2]
                     current_item = &table[start, end, lhs]
                     for k in range(start+1, end):
                         left_item = &table[start,k,rhs1]
                         right_item = &table[k,end,rhs2]
                         if not left_item.initialized or not right_item.initialized:
                             continue
                         score = left_item.score + right_item.score + span_scores[start, end] + \
                                 label_scores[start, end, lhs]
                         if score > current_item.score:
                             item_assign(current_item, start, end, lhs, score, left_item, right_item)

        current_item = &table[0,sent_length,1]
        for label_idx in range(1, label_count):
            i_item = &table[0,sent_length,label_idx]
            if i_item.score > current_item.score:
                current_item = i_item
        return current_item


cdef class CKYBinaryRuleDecoder(object):
    cdef void* pool
    def __cinit__(self, max_sentence_size, label_count):
        self.pool = malloc(max_sentence_size * (max_sentence_size + 1) * label_count * sizeof(Item))

    def __call__(self, rules, span_scores, label_scores,
                leaftag_scores, leaftag_to_label, internal_labels):
        cdef int sent_length =  span_scores.shape[0]
        cdef int label_count = label_scores.shape[2]
        cdef Item[:,:,:] table = <Item[:sent_length, :(sent_length+1), :label_count]> self.pool
        cdef Item* current_item

        memset(self.pool, 0, sent_length * (sent_length + 1) * label_count * sizeof(Item))

        current_item = cky_binary_rule_decoder(
            rules, span_scores, label_scores,
            leaftag_scores, leaftag_to_label, table)

        if not current_item.initialized:
            raise ArithmeticError("Can't decode.")
        ret = ItemWrapped.create(current_item)
        return ret

    def __dealloc__(self):
        free(self.pool)


cdef class CKYRuleFreeDecoder(CKYBinaryRuleDecoder):
    def __cinit__(self, max_sentence_size, label_count):
        pass

    def __call__(self, rules, span_scores, label_scores,
                 leaftag_scores, leaftag_to_label, internal_labels):
        return cky_decoder_2(span_scores, label_scores, leaftag_scores,
                             leaftag_to_label, internal_labels)


@cython.boundscheck(False)
@cython.wraparound(False)
def cky_decoder_2(np.float64_t[:, :] span_scores,
                  np.float64_t[:, :, :] label_scores,
                  np.float64_t[:, :] leaftag_scores,
                  np.int32_t[:] leaftag_to_label,
                  np.int32_t[:] internal_labels
                  ):
    cdef int sent_length =  span_scores.shape[0]
    cdef int label_count = label_scores.shape[2]
    cdef int leaftag_count = leaftag_to_label.shape[0]
    cdef int internal_count = internal_labels.shape[0]
    cdef bint use_leaftag = leaftag_scores is not None

    cdef void* pool = malloc(
        (sent_length + 1) * (sent_length + 1) * 2 * sizeof(Item))
    cdef Item[:,:,:] table = <Item[:(sent_length+1), :(sent_length+1), :2]> pool

    cdef int i, label_idx, leaftag_idx, internal_idx, span_length, start, k, end
    cdef int sub_label_idx, best_label_idx = 0, best_label_idx_nonempty = 0
    cdef double score, best_label_score, best_label_score_nonempty

    cdef Item* item
    cdef Item* item_nonempty
    cdef Item* left_item
    cdef Item* right_item
    cdef Item* current_item
    cdef Item* current_item_nonempty

    with nogil:
        for i in range(sent_length):
            item = &table[i,i+1,0]  # empty or noempty
            item_nonempty = &table[i,i+1,1]  # noempty
            item.score = -INFINITY
            for leaftag_idx in range(leaftag_count):
                label_idx = leaftag_to_label[leaftag_idx]
                if use_leaftag:
                    score = leaftag_scores[i, leaftag_idx]
                else:
                    score = label_scores[i, i+1, label_idx]
                if score > item.score:
                    item_assign(item, i, i+1, label_idx, score, NULL, NULL)
                    item_assign(item_nonempty, i, i+1, label_idx, score, NULL, NULL)

        for span_length in range(2, sent_length + 1):
             for start in range(sent_length - span_length + 1):
                 end = start + span_length
                 current_item = &table[start, end, 0]
                 current_item_nonempty = &table[start, end, 1]
                 current_item.score = -INFINITY
                 current_item_nonempty.score = -INFINITY
                 best_label_score = -INFINITY
                 best_label_score_nonempty = -INFINITY
                 for internal_idx in range(internal_count):
                     label_idx = internal_labels[internal_idx]
                     score = label_scores[start, end, label_idx]
                     if score > best_label_score:
                         best_label_score = score
                         best_label_idx = label_idx
                     if score > best_label_score_nonempty and label_idx != 0:
                         best_label_score_nonempty = score
                         best_label_idx_nonempty = label_idx
                 for k in range(start+1, end):
                     left_item = &table[start,k,0]
                     right_item = &table[k,end,1]
                     score = left_item.score + right_item.score + span_scores[start, end] + \
                             best_label_score
                     if score > current_item.score:
                         item_assign(current_item, start, end, best_label_idx, score, left_item, right_item)

                     score = left_item.score + right_item.score + span_scores[start, end] + \
                             best_label_score_nonempty
                     if score > current_item_nonempty.score:
                         item_assign(current_item_nonempty, start, end, best_label_idx_nonempty, score, left_item, right_item)


    current_item = &table[0,sent_length,1]
    ret = ItemWrapped.create(current_item)
    free(pool)
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
def cky_binary_rule_decoder_unlabeled(np.float64_t[:, :] span_scores):
    cdef int sent_length =  span_scores.shape[0]
    cdef double score
    cdef int i, k, l, start, end, label_idx, lhs, rule
    cdef Item *left_item
    cdef Item *right_item
    cdef Item *i_item
    cdef Item *current_item

    cdef void* pool = calloc(sent_length * (sent_length + 1), sizeof(Item))
    cdef Item[:,:] table = <Item[:sent_length, :(sent_length+1)]> pool


    with nogil:
        for l in range(2, sent_length + 1):
             for start in range(sent_length - l + 1):
                 end = start + l
                 current_item = &table[start, end]
                 current_item.score = -INFINITY
                 for k in range(start+1, end):
                     left_item = &table[start,k]
                     right_item = &table[k,end]
                     if not left_item.initialized or not right_item.initialized:
                         continue
                     score = left_item.score + right_item.score + span_scores[start, end]
                     if score > current_item.score:
                         item_assign(current_item, start, end, 0, score, left_item, right_item)
    current_item = &table[0,sent_length]
    ret = ItemWrapped.create(current_item)
    free(pool)
    return ret


cdef class CKYBinaryRuleDecoderUnlabeled(object):
    cdef void* pool
    def __cinit__(self, max_sentence_size, label_count):
        self.pool = malloc(max_sentence_size * (max_sentence_size + 1) * label_count * sizeof(Item))

    def __call__(self, rules, span_scores, label_scores,
                leaftag_scores, leaftag_to_label, internal_labels):
        cdef int sent_length =  span_scores.shape[0]
        cdef int label_count = label_scores.shape[2]
        cdef Item[:,:,:] table = <Item[:sent_length, :(sent_length+1), :label_count]> self.pool
        cdef Item* current_item

        memset(self.pool, 0, sent_length * (sent_length + 1) * label_count * sizeof(Item))

        current_item = cky_binary_rule_decoder(
            rules, span_scores, label_scores,
            leaftag_scores, leaftag_to_label, table)

        if not current_item.initialized:
            raise ArithmeticError("Can't decode.")
        ret = ItemWrapped.create(current_item)
        return ret

    def __dealloc__(self):
        free(self.pool)


cdef struct TreeInfo:
    short left
    short right
    bint is_leaf


cdef inline void tree_info_assign(
        TreeInfo* item,
        short left, short right, bint is_leaf) nogil:
    item.left = left
    item.right = right
    item.is_leaf = is_leaf


cdef struct TreeLabelItem:
    short left_idx
    short right_idx
    int label
    bint initialized

cdef inline void tree_label_item_assign(TreeLabelItem* item,
                                        short left_idx,
                                        short right_idx,
                                        int label
                                        ) nogil:
    item.left_idx = left_idx
    item.right_idx = right_idx
    item.label = label
    item.initialized = True


@cython.boundscheck(False)
@cython.wraparound(False)
def assign_label_by_rules(
        int sent_length,
        int[:, :] rules,
        gold_tree,
        np.float64_t[:, :] label_scores,
        np.float64_t[:, :] leaftag_scores,
        np.int32_t[:] leaftag_to_label,
        ):
    cdef int nodes_count = label_scores.shape[0]
    cdef int label_count = label_scores.shape[1]
    cdef int leaftag_count = leaftag_to_label.shape[0]
    cdef int rule_count = rules.shape[0]
    cdef bint use_leaftag = leaftag_scores is not None
    cdef bint is_leaf

    # cdef void* info_pool = calloc(nodes_count, sizeof(Item))
    # cdef TreeInfo[:] table = <TreeInfo[:nodes_count]> memory_pool
    #
    # for tree_node in gold_tree.generate_rules():
    #     is_leaf = isinstance(tree_node.children[0], Lexicon)
