from libc.stdlib cimport malloc, calloc, free
import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libcpp.vector cimport vector
from cviterbi cimport *
from numpy.math cimport INFINITY


cdef struct Item:
    np.float64_t score
    int back_pointer


cdef inline void item_assign(Item* item, np.float64_t score, int back_pointer) nogil:
    item.score = score
    item.back_pointer = back_pointer


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cviterbi(np.float64_t[:, :] tag_scores,
            np.float64_t[:, :] transition_scores,
            np.int8_t[:, :] valid_transition_map,
            Item[:,:] table
            ) nogil:
    cdef int sent_length = tag_scores.shape[0]
    cdef int tag_count = tag_scores.shape[1]
    cdef int virtual_start = tag_count
    cdef int virtual_end = tag_count + 1

    cdef int i, tag_i, tag_previous, max_tag_previous, max_last_tag
    cdef np.float64_t max_score, max_last_score, new_score

    # initial
    for tag_i in range(tag_count):
        if valid_transition_map[tag_i, virtual_start]:
            item_assign(&table[0, tag_i],
                        tag_scores[0, tag_i] + transition_scores[tag_i, virtual_start],
                        -1)
        else:
            item_assign(&table[0, tag_i], -INFINITY, -1)

    # dp
    for i in range(1, sent_length):
        for tag_i in range(tag_count):
            max_score = -INFINITY
            max_tag_previous = -1
            for tag_previous in range(tag_count):
                if not valid_transition_map[tag_i, tag_previous]:
                    continue
                new_score = tag_scores[i, tag_i] + \
                            transition_scores[tag_i, tag_previous] + \
                            table[i - 1, tag_previous].score
                if new_score > max_score:
                    max_score = new_score
                    max_tag_previous = tag_previous
            item_assign(&table[i, tag_i], max_score, max_tag_previous)

    # traceback
    max_last_score = -INFINITY
    max_last_tag = -1
    for tag_i in range(tag_count):
        if valid_transition_map[virtual_end, tag_i]:
            new_score = table[sent_length-1, tag_i].score + \
                        transition_scores[virtual_end, tag_i]
            if new_score > max_last_score:
                max_last_score = new_score
                max_last_tag = tag_i
    return max_last_tag


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int cviterbi_beam_saerch(np.float64_t[:, :] tag_scores,
            np.float64_t[:, :] transition_scores,
            np.int8_t[:, :] valid_transition_map,
            Item[:,:] table,
            int beam_size
            ) nogil:
    cdef int sent_length = tag_scores.shape[0]
    cdef int tag_count = tag_scores.shape[1] - 2
    cdef int virtual_start = tag_count
    cdef int virtual_end = tag_count + 1
    cdef AgendaBeam old_beam, beam

    cdef int i, tag_i, tag_previous, max_tag_previous, max_last_tag, current_beam_size, idx
    cdef np.float64_t max_score, max_last_score, new_score

    beam.setBeamSize(beam_size)

    # initial
    for tag_i in range(tag_count):
        if valid_transition_map[tag_i, virtual_start]:
            new_score = tag_scores[0, tag_i] + transition_scores[tag_i, virtual_start]
            item_assign(&table[0, tag_i], new_score, -1)
            beam.emplaceItem(new_score, tag_i)
        else:
            item_assign(&table[0, tag_i], -INFINITY, -1)

    # dp
    for i in range(1, sent_length):
        old_beam = beam
        current_beam_size = old_beam.size()
        beam.clear()
        for tag_i in range(tag_count):
            max_score = -INFINITY
            max_tag_previous = -1
            for idx in range(current_beam_size):
                tag_previous = old_beam[idx].pointer
                if not valid_transition_map[tag_i, tag_previous]:
                    continue
                new_score = tag_scores[i, tag_i] + \
                            transition_scores[tag_i, tag_previous] + \
                            table[i - 1, tag_previous].score
                if new_score > max_score:
                    max_score = new_score
                    max_tag_previous = tag_previous
            item_assign(&table[i, tag_i], max_score, max_tag_previous)
            beam.emplaceItem(max_score, tag_i)

    # traceback
    max_last_score = -INFINITY
    max_last_tag = -1
    for tag_i in range(tag_count):
        if valid_transition_map[virtual_end, tag_i]:
            new_score = table[sent_length-1, tag_i].score + \
                        transition_scores[virtual_end, tag_i]
            if new_score > max_last_score:
                max_last_score = new_score
                max_last_tag = tag_i
    return max_last_tag


def viterbi(np.float64_t[:, :] tag_scores,
            np.float64_t[:, :] transition_scores,
            np.int8_t[:, :] valid_transition_map,
            beam_size=None
            ):
    cdef int sent_length = tag_scores.shape[0]
    cdef int tag_count = tag_scores.shape[1]
    cdef void* pool = calloc(sent_length * tag_count, sizeof(Item))
    cdef Item[:,:] table = <Item[:sent_length, :tag_count]> pool
    cdef int c_beam_size

    cdef int last_tag

    if beam_size is None:
        with nogil:
            last_tag = cviterbi(tag_scores, transition_scores,
                                     valid_transition_map, table)
    else:
        c_beam_size = beam_size
        with nogil:
            last_tag = cviterbi_beam_saerch(tag_scores, transition_scores,
                                     valid_transition_map, table, c_beam_size)
    path = [last_tag]
    for i in range(sent_length-1, 0, -1):
        tag_previous = table[i, last_tag].back_pointer
        path.append(tag_previous)
        last_tag = tag_previous
    free(pool)
    return path[::-1]
