import numpy as np
from libcpp.vector cimport vector
cimport numpy as np

cdef extern from "cviterbi.h" namespace "CViterbi":
    cdef cppclass BeamItem:
        BeamItem(np.float64_t, int) nogil
        np.float64_t score
        int pointer

    cdef cppclass AgendaBeam:
        AgendaBeam() nogil
        void setBeamSize(size_t) nogil
        void emplaceItem(double, int) nogil
        void clear() nogil
        size_t size() nogil
        BeamItem& operator[](size_t idx) nogil
        vector[BeamItem] getItems() nogil
