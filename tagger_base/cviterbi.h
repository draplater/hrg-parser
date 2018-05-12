#include <utility>
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>


namespace CViterbi {
    class BeamItem {
    public:
        double score = -INFINITY;
        int pointer = -1;

        BeamItem() {}
        BeamItem(double score, int pointer) : score(score), pointer(pointer) {}

        inline bool operator<(const BeamItem &other) const {
            return this->score < other.score;
        }

        inline bool operator<=(const BeamItem &other) const {
            return this->score <= other.score;
        }

        inline bool operator>(const BeamItem &other) const {
            return this->score > other.score;
        }
    };

    class AgendaBeam {
    private:
        std::vector<BeamItem> beam;
        size_t maxSize = 64;
        bool m_bItemSorted = false;
        void heap_shift_up(size_t base) {
            while (base > 0) {
                size_t next_base = (base - 1) >> 1;
                if (beam[next_base] > beam[base]) {
                    std::swap(beam[next_base], beam[base]);
                    base = next_base;
                } else {
                    break;
                }
            }
        }

        void heap_shift_down(size_t x) {
            size_t beam_size = beam.size();
            size_t i, m;
            for(i=x;
                (i * 2 + 2) < beam_size && // has two children
                beam[i] < beam[m=(beam[i*2+1] > beam[i*2+2])?(i*2+1):(i*2+2)];
                i=m) {
                // swap with the largest children
                std::swap(beam[i], beam[m]);
            }
            if(i*2+1<beam_size && beam[i] < beam[i*2+1]) {
                // has only left children
                std::swap(beam[i], beam[i*2+1]);
            }
        }
    public:
        AgendaBeam() = default;

        void setBeamSize(size_t maxSize) {
            this->maxSize = maxSize;
        }

        ~AgendaBeam() = default;

        void clear() {
            beam.clear();
            m_bItemSorted = false;
        }

        size_t size() {
            return beam.size();
        }

        BeamItem& operator[](size_t idx) {
            return beam[idx];
        }

        std::vector<BeamItem> getItems() {
            return beam;
        }

        void emplaceItem(double score, int pointer) {
            if (beam.size() == maxSize) {
                if (score > beam[0].score) {
                    beam[0].score = score;
                    beam[0].pointer = pointer;
                    heap_shift_down(0);
                } else {
                    return;
                }
            }
            beam.emplace_back(score, pointer);
            heap_shift_up(beam.size() - 1);
        }
    };
}
