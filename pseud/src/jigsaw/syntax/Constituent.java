package jigsaw.syntax;

import java.io.Serializable;
import java.util.Collection;

public class Constituent<L> implements Serializable{
  private static final long serialVersionUID = 1L;
  L label;
  int start;
  int end;

  public L getLabel() {
    return label;
  }

  public int getStart() {
    return start;
  }

  public int getEnd() {
    return end;
  }
  
  public int getLength() {
      return end-start+1;
  }

  public String toString() {
    return "<"+label+" : "+start+", "+end+">"; 
  }

  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Constituent)) return false;

    final Constituent<L> constituent = (Constituent<L>) o;

    if (end != constituent.end) return false;
    if (start != constituent.start) return false;
    if (label != null ? !label.equals(constituent.label) : constituent.label != null) return false;

    return true;
  }

  public int hashCode() {
    int result;
    result = (label != null ? label.hashCode() : 0);
    result = 29 * result + start;
    result = 29 * result + end;
    return result;
  }
  
  /**
   * Detects whether this constituent overlaps any of a Collection of
   * Constituents without nesting, that is, 
   * whether it "crosses" any of them.
   *
   * @param constColl The set of constituent to check against
   * @return True if some constituent in the collection is crossed
   * @throws ClassCastException If some member of the Collection isn't
   *                            a Constituent
   */
  public boolean crosses(Collection<Constituent<L>> constColl) {
    for (Constituent<L> c : constColl) {
      if (crosses(c)) {
        return true;
      }
    }
    return false;
  }
  /**
   * Detects whether this constituent overlaps a constituent without
   * nesting, that is, whether they "cross".
   *
   * @param c The constituent to check against
   * @return True if the two constituents cross
   */
  public boolean crosses(Constituent<L> c) {
    // bug fixed by WS
    return (start < c.start && c.start <= end && end < c.end) || (c.start < start && start <= c.end && c.end < end);
  }


  public Constituent(L label, int start, int end) {
    this.label = label;
    this.start = start;
    this.end = end;
  }
}
