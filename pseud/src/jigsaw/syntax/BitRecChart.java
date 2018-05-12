package jigsaw.syntax;

import java.util.BitSet;
import java.util.Stack;

/**
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 22, 2010
 * Bit vector representation of a recognition chart.
 * A boolean value is used to represent whether 
 */
public class BitRecChart implements RecChart {

  public BitRecChart() {
    
  }
  
  public BitRecChart(Grammar g, short maxpos) {
    _g = g;
    _maxpos = maxpos;
    _maxs = _g.nts().size() - 1;
    _pagesize = _maxpos*(_maxpos+1)/2;
    _size = _pagesize * (_maxs+1);
    _bc1 = new BitSet(_size);
    _bc2 = new BitSet(_size);
    _chainvec = _g.chainvec();
  }
  
  /** constructor without given a grammar
   * most likely used for constructing a lexical input chart
   * @param maxpos
   * @param maxs
   */
  public BitRecChart(short maxpos, int maxs) {
    _maxpos = maxpos;
    _maxs = maxs;
    int pagenum = _maxs + 1;
    _pagesize = _maxpos * (maxpos + 1) /2;
    _size = _pagesize * pagenum;
    _bc1 = new BitSet(_size);
    _bc2 = new BitSet(_size);
  }
  
  private BitSet _bc1 = null; // e-adjacent
  private BitSet _bc2 = null; // b-adjacent
  private int _pagesize = -1; // _maxpos*(_maxpos+1)/2
  private int _size = -1; // number of chart cells 
  private BitSet [] _chainvec = null; // chainvec[A] for a NTs

  public BitSet chainvec(int nt) {
    return _chainvec[nt];
  }
  

  
  @Override
  public void clear(short start, short end, int nt) {
    assert start >= 0 && end <= _maxpos && start < end && nt <= _maxs;
    int pageidx = nt * _pagesize;
    int offset1 = start * _maxpos - start * (start + 1) / 2 + end - 1;
    int offset2 = end * (end - 1) / 2 + start;
    _bc1.clear(pageidx + offset1);
    _bc2.clear(pageidx + offset2);
  }

  @Override
  public void clearAll() {
    _bc1.clear();
    _bc2.clear();   
  }

  @Override
  public boolean get(short start, short end, int s) {
    assert start >= 0 && end <= _maxpos && start < end && s <= _maxs;
    int offset = end * (end - 1) / 2 + start;
    return _bc2.get(s * _pagesize + offset);   
  }

  @Override
  public void set(short start, short end, int s) {
    assert start >= 0 && end <= _maxpos && start < end && s <= _maxs;
    int pageidx = s * _pagesize;
    int offset1 = start * _maxpos - start * (start + 1) / 2 + end - 1;
    int offset2 = end * (end - 1) / 2 + start;
    _bc1.set(pageidx + offset1);
    _bc2.set(pageidx + offset2);
  }

  public void or(short start, short end, BitSet vec) {
    assert start >= 0 && end <= _maxpos && start < end && vec.size() == _maxs + 1;
    for (int i = 0; i < vec.size(); i ++) {
      if (vec.get(i))
        set(start, end, i);
    }
  }
  public BitSet getVectorByStart(int start, short length, int s) {
    assert start >= 0 && start < _maxpos && s <= _maxs;
    int idx = s * _pagesize + start * _maxpos - start * (start - 1) / 2;
    return _bc1.get(idx, idx + length);
  }
  
  public BitSet getVectorByEnd(int end, short length, int s) {
    assert end > 0 && end <= _maxpos && s <= _maxs;
    int idx = s * _pagesize + end * (end + 1) / 2;
    return _bc2.get(idx - length, idx);
  }
  
  /** Returns a filtered chart for viterbi parsing */
  public BitRecChart filter() {
    // TODO filter the chart for Viterbi parse
    BitRecChart newchart = new BitRecChart(_g, _maxpos);
    filterSubtree((short)0, _maxpos, _g.NT(_g.Start()), newchart);
    return newchart;
  }

  /* recursive implementation, not efficient
  private void filterSubtree(short start, short end, int nt, BitRecChart newchart) {
    if (newchart.get(start, end, nt)) // we have been here
      return;
    newchart.set(start, end, nt); // TODO one might add a counter here to see the effect of filtering
    for (Rule r : _g.chainrules(nt)) {
      if (get(start, end, r.rhs()[0]))
        filterSubtree(start, end, r.rhs()[0], newchart);
    }
    for (Rule r : _g.birules(nt)) {
      BitSet vec1 = getVectorByStart(start, (short)(end - start - 1), r.rhs()[0]);
      BitSet vec2 = getVectorByEnd(end, (short)(end - start - 1), r.rhs()[1]);
      vec1.and(vec2);
      for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1)) {
        filterSubtree(start, (short)(start+m+1), r.rhs()[0], newchart);
        filterSubtree((short)(start+m+1), end, r.rhs()[1], newchart);
      }
    }
  } */
  
  private void filterSubtree(short start, short end, int nt, BitRecChart newchart) {
    Stack<int[]> stack = new Stack<int[]>();
    stack.push(new int[]{start,end,nt});
    while (!stack.empty()) {
      int[] item = stack.pop();
      short s = (short)item[0];
      short e = (short)item[1];
      int A = item[2];
      if (newchart.get(s,e,A))
        continue;
      newchart.set(s, e, A);
      for (Rule r : _g.chainrules(A)) {
        if (get(s, e, r.rhs()[0]) && !newchart.get(s, e, r.rhs()[0]))
          stack.push(new int[]{s,e,r.rhs()[0]});
      }
      for (Rule r : _g.birules(A)) {
        BitSet vec1 = getVectorByStart(s, (short)(e-s-1), r.rhs()[0]);
        BitSet vec2 = getVectorByEnd(e, (short)(e-s-1), r.rhs()[1]);
        vec1.and(vec2);
        for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1)) {
          if (!newchart.get(s, (short)(s+m+1), r.rhs()[0]))
            stack.push(new int[]{s, s+m+1, r.rhs()[0]});
          if (!newchart.get((short)(s+m+1), e, r.rhs()[1]))
            stack.push(new int[]{s+m+1, e, r.rhs()[1]});
        }
      }
    }
  }
  
  public double fillrate() {
    return ((double)_bc1.cardinality()) / _size; 
  }
  
  /** not tested */
  public Object clone() {
    BitRecChart copy = new BitRecChart();
    copy._bc1 = (BitSet)_bc1.clone();
    copy._bc2 = (BitSet)_bc2.clone();
    copy._g = _g;
    copy._maxs = _maxs;
    copy._maxpos = _maxpos;
    copy._size = _size;
    copy._pagesize = _pagesize;
    copy._chainvec = _chainvec; 
    return copy;  
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub

  }

  private Grammar _g = null;
  
  private int _maxs = -1;
  
  @Override
  public int maxS() {
    return _maxs;
  }

  private short _maxpos = 0;
  
  @Override
  public int maxpos() { 
    return _maxpos;
  }
  
  public int pagesize() {
    return _pagesize;
  }
  public int size() {
    return _size;
  }
  public boolean recognized() {
    return this.get((short)0, _maxpos, _g.NT(_g.Start()));
  }

}
