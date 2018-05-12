package jigsaw.syntax;

import java.io.Serializable;
import java.util.Arrays;

/**
 * 
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 22, 2010
 *
 */
public class Rule implements Serializable {

  private static final long serialVersionUID = 4902245258139401162L;

  public Rule(int lhs, int[] rhs) {
    _lhs = lhs;
    _rhs = rhs;
  }
  public Rule(int lhs, int[] rhs, int freq) {
    this(lhs, rhs);
    _freq = freq;
  }
  public int lhs() {
    return _lhs;
  }
  public int[] rhs() {
    return _rhs;
  }
  public int freq() {
    return _freq;
  }
  public int incr() {
    return ++_freq;
  }
  public int arity() {
    return _rhs.length;
  }
  public int id() {
    return _id;
  }
  public void id(int i) {
    _id = i;
  }
  
  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (! (o instanceof Rule))
      return false;
    Rule that = (Rule)o;
    return this._lhs == that._lhs &&
        Arrays.equals(this._rhs, that._rhs);
  }
  
  @Override
  public int hashCode() {
    final int prime = 97;
    int result = 1;
    result = prime * result + _lhs;
    for (int c : _rhs) {
      result = prime * result + c;
    }
    return result;
  }

  public String toString() {
    StringBuffer sb = new StringBuffer();
    sb.append(_freq);
    sb.append(" ");
    sb.append(_lhs);
    for (int r : _rhs) {
      sb.append(" ");
      sb.append(r);
    }
    return sb.toString();
  }
  
  private int _id = -1;
  private int _lhs = -1;
  private int [] _rhs = null;
  private int _freq = 0;
  
  public static void main(String [] args) {
    int [] rhs = {3, 17, -23};
    Rule r = new Rule(253, rhs, 16);
    System.out.println(r.toString());
    int [] newrhs = r.rhs();
    newrhs[0] = 15;
    System.out.println(r.toString());
  }
}
