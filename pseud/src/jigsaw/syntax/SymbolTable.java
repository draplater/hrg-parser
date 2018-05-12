package jigsaw.syntax;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Vector;

/**
 * Symbol Table: maintaining string-to-int records.
 * 
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 22, 2010
 *
 */
public class SymbolTable implements Serializable {
  
  private static final long serialVersionUID = 1L;

  public SymbolTable() {
    
  }
  
  public void clear() {
    _st = new HashMap<String,Integer>();
    _ts = new Vector<String>();
    _num = 0;
  }
  
  public int size() {
    return _num;
  }
  
  public int register(String t) {
    if (_st.containsKey(t)) {
      return _st.get(t);
    } else {
      _st.put(t, _num);
      _ts.add(t);
      return _num++;
    }
  }
  
  public int lookup(String s) {
    if (_st.containsKey(s))
      return _st.get(s);
    else
      return -1;
  }
  
  public String lookup(int t) {   
    if (t <= _ts.size() - 1 && t >= 0)
      return _ts.get(t);
    else
      return null;
  }

  public String[] symbols() {
    return _ts.toArray(new String[size()]);
  }
  
  private HashMap<String,Integer> _st = new HashMap<String,Integer>();
  private Vector<String> _ts = new Vector<String>(); 
  private int _num = 0;
  
  public static void main(String [] args) {
    SymbolTable st = new SymbolTable();
    System.out.println("Size:"+st.size());
    System.out.println("Register S:"+st.register("S"));
    System.out.println("Register NP:"+st.register("NP"));
    System.out.println("Size:"+st.size());
    System.out.println("Lookup 0:"+st.lookup(0));
    System.out.println("Lookup 1:"+st.lookup(1));
    System.out.println("Lookup S:"+st.lookup("S"));
    System.out.println("Lookup NP:"+st.lookup("NP"));
    System.out.println("Lookup X:"+st.lookup("X"));
    System.out.println("Cleaning ...");
    st.clear();
    System.out.println("Size:"+st.size());
  }
}
