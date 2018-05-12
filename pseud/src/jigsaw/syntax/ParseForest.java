/**
 * 
 */
package jigsaw.syntax;

import java.util.BitSet;
import java.util.HashMap;
import java.util.Vector;

/**
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 27, 2010
 *
 */
public class ParseForest {

  public ParseForest() {
    
  }
  public ParseForest(BitRecChart chart, BitRecChart ichart, Grammar g) {
    _chart = chart;
    _g = g;
    _ichart = ichart;
    buildSubtree((short)0, (short)_chart.maxpos(), g.NT(g.Start()));
  }
  
  public int newNode(short start, short end, int A) {
    int n = catname.size();
    catname.add(A);
    int idx = _chart.pagesize() * A + end *(end - 1) / 2 + start;
    _nodemap.put(idx, n);
    firstanalysis.add(firstchild.size());
    return n;
  }
  
  public int getNode(short start, short end, int A) {
    int idx = _chart.pagesize() * A + end * (end - 1) / 2 + start;
    if (_nodemap.containsKey(idx))
      return _nodemap.get(idx);
    else    
      return -1;
  }
  
  public void addAnalysis(int node, int ridx, short mid, Vector<int[]> localas) {
    int first = child.size();
    child.add(-1);
    child.add(-1);
    firstchild.add(first);
    int [] analysis = {ridx, mid};
    localas.add(analysis);
    rulenumber.add(ridx);
  }
  
  public void addAnalysis(int node, int ridx, Vector<int[]> localas) {
    int first = child.size();
    child.add(-1);
    firstchild.add(first);
    int [] analysis = {ridx, -1};
    localas.add(analysis);
    rulenumber.add(ridx);
  }
  
  public void addChild(int cidx, int cnode) {
    child.set(cidx, cnode);
  } 
  
  public int buildSubtree(short start, short end, int A) {
   int n = getNode(start, end, A);
   if (n == -1) {
     n = newNode(start, end, A);
     Vector<int[]> analyses = new Vector<int[]>();
     /*
     if (end == start + 1) { // change this to check for input chart
       for (Rule r : _g.trules(A))
         if (_inputs.get(start) == r.rhs()[0]) {
           addAnalysis(n, r.id(), analyses);
         }
     }*/
     for (Rule r : _g.trules(A)) { // A -> tag, tag -> w_{s-e}
       int ti = _g.T2i(r.rhs()[0]);
       if (_ichart.get(start, end, ti))
         addAnalysis(n, r.id(), analyses);
     }
     for (Rule r : _g.chainrules(A)) {
       if (_chart.get(start, end, r.rhs()[0])) {
         addAnalysis(n, r.id(), analyses);
       }
     }
     if (end > start + 1) { // only try binary rules when the spanning is larger than 1
       for (Rule r : _g.birules(A)) {
         // bit vector operation for quick block access of the chart
         BitSet vec1 = _chart.getVectorByStart(start, (short)(end - start - 1), r.rhs()[0]);
         BitSet vec2 = _chart.getVectorByEnd(end, (short)(end - start - 1), r.rhs()[1]);
         vec1.and(vec2);
         for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1))
           addAnalysis(n, r.id(), (short)(start+m+1), analyses);
       }
     }
     for (int i = 0; i < analyses.size(); i ++) {
       int [] a = analyses.get(i);
       int aidx = firstanalysis.get(n)+i;
       Rule r = _g.rules().get(a[0]);
       int fc = firstchild.get(aidx);
       if (a[1] == -1) { // unary rule
          if (r.rhs()[0] < 0) { // preterminal rule
            addChild(fc, -end);
          } else { // chain rule
            int d = buildSubtree(start, end, r.rhs()[0]);
            addChild(fc, d);
          }
       } else { // binary rule
         short mid = (short)a[1];
         int d = buildSubtree(start, mid, r.rhs()[0]);
         addChild(fc, d);
         d = buildSubtree(mid, end, r.rhs()[1]);
         addChild(fc + 1, d);
       }
     }    
   }
   return n;
  }
  
  private BitRecChart _chart = null;
  private Grammar _g = null;
  private BitRecChart _ichart = null;
  
  protected Vector<Integer> catname = new Vector<Integer>(); // the category of the nth constituent
  protected Vector<Integer> firstanalysis = new Vector<Integer>(); // the index of the first analysis of the nth constituent
  protected Vector<Integer> rulenumber = new Vector<Integer>(); // the rule number of analysis a
  protected Vector<Integer> firstchild = new Vector<Integer>(); // the index of the first child for analysis a
  protected Vector<Integer> child = new Vector<Integer>(); // array of child nodes
   
  // mapping from <start,end,A> to node index
  private HashMap<Integer,Integer> _nodemap = new HashMap<Integer, Integer>(); 
 
  public void debugPrint() {
    System.out.print("catname=[");
    for (int name : catname) {
      System.out.print(" "+_g.NT(name));
    }
    System.out.println(" ]");
    System.out.print("catnum=[");
    for (int num : catname) {
      System.out.print(" "+num);
    }
    System.out.println(" ]");
    System.out.print("first-analysis=[");
    for (int fa : firstanalysis) {
      System.out.print(" "+fa);
    }
    System.out.println(" ]");
    System.out.print("rule-number=[");
    for (int rn : rulenumber) {
      System.out.print(" "+rn);
    }
    System.out.println(" ]");
    System.out.print("first-child=[");
    for (int fc : firstchild) {
      System.out.print(" "+fc);
    }
    System.out.println(" ]");
    System.out.print("child=[");
    for (int c : child) {
      System.out.print(" "+c);
    }
    System.out.println(" ]");
  }
  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub

  }
}
