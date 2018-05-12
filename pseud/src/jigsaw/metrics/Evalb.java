package jigsaw.metrics;


import java.util.*;
import java.io.*;

import jigsaw.syntax.Constituent;
import jigsaw.syntax.Tree;

public class Evalb implements SetEvaluator {

  private static final boolean DEBUG = false;

  protected final String str;
  protected final boolean runningAverages;

  protected double precision = 0.0;
  protected double recall = 0.0;
  protected double f1 = 0.0;
  protected double num = 0.0;
  protected double exact = 0.0;

  protected double precision2 = 0.0;
  protected double recall2 = 0.0;
  protected double pnum2 = 0.0;
  protected double rnum2 = 0.0;
  
  protected double curF1 = 0.0;
  
  public Evalb(String str, boolean runningAverages) {
    this.str = str;
    this.runningAverages = runningAverages;
  }
  
  public double getSentAveF1() {
    return f1 / num;
  }

  public double getEvalbF1() {
    return 2.0 / (rnum2 / recall2 + pnum2 / precision2);
  }
  
  /**
   * Return the evalb F1% from the last call to {@link #evaluate}.
   * 
   * @return The F1 percentage
   */
  public double getLastF1() {
    return curF1 * 100.0;
  }

  /** @return The evalb (micro-averaged) F1 times 100 to make it
   *  a number between 0 and 100.
   */
  public double getEvalbF1Percent() {
    return getEvalbF1() * 100.0;
  }

  public double getExact() {
    return exact / num;
  }

  public double getExactPercent() {
    return getExact() * 100.0;
  }

  public int getNum() {
    return (int) num;
  }

  // should be able to pass in a comparator!
  protected static double precision(Set<?> s1, Set<?> s2) {
    double n = 0.0;
    double p = 0.0;
    for (Object o1 : s1) {
      if (s2.contains(o1)) {
        p += 1.0;
      }
      if (DEBUG) {
        if (s2.contains(o1)) {
          System.err.println("Eval Found: "+o1);
        } else {
          System.err.println("Eval Failed to find: "+o1);
        }
      }
      n += 1.0;
    }
    if (DEBUG) System.err.println("Matched " + p + " of " + n);
    return (n > 0.0 ? p / n : 0.0);
  }

  protected Set<Constituent<String>> makeObjects(Tree<String> tree) {
    if(tree == null) 
      return null;

    //evalb only evaluates phrasal categories, thus constituents() does not
    //return objects for terminals and pre-terminals
    Set<Constituent<String>> set = new HashSet<Constituent<String>>();    
    Map<Tree<String>, Constituent<String>> tcmap = tree.getConstituents();
    for (Tree<String> node : tcmap.keySet()){
      if (node == tree || node.isLeaf() || node.isPreTerminal())
        continue;
      set.add(tcmap.get(node));      
    }
    return set;
  }

  public void evaluate(Tree<String> guess, Tree<String> gold) {
    evaluate(guess, gold, new PrintWriter(System.out, true));
  }

  /* Evaluates precision and recall by calling makeObjects() to make a
   * set of structures for guess Tree and gold Tree, and compares them
   * with each other.
   */
  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw) {
    evaluate(guess, gold, pw, 1.0);
  }
  
  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw, double weight) {
    if(gold == null || guess == null) {
      System.err.printf("%s: Cannot compare against a null gold or guess tree!\n",this.getClass().getName());
      return;
    }
    Set<?> dep1 = makeObjects(guess);
    Set<?> dep2 = makeObjects(gold);
    final double curPrecision = precision(dep1, dep2);
    final double curRecall = precision(dep2, dep1);
    curF1 = (curPrecision > 0.0 && curRecall > 0.0 ? 2.0 / (1.0 / curPrecision + 1.0 / curRecall) : 0.0);
    precision += curPrecision * weight;
    recall += curRecall * weight;
    f1 += curF1 * weight;
    num += weight;

    precision2 += dep1.size() * curPrecision * weight;
    pnum2 += dep1.size() * weight;

    recall2 += dep2.size() * curRecall * weight;
    rnum2 += dep2.size() * weight;

    if (curF1 > 0.9999) {
      exact += 1.0;
    }
    if (pw != null) {
      double cF1 = 2.0 / (rnum2 / recall2 + pnum2 / precision2);
      pw.format("%s [current] P: %.2f R: %.2f F1: %.2f", str, curPrecision*100, curRecall*100, curF1*100);
//      pw.print(str + " [current] P: " + ((int) (curPrecision * 10000)) / 100.0 + 
//               " R: " + ((int) (curRecall * 10000)) / 100.0 +
//               " F1: " + ((int) (curF1 * 10000)) / 100.0);
      if (runningAverages) {
//        pw.print(str + " [average-s] P: " + ((int) (precision * 10000 / num)) / 100.0 +
//                 " R: " + ((int) (recall * 10000 / num)) / 100.0 +
//                 " F1: " + ((int) (10000 * f1 / num)) / 100.0);
        pw.format(" - [average] P: %.2f R: %.2f F1: %.2f Ex: %.2f N: %d", precision2*100/pnum2, recall2*100/rnum2, cF1*100, 100*exact/num, getNum());
//        pw.print(" [average] P: " + ((int) (precision2 * 10000 / pnum2)) / 100.0 +
//                 " R: " + ((int) (recall2 * 10000 / rnum2)) / 100.0 +
//                 " F1: " + ((int) (10000 * cF1)) / 100.0);
//        pw.print(" Ex: " + ((int) (10000 * exact / num)) / 100.0 +
//                   " N: " + getNum());
      }
      pw.println();
    }
  }

  public void display(boolean verbose) {
    display(verbose, new PrintWriter(System.out, true));
  }

  public void display(boolean verbose, PrintWriter pw) {
    double prec = precision2 / pnum2;//(num > 0.0 ? precision/num : 0.0);
    double rec = recall2 / rnum2;//(num > 0.0 ? recall/num : 0.0);
    double f = 2.0 / (1.0 / prec + 1.0 / rec);//(num > 0.0 ? f1/num : 0.0);
    pw.format("%s [summary]: LP: %.2f LR: %.2f F1: %.2f Ex: %.2f N: %d%n", str, 100*prec, 100*rec, 100*f, 100*exact/num, getNum());
//    pw.println(str + " [summary evalb]: LP: " + ((int) (10000.0 * prec)) / 100.0 + " LR: " + ((int) (10000.0 * rec)) / 100.0 + " F1: " + ((int) (10000.0 * f)) / 100.0 + " Exact: " + ((int) (10000.0 * exact / num)) / 100.0 + " N: " + getNum());
  }


  public static class CBEval extends Evalb {

    private double cb = 0.0;
    private double num = 0.0;
    private double zeroCB = 0.0;

    protected void checkCrossing(Set<Constituent<String>> s1, Set<Constituent<String>> s2) {
      double c = 0.0;
      for (Constituent<String> constit : s1) {
        if (constit.crosses(s2)) {
          c += 1.0;
        }
      }
      if (c == 0.0) {
        zeroCB += 1.0;
      }
      cb += c;
      num += 1.0;
    }

    @Override
    public void evaluate(Tree<String> t1, Tree<String> t2, PrintWriter pw) {
      Set<Constituent<String>> b1 = makeObjects(t1);
      Set<Constituent<String>> b2 = makeObjects(t2);
      checkCrossing(b1, b2);
      if (pw != null && runningAverages) {
        pw.println("AvgCB: " + ((int) (10000.0 * cb / num)) / 100.0 +
            " ZeroCB: " + ((int) (10000.0 * zeroCB / num)) / 100.0 + " N: " + getNum());
      }
    }

    @Override
    public void display(boolean verbose, PrintWriter pw) {
      pw.println(str + " AvgCB: " + ((int) (10000.0 * cb / num)) / 100.0 +
          " ZeroCB: " + ((int) (10000.0 * zeroCB / num)) / 100.0);
    }

    public CBEval(String str, boolean runningAverages) {
      super(str, runningAverages);
    }

  }

}
