package jigsaw.metrics;

import java.io.PrintWriter;
import java.util.*;

import jigsaw.syntax.Tree;
import jigsaw.util.Pair;


public class TaggingCatEval extends TaggingEval {

  public TaggingCatEval(String str, boolean runningAverages) {
    super(str, runningAverages);
  }

  private double totalToks = 0.0;
  private double totalCorrect = 0.0;
  private HashMap<String, Double> catCorrect = new HashMap<String, Double>();
  private HashMap<String, Double> catCount = new HashMap<String, Double>();
  private HashMap<Pair<String,String>, Double> errorCount = new HashMap<Pair<String,String>, Double>();
  
  @Override
  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw, double weight) {
    List<Tree<String>> guesspts = guess.getPreTerminals();
    List<Tree<String>> goldpts = gold.getPreTerminals();
    
    if (guesspts.size() != goldpts.size())
      pw.println("Warning: yield length differs: Guess " + guesspts.size() + " / Gold" + goldpts.size());
    
    double currCorrect = 0.0;
    for (int i = 0; i < goldpts.size(); i ++) {
      String guesspos = guesspts.get(i).getLabel();
      String goldpos = goldpts.get(i).getLabel();
      double c = weight + (catCount.containsKey(goldpos)?catCount.get(goldpos):0.0);
      catCount.put(goldpos, c);
      if (guesspos.equals(goldpos)) {
        currCorrect += 1.0;
        c = weight + (catCorrect.containsKey(goldpos)?catCorrect.get(goldpos):0.0);
        catCorrect.put(goldpos, c);
      } else {
        Pair<String,String> p = new Pair(goldpos, guesspos);
        c = weight + (errorCount.containsKey(p)?errorCount.get(p):0.0);
        errorCount.put(p, c);
      }
    }
    double currAcc = currCorrect / goldpts.size();
    totalCorrect += currCorrect * weight;
    totalToks += goldpts.size() * weight;
    pw.format("%s [current] Acc: %.2f" , str, currAcc * 100);
    if (runningAverages) {
      pw.format(" - [average] Acc: %.2f", totalCorrect/totalToks*100);
    }
    pw.println();
  }
  
  @Override
  public void display(boolean verbose, PrintWriter pw) {   
    pw.format("%s [summary] Acc: %.2f T#: %d%n", str, totalCorrect/totalToks*100, (int)totalToks);
    if (verbose) {
      HashMap<String,Double> catAccuracy = new HashMap<String,Double>();
      for (String pos : catCorrect.keySet()) {
        catAccuracy.put(pos, catCorrect.get(pos)/ catCount.get(pos));
      }
      List<String> catacclist = SortByValue(catAccuracy);
      pw.println("Top-10 low accuracy tags : ");
      for (int i = 0; i < 10 && i < catacclist.size(); i ++) {
        String pos = catacclist.get(i);
        pw.format(" %s : %.0f / %.0f = %.2f %n", pos, catCorrect.get(pos), catCount.get(pos), catAccuracy.get(pos)*100);
      }
      pw.println("Top-10 tagging errors : ");
      List<Pair<String,String>> caterrlist = SortByValue(errorCount);
      for (int i = 0; i < 10 && i < caterrlist.size(); i ++) {
        Pair<String,String> pair = caterrlist.get(caterrlist.size()-i-1);
        pw.format(" %s -> %s %.0f%n", pair.getFirst(), pair.getSecond(), errorCount.get(pair));
      }
    }
  }
  
  
  public static <K, V> List<K> SortByValue(final Map<K,V> m) {
    List<K> keys = new ArrayList<K>();
    keys.addAll(m.keySet());
    Collections.sort(keys, new Comparator<K>() {
      public int compare(Object o1, Object o2) {
        Object v1 = m.get(o1);
        Object v2 = m.get(o2);
        if (v1 == null) {
          return (v2 == null) ? 0 : 1;
        }
        else if (v1 instanceof Comparable) {
          return ((Comparable) v1).compareTo(v2);
        }
        else {
          return 0;
        }
      }
    });
    return keys;
  }
  
}
