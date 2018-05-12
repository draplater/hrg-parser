package jigsaw.metrics;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import jigsaw.syntax.Constituent;
import jigsaw.syntax.Tree;


public class CatEvalb extends Evalb {
  public CatEvalb(String str, boolean runningAverages) {
    super(str, runningAverages);
  }
  
  HashMap<String,Double> catGold = new HashMap<String,Double>();
  HashMap<String,Double> catGuess = new HashMap<String,Double>(); 
  HashMap<String,Double> catPrecision = new HashMap<String,Double>();
  HashMap<String,Double> catRecall = new HashMap<String,Double>();
  

  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw, double weight) {
    if(gold == null || guess == null) {
      System.err.printf("%s: Cannot compare against a null gold or guess tree!\n",this.getClass().getName());
      return;
    }
    Set<Constituent<String>> dep1 = makeObjects(guess);
    Set<Constituent<String>> dep2 = makeObjects(gold);
    final double curPrecision = precision(dep1, dep2);
    final double curRecall = precision(dep2, dep1);
    recordCat(dep1, dep2, weight);
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
      if (runningAverages) {
        pw.format(" - [average] P: %.2f R: %.2f F1: %.2f Ex: %.2f N: %d", precision2*100/pnum2, recall2*100/rnum2, cF1*100, 100*exact/num, getNum());
      }
      pw.println();
    }
  }
  public void increaseCatGold(String cat, double count) {
    double c = count + (catGold.containsKey(cat)?catGold.get(cat):0.0);
    catGold.put(cat, c);
  }
  public void increaseCatGuess(String cat, double count) {
    double c = count + (catGuess.containsKey(cat)?catGuess.get(cat):0.0);
    catGuess.put(cat, c);
  }
  public void increaseCatPrecision(String cat, double count) {
    double c = count + (catPrecision.containsKey(cat)?catPrecision.get(cat):0.0);
    catPrecision.put(cat, c);
  }
  public void increaseCatRecall(String cat, double count) {
    double c = count + (catRecall.containsKey(cat)?catRecall.get(cat):0.0);
    catRecall.put(cat, c);
  }
  public void recordCat(Set<Constituent<String>> guessSet, Set<Constituent<String>> goldSet, double weight) {
    for (Constituent<String> guess : guessSet) {
      String guesscat = guess.getLabel();
      increaseCatGuess(guesscat, weight);
      if (goldSet.contains(guess)) {
        increaseCatPrecision(guesscat, weight);
      }
    }
    for (Constituent<String> gold : goldSet) {
      String goldcat = gold.getLabel();
      increaseCatGold(goldcat, weight);
      if (guessSet.contains(gold)) {
        increaseCatRecall(goldcat, weight);
      }
    }
  }
  
  public void display(boolean verbose, PrintWriter pw) {
    double prec = precision2 / pnum2;//(num > 0.0 ? precision/num : 0.0);
    double rec = recall2 / rnum2;//(num > 0.0 ? recall/num : 0.0);
    double f = 2.0 / (1.0 / prec + 1.0 / rec);//(num > 0.0 ? f1/num : 0.0);
    pw.format("%s [summary]: LP: %.2f LR: %.2f F1: %.2f Ex: %.2f N: %d%n", str, 100*prec, 100*rec, 100*f, 100*exact/num, getNum());
    if (verbose) {
      HashMap<String, Double> catP = new HashMap<String, Double>();
      for (String cat : catGuess.keySet())
        catP.put(cat, (catPrecision.containsKey(cat)?catPrecision.get(cat):0.0) / catGuess.get(cat));
      HashMap<String, Double> catR = new HashMap<String, Double>();
      for (String cat : catGold.keySet())
        catR.put(cat, (catRecall.containsKey(cat)?catRecall.get(cat):0.0) / catGold.get(cat));
      HashMap<String, Double> catF1 = new HashMap<String, Double>();
      for (String cat : catGold.keySet()) {
        double p = catP.containsKey(cat)?catP.get(cat):0.0;
        double r = catR.containsKey(cat)?catR.get(cat):0.0;
        double f1 = (p > 0.0 && r > 0.0)?(2.0 / (1.0 / p + 1.0 / r)):0.0;
        catF1.put(cat, f1);
      }
      pw.println("Top low precision categories : ");
      List<String> catplist = TaggingCatEval.SortByValue(catP);
      for (int i = 0; i < catplist.size(); i ++) {
        String cat = catplist.get(i);
        pw.format(" %s : %.0f / %.0f = %.2f %n", cat, catPrecision.containsKey(cat)?catPrecision.get(cat):0.0, catGuess.get(cat), catP.get(cat)*100);
      }
      pw.println("Top low recall categories : ");
      List<String> catrlist = TaggingCatEval.SortByValue(catR);
      for (int i = 0; i < catrlist.size(); i ++) {
        String cat = catrlist.get(i);
        pw.format(" %s : %.0f / %.0f = %.2f %n", cat, catRecall.containsKey(cat)?catRecall.get(cat):0.0, catGold.get(cat), catR.get(cat)*100);
      }
      pw.println("Top low F1 categories : ");
      List<String> catf1list = TaggingCatEval.SortByValue(catF1);
      for (int i = 0; i < catf1list.size(); i ++) {
        String cat = catf1list.get(i);
        pw.format(" %s : P=%.0f/%.0f=%.2f R=%.0f/%.0f=%.2f F1=%.2f %n", cat,
            catPrecision.containsKey(cat)?catPrecision.get(cat):0.0, catGuess.containsKey(cat)?catGuess.get(cat):0.0,
            catP.containsKey(cat)?catP.get(cat)*100:0.0, 
            catRecall.containsKey(cat)?catRecall.get(cat):0.0, catGold.containsKey(cat)?catGold.get(cat):0.0,
            catR.containsKey(cat)?catR.get(cat)*100:0.0,
            catF1.get(cat)*100);
      }
    }
  }
}
