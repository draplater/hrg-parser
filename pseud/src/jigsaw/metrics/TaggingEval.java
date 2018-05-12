package jigsaw.metrics;

import java.io.PrintWriter;
import java.util.*;

import jigsaw.syntax.Constituent;
import jigsaw.syntax.Tree;


public class TaggingEval implements SetEvaluator {

  private double totalToks = 0.0;
  private double totalCorrect = 0.0;
  
  protected final String str;
  protected final boolean runningAverages;

  public TaggingEval(String str) {
    this(str, true);
  }

  public TaggingEval(String str, boolean runningAverages) {
    this.str = str;
    this.runningAverages = runningAverages;
  }

  public void evaluate(Tree<String> guess, Tree<String> gold) {
    evaluate(guess, gold, new PrintWriter(System.out, true));
  }

  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw) {
    evaluate(guess, gold, pw, 1.0);
  }

  @Override
  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw, double weight) {
    List<Tree<String>> guesspts = guess.getPreTerminals();
    List<Tree<String>> goldpts = gold.getPreTerminals();
    
    if (guesspts.size() != goldpts.size())
      pw.println("Warning: yield length differs: Guess " + guesspts.size() + " / Gold" + goldpts.size());
    
    double currCorrect = 0.0;
    for (int i = 0; i < guesspts.size(); i ++){
      String guesspos = guesspts.get(i).getLabel();
      String goldpos = goldpts.get(i).getLabel();
      if (guesspos.equals(goldpos)) 
        currCorrect += 1.0;
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
  
  public void display(boolean verbose) {
    display(verbose, new PrintWriter(System.out, true));
  }
  
  @Override
  public void display(boolean verbose, PrintWriter pw) {
    pw.format("%s [summary] Acc: %.2f T#: %d\n", str, totalCorrect/totalToks*100, (int)totalToks);
//    pw.println(str + " [summary tag] Acc: " + ((int) (10000.0 * totalCorrect / totalToks)) / 100.0 + " T#: " + (int)totalToks);
  }
}
