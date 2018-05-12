package jigsaw.grammar;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import jigsaw.syntax.Grammar;
import jigsaw.syntax.Parser;
import jigsaw.syntax.Tree;
import jigsaw.treebank.EnglishPennTreebankParseEvaluator;
import jigsaw.treebank.PennTreebankReader;
import jigsaw.treebank.Trees;
import jigsaw.treebank.EnglishPennTreebankParseEvaluator.LabeledConstituentEval;
import jigsaw.treebank.EnglishPennTreebankParseEvaluator.RuleEval;
import jigsaw.util.StringUtils;


public class SimplePtbPcfgTester {
  
  public static String [] sentence = 
     {"But", "they", "cited", "the", "UAL", "and", "AMR", "examples", "as",
      "reasons", "to", "move", "quickly", "to", "enact", "this", "legislation", "."};
  
  public static void main(String [] args) throws IOException {
    // [-gpos] -d|-t|-sxx ptbroot grammar
    if (args.length != 3 && args.length != 4 && args.length != 5 && args.length != 6) {
      System.err.println("Usage: [-gpos] -d|-t|-sxx ptbroot grammar [gold parsed]");
      System.exit(1);
    }
    Collection<Tree<String>> trees = null;
    String target = null;
    String ptbroot = null;
    String gfile = null;
    BufferedWriter gout = null;
    BufferedWriter pout = null;
 
    boolean gpos = false;
    switch (args.length) {
    case 3:
      target = args[0];
      ptbroot = args[1];
      gfile = args[2];
      break;
    case 4:
      gpos = true;
      target = args[1];
      ptbroot = args[2];
      gfile = args[3];
      break;
    case 5:
      target = args[0];
      ptbroot = args[1];
      gfile = args[2];
      gout = new BufferedWriter(new FileWriter(args[3]));
      pout = new BufferedWriter(new FileWriter(args[4]));
      break;
    case 6:
      gpos = true;
      target = args[1];
      ptbroot = args[2];
      gfile = args[3];
      gout = new BufferedWriter(new FileWriter(args[4]));
      pout = new BufferedWriter(new FileWriter(args[5]));
      break;
    default:
      System.err.println("Invalid argument");
      System.exit(1);
    }
    if (target.equals("-s00")) {
      trees = PennTreebankReader.readTrees(ptbroot, 0, 99);
    } else if (target.equals("-t") || target.equals("-s23")) {
      trees = PennTreebankReader.readTrees(ptbroot, 2300, 2399);
    } else if (target.equals("-s01")) {
      trees = PennTreebankReader.readTrees(ptbroot, 100, 199);
    } else if (target.equals("-s24")) {
      trees = PennTreebankReader.readTrees(ptbroot, 2400, 2454);
    } else if (target.equals("-ds")) {
      trees = PennTreebankReader.readTrees(ptbroot, 2200, 2219);
    } else if (target.equals("-d")) {
      trees = PennTreebankReader.readTrees(ptbroot, 2200, 2299);
    } else if (target.equals("-s22s")) {
      trees = PennTreebankReader.readTrees(ptbroot, 2200, 2219);
    } else {
      System.err.println("Unknown target evaluation set");
      System.exit(1);
    }
    LabeledConstituentEval<String> eval = new LabeledConstituentEval<String>(Collections.singleton("ROOT"), new HashSet<String>());
    //RuleEval<String> rule_eval = new RuleEval<String>(Collections.singleton("ROOT"), new HashSet<String>());
    try {
      Parser parser = new Parser(gfile);
      long totalTime = 0;
      int totalSentences = 0;
      double totalTokens = 0;
      for (Tree<String> tree : trees) {
        Tree<String> gold = (new Trees.StandardTreeNormalizer()).transformTree(tree);
        if (gold != null) {
          totalSentences ++;
          List<String> tokens = gold.getYield();
          List<Tree<String>> pts = gold.getPreTerminals();
          totalTokens += tokens.size();
          System.out.print(StringUtils.join(tokens, " "));          
          long startTime = System.currentTimeMillis();
          Tree<String> guess = null;
          if (!gpos)
            guess = parser.parseR(gold.getYield());
          else
            guess = parser.parseGoldPos(pts);
          long endTime = System.currentTimeMillis();
          totalTime += endTime - startTime;
          System.out.println(" ("+(endTime-startTime)+" ms)");
          if (gout != null) {
            gout.write(gold.toString()+"\n");
            pout.write(guess.toString()+"\n");
//            gout.write(Trees.PennTreeRenderer.render(gold)+"\n");
//            pout.write(Trees.PennTreeRenderer.render(guess)+"\n");
          }
          eval.evaluate(guess, gold);
          //rule_eval.evaluate(guess, gold);
        }
      }
      eval.display(true);
      //rule_eval.display(true);
      
      System.out.println("Average speed : "+(totalTokens/totalTime *1000)+" toks/s");
      if (args.length == 6) {
        gout.close();
        pout.close();
      }
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
