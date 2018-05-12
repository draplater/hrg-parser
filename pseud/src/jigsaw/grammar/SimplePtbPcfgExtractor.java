package jigsaw.grammar;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Collection;
import java.util.List;

import jigsaw.syntax.Grammar;
import jigsaw.syntax.Rule;
import jigsaw.syntax.Tree;
import jigsaw.treebank.PennTreebankReader;
import jigsaw.treebank.Trees;
import jigsaw.util.StringUtils;

import jigsaw.syntax.*;
import jigsaw.treebank.*;
import jigsaw.util.*;

/**
 * Extracting PCFG from PTB
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Apr 12, 2011
 *
 */
public class SimplePtbPcfgExtractor {

  public SimplePtbPcfgExtractor() {
    
  }
  
  public static Grammar extractPcfg(String ptbroot, int begin, int end) {
    Grammar g = new Grammar();
    // TODO
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(ptbroot, begin, end);
    g.registerS("ROOT");
    int tcount = 0;
    for (Tree<String> tree : trees) {
      tcount ++;
      tree = (new Trees.StandardTreeNormalizer()).transformTree(tree);
      //System.out.println(Trees.PennTreeRenderer.render(tree));
      for (Tree<String> node : tree) { // traverse the tree in pre-order
        if (node.isLeaf()) { // word
          continue;
        } else if (node.isPreTerminal()) { // POS
          int t = g.registerT(node.getLabel());
          int tag = g.T2i(t);
          List<String> yield = node.getYield();
          String word = StringUtils.join(yield, " ");
          g.incrSeenCount(word, tag, 1);  // TODO we do not have access to loc of unseen words afterwards
        } else { // phrase
          int lhs = g.registerNT(node.getLabel());
          int [] rhs = new int[node.getChildren().size()];
          int i = 0;
          for (Tree<String> c : node.getChildren()) {
            if (c.isPreTerminal()) {
              rhs[i] = g.registerT(c.getLabel());
            } else {
              rhs[i] = g.registerNT(c.getLabel());
            }
            i ++;
          }
          Rule r = new Rule(lhs, rhs);
          g.incrSeenCount(r);
        }
      }
    }
    
    System.out.println(tcount + " trees extracted.");

    return g;
  }
  
  public static void main(String [] args) {
    if (args.length != 2 && args.length != 4) {
      System.err.println("Usage: <ptb directory> <output grammar name>");
      System.err.println("       <begin> <end> <ptb directory> <output grammar name>");
      
      System.exit(1);
    }
    try {
      String ptbroot = null;
      int begin = 200;
      int end = 2172;
      String gname = null;
      switch (args.length) {
      case 2:
        ptbroot = args[0];
        gname = args[1];
        break;
      case 4:
        begin = Integer.parseInt(args[0]);
        end = Integer.parseInt(args[1]);
        ptbroot = args[2];
        gname = args[3];
        break;
      default:
        System.err.println("Invalid arguments.");
        System.exit(1);
      }
      Grammar g = SimplePtbPcfgExtractor.extractPcfg(ptbroot, begin, end);
      g.dumpGrammar(gname);
      System.out.println("Dumped");
      System.out.println();
      System.out.println("Unnormalaized grammar:");
      System.out.println("Rules : "+g.rules().size());
      System.out.println("Non-Terminals : "+g.nts().size());
      System.out.println("Preterminal Tags : "+g.ts().size());
      g.normalize();
      g.buildProb();
      g.lexicon().buildUWModel();
      System.out.println("Normalized");
      g.dumpGrammar(gname+"-normalized");
      System.out.println("Dumped normalized grammmar");
      ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(gname+".gr"));
      oos.writeObject(g);
      System.out.println("Serialized");
      System.out.println();
      System.out.println("Normalaized grammar:");
      System.out.println("Rules : "+g.rules().size());
      System.out.println("Non-Terminals : "+g.nts().size());
      System.out.println("Preterminal Tags : "+g.ts().size());
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
