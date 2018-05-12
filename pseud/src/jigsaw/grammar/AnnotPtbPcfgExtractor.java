package jigsaw.grammar;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.regex.Pattern;

import jigsaw.syntax.CollinsHeadFinder;
import jigsaw.syntax.Grammar;
import jigsaw.syntax.Rule;
import jigsaw.syntax.Tree;
import jigsaw.treebank.PennTreebankReader;
import jigsaw.treebank.Trees;
import jigsaw.treebank.Trees.EmptyNodeStripper;
import jigsaw.treebank.Trees.NPTmpRetainingFunctionNodeStripper;
import jigsaw.treebank.Trees.TreeTransformer;
import jigsaw.treebank.Trees.XOverXRemover;
import jigsaw.util.StringUtils;


/** Extract automatically annotated PCFG from PTB
 * 
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Apr 19, 2011
 *
 */
public class AnnotPtbPcfgExtractor {

  public AnnotPtbPcfgExtractor() {
    
  }
  
  public static class DummyDelimAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (!tree.isLeaf() && !tree.getLabel().equals("ROOT"))
        tree.setLabel(tree.getLabel()+"@");
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
  }
  
  public static class GPAnnotator implements TreeTransformer<String> {

    public GPAnnotator(int gplevel) {
      _gplevel = gplevel;
    }
    public GPAnnotator(int gplevel, boolean tagpa) {
      this(gplevel);
      _tagpa = tagpa;      
    }
    
    private int _gplevel = 2;
    private boolean _tagpa = false;
    
    @Override
    public Tree<String> transformTree(Tree<String> tree) {
      // TODO
      ArrayList<Tree<String>> path = new ArrayList<Tree<String>>();
      path.add(tree);
      Tree<String> newtree = transformTree(path);
      return newtree;
    }
    
    public Tree<String> transformTree(List<Tree<String>> path) {
      Tree<String> oldnode = path.get(0);
      if (oldnode.isLeaf() || (oldnode.isPreTerminal() && !_tagpa)) {
        path.remove(0);
        return oldnode;
      }
      StringBuffer sb = new StringBuffer(oldnode.getLabel());
      for (int i = 1; i <= _gplevel && i < path.size(); i ++) {
        sb.append("^");
        sb.append(AnnotTmpRemover.transformLabel(path.get(i).getLabel()));
      }
      ArrayList<Tree<String>> children = new ArrayList<Tree<String>>();
      Tree<String> newnode = new Tree<String>(sb.toString());
      newnode.setChildren(children);
      for (Tree<String> child : oldnode.getChildren()) {
        path.add(0, child);
        children.add(transformTree(path));
      }
      path.remove(0);
      return newnode;
    }
  
  }
  
  public static class AnnotRemover implements TreeTransformer<String> {
  
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      String label = tree.getLabel();
      String cat = transformLabel(label);
      ArrayList<Tree<String>> children = new ArrayList<Tree<String>>();
      Tree<String> newtree = new Tree<String>(cat, children);
      for (Tree<String> child : tree.getChildren())
        children.add(transformTree(child));
      return newtree;
    }
    
    public static String transformLabel(String annotlabel) {
      return annotlabel.split("@")[0];
    }
  }
  public static class AnnotTmpRemover implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      String label = tree.getLabel();
      String cat = transformLabel(label);
      ArrayList<Tree<String>> children = new ArrayList<Tree<String>>();
      Tree<String> newtree = new Tree<String>(cat, children);
      for (Tree<String> child : tree.getChildren())
        children.add(transformTree(child));
      return newtree;
    }
    public static String transformLabel(String annotlabel) {
      return annotlabel.split("@")[0].split("-")[0];
    }
  }
  public static class UIAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return tree;
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      if (tree.getChildren().size() == 1 && !tree.getLabel().startsWith("ROOT"))
        tree.setLabel(tree.getLabel()+"-U");
      return tree;
    }
  }
  
  public static class UDTAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      return transformTreeHelper(tree, true);
    }
    private Tree<String> transformTreeHelper(Tree<String> tree, boolean nosibling) {
      if (tree.isLeaf())
        return tree;
      for (Tree<String> child : tree.getChildren())
        transformTreeHelper(child, tree.getChildren().size()==1);
      if (tree.isPreTerminal() && tree.getLabel().startsWith("DT") && nosibling)
        tree.setLabel(tree.getLabel()+"^U");
      return tree;
    }
  }
  
  /**
   * Six-way classification of IN
   * @author Yi Zhang <yzhang@coli.uni-sb.de>
   * @date Apr 28, 2011
   *
   */
  public static class INAnnotator implements TreeTransformer<String> {
    
    public Tree<String> transformTree(Tree<String> tree) {
      ArrayList<Tree<String>> path = new ArrayList<Tree<String>>();
      path.add(tree);
      return transformTreeHelper(path);
    }
    public Tree<String> transformTreeHelper(List<Tree<String>> path) {
      Tree<String> node = path.get(0); 
      if (node.isPreTerminal() && node.getLabel().startsWith("IN")) {
        String cat = node.getLabel();
        Tree<String> parent = null;
        if (path.size() > 1)
          parent = path.get(1);
        Tree<String> gparent = null;
        if (path.size() > 2)
          gparent = path.get(2);
        String pstr = (parent!=null)?parent.getLabel():"";
        String gpstr = (gparent!=null)?gparent.getLabel():"";       
        if (gpstr.startsWith("N") && (pstr.startsWith("P") || pstr.startsWith("A"))) {
          // noun postmodifier PP (or so-called ADVP like "outside India")
          cat += "-N";
        } else if (pstr.startsWith("Q") && (gpstr.startsWith("N") || gpstr.startsWith("ADJP"))) {
          // about, than, between, etc. in a QP preceding head of NP
          cat += "-Q";
        } else if (gpstr.startsWith("S@")) {
          if (pstr.startsWith("SBAR")) {
            // sentential subordinating conj: although, if, until, as, while
            cat += "-SCC";
          } else {
            // PP adverbial clause: among, in, for, after
            cat += "-SC";
          }
        } else if (pstr.startsWith("SBAR") || pstr.startsWith("WHNP")) {
          // that-clause complement of VP or NP (or whether, if complement)
          // but also VP adverbial because, until, as, etc.
          cat += "-T";
        }       
        node.setLabel(cat);
      }
      for (Tree<String> child : node.getChildren()) {
        path.add(0, child);
        transformTreeHelper(path);
      }
      path.remove(0);
      return node;
    }
  }
  
  public static class URBAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      return transformTreeHelper(tree, true);
    }
    private Tree<String> transformTreeHelper(Tree<String> tree, boolean nosibling) {
      if (tree.isLeaf())
        return tree;
      for (Tree<String> child : tree.getChildren())
        transformTreeHelper(child, tree.getChildren().size()==1);
      if (tree.isPreTerminal() && tree.getLabel().startsWith("RB") && nosibling)
        tree.setLabel(tree.getLabel()+"^U");
      return tree;
    }
  }

  public static class AUXAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      ArrayList<Tree<String>> path = new ArrayList<Tree<String>>();
      path.add(tree);
      return transformTreeHelper(path);
    }
    private Tree<String> transformTreeHelper(List<Tree<String>> path) {
      Tree<String> node = path.get(0);
      if (node.isPreTerminal()) {
        String cat = node.getLabel();
        String baseCat = AnnotTmpRemover.transformLabel(node.getLabel());
        String word = StringUtils.join(node.getYield(), " ");
        Tree<String> parent = null;
        if (path.size() > 1)
          parent = path.get(1);
        if ((baseCat.equals("VBZ") || baseCat.equals("VBP") || baseCat.equals("VBD")
            || baseCat.equals("VBN") || baseCat.equals("VBG") || baseCat.equals("VB"))) {
          if (parent != null 
              && (word.equalsIgnoreCase("'s") || word.equalsIgnoreCase("s"))) {  // a few times the apostrophe is missing!
            List<Tree<String>> sisters = parent.getChildren();
            int i = 0;
            for (; i  < sisters.size(); i ++)
              if (sisters.get(i).getLabel().startsWith("VBZ"))
                break;
            boolean annotateHave = false;  // VBD counts as an erroneous VBN!
            for (int j = i+1; j < sisters.size() && !annotateHave; j++) {
              if (sisters.get(j).getLabel().startsWith("VP")) {
                for (Tree<String> kid : sisters.get(j).getChildren()) {
                  if (kid.getLabel().startsWith("VBN") || 
                      kid.getLabel().startsWith("VBD")) {
                    annotateHave = true;
                    break;
                  }
                }
              }
            }
            if (annotateHave) {
              cat += "-HV";
              // System.out.println("Went with HAVE for " + parent);
            } else {
              cat += "-BE";
            }
          } else {
            if (word.equalsIgnoreCase("am") || word.equalsIgnoreCase("is") || 
                word.equalsIgnoreCase("are") || word.equalsIgnoreCase("was") || 
                word.equalsIgnoreCase("were") || word.equalsIgnoreCase("'m") || 
                word.equalsIgnoreCase("'re") || word.equalsIgnoreCase("be") || 
                word.equalsIgnoreCase("being") || word.equalsIgnoreCase("been") || word.equalsIgnoreCase("ai")) { // allow "ai n't"
              cat += "-BE";
            } else if (word.equalsIgnoreCase("have") || word.equalsIgnoreCase("'ve") ||
                word.equalsIgnoreCase("having") || word.equalsIgnoreCase("has") || 
                word.equalsIgnoreCase("had") || word.equalsIgnoreCase("'d")) {
              cat += "-HV";
            } else if (word.equalsIgnoreCase("do") || word.equalsIgnoreCase("did") || 
                word.equalsIgnoreCase("does") || word.equalsIgnoreCase("done") || 
                word.equalsIgnoreCase("doing")) {
              // both DO and HELP take VB form complement VP
              cat += "-DO";
            }
          }
          node.setLabel(cat);
        }
      }

      for (Tree<String> child : node.getChildren()) {
        path.add(0, child);
        transformTreeHelper(path);
      }
      path.remove(0);
      return node;
    }
  }
  
  public static class VPAnnotator implements TreeTransformer<String> {
    private CollinsHeadFinder headFinder = new CollinsHeadFinder();
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return tree;
      String cat = tree.getLabel();
      String baseCat = AnnotTmpRemover.transformLabel(cat);
      String baseTag = AnnotTmpRemover.transformLabel(headFinder.determineHead(tree).getLabel());
      if (baseCat.equals("VP")) {
        if (baseTag.equals("VBZ") || baseTag.equals("VBD") || baseTag.equals("VBP") || baseTag.equals("MD")) {
          cat += "-VBF";
        } else if (baseTag.equals("TO") || baseTag.equals("VBG") || baseTag.equals("VBN") || baseTag.equals("VB")) {
          cat += "-" + baseTag;
        }
        tree.setLabel(cat);
      }
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
  }
  public static class BNPAnnotator implements TreeTransformer<String> {
    
    public Tree<String> transformTree(Tree<String> tree) {
      if (isBaseNP(tree))
        tree.setLabel(tree.getLabel()+"-B");
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
    
    private boolean isBaseNP(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return false;
      String cat = AnnotRemover.transformLabel(tree.getLabel());
      if (!cat.startsWith("NP"))
        return false;
      for (Tree<String> child : tree.getChildren())
        if (!child.isPreTerminal())
          return false;
      return true;
    }
  }
  
  public static class DVAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return tree;
      boolean hasv = false;
      for (Tree<String> pt : tree.getPreTerminals()) {
        String tag = AnnotRemover.transformLabel(pt.getLabel());
        if (tag.startsWith("V") || tag.startsWith("MD")) {
           hasv = true;
           break;
        }
      }
      if (hasv && !tree.getLabel().startsWith("ROOT")) {
        tree.setLabel(tree.getLabel()+"-v");
      }
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
    
  }
  
  public static class RRNPAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (rightRecNP(tree))
        tree.setLabel(tree.getLabel()+"-RN");
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
    
    private static boolean rightRecNP(Tree<String> t) {
      if (t.isLeaf() || t.isPreTerminal())
        return false;
      String cat = AnnotRemover.transformLabel(t.getLabel());
      if (!cat.startsWith("NP"))
        return false;
      while (!t.isLeaf()) {
        t = t.getChildren().get(t.getChildren().size()-1);
        String str = t.getLabel();
        if (str.startsWith("NP")) {
          return true;
        }
      }
      return false;
    }
  }
  
  public static class CCAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      if (tree.isPreTerminal() && tree.getLabel().startsWith("CC")) {
        String word = StringUtils.join(tree.getYield(), " ");
        if (word.equals("and") || word.equals("or")) {
          tree.setLabel(tree.getLabel() + "-C");
        }
      }
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
  }
  public static class PercentAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      if (tree.isPreTerminal() && StringUtils.join(tree.getYield(), " ").equals("%"))
        tree.setLabel(tree.getLabel()+"-%");
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }    
  }
  public static class PossAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return tree;
      if (tree.getLabel().startsWith("NP") &&
          tree.getChildren().get(tree.getChildren().size()-1).getLabel().startsWith("POS"))
        tree.setLabel(tree.getLabel()+"-P");
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }    
  }
  
  public static class NPTMPAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      if (AnnotRemover.transformLabel(tree.getLabel()).equals("NP-TMP"))
        annotateTmpRec(tree);
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
    CollinsHeadFinder headFinder = new CollinsHeadFinder();
    private void annotateTmpRec(Tree<String> tree) {
      Tree<String> t = headFinder.determineHead(tree);
      while (t != null && !t.isLeaf()) {
        if (t.isPreTerminal()) {
          t.setLabel(t.getLabel()+"^TMP");
          return;
        }
        else
          t = headFinder.determineHead(t);
      }
    }
    
  }
  public static class KMAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      tree = (new DummyDelimAnnotator()).transformTree(tree);
      tree = (new GPAnnotator(1, true)).transformTree(tree);
      tree = (new UIAnnotator()).transformTree(tree);
      tree = (new UDTAnnotator()).transformTree(tree);
      tree = (new URBAnnotator()).transformTree(tree);
      tree = (new INAnnotator()).transformTree(tree);
      tree = (new AUXAnnotator()).transformTree(tree);
      tree = (new CCAnnotator()).transformTree(tree);
      tree = (new PercentAnnotator()).transformTree(tree); //
      tree = (new NPTMPAnnotator()).transformTree(tree);
      tree = (new PossAnnotator()).transformTree(tree); //
      tree = (new VPAnnotator()).transformTree(tree);
      tree = (new BNPAnnotator()).transformTree(tree);
      tree = (new DVAnnotator()).transformTree(tree);
      tree = (new RRNPAnnotator()).transformTree(tree);
      return tree;
    }
  }
  
  public static Grammar extractPcfgWithGP(String ptbroot, int gplevel, boolean tagpa) {
    Grammar g = new Grammar();
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(ptbroot, 200, 2172);
    g.registerS("ROOT");
    int tcount = 0;
    for (Tree<String> tree : trees) {
      tcount ++;
      tree = (new Trees.StandardTreeNormalizer()).transformTree(tree);
      GPAnnotator annot = new GPAnnotator(gplevel, tagpa);
      tree = annot.transformTree(tree);
      //System.out.println(Trees.PennTreeRenderer.render(tree));
      // transform the tree here
      for (Tree<String> node : tree) { // traverse the tree in pre-order
        if (node.isLeaf()) { // word
          continue;
        } else if (node.isPreTerminal()) { // POS
          int t = g.registerT(node.getLabel());
          int tag = g.T2i(t);
          List<String> yield = node.getYield();
          String word = StringUtils.join(yield, " ");
          g.incrSeenCount(word, tag, 1);  
          // TODO we do not have access to loc of unseen words afterwards
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
  
  public static class NPTmpRetainingTreeNormalizer implements TreeTransformer<String> {
    EmptyNodeStripper emptyNodeStripper = new EmptyNodeStripper();
    XOverXRemover<String> xOverXRemover = new XOverXRemover<String>();
    NPTmpRetainingFunctionNodeStripper functionNodeStripper = new NPTmpRetainingFunctionNodeStripper();

    public Tree<String> transformTree(Tree<String> tree) {
      tree = functionNodeStripper.transformTree(tree);
      tree = emptyNodeStripper.transformTree(tree);
      tree = xOverXRemover.transformTree(tree);
      return tree;
    }
  }
  public static Grammar extractPcfgWithAnnot(TreeTransformer<String> annot, String ptbroot) {
    Grammar g = new Grammar();
    Collection<Tree<String>> trees = PennTreebankReader.readTrees(ptbroot, 200, 2172);
    g.registerS("ROOT");
    int tcount = 0;
    for (Tree<String> tree : trees) {
      tcount ++;
      tree = (new NPTmpRetainingTreeNormalizer()).transformTree(tree);
      tree = annot.transformTree(tree);
      //System.out.println(Trees.PennTreeRenderer.render(tree));
      // transform the tree here
      for (Tree<String> node : tree) { // traverse the tree in pre-order
        if (node.isLeaf()) { // word
          continue;
        } else if (node.isPreTerminal()) { // POS
          int t = g.registerT(node.getLabel());
          int tag = g.T2i(t);
          List<String> yield = node.getYield();
          String word = StringUtils.join(yield, " ");
          g.incrSeenCount(word, tag, 1);  
          // TODO we do not have access to loc of unseen words afterwards
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
    if (args.length != 2) {
//      System.err.println("Usage: <ptb directory> <gp level> <output grammar name>");
      System.err.println("Usage: <ptb directory> <output grammar name>");
      System.exit(1);
    }
    try {
//      Grammar g = AnnotPtbPcfgExtractor.extractPcfgWithGP(args[0], Integer.parseInt(args[1]), true);
      KMAnnotator annot = new KMAnnotator();
      Grammar g = AnnotPtbPcfgExtractor.extractPcfgWithAnnot(annot, args[0]);
      g.dumpGrammar(args[1]);
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
      g.dumpGrammar(args[1]+"-normalized");
      System.out.println("Dumped normalized grammmar");
      ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(args[1]+".gr"));
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
