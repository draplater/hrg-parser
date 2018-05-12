package jigsaw.grammar;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;

import fig.basic.IOUtils;

import jigsaw.syntax.*;
import jigsaw.treebank.*;
import jigsaw.treebank.Trees.TreeTransformer;
import jigsaw.util.StringUtils;

/** Extract automatically annotated PCFG from PTB
 * 
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Apr 19, 2011
 *
 */
public class AnnotRedwoodsPcfgExtractor {

  public AnnotRedwoodsPcfgExtractor() {
    
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
        sb.append(AnnotRemover.transformLabel(path.get(i).getLabel()));
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
  public static class LEReplacer implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      if (tree.isPreTerminal()) {
        if (_lemap.containsKey(tree.getLabel())) {
          tree.setLabel(_lemap.get(tree.getLabel()));
        }
      }
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
    
    private HashMap<String,String> _lemap = new HashMap<String,String>();
    
    public LEReplacer(List<File> lexfiles) throws IOException {
      for (File lex : lexfiles) {
        BufferedReader br = new BufferedReader(new FileReader(lex));
        String line = null;
        Pattern r = Pattern.compile("^([^ ]+)\\s*:=\\s*([^ ]+)\\s*&");
        while ((line=br.readLine()) != null) {
          Matcher m = r.matcher(line);
          if (m.find()) {
            _lemap.put(m.group(1), m.group(2));
          }
        }
      }
    }
  }
  
  public static class LHCAnnotator implements TreeTransformer<String> {
    private ERGHeadFinder _headFinder = new ERGHeadFinder();
    
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return tree;
      if (!tree.getLabel().startsWith("ROOT")) {
        Tree<String> lexhead = _headFinder.determineLexicalHead(tree);
        if (lexhead != null) {
          String cat = AnnotRemover.transformLabel(lexhead.getLabel());
          cat = ERGHeadFinder.getLECategory(cat);
          tree.setLabel(tree.getLabel()+":"+cat);
        }
      }
      for (Tree<String> child : tree.getChildren())
        transformTree(child);
      return tree;
    }
  }
  
  public static class MPHAnnotator implements TreeTransformer<String> {

    public Tree<String> transformTree(Tree<String> tree) { 
      Tree<String> newtree = null;
      if (tree.getYield().size() == 1 && 
          ERGHeadFinder.isLR(tree.getLabel())) {
        newtree = transformTreeHelper(tree);      
      } else {
        newtree = new Tree<String>(tree.getLabel());
        ArrayList<Tree<String>> newchildren = new ArrayList<Tree<String>>();
        for (Tree<String> child : tree.getChildren())
          newchildren.add(transformTree(child));
        newtree.setChildren(newchildren);
      }
      return newtree;
    }
    private Tree<String> transformTreeHelper(Tree<String> tree) {
      Tree<String> t = tree;
      ArrayList<String> labels = new ArrayList<String>();
      while (!t.isLeaf()) {
        labels.add(0, t.getLabel());
        t = t.getChildren().get(0);
      }
      Tree<String> newtree = new Tree<String>(StringUtils.join(labels,"&"), 
                                              new ArrayList<Tree<String>>());
      newtree.getChildren().add(t);
      return newtree;
    }
  }
  
  public static class MPHUnannotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      Tree<String> newtree = null;
      if (tree.isPreTerminal()) {
        Tree<String> terminal = tree.getChildren().get(0);
        Tree<String> top = null;
        Tree<String> previous = terminal;
        String [] chain = tree.getLabel().split("&");
        for (String rulename : chain) {
          if (!rulename.equals("")) {
            top = new Tree<String>(rulename, new ArrayList<Tree<String>>());
            top.getChildren().add(previous);
            previous = top;
          }
        }
        return top;
      } else {
        newtree = new Tree<String>(tree.getLabel(), new ArrayList<Tree<String>>());
        for (Tree<String> child : tree.getChildren())
          newtree.getChildren().add(transformTree(child));
      }
      return newtree;
    }
  }
  
  
  public static class POSInserter implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf() || tree.isPreTerminal())
        return tree;
      ArrayList<Tree<String>> newchildren = new ArrayList<Tree<String>>();
      for (Tree<String> child : tree.getChildren()) {
        if (child.isPreTerminal()) {
          String letype = AnnotRemover.transformLabel(child.getLabel());
          Tree<String> post = new Tree<String>(Grammar.MODPREFIX+"P-"+ERGHeadFinder.getLECatAndComps(letype));
          ArrayList<Tree<String>> postkids = new ArrayList<Tree<String>>();
          postkids.add(child);
          post.setChildren(postkids);
          newchildren.add(post);
        } else {
          transformTree(child);
          newchildren.add(child);
        }
      }
      tree.setChildren(newchildren);
      return tree;
    }    
  }

  public static class LECollapser implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      if (tree.isLeaf())
        return tree;
      if (tree.isPreTerminal()) {
        String letype = AnnotRemover.transformLabel(tree.getLabel());
        tree.setLabel(ERGHeadFinder.getLECatAndComps(letype));
      }
      for (Tree<String> child : tree.getChildren()) {
        transformTree(child);
      }
      return tree;
    }
  }
  
  
  public static class ZAnnotator implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
//      tree = (new LECollapser()).transformTree(tree);
      tree = (new MPHAnnotator()).transformTree(tree);
      tree = (new DummyDelimAnnotator()).transformTree(tree);
//      tree = (new POSInserter()).transformTree(tree); // insert pre-preterminal POS nodes
      tree = (new GPAnnotator(1, false)).transformTree(tree); // good with gp1 and no tagpa
      //tree = (new LHCAnnotator()).transformTree(tree); // hurts coverage already
     return tree;
    }
  }
  public static class StandardNormalizer implements TreeTransformer<String> {
    LEReplacer _lereplacer = null;
    ERGHeadFinder.PUNCTForker _punctforker = null;
    public StandardNormalizer() {
      try {
    	  // wsun: I don't know what is lexicon_all.tdl. I suppose it is the lexicon of the erg. :(
//        File lexfile = new File("erg/lexicon_all.tdl");
    	  File lexfile = new File("lexicon.tdl");
        ArrayList<File> lexfiles = new ArrayList<File>();
        lexfiles.add(lexfile);
        _lereplacer = new LEReplacer(lexfiles);
        _punctforker = new ERGHeadFinder.PUNCTForker();
      } catch (IOException ex) {
        ex.printStackTrace();
      }
    }
    public Tree<String> transformTree(Tree<String> tree) {
      tree = _lereplacer.transformTree(tree);
      tree = _punctforker.transformTree(tree);
      return tree;
    }
  }
  public static Grammar extractPcfgWithAnnot(Collection<Tree<String>> trees, TreeTransformer<String> annot, boolean debug, PrintWriter pw) {
    Grammar g = new Grammar(new RedwoodsLexicon());
//    Collection<Tree<String>> trees = RedwoodsTreebankReader.readLogonTrainingTrees(tsdbhome);
//    Collection<Tree<String>> trees = RedwoodsTreebankReader.readWSTrainingTrees(tsdbhome);
    g.registerS("ROOT");
    StandardNormalizer normalizer = new StandardNormalizer();
    int tcount = 0;
    for (Tree<String> tree : trees) {
      tcount ++;
      if (tcount % 100 == 0)
        System.out.print(".");
      try {
      tree = normalizer.transformTree(tree);
      pw.print(tree.toString());
      pw.print("\t");
      // transform the tree here      
      tree = annot.transformTree(tree);
      pw.print(tree.toString());
      pw.println();
      if (debug)
        System.out.println(RedwoodsTreeReader.RedwoodsTreeIndentedRenderer.render(tree));
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
      } catch (Exception ex) {
        System.err.print("!");
//        System.err.println(RedwoodsTreeReader.RedwoodsTreeIndentedRenderer.render(tree));
        tcount --;
      }
    }
    
    System.out.println(tcount + " trees extracted.");

    return g;
  }
  
  public static void main(String [] args) {
    if (args.length < 4 || args.length > 5) {
//      System.err.println("Usage: <tsdbhome directory> <gp level> <output grammar name>");
      System.err.println("Usage: [-v] -t <ws|logon|logon+ws|wws> <tsdbhome directory> <output grammar name>");
      System.exit(1);
    }
    try {
//      Grammar g = AnnotPtbPcfgExtractor.extractPcfgWithGP(args[0], Integer.parseInt(args[1]), true);      
      Options options = new Options();
      options.addOption("v", false, "verbose mode");
      Option treebank = OptionBuilder.withArgName("trainingset").isRequired().hasArg().withDescription("use given treebank for training" ).create("t");
      options.addOption(treebank);
      CommandLineParser parser = new PosixParser();
      CommandLine cmd = parser.parse(options, args);
      args = cmd.getArgs();
      
      Collection<Tree<String>> trees = null;
      String trainingset = cmd.getOptionValue("t");
      if (trainingset == null) {
        System.err.println("Usage: [-v] -t <ws|logon|logon+ws|ww|wws> <tsdbhome directory> <output grammar name>");
        System.exit(1);
      }
      if (trainingset.equals("logon"))
        trees = RedwoodsTreebankReader.readLogonTrainingTrees(args[0]);
      else if (trainingset.equals("ws")) 
        trees = RedwoodsTreebankReader.readWSTrainingTrees(args[0]);
      else if (trainingset.equals("logon+ws"))
        trees = RedwoodsTreebankReader.readLogonAndWSTrainingTrees(args[0]);
      else if (trainingset.equals("ww"))
        trees = RedwoodsTreebankReader.readWWTrainingTrees(args[0]);
      else if (trainingset.equals("wws"))
        trees = RedwoodsTreebankReader.readWWSmallTrainingTrees(args[0]);
      else {
        System.err.println("Unknown training set " + trainingset + " specified. Abort.");
        System.exit(1);
      }
      ZAnnotator annot = new ZAnnotator();
      PrintWriter tPw = IOUtils.openOutEasy("/tmp/normalizedERGTrees");
      Grammar g = AnnotRedwoodsPcfgExtractor.extractPcfgWithAnnot(trees, annot, cmd.hasOption("v"), tPw);
      tPw.close();
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
      System.out.println("Dumped normalized grammar");
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
