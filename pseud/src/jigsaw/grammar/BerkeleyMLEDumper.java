package jigsaw.grammar;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

import jigsaw.syntax.Grammar;

import jigsaw.syntax.*;

public class BerkeleyMLEDumper {

  public static void main(String [] args) {
    if (args.length != 3) {
      System.err.println("Usage: <mle-grammar> <lexicon> <dump-output.gr>");
      System.exit(1);
    }
    try {
      Grammar g = new Grammar();
      g.loadProbGrammar(args[0]);
      g.lexicon().loadLexicon(args[1]); // this is not the output of Berkeley Parser
      ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(args[2]));
      oos.writeObject(g);
    } catch (Exception ex) {
      ex.printStackTrace();
    }
    
  }
}
