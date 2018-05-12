package jigsaw.metrics;

import java.io.PrintWriter;

import jigsaw.syntax.Tree;


public interface SetEvaluator {
  public void evaluate(Tree<String> guess, Tree<String> gold, PrintWriter pw, double weight);
  public void display(boolean verbose, PrintWriter pw);
}
