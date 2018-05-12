package jigsaw.syntax;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.Writer;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import jigsaw.treebank.Trees;
import jigsaw.util.StringUtils;


/**
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 26, 2010
 *
 */
public class Parser {

  public Parser(Grammar g, Lexicon l) {
    _g = g;
    _l = l;
  }
  public Parser(Grammar g) {
    _g = g;
    _l = g.lexicon();
  }
  public Parser(String grammarfile) throws IOException, ClassNotFoundException {
    _g = (Grammar)(new ObjectInputStream(new FileInputStream(grammarfile))).readObject();
    System.out.println("done.");
    _l = _g.lexicon();
  }
  
  public Tree<String> parse(String sentence) {
    List<String> inputs = tokenize(sentence);
    return parse(inputs);
  }
  
  /** parsing with gold POSs */
  public Tree<String> parseGoldPos(List<Tree<String>> sentence) {
    _stats.totalSentences ++;
    long startTime = System.currentTimeMillis();
    lexparseGoldPos(sentence);
    recognize();
    Tree<String> result = null;
    if (_chart.recognized()) {
      _stats.successSentences ++;
      _stats.successTokens += _stats.lastTokens;
      viterbi();
      result = buildVParse();
    } else {
      viterbi();      
      result = new Tree<String>(_g.Start());
      List<Tree<String>> children = partialParse();
      result.setChildren(children);
    }
    result = (new Trees.GrammarInternalNodeStripper()).transformTree(result);
    long endTime = System.currentTimeMillis();
    _stats.lastParseTime = endTime - startTime;
    _stats.totalParseTime += _stats.lastParseTime;
    return result;
  }
  public Tree<String> parse(List<String> sentence) {
    return parse(sentence, true);
  }
  public Tree<String> parse(List<String> sentence, boolean word_lattice) {
    _stats.totalSentences ++;
    long startTime = System.currentTimeMillis();
    if (word_lattice)
      lexparse(sentence);
    else
      lexparseNoLattice(sentence);
    recognize();
    Tree<String> result = null;
    if (_chart.recognized()) { 
      _stats.successSentences ++;
      _stats.successTokens += _stats.lastTokens;
      viterbi();
      result = buildVParse();
      result = (new Trees.GrammarInternalNodeStripper()).transformTree(result);
    }
    long endTime = System.currentTimeMillis();
    _stats.lastParseTime = endTime - startTime;
    _stats.totalParseTime += _stats.lastParseTime;
    return result;
  }
  
  public Tree<String> parseR(List<String> sentence) {
    return parseR(sentence, true);
  }
  /** robust version of the parser, never returns null */
  public Tree<String> parseR(List<String> sentence, boolean word_lattice) {
    _stats.totalSentences ++;
    long startTime = System.currentTimeMillis();
    if (word_lattice)
      lexparse(sentence);
    else
      lexparseNoLattice(sentence);
    recognize();
    Tree<String> result = null;
    if (_chart.recognized()) {
      _stats.successSentences ++;
      _stats.successTokens += _stats.lastTokens;
      viterbi();
      result = buildVParse();
    } else {
      viterbi();
      result = new Tree<String>(_g.Start());
      List<Tree<String>> children = partialParse();
      result.setChildren(children);
    }
    result = (new Trees.GrammarInternalNodeStripper()).transformTree(result);
    long endTime = System.currentTimeMillis();
    _stats.lastParseTime = endTime - startTime;
    _stats.totalParseTime += _stats.lastParseTime;
    return result;
  }

  public List<Tree<String>> partialParse() {
    ArrayList<Tree<String>> partials = new ArrayList<Tree<String>>();
    // TODO
    for (short i = 0; i < _ichart.maxpos(); i ++) {
      double maxscore = Double.NEGATIVE_INFINITY;
      Tree<String> tnode = new Tree<String>(_inputs.get(i));
      ArrayList<Tree<String>> children = new ArrayList<Tree<String>>();
      children.add(tnode);
      Tree<String> ptnode = null;
      for (int tag = 0; tag < _ichart.maxS(); tag ++) {
        if (_ichart.get(i, (short)(i+1), tag) && 
            logLP(i, (short)(i+1), tag) > maxscore) {
          ptnode = new Tree<String>(_g.T(_g.i2T(tag)));
          maxscore = logLP(i, (short)(i+1), tag);
        }
      }
      ptnode.setChildren(children);
      partials.add(ptnode);
    }
    return partials;
  }
  
  public static List<String> tokenize(String sentence) {
    sentence = sentence.replace(",", " ,");
    sentence = sentence.replace(".", " .");
    sentence = sentence.replace("!", " !");
    sentence = sentence.replace("?", " ?");
    sentence = sentence.replace("'", " '");
    String [] toks = sentence.split("\\s+");
    Vector<String> tokenized = new Vector<String>(); 
    for (String tok : toks)
      tokenized.add(tok);
    return tokenized;
  }
  
  public void recognize() {   
    short maxpos = (short)_chart.maxpos();
    int N = _g.nts().size();
    BitSet vec = new BitSet(N);
    for (short end = 1; end <= maxpos; end ++) {
      /* @{
       * the following block is to be moved to lexparse() to allow integration with a lexicon.
       * the recognizer will then operate on a partially filled chart, i.e. dag input.  
       *//*
      vec.clear();
      for (Rule r : _g.trules()) {
        if ( r.rhs()[0] == inputs.get(end-1)) {
          vec.or(_chart.chainvec(r.lhs()));
        }
      }
      _chart.or((short)(end-1), end, vec);*/
      /* end of the block
       * @} */
      for (short begin = (short)(end - 2); begin >= 0; begin --) {
        vec.clear();
        for (int A = 0; A < N; A ++) {
          if (!vec.get(A)) {
            //derivable()
            for (Rule r : _g.birules(A)) {
              BitSet vec1 = _chart.getVectorByStart(begin, (short)(end - begin - 1), r.rhs()[0]);
              BitSet vec2 = _chart.getVectorByEnd(end, (short)(end - begin - 1), r.rhs()[1]);
              vec1.and(vec2);
              if (!vec1.isEmpty()) {
                vec.or(_chart.chainvec(A));
                break;
              }
            }
          }
        }
        _chart.or(begin, end, vec);
      }
    }
    _stats.lastCFR = _chart.fillrate();
    _stats.totalCFR += _stats.lastCFR;
  }
  
  public void initlogLP() {
    _logLP = new double [_ichart.size()]; 
    for (int i = 0; i < _logLP.length; i ++)
      _logLP[i] = Double.NaN;
  }
  public void logLP(short start, short end, int tag, double logp) {
    _logLP[tag*_ichart.pagesize()+end*(end-1)/2+start] = logp;
  }
  public double logLP(short start, short end, int tag) {
    return _logLP[tag*_ichart.pagesize()+end*(end-1)/2+start];
  }
  public void initlogP() {
    _logP = new double [_chart.size()]; 
    for (int i = 0; i < _logP.length; i ++)
      _logP[i] = Double.NaN;
  }
  public void logP(short start, short end, int nt, double logp) {
    _logP[nt*_chart.pagesize()+end*(end-1)/2+start] = logp;
  }
  public double logP(short start, short end, int nt) {
    return _logP[nt*_chart.pagesize()+end*(end-1)/2+start];
  }
  
  /** lexparse with god POS inputs, implicitly no word-lattice */
  public void lexparseGoldPos(List<Tree<String>> inputs) {
    short maxpos = (short)inputs.size();
    _stats.lastTokens = maxpos;
    _stats.totalTokens += _stats.lastTokens;
    _inputs = new Vector<String>();
    _chart = new BitRecChart(_g, maxpos);
    _ichart = new BitRecChart(maxpos, _g.ts().size()-1);
    initlogLP();
    int N = _g.nts().size();
    BitSet vec = new BitSet(N);
    int i = 0;
    for (Tree<String> pt : inputs) {
      String word = StringUtils.join(pt.getYield()," ");
      _inputs.add(word);
      int t = _g.T(pt.getLabel());
      int ti = _g.T2i(t);
      Iterator<Integer> tagiter = null;
      boolean knownGPOS = false; 
      if (ti != -1) {
        knownGPOS = true;
        ArrayList<Integer> taglist = new ArrayList<Integer>();
        taglist.add(ti);
        tagiter = taglist.iterator();
      } else {
        tagiter = _l.tagIteratorByWord(word, i, true);
      }
      vec.clear();
      while (tagiter.hasNext()) {
        ti = tagiter.next();
        t = _g.T(_l.tag(ti));
        _ichart.set((short)i, (short)(i+1), ti);
        logLP((short)i, (short)(i+1), ti, knownGPOS?0.0:_l.score(word, ti, i));
        for (Rule r : _g.trules()) {
          if (r.rhs()[0] == t) {
            vec.or(_chart.chainvec(r.lhs()));
          }
        }
      }
      _chart.or((short)i, (short)(i+1), vec);
      i ++;
    }
  }
  /** build word-lattice based on the input token sequence*/
  public void lexparse(List<String> inputs) {
    short maxpos = (short)inputs.size();
    _stats.lastTokens = maxpos;
    _stats.totalTokens += _stats.lastTokens;
    _inputs = new Vector<String>(inputs);
    _chart = new BitRecChart(_g, maxpos);
    _ichart = new BitRecChart(maxpos, _g.ts().size()-1);
    initlogLP();
    int N = _g.nts().size();
    BitSet vec = new BitSet(N);
    BitSet covered = new BitSet(maxpos); // keep track of lexical gaps
    // 1. lookup lexicon without guessing for unknowns
    for (short start = 0; start < maxpos; start ++) {
      StringBuffer word = new StringBuffer();
      for (short end = (short)(start + 1); end <= maxpos; end ++) {
        if (word.length() > 0)
          word.append(" ");
        word.append(_inputs.get(end-1));
        Iterator<Integer> iter = _l.tagIteratorByWord(word.toString(), start, false);
        if (iter.hasNext())
          covered.set(start, end);
        vec.clear();
        while (iter.hasNext()) { // iterate through the possible tags
          int tag = iter.next();
          int t = _g.T(_l.tag(tag)); // TODO this could be avoided when the grammar and lexicon share the same terminal symbol table
          _ichart.set(start, end, tag); // TODO this assumes a shared terminal symbol table
          logLP(start, end, tag, _l.score(word.toString(), tag, start));
          for (Rule r : _g.trules()) { // find rules that can produce the tag
            if (r.rhs()[0] == t) {
              vec.or(_chart.chainvec(r.lhs()));
            }
          }
        }
        _chart.or((short)start, (short)end, vec);
      }
    }
    // 2. fill in guessed tags for lexical gaps
    for (short i = 0; i < maxpos; i ++) {
      if (!covered.get(i)) {
        vec.clear();
        Iterator<Integer> iter = _l.tagIteratorByWord(_inputs.get(i), i, true);
        while (iter.hasNext()) {
          int tag = iter.next();
          int t = _g.T(_l.tag(tag));
          _ichart.set(i, (short)(i+1), tag);
          logLP(i, (short)(i+1), tag, _l.score(_inputs.get(i), tag, i));
          for (Rule r : _g.trules()) {
            if (r.rhs()[0] == t) {
              vec.or(_chart.chainvec(r.lhs()));
            }
          }
        }
        _chart.or(i, (short)(i+1), vec);
      }
    }
  }
  /** Lexical parsing without word-lattice (gold tokenization) */
  public void lexparseNoLattice(List<String> inputs) {
    short maxpos = (short)inputs.size();
    _stats.lastTokens = maxpos;
    _stats.totalTokens += _stats.lastTokens;
    _inputs = new Vector<String>(inputs);
    _chart = new BitRecChart(_g, maxpos);
    _ichart = new BitRecChart(maxpos, _g.ts().size()-1);
    initlogLP();
    int N = _g.nts().size();
    BitSet vec = new BitSet(N);
    for (short start = 0; start < maxpos; start ++) {     
      // 1. lookup lexicon without guessing for unknowns
      Iterator<Integer> iter = _l.tagIteratorByWord(inputs.get(start), start, false);
      // 2. fill in guessed tags for lexical gaps
      if (!iter.hasNext())
        iter = _l.tagIteratorByWord(_inputs.get(start), start, true);
      vec.clear();
      while (iter.hasNext()) { // iterate through the possible tags
        int tag = iter.next();
        int t = _g.T(_l.tag(tag)); // TODO this could be avoided when the grammar and lexicon share the same terminal symbol table
        _ichart.set(start, (short)(start+1), tag); // TODO this assumes a shared terminal symbol table
        logLP(start, (short)(start+1), tag, _l.score(inputs.get(start), tag, start));
        for (Rule r : _g.trules()) { // find rules that can produce the tag
          if (r.rhs()[0] == t) {
            vec.or(_chart.chainvec(r.lhs()));
          }
        }
      }
      _chart.or((short)start, (short)(start+1), vec);
    }
  }
  /*
  public void viterbi(Vector<Integer> inputs) {
    _chart = _chart.filter();
    initlogP();
    for (short end = 1; end <= _chart.maxpos(); end ++) {
      for (Rule r: _g.trules()) {
        if (r.rhs()[0] == inputs.get(end - 1) &&
            _chart.get((short)(end - 1), end, r.lhs()))
          addProb((short)(end-1), (short)0, end, r);   
      }
      for (short start = (short)(end - 2); start >= 0; start --) {
        for (int A = 0; A <= _chart.maxS(); A ++) {
          if (_chart.get(start, end, A)) {
            for (Rule r : _g.birules(A)) {
              BitSet vec1 = _chart.getVectorByStart(start, (short)(end - start - 1), r.rhs()[0]);
              BitSet vec2 = _chart.getVectorByEnd(end, (short)(end - start - 1), r.rhs()[1]);
              vec1.and(vec2);
              for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1))
                addProb(start, (short)(start+m+1), end, r);
            }
          }
        }
      }
    }
  }*/
  public void viterbi() {
    //long startTime = System.currentTimeMillis();
    //_chart = _chart.filter(); // whether to filter or not depends on the actual grammar, try it out before deciding
    //long endTime = System.currentTimeMillis();
    //System.out.println("filter() : "+(endTime-startTime)+" ms");
    //System.out.println("Chart filling rate (after filtering): "+_chart.fillrate());
    //startTime = System.currentTimeMillis();
    initlogP();
    //endTime = System.currentTimeMillis();
    //System.out.println("initlogP() : "+(endTime-startTime)+" ms");
    for (short end = 1; end <= _chart.maxpos(); end ++) {
      for (short start = (short)(end - 1); start >= 0; start --) {
        for (Rule r : _g.trules()) {
          int ti = _g.T2i(r.rhs()[0]);
          if (_chart.get(start, end, r.lhs()) &&
              _ichart.get(start, end, ti)) {
            addProb(start, (short)0, end, r); // addProb() also adds probs for chain rules
          }
        }
      }
      for (short start = (short)(end - 2); start >= 0; start --) {
        for (int A = 0; A <= _chart.maxS(); A ++) {
          if (_chart.get(start, end, A)) {
            for (Rule r : _g.birules(A)) {
              BitSet vec1 = _chart.getVectorByStart(start, (short)(end - start - 1), r.rhs()[0]);
              BitSet vec2 = _chart.getVectorByEnd(end, (short)(end - start - 1), r.rhs()[1]);
              vec1.and(vec2);
              for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1))
                addProb(start, (short)(start+m+1), end, r);
            }
          }
        }
      }
    }
  }

  public void addProb(short start, short mid, short end, Rule r) {
    double logp = _g.logProb(r.id());
    if (r.arity() == 1 && r.rhs()[0] < 0) { // lexical probability
      // TODO
      int ti = _g.T2i(r.rhs()[0]);
      logp += logLP(start, end, ti);
    } else if (r.arity() == 1) { // unary rule
      logp += logP(start, end, r.rhs()[0]);
    } else if (r.arity() == 2) { // binary rule
      logp += logP(start, mid, r.rhs()[0])
                + logP(mid, end, r.rhs()[1]);                   
    }
    if (logp == Double.NEGATIVE_INFINITY) {
      System.out.println("break here 1");
      System.exit(1);
    }
    double oldlp = logP(start, end, r.lhs());
    if (Double.isNaN(oldlp) || oldlp < logp) {
      if (logp == Double.NEGATIVE_INFINITY) {
        System.out.println("break here 2");
        System.exit(2);
      }
      logP(start, end, r.lhs(), logp);
      for (Rule rc : _g.revchainrules(r.lhs()))
        addProb(start, (short)0, end, rc);
    }
  }
  
  public Tree<String> buildVParse() {
    return buildVParse((short)0, (short)_chart.maxpos(), _g.S());
  }
  
  int debugcount = 0;
  public Tree<String> buildVParse(short start, short end, int A) {
    Tree<String> node = new Tree<String>(_g.NT(A));
    List<Tree<String>> children = new ArrayList<Tree<String>>();
    node.setChildren(children);
    for (Rule r : _g.trules(A)) { // preterminal A generates a tag in the input chart
      int ti = _g.T2i(r.rhs()[0]);
      if (_ichart.get(start, end, ti) &&
          logP(start, end, A) == _g.logProb(r.id()) + logLP(start,end, ti)) {
        // create tag and word nodes
        Tree<String> ctag = new Tree<String>(_g.T(r.rhs()[0]));
        Tree<String> cword = new Tree<String>(StringUtils.join(_inputs.subList(start, end), " "));
        List<Tree<String>> cs = new ArrayList<Tree<String>>();
        cs.add(cword);
        ctag.setChildren(cs);
        children.add(ctag);
        return node;
      }
    }
    for (Rule r : _g.chainrules(A)) { // add chain rule NT nodes
      if (_chart.get(start, end, r.rhs()[0]) &&
          logP(start, end, A) > Double.NEGATIVE_INFINITY &&
          logP(start, end, A) == 
            logP(start, end,r.rhs()[0]) + _g.logProb(r.id())) {
        debugcount ++;
        if (debugcount == 1000) {
          System.out.println("break here 3");
        }
        node.getChildren().add(buildVParse(start, end, r.rhs()[0]));
        return node;
      }
    }
    for (Rule r : _g.birules(A)) { // add binary rule NT nodes
      BitSet vec1 = _chart.getVectorByStart(start, (short)(end - start - 1), r.rhs()[0]);
      BitSet vec2 = _chart.getVectorByEnd(end, (short)(end - start - 1), r.rhs()[1]);
      vec1.and(vec2);
      for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1)) {
        if (logP(start, end, A) ==
              logP((short)start, (short)(start+m+1), r.rhs()[0]) + //[r.rhs()[0]*_chart.pagesize()+(start+m+1)*(start+m)/2+start] +
              logP((short)(start+m+1), (short)end, r.rhs()[1]) + //[r.rhs()[1]*_chart.pagesize()+end*(end-1)/2+start+m+1] +
              _g.logProb(r.id())) {
          Tree<String> lc = buildVParse(start, (short)(start+m+1), r.rhs()[0]);
          Tree<String> rc = buildVParse((short)(start+m+1), end, r.rhs()[1]);
          node.getChildren().add(lc);
          node.getChildren().add(rc);
          return node;
        }
      }
    }
    return null;
  }
  public void printVParse(Writer w) throws IOException {
    w.write(Math.exp(logP((short)0,(short)_chart.maxpos(),_g.NT(_g.Start())))+":");
    printVParse((short)0, (short)_chart.maxpos(), _g.NT(_g.Start()), w);
    w.flush();
  }
  
  public void printVParse(short start, short end, int A, Writer w) throws IOException {
    w.write("("+_g.NT(A)+" ");
    
    for (Rule r : _g.trules(A)) {
      int ti = _g.T2i(r.rhs()[0]);
      if (_ichart.get(start, end, ti) &&
          logP(start, end, A) == _g.logProb(r.id()) + logLP(start, end, ti)) {
        w.write("("+_g.T(r.rhs()[0])+" \""+_inputs.get(start));
        for (int i = start+1; i < end; i ++)
          w.write(" "+_inputs.get(i));
        w.write("\"))");
        return;
      }
    }
    for (Rule r : _g.chainrules(A)) {
      if (_chart.get(start, end, r.rhs()[0]) &&
          logP(start, end, A) == 
            logP(start, end, r.rhs()[0]) + _g.logProb(r.id())) {
        printVParse(start, end, r.rhs()[0], w);
        w.write(")");
        return;
      }
    }
    for (Rule r : _g.birules(A)) {
      BitSet vec1 = _chart.getVectorByStart(start, (short)(end - start - 1), r.rhs()[0]);
      BitSet vec2 = _chart.getVectorByEnd(end, (short)(end - start - 1), r.rhs()[1]);
      vec1.and(vec2);
      for (int m = vec1.nextSetBit(0); m >= 0; m = vec1.nextSetBit(m+1)) {
        if (logP(start, end, A) ==
              logP((short)start, (short)(start+m+1), r.rhs()[0]) + //[r.rhs()[0]*_chart.pagesize()+(start+m+1)*(start+m)/2+start] +
              logP((short)(start+m+1), (short)end, r.rhs()[1]) + //[r.rhs()[1]*_chart.pagesize()+end*(end-1)/2+start+m+1] +
              _g.logProb(r.id())) {
          printVParse(start, (short)(start+m+1), r.rhs()[0], w);
          printVParse((short)(start+m+1), end, r.rhs()[1], w);
          w.write(")");
          return;
        }
      }
    }
  }
  private double [] _logP = null; // viterbi log probability 
  private double [] _logLP = null; // viterbi log lexical probability
  
  private Vector<String> _inputs = null; // tokenized input sentence to be parsed 
  private BitRecChart _chart = null; // parse chart (syntax)
  private BitRecChart _ichart = null; // input chart (pos/tag dag)
  private Grammar _g = null;
  private Lexicon _l = null;
  private Statistics _stats = new Statistics(); 
  
  public Statistics stats() {
    return _stats;
  }
  
  public BitRecChart chart() {
    return _chart;
  }
  public BitRecChart ichart() {
    return _ichart;
  }
  
  public static class Statistics {
    public int totalSentences = 0; // total number of sentences parsed
    public int totalTokens = 0; // total number of tokens parsed
    public int successSentences = 0; // successfully parsed sentences
    public int successTokens = 0; // sum of tokens from successfully parsed sentences
    public long totalParseTime = 0l; // total time (in milliseconds) spent on parsing 
    public double totalCFR = 0.0; // total sum of chart filling rates
    public double lastCFR = 0.0; // chart filling rate of the last parse
    public long lastParseTime = 0l; // parsing time of the last parse
    public int lastTokens = 0; // length of the last parse input
  
    public double coverage() { // %
      return ((double)successSentences) / totalSentences;
    }
    public double averageSentenceLength() { // token / sentence
      return ((double)totalTokens) / totalSentences;
    }
    public double averageCFR() { // CFR
      return totalCFR / totalSentences;
    }
    public double averageSuccessSentenceLength() { // token / sentence
      return ((double)successTokens) / successSentences;
    }
    public double averageSpeed() { // token / second
      return (1000.0 * totalTokens) / totalParseTime;
    }
    public double averageParseTimePerSentence() { // milliseconds / sentence
      return ((double)totalParseTime) / totalSentences;
    }
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    try {
      if (args[0].equals("-f")) {
        Grammar grammar = new Grammar();
        grammar.loadGrammar(args[1]); 
        grammar.normalize();
        Lexicon lexicon = new Lexicon(grammar.ts());
        lexicon.loadLexicon(args[2]);
        Parser parser = new Parser(grammar, lexicon);
        parser.lexparse(parser.tokenize(args[3]));
        parser.recognize();
        if (parser._chart.recognized())
          System.out.println("Yes");
        else
          System.out.println("No");
        ParseForest forest = new ParseForest(parser._chart, parser._ichart, grammar);
        forest.debugPrint();
        System.out.println("Done.");
      } else if (args[0].equals("-v")) {
        System.out.print("Loading grammar ... ");
        Grammar grammar = new Grammar();
        grammar.loadGrammar(args[1]);
        grammar.normalize();
        System.out.println("done.");
        System.out.print("Loading lexicon ... ");
        Lexicon lexicon = new Lexicon(grammar.ts());
        lexicon.loadLexicon(args[2]);
        System.out.println("done.");
        Parser parser = new Parser(grammar, lexicon);
        long startTime = System.currentTimeMillis();
        parser.lexparse(parser.tokenize(args[3]));
        long endTime = System.currentTimeMillis();
        System.out.println("lexparse(): "+(endTime-startTime)+" ms");
        startTime = System.currentTimeMillis();
        parser.recognize();
        endTime = System.currentTimeMillis();
        System.out.println("recognize(): "+(endTime-startTime)+" ms");
        if (parser._chart.recognized())
          System.out.println("Yes");
        else {
          System.out.println("No");
          System.exit(1);
        }
        startTime = System.currentTimeMillis();
        parser.viterbi();
        endTime = System.currentTimeMillis();
        System.out.println("viterbi(): "+(endTime-startTime)+" ms");
        //parser.printVParse(new OutputStreamWriter(System.out));
        startTime = System.currentTimeMillis();
        Tree<String> parse = parser.buildVParse();
        endTime = System.currentTimeMillis();
        System.out.println("buildVParse(): "+(endTime-startTime)+" ms");
        Tree<String> result = (new Trees.GrammarInternalNodeStripper()).transformTree(parse);
        System.out.println(Trees.PennTreeRenderer.render(result));
        System.out.println();
      } else if (args[0].endsWith("-b")) {
        System.out.print("Loading grammar ... ");
        Grammar grammar = (Grammar)(new ObjectInputStream(new FileInputStream(args[1]))).readObject();
        System.out.println("done.");
        Parser parser = new Parser(grammar);

        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String sentence = null;
        while ((sentence=br.readLine()) != null) {
          if (sentence.equals(""))
            break;
          long startTime = System.currentTimeMillis();
          parser.lexparse(parser.tokenize(sentence));
          long endTime = System.currentTimeMillis();
          System.out.println("lexparse(): "+(endTime-startTime)+" ms");
          startTime = System.currentTimeMillis();
          parser.recognize();
          endTime = System.currentTimeMillis();
          System.out.println("recognize(): "+(endTime-startTime)+" ms");
          System.out.println("Chart filling rate (before filtering): "+parser._chart.fillrate());
          if (parser._chart.recognized())
            System.out.println("Yes");
          else {
            System.out.println("No");
            parser.viterbi();      
            Tree<String> rtree = new Tree<String>(parser._g.Start());
            List<Tree<String>> children = parser.partialParse();
            rtree.setChildren(children);
            Tree<String> result = (new Trees.GrammarInternalNodeStripper()).transformTree(rtree);
            System.out.println(Trees.PennTreeRenderer.render(result));
            continue;
          }
          startTime = System.currentTimeMillis();
          parser.viterbi();
          endTime = System.currentTimeMillis();
          System.out.println("viterbi(): "+(endTime-startTime)+" ms");
          //parser.printVParse(new OutputStreamWriter(System.out));
          startTime = System.currentTimeMillis();
          Tree<String> parse = parser.buildVParse();
          endTime = System.currentTimeMillis();
          System.out.println("buildVParse(): "+(endTime-startTime)+" ms");
          Tree<String> result = (new Trees.GrammarInternalNodeStripper()).transformTree(parse);
          System.out.println(Trees.PennTreeRenderer.render(result));
          System.out.println();
        }
      }
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

}
