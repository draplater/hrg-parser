package jigsaw.syntax;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Vector;

/**
 * Context-Free Grammar
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 22, 2010
 *
 */
public class Grammar implements Serializable {

  private static final long serialVersionUID = 3779488872217828209L;

  public Grammar() {
    _l = new Lexicon(_ts);
  }
  
  public Grammar(Lexicon lex) {
    _l = lex;
    _ts = _l.tags();
  }
  private Lexicon _l = null;
  
  public Lexicon lexicon() {
    return _l;
  }
  
  /** Symbol table for terminals */
  private SymbolTable _ts = new SymbolTable();
  /** Symbol table for nonterminals */
  private SymbolTable _nts = new SymbolTable();
  /** List of rules */
  private Vector<Rule> _rules = new Vector<Rule>();
  /** HashMap of rules */
  private HashMap<Rule,Rule> _rulemap = new HashMap<Rule,Rule>();
  /** List of pre-terminal rules */
  private Vector<Rule> _trules = new Vector<Rule>();
  /** List of chain/unary rules */
  private Vector<Rule> _chainrules = new Vector<Rule>();
  /** List of binary rules */
  private Vector<Rule> _birules = new Vector<Rule>();
  /** Empty list containing no rules */
  private Vector<Rule> _norules= new Vector<Rule>();
  
  // LHS: left of rule
  /** Rules indexed by their LHS */
  private Vector<Rule> [] _idx_rules = null;
  /** Binary rules indexed by their LHS */
  private Vector<Rule> [] _idx_birules = null;
  /** Preterminal rules indexed by their LHS */
  private Vector<Rule> [] _idx_trules = null;
  /** Chain/unary rules indexed by their LHS */
  private Vector<Rule> [] _idx_chainrules = null;
  /** Chain/unary rules indexed by their RHS[0] */
  private Vector<Rule> [] _idx_rev_chainrules = null;
  
  private double [] _logprob = null;
  private int _start = 0;
  public final static char MODPREFIX = '@'; // prefix for the internal symbols
  
  public void buildProb() {
    _logprob = new double[_rules.size()];
    int [] lhssum = new int[_nts.size()];
    for (int i = 0; i < lhssum.length; i ++)
      lhssum [i] = 0;
    for (int i = 0; i < _logprob.length; i ++) {
      int lhs = _rules.get(i).lhs();
      if (lhssum[lhs] == 0) {
        for (Rule r : rules(lhs))
          lhssum[lhs] += r.freq();
      }
      _logprob[i] = Math.log((double)_rules.get(i).freq() / lhssum[lhs]);
    }
  }
  
  public void analyzeConnectivity() {
    BitSet [] offspring = new BitSet[_nts.size()];
    for (Rule r : _rules) {
      if (offspring[r.lhs()] == null) { 
        offspring[r.lhs()] = new BitSet(_nts.size());
        offspring[r.lhs()].set(r.lhs());
      }
      for (int c : r.rhs()) {
        if (c >= 0)
          offspring[r.lhs()].set(c);
      }
    }
    boolean changed = true;
    int iter = 0;
    while (changed) {
      changed = false;
      iter ++;
      for (int i = 0; i < _nts.size(); i ++) {
        for (int j = 0; j < _nts.size(); j ++) {
          if (i != j && offspring[i].get(j)) {
            int card = offspring[i].cardinality();
            offspring[i].or(offspring[j]);
            if (card != offspring[i].cardinality())
              changed = true;
          }
        }
      }
    }
    System.err.println("Closure found in "+iter+" iterations");
//    BitSet [] offspring2 = new BitSet[_nts.size()];
//    for (Rule r : _rules) {
//      for (int c : r.rhs()) {
//        if (c >= 0) {
//          if (offspring2[c] == null)
//            offspring2[c] = new BitSet(_nts.size());
//          offspring2[c].set(r.lhs());
//        }
//      }
//    }
//    for (int k = 0; k < _nts.size(); k ++) {
//      for (int i = 0; i < _nts.size(); i ++) {
//        for (int j = 0; j < _nts.size(); j ++) {
//          if (offspring[i].get(k) && offspring[k].get(j))
//            offspring[i].set(j);
//        }
//      }
//    }
    for (int A = 0; A < _nts.size(); A ++) {
      System.out.print(NT(A)+"\t"+offspring[A].cardinality()+"\t");
      
      for (int i = offspring[A].nextSetBit(0); i >= 0; i = offspring[A].nextSetBit(i+1)) {
        if (NT(i).startsWith(MODPREFIX+"T")) {
          System.out.print(" "+T(trules(i).firstElement().rhs()[0]));
        } else {
          System.out.print(" "+NT(i));
        }
      }
      System.out.println();
    }
  }
  
  public double logProb(int ridx) {
    return _logprob[ridx];
  }
  public Vector<Rule> rules() {
    return _rules;
  }
  public Vector<Rule> birules() {
    return _birules;
  }
  public Vector<Rule> trules() {
    return _trules;
  }
  public Vector<Rule> chainrules() {
    return _chainrules;
  }
  public Vector<Rule> rules(int lhs) {
    if (_idx_rules[lhs] != null)
      return _idx_rules[lhs];
    else
      return _norules;
  }
  public Vector<Rule> birules(int lhs) {
    if (_idx_birules[lhs] != null)
      return _idx_birules[lhs];
    else
      return _norules; 
  }
  public Vector<Rule> trules(int lhs) {
    if (_idx_trules[lhs] != null)
      return _idx_trules[lhs];
    else
      return _norules;
  }
  public Vector<Rule> chainrules(int lhs) {
    if (_idx_chainrules[lhs] != null)
      return _idx_chainrules[lhs];
    else
      return _norules;
      
  }
  public Vector<Rule> revchainrules(int rhs) {
    if (_idx_rev_chainrules[rhs] != null)
      return _idx_rev_chainrules[rhs];
    else
      return _norules;
  }
  
  
  public void loadProbGrammar(String filename) throws IOException, ParserException {
    BufferedReader br = new BufferedReader(new FileReader(filename));
    String line = null;
    ArrayList<Double> logprobs = new ArrayList<Double>();
    while ((line = br.readLine()) != null) {
      String [] toks = line.split("\\s+");
      if (toks.length < 4)
        throw new ParserException("Invalid line in grammar " + filename);
      if (toks[0].equals(toks[2]) && toks[3].equals("1.0"))
        continue;
      _nts.register(toks[0].substring(0, toks[0].indexOf('_')));
    }
    br.close();
    br = new BufferedReader(new FileReader(filename));
    while ((line = br.readLine()) != null) {
      String [] toks = line.split("\\s+");
      if (toks[0].equals(toks[2]) && toks[3].equals("1.0"))
        continue;
      int lhs = _nts.lookup(toks[0].substring(0, toks[0].indexOf('_')));
      int [] rhs = new int[toks.length-3];
      for (int i = 0; i < rhs.length; i ++) {
        int r = _nts.lookup(toks[i+2].substring(0, toks[i+2].indexOf('_')));
        if (r != -1) { // non-terminal
          rhs[i] = r;
        } else { // terminal
          r = _ts.register(toks[i+2].substring(0, toks[i+2].indexOf('_')));
          rhs[i] = -r - 2;
        }
      }
      Rule rule = new Rule(lhs, rhs, 0);
      _rules.add(rule);
      logprobs.add(Math.log(Double.parseDouble(toks[toks.length-1])));
    }
    br.close();
    preterminalize();
    _logprob = new double[_rules.size()];
    for (int i = 0; i < _logprob.length; i ++) {
      if (i < logprobs.size())
        _logprob[i] = logprobs.get(i);
      else
        _logprob[i] = 0.0;
    }
    indexRules();
  }
  
  /**
   * Load grammar from one PCFG file 
   * @param filename
   * @throws IOException
   * @throws ParserException
   */
  public void loadGrammar(String filename) throws IOException, ParserException {
    // First parse: record all nonterminals
    BufferedReader br = new BufferedReader(new FileReader(filename));
    String line = null;
    while ((line = br.readLine()) != null) {
      String [] toks = line.split("\\s+");
      if (toks.length < 3)
        throw new ParserException("Invalid line in grammar "+filename);
      _nts.register(toks[1]);
    }
    br.close();
    // Second parse: read terminals, rules and frequencies
    br = new BufferedReader(new FileReader(filename));
    while ((line = br.readLine()) != null) {
      String [] toks = line.split("\\s+");
      int lhs = _nts.lookup(toks[1]);
      int [] rhs = new int[toks.length-2];
      for (int i = 0; i < rhs.length; i ++) {
        int r = _nts.lookup(toks[i+2]);
        if (r != -1) { // non-terminal
          rhs[i] = r;
        } else { // terminal
          r = _ts.register(toks[i+2]);
          rhs[i] = -r - 2;
        }
      }
      Rule rule = new Rule(lhs, rhs, Integer.parseInt(toks[0]));
      _rules.add(rule);
    }
    br.close();
  }
  
  /**
   * Load grammar from a PCFG file and a lexicon file 
   * @param gfile
   * @param lfile
   * @throws IOException
   * @throws ParserException
   */
  public void loadGrammar(String gfile, String lfile) throws IOException, ParserException {
    BufferedReader br = new BufferedReader(new FileReader(gfile));
    String line = null;
    while ((line=br.readLine())!=null) {
      String [] toks = line.split("\\s+");
      if (toks.length < 3)
        throw new ParserException("Invalid line in grammar "+gfile);
      int lhs = _nts.register(toks[1]);
      int [] rhs = new int[toks.length-2];
      for (int i = 0; i < rhs.length; i ++)
        rhs[i] = _nts.register(toks[i+2]);
      Rule rule = new Rule(lhs, rhs, Integer.parseInt(toks[0]));
      _rules.add(rule);
    }
    br.close();
    br = new BufferedReader(new FileReader(lfile));
    while ((line=br.readLine())!=null) {
      String [] toks = line.split("\\t");
      if (toks.length < 2)
        throw new ParserException("Invalid line in lexicon "+lfile);
      int t = _ts.register(toks[0]);
      for (int i = 1; i < toks.length; i ++) {
        String [] parts = toks[i].split(" ");
        Rule rule = new Rule(_nts.register(parts[0]), new int[]{-t-2}, Integer.parseInt(parts[1]));
        _rules.add(rule);
        // TODO
      }
    }
    br.close();
  }
  
  public void incrSeenCount(Rule r) {   
    if (_rulemap.containsKey(r)) {
      Rule rule = _rulemap.get(r);
      rule.incr();
    } else {
      r.incr();
      _rules.add(r);
      _rulemap.put(r, r);
    }
  }
  
  public void incrSeenCount(String word, int tag, int count) {
    _l.incrSeenCount(word, tag, count);
  }
  
  public String T(int t) {
    return _ts.lookup(-t-2);
  }
  public int T(String t) {
    return -_ts.lookup(t)-2;
  }
  public int T2i(int t) {
    return -t-2;
  }
  public int i2T(int i) {
    return -i-2;
  }
  public String NT(int nt) {
    return _nts.lookup(nt);
  }
  public int NT(String nt) {
    return _nts.lookup(nt);
  }
  public String Symbol(int x) {
    if (x == -1)
      return null;
    if (x > 0)
      return NT(x);
    else
      return T(x);
  }
  public SymbolTable nts() {
    return _nts;
  }
  public SymbolTable ts() {
    return _ts;
  }
  public String Start() {
    return _nts.lookup(_start);
  }
  public int S() {
    return _start;
  }
  public int registerNT(String nt) {
    return _nts.register(nt);
  }
  public int registerT(String t) {
    return -_ts.register(t)-2;
  }
  public int registerS(String s) {
    _start = _nts.register(s);
    return _start;
  }
  /** Assume the input grammar is already epsilon free, convert the 
   * grammar into CNF. Newly added categories are named as "@n" */
  public void normalize() {
    preterminalize();
    binarize();
    topoSort();
    indexRules();
  }
  
  public void indexRules() {
    _idx_rules = new Vector[_nts.size()];
    _idx_birules = new Vector[_nts.size()];
    _idx_trules = new Vector[_nts.size()];
    _idx_chainrules = new Vector[_nts.size()];
    _idx_rev_chainrules = new Vector[_nts.size()];
    int id = 0; 
    for (Rule r: _rules) {
      r.id(id++);
      if (_idx_rules[r.lhs()] == null) {
        _idx_rules[r.lhs()] = new Vector<Rule>();
      }
      _idx_rules[r.lhs()].add(r);
      if (r.arity() == 2) {
        _birules.add(r);
        if (_idx_birules[r.lhs()] == null) {
          _idx_birules[r.lhs()] = new Vector<Rule>();
        }
        _idx_birules[r.lhs()].add(r);
      }
      else if (r.arity() == 1) {
        if (r.rhs()[0] >= 0) {
          _chainrules.add(r);
          if (_idx_chainrules[r.lhs()] == null) {
            _idx_chainrules[r.lhs()] = new Vector<Rule>();
          }
          _idx_chainrules[r.lhs()].add(r);
          if (_idx_rev_chainrules[r.rhs()[0]] == null) {
            _idx_rev_chainrules[r.rhs()[0]] = new Vector<Rule>();
          }
          _idx_rev_chainrules[r.rhs()[0]].add(r);
        }
        else {
          _trules.add(r);
          if (_idx_trules[r.lhs()] == null) {
            _idx_trules[r.lhs()] = new Vector<Rule>();
          }
          _idx_trules[r.lhs()].add(r);
        }
      }
    }
    initChainVectors();
  }
  
  /** add preterminal NT symbol to replace the terminals on the RHS of 
   * non-unary rules.
   */
  public void preterminalize() {
    int x = 1;
    // make sure terminal symbols are only in unary rules 
    HashMap<Integer,Integer> pts = new HashMap<Integer,Integer>();
    HashMap<Integer,Integer> ptc = new HashMap<Integer,Integer>();
    for (Rule r :_rules) {
      if (r.rhs().length == 1)
        continue;
      for (int i = 0; i < r.rhs().length; i ++) {
        if (r.rhs()[i] < 0) { // terminal, replace
          if (pts.containsKey(r.rhs()[i])) {
            ptc.put(r.rhs()[i], ptc.get(r.rhs()[i])+r.freq());  
            r.rhs()[i] = pts.get(r.rhs()[i]);              
          } else {
            int nt = _nts.register(MODPREFIX+"T"+x);
            x++;
            pts.put(r.rhs()[i], nt);
            ptc.put(r.rhs()[i], r.freq());
            r.rhs()[i] = nt;
          }
        }
      }
    }
    for (int t: pts.keySet()) {
      int [] rhs = { t };
      _rules.add(new Rule(pts.get(t), rhs, ptc.get(t)));
    }
  }
  public void binarize() {
    int x = 1;
    Vector<Rule> todo = _rules;
    Vector<Rule> tmp = new Vector<Rule>();
    Vector<Rule> binarized = new Vector<Rule>();
    while (!todo.isEmpty()) {
      HashMap<String,Integer> paircounts = new HashMap<String,Integer>();
      int maxcount = -1;
      int maxA = -1;
      int maxB = -1;
      for (Rule r : todo) {
        if (r.arity() <= 2) {
          // already binarized, move to the new rule set
          binarized.add(r);
        } else {
          // not yet binary, move to the temp
          tmp.add(r);
          // count bigrams
          // NOTE: in case of S-> A A A, only count the bigram <A,A> once
          // TODO: this could be extended for head-oriented binarization strategy 
          int previous = -1;
          for (int i = 0; i < r.arity() - 1; i ++) {
            if (r.rhs()[i] == r.rhs()[i+1] &&
                r.rhs()[i] == previous) {
              previous = -1;
              continue;
            }
            String pair = r.rhs()[i]+":"+r.rhs()[i+1];
            int c = paircounts.containsKey(pair)?paircounts.get(pair):0;
            paircounts.put(pair, c+r.freq());
            if (c+r.freq() > maxcount) {
              maxcount = c+r.freq();
              maxA = r.rhs()[i];
              maxB = r.rhs()[i+1];
            }
            previous = r.rhs()[i];
          }
        }
      }      
      if (tmp.isEmpty())
        break;
      // replace all bigram <maxA,maxB> with new nt
      todo.clear();
      int nt = _nts.register(MODPREFIX+"B"+x);
      x++;     
      for (Rule r: tmp) {
        Vector<Integer> newrhs = new Vector<Integer>();
        boolean replaced = false;
        for (int i = 0; i < r.arity(); i ++) {
          if (i < r.arity() - 1 &&
              r.rhs()[i] == maxA &&
              r.rhs()[i+1] == maxB) {
            replaced = true;
            i++;
            newrhs.add(nt);
          } else
            newrhs.add(r.rhs()[i]);
        }
        if (!replaced)
          todo.add(r);
        else {
          int [] nrhs = new int[newrhs.size()];
          for (int i = 0; i < newrhs.size(); i ++)
            nrhs[i] = newrhs.get(i);
          todo.add(new Rule(r.lhs(),nrhs,r.freq()));
        }
      }
      // add new binary rule to the result rule set
      binarized.add(new Rule(nt,new int[]{maxA,maxB},maxcount));
      tmp.clear();
    }
    _rules = binarized;
  }
  
  public void topoSort() {
    Vector<Rule> [] ruleidx = new Vector[_nts.size()];
    for (Rule r: _rules) {
      if (ruleidx[r.lhs()] == null)
        ruleidx[r.lhs()] = new Vector<Rule>();
      ruleidx[r.lhs()].add(r);
    }
    // first, topo sort NTs
    SymbolTable newnts = new SymbolTable();
    newnts.register(_nts.lookup(_start));
    Queue<Integer> q = new LinkedList<Integer>();
    q.add(_start);
    while (!q.isEmpty()) {
      int nt = q.poll();
      for (Rule r : ruleidx[nt]) {
        for (int i = 0; i < r.arity(); i ++) {
          if (r.rhs()[i] >= 0 && 
              newnts.lookup(_nts.lookup(r.rhs()[i])) == -1) {
            // an nt which has not been registered in the new table
            newnts.register(_nts.lookup(r.rhs()[i]));
            q.add(r.rhs()[i]);
          }
        }
      }
    }
    for (String nt : _nts.symbols()) {
      if (newnts.lookup(nt) == -1)
        newnts.register(nt);
    }
    
    // now, change the rules
    Vector<Rule> newrules = new Vector<Rule>();
    for (int nt = 0; nt < newnts.size(); nt ++) {
      int oldnt = _nts.lookup(newnts.lookup(nt));
      for (Rule r: ruleidx[oldnt]) {
        int [] newrhs = new int[r.arity()];
        for (int i = 0; i < r.arity(); i ++) {
          if (r.rhs()[i] >= 0) 
            newrhs[i] = newnts.lookup(_nts.lookup(r.rhs()[i]));
          else
            newrhs[i] = r.rhs()[i];
        }
        newrules.add(new Rule(nt, newrhs, r.freq()));
      }
    }
    // now we can throw away the old nt symbol table and rules
    _nts = newnts;
    _rules = newrules;
  }
  
  private BitSet [] _chainvec = null;
  public void initChainVectors() {
    int _maxs = _nts.size();
    _chainvec = new BitSet[_maxs + 1];
    for (int i = 0; i <= _maxs; i ++) {
      _chainvec[i] = new BitSet(_maxs + 1);
      _chainvec[i].set(i);
    }
    for (Rule r : rules()) {
      if (r.arity() == 1 &&
          r.rhs()[0] >= 0) {
        _chainvec[r.rhs()[0]].set(r.lhs());
      }
    }
//    // compute the transitive closure with Warshall's algorithm
//    for (int i = 0; i <= _maxs; i ++) {
//      for (int j = 0; j <= _maxs; j ++) {
//        if (_chainvec[i].get(j)) {
//          _chainvec[i].or(_chainvec[j]);
//        }
//      }
//    }
    boolean changed = true;
    while (changed) {
      changed = false;
      for (int i = 0; i <= _maxs; i ++) {
        for (int j = 0; j <= _maxs; j ++) {
          if (_chainvec[i].get(j)) {
            int card = _chainvec[i].cardinality();
            _chainvec[i].or(_chainvec[j]);
            if (card != _chainvec[i].cardinality())
              changed = true;
          }
        }
      }
    }
  }
  public BitSet[] chainvec() {
    return _chainvec; 
  }
  public void dumpGrammar(String filename) throws IOException {
    BufferedWriter bw = new BufferedWriter(new FileWriter(filename+".grammar"));
    for (Rule r : _rules) {
      bw.write(r.freq()+" "+NT(r.lhs()));
      for (int x : r.rhs()) {
        bw.write(" "+Symbol(x));
      }
      bw.write("\n");
    }
    bw.close();
    _l.dumpLexicon(filename);
  }
  
  public static void main(String [] args) {
//    try {
//      System.out.println("Memory "+Runtime.getRuntime().totalMemory());
//      Grammar g = new Grammar();
//      String gfile = args[0];
//      String lfile = null;
//      String dfile = args[1];
//      if (args.length == 3) {
//        lfile = args[1];
//        dfile = args[2];
//      }
//      if (lfile == null) {
//        System.out.println("Loading grammar "+gfile+" ...");
//        g.loadGrammar(gfile);
//      } else {
//        System.out.println("Loading grammar "+gfile+" and lexicon "+lfile+" ...");
//        g.loadGrammar(gfile, lfile);
//      }
//      System.out.println("Rules: "+g._rules.size());
//      System.out.println("Terminals: "+g._ts.size());
//      System.out.println("Non-Terminals: "+g._nts.size());
//      System.out.println("Start Symbol: "+g._nts.lookup(0));
//      System.out.println("Memory "+Runtime.getRuntime().totalMemory());
//      System.out.println();
//      System.out.println("Normalizing...");
//      g.normalize();
//      System.out.println("Rules: "+g._rules.size());
//      System.out.println("Terminals: "+g._ts.size());
//      System.out.println("Non-Terminals: "+g._nts.size());
//      System.out.println("Start Symbol: "+g._nts.lookup(0));
//      System.out.println("Memory "+Runtime.getRuntime().totalMemory());
//      System.out.println("Dumping grammar to "+dfile+" ...");
//      g.dumpGrammar(dfile);
//      System.out.println("Done.");
//    } catch (Exception ex) {
//      ex.printStackTrace();
//    }
    try {
      System.err.print("Loading ... ");
      Grammar g = null;
      if (args[0].endsWith(".gr"))
        g = (Grammar)(new ObjectInputStream(new FileInputStream(args[0])).readObject());
      else {
        g = new Grammar();
        g.loadGrammar(args[0]);
      }
      System.err.println("done");
      g.analyzeConnectivity();
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }
}
