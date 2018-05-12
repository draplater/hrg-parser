package jigsaw.treebank;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

import fig.basic.LogInfo;

import jigsaw.grammar.AnnotRedwoodsPcfgExtractor;
import jigsaw.treebank.Trees.*;
import jigsaw.syntax.Tree;
import jigsaw.util.Filter;
import jigsaw.util.StringUtils;
import org.javatuples.Pair;


public class RedwoodsTreeReader implements Iterator<Tree<String>> {

  public String _currentParseId;

  public enum TreeBankType {
    REDWOODS,
    REDWOODS_POS,
    REDWOODS_MORPH,
    REDWOODS_POSMORPH
  }

  public RedwoodsTreeReader() {
	  ;
  }
  
  public RedwoodsTreeReader(File profile, long fromID, long toID, boolean firstReading) throws IOException{
    _fromID = fromID;
    _toID = toID;
    _profile = profile;
    _firstReading = firstReading;
    initialize();
  }
  
  public RedwoodsTreeReader(File profile, int fromID, int toID) throws IOException {
    this(profile, fromID, toID, false);
  }
  private long _fromID = Long.MIN_VALUE;
  private long _toID = Long.MAX_VALUE;
  private boolean _firstReading = false;
  
  public RedwoodsTreeReader(File profile) throws IOException {
    this(profile, Integer.MIN_VALUE, Integer.MAX_VALUE);
  }
  
  public RedwoodsTreeReader(File profile, boolean firstReading) throws IOException {
    this(profile, Integer.MIN_VALUE, Integer.MAX_VALUE, firstReading);
  }
  
  public void initialize() throws IOException {
    loadRelations();
    if (_firstReading)
      loadFirsts();
    else
      loadActives();
    openResults();
  }
  
  private File _profile = null;
   
  private BufferedReader _br = null; // Reader for the result fname
    
  /** a mapping from fields name to the field index position
   * E.g. tree:t-version -> 1
   */
  private HashMap<String,Integer> _rel = null;
  
  public boolean hasNext() {
    return !_remaining_results.isEmpty();
  }
  
  private MorphLexStripper _morphLexStripper = new MorphLexStripper();
  private LENameStripper _leNameStripper = new LENameStripper();
  private XOverXRemover<String> _xOverXRemover = new XOverXRemover<String>();
  private LEType2POSStripper _leType2POSStripper = new LEType2POSStripper();
  private MorphLexMarker _morphLexMarker = new MorphLexMarker();
  
  public void loadRelations() throws IOException {
    File rfile = new File(_profile, "relations");
    BufferedReader rbr = null;
    if (rfile.exists()) {
      rbr = new BufferedReader(new FileReader(rfile));
    } else {
      rfile = new File(_profile, "relations.gz");
      if (rfile.exists()) {
        rbr = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(rfile))));
      }
    }
    if (rbr == null)
      return;
    _rel = new HashMap<String,Integer>();
    Pattern r = Pattern.compile("^ +([^ ]+) .*");
    String line = null;
    String section = "";
    int idx = 0;
    while ((line=rbr.readLine()) != null) {
      if (line.matches("^[a-z]+:")) {
        section = line.substring(0, line.indexOf(':'));
        idx = 0;
      }
      else if (line.matches("^ +[^ ]+ .*")){
        Matcher m = r.matcher(line);
        if (m.find()) {
          String fname = m.group(1);
          setRelIdx(section+":"+fname, idx);
          idx ++;
        }
      }
    }
    rbr.close();
  }
  
  private void setRelIdx(String rname, int idx) {
    _rel.put(rname, idx);
  }
  private int getRelIdx(String rname) {
    return _rel.get(rname);
  }
  /** should avoid the getRel when one want to access multiple 
   * fields in the same record, for multiple string splitting would be wasted */
  private String getRel(String record, String rname) {
    String [] fields = record.split("@");
    return fields[getRelIdx(rname)];
  }


  final HashMap<Pair<String, Integer>, Pair<Integer, Integer>> _spans = new HashMap<>();
  /** parse-id -> i-id */
  private HashMap<String,String> _idmap = new HashMap<String,String>(); 
  /** load active readings from the profile 
   * 1. read `parse' fname for a mapping from `parse-id' to `i-id' 
   * 2. read `tree' fname for a list of `parse-id+t-version'
   *    Note: a. only keep the highest t-version for each parse-id, 
   *          b. for the highest t-version, only keep when t-active == 1
   * 3. read `preference' fname for the corresponding `result-id' 
   * */
  public void loadActives() throws IOException {    
    // 1. read `parse' fname for a mapping from `parse-id' to `i-id'
    File file = new File(_profile, "parse");
    BufferedReader br = null;
    if (file.exists()) {
      br = new BufferedReader(new FileReader(file));
    } else {
      file = new File(_profile, "parse.gz");
      if (file.exists()) {
        br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))));
      }
    }
    if (br == null)
      return;
    String line = null;
    while ((line=br.readLine()) != null) {
      String [] fields = line.split("@");
      String parse_id = fields[getRelIdx("parse:parse-id")];
      String i_id = fields[getRelIdx("parse:i-id")];
      _idmap.put(parse_id, i_id);

      // extract (parse_id, word_id) -> spans mapping
      final String[] lexiconFields = {"parse:p-input", "parse:p-tokens"};
      for(String field: lexiconFields) {
        String pInput = fields[getRelIdx(field)];
        Pattern pInputPattern = Pattern.compile("\\((\\d+), \\d+, \\d+, <(\\d+):(\\d+)>, \\d+, \".*?\", \\d+, \".*?\"(?:, )?(?:\".*?\" [\\d.]+ ?)*\\)");
        Matcher matcher = pInputPattern.matcher(pInput);
        while (matcher.find()) {
          _spans.put(Pair.with(parse_id, Integer.parseInt(matcher.group(1))),
                  Pair.with(Integer.parseInt(matcher.group(2)),
                          Integer.parseInt(matcher.group(3))
                  ));
        }
      }
    }
    br.close();
    // 2. read `tree' fname for a list of `parse-id+t-version'
    br = null;
    file = new File(_profile, "tree");
    if (file.exists()) {
      br = new BufferedReader(new FileReader(file));
    } else {
      file = new File(_profile, "tree.gz");
      if (file.exists()) {
        br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))));
      }
    }
    if (br == null)
      return;
    // tree:parse-id -> t-version
    HashMap<String,Integer> activetrees = new HashMap<String,Integer>();
    while ((line=br.readLine()) != null) {
      String[] fields = line.split("@");
      String parse_id = fields[getRelIdx("tree:parse-id")];
      int t_version = Integer.parseInt(fields[getRelIdx("tree:t-version")]);
      String t_active = fields[getRelIdx("tree:t-active")];
      if (activetrees.containsKey(parse_id)) {
        int old_t_version = activetrees.get(parse_id);
        if (t_version > old_t_version)
          activetrees.remove(parse_id);
        else
          continue;
      }
//      if (t_active.equals("1") || t_active.equals("2"))
       activetrees.put(parse_id, t_version);
    }
    br.close();
    // 3. read `preference' fname for the corresponding `result-id'
    br = null;
    file = new File(_profile, "preference");
    if (file.exists()) {
      br = new BufferedReader(new FileReader(file));
    } else {
      file = new File(_profile, "preference.gz");
      if (file.exists()) {
        br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))));
      }
    }
    if (br == null)
      return;
    /** parse-id -> result-id */
    _active_results = new HashMap<String,String>();
    _remaining_results = new HashSet<String>();
    while ((line=br.readLine()) != null) {
      String[] fields = line.split("@");
      String parse_id = fields[getRelIdx("preference:parse-id")];
      int t_version = Integer.parseInt(fields[getRelIdx("preference:t-version")]);
      String result_id = fields[getRelIdx("preference:result-id")];
      if (activetrees.containsKey(parse_id) && activetrees.get(parse_id)==t_version) {
        long iid = Long.parseLong(_idmap.get(parse_id));
        if (iid >= _fromID && iid <= _toID) {
          _active_results.put(parse_id, result_id);
          _remaining_results.add(parse_id);
        }
      }
    }
    br.close();
  }


  /** load first reading for each parsed sentence from the profile
   * useful for raw (unannotated) profiles such as wikiwoods
   * @throws IOException
   */
  public void loadFirsts() throws IOException {
    // read `parse' fname for a mapping from `parse-id' to `i-id' 
    File file = new File(_profile, "parse");
    BufferedReader br = null;
    if (file.exists()) {
      br = new BufferedReader(new FileReader(file));
    } else {
      file = new File(_profile, "parse.gz");
      if (file.exists()) {
        br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))));
      }
    }
    if (br == null)
      return;
    /** parse-id -> result-id */
    _active_results = new HashMap<String,String>();
    _remaining_results = new HashSet<String>();
    String line = null;
    while ((line=br.readLine()) != null) {
      String [] fields = line.split("@");
      String parse_id = fields[getRelIdx("parse:parse-id")];
      String i_id = fields[getRelIdx("parse:i-id")];
      _idmap.put(parse_id, i_id);
      long iid = Long.parseLong(i_id);
      int readings = Integer.parseInt(fields[getRelIdx("parse:readings")]);
      if (iid >= _fromID && iid <= _toID && readings > 0) {
        _active_results.put(parse_id, "0");
        _remaining_results.add(parse_id);
      }
    }
    br.close();
    
  }
  public void openResults() throws IOException {
    File file = new File(_profile, "result");
    _br = null;
    if (file.exists()) {
      _br = new BufferedReader(new FileReader(file));
    } else {
      file = new File(_profile, "result.gz");
      if (file.exists()) {
        _br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file))));
      }
    }
  }

  /** parse-id -> result-id */
  private HashMap<String,String> _active_results = null;
  /** parse-id */
  private HashSet<String> _remaining_results = null;
  
  public int size() {
    return _active_results.size();
  }
  
  public Tree<String> next() {
    String line = null;
    try {
      while ((line=_br.readLine()) != null) {
        String [] fields = line.split("@");
        String parse_id = fields[getRelIdx("result:parse-id")];
        String result_id = fields[getRelIdx("result:result-id")];
        if (_active_results.containsKey(parse_id) && 
            _active_results.get(parse_id).equals(result_id)) {
          String derivstr = fields[getRelIdx("result:derivation")];
          //    System.out.println(_line);
          this._currentParseId = parse_id;
          Tree<String> tree = createTree(derivstr);
          Tree<String> root = new Tree<String>("ROOT");
          root.setChildren(new ArrayList<Tree<String>>()); 
          root.getChildren().add(tree);
          _remaining_results.remove(parse_id);
          return root;
        }
      }
    } catch (IOException ex) {
      ex.printStackTrace();
    }
    return null;
  }
  
  public void close() {
    try {
      _br.close();
    } catch (IOException ex) {
      ex.printStackTrace();
    }
  }
  
  public Tree<String> createTree(String tstr) {
//	LogInfo.begin_track("");
//	  LogInfo.logs(tstr);
//	  LogInfo.end_track();
    tstr = tstr.trim();
    if (tstr.length() == 0)
      return null;
    Tree<String> tree = new Tree<String>(null);
    if (!tstr.startsWith("(") || !tstr.endsWith(")"))
      return null;
    tstr = tstr.substring(1,tstr.length()-1);
    if (tstr.startsWith("\"")) {
      // This is a leaf node
      tree.setLabel(parseTerminal(tstr));       
    } else { // non-leaf
      Pattern r = null;
      if (tstr.matches("^[^ ]+ \\(.*")){ // root
        r = Pattern.compile("^([^ ]+) (.*)$");
      } else { // non-root non-terminals
        r = Pattern.compile("^\\d+ ([^ ]+) [0-9e.-]+ \\d+ \\d+ (.*)$");
      }
      Matcher m = r.matcher(tstr);
      if (m.find()) {
        tree.setLabel(m.group(1));
        String dtrstr = m.group(2);
        tree.setChildren(createDaughters(dtrstr));
      }
    }
    return tree;
  }
  
  public String parseTerminal(String tstr) {
    // DO IT THE PROPER WAY, GET YIELD FROM THE TOKEN FS !
    //Pattern r = Pattern.compile("\"(.+?[^\\\\])\" \\d+ \"token \\[ .*? \\]\"");
    ArrayList<String> words = new ArrayList<String>();
//    Pattern r = Pattern.compile("\\\"token \\[ .*? \\+FORM \\\\\\\\\"(.*?)\\\\\\\\\" \\]\\\"");
    Pattern r = Pattern.compile("(\\d+) \"token \\[.*? \\+FORM (.+?) .*?\\]\"");
    Matcher m = r.matcher(tstr);
    while (m.find()) {
      String tokenfs = m.group(0);
      int lexId = Integer.parseInt(m.group(1));
      String spanRepr;
      try {
        Pair<Integer, Integer> span = this._spans.get(Pair.with(_currentParseId, lexId));
        spanRepr = String.format("#__#[%d,%d]", span.getValue0(), span.getValue1());
      } catch(NullPointerException e) {
        spanRepr = "#__#[NotExist]";
      }
      String form = m.group(2);
      if (form.startsWith("\\\\\"") && form.endsWith("\\\\\"")) {
        words.add(form.substring(3, form.length()-3) + spanRepr);
      } else if (form.startsWith("#")) {
        Matcher idxm = Pattern.compile("^#\\d+").matcher(form);
        String idx = "";
        if (idxm.find()) {
          idx = idxm.group(0);
        }
        int pos = tokenfs.indexOf(idx+"=");
        if (pos != -1) {
          Matcher valm = Pattern.compile("\\\\\\\\\"(.*?)\\\\\\\\\"").matcher(tokenfs.substring(pos));
          if (valm.find()) {
            words.add(valm.group(1) + spanRepr);
          }
        }
      }
    }
    return StringUtils.join(words, " ");
  }
  
  public List<Tree<String>> createDaughters(String dstr) {
    ArrayList<Tree<String>> dtrs = new ArrayList<Tree<String>>(); 
    int p = splitDtrStr(dstr);
    while (p != -1) {
      dtrs.add(createTree(dstr.substring(0,p)));
      if (p < dstr.length())
        dstr = dstr.substring(p).trim();
      else 
        break;
      p = splitDtrStr(dstr);
    }
    return dtrs;
  }
  private int splitDtrStr(String str) {
    if (!str.startsWith("("))
      return -1;
    int pl = 1;
    int p = 1;
    boolean quoted = false;
    while (pl != 0 && p < str.length()) {
      if (!quoted) {
        if (str.charAt(p) == '(')
          pl ++;
        else if (str.charAt(p) == ')')
          pl --;
        else if (str.charAt(p) == '"')
          quoted = true;
      } else {
        if (str.substring(p).startsWith("\\\""))
          p ++;
        else if (str.charAt(p) == '"')
          quoted = false;
      }
      p ++;
    }
    if (pl == 0)
      return p;
    else
      return -10;
  }
  public static void main(String [] args) {
    ForkJoinPool pool = new ForkJoinPool(10);
    try {
      if (args.length != 1 &&
          args.length != 2) {
        System.err.println("Usage: <profile dir>");
        System.exit(1);
      }
      String profileDir = args[0];
      File[] profiles = new File(profileDir).listFiles();
      assert profiles != null;
      Arrays.sort(profiles);
      pool.submit(() ->
        Arrays.stream(profiles).parallel()
                .filter(x -> x.getName().startsWith("wsj"))
                .forEach(
              profile -> {
                try {
                  TreeTransformer<String> transformer = new AnnotRedwoodsPcfgExtractor.StandardNormalizer();
                  RedwoodsTreeReader reader = new RedwoodsTreeReader(profile);
                  String[] pathSegments = profile.getPath().split("/");
                  String name = pathSegments[pathSegments.length-1];
                  BufferedWriter out = new BufferedWriter(
                          new FileWriter("out/" + name));
                  while (reader.hasNext()) {
                    try {
                      Tree<String> tree = reader.next();
                      tree = transformer.transformTree(tree);
                      out.write("#" + reader._currentParseId + "\n");
                      out.write(RedwoodsTreeIndentedRenderer.render(tree)
                              .replace("\n", " ")
                              .replaceAll(" +", " "));
                      out.write("\n");
                    } catch(NullPointerException e) {
                      e.printStackTrace();
                    }
                  }
                  out.close();
                } catch (Exception e) {
                  System.out.println(profile);
                  e.printStackTrace();
                }
              }
      )).get();
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

  @Override
  public void remove() {
//  should not be invoked anyway   
  }
  
  /**
   * remove_ morphological (including punctuations) and lexical rules from the derivation tree
   *
   */
  public static class MorphLexStripper implements TreeTransformer<String> {
    
    public Tree<String> transformTree(Tree<String> tree) {
      return Trees.spliceNodes(tree, new Filter<String>() {
        public boolean accept(String t) {
          return (morphr.contains(t.toUpperCase()));
        }
      });
    }
    
    public final static String[] morph_rules = {
      "PLUR_NOUN_ORULE", "THIRD_SG_FIN_VERB_ORULE", "PSP_VERB_ORULE", "PAST_VERB_ORULE", "PRP_VERB_ORULE",
      "BSE_VERB_IRULE", "NON_THIRD_SG_FIN_VERB_IRULE", "BSE_OR_NON3SG_VERB_IRULE", "SING_NOUN_IRULE", 
      "MASS_NOUN_IRULE", "MASS_COUNT_IRULE", "PLUR_NUMCOMP_NOUN_IRULE", "POS_ADJ_IRULE", "PASSIVE_ORULE",
      "PREP_PASSIVE_ORULE", "PREP_PASSIVE_TRANS_ORULE", "CP_PASSIVE_ORULE", "DATIVE_PASSIVE_ORULE",
      "PUNCT_PERIOD_ORULE", "PUNCT_QMARK_ORULE", "PUNCT_QQMARK_ORULE", "PUNCT_QMARK_BANG_ORULE",
      "PUNCT_COMMA_ORULE", "PUNCT_BANG_ORULE", "PUNCT_SEMICOL_ORULE", "PUNCT_RPAREN_ORULE",
      "PUNCT_RP_COMMA_ORULE", "PUNCT_LPAREN_ORULE", "PUNCT_RBRACKET_ORULE", "PUNCT_LBRACKET_ORULE",
      "PUNCT_DQRIGHT_ORULE", "PUNCT_DQLEFT_ORULE", "PUNCT_SQRIGHT_ORULE", "PUNCT_SQLEFT_ORULE",
      "PUNCT_HYPHEN_ORULE", "PUNCT_COMMA_INFORMAL_ORULE", "PUNCT_ITALLEFT_ORULE", "PUNCT_ITALRIGHT_ORULE",
      "PUNCT_DROP_ILEFT_ORULE", "PUNCT_DROP_IRIGHT_ORULE", "SAILR", "ADVADD", "VPELLIPSIS_REF_LR",
      "VPELLIPSIS_EXPL_LR", "INTRANSNG", "INTR_PP_NG", "TRANSNG", "MONTHDET", "WEEKDAYDET", "ATTR_ADJ",
      "ATTR_ADJ_VERB_PART", "ATTR_ADJ_VERB_TR_PART", "ATTR_ADJ_VERB_PSV_PART", "ATTR_VERB_PART_NAMEMOD",
      "ATTR_VERB_PART_TR_NAMEMOD", "PART_PPOF_AGR", "PART_PPOF_NOAGR", "PART_NOCOMP", "NP_PART_LR",
      "DATIVE_LR", "MINUTE_LR", "TAGLR", "BIPART", "FOREIGN_WD", "INV_QUOTE", "NOUN_ADJ_ORULE",
      "RE_VERB_PREFIX", "PRE_VERB_PREFIX", "MIS_VERB_PREFIX", "CO_VERB_PREFIX", "W_PERIOD_PLR"
    };
    public final static HashSet<String> morphr = new HashSet<String>(Arrays.asList(morph_rules));
  }
  
  /**
   * remove_ morphological (including punctuations) and lexical rule nodes from the derivation tree
   * and mark the enlisted ones on the preterminal node (assuming the preterminal is not morph/lex rule).
   * 
   */
  public static class MorphLexMarker implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      List<Tree<String>> roots = transformTreeHelper(tree, "");
      if (roots.size() != 1)
        return null;
      return roots.get(0);
    }
    
    public List<Tree<String>> transformTreeHelper(Tree<String> tree, String morphmarking) {
      if (tree.isPreTerminal()) {
        if (morphmarking.equals(""))
          return Collections.singletonList(tree);
        else {
          Tree<String> newtree = tree.shallowClone();
          newtree.setLabel(tree.getLabel()+morphmarking);
          return Collections.singletonList(newtree);
        }
      }
      else if (MorphLexStripper.morphr.contains(tree.getLabel())) {
        String newmorphmarking = morphmarking; 
        if (morphmr.containsKey(tree.getLabel())) {
          newmorphmarking += morphmr.get(tree.getLabel());
        }
        ArrayList<Tree<String>> transformedChildren = new ArrayList<Tree<String>>();        
        for (Tree<String> child: tree.getChildren()) {
          transformedChildren.addAll(transformTreeHelper(child,newmorphmarking));
        }
        return transformedChildren;
      } else {
        ArrayList<Tree<String>> transformedChildren = new ArrayList<Tree<String>>();
        for (Tree<String> child: tree.getChildren()) {
          transformedChildren.addAll(transformTreeHelper(child,morphmarking));
        }
        Tree<String> newtree = tree.shallowCloneJustRoot();
        newtree.setChildren(transformedChildren);
        return Collections.singletonList(newtree);
      }
    }
    public static String[][] morph_marking_rules = {
      {"PLUR_NOUN_ORULE", "+PL"},
      {"THIRD_SG_FIN_VERB_ORULE", "+3SG"},
      {"PSP_VERB_ORULE","+PSP"},
      {"PAST_VERB_ORULE", "+PAST"},
      {"PRP_VERB_ORULE", "+PRP"},
//      {"BSE_VERB_IRULE", ""},
//      {"NON_THIRD_SG_FIN_VERB_IRULE", ""},
//      {"BSE_OR_NON3SG_VERB_IRULE", ""},
//      {"SING_NOUN_IRULE", ""},
//      {"MASS_NOUN_IRULE", ""},
//      {"MASS_COUNT_IRULE", ""},
//      {"PLUR_NUMCOMP_NOUN_IRULE", ""},
//      {"POS_ADJ_IRULE", ""},
//      {"PASSIVE_ORULE", ""},
//      {"PREP_PASSIVE_ORULE", ""},
//      {"PREP_PASSIVE_TRANS_ORULE", ""},
//      {"CP_PASSIVE_ORULE", ""},
//      {"DATIVE_PASSIVE_ORULE", ""},
            {"W_PERIOD_PLR", "+FS"},
      {"PUNCT_PERIOD_ORULE", "+FS"},
//      {"PUNCT_QMARK_ORULE", ""},
//      {"PUNCT_QQMARK_ORULE", ""},
//      {"PUNCT_QMARK_BANG_ORULE", ""},
      {"PUNCT_COMMA_ORULE", "+COMMA"},
      {"PUNCT_BANG_ORULE", "+FS"},
      {"PUNCT_SEMICOL_ORULE", ""},
      {"PUNCT_RPAREN_ORULE", "+RPR"},
      {"PUNCT_RP_COMMA_ORULE", "+COMMA"},
      {"PUNCT_LPAREN_ORULE", "+LPR"},
      {"PUNCT_RBRACKET_ORULE", "+RPR"},
      {"PUNCT_LBRACKET_ORULE", "+LPR"},
      {"PUNCT_DQRIGHT_ORULE", "+RPR"},
      {"PUNCT_DQLEFT_ORULE", "+LPR"},
      {"PUNCT_SQRIGHT_ORULE", "+RPR"},
      {"PUNCT_SQLEFT_ORULE", "+LPR"},
//      {"PUNCT_HYPHEN_ORULE", ""},
      {"PUNCT_COMMA_INFORMAL_ORULE", "+COMMA"},
//      {"PUNCT_ITALLEFT_ORULE", ""},
//      {"PUNCT_ITALRIGHT_ORULE", ""},
//      {"PUNCT_DROP_ILEFT_ORULE", ""},
//      {"PUNCT_DROP_IRIGHT_ORULE", ""},
//      {"SAILR", ""},
//      {"ADVADD", ""},
//      {"VPELLIPSIS_REF_LR", ""},
//      {"VPELLIPSIS_EXPL_LR", ""},
//      {"INTRANSNG", ""},
//      {"INTR_PP_NG", ""},
//      {"TRANSNG", ""},
//      {"MONTHDET", ""},
//      {"WEEKDAYDET", ""},
//      {"ATTR_ADJ", ""},
//      {"ATTR_ADJ_VERB_PART", ""},
//      {"ATTR_ADJ_VERB_TR_PART", ""},
//      {"ATTR_ADJ_VERB_PSV_PART", ""},
//      {"ATTR_VERB_PART_NAMEMOD", ""},
//      {"ATTR_VERB_PART_TR_NAMEMOD", ""},
//      {"PART_PPOF_AGR", ""},
//      {"PART_PPOF_NOAGR", ""},
//      {"PART_NOCOMP", ""},
//      {"NP_PART_LR", ""},
//      {"DATIVE_LR", ""},
//      {"MINUTE_LR", ""},
//      {"TAGLR", ""},
//      {"BIPART", ""},
//      {"FOREIGN_WD", ""},
//      {"INV_QUOTE", ""},
//      {"NOUN_ADJ_ORULE", ""},
//      {"RE_VERB_PREFIX", ""},
//      {"PRE_VERB_PREFIX", ""},
//      {"MIS_VERB_PREFIX", ""},
//      {"CO_VERB_PREFIX", ""}
    };
    
    public final static HashMap<String,String> morphmr = new HashMap<String,String>(morph_marking_rules.length);
    static {
      for (String[] mapping : morph_marking_rules)
      {
          morphmr.put(mapping[0], mapping[1]);
      }
    }
  }
  
  /**
   * remove_ lexical entry names from the derivation tree.
   * this should be applied before MorphLexMarker transformers.
   *
   */
  public static class LENameStripper implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      List<Tree<String>> roots = transformTreeHelper(tree);
      if (roots.size() != 1)
        return null;
      return roots.get(0);
    }
    public List<Tree<String>> transformTreeHelper(Tree<String> tree) {
      if (tree.isPreTerminal())
        return tree.getChildren();
      else {
        ArrayList<Tree<String>> transformedChildren = new ArrayList<Tree<String>>();
        for (Tree<String> child: tree.getChildren()) {
          transformedChildren.addAll(transformTreeHelper(child));
        }
        Tree<String> newtree = tree.shallowCloneJustRoot();
        newtree.setChildren(transformedChildren);
        return Collections.singletonList(newtree);
      }
    }
  }
  
  public static class LEType2POSStripper implements TreeTransformer<String> {
    public Tree<String> transformTree(Tree<String> tree) {
      final String transformedLabel = transformLabel(tree);
      if (tree.isLeaf()) {
          return tree.shallowCloneJustRoot();
      }
      final List<Tree<String>> transformedChildren = new ArrayList<Tree<String>>();
      for (final Tree<String> child : tree.getChildren()) {
          transformedChildren.add(transformTree(child));
      }
      return new Tree<String>(transformedLabel, transformedChildren);
    }

    public static String transformLabel(Tree<String> tree) {
      String transformedLabel = tree.getLabel();
      if (transformedLabel.toUpperCase().endsWith("_LE")) {
        int cutIndex = transformedLabel.indexOf('_');
        if (cutIndex != -1)
          return transformedLabel.substring(0,cutIndex);
      }
      return transformedLabel;
    }
  }

  public static class RedwoodsTreeNormalizer implements TreeTransformer<String> {
    MorphLexStripper morphLexStripper = new MorphLexStripper();
    LENameStripper leNameStripper = new LENameStripper();
    XOverXRemover<String> xOverXRemover = new XOverXRemover<String>();
    public Tree<String> transformTree(Tree<String> tree) {
      tree = morphLexStripper.transformTree(tree);
      tree = leNameStripper.transformTree(tree);
      tree = xOverXRemover.transformTree(tree);
      return tree;
    }
  }
  public static class RedwoodsTreeRenderer {
    public static String render(Tree<String> tree) {
      StringBuilder sb = new StringBuilder();
      renderTree(tree, sb);
      return sb.toString();
    }
    public static void renderTree(Tree<String> tree, StringBuilder sb) {
      sb.append("(");
      if (tree.isLeaf()) {
        sb.append("\"");
        sb.append(tree.getLabel());
        sb.append("\"");
      } else {
        sb.append(tree.getLabel());
        for (Tree<String> child : tree.getChildren()) {
          sb.append(' ');
          renderTree(child,sb);
        }
      }
      sb.append(")");
    }
  }
  public static class RedwoodsTreeIndentedRenderer {
    public static String render(Tree<String> tree) {
      StringBuilder sb = new StringBuilder();
      renderTree(tree, 0, sb);
      return sb.toString();
    }
    public static void renderTree(Tree<String> tree, int indent,
                                  StringBuilder sb) {
      for (int i = 0; i < indent; i ++)
        sb.append("  ");
      if (tree.isPreTerminal()) {
        sb.append("(");
        sb.append(tree.getLabel());
//        sb.append(" \"");
//        sb.append(tree.getChildren().get(0).getLabel());
//        sb.append("\")");
        sb.append(" ");
        String terminal = tree.getChildren().get(0).getLabel()
                .replace(" ", "_")
                .replace("(", "-LPAR-")
                .replace(")", "-RPAR-");
        if(terminal.isEmpty()) {
          terminal = "-EMPTY-";
        }
        sb.append(terminal);
        sb.append(")");
      } else {
        sb.append("(");
        sb.append(tree.getLabel());
        for (Tree<String> child : tree.getChildren()) {
          sb.append("\n");
          renderTree(child,indent+1,sb);
        }
        sb.append(")");
      }
    }
  }
}
