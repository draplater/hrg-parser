package jigsaw.syntax;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import jigsaw.grammar.AnnotRedwoodsPcfgExtractor;
import jigsaw.treebank.Trees.TreeTransformer;
import jigsaw.util.StringUtils;

public class ERGHeadFinder implements HeadFinder {

  private static final long serialVersionUID = 7927946897317413154L;
  @Override
  public Tree<String> determineHead(Tree<String> t) {
    if (t.isLeaf()) {
      return null;
    }
    List<Tree<String>> children = t.getChildren();
    if (children.size() == 1)
      return children.get(0);
    String rule = AnnotRedwoodsPcfgExtractor.AnnotRemover.transformLabel(t.getLabel());
    if (_headTable.containsKey(rule)) {
      return children.get(_headTable.get(rule));
    }
    return children.get(0); // by default take the left daughter
  }

  /** returns the lexical head (preterminal) of the given tree */
  public Tree<String> determineLexicalHead(Tree<String> t) {
    if (t == null || t.isLeaf())
      return null;
    else if (t.isPreTerminal())
      return t;
    else
      return determineLexicalHead(determineHead(t));
  }
  public static String getLECategory(String letype) {
    String cat = letype;
    if (cat.indexOf('_') != -1)
      cat = cat.substring(0, cat.indexOf('_'));
    return cat;
  }
  private static final Pattern p = Pattern.compile("^([^_]+_[^_]+)_.*");
  public static String getLECatAndComps(String letype) {
    String cat = letype;
    Matcher m = p.matcher(letype);
    if (m.matches()) {
      cat = m.group(1);
    }
    return cat;
  }
  private HashMap<String, Integer> _headTable = null;
  private final static String [] _head_init = {
    "hd-cmp_u_c", "hd-aj_scp_c", "hd-aj_scp-pr_c", "hd-aj_int-unsl_c", "hd-aj_int-sl_c",
    "hd-aj_vmod_c", "hd-aj_vmod-s_c", "hdn-aj_rc_c", "hdn-aj_rc-pr_c", "hdn-aj_redrel_c",
    "hdn-aj_redrel-pr_c", "hdn-np_app_c", "hdn-np_app-pr_c", "hdn-np_app-idf_c",
    "hdn-np_app-idf-p_c", "hdn-np_app-nbr_c", "n-num_mnp_c", "hd-cl_fr-rel_c",
    "mrk-nh_evnt_c", "mrk-nh_cl_c", "mrk-nh_ajlex_c", "mrk-nh_nom_c", "mrk-nh_n_c",
    "mrk-nh_atom_c", "vp-vp_crd-fin-t_c", "vp-vp_crd-fin-m_c", "vp-vp_crd-fin-im_c",
    "vp-vp_crd-nfin-t_c", "vp-vp_crd-nfin-m_c", "vp-vp_crd-nfin-im_c", "v-v_crd-fin-ncj_c",
    "cl-cl_crd-t_c", "cl-cl_crd-int-t_c", "cl-cl_crd-m_c", "cl-cl_crd-im_c",
    "cl-cl_crd-rc-t_c", "pp-pp_crd-t_c", "pp-pp_crd-m_c", "pp-pp_crd-im_c", "r-r_crd-t_c",
    "r-r_crd-m_c", "r-r_crd-im_c", "np-np_crd-t_c", "np-np_crd-i-t_c", "np-np_crd-i2-t_c",
    "np-np_crd-i3-t_c", "np-np_crd-m_c", "np-np_crd-im_c", "np-np_crd-nc-t_c",
    "np-np_crd-nc-m_c", "n-n_crd-nc-m_c", "n-n_crd-t_c", "n-n_crd-2-t_c", "n-n_crd-3-t_c",
    "n-n_crd-m_c", "n-n_crd-im_c", "n-n_crd-asym-t_c", "n-n_crd-asym2-t_c", "n-j_crd-t_c",
    "j-n_crd-t_c", "j-j_crd-att-t_c", "j-j_crd-prd-t_c", "j-j_crd-prd-m_c",
    "j-j_crd-prd-im_c", "jpr-jpr_crd-t_c", "jpr-jpr_crd-m_c", "jpr-jpr_crd-im_c",
    "jpr-vpr_crd-t_c", "jpr-vpr_crd-m_c", "jpr-vpr_crd-im_c", "vppr-vppr_crd-t_c",
    "vppr-vppr_crd-m_c", "vppr-vppr_crd-im_c", "vpr-vpr_crd-t_c", "vpr-vpr_crd-m_c",
    "vpr-vpr_crd-im_c", "ppr-ppr_crd-t_c", "ppr-ppr_crd-m_c", "ppr-ppr_crd-im_c",
    "hd-hd_rnr_c", "np-aj_frg_c", "np-aj_rorp-frg_c", "np-aj_j-frg_c", "nb-aj_frg_c",
    "pp-aj_frg_c", "j-aj_frg_c", "hdn-cl_prnth_c", "hdn-n_prnth_c", "hdn-cl_dsh_c",
    "hd-pct_c", "cl-cl_runon_c", "cl-cl_runon-cma_c", "cl-np_runon_c", "cl-np_runon-prn_c"
  };
  private final static String [] _head_final = {
    "sb-hd_mc_c", "sb-hd_nmc_c", "sb-hd_q_c", "sp-hd_n_c", "sp-hd_hc_c", "aj-hd_scp_c",
    "aj-hd_scp-xp_c", "aj-hd_scp-pr_c", "aj-hd_int_c", "aj-hd_adjh_c", "aj-hd_int-inv_c",
    "aj-hd_int-rel_c", "aj-hdn_norm_c", "aj-hdn_adjn_c", "flr-hd_nwh_c",
    "flr-hd_nwh-nc_c", "flr-hd_wh-mc_c", "flr-hd_wh-mc-sb_c", "flr-hd_wh-nmc-fin_c",
    "flr-hd_wh-nmc-inf_c", "flr-hd_rel-fin_c", "flr-hd_rel-inf_c", "np-hdn_cpd_c",
    "np-hdn_ttl-cpd_c", "np-hdn_nme-cpd_c", "np-hdn_num-cpd_c", "np-hdn_cty-cpd_c",
    "n-hdn_cpd_c", "n-hdn_j-n-cpd_c", "n-hdn_ttl-cpd_c", "n-nh_vorj-cpd_c",
    "n-nh_j-cpd_c", "j-n_n-ed_c", "num-n_mnp_c", "np-prdp_vpmod_c", "aj-np_frg_c",
    "aj-np_int-frg_c", "aj-pp_frg_c", "aj-r_frg_c", "w-w_fw-seq-m_c", "w-w_fw-seq-t_c"
  };
  public ERGHeadFinder() {
    _headTable = new HashMap<String, Integer>();
    for (String rule : _head_init)
      _headTable.put(rule, 0);
    for (String rule : _head_final)
      _headTable.put(rule, 1);
  }
  public static boolean isCR(String rulename) {
    return rulename!=null && rulename.endsWith("_c");
  }
  public static boolean isLR(String rulename) {
    return rulename!=null && rulename.endsWith("lr");
  }
  public static boolean isPLR(String rulename) {
    return rulename!=null && rulename.endsWith("_plr");
  }
  public static boolean isOLR(String rulename) {
    return rulename!=null && rulename.endsWith("_olr");
  }
  public static boolean isODLR(String rulename) {
    return rulename!=null && rulename.endsWith("_odlr");
  }
  public static boolean isDLR(String rulename) {
    return rulename!=null && rulename.endsWith("_dlr");
  }
  public static boolean isILR(String rulename) {
    return rulename!=null && rulename.endsWith("_ilr");
  }
  public static boolean isOrthChangingLR(String rulename) {
    return rulename!=null && (isOLR(rulename) || isODLR(rulename));
  }
  public static boolean isOrthInvariantLR(String rulename) {
    return rulename!=null && (isILR(rulename) || isDLR(rulename));
  }
  public static boolean isInflLR(String rulename) {
    return rulename!=null && (isILR(rulename) || isOLR(rulename));
  }
  public static boolean isDerivLR(String rulename) {
    return rulename!=null && (isDLR(rulename) || isODLR(rulename));
  }

  public static boolean isPrefixPLR(String rulename) {
    if (!isPLR(rulename))
      return false;
    if (rulename.equals("w_lparen_plr") || rulename.equals("w_lbrack_plr") ||
        rulename.equals("w_dqleft_plr") || rulename.equals("w_sqleft_plr") ||
        rulename.equals("w_italleft_plr") || rulename.equals("w_drop-ileft_plr"))
      return true;
    return false;
  }

  public static boolean isSuffixPLR(String rulename) {
    if (!isPLR(rulename))
      return false;
    return !isPrefixPLR(rulename);
  }

  public static class PUNCTForker implements TreeTransformer<String> {
    private static String [][] _plrs =
    {
      { "w_period_plr", "punct_period", "suffix", "." },
      { "w_qmark_plr", "punct_qmark", "suffix", "?" },
      { "w_qqmark_plr", "punct_qmark", "suffix", "?" },
      { "w_qmark-bang_plr", "punct_bang", "suffix", "!" },
      { "w_comma_plr", "punct_comma", "suffix", "," },
      { "w_bang_plr", "punct_bang", "suffix", "!" },
      { "w_semicol_plr", "punct_semicol", "suffix", ";" },
      { "w_double_semicol_plr", "punct_semicol", "suffix", ";;" },
      { "w_rparen_plr", "punct_rparen", "suffix", ")" },
      { "w_comma-rp_plr", "punct_rparen", "suffix", ",)" },
      { "w_lparen_plr", "punct_lparen", "prefix", "(" },
      { "w_rbrack_plr", "punct_rbrack", "suffix", "]", "}" },
      { "w_lbrack_plr", "punct_lbrack", "prefix", "[", "{" },
      { "w_dqright_plr", "punct_rdq", "suffix", "”", "\"", "''" },
      { "w_dqleft_plr", "punct_ldq", "prefix", "“", "\"", "``" },
      { "w_sqright_plr", "punct_rsq", "suffix", "’", "'"},
      { "w_sqleft_plr", "punct_lsq", "prefix", "‘", "'", "`" },
      { "w_hyphen_plr", "punct_hyphen", "suffix", "-" },
      { "w_comma-nf_plr", "punct_comma", "suffix", "," },
      { "w_italleft_plr", "punct_lit", "prefix", "¦i" },
      { "w_italright_plr", "punct_rit", "suffix", "i¦", "”", "\"", "''" },
      { "w_drop-ileft_plr", "punct_lit", "prefix", "¦i" },
      { "w_drop-iright_plr", "punct_rit", "suffix", "i¦" },
      { "w_threedot_plr", "punct_threedot", "suffix", "..."},
      { "w_asterisk_plr", "punct_asterisk", "suffix", "*"},
      { "w_asterisk_pre_plr", "punct_asterisk_pre", "prefix", "*"},
      { "w_asterisk-pre_plr", "punct_asterisk-pre", "prefix", "*"},
    };
    private HashMap<String,String[]> _plrmap = null;

    public PUNCTForker() {
      _plrmap = new HashMap<String,String[]>();
      for (String [] plr : _plrs) {
        _plrmap.put(plr[0], plr);
      }
    }
    public Tree<String> transformTree(Tree<String> tree) {
      Tree<String> newtree = null;
      String label = tree.getLabel();
      if (isPLR(label)) {
        String yieldFull = StringUtils.join(tree.getYield(), " ");
        String[] tmp = StringUtils.rsplit(yieldFull, "#__#", 1);
        String yield = tmp[0];
        String spanRepr = tmp[1];
        Pattern p = Pattern.compile("\\[(\\d+),(\\d+)\\]");
        Matcher m = p.matcher(spanRepr);
        if(!m.find()) {
          System.out.println(spanRepr);
        }
        int spanStart = Integer.parseInt(m.group(1));
        int spanEnd = Integer.parseInt(m.group(2));
        Tree<String> terminal = tree.getTerminals().get(0);
        String [] plr = _plrmap.get(label);
        if(plr == null) {
            System.out.println(label);
        }
        String stem = yield;
        String affix = "_";
        boolean isPrefix = plr[2].equals("prefix");
        for (int i = 3; i < plr.length; i ++) {
          if (!isPrefix && yield.endsWith(plr[i])) {
            affix = plr[i];
            stem = yield.substring(0,yield.length()-plr[i].length());
            break;
          } else if (isPrefix && yield.startsWith(plr[i])) {
            affix = plr[i];
            stem = yield.substring(plr[i].length());
            break;
          }
        }
        terminal.setLabel(stem + "#__#" + spanRepr);
        String puctAdditionalInfo;
        if(isPrefix) {
          puctAdditionalInfo = String.format("#__#[%d,%d]", spanStart, spanStart);
        } else {
          puctAdditionalInfo = String.format("#__#[%d,%d]", spanEnd, spanEnd);
        }
        Tree<String> pnct = new Tree<String>(affix + puctAdditionalInfo);
        Tree<String> pnct_pt = new Tree<String>(plr[1],new ArrayList<Tree<String>>());
        pnct_pt.getChildren().add(pnct);
        Tree<String> newchild  = transformTree(tree.getChildren().get(0));
        newtree = new Tree<String>(newchild.getLabel(), new ArrayList<Tree<String>>());
        newtree.getChildren().add(newchild);
        if (isPrefix)
          newtree.getChildren().add(0, pnct_pt);
        else
          newtree.getChildren().add(pnct_pt);
      } else {
        newtree = new Tree<String>(tree.getLabel(), new ArrayList<Tree<String>>());
        for (Tree<String> child : tree.getChildren())
          newtree.getChildren().add(transformTree(child));
      }
      return newtree;
    }

    public static String[] splitPunct(String yield, String plr) {
      String [] result = new String[2];
      return result;
    }
  }
}
