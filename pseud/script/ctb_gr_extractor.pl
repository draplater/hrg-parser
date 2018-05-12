#!/usr/bin/perl
#
# Function: 
#   Deriving deep grammatical relations from CTB annotations or
#     function-tag-enriched PCFG parser outputs.
#
# Note:
#   The "tapping implict grammatical function information" part 
#      is mainly based on Dr. Xue's dependency tree extraction script.
#  
#
# Author: an NLPer who is also a Perl lover.
# Start: 19/12/2013
#

use Encode;

use strict;
use Getopt::Std;

#///////////////////////////////
#//  LinguaView's XML markup  //
#///////////////////////////////
my $lingua_view_begin = <<EOT
<?xml version="1.0" encoding="utf-8" ?>
<viewer>
EOT
;
my $lingua_view_end = <<EOT
</viewer>
EOT
;

my ($help) = <<EOT

   [perl] $0 [OPTIONS] 

  This script enriches CTB annotations and derives deep grammatical relations
  from it.

  Options:
     -d    <directory>  ctb directory
     -D    <directory>  output directory
     -g                 present GPSG tree

     -c    <filename>   collect grammatical relations only
                        the input file contains "enriched" CTB trees
     -t                 trace back (together with -c)

     -e                 output dependency graph with empty category
     -s                 show file list and exit

     -v                 show the version number
     -h                 print this help text and exit

EOT
;

use vars qw($opt_d $opt_D $opt_v $opt_h $opt_g $opt_s $opt_e $opt_c $opt_t);
if (!getopts('d:D:c:etsgvh') || defined($opt_h)) {
  exit_with_msg($help);
}

if (defined($opt_v)) {
  exit_with_msg("version 1.0\n");
}

if (defined($opt_s)) {
  my $msg = "Training files:\n    ";
  $msg .= (join ", ", conlltrain()); 
  $msg .= "\n";
  $msg .= "Development files\n    ";
  $msg .= (join ", ", conlldev());
  $msg .= "\n";
  $msg .= "Test files:\n    ";
  $msg .= (join ", ", conlltest());
  $msg .= "\n";
  exit_with_msg("\n");
}

if (not defined($opt_c) and not defined($opt_d) and not defined($opt_D)) {
  exit_with_msg($help);
}

# Global var
my @tuples;
my $min_valid_trace_id = 1000;
my $delimiter = "#";

if (defined($opt_c)) {
  my $xmlfile = $opt_c . ".xml";
  my $conllfile = $opt_c . ".conll";

  open(OUT, ">$xmlfile");        # To the input of our LinguaView
  open(COUT, ">$conllfile");   # To CoNLL 2008 ST format 
  print OUT $lingua_view_begin;
  process_one_file("$opt_c", \*OUT, \*COUT);
  print OUT $lingua_view_end;
  close OUT; 
  close COUT;

  exit;
}

my $ctbdir = $opt_d;
die "BAD ctb directory $opt_d\n" if (not -e $ctbdir);

my $outdir = $opt_D;
`mkdir $outdir` if (not -e $outdir);
`mkdir $outdir/trn` if (not -e "$outdir/trn");
`mkdir $outdir/dev` if (not -e "$outdir/dev");
`mkdir $outdir/tst` if (not -e "$outdir/tst");
`mkdir $outdir/other` if (not -e "$outdir/other");

# Split training/development/test data sets according to CoNLL 2009 shared task.
my @trnflist = conlltrain();
my @tstflist = conlltest();
my @devflist = conlldev();
my @othflist = conlldev();


#/////////////////
#//  Main Loop  // 
#/////////////////

foreach my $ctbfile (split(/\n/, `ls -1 $ctbdir`)) { 
  $ctbfile =~ /(\d\d\d\d)/;
  my $file_id = $1;
  my $subdir;
  if    ($file_id ~~ @trnflist) { $subdir = "trn"; } 
  elsif ($file_id ~~ @tstflist) { $subdir = "tst"; } 
  elsif ($file_id ~~ @devflist) { $subdir = "dev"; } 
  else                          { $subdir = "other"; } 

  open(OUT, ">$outdir/$subdir/$ctbfile.xml");        # To the input of our LinguaView
  open(COUT, ">$outdir/$subdir/$ctbfile.conll08");   # To CoNLL 2008 ST format 
  print OUT $lingua_view_begin;
  process_one_file("$ctbdir/$ctbfile", \*OUT, \*COUT);
  print OUT $lingua_view_end;
  close OUT; 
  close COUT;
}

#////////////////////
#//  Subfunctions  // 
#////////////////////

sub exit_with_msg {
  my $msg = shift;
  print $msg;
  exit;
}

# process one ctb .fid file
sub process_one_file {
  my ($file, $outfh, $conllfh) = @_;
  open IF, "$file";

  my $sent_id = 0;
  my $tbn;
  while (defined($tbn = TBNode::nextSentence(\*IF))) {
    $sent_id ++;
    my $firstchild = $tbn->firstChild();
    my @terms_orig = $firstchild->terminals();
    if ($firstchild) {
      if (defined($opt_d)) {
        # add a unique function tag to each non-terminal
        add_grammatical_function_to_parse_node($firstchild);
  
        # correct some function tags (the ones stored in the GRAMREL, rather than rawLabel)
        correct_annotation_error_of_function_tag($firstchild);
  
        # make the tree less flat
        heighten($firstchild);
        heighten($firstchild);
        #$tbn->re_parent();
  
        # raising/control: This kind of bounded non-local dependecies are not annotated by the ctb.
        add_trace_for_control_raising_construction($firstchild);
  
        # relative construction: tracing empty categories under an IP whose "boss" is a CP#relative
        add_trace_for_relative_construction_via_path($firstchild);
  
        # tracing big PRO under LCP
        add_trace_for_lcp_via_path($firstchild);
  
        # tracing big PRO generally
        trace_big_pro_via_path($firstchild);
        #link_big_pro_to_big_pro_c_command_it($firstchild);
  
        # deal with BA/Long-Bei
        deal_with_babei_structure($firstchild);
  
        # deal with short bei
        deal_with_short_bei($firstchild);
  
        # deal with short bei
        deal_with_serial_verbal_construction_easy($firstchild);
  
      } elsif (defined($opt_c)) {

        # TODO: There is a bug to be fixed
        if (defined($opt_t)) {
          trace_a_gpsg_tree($tbn);
        }
      }
      # bottom-up passing head words to each non-terminal
      update_headword_index($tbn);

      my @terms = $tbn->terminals();
      my @term_words_no_ec = ("#ROOT#");
      my @term_poss_no_ec = ("#ROOT#");
      my @termidx_map = ();
      my $term_idx = 0;
      foreach my $term (@terms) {
	if ($term->isTrace()) {
          push @termidx_map, -2;
        } else {
          push @termidx_map, $term_idx;
          push @term_words_no_ec, $term->data();
          push @term_poss_no_ec, $term->label();
          $term_idx ++;
        }
      }

      # output basic lexical information. 
      printf $outfh "  <sentence id = \"%i\">\n", $sent_id;
      printf $outfh "  <wordlist length = \"%i\">\n", scalar(@terms);
      if (defined($opt_e)) {
        my $wordi = 0;
        foreach my $term (@terms) {
          printf $outfh "    <tok id=\"%i\" pos=\"%s\" head=\"%s\" />\n", $wordi, $term->label(), $term->data();
          $wordi ++;
        }
      } else {
        for (my $wi = 0; $wi <= $#terms; $wi ++) { 
          if ($termidx_map[$wi] >= 0) {
            printf $outfh "    <tok id=\"%i\" pos=\"%s\" head=\"%s\" />\n", 
            $termidx_map[$wi], $terms[$wi]->label(), $terms[$wi]->data();
          }
        }
      }
      printf $outfh "  </wordlist>\n";

      # output constitutent tree
      $firstchild->map_c_and_f_struct();
      my $tree_str;
      if (defined($opt_g)) {
        $tbn->to_gpsg_tree();
        $tree_str = $tbn->to_noec_str();
      } else {
        $tree_str = $tbn->to_str();
      }
      $tree_str =~ s/  */ /g;
      $tree_str =~ s/TOP$delimiter//; # The TOP node is not supported by current version of our LinguaView program.
      printf $outfh "  <constree>\n";
      printf $outfh "    %s\n", $tree_str;
      printf $outfh "  </constree>\n";

#     my $func_tree_str = $firstchild->to_func_tree_str();
#     $func_tree_str =~ s/  */ /g;
#     printf $outfh "  <functree>";
#     printf $outfh "    %s", $func_tree_str;
#     printf $outfh "  </functree>\n";

      # extract bi-lexical dependency pairs
      @tuples = ();
      collect_tuples($firstchild, \@tuples, \@terms);

      # output dependency graph
      my %pairs = ();
      printf $outfh "  <deepdep>\n"; 
      foreach my $tuple (@tuples) {
        my ($hi, $di, $rel) = (split "\t", $tuple);
        if (defined($opt_e)) {
          printf $outfh  "    (%i, %i, %s)\n", $hi, $di, $rel; 
        } else {
          if ($termidx_map[$hi] >= 0 and $termidx_map[$di] >= 0) {
            printf $outfh  "    (%i, %i, %s)\n", $termidx_map[$hi], $termidx_map[$di], $rel; 
            $pairs{$termidx_map[$hi]+1}{$termidx_map[$di]+1} = $rel; 
          }
        }
      }
      foreach my $hi ($firstchild->headIdx()) {
        if (defined($opt_e)) {
          printf $outfh  "    (%i, %i, %s)\n", -1, $hi, "ROOT"; 
        } else {
          if ($termidx_map[$hi] >= 0) {
            printf $outfh  "    (%i, %i, %s)\n", -1, $termidx_map[$hi], "ROOT"; 
            $pairs{0}{$termidx_map[$hi]+1} = "ROOT";
          }
        }
      }
      printf $outfh "  </deepdep>\n"; 
      printf $outfh "  </sentence>\n";

      if (not defined($opt_e)) {
        # represent as propsitions
        my $props = SRL::props->new;
        $props->init;
        $props->{length} = scalar(@term_words_no_ec);
        foreach my $pred_posi (sort {$a <=> $b} (keys %pairs)) {
          my $prop = SRL::prop->new;
          $prop->{predpos} = $pred_posi;
          $prop->{pred} = $term_words_no_ec[$pred_posi];
          foreach my $arg_posi (keys %{$pairs{$pred_posi}}) {
            my $arg = new SRL::phrase;
            $arg->{start} = $arg_posi;
            $arg->{end} = $arg_posi;
            $arg->{type} = $pairs{$pred_posi}{$arg_posi};
            $prop->add_arg($arg);
          }
          $props->add_prop($prop);
        }
        my @output_str_list = $props->to_SE_props();
        for (my $k = 0; $k <= $#term_words_no_ec; $k ++) {
          printf $conllfh "%i %s %s %s %s _ _ _ _ _ %s\n", ($k+1),      # ID
                          $term_words_no_ec[$k], $term_words_no_ec[$k], # Word 
                          $term_poss_no_ec[$k], $term_poss_no_ec[$k],   # POS
                          $output_str_list[$k];                         # As predicate-argument tuple
        }
        print $conllfh "\n";
      }
    }

    if (defined($opt_d)) {
      my @terms_modi = $firstchild->terminals();
      if (scalar(@terms_orig) != scalar(@terms_modi)) { # check once more.
        print STDERR "ERROR!\n";
      }
    }
  }
  close IF;
}

#
# recursively process parse tree nodes; add explict and implict function tags.
#
sub add_grammatical_function_to_parse_node {
  my $node = shift;

  if ($node->isTerminal()){ 
    #this is where to add auxiliaries for verbs, if there are any 
  } else {
    my $synt_cat = $node->label();
    my $relation_headaddr = get_head_and_relation($node);
    my ($relation, $headaddr) = split(/\|/, $relation_headaddr);
    #print $relation_headaddr , "\t", $relation, "\t", $headaddr, "\n";

    my @children = $node->children();
    # whether new nodes are added to make the whole tree less flat.
    my $change_topo_struc = 0;
    $change_topo_struc = 1 if ($synt_cat eq "NP");
    if ($change_topo_struc) {
      if ($synt_cat eq "NP") { 
        if ($relation eq "comp") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "COMP"); 
          foreach my $child (@children) { 
            add_grammatical_function_to_parse_node($child);
          } 
        } elsif ($relation eq "flat") { 
          foreach my $child (@children){ 
            if ($headaddr eq $child->termno().":".$child->height()) {
              $child->iAmHeadPhrase(); 
            } elsif ($child->label() eq "PU") { 
              # pass
            } elsif ($child->label() eq "DT") {
              $child->gramRel("DMOD"); 
            } elsif ($child->label() eq "JJ") {
              $child->gramRel("AMOD"); 
            } elsif($child->label() eq "NN" || $child->label() eq "NR" || $child->label() eq "NT") {
              $child->gramRel("NMOD"); 
            } else {
              $child->gramRel("UNSPEC"); 
            } 
            add_grammatical_function_to_parse_node($child);
          } 
        } elsif($relation eq "adj") { 
          foreach my $child (@children){ 
            if ($headaddr eq $child->termno().":".$child->height()) {
              $child->iAmHeadPhrase(); 
            } elsif ($child->label() eq "PU") {
              # pass 
            } elsif ($child->label() eq "DP" || $child->label() eq "QP") {
              $child->gramRel("DMOD"); 
            } elsif($child->label() eq "CP") {
              $child->gramRel("RELATIVE"); 
            } elsif($child->label() eq "IP") {
              $child->gramRel("APP"); 
            } elsif($child->label() eq "ADJP") {
              $child->gramRel("AMOD"); 
            } elsif($child->label() eq "PP") {
              $child->gramRel("AMOD"); 
            } elsif($child->label() eq "NP" || $child->label() eq "DNP") {
              $child->gramRel("NMOD"); 
            } else {
              my $ftag=get_function_tag($child); 
              $ftag = "UNSPEC" if ($ftag eq "NONE");
              $child->gramRel($ftag); 
            } 
            add_grammatical_function_to_parse_node($child);
          }
        } elsif ($relation eq "coord") { 
          foreach my $child (@children) {
            add_grammatical_function_to_parse_node($child); 
          }

          my @new_children = ();

          for (my $i=0; $i<=$#children; $i++) { 
            my $child = $children[$i]; 
            if (is_conj($child)) { 
              $child->iAmConjunction(); 
              push @new_children, $child;
            } elsif (not $child->isTerminal()) { 
              $child->iAmConjunct(); 
              push @new_children, $child;
            } else { 
              # heighten flat NP coordination structures 
              # i.e. combining multiple terminals (wsun: be careful!)

              my $new_np_node = TBNode->new("NP");
              $new_np_node->termno($child->termno());
              $new_np_node->iAmConjunct(); 
              push @new_children, $new_np_node;

              my $end = $i; 
              my $ch = $child; 
              while (defined($ch) and ($ch->isTerminal()) and !is_conj($ch)) { 
                $end++; 
                $ch = $children[$end]; 
              } 
              $end--;

              for (my $k = $i; $k < $end; $k ++) {
                $children[$k]->gramRel("NMOD");
                $new_np_node->addChild($children[$k]);
              } 
              $children[$end]->iAmHeadPhrase();
              $new_np_node->addChild($children[$end]);

              $i = $end;
            }
          } 
          $node->clean_children();
          foreach my $new_chd (@new_children) {
            $node->addChild($new_chd);
          }
          
#         print STDERR "} ", $node->show(), "\n";
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC"); 
          foreach my $child ($node->children()) { 
            add_grammatical_function_to_parse_node($child);
          }		
        } 
      }
    } else {
      if ($synt_cat eq "IP") {
        if ($relation eq "pred") {
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif ($relation eq "coord") { 
          update_gramrel_coordination_hard(\@children); 
        } elsif ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "ADV");
        } 
      } elsif ($synt_cat eq "VP") { 
        if ($relation eq "comp") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "COMP"); 
        } elsif ($relation eq "coord") { 
          update_gramrel_coordination_hard(\@children); 
        } elsif ($relation eq "flat") {
          update_gramrel_verb_complement_easy(\@children, $headaddr, "OTHER"); 
        } elsif($relation eq "aux"){ 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "AUX"); 
        } elsif ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif ($synt_cat eq "VRD") { 
        if ($relation eq "comp") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } elsif($relation eq "flat") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } else { 
          update_gramrel_head_initial_easy(\@children, "UNSPEC"); 
        } 
      } elsif ($synt_cat eq "VSB") { 
        if ($relation eq "comp") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } elsif ($relation eq "flat") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC"); 
        } 
      } elsif ($synt_cat eq "VNV") { 
        if ($relation eq "comp") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } elsif($relation eq "flat") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } else { 
          update_gramrel_head_initial_easy(\@children, "UNSPEC"); 
        } 
      } elsif($synt_cat eq "VCD") { 
        if ($relation eq "coord") { 
          update_gramrel_verb_coordination_easy(\@children)
        } elsif ($relation eq "flat") { 
          my $firstchild = $children[0]; 
          #reset the headaddr 
          $headaddr = $firstchild->termno().":".$firstchild->height(); 
  
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } else { 
          update_gramrel_head_initial_easy(\@children, "UNSPEC"); 
        } 
      } elsif ($synt_cat eq "VPT") { 
        if ($relation eq "comp") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } elsif ($relation eq "flat") { 
          update_gramrel_verb_complement_easy(\@children, $headaddr, "PRT"); 
        } else { 
          update_gramrel_head_initial_easy(\@children, "UNSPEC"); 
        } 
      } elsif ($synt_cat eq "UCP") { 
        if ($relation eq "coord") { 
          update_gramrel_coordination_easy(\@children)
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC"); 
        } 
      } elsif ($synt_cat eq "PP") { 
        if ($relation eq "comp") { 
          foreach my $child (@children) { 
            if ($headaddr eq $child->termno().":".$child->height()) {
              $child->iAmHeadPhrase(); 
            } elsif ($child->label() eq "PU") { 
              # pass
            } elsif ($child->label() eq "NP") {
              $child->gramRel("OBJ"); 
            } else {
              $child->gramRel("COMP"); 
            } 
          } 
        } elsif ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif ($synt_cat eq "DP") { 
        if ($relation eq "comp") { 
          foreach my $child (@children) { 
            if ($headaddr eq $child->termno().":".$child->height()) {
              $child->iAmHeadPhrase(); 
            } elsif($child->label() eq "PU"){
              # pass
            } elsif ($child->label() eq "NP") {
              $child->gramRel("OBJ"); 
            } else {
              $child->gramRel("COMP"); 
            } 
          } 
        } elsif ($relation eq "flat") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif ($synt_cat eq "QP") { 
        if ($relation eq "comp") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "COMP"); 
        } elsif($relation eq "coord") { 
          update_gramrel_coordination_easy(\@children)
        } elsif ($relation eq "flat") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif ($synt_cat eq "CLP") { 
        if ($relation eq "comp") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "COMP"); 
        } elsif($relation eq "flat") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif ($synt_cat eq "WHNP" or $synt_cat eq "WHPP") { 
        if (@children == 1) {
          $children[0]->iAmHeadPhrase();
        } else {
          # pass
        }
      } elsif ($synt_cat eq "ADJP") {
        if ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif($relation eq "coord"){ 
          update_gramrel_coordination_hard(\@children); 
        } elsif($relation eq "flat") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else {
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif($synt_cat eq "ADVP") { 
        if ($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif ($relation eq "coord") { 
          if ($#children == 1) { #there is only one child, make it the head
            my $child = $children[0];
            $child->iAmHeadPhrase(); 
          } else { 
            update_gramrel_coordination_hard(\@children); 
          }
        } elsif ($relation eq "flat") {
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else {
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        }
      } elsif ($synt_cat eq "CP") {
        if ($relation eq "comp") {
          foreach my $child (@children) {
            if ($headaddr eq $child->termno().":".$child->height()) { 
              #$child->gramRel("SUBORD"); 
              $child->iAmHeadPhrase();
            } elsif ($child->label() eq "PU") {
              # pass
            } else {
              $child->gramRel("COMP"); 
            } 
          } 
        } elsif($relation eq "adj") { 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        } 
      } elsif($synt_cat eq "LCP") { 
        if($relation eq "comp"){ 
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "COMP"); 
        } elsif ($relation eq "adj") {
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        }
      } elsif ($synt_cat eq "DNP"){
        if($relation eq "comp"){
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "COMP"); 
        } else { 
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        }
      } elsif ($synt_cat eq "DVP"){
        if($relation eq "comp"){
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "COMP"); 
        } else {
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        }
      } elsif ($synt_cat eq "PRN"){
        if($relation eq "adj"){
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } elsif ($relation eq "flat"){
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "ADV"); 
        } else {
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        }
      } elsif ($synt_cat eq "FRAG") {
        if($relation eq "flat"){
          update_gramrel_head_nonhead_easy(\@children, $headaddr, "UNSPEC"); 
        } else {
          update_gramrel_head_final_easy(\@children, "UNSPEC");
        }
      } else { 
        update_gramrel_head_final_easy(\@children, "UNSPEC");
      }

      # Recursively add grammatical relations to its all child nodes;
      foreach my $child (@children){
        add_grammatical_function_to_parse_node($child);
      }
    }
  }
}

#
# Some original function tag annotations are problematic.
# Let's correct them.
#
sub correct_annotation_error_of_function_tag {
  my $node = shift;  

  if ($node->isTerminal()){ 
  } else {
    my $synt_cat = $node->label();
    my @children = $node->children();
    if ($synt_cat eq "VP") {
      foreach my $child (@children) {
        if ($child->label() eq "IP") {
          # Sometimes, the function tag of the IP under a VP is annotated as a OBJ.
          $child->gramRel("COMP");
        }
      }
    }
    if ($synt_cat eq "IP" and @children == 1 and $children[0]->label() eq "IP") {
      $node->children($children[0]->children());
      foreach my $chd ($node->children()) {
        $chd->parent($node);
      }
    }

    foreach my $child (@children){
      correct_annotation_error_of_function_tag($child);
    }
  }
}

#
# Take the raising/control structure as bounded dependency.
# Thus find the SUBJECT of the outside verb as the missing one of the embedded clause.
#
sub add_trace_for_control_raising_construction {
  my $node = shift;  

  if ($node->isTerminal()){ 
  } else {
    my $synt_cat = $node->label();
    my @children = $node->children();
    if ($synt_cat eq "VP") {
      if ($#children == 2) {
        my $head_verb = $children[0];
        my $middle_node = $children[1];
        my $ip_comp = $children[2];
        if ($head_verb->label() =~ /^V/ and $middle_node->label eq "NP" and $ip_comp->label() eq "IP") {
          # The structure we tackle is IP->VV NP-OBJ IP-COMP
          my @terms_of_ip = $ip_comp->terminals();
          if ($terms_of_ip[0]->data() eq "*PRO*") {
            my $big_pro = $terms_of_ip[0];
            my @terms_of_middle = $middle_node->terminals();
            if (scalar(@terms_of_middle) == 1 and $terms_of_middle[0]->isTrace()) {
              # Check the middle node: If it is a empty cat, link the big PRO with "outside subject."
              my $npar = $node->parent();
              while (defined($npar) and ($npar->label() ne "IP")) {
                $npar = $npar->parent();
              }
              if (defined($npar) and $npar->label() eq "IP") {
                foreach my $nchd ($npar->children()) {
                  if ($nchd->gramRel() eq "SBJ") {
                    $big_pro->data($big_pro->data() . "-" . $min_valid_trace_id);
                    $nchd->rawLabel($nchd->rawLabel() . "-". $min_valid_trace_id);
                    $min_valid_trace_id ++;

                    last;
                  }
                }
              }
            } else {
              # Check the middle node: If it is not a empty cat, link it with the big PRO.
              $big_pro->data($big_pro->data() . "-" . $min_valid_trace_id);
              $middle_node->rawLabel($middle_node->rawLabel() . "-" . $min_valid_trace_id);
              $min_valid_trace_id ++;
            }
          }
        }
      }
    }

    foreach my $child (@children){
      add_trace_for_control_raising_construction($child);
    }
  }
}

#
# Thus find the SUBJECT of the outside verb as the missing one of the embedded clause.
# An old version to deal with the relative construction.
# DO NOT USE IT as possible as you can.
#
sub add_trace_for_dec_construction {
  my $node = shift;  

  if ($node->isTerminal()){ 
  } else {
    my $synt_cat = $node->label();
    my @children = $node->children();
    if ($synt_cat eq "CP") {
      if ($#children == 1) {
        my $ip = $children[0];
        my $dec = $children[1];
        if ($ip->label() eq "IP" and $dec->label() eq "DEC") {
          # Make sure there is a modified.
          my $npar = $node->parent();
          while (defined($npar) and $npar->label() ne "NP") {
            $npar = $npar->parent();
          }
          if ($npar->label() eq "NP" and $npar->children() >= 2) {
            my @np_children = $npar->children();
            my $np_head = $np_children[-1];

            my @ip_children = $ip->children();
            # Check subject first, if it is a trace without link to any "real" constitute,
            # link it with the modified of the DE-construction. 
            # If the subject is not a trace of a alreadly linked trace, check object(s) 
            # in a similar way. 
            # Yes! this solution doesn't consider the case that both sbj and objs are empty.
            # Let me improve it in future.
            my $subj = $ip_children[0];
            my @terms_of_subj = $subj->terminals();
  
            my $update = 0;
            if (@terms_of_subj == 1 and $terms_of_subj[0]->data() =~ /\*T\*/) {
              my $trace = $terms_of_subj[0];

              if ($trace->traceIdEquivsButNonTrace() == 0) {
                # Link the np-head and the subject
                $np_head->rawLabel($np_head->rawLabel() . "-" . $trace->traceId());
                $update = 1;
              }
            }
            if (not $update) {
              # find the vp
              my $vp_complementation = $ip_children[-1];
              my @objs = ();
              while (defined($vp_complementation) and (not $vp_complementation->isTerminal())) {
                my @tmp_children = $vp_complementation->children();
                if ($tmp_children[0]->isHead()) {
                  for (my $k = 1; $k <= $#tmp_children; $k ++) {
                    if ($tmp_children[$k]->only_contains_one_trace()) {
                      my @tmp_terms = $tmp_children[$k]->terminals();
                      if ($tmp_terms[0]->data() =~ /\*T\*/) {
                        push @objs, $tmp_terms[0];
                      }
                    }
                  }
                  last;
                } else {
                  $vp_complementation = $tmp_children[-1];
                }
              }
              if (@objs > 0) {
                my $trace = $objs[0]; # only update the first one and I suppose there is only one.
                if ($trace->traceIdEquivsButNonTrace() == 0) {
                  # Link the np-head and the object
                  $np_head->rawLabel($np_head->rawLabel() . "-" . $trace->traceId());
                }
              }
            }
          }
        }
      }
    }

    foreach my $child (@children){
      add_trace_for_dec_construction($child);
    }
  }
}

#
# DO NOT USE THIS OLD VERSION as possible as you can.
#
sub add_trace_for_relative_construction {
  my $node = shift;  

  if (not $node->isTerminal()) { 
    my @children = $node->children();
    if ($node->gramRel() eq "RELATIVE") {
      # Make sure there is a modified.
      my $npar = $node->parent();
      while (defined($npar) and $npar->label() ne "NP") {
        $npar = $npar->parent();
      }
      if ($npar->label() eq "NP" and $npar->children() >= 2) {
        my @np_children = $npar->children();
        my $np_head = $np_children[-1];

        # Try to link np_head with a trace under the IP intermediately inside the CP#relative
        my $ip;
        if (@children == 2 and $children[-1]->label() eq "CP") {
          # there is a function word like "DEC"
          my @tmp_children = $children[-1]->children();
          $ip = $tmp_children[0];
          print STDERR "oops! let me debug\n" if $ip->label() ne "IP";
        } elsif ($children[-1]->label() eq "IP") {
          # there is an "empty" function word 
          $ip = $children[-1];
        }
        if (defined($ip)) {
          my @ip_children = $ip->children();
          # Check subject first, if it is a trace without link to any "real" constitute,
          # link it with the modified of the DE-construction. 
          # If the subject is not a trace of a alreadly linked trace, check object(s) 
          # in a similar way. 
          # Yes! this solution doesn't consider the case that both sbj and objs are empty.
          # Let me improve it in future.
          my $subj = $ip_children[0];
          my @terms_of_subj = $subj->terminals();
  
          my $update = 0;
          if (@terms_of_subj == 1 and $terms_of_subj[0]->data() =~ /\*T\*/) {
            my $trace = $terms_of_subj[0];

            if ($trace->traceIdEquivsButNonTrace() == 0) {
              # Link the np-head and the subject
              $np_head->rawLabel($np_head->rawLabel() . "-" . $trace->traceId());
              $update = 1;
            }
          }
          if (not $update) {
            # find the vp
            my $vp_complementation = $ip_children[-1];
            my @objs = ();
            while (defined($vp_complementation) and (not $vp_complementation->isTerminal())) {
              my @tmp_children = $vp_complementation->children();
              if ($tmp_children[0]->isHead()) {
                for (my $k = 1; $k <= $#tmp_children; $k ++) {
                  if ($tmp_children[$k]->only_contains_one_trace()) {
                    my @tmp_terms = $tmp_children[$k]->terminals();
                    if ($tmp_terms[0]->data() =~ /\*T\*/) {
                      push @objs, $tmp_terms[0];
                    }
                  }
                }
                last;
              } else {
                $vp_complementation = $tmp_children[-1];
              }
            }
            if (@objs > 0) {
              my $trace = $objs[0]; # only update the first one and I suppose there is only one.
              if ($trace->traceIdEquivsButNonTrace() == 0) {
                # Link the np-head and the object
                $np_head->rawLabel($np_head->rawLabel() . "-" . $trace->traceId());
              }
            }
          }
        }
      }
    }

    foreach my $child (@children){
      add_trace_for_relative_construction($child);
    }
  }
}

#
# Deal with relative construction.
# According to the X-bar syntax, we give a more elegant solution based on category path.
#
sub add_trace_for_relative_construction_via_path {
  my $cp = shift;  

  if (not $cp->isTerminal()) { 
    my @children = $cp->children();
    if ($cp->gramRel() eq "RELATIVE") {
      # Make sure there is a modified.
      my $npar = $cp->parent();
      while (defined($npar) and $npar->label() ne "NP") {
        $npar = $npar->parent();
      }
      if ($npar->label() eq "NP" and $npar->children() >= 2) {
        my @np_children = $npar->children();
        my $np_head = $np_children[-1];

        foreach my $trace ($cp->terminals()) {
          if ($trace->data() =~ /\*T\*/) {
            if ($trace->traceIdEquivsButNonTrace() == 0) {
              # find path
              my $path_str = "";
              my $ancestor = $trace->parent();
              while (defined($ancestor) and $ancestor != $cp) {
                $path_str .= $ancestor->label();
                $ancestor = $ancestor->parent();
              }

              if ($path_str =~ /^NP(VP)*(IP)*(CP)*$/) {
                $np_head->rawLabel($np_head->rawLabel() . "-" . $trace->traceId());
              }
            }
          }
        }
      }
    }

    foreach my $child (@children){
      add_trace_for_relative_construction_via_path($child);
    }
  }
}

#
# Deal with LCP
# According to the X-bar syntax, we give a more elegant solution based on category path.
#
sub add_trace_for_lcp_via_path {
  my $lcp = shift;  

  if (not $lcp->isTerminal()) { 
    my @children = $lcp->children();
    foreach my $child (@children){
      add_trace_for_lcp_via_path($child);
    }

    if ($lcp->label() eq "LCP") {
      my @terms = $lcp->terminals();
      my $big_pro = $terms[0];
      return if ($big_pro->data() ne "*PRO*");

      # Find the subject
      my $npar = $lcp->parent();
      while (defined($npar) and $npar->label() ne "IP") {
        $npar = $npar->parent();
      }
      if (defined($npar) and $npar->label() eq "IP" and $npar->children() >= 2) {
        my @ip_children = $npar->children();
        my $subj = $ip_children[0];

        if ($subj->gramRel() eq "SBJ") {
          $big_pro->data($big_pro->data() . "-" . $min_valid_trace_id);
          $subj->rawLabel($subj->rawLabel() . "-" . $min_valid_trace_id);
          $min_valid_trace_id ++;
        }
      }
    }

  }
}

#
# According to the X-bar syntax, we give a more elegant solution based on category path.
#
sub trace_big_pro_via_path {
  my $node = shift;  

  foreach my $big_pro ($node->terminals()) {
    next if $big_pro->data() !~ /\*PRO\*/;
    next if defined($big_pro->traceId());

    # Case 1: IP->VP/PP->IP->PRO(subject)
    my $np = $big_pro->parent();
    if ($np->gramRel() eq "SBJ") {
      my $ip_low = $np->parent();
      while (defined($ip_low) and $ip_low->label() ne "IP") {
        $ip_low = $ip_low->parent();
      }
      if (defined($ip_low)) {
        my $vp = $ip_low->parent();
        while (defined($vp) and ($vp->label() ne "VP" and $vp->label() ne "PP")) {
          $vp = $vp->parent();
        }
        if (defined($vp)) {
          my $ip_high = $vp->parent();
          while (defined($ip_high) and $ip_high->label() ne "IP") {
            $ip_high = $ip_high->parent();
          }
          if (defined($ip_high)) {
            # is it a good one?
            my $path_str;
            my $tmp_par = $np;
            while (defined($tmp_par) and $tmp_par != $ip_high) {
              $path_str .= $tmp_par->label();
              $tmp_par = $tmp_par->parent();
            }
#            print STDERR "$path_str\n";
            if ($path_str =~ /^NP(IP)*(VP)*$/ or 
                $path_str =~ /^NP(IP)*(PP)*$/ or
                $path_str =~ /^NP(IP)*PP(VP)*$/  
            ) {
              # when the path is good, let us link
              foreach my $np_subj ($ip_high->children()) {
                if ($np_subj->gramRel() eq "SBJ") {
                  if ($np_subj->only_contains_one_trace()) { 
                    # This case is rather tricky, namely link PRO to PRO
                    # We take it account a language-specific property: 
                    # the c-commonder PRO is preceding the c-commonded.
                    my @tmp_terms = $np_subj->terminals();
                    my $np_subj_trace = $tmp_terms[0];
                    if (defined($np_subj_trace->traceId())) {
                      $big_pro->data($big_pro->data() . "-" . $np_subj_trace->traceId());
                      last;
                    }
                  }
                  $big_pro->data($big_pro->data() . "-" . $min_valid_trace_id);
                  $np_subj->rawLabel($np_subj->rawLabel() . "-" . $min_valid_trace_id);
                  $min_valid_trace_id ++;
                  last;
                }
              }
            }

          }
        }
      }
      next;
    }
  }
}

#
#
sub deal_with_short_bei {
  my $node = shift;  

  foreach my $sb ($node->terminals()) {
    next if $sb->label() ne "SB";

    my $ip = $sb->parent();
    while (defined($ip) and $ip->label() ne "IP") {
      $ip = $ip->parent();
    }
    if (defined($ip)) {
      foreach my $np_subj ($ip->children()) {
        $np_subj->gramRel("SBJbei") if ($np_subj->gramRel() eq "SBJ");
      }
    }
  }
}

#
# Deal with BA-construction
#
sub deal_with_babei_structure {
  my $subj = shift;  

  if (not $subj->isTerminal()) { 
    my @children = $subj->children();
    foreach my $child (@children){
      deal_with_babei_structure($child);
    }

    if ($subj->gramRel() eq "SBJ") {
      my $npar = $subj->parent();
      if (defined($npar) and $npar->label() eq "IP") {
        $npar = $npar->parent();
        if (defined($npar) and $npar->label() eq "VP") {
          my @tmp_children = $npar->children();
          $subj->gramRel("SBJ*ba") if $tmp_children[0]->label() eq "BA";
          $subj->gramRel("SBJ*bei") if $tmp_children[0]->label() eq "LB";
        }
      }
    }
  }
}

#
sub deal_with_serial_verbal_construction_easy {
  my $vp = shift;  

  if (not $vp->isTerminal()) { 
    my @children = $vp->children();
    foreach my $child (@children){
      deal_with_serial_verbal_construction_easy($child);
    }
    if ($vp->label() eq "VP" and
        $#children == 1 and 
        $children[0]->label() eq "VP" and
        $children[1]->label() eq "VP") {
      $children[0]->iAmSerialVP();
      $children[1]->iAmSerialVP();
    }
  }
}

#
#
sub heighten {
  my $node = shift;  

  if (not $node->isTerminal()) { 
    foreach my $child ($node->children) { # BE CAREFUL HERE: the children have be modified
      heighten($child);
    }
    my $relation_headaddr = get_head_and_relation($node);
    my ($relation, $headaddr) = split(/\|/, $relation_headaddr);
    my $synt_cat = $node->label();
    my @children = $node->children();

    if (@children > 2) {
      if ($synt_cat =~ /^[NV]P$/ and $relation eq "adj" and 
          $children[-1]->isHead() and $children[-1]->label() eq $synt_cat) {
        # this structure consists of a number of modifiers and finally the head phrase.
        my $last_np_node = $children[-1];
        for (my $k = $#children-1; $k >= 1; $k --) {
          my $child = $children[$k];
          my $new_np_node = TBNode->new($synt_cat);
          $new_np_node->termno($child->termno());
          $new_np_node->iAmHeadPhrase(); 

          $new_np_node->addChild($child);
          $new_np_node->addChild($last_np_node);

          $last_np_node = $new_np_node;
        }
        $last_np_node->iAmHeadPhrase();

        $node->clean_children();
        $node->addChild($children[0]);
        $node->addChild($last_np_node);
      }
    }
    if ($synt_cat eq "IP" and $relation ne "coord") {
      my $find_comp = 0;
      my $find_adj = 0;
      my $head_posi = -1;
      my $subj_posi = -1;
      my $idx = 0;
      foreach my $chd (@children) {
        $find_comp = 1 and $subj_posi = $idx if $chd->gramRel() eq "SBJ";
        $find_adj = 1 if $chd->gramRel() eq "ADV";
        $head_posi = $idx if $chd->isHead();
        $idx ++;
      }
      if ($find_comp and $find_adj and $head_posi != -1 and $subj_posi != -1) {
        my $new_ip_node = TBNode->new("IP");
        $new_ip_node->termno($children[$subj_posi]->termno());
        $new_ip_node->iAmHeadPhrase(); 

        my @new_children = ();
        for (my $k = 0; $k < $subj_posi; $k ++) {
          push @new_children, $children[$k];
        }
        for (my $k = $subj_posi; $k <= $head_posi; $k ++) {
          $new_ip_node->addChild($children[$k]);
        }
        push @new_children, $new_ip_node;
        for (my $k = $head_posi+1; $k < @children; $k ++) {
          push @new_children, $children[$k];
        }
        $node->clean_children();
        foreach my $new_chd (@new_children) {
          $node->addChild($new_chd);
        }
      }
    }
    if (@children >= 4 and $children[0]->label() eq "PU" and $children[-1]->label() eq "PU") {
      my $new_node = TBNode->new($synt_cat);
      $new_node->termno($children[1]->termno());
      $new_node->iAmHeadPhrase(); 

      for (my $k = 1; $k < $#children; $k ++) {
        $new_node->addChild($children[$k]);
      }

      my @new_children = ();
      push @new_children, $children[0];
      push @new_children, $new_node;
      push @new_children, $children[-1];

      $node->clean_children();
      foreach my $new_chd (@new_children) {
        $node->addChild($new_chd);
      }
    }
  }
}

sub trace_a_gpsg_tree {
  my $top_n = shift;
  my $predref = sub { 
    my $node = shift;
    if ($node->label() =~ /\//) {
      my $insert = 1;
      foreach my $c ($node->children()) {
        if ($c->label() =~ /\//) {
          $insert = 0;
          last;
        }
      }
      return $insert;
    }
    return 0;
  };
  my $findallref = sub { return 1; };
  my @all_nodes = $top_n->find($findallref);

  foreach my $n ($top_n->find($predref)) {
    my $npar = $n->parent();
#   print STDERR $n->label(), "\n";
    while (defined($npar)) {
#     print STDERR "\t", $npar->label(), "\n";
      last if ($npar->label() !~ /\//);
      $npar = $npar->parent();
    }
    if (defined($npar)) {
      foreach my $antic ($npar->children()) {
        next if ($antic->start() <= $n->start() and $antic->end() >= $n->end());

        my $trace = TBNode->new("-NONE-");
        $n->label() =~ /\/(.*)/;
        $trace->gramRel($1);

        $trace->data("*-$min_valid_trace_id");
        $antic->rawLabel($antic->rawLabel() . "-". $min_valid_trace_id);
        $min_valid_trace_id ++;

        $n->addChild($trace);

#       my $end = $n->end();
#       $trace->termno($end+1);
#       foreach my $nt (@all_nodes) {
#         if ($nt->termno() > $end) {
#           $nt->termno($nt->termno()+1);
#         }
#       }

        last;
      }
    }
  }
}

#
# recursively update the head words informations of every nodes
#
sub update_headword_index {
  my $node = shift;  

  if ($node->isTerminal()) { 
    $node->addOneHeadWord($node->start());
  } else {
    foreach my $child ($node->children()) {
      update_headword_index($child);
      if ($child->isHead() or $child->isConjunct()) {
        foreach my $hi ($child->headIdx()) {
          $node->addOneHeadWord($hi);
        }
      } 
    }
  }
}

#
# get all bi-lexical dependencies.
#
sub collect_tuples {
  my $node = shift;  
  my $tuples_lref = shift;  
  my $terms_lref = shift;  

  #my @tuples = @{$tuples_lref};

  my @terms = @{$terms_lref};

  if (not $node->isTerminal()){ 
    my $relation_headaddr = get_head_and_relation($node);
    my ($relation, $headaddr) = split(/\|/, $relation_headaddr);

    my @children = $node->children();
    my $hi = -1;
    for ($hi = 0; $hi <= $#children; $hi ++) {
      last if $children[$hi]->isHead();
    }
    if ($hi > $#children) {
      # there is no head here => do nothing.
    } else {
      my $hchild = $children[$hi]; 
      for (my $nhi = 0; $nhi <= $#children; $nhi ++) {
        if ($nhi != $hi) {
          my $dchild = $children[$nhi];
          if (! $dchild->isConjunction() and ! $dchild->isConjunct() and ! $dchild->isHead()) {
            foreach my $dword ($dchild->headIdx()) {
              next if $dchild->label() eq "PU";
              next if $dchild->gramRel() eq "UNSPEC";

              foreach my $hword ($hchild->headIdx()) {
                next if $hchild->label() eq "PU";
                my $tuple = $hword ."\t". $dword ."\t". $dchild->gramRel(); 
                push @tuples, $tuple;

                my @ldd_nodes = ();
                if ($terms[$dword]->isTrace()) {
                  foreach my $dw ($terms[$dword]->traceIdEquivs()) {
                    if ($dw->start() != $terms[$dword]->start()) {
                      push @ldd_nodes, $dw;
                    }
                  }
                }

                foreach my $ldd_node (@ldd_nodes) {
                  foreach my $ldd_word ($ldd_node->headIdx()) {
                    my $tuple = $hword ."\t". $ldd_word ."\t". $dchild->gramRel(). "*ldd*"; #. $dchild->data(); 
                    push @tuples, $tuple;
                  }
                }
              }
            }
          }
        }
      }
    }

    foreach my $child (@children) {
      collect_tuples($child, \@tuples, \@terms);
    }
  }
}

#
# determine if the structure is one of adj(unction), comp(lementation),
# aux(iliary), coord(ination), pred(ication), or flat.
# return values: "adj", "comp", "aux", "coord", "pred", "flat" 
# as well as the head of this constituent
#
sub get_head_and_relation{ 
    my ($node) = @_;
    my $nodeLabel = $node->label();
    my @children = $node->children();
    my $nonterm_head = ""; #non-terminal head, head-final 
    my $term_head = ""; #terminal head
    my $CCs = 0;
    my $PUs = 0;
    my $terminals = 0; #number of terminal chidren in constituent that are NOT punctuation marks
    my $type = "flat"; #default relation is flat
    my $allterminal = 1; #a boolean to determine if all the children are terminals
    my %allnonterms;
    my %allterminals;
    my $maxNonterm = 0; #maximum number of nonterms that have the same label
    my $uniqlabels = 0; #number of unique labels in the non-terminal children
    my $uniqtermlabels = 0;

    foreach my $child (@children){
	if($child->height() == 0){
	    if($child->label() eq "CC"){
		$CCs++;
	    }elsif($child->label() eq "ETC"){
		$CCs++;
	    }elsif($child->label() eq "PU"){
		$PUs++;
	    }elsif($child->label() eq "AS"){
		#don't update the terminal head
	    }else{
		$terminals++;
		$term_head = $child;
		$allterminals{$child->label()}++;
	    }
	}elsif(is_compound($child)){
	    #treat compounds as terminals
	    $term_head = $child;
	}else{
	    if($child->label() ne "PRN"){
		$allterminal = 0;
		$allnonterms{$child->label()}++;
		$nonterm_head = $child;
	    }
	}
    }

    foreach my $k (keys %allterminals){
	$uniqtermlabels++;
    }


    foreach my $k (keys %allnonterms){
	$uniqlabels++;
	if($allnonterms{$k} > $maxNonterm){
	    $maxNonterm = $allnonterms{$k};
	}
    }

    #print "test1\n";
    #the last non-punctuation terminal is CC
    if($CCs || #if there is a CC
       $nodeLabel eq "VCD"||
       ($terminals > 1 && $allterminal && $PUs > 0 && $uniqtermlabels == 1)||
       (!$term_head && $maxNonterm >1 && $PUs > 0 && $uniqlabels == 1) || #or no terminals other than PU
                                            #there are two or more nonterms with the same label  
       (!$term_head && $nodeLabel eq "UCP") ){
	if($nonterm_head){ 
	    my $headaddr = create_id_easy($nonterm_head);
	    return "coord|$headaddr";
	}elsif($term_head){
	    my $headaddr = create_id_easy($term_head);
	    return "coord|$headaddr";
	}else{
	    return "coord|unspecified"; #do not specify a head, e.g., conjunctions
	}
    }

    #print "test2\n";
    #the structure is all flat
    if($allterminal){
	if($term_head){  
	    my $headaddr = create_id_easy($term_head);
	    return "flat|$headaddr";
	}else{
	    #the head is empty because there is only PU or CC
	    #use the first child
	    my $headaddr = create_id_easy($children[0]);
	    return "flat|$headaddr";
	}
    }

    #print "test3\n";
    #if the last non-terminal structure is VC or VV and
    #its right sibling is a VP, this is an auxiliary
    #head is the VP
    if($term_head && $term_head->nextSibling() && $term_head->nextSibling()->label() eq "VP"){
	my $headaddr = create_id_easy($term_head->nextSibling()); 
	return "aux|$headaddr";
    }
    #print "test4\n";
    #if there is a terminal, it's comp
    if($term_head){
	my $headaddr = create_id_easy($term_head);
	return "comp|$headaddr";
    }

    #if this is an IP node and the headnode is a VP, then this is a predication
    if (!$term_head && $nodeLabel =~ /^IP/) {
      if ($nonterm_head->label() eq "VP"){
 	my $headaddr = create_id_easy($nonterm_head);
	return "pred|$headaddr";
      }
      # add by wsun according to the CTB manual (page 6).
      my @ftags = $nonterm_head->ftags();
      foreach my $ftag (@ftags) {
        if ($ftag eq "PRD") {
 	  my $headaddr = create_id_easy($nonterm_head);
	  return "pred|$headaddr";
        }
      }
    }

    #everything else is treated as adjunction and the headnode is the last non-punctuation node
    my $headaddr = create_id_easy($nonterm_head);
    return "adj|$headaddr";
}

sub update_gramrel_coordination_easy {
  my $children_lref = shift;
  foreach my $child (@$children_lref) {
    if (is_conj($child)) {
      $child->iAmConjunction();
    } else { 
      $child->iAmConjunct();
    }
  }
}

sub update_gramrel_coordination_hard {
  my $children_lref = shift;
  foreach my $child (@$children_lref) {
    if (is_conj($child)) {
      $child->iAmConjunction();
    } elsif ($child->label() eq "PU") { 
      # pass
    } else { 
      $child->iAmConjunct();
    }
  }
}

sub update_gramrel_head_nonhead_easy {
  my $children_lref = shift;
  my $headaddr = shift;
  my $default_tag = shift;
  foreach my $child (@$children_lref) {
    if ($headaddr eq $child->termno().":".$child->height()) { 
      $child->iAmHeadPhrase();
    } elsif ($child->label() eq "PU") { 
      # pass
    } else { 
      my $ftag = get_function_tag($child);
      $ftag = $default_tag if $ftag eq "NONE"; # default tag
      $child->gramRel($ftag);
    }
  }
}

sub update_gramrel_head_initial_easy {
  my $children_lref = shift;
  my $default_tag = shift;
  my @children = @$children_lref;

  $children[0]->iAmHeadPhrase();
  for (my $k = 1; $k <= $#children; $k ++) {
    my $child = $children[$k];
    my $ftag = get_function_tag($child);
    $ftag = $default_tag if $ftag eq "NONE"; # default tag
    $child->gramRel($ftag);
  }
}

sub update_gramrel_head_final_easy {
  my $children_lref = shift;
  my $default_tag = shift;
  my @children = @{$children_lref};

  for (my $k = 0; $k < $#children; $k ++) {
    my $child = $children[$k];
    #  print STDERR $child->rawdata();
    my $ftag = get_function_tag($child);
    $ftag = $default_tag if $ftag eq "NONE"; # default tag
    $child->gramRel($ftag);
  }
  #print STDERR scalar(@children), "\n";
  $children[-1]->iAmHeadPhrase();
}

sub update_gramrel_verb_complement_easy {
  my $children_lref = shift;
  my $headaddr = shift;
  my $default_tag = shift;
  foreach my $child (@$children_lref) {
    if ($headaddr eq $child->termno().":".$child->height()) { 
      $child->iAmHeadPhrase();
    } elsif ($child->label() eq "AS") { 
      $child->gramRel("PRT");
    } elsif ($child->label() eq "PU") { 
      # pass
    } else { 
      my $ftag = get_function_tag($child);
      $ftag = $default_tag if $ftag eq "NONE"; # default tag
      $child->gramRel($ftag);
    }
  }
}

sub update_gramrel_verb_coordination_easy {
  my $children_lref = shift;
  foreach my $child (@$children_lref) {
    if (is_conj($child)) {
      $child->iAmConjunction();
    } elsif ($child->label() eq "PU") { 
      # pass
    } elsif (is_verb($child)) { 
      $child->iAmConjunct();
    } else { 
      $child->iAmConjunct();
    }
  }
}

sub is_conj{
    my ($node) = @_;
    if($node->height() == 0 && 
       ($node->label() eq "CC" || 
	$node->label() eq "ETC" || 
	($node->label() eq "PU" &&
	 $node->data() ne "。"))){
	return 1;
    }
    return 0;
}

sub is_compound{
    my ($node) = @_;
    if($node->label() eq "VCD" || 
       $node->label() eq "VRD" || 
       $node->label() eq "VNV" || 
       $node->label() eq "VSB" || 
       $node->label() eq "VPT" || 
       $node->label() eq "VCP"){
	return 1;
    }
    return 0;
}

sub isFlat{
    my (@nodes) = @_;
    my $isflat = 1;
    foreach my $node (@nodes){
	if($node->height() != 0){
	    $isflat = 0;
	}
    }
    return $isflat;
}

sub is_verb{
    my ($node) = @_;
    if($node->label() =~ /^V/){
	return 1;
    }
    return 0;
}

sub printnode{
    my ($cnode, $outfh) = @_;
    print $outfh $cnode -> rawdata();
}


sub printConj {
    my ($outfh) = @_;
    print $outfh "(CONJOINED T)";
}

sub is_the_same_span {
    my ($node1, $node2) =@_;
    return $node1->termno() == $node2->termno() && $node1->height() == $node2-> height();
}

sub get_function_tag {
  # filtering out the irrelevant tags
  my ($node) = @_;
  # put the nodes in a hash
  my @ftags = $node->ftags();
  foreach my $ftag (@ftags){
    if    ($ftag eq "ADV") { return "ADV"; }
    elsif ($ftag eq "APP") { return "APP"; }
    elsif ($ftag eq "BNF") { return "BNF"; }
    elsif ($ftag eq "CND") { return "CND"; }
    elsif ($ftag eq "DIR") { return "DIR"; }
    elsif ($ftag eq "EXT") { return "EXT"; }
    elsif ($ftag eq "FOC") { return "FOC"; }
    elsif ($ftag eq "IO")  { return "IO"; }
    elsif ($ftag eq "LGS") { return "LGS"; }
    elsif ($ftag eq "LOC") { return "LOC"; }
    elsif ($ftag eq "MNR") { return "MNR"; }
    elsif ($ftag eq "PRD") { return "PRD"; }
    elsif ($ftag eq "PRP") { return "PRP"; }
    elsif ($ftag eq "SBJ") { return "SBJ"; }
    elsif ($ftag eq "TMP") { return "TMP"; }
    elsif ($ftag eq "TPC") { return "TPC"; }
    elsif ($ftag eq "VOC") { return "VOC"; }
    elsif ($ftag eq "OBJ") { return "OBJ"; }
  }
  return "NONE";
}

sub printSemFeature{
    my ($outfh, $node) = @_;
    my @ftags = $node->ftags();
    my $ftag = $ftags[0];
    if(defined($ftag)){
	if($ftag eq "MNR"){
	    print $outfh "(SEM-FEATURE MNR)";
	} elsif ($ftag eq "CND"){
	    print $outfh "(SEM-FEATURE CONDITION)";
	} elsif ($ftag eq "DIR"){
	    print $outfh "(SEM-FEATURE DIR)";
	} elsif ($ftag eq "PRP"){
	    print $outfh "(SEM-FEATURE PRP)";
	} elsif ($ftag eq "EXT"){
	    print $outfh "(SEM-FEATURE EXT)";
	} elsif ($ftag eq "Q"){
	    print $outfh "(MOOD INTERROGATIVE)";
	} elsif ($ftag eq "FOC"){
	    print $outfh "(FOCUS T)";
	} elsif ($ftag eq "HLN"){
	    print $outfh "(GENRE-TYPE HLN)";
	} elsif ($ftag eq "TMP"){
	    print $outfh "(SEM-FEATURE TMP)";
	} elsif ($ftag eq "IMP"){
	    print $outfh "(MOOD IMPERATIVE)";
	} elsif ($ftag eq "TTL"){
	    print $outfh "(GENRE-TYPE TTL)";
	} elsif ($ftag eq "LOC"){
	    print $outfh "(SEMFEATURE LOC)";
	}
    }

}

sub create_id_easy {
  my ($node) = @_;
  if (defined($node)) {
    my $termno = $node->termno(); 
    my $height = $node->height();
    my $id = $termno . ":" . $height; 
    return $id; 
  } else { 
    print STEDRR "Error!\n"; 
  }
  return "NIL"; 
}

#sub to_conll08 {
#  my $tuples_lref = shift;
#
#  my @tuples = @$tuples_lref;
#  foreach my $tuple (@tuples) {
#    my ($hi, $di, $rel) = (split "\t", $tuple);
#    printf $outfh  "    (%i, %i, %s)\n", $hi, $di, $rel; 
#    prop = new SRL::prop;
#  }
#}

sub conlltrain {
  my @flist = ("0081", "0083", "0085", "0087", "0089", "0091", "0093", "0095", "0097", "0099", "0101", "0103", "0105", "0107", "0109", "0111", "0113", "0115", "0117", "0119", "0121", "0123", "0125", "0127", "0129", "0131", "0133", "0135", "0137", "0139", "0141", "0143", "0145", "0147", "0149", "0151", "0153", "0155", "0157", "0159", "0161", "0163", "0165", "0167", "0169", "0171", "0173", "0175", "0177", "0179", "0181", "0183", "0185", "0187", "0189", "0191", "0193", "0195", "0197", "0199", "0201", "0203", "0205", "0207", "0209", "0211", "0213", "0215", "0217", "0219", "0221", "0223", "0225", "0227", "0229", "0231", "0233", "0235", "0237", "0239", "0241", "0243", "0245", "0247", "0249", "0251", "0253", "0255", "0257", "0259", "0261", "0263", "0265", "0267", "0269", "0271", "0273", "0275", "0277", "0279", "0281", "0283", "0285", "0287", "0289", "0291", "0293", "0295", "0297", "0299", "0301", "0303", "0305", "0307", "0309", "0311", "0313", "0315", "0317", "0319", "0321", "0323", "0325", "0401", "0403", "0405", "0407", "0409", "0411", "0413", "0415", "0417", "0419", "0421", "0423", "0425", "0427", "0429", "0431", "0435", "0437", "0439", "0441", "0443", "0445", "0447", "0449", "0451", "0453", "0501", "0503", "0505", "0507", "0509", "0511", "0513", "0515", "0517", "0519", "0521", "0523", "0525", "0527", "0529", "0531", "0533", "0535", "0537", "0539", "0541", "0543", "0545", "0547", "0549", "0551", "0553", "0591", "0595", "0601", "0603", "0605", "0607", "0609", "0611", "0613", "0615", "0617", "0619", "0621", "0623", "0625", "0627", "0631", "0633", "0635", "0637", "0639", "0641", "0643", "0645", "0647", "0649", "0651", "0653", "0655", "0657", "0659", "0661", "0663", "0665", "0667", "0669", "0671", "0673", "0675", "0677", "0679", "0681", "0683", "0685", "0687", "0689", "0691", "0693", "0695", "0697", "0699", "0701", "0703", "0705", "0707", "0709", "0711", "0713", "0715", "0717", "0719", "0721", "0723", "0725", "0727", "0729", "0731", "0733", "0735", "0737", "0739", "0741", "0743", "0745", "0747", "0749", "0751", "0753", "0755", "0757", "0759", "0761", "0763", "0765", "0767", "0769", "0771", "0773", "0775", "0777", "0779", "0781", "0783", "0785", "0787", "0789", "0791", "0793", "0797", "0799", "0801", "0803", "0805", "0807", "0809", "0811", "0813", "0817", "0819", "0821", "0823", "0825", "0827", "0829", "0831", "0833", "0835", "0837", "0839", "0841", "0843", "0845", "0847", "0849", "0851", "0853", "0855", "0857", "0859", "0861", "0863", "0865", "0867", "0869", "0871", "0873", "0875", "0877", "0879", "0881", "0883", "0885", "1001", "1003", "1005", "1007", "1009", "1011", "1013", "1017", "1019", "1021", "1023", "1025", "1027", "1029", "1031", "1033", "1035", "1037", "1039", "1041", "1043", "1045", "1047", "1049", "1051", "1053", "1055", "1057", "1059", "1063", "1065", "1067", "1069", "1071", "1073", "1075", "1077", "1103", "1105", "1107", "1109", "1111", "1115", "1131", "1133", "1137", "1145", "1147", "1149", "1151", "2001", "2003", "2005", "2007", "2009", "2011", "2013", "2015", "2017", "2019", "2021", "2023", "2025", "2027", "2029", "2031", "2033", "2035", "2037", "2039", "2041", "2043", "2045", "2047", "2049", "2051", "2053", "2055", "2057", "2059", "2061", "2063", "2065", "2067", "2069", "2071", "2073", "2075", "2077", "2079", "2081", "2083", "2085", "2087", "2089", "2091", "2093", "2095", "2097", "2099", "2101", "2103", "2105", "2107", "2109", "2111", "2113", "2115", "2117", "2119", "2121", "2123", "2125", "2127", "2129", "2131", "2133", "2135", "2137", "2139", "2161", "2163", "2181", "2183", "2185", "2187", "2189", "2191", "2193", "2195", "2197", "2199", "2201", "2203", "2205", "2207", "2209", "2211", "2213", "2215", "2217", "2219", "2221", "2223", "2225", "2227", "2229", "2231", "2233", "2235", "2237", "2239", "2241", "2243", "2245", "2247", "2249", "2251", "2253", "2255", "2257", "2259", "2261", "2263", "2265", "2267", "2269", "2271", "2273", "2275", "2277", "2279", "2311", "2313", "2315", "2317", "2319", "2321", "2323", "2325", "2327", "2329", "2331", "2333", "2335", "2337", "2339", "2341", "2343", "2345", "2347", "2349", "2351", "2353", "2355", "2357", "2359", "2361", "2363", "2365", "2367", "2369", "2371", "2373", "2375", "2377", "2379", "2381", "2383", "2385", "2387", "2389", "2391", "2393", "2395", "2397", "2399", "2401", "2403", "2405", "2407", "2409", "2411", "2413", "2415", "2417", "2419", "2421", "2423", "2425", "2427", "2429", "2431", "2433", "2435", "2437", "2439", "2441", "2443", "2445", "2447", "2449", "2451", "2453", "2455", "2457", "2459", "2461", "2463", "2465", "2467", "2469", "2471", "2473", "2475", "2477", "2479", "2481", "2483", "2485", "2487", "2489", "2491", "2493", "2495", "2497", "2499", "2501", "2503", "2505", "2507", "2509", "2511", "2513", "2515", "2517", "2519", "2521", "2523", "2525", "2527", "2529", "2531", "2533", "2535", "2537", "2539", "2541", "2543", "2545", "2547", "2549", "2603", "2605", "2607", "2609", "2611", "2613", "2615", "2617", "2619", "2621", "2623", "2625", "2627", "2629", "2631", "2633", "2635", "2637", "2639", "2641", "2643", "2645", "2647", "2649", "2651", "2653", "2655", "2657", "2659", "2661", "2663", "2665", "2667", "2669", "2671", "2673", "2675", "2677", "2679", "2681", "2683", "2685", "2687", "2689", "2691", "2693", "2695", "2697", "2699", "2701", "2703", "2705", "2707", "2709", "2711", "2713", "2715", "2717", "2719", "2721", "2723", "2725", "2727", "2729", "2731", "2733", "2735", "2737", "2739", "2741", "2743", "2745", "2747", "2749", "2751", "2753", "2755", "2757", "2759", "2761", "2763", "2765", "2767", "2769", "2771", "2773", "2821", "2823", "2825", "2827", "2829", "2831", "2833", "2835", "2837", "2839", "2841", "2843", "2845", "2847", "2849", "2851", "2853", "2855", "2857", "2859", "2861", "2863", "2865", "2867", "2869", "2871", "2873", "2875", "2877", "2879", "2881", "2883", "2885", "2887", "2889", "2891", "2893", "2895", "2897", "2899", "2901", "2903", "2905", "2907", "2909", "2911", "2913", "2915", "2917", "2919", "2921", "2923", "2925", "2927", "2929", "2931", "2933", "2935", "2937", "2939", "2941", "2943", "2945", "2947", "2949", "2951", "2953", "2955", "2957", "2959", "2961", "2963", "2965", "2967", "2969", "2971", "2973", "2975", "2977", "2979", "2981", "2983", "2985", "2987", "2989", "2991", "2993", "2995", "2997", "2999", "3001", "3003", "3005", "3007", "3009", "3011", "3013", "3015", "3017", "3019", "3021", "3023", "3025", "3027", "3029", "3031", "3033", "3035", "3037", "3039", "3041", "3043", "3045", "3047", "3049", "3051", "3053", "3055", "3057", "3059", "3061", "3063", "3065", "3067", "3069", "3071", "3073", "3075", "3077", "3079", "0082", "0084", "0086", "0088", "0090", "0092", "0094", "0096", "0098", "0100", "0102", "0104", "0106", "0108", "0110", "0112", "0114", "0116", "0118", "0120", "0122", "0124", "0126", "0128", "0130", "0132", "0134", "0136", "0138", "0140", "0142", "0144", "0146", "0148", "0150", "0152", "0154", "0156", "0158", "0160", "0162", "0164", "0166", "0168", "0170", "0172", "0174", "0176", "0178", "0180", "0182", "0184", "0186", "0188", "0190", "0192", "0194", "0196", "0198", "0200", "0202", "0204", "0206", "0208", "0210", "0212", "0214", "0216", "0218", "0220", "0222", "0224", "0226", "0228", "0230", "0232", "0234", "0236", "0240", "0242", "0244", "0246", "0248", "0250", "0252", "0254", "0256", "0258", "0260", "0262", "0264", "0266", "0268", "0270", "0272", "0274", "0276", "0278", "0280", "0282", "0284", "0286", "0288", "0290", "0292", "0294", "0296", "0298", "0300", "0302", "0304", "0306", "0308", "0312", "0314", "0316", "0318", "0320", "0322", "0324", "0400", "0402", "0404", "0406", "0408", "0410", "0412", "0414", "0416", "0418", "0422", "0424", "0426", "0428", "0430", "0432", "0434", "0436", "0438", "0440", "0442", "0444", "0446", "0448", "0450", "0452", "0454", "0500", "0502", "0504", "0506", "0508", "0510", "0512", "0514", "0516", "0518", "0520", "0522", "0524", "0526", "0528", "0530", "0532", "0534", "0536", "0538", "0540", "0542", "0544", "0546", "0550", "0552", "0554", "0590", "0592", "0594", "0596", "0600", "0602", "0604", "0606", "0608", "0610", "0616", "0618", "0620", "0622", "0624", "0626", "0628", "0630", "0632", "0634", "0636", "0638", "0640", "0642", "0644", "0646", "0648", "0650", "0652", "0654", "0656", "0658", "0660", "0662", "0664", "0666", "0668", "0670", "0672", "0674", "0676", "0678", "0680", "0682", "0684", "0686", "0688", "0690", "0692", "0694", "0696", "0698", "0700", "0702", "0704", "0706", "0708", "0710", "0712", "0714", "0716", "0718", "0720", "0722", "0724", "0726", "0728", "0730", "0732", "0734", "0736", "0738", "0740", "0742", "0744", "0746", "0750", "0752", "0754", "0756", "0758", "0760", "0762", "0764", "0766", "0768", "0770", "0772", "0774", "0776", "0778", "0780", "0782", "0784", "0786", "0788", "0790", "0792", "0794", "0796", "0798", "0800", "0802", "0804", "0806", "0808", "0810", "0812", "0814", "0816", "0818", "0820", "0822", "0824", "0826", "0828", "0830", "0832", "0834", "0836", "0838", "0840", "0842", "0844", "0846", "0848", "0850", "0852", "0854", "0856", "0860", "0862", "0864", "0866", "0868", "0870", "0872", "0874", "0876", "0878", "0880", "0882", "0884", "0900", "1002", "1004", "1006", "1008", "1010", "1012", "1014", "1016", "1022", "1024", "1028", "1030", "1032", "1034", "1038", "1040", "1042", "1046", "1048", "1050", "1052", "1054", "1056", "1058", "1062", "1064", "1066", "1068", "1070", "1074", "1076", "1078", "1100", "1102", "1104", "1106", "1108", "1110", "1112", "1114", "1130", "1134", "1136", "1138", "1140", "1146", "1150", "2000", "2002", "2004", "2006", "2008", "2010", "2012", "2014", "2016", "2018", "2020", "2022", "2024", "2026", "2028", "2030", "2032", "2034", "2036", "2038", "2040", "2042", "2044", "2046", "2048", "2050", "2052", "2054", "2056", "2058", "2060", "2062", "2064", "2066", "2068", "2070", "2072", "2074", "2076", "2078", "2080", "2082", "2084", "2086", "2088", "2090", "2092", "2094", "2096", "2098", "2100", "2102", "2104", "2106", "2108", "2110", "2112", "2114", "2116", "2118", "2120", "2122", "2124", "2126", "2128", "2130", "2132", "2134", "2136", "2138", "2160", "2162", "2164", "2182", "2184", "2186", "2188", "2190", "2192", "2194", "2196", "2198", "2200", "2202", "2204", "2206", "2208", "2210", "2212", "2214", "2216", "2218", "2220", "2222", "2224", "2226", "2228", "2230", "2232", "2234", "2236", "2238", "2240", "2242", "2244", "2246", "2248", "2250", "2252", "2254", "2256", "2258", "2260", "2262", "2264", "2266", "2268", "2270", "2272", "2274", "2276", "2278", "2312", "2314", "2316", "2318", "2320", "2322", "2324", "2326", "2328", "2330", "2332", "2334", "2336", "2338", "2340", "2342", "2344", "2346", "2348", "2350", "2352", "2354", "2356", "2358", "2360", "2362", "2364", "2366", "2368", "2370", "2372", "2374", "2376", "2378", "2380", "2382", "2384", "2386", "2388", "2390", "2392", "2394", "2396", "2398", "2400", "2402", "2404", "2406", "2408", "2410", "2412", "2414", "2416", "2418", "2420", "2422", "2424", "2426", "2428", "2430", "2432", "2434", "2436", "2438", "2442", "2444", "2446", "2448", "2450", "2452", "2454", "2456", "2458", "2460", "2462", "2464", "2466", "2468", "2470", "2472", "2474", "2476", "2478", "2480", "2482", "2484", "2486", "2488", "2490", "2492", "2494", "2496", "2498", "2500", "2502", "2504", "2506", "2508", "2510", "2512", "2514", "2516", "2518", "2520", "2522", "2524", "2526", "2528", "2530", "2532", "2534", "2536", "2538", "2540", "2542", "2544", "2546", "2548", "2604", "2606", "2608", "2610", "2612", "2614", "2616", "2618", "2620", "2622", "2624", "2626", "2628", "2630", "2632", "2634", "2636", "2638", "2640", "2642", "2644", "2646", "2648", "2650", "2652", "2654", "2656", "2658", "2660", "2662", "2664", "2666", "2668", "2670", "2672", "2674", "2676", "2678", "2680", "2682", "2684", "2686", "2688", "2690", "2692", "2694", "2696", "2698", "2700", "2702", "2704", "2706", "2708", "2710", "2712", "2714", "2716", "2718", "2720", "2722", "2724", "2726", "2728", "2730", "2732", "2734", "2736", "2738", "2740", "2742", "2744", "2746", "2748", "2750", "2752", "2754", "2756", "2758", "2760", "2762", "2764", "2766", "2768", "2770", "2772", "2774", "2820", "2822", "2824", "2826", "2828", "2830", "2832", "2834", "2836", "2838", "2840", "2842", "2844", "2846", "2848", "2850", "2852", "2854", "2856", "2858", "2860", "2862", "2864", "2866", "2868", "2870", "2872", "2874", "2876", "2878", "2880", "2882", "2884", "2886", "2888", "2890", "2892", "2894", "2896", "2898", "2900", "2902", "2904", "2906", "2908", "2910", "2912", "2914", "2916", "2918", "2920", "2922", "2924", "2926", "2928", "2930", "2934", "2936", "2938", "2940", "2942", "2944", "2946", "2948", "2950", "2952", "2954", "2956", "2958", "2960", "2962", "2964", "2966", "2968", "2970", "2972", "2974", "2976", "2978", "2980", "2982", "2984", "2986", "2988", "2990", "2992", "2994", "2996", "2998", "3000", "3002", "3004", "3006", "3008", "3010", "3012", "3014", "3016", "3018", "3020", "3022", "3024", "3026", "3028", "3030", "3032", "3034", "3036", "3038", "3040", "3042", "3044", "3046", "3048", "3050", "3052", "3054", "3058", "3060", "3062", "3064", "3066", "3068", "3070", "3072", "3074", "3076", "3078");  
}
sub conlldev {
  my @flist = ("0041", "0042", "0043", "0044", "0045", "0046", "0047", "0048", "0049", "0050", "0051", "0052", "0054", "0055", "0056", "0057", "0058", "0059", "0060", "0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069", "0070", "0071", "0072", "0073", "0074", "0075", "0076", "0077", "0078", "0079", "0080", "1120", "1121", "1122", "1123", "1124", "1125", "1126", "2140", "2141", "2142", "2143", "2144", "2145", "2146", "2147", "2148", "2149", "2150", "2151", "2152", "2153", "2154", "2155", "2156", "2157", "2158", "2159", "2280", "2281", "2282", "2283", "2284", "2285", "2286", "2287", "2288", "2289", "2290", "2291", "2292", "2293", "2294", "2550", "2551", "2552", "2553", "2554", "2555", "2556", "2557", "2558", "2559", "2560", "2561", "2562", "2563", "2564", "2565", "2566", "2567", "2568", "2569", "2775", "2776", "2777", "2778", "2779", "2780", "2781", "2782", "2783", "2784", "2785", "2786", "2787", "2788", "2789", "2790", "2791", "2792", "2793", "2794", "2795", "2796", "2797", "2798", "2799", "3080", "3081", "3082", "3083", "3084", "3085", "3086", "3087", "3088", "3089", "3090", "3091", "3092", "3093", "3094", "3095", "3096", "3097", "3098", "3099", "3100", "3101", "3102", "3103", "3104", "3105", "3106", "3107", "3108", "3109");
}
sub conlltest {
  my @flist = ("0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020", "0021", "0022", "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031", "0032", "0033", "0034", "0035", "0036", "0037", "0038", "0039", "0040", "0902", "0903", "0904", "0905", "0906", "0907", "0908", "0909", "0910", "0911", "0912", "0913", "0914", "0915", "0916", "0917", "0918", "0919", "0920", "0921", "0922", "0923", "0924", "0925", "0926", "0927", "0928", "0929", "0930", "0931", "1020", "1044", "1060", "1061", "1072", "1118", "1119", "1132", "1141", "1142", "1148", "2165", "2166", "2167", "2168", "2169", "2170", "2171", "2172", "2173", "2174", "2175", "2176", "2177", "2178", "2179", "2180", "2295", "2296", "2297", "2298", "2299", "2300", "2301", "2302", "2303", "2304", "2305", "2306", "2307", "2308", "2309", "2310", "2570", "2571", "2572", "2573", "2574", "2575", "2576", "2577", "2578", "2579", "2580", "2581", "2582", "2583", "2584", "2585", "2586", "2587", "2588", "2589", "2590", "2591", "2592", "2593", "2594", "2595", "2596", "2597", "2598", "2599", "2600", "2601", "2602", "2800", "2801", "2802", "2803", "2804", "2805", "2806", "2807", "2808", "2809", "2810", "2811", "2812", "2813", "2814", "2815", "2816", "2817", "2818", "2819", "3110", "3111", "3112", "3113", "3114", "3115", "3116", "3117", "3118", "3119", "3120", "3121", "3122", "3123", "3124", "3125", "3126", "3127", "3128", "3129", "3130", "3131", "3132", "3133", "3134", "3135", "3136", "3137", "3138", "3139", "3140", "3141", "3142", "3143", "3144", "3145"); 
}
#
#  Package: Phrase
#
package SRL::phrase;

sub new {
  my ($pkg) = @_;
  my $p = {};
  bless $p, $pkg;

  $p->{start} = undef;
  $p->{end} = undef;
  $p->{type} = undef;

  return $p;
}

sub set_position {
  my $p = shift;
  my $start = shift;
  my $end = shift;
  my $type = shift;

  if ($start > $end) { return 0; }

  $p->{start} = $start;
  $p->{end} = $end;
  $p->{type} = $type;

  return 1;
}

#check if phrase is valid
sub valid {
  my $p = shift;

  return (defined($p->{start}) && defined($p->{end}) && ($p->{start} >= 0) && ($p->{start} <= $p->{end}));
}

#return list of SE tags
sub to_SE {
  my $p = shift;
  my $i;
  my $SE = [];

  if ($p->valid) {
#   $SE->[0] = "($p->{type}*";
    $SE->[0] = "$p->{type}";
    for ($i = $p->{start} + 1; $i <= $p->{end}; $i++) {
      push @$SE, "_";
    }
#   $SE->[$p->{end} - $p->{start}] .= "$p->{type})";
    $SE->[$p->{end} - $p->{start}] = "$p->{type}";
  } else {
    print STDERR "phrase->to_SE: Cannot generate SE tags for an invalid phrase\n";
  }

  return $SE;
}

sub type_to_suffix {
  my $type = shift;

  if (!defined($type)) { return ""; }
  if ($type eq "") { return ""; }
  return "-$type";
}


1;

#
# Package: Proposition    
#   
package SRL::prop;

sub new {
  my ($pkg) = @_;

  my $p = {};

  bless $p, $pkg;

  $p->init;

  return $p;
}

sub init {
  my $prop = shift;

  $prop->{pred} = undef;    # predicate
  $prop->{predpos} = undef; # predicate position
  $prop->{args} = [];       # list of arguments
  $prop->{start} = 0;       # the position of prop's start 
  $prop->{end} = 0;         # the position of prop's end
}

sub add_arg {
  my $prop = shift;
  my $arg = shift;

  push @{$prop->{args}}, $arg;
}

#return list of SE tags
sub to_SE {
  my $prop = shift;
  my $length = shift;

  my $tags = [];

  my $tmptags;
  my $a;
  my ($i, $j);

  my $ok = 1;

  foreach $a (@{$prop->{args}}) {
    if ($a->valid) {
      if ($a->{end} < $length) {
        $tmptags = $a->to_SE;
 
        for ($i = $a->{start}, $j = 0; $i <= $a->{end}; $i++, $j++) {
          if (!defined($tags->[$i])) {

            $tags->[$i] = $tmptags->[$j];

          } else {
            print STDERR "prop->to_SE: Arguments overlap\n";
            $ok = 0; last;
          }
        }
      } else {
        print STDERR "prop->to_SE: Arguments are off the sentence length.\n";
      }
    } else {
      print STDERR "prop->to_SE: Invalid arguments.\n";
    }
  }

  if ($ok) {
    for ($i = 0; $i < $length; $i++) {
      if (!defined($tags->[$i])) { $tags->[$i] = "_"; }
    }
  } else {
    for ($i = 0; $i < $length; $i++) {
      $tags->[$i] = "_";
    }
  }

  return $tags;
}

#return list of SE tags
sub to_SE_column {
  my $prop = shift;
  my $length = shift;

  my $tags_lref = $prop->to_SE($length);

  return (join " ", @$tags_lref);
}

1;

#
#   Package: Propositions
#
package SRL::props;

sub new {
  my ($pkg) = @_;
  my $p = {};
  bless $p, $pkg;

  $p->init;

  return $p;
}

sub init {
  my $props = shift;

  $props->{length} = 0;     # sentence length
  $props->{props} = {};     # list of props indexed by their positions
}

#number of propositions
sub num_prop {
  my $props = shift;

  return scalar(keys %{$props->{props}});
}

#add propsition
#return 1 if succeeded, i.e. prop didn't exist at the position of prop to add.
sub add_prop {
  my $props = shift;
  my $prop = shift;

  if (!defined($props->{props}->{$prop->{predpos}})) {
    $props->{props}->{$prop->{predpos}} = $prop;
    return 1;
  }
  return 0;
}

#return prop position for prop indexed by order
sub index_to_predpos {
  my $props = shift;
  my $i = shift;

  my @proplist = sort {$a <=> $b} (keys %{$props->{props}});

  return $proplist[$i];
}

#return predicate column;
sub to_pred_column {
  my $props = shift;
  my $preds = [];
  my $i;
  my $prop;

  foreach $i (sort {$a <=> $b} (keys %{$props->{props}})) {
    $prop = $props->{props}->{$i};
    if ($prop->{predpos} < $props->{length}) {
      $preds->[$prop->{predpos}] = $prop->{pred};
    }
  }

  for ($i = 0; $i < $props->{length}; $i++) {
    if (!defined($preds->[$i])) {
      $preds->[$i] = "_";
    }
  }

  return $preds;
}

#return arg column in SE format
sub to_SE_args_by_predpos {
  my $props = shift;
  my $i = shift;

  return $props->{props}->{$i}->to_SE($props->{length});
}

sub to_SE_args_by_index {
  my $props = shift;
  my $i = $props->index_to_predpos(shift);

  return $props->{props}->{$i}->to_SE($props->{length});
}

#return all arg columns in SE format
sub to_SE_args {
  my $props = shift;
  my $prop;

  my $args = [];

  foreach my $pos (sort {$a <=> $b} (keys %{$props->{props}})) {
    $prop = $props->{props}->{$pos};
    push @$args, $prop->to_SE($props->{length});
  }

  return $args;
}

#write to file
sub to_SE_props {
  my $props = shift;

  my @output = ();

  my $preds = $props->to_pred_column;
  my $args = $props->to_SE_args;
  for (my $i = 0; $i < scalar(@$preds); $i ++) {
    my $str = $preds->[$i];
    for (my $j = 0; $j < @$args; $j ++) {
      $str .= " " . $args->[$j]->[$i];
    }
    push @output, $str;
  }

  return @output;
}


1;


###################################################################
#   Package    TBNode objects and reading from file.
#
# This module attempts to give a clean interface to interacting 
# with penn treebank style data in perl.
# 
# Author: scott cotton
# Date: 20030707
#
###################################################################

package TBNode;

#
# constructor, inheritable style
#
sub new {
    my $this = shift;
    my $class = ref($this) || $this;
    my $self = {};
    bless $self, $class;
    $self->{LABEL} = shift;
    $self->{DATA} = shift;
    $self->{PARENT} = undef;
    $self->{CHILDREN} = [];
    $self->{NEXTSIBLING} = undef;
    $self->{PREVSIBLING} = undef;
    $self->{TERMNO} = undef;
    $self->{HEIGHT} = undef;

    $self->{HEADWORDIDX} = {};  # wsun

    $self->{ISHEADPHRASE} = 0;  # 
    $self->{GRAMREL} = undef;   # 
    $self->{ISCONJUNCTION} = 0; # 
    $self->{ISCONJUNCT} = 0;    # 
    $self->{CORR} = "";          # 

    return $self;
}

# (* wsun
#
# get/set c/f-structure correspondence
# 
sub correspondence {
  my $self = shift;
  if (@_) { $self->{CORR} = shift; }
  return $self->{CORR};
}

#
# get/set grammatical relation
# 
sub gramRel {
  my $self = shift;
  if (@_) { $self->{GRAMREL} = shift; }
  return $self->{GRAMREL};
}

#
# get/set grammatical relation
# 
sub headIdx {
  my $self = shift;
  if (@_) { $self->{HEADWORDIDX} = shift; }
  return (keys %{$self->{HEADWORDIDX}});
}

#
# get/set grammatical relation
# 
sub addOneHeadWord {
  my $self = shift;
  my $hi = shift;
  $self->{HEADWORDIDX}->{$hi} = 1;
}

sub iAmHeadPhrase {
  my $self = shift;
  $self->{ISHEADPHRASE} = 1; 
  $self->{GRAMREL} = "HEAD"; 
}

sub isHead {
  my $self = shift;
# return $self->{GRAMREL} eq "HEAD";
  return $self->{ISHEADPHRASE};
}

sub iAmConjunction {
  my $self = shift;
  $self->{ISCONJUNCTION} = 1; 
  $self->{GRAMREL} = "CONJ"; 
}

sub isConjunction {
  my $self = shift;
  return $self->{ISCONJUNCTION};
}

sub iAmConjunct {
  my $self = shift;
  $self->{ISCONJUNCT} = 1; 
  $self->{GRAMREL} = "CONJt"; 
}

sub iAmSerialVP {
  my $self = shift;
  $self->{ISCONJUNCT} = 1; 
  $self->{GRAMREL} = "SerialVP"; 
}

sub isConjunct {
  my $self = shift;
  return $self->{ISCONJUNCT};
}

#
# get the index of the first word in this node; 
#
sub start {
  my $self = shift;
  return $self->termno();
}

#
# get the index of the last word in this node; 
#
sub end {
  my $self = shift;
  my $node = $self;
  if ($node->isTerminal) {
    return $node->termno();
  } else {
    my @children = $node->children();
    return $children[-1]->end();
  }
}

sub headword_indices {
  my $self = shift;
  return $self->{HEADWORDIDX};
}

#
# 
#
sub map_c_and_f_struct {
  my $self = shift;

    foreach my $child ($self->children()) {
      $child->map_c_and_f_struct();
    }

    my $rule = "";
    if ($self->isHead()) {
      $rule = "^=!";
    } elsif ($self->isConjunct()) {
      $rule = "!@^";
    } elsif ($self->isConjunction()) {
      # pass
    } else {
      my $func = $self->gramRel();
      if ($func eq "ADV" or $func =~ /MOD/) {
        $rule = "!@^.ADJ"
      } else {
        $rule = "^.$func=!"
      }
    }

    $self->correspondence($rule);
}

sub to_gpsg_tree {
  my $self = shift;

  my @terms = $self->terminals();
  foreach my $term (@terms) {
    if ($term->isTrace()) {
      my $term_par = $term;
      while (defined($term_par)) {
        if ($term_par->parent()->start() == $term->start() and $term_par->parent()->end() == $term->end()) {
          $term_par = $term_par->parent();
        } else {
          last;
        }
      }
      my $missing_gr = $term_par->gramRel();
      if (not defined($missing_gr) or $missing_gr eq "") {
        next;
      }
      my @good_antic = $term->traceIdEquivsButNonTrace();
      if (@good_antic > 0) {
        my $antic = $good_antic[0];
        my $npar = $term;
        while (defined($npar)) {
#         printf STDERR "%i %i %i %i\n", $npar->{end}, $antic->{end}, $npar->{start}, $antic->{start};
          last if ($npar->end() >= $antic->end() and $npar->start() <= $antic->start());

          my @parts = (split '-', $npar->rawLabel());
          $parts[0] .= "/" . $missing_gr;
          $npar->rawLabel((join '-', @parts));

          $npar = $npar->parent();
        }
      }
    }
  }
}

#
#
sub to_noec_str {
  my $self = shift;

  return "" if ($self->only_contains_trace());

  my $tag = $delimiter . $self->gramRel(); 
  my @orig_ftags = $self->ftags(); 
  if (@orig_ftags > 0) {
    $tag .= "-" . (join "-", @orig_ftags);
  }
  if ($self->isTerminal()) {
    return "(" . $self->label() . $tag . " " . $self->data() . ")";
  } else {
    my $str = "";
#   my $missing = "";
    foreach my $child ($self->children()) {
      if ($child->only_contains_trace()) {
#       my @terms = $child->terminals();
#       my @good_antic = $terms[0]->traceIdEquivsButNonTrace();
#       if (@good_antic > 0) {
#         $missing = $child->gramRel();
#       }
      } else {
        $str .= " " . $child->to_noec_str();
      }
    }
#   if ($missing ne "") {
#     $tag = $self->label() .  "/" . $missing  . $tag;
#   } else {
      $tag = $self->label() . $tag;
#   }

    return "(" . $tag . $str . ")";
  }
}

#
# to lisp tree string
#
sub to_str {
  my $self = shift;
# my $str = "(". $self->rawLabel(). $delimiter . $self->gramRel(). " ";
  my $str = "(". $self->label(). $delimiter . $self->gramRel(); 
  my @orig_ftags = $self->ftags(); 
  if (@orig_ftags > 0) {
    $str .= "-" . (join "-", @orig_ftags);
  }
  $str .= " ";
  if ($self->isTerminal()) {
    $str .= $self->data();
  } else {
    foreach my $child ($self->children()) {
      $str .= " ";
      $str .= $child->to_str();
    }
  }
  $str .= ")";
  return $str;
}

#
#
sub to_func_tree_str {
  my $self = shift;

  if ($self->only_contains_trace()) {
    return "";
  }

  # What is the syntactic category of a constituent?
  # It functions as a child of its parent.
  my $par = $self->parent();
  my $par_lab = $par->label();
  $par_lab = $par_lab . $delimiter . $self->gramRel(); 

  if ($self->isTerminal()) {
    return "(" . $par_lab . " " . $self->data() . ")";
  } else {
    my @non_uniry_children = $self->children();
    while (@non_uniry_children == 1 and (not $non_uniry_children[0]->isTerminal())) {
      @non_uniry_children = $non_uniry_children[0]->children();
    }

    my $str = "";
    if (@non_uniry_children == 1) {
      $str = " " . $non_uniry_children[0]->data();
    } else {
      foreach my $child (@non_uniry_children) {
        $str .= " " . $child->to_func_tree_str();
      }
    }
    return "(" . $par_lab . $str . ")";
  }
}

#
# find all terminals in linear order
#
sub terminals {
  my $self = shift;
  my $isterm = sub { my $node = shift; return $node->isTerminal(); };
  my @terms = $self->find($isterm);
  return @terms;
}

#
# if the node contains only one terminal and the terminal is a trace
#
sub only_contains_one_trace {
  my $self = shift;

  if ($self->isTerminal()) { 
    return $self->isTrace();
  } else {
    my @terms = $self->terminals();
    return  (@terms == 1 and $terms[0]->isTrace());
  }
}

#
# if the node contains only one terminal and the terminal is a trace
#
sub only_contains_trace {
  my $self = shift;

  if ($self->isTerminal()) { 
    return $self->isTrace();
  } else {
    foreach my $term ($self->terminals()) {
      if (not $term->isTrace()) {
        return 0;
      }
    }
    return 1;
  }
}

#
# all nodes with the same trace id but have real content.
#
sub traceIdEquivsButNonTrace {
  my $self = shift;
  my @equivs = $self->traceIdEquivs();
  my @eq = ();
  foreach my $equiv (@equivs) {
    if (not $equiv->only_contains_one_trace()) {
      push @eq, $equiv;
    }
  }
  return @eq;
}

sub clean_children {
  my $self = shift;
  $self->{CHILDREN} = [];
}

sub re_parent() {
  my $self = shift;
  if (not $self->isTerminal()) {
    foreach my $child ($self->children()) {
      $child->parent($self);
      $child->re_parent();
    }
  }
}

# wsun *)

#
#
# get/set the label associated with the node, this is the 
# whole label with function tags and all
#
sub rawLabel {
    my $self = shift;
    if (@_) { $self->{LABEL} = shift; }
    return $self->{LABEL};
}

#
# return the primary label of this node
#
sub label {
    my $self = shift;
    return "-NONE-" if $self->rawLabel =~ /-NONE-/;

    my @parts = split('-', $self->rawLabel());
    return $parts[0];
}


#
# get the function tags associated with the node, including NONE, and
# any numeric trace identifiers
#
sub ftags {
    my $self = shift;
    my @tags = split('-', $self->rawLabel());
    return @tags[1 .. $#tags];
}


#
# get/set the data associated with the node
#
sub data {
    my $self = shift;
    if (@_) { $self->{DATA} = shift; }
    return $self->{DATA};
}

#
# get/set the "terminal number" of the node
# this is the index of the least terminal (empty constituents included) 
# which is dominated by this node.
#
sub termno {
    my $self = shift;
    if (@_) { $self->{TERMNO} = shift; }
    return $self->{TERMNO};
}


#
# get/set the height associated with this node.
#
sub height {
    my $self = shift;
    if (@_) { $self->{HEIGHT} = shift; }
    return $self->{HEIGHT};
}


#
# get/set the parent of this node
# returns undef if last sibling
sub parent {
    my $self = shift;
    if (@_) { $self->{PARENT} = shift; }
    return $self->{PARENT};
}

#
# get/set the right sibling of this node
# returns undef if last sibling
#
sub nextSibling {
    my $self = shift;
    if (@_) { $self->{NEXTSIBLING} = shift; }
    return $self->{NEXTSIBLING};
}

#
# get/set the left sibling of this node
# returns undef if this node is leftmost
#
sub prevSibling {
    my $self = shift;
    if (@_) { $self->{PREVSIBLING} = shift; }
    return $self->{PREVSIBLING};
}

#
# get/set the children of this node
#
sub children {
    my $self = shift;
    if (@_) { @{$self->{CHILDREN}} = @_; }
    return @{$self->{CHILDREN}};
}

#
# the first child of this node if it is not a terminal,
# otherwise return undef
#
sub firstChild {
    my $self = shift;
    my @kids = $self->children();
    if ($#kids >= 0) {
        return $kids[0];
    } else {
        return undef;
    }
}

#
# get the root node of this syntax tree
#
sub root {
    my $self = shift;
    my $node = $self;
    while(defined $node->parent()) {
        $node = $node->parent();
    }
    return $node;
}

#
# return the highest index of the list of this nodes children
# eg, a node with a single child would return 0,
# a node with no children would return -1
#
sub nchildren {
    my $self = shift;
    my @foo = $self->children();
    return $#foo;
}

#
# add a child to this node, linking next/prev-Sibling and parent-child
# relations accordingly.
#
sub addChild {
    my $self = shift;
    my $new_child = shift;
    $new_child->parent($self);
    my @kids = $self->children();
    push(@kids, $new_child);
    $self->children(@kids);
    if ($#kids > 0) {
        my $last_child = $kids[$#kids - 1];
        $last_child->nextSibling($new_child);
        $new_child->prevSibling($last_child);
    }
}

#
# print out some stuff that looks vaguely like a treebank file
#
sub show {
    my $self = shift;
    print "(", $self->rawLabel(), $delimiter , $self->gramRel(), " ";
    #print "(", $self->label(), " ";
    #print "(", $self->rawLabel(), " ";
    if ($self->nchildren() >= 0) {
        foreach my $child ($self->children()) {
            $child->show();
        }
        print ")";
    } else {
        print $self->data(), ")";
    }
}


#
# used by nextSentence below to set the terminal number and height of
# internal and root nodes.  not for external use.
#
sub percolate {
    my $self = shift;
    my $lastnode = $self;
    my $node = $self->parent();
    while(defined($node)) {
        last if (defined($node->height()));
        $node->height($lastnode->height() + 1);
        $node->termno($lastnode->termno());
        $lastnode = $node;
        $node = $node->parent();
    }
}


#
# given a predicate and optionall a list ref,
# populate the list with all the nodes either dominated by
# this node or this node which satisfy the predicate.
# return the list.
#
sub find {
    my $self = shift;
    my $predicate = shift;
    my $res = shift || []; 
    if (&$predicate($self)) {
        push(@{$res}, $self);
    }
    foreach my $child ($self->children()) {
        $child->find($predicate, $res);
    }
    return @{$res};
}

#
# iterate code over this node and all its descendents
#
sub iter {
    my $self = shift;
    my $code = shift;
    &$code($self);
    foreach my $child ($self->children()) {
        $child->iter($code);
    }
}


#
# return the trace index associated with this node,
# or undef if there is no such thing.
#
sub traceId {
    my $self = shift;
    my @ftags = $self->ftags();
    foreach my $tag (@ftags) {
        return $tag if $tag =~ /^\d+$/;
    }
    if ($self->data() =~ /^\*.*-(\d+).*/) { return $1; }
    return undef;
}

#
# find all the nodes in this sentence with the same trace id as this node
# 
sub traceIdEquivs {
    my $self = shift;
    my $tid = $self->traceId();
    my $predref = sub { 
        my $node = shift;
        return ($node->traceId() == $tid);
    };
    if (defined ($tid)) { return $self->root()->find($predref); }
    return ($self);
}

#
# return true iff this node has no children
#
sub isTerminal {
    my $self = shift;
    return $self->nchildren() == -1 or $self->nchildren() == 0;
}


#
# return 1 iff this node is "a trace", ie a node without
# children, without a primary label, with a "NONE" function
# tag label.  By convention, nodes with "NONE" in the function
# tags never have primary labels.
# 
# wsun: BUG FIXED!
#
sub isTrace {
  my $self = shift;
  if (! $self->isTerminal() ) { return 0; }
  foreach my $tag ($self->ftags()) {
    if ($tag eq "NONE") {
      return 1;
    }
  }
  return 0;
}


#
# return true iff this node has exactly one
# child which is a trace
#
sub isEmptyNonTerm {
    my $self = shift;
    return $self->nchildren() == 0 && $self->firstChild()->isTrace();
}

#
# follow a numeric trace.
# eg, if we have a tree (A (B-2 b) (C-1 (-NONE- *-2)) (D ...) (E (-NONE- *T*-1)))
# and we take the node (-NONE- *T*-1) as $self, then we return the 
# array ((-NONE- *T*-1), (C-1 (-NONE- *-2)), (-NONE- *-2), (B-2 b))
#
sub followNumeric {
    my $self = shift;
    my %done = ($self => 1);
    my $found = 1;
    my @current = ($self);
    my @res=($self);
    while($found) {
        $found = 0;
        my @tidnodes = ();
        foreach my $c (@current) { push(@tidnodes, $c->traceIdEquivs()); }
        @current = ();
        foreach my $tn (@tidnodes) {
            if ($tn->isEmptyNonTerm() && !defined($done{$tn->firstChild()})) {
                push(@current, $tn->firstChild());
                $found = 1;
            }
            if (!defined($done{$tn})) {
                $done{$tn} = 1;
                push(@res, $tn);
            }
        }
    }
    return @res;
}


sub rawdata {
    my $self = shift;
    my $isterm = sub { my $node = shift; return $node->height() == 0; };
    my @terms = $self->find($isterm);
    my $result = "";
    for(my $i=0; $i<$#terms; $i++) {
        $result .= $terms[$i]->data() . " ";
    }
    $result .= $terms[$#terms]->data();
}


#
# a "static" method which finds the next root node from a file handle
#
sub nextSentence {
  my $fh = shift;
  my @node_stack = ();
  my $buf;
  my $readres;
  my $char;
  my $in_sent=0;
  my $nlabel;
  my $root;
  my $terminal=0;
  while (($readres = read($fh, $char, 1)) != 0) {
    die "unable to read from $fh! $!\n" unless (defined $readres);
    if ($char eq "(") {
      $nlabel = "";
      $in_sent = 1;
      my $firstwhite = 1;
      while(($readres = read($fh, $char, 1)) != 0) {
        die "unable to read from $fh! $!\n" unless (defined $readres);
        if ($firstwhite) {
          if ($char =~ /\s/) { next; }
          elsif ($char eq "(") {
            my $node = TBNode->new("TOP");
            if ($#node_stack >= 0) {
              my $nparent = $node_stack[$#node_stack];
              $nparent->addChild($node);
            } else {
              $root = $node;
            }
            push (@node_stack, $node);
            $firstwhite=1;
            $nlabel="";
            next;
          }
          $nlabel .= $char;
          $firstwhite=0;
          next;
        }
        last if ($char =~ /\s/);
        $nlabel .= $char;
      }
      my $gram_lab;
      if ($nlabel =~ s/$delimiter([^\-]*)//) {
        $gram_lab = $1;
      }
      my $node = TBNode->new($nlabel);
      if (defined($gram_lab)) {
      #print STDERR "$nlabel\t$gram_lab\n";
        if ($gram_lab eq "HEAD") {
          $node->iAmHeadPhrase();
        } elsif ($gram_lab eq "CONJ") {
          $node->iAmConjunction();
        } elsif ($gram_lab eq "CONJt") {
          $node->iAmConjunct();
        } elsif ($gram_lab eq "SerialVP") {
          $node->iAmSerialVP();
        } else {
          $node->gramRel($gram_lab);
        }
      }
      if ($#node_stack >= 0) {
        my $nparent = $node_stack[$#node_stack];
        $nparent->addChild($node);
      } else {
        $root = $node;
      }
      push(@node_stack, $node);
    } elsif ($in_sent && $char eq ")") {
      pop @node_stack;
      if ($#node_stack < 0) { $in_sent=0; last; }
    } elsif ($in_sent && $char !~ /\s/) {
      my $contents = $char;
      while(($readres = read($fh, $char, 1)) != 0) {
        die "unable to read from $fh! $!\n" unless (defined $readres);
        next if ($char =~ /\s/);
        if ($char eq ")") {
          my $node = pop @node_stack;
          if ($#node_stack < 0) { $in_sent = 0; }
          $node->data($contents);
          $node->termno($terminal);
          $node->height(0);
          $node->percolate();
          $terminal++;
          last;
        } 
        $contents .= $char;
      }
    } else { 
      next; 
    }
  }
  if ($#node_stack != -1) {
    die "mismatched parens. $#node_stack\n";
  }
  return $root;
}

1;

