package scripts;

import edu.berkeley.nlp.syntax.Tree;
import edu.berkeley.nlp.syntax.Trees;
import edu.berkeley.nlp.util.BufferedIterator;
import edu.berkeley.nlp.util.LogInfo;
import edu.pku.coli.ltag.PCFGApprox.GrammarTransformer;
import edu.pku.coli.ltag.PCFGApprox.PCFGApproximationTree;
import edu.pku.coli.syntax.LTAGBasicDerivedTreeBuilder;
import edu.pku.coli.syntax.LTAGElementaryTreeTemplatePosAnchorTriplet;
import edu.pku.coli.syntax.LTAGTreeContainer;

import java.io.*;
import java.util.Scanner;

public class LTAGPTB {
    public static void main(String[] args) throws IOException {
        LTAGBasicDerivedTreeBuilder bdtb = null;
        try {
            bdtb = bdtb.buildBDTBuilder();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        Scanner scanner = new Scanner(
                new BufferedReader(
                        new FileReader("/home/chenyufei/work/ijcai18/span/ptb-sdp-kfold-independent/all.cfg")));
        BufferedWriter output = new BufferedWriter(new FileWriter("/home/chenyufei/work/ijcai18/span/ptb-sdp-kfold-independent/all-ltag.cfg"));
        BufferedWriter failed = new BufferedWriter(new FileWriter("all-ltag.failed"));
        String sentId = "";
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine().trim();
            if (line.startsWith("#")) {
                sentId = line;
                continue;
            }
            try {
                Trees.PennTreeReader reader = new Trees.PennTreeReader(new StringReader(
                        line));
                Tree<String> tree = reader.next();
                LTAGTreeContainer treeContainer = new LTAGTreeContainer(tree, bdtb);
                treeContainer.buildDerivedTree();
                treeContainer.buildDerivationTree();
                PCFGApproximationTree transformed = GrammarTransformer
                        .toPCFGTree(treeContainer.getDerivationTree());
                output.write(sentId + "\n");
                output.write(transformed.toString() + "\n");
            } catch(IOException e) {
                e.printStackTrace();
                failed.write(sentId + "\n");
            }
        }
        output.close();
        failed.close();
    }
}
