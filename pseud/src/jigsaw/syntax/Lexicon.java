package jigsaw.syntax;

import java.awt.image.LookupOp;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.io.Writer;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.Vector;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import jigsaw.util.StringUtils;

/**
 * The lexicon of a grammar. The main function of the lexicon is to return
 * P(w|t) through the score() function.
 * 
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Nov 3, 2010
 * 
 * 
 */
public class Lexicon implements Serializable {
	private static final long serialVersionUID = -537673587380271408L;

	public Lexicon(SymbolTable tags) {
		_tags = tags;
	}

	public Lexicon() {
		this(new SymbolTable());
	}

	// known word surface forms
	private SymbolTable _words = new SymbolTable();
	// tags
	private SymbolTable _tags = null;

	public SymbolTable tags() {
		return _tags;
	}

	public String word(int w) {
		return _words.lookup(w);
	}
	
	// C.Wang add at 5th, May
	public int word(String word){
		return _words.lookup(word);
	}

	public String tag(int t) {
		return _tags.lookup(t);
	}

	public void loadLexicon(String filename) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		Pattern p = Pattern.compile("([^ ]+) ([0-9]+)");

		while ((line = br.readLine()) != null && line.length() > 0) { // end with an empty line
			String[] toks = line.split("\\t");
			String wordS = toks[0];
			int word = _words.register(wordS);
			for (int i = 1; i < toks.length; i++) {
				Matcher m = p.matcher(toks[i]);
				if (m.matches()) {
					String tagS = m.group(1);
					int count = Integer.parseInt(m.group(2));
					// TODO
					int tag = _tags.register(tagS);
					incrSeenCount(word, tag, count);
				}
			}
		}
		buildUWModel();
		br.close();
	}

	public void loadLexicon(BufferedReader br) throws IOException {
		String line = null;
		Pattern p = Pattern.compile("([^ ]+) ([0-9]+)");

		while ((line = br.readLine()) != null
				 && line.trim().length() > 0) { //XXX modified 05/08
			String[] toks = line.split("\\t");
			String wordS = toks[0];
			int word = _words.register(wordS);
			for (int i = 1; i < toks.length; i++) {
				Matcher m = p.matcher(toks[i]);
				if (m.matches()) {
					String tagS = m.group(1);
					int count = Integer.parseInt(m.group(2));
					// TODO
					int tag = _tags.register(tagS);
					incrSeenCount(word, tag, count);
				}
			}
		}
		buildUWModel();
	}

	public void dumpLexicon(String filename) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename
				+ ".lexicon"));
		for (int word : _wordTags.keySet()) {
			String wordS = _words.lookup(word);
			bw.write(wordS);
			for (int tag : _wordTags.get(word)) {
				bw.write("\t" + this.tag(tag) + " "
						+ getWordTagCount(word, tag));
			}
			bw.write("\n");
		}
		bw.close();
	}

	// XXX modified 04/21 let's prune it! by C.Wang
	public void dumpLexicon(String filename, int thres) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(filename
				+ ".lexicon"));
		//no more empty line is allowed. Don't call dump(bw, thres) function;
		for (int word : _wordTags.keySet()) {
			if (thres > 0 && _wc.get(word) < thres)
				continue;
			String wordS = _words.lookup(word);
			bw.write(wordS);
			for (int tag : _wordTags.get(word)) {
				bw.write("\t" + this.tag(tag) + " "
						+ getWordTagCount(word, tag));
			}
			bw.write("\n");
		}
		bw.close();
	}

	// XXX modified 04/21 to bufferedReader by C.Wang
	public void dumpLexicon(Writer pw, int thres) throws IOException {
		for (int word : _wordTags.keySet()) {
			if (thres > 0 && _wc.get(word) < thres)
				continue;
			String wordS = _words.lookup(word);
			pw.write(wordS);
			for (int tag : _wordTags.get(word)) {
				pw.write("\t" + this.tag(tag) + " "
						+ getWordTagCount(word, tag));
			}
			pw.write("\n");
		}
		pw.write("\n");
	}

	/**
	 * This scans the original word list for low frequency words, increase the
	 * frequency count of the unknowns. It also builds signatures for the
	 * unknowns for the guessing mechanism.
	 */
	public void buildUWModel() {
		for (int w : _wc.keySet()) {
			if (getWordCount(w) < uwThreshold) { // a rare word
				for (int t : _wordTags.get(w)) { // accumulate tag counts
					String sig = getSignature(_words.lookup(w), -1);
					incrUnseenCount(sig, t, getWordTagCount(w, t));
				}
			}
		}
	}

	public boolean isKnown(String word) {
		return getWordCount(word) > 0;
	}

	public boolean isKnown(int word) {
		return getWordCount(word) > 0;
	}

	/** seen tags indexed by each word */
	private HashMap<Integer, HashSet<Integer>> _wordTags = new HashMap<Integer, HashSet<Integer>>();

	/** returns an iterator of possible tags for the given word */
	public Iterator<Integer> tagIteratorByWord(String word, int loc,
			boolean smoothUnknown) {
		if (isKnown(word))
			return tagIteratorByWord(_words.lookup(word), loc);
		else if (smoothUnknown) { // build signature
			// TODO
			Vector<Integer> tags = new Vector<Integer>();
			for (int t = 0; t < _tags.size(); t++) {
				double s = scoreD(word, t, loc);
				if (s >= pruneThreshold)
					tags.add(t);
			}
			return tags.iterator();
		} else
			return new Vector<Integer>().iterator();
	}

	// XXX modified 03/21
	// public Set<String> lookup(String word) {
	// int wordInt = _words.lookup(word);
	// if(wordInt != -1 && _wordTags.containsKey(wordInt)){
	// Set<String> tags = new HashSet<String>();
	// for (Integer i: _wordTags.get(wordInt))
	// tags.add(tag(i));
	// return tags;
	// }
	// else return Collections.<String>emptySet();
	// }

	// XXX modified 03/21
	public Set<Integer> lookup(String word) {
		int wordInt = _words.lookup(word);
		if (wordInt != -1 && _wordTags.containsKey(wordInt)) {
			// Set<String> tags = new HashSet<String>();
			// for (Integer i: _wordTags.get(wordInt))
			// tags.add(tag(i));
			// return tags;
			return _wordTags.get(wordInt); //just return null if not exist
		} else
			return Collections.<Integer> emptySet();
	}

	/**
	 * TODO here maybe a problem..word check range
	 * 
	 * @param word
	 * @param loc
	 * @return
	 */
	public Iterator<Integer> tagIteratorByWord(int word, int loc) {
		// TODO do we prune here?
		return _wordTags.get(word).iterator();
	}

	/**
	 * returns P(word | tag)
	 * 
	 * @param word
	 * @param tag
	 * @param loc
	 * @return
	 */
	public double scorePetrov(String word, int tag, int loc) {
		int w = _words.lookup(word); // unseen will be w=-1
		double wc = getWordCount(w);
		if (wc == 0 && loc == 0) { // if an unseen at sentence initial position,
									// try the uncapitalized word
			int w_uc = _words.lookup(StringUtils.uncapitalize(word));
			wc = getWordCount(w_uc);
			if (wc != 0) // if the uncapitalized word was seen before, use it;
							// otherwise continue with the capitalized word
				w = w_uc;
		}
		if (wc > 0.0) { // seen
			double tc = getTagCount(tag);
			double wtc = getWordTagCount(w, tag);
			double probTagGivenWord = Double.NEGATIVE_INFINITY;
			if (wc > smoothInUnknownsThreshold) {
				probTagGivenWord = wtc / wc; // prob tag given word;
			} else {
				double probTagGivenUnseen = getUnseenTagCount(tag)
						/ _totalUnseen;
				probTagGivenWord = (wtc + (smooth[0] * probTagGivenUnseen))
						/ (wc + smooth[0]);
			}
			double probTag = tc / _totalSeen;
			double probWord = wc / _totalSeen;
			if (probTag == 0.0) {
				System.out.println("break lex3 0.0");
				return 0.0;
			} else {
				double s = probTagGivenWord * probWord / probTag;
				if (s == 0.0) {
					System.out.println("break lex4 0.0");
				}
				return probTagGivenWord * probWord / probTag;
			}
		} else { // unseen
			String sig = getSignature(word, loc);
			double sc = getUnseenSigCount(sig);
			double stc = getUnseenSigTagCount(sig, tag);
			double utc = getUnseenTagCount(tag);
			double tc = getTagCount(tag);
			double probTagGivenUnseen = utc / _totalUnseen;
			double probTagGivenSig = (stc + (smooth[1] * probTagGivenUnseen))
					/ (sc + smooth[1]);
			// double probTag = tc / _totalSeen;
			// double probSig = sc / _totalSeen;
			// if (sc == 0)
			// return 0.0;
			// else
			// return probTagGivenSig * probSig / probTag / sc;
			if (probTagGivenSig == 0.0) {
				System.out.println("break lex 0.0");
			}
			return probTagGivenSig / tc;
		}
	}

	/**
	 * returns P(tag | word)
	 * 
	 * @param word
	 * @param tag
	 * @param loc
	 * @return
	 */
	public double scoreD(String word, int tag, int loc) {
		int w = _words.lookup(word); // unseen will be w=-1
		double wc = getWordCount(w);
		if (wc > 0.0) { // seen
			double wtc = getWordTagCount(w, tag);
			double probTagGivenWord = Double.NEGATIVE_INFINITY;
			if (wc > smoothInUnknownsThreshold) {
				probTagGivenWord = wtc / wc; // prob tag given word;
			} else {
				double probTagGivenUnseen = getUnseenTagCount(tag)
						/ _totalUnseen;
				probTagGivenWord = (wtc + (smooth[0] * probTagGivenUnseen))
						/ (wc + smooth[0]);
			}
			return probTagGivenWord;
		} else { // unseen
			String sig = getSignature(word, loc);
			double sc = getUnseenSigCount(sig);
			double stc = getUnseenSigTagCount(sig, tag);
			double utc = getUnseenTagCount(tag);
			double probTagGivenUnseen = utc / _totalUnseen;
			double probTagGivenSig = (stc + (smooth[1] * probTagGivenUnseen))
					/ (sc + smooth[1]);
			return probTagGivenSig;
		}
	}

	public double score(String word, int tag, int loc) {
		int w = _words.lookup(word);
		double c_TW = getWordTagCount(w, tag);
		double c_W = getWordCount(w);
		double total = _totalSeen;
		double totalUnseen = _totalUnseen;
		double c_T = getTagCount(tag);
		double c_Tunseen = getUnseenTagCount(tag);

		double pb_W_T; // always set below
		boolean seen = (c_W > 0.0);

		if (seen) {
			// known word model for P(T|W)
			double p_T_U;
			if (useSignatureForKnownSmoothing) { // only works for English
													// currently
				p_T_U = scoreProbTagGivenWordSignature(word, tag, loc,
						smooth[0]);
			} else {
				p_T_U = c_Tunseen / totalUnseen;
			}

			double pb_T_W; // always set below
			if (c_W > smoothInUnknownsThreshold) {
				// we've seen the word enough times to have confidence in its
				// tagging
				pb_T_W = c_TW / c_W;
			} else {
				// we haven't seen the word enough times to have confidence in
				// its
				// tagging
				// double pb_T_W = (c_TW+smooth[1]*x_TW)/(c_W+smooth[1]*x_W);
				pb_T_W = (c_TW + smooth[1] * p_T_U) / (c_W + smooth[1]);
			}
			double p_T = (c_T / total);
			double p_W = (c_W / total);
			pb_W_T = Math.log(pb_T_W * p_W / p_T);

		} else { // when unseen
			if (loc >= 0) {
				pb_W_T = scoreUnknown(word, tag, loc, c_T, total, smooth[0]);
			} else {
				// For negative we now do a weighted average for the dependency
				// grammar :-)
				double pb_W0_T = scoreUnknown(word, tag, 0, c_T, total,
						smooth[0]);
				double pb_W1_T = scoreUnknown(word, tag, 1, c_T, total,
						smooth[0]);
				pb_W_T = Math
						.log((Math.exp(pb_W0_T) + 2 * Math.exp(pb_W1_T)) / 3);
			}
		}

		// Categorical cutoff if score is too low
		if (pb_W_T > -100.0) {
			return pb_W_T;
		}
		return Double.NEGATIVE_INFINITY;
	} // end score()

	public double scoreUnknown(String word, int tag, int loc, double c_Tseen,
			double total, double smooth) {
		double pb_T_S = scoreProbTagGivenWordSignature(word, tag, loc, smooth);
		double p_T = (c_Tseen / total);
		double p_W = 1.0 / total;
		double pb_W_T = Math.log(pb_T_S * p_W / p_T);

		if (pb_W_T > -100.0) {
			return pb_W_T;
		}
		return Double.NEGATIVE_INFINITY;
	}

	public double scoreProbTagGivenWordSignature(String word, int tag, int loc,
			double smooth) {
		// unknown word model for P(T|S)
		String sig = getSignature(word, loc);
		double c_TS = getUnseenSigTagCount(sig, tag);
		double c_S = getUnseenSigCount(sig);
		double c_U = _totalUnseen;
		double c_T = getUnseenTagCount(tag);

		double p_T_U = (c_T + smooth) / (c_U + smooth); // TODO this smoothing
														// was added to make
														// sure p_T_U is
														// non-zero
		// if (unknownLevel == 0) {
		// c_TS = 0;
		// c_S = 0;
		// }
		return (c_TS + smooth * p_T_U) / (c_S + smooth);
	}

	public String getSignature(String word, int loc) {
		StringBuilder sb = new StringBuilder();
		int wlen = word.length();
		int numCaps = 0;
		boolean hasDigit = false;
		boolean hasDash = false;
		boolean hasLower = false;
		for (int i = 0; i < wlen; i++) {
			char ch = word.charAt(i);
			if (Character.isDigit(ch)) {
				hasDigit = true;
			} else if (ch == '-') {
				hasDash = true;
			} else if (Character.isLetter(ch)) {
				if (Character.isLowerCase(ch)) {
					hasLower = true;
				} else if (Character.isTitleCase(ch)) {
					hasLower = true;
					numCaps++;
				} else {
					numCaps++;
				}
			}
		}
		char ch0 = word.charAt(0);
		String lowered = word.toLowerCase();
		if (Character.isUpperCase(ch0) || Character.isTitleCase(ch0)) {
			if (loc == 0 && numCaps == 1) {
				sb.append("-INITC");
				if (isKnown(lowered)) {
					sb.append("-KNOWNLC");
				}
			} else {
				sb.append("-CAPS");
			}
		} else if (!Character.isLetter(ch0) && numCaps > 0) {
			sb.append("-CAPS");
		} else if (hasLower) { // (Character.isLowerCase(ch0)) {
			sb.append("-LC");
		}
		if (hasDigit) {
			sb.append("-NUM");
		}
		if (hasDash) {
			sb.append("-DASH");
		}
		if (lowered.endsWith("s") && wlen >= 3) {
			// here length 3, so you don't miss out on ones like 80s
			char ch2 = lowered.charAt(wlen - 2);
			// not -ess suffixes or greek/latin -us, -is
			if (ch2 != 's' && ch2 != 'i' && ch2 != 'u') {
				sb.append("-s");
			}
		} else if (word.length() >= 5 && !hasDash && !(hasDigit && numCaps > 0)) {
			// don't do for very short words;
			// Implement common discriminating suffixes
			if (lowered.endsWith("ed")) {
				sb.append("-ed");
			} else if (lowered.endsWith("ing")) {
				sb.append("-ing");
			} else if (lowered.endsWith("ion")) {
				sb.append("-ion");
			} else if (lowered.endsWith("er")) {
				sb.append("-er");
			} else if (lowered.endsWith("est")) {
				sb.append("-est");
			} else if (lowered.endsWith("ly")) {
				sb.append("-ly");
			} else if (lowered.endsWith("ity")) {
				sb.append("-ity");
			} else if (lowered.endsWith("y")) {
				sb.append("-y");
			} else if (lowered.endsWith("al")) {
				sb.append("-al");
			}
		}
		return sb.toString();
	}

	public void train() {

	}

	/** probability threshold of P(t|w) for pruning unlikely lexical entries */
	public double pruneThreshold = 1e-3;

	/**
	 * if a word occurs less than this threshold, it is considered to be
	 * unknown, and used for calculating unseen probability
	 */
	public int uwThreshold = 2;

	/**
	 * if a seen word occurs less than this threshold, its probability is
	 * smoothed with tag prob given unseen
	 */
	public int smoothInUnknownsThreshold = 10;

	/** Using signature for known word score smoothing */
	public boolean useSignatureForKnownSmoothing = true;

	/** smoothing factor */
	// public double smooth = 0.1;

	public double[] smooth = { 1.0, 1.0 };

	private HashMap<String, Integer> _wtc = new HashMap<String, Integer>();
	private HashMap<Integer, Integer> _wc = new HashMap<Integer, Integer>();// could
																			// be
																			// optimized
																			// to
																			// use
																			// an
																			// array
	private HashMap<Integer, Integer> _tc = new HashMap<Integer, Integer>();// could
																			// be
																			// optimized
																			// to
																			// use
																			// an
																			// array
	private HashMap<Integer, Integer> _unseentc = new HashMap<Integer, Integer>();
	private HashMap<String, Integer> _unseenstc = new HashMap<String, Integer>();
	private HashMap<String, Integer> _unseensc = new HashMap<String, Integer>();
	private int _totalSeen = 0;
	private int _totalUnseen = 0;

	public void incrSeenCount(int word, int tag, int count) {
		String wt = word + "+" + tag;
		int wtc = count + (_wtc.containsKey(wt) ? _wtc.get(wt) : 0);
		_wtc.put(wt, wtc);
		int wc = count + (_wc.containsKey(word) ? _wc.get(word) : 0);
		_wc.put(word, wc);
		int tc = count + (_tc.containsKey(tag) ? _tc.get(tag) : 0);
		_tc.put(tag, tc);
		_totalSeen += count;
		HashSet<Integer> seentags = _wordTags.containsKey(word) ? _wordTags
				.get(word) : (new HashSet<Integer>());
		seentags.add(tag);
		_wordTags.put(word, seentags);
	}

	public void incrUnseenCount(String sig, int tag, int count) {
		String st = sig + "+" + tag;
		int stc = count + (_unseenstc.containsKey(st) ? _unseenstc.get(st) : 0);
		_unseenstc.put(st, stc);
		int tc = count + (_unseentc.containsKey(tag) ? _unseentc.get(tag) : 0);
		_unseentc.put(tag, tc);
		int sc = count + (_unseensc.containsKey(sig) ? _unseensc.get(sig) : 0);
		_unseensc.put(sig, sc);
		_totalUnseen += count;
	}

	public int getWordCount(int word) {
		return _wc.containsKey(word) ? _wc.get(word) : 0;
	}

	public int getWordCount(String word) {
		int w = _words.lookup(word);
		if (w == -1)
			return 0;
		else
			return getWordCount(w);
	}

	public int getTagCount(int tag) {
		return _tc.containsKey(tag) ? _tc.get(tag) : 0;
	}

	public int getWordTagCount(int word, int tag) {
		return _wtc.containsKey(word + "+" + tag) ? _wtc.get(word + "+" + tag)
				: 0;
	}

	public int getWordTagCount(String word, int tag) {
		int w = _words.lookup(word);
		if (w == -1)
			return 0;
		else
			return getWordTagCount(w, tag);
	}
	
	// Du Yantao added on 17th, Oct, 2013
	public int getWordTagCount(String word, String tag) {
		int w = _words.lookup(word);
		if (w == -1) {
			return 0;
		}
		int t = _tags.lookup(tag);
		if (t == -1) {
			return 0;
		}
		return getWordTagCount(w, t);
	}

	public int getUnseenTagCount(int tag) {
		return _unseentc.containsKey(tag) ? _unseentc.get(tag) : 0;
	}

	public int getUnseenSigTagCount(String sig, int tag) {
		return _unseenstc.containsKey(sig + "+" + tag) ? _unseenstc.get(sig
				+ "+" + tag) : 0;
	}

	public int getUnseenSigCount(String sig) {
		return _unseensc.containsKey(sig) ? _unseensc.get(sig) : 0;
	}

	public void incrSeenCount(String wordS, int tag, int count) {
		int word = _words.register(wordS);
		incrSeenCount(word, tag, count);
	}

	// XXX modified 04/21
	public void incrSeenCount(String wordS, String tag, int count) {
		int word = _words.register(wordS);
		int stag = _tags.register(tag);
		incrSeenCount(word, stag, count);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			Lexicon lexicon = new Lexicon();
			lexicon.loadLexicon(args[0]);
			String word = null;
			System.out.println("Number of tags: " + lexicon._tags.size());
			System.out.println("Total seen words (form/instance): "
					+ lexicon._words.size() + "/" + lexicon._totalSeen);
			System.out.println("Total unseen words (instance): "
					+ lexicon._totalUnseen);
			BufferedReader br = new BufferedReader(new InputStreamReader(
					System.in));
			while ((word = br.readLine()) != null) {
				if (word.equals(""))
					break;
				System.out.println(word + ": ");
				Iterator<Integer> iter = lexicon.tagIteratorByWord(word, 1,
						true);
				while (iter.hasNext()) {
					int tag = iter.next();
					double score = lexicon.score(word, tag, 0);
					double scoreD = lexicon.scoreD(word, tag, 0);
					System.out.println(lexicon.tag(tag) + "\t" + scoreD + "\t"
							+ score);
				}
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

}
