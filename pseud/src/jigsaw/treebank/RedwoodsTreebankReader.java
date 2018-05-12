package jigsaw.treebank;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import jigsaw.syntax.Tree;
import jigsaw.util.ConcatenationIterator;

public class RedwoodsTreebankReader {
  
  static class Profile {
    File profdir;
    long fromID;
    long toID;
    boolean firstReading = false;
    
    public Profile(File pdir, boolean fr) {
      this(pdir, Long.MIN_VALUE, Long.MAX_VALUE, fr);
    }
    public Profile(File pdir, long from, long to) {
      this(pdir, from, to, false);
    }
    public Profile(File pdir, long from, long to, boolean fr) {
      profdir = pdir;
      fromID = from;
      toID = to;
      firstReading = fr;
    }
  }
  static class TreeCollection extends AbstractCollection<Tree<String>> {

    List<Profile> profiles;
    
    static class TreeIteratorIterator implements Iterator<Iterator<Tree<String>>> {
      Iterator<Profile> profileIterator;
      Iterator<Tree<String>> nextTreeIterator;

      public boolean hasNext() {
        return nextTreeIterator != null;
      }

      public Iterator<Tree<String>> next() {
        Iterator<Tree<String>> currentTreeIterator = nextTreeIterator;
        advance();
        return currentTreeIterator;
      }

      public void remove() {
        throw new UnsupportedOperationException();
      }

      private void advance() {
        nextTreeIterator = null;
        while (nextTreeIterator == null && profileIterator.hasNext()) {
          try {
            Profile profile = profileIterator.next();
            nextTreeIterator = new RedwoodsTreeReader(profile.profdir, profile.fromID, profile.toID, profile.firstReading);
          } catch (FileNotFoundException e) {
          } catch (IOException ex) {
            ex.printStackTrace();
          }
        }
      }

      TreeIteratorIterator(List<Profile> profiles) {
        this.profileIterator = profiles.iterator();
        advance();
      }
    }

    @Override
    public Iterator<Tree<String>> iterator() {
      return new ConcatenationIterator<Tree<String>>(new TreeIteratorIterator(profiles));
    }

    @Override
    public int size() {
      // TODO
      int size = 0;
      Iterator<Tree<String>> i = iterator();
      while (i.hasNext()) {
        size++;
        i.next();
      }
      return size;
    }
    
    public TreeCollection(List<Profile> profiles) {
      this.profiles = profiles;
    }
  }
  
  public static Collection<Tree<String>> readTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>(); 
    for (File pdir : tsdbhome.listFiles()) {
      if (pdir.isDirectory() && (new File(pdir, "relations")).exists()) {
        profiles.add(new Profile(pdir, Integer.MIN_VALUE, Integer.MAX_VALUE));
      }
    }
    return new TreeCollection(profiles);
  }
  
  /** Every section but jh4 are used for training */
  public static Collection<Tree<String>> readLogonTrainingTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "jh0"), 3000011L, 3000922L));
    profiles.add(new Profile(new File(tsdbhome, "jh1"), 3010011L, 3015003L));
    profiles.add(new Profile(new File(tsdbhome, "jh2"), 3020011L, 3024863L));
    profiles.add(new Profile(new File(tsdbhome, "jh3"), 3030011L, 3035483L));
    //profiles.add(new Profile(new File(tsdbhome, "jh4"), 3040011L, 3046163L));
    profiles.add(new Profile(new File(tsdbhome, "jh5"), 3050011L, 3051733L));
    profiles.add(new Profile(new File(tsdbhome, "ps"), 3060011L, 3064003L));
    profiles.add(new Profile(new File(tsdbhome, "tg1"), 3090011L, 3170582L));
    profiles.add(new Profile(new File(tsdbhome, "tg2"), 3180011L, 3270582L));
    return new TreeCollection(profiles);
  }
  /** First 750(*) sentences from jh4 are used for development */
  public static Collection<Tree<String>> readLogonDevTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "jh4"), 3040011L, 3042863L));
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readLogonDevTrees(String path, long from, long to) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "jh4"), from, to));
    return new TreeCollection(profiles);
  }
  /** First 375(*) sentences from jh4 are used for quick parameter tuning (small dev) */
  public static Collection<Tree<String>> readLogonSmallDevTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "jh4"), 3040011L, 3041413L));
    return new TreeCollection(profiles);
  }
  /** The rest of jh4 are used for final testing */
  public static Collection<Tree<String>> readLogonTestTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "jh4"), 3042871, 3046163));
    return new TreeCollection(profiles);
  }
  
  public static Collection<Tree<String>> readWSTrainingTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "ws01"), 10010010L, 10040470L));
    profiles.add(new Profile(new File(tsdbhome, "ws02"), 10050010L, 10150220L));
    profiles.add(new Profile(new File(tsdbhome, "ws03"), 10160010L, 10230480L));
    profiles.add(new Profile(new File(tsdbhome, "ws04"), 10240010L, 10280140L));
    profiles.add(new Profile(new File(tsdbhome, "ws05"), 10290010L, 10311870L));
    profiles.add(new Profile(new File(tsdbhome, "ws06"), 10320010L, 10371850L));
    profiles.add(new Profile(new File(tsdbhome, "ws07"), 10380010L, 10422480L));
    profiles.add(new Profile(new File(tsdbhome, "ws08"), 10430010L, 10481420L));
    profiles.add(new Profile(new File(tsdbhome, "ws09"), 10490010L, 10551290L));
    profiles.add(new Profile(new File(tsdbhome, "ws10"), 10560010L, 10631820L));
    profiles.add(new Profile(new File(tsdbhome, "ws11"), 10640010L, 10702180L));
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readWSDevTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "ws12"), 10710010L, 10771020L));
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readWSSmallDevTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "ws12"), 10710010L, 10730730L));
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readWSTestTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "ws13"), 10780010L, 10860110L));
    return new TreeCollection(profiles);
  }
  
  public static Collection<Tree<String>> readLogonAndWSTrainingTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    profiles.add(new Profile(new File(tsdbhome, "jh0"), 3000011L, 3000922L));
    profiles.add(new Profile(new File(tsdbhome, "jh1"), 3010011L, 3015003L));
    profiles.add(new Profile(new File(tsdbhome, "jh2"), 3020011L, 3024863L));
    profiles.add(new Profile(new File(tsdbhome, "jh3"), 3030011L, 3035483L));
    //profiles.add(new Profile(new File(tsdbhome, "jh4"), 3040011L, 3046163L));
    profiles.add(new Profile(new File(tsdbhome, "jh5"), 3050011L, 3051733L));
    profiles.add(new Profile(new File(tsdbhome, "ps"), 3060011L, 3064003L));
    profiles.add(new Profile(new File(tsdbhome, "tg1"), 3090011L, 3170582L));
    profiles.add(new Profile(new File(tsdbhome, "tg2"), 3180011L, 3270582L));
    profiles.add(new Profile(new File(tsdbhome, "ws01"), 10010010L, 10040470L));
    profiles.add(new Profile(new File(tsdbhome, "ws02"), 10050010L, 10150220L));
    profiles.add(new Profile(new File(tsdbhome, "ws03"), 10160010L, 10230480L));
    profiles.add(new Profile(new File(tsdbhome, "ws04"), 10240010L, 10280140L));
    profiles.add(new Profile(new File(tsdbhome, "ws05"), 10290010L, 10311870L));
    profiles.add(new Profile(new File(tsdbhome, "ws06"), 10320010L, 10371850L));
    profiles.add(new Profile(new File(tsdbhome, "ws07"), 10380010, 10422480L));
    profiles.add(new Profile(new File(tsdbhome, "ws08"), 10430010L, 10481420L));
    profiles.add(new Profile(new File(tsdbhome, "ws09"), 10490010L, 10551290L));
    profiles.add(new Profile(new File(tsdbhome, "ws10"), 10560010L, 10631820L));
    profiles.add(new Profile(new File(tsdbhome, "ws11"), 10640010L, 10702180L));
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readWWTrainingTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    for (File dir : tsdbhome.listFiles()) {
      if (dir.isDirectory()) {
        profiles.add(new Profile(dir, true));
      }
    }
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readWWSmallTrainingTrees(String path) {
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    for (File dir : tsdbhome.listFiles()) {
      if (dir.isDirectory()) {
        int seg = Integer.parseInt(dir.getName());
        if (seg < 300)
          profiles.add(new Profile(dir, true));
      }
    }
    return new TreeCollection(profiles);
  }
  public static Collection<Tree<String>> readWWTrainingTrees(String path, int i) {
    if (i<0 || i>9)
      return null;
    File tsdbhome = new File(path);
    ArrayList<Profile> profiles = new ArrayList<Profile>();
    for (File dir : tsdbhome.listFiles()) {
      if (dir.isDirectory()) {
        int seg = Integer.parseInt(dir.getName());
        if (seg % 10 == i)
          profiles.add(new Profile(dir, true));
      }
    }
    return new TreeCollection(profiles);
  }
}
