package jigsaw.syntax;

/**
 * @author Yi Zhang <yzhang@coli.uni-sb.de>
 * @date Oct 22, 2010
 *
 */
public interface RecChart {
  public int maxpos();
  public boolean get(short start, short end, int symbol);
  public void set(short start, short end, int symbol);
  public void clear(short start, short end, int symbol);
  public void clearAll();
  public int maxS();
}
