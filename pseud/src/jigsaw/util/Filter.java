package jigsaw.util;

public interface Filter<T> {
  boolean accept(T t);
}
