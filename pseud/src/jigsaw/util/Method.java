package jigsaw.util;

/**
 * A function wrapping interface.
 */
public interface Method<I, O> {
    public O call(I obj);
}