package jigsaw.util;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.AbstractList;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.RandomAccess;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * @author Dan Klein
 */
public class CollectionUtils
{
    public interface CollectionFactory<V>
    {

        Collection<V> newCollection();

    }

    public static <E extends Comparable<E>> List<E> sort(Collection<E> c)
    {
        List<E> list = new ArrayList<E>(c);
        Collections.sort(list);
        return list;
    }

    public static <E> boolean isSublistOf(List<E> bigger, List<E> smaller)
    {
        if (smaller.size() > bigger.size()) return false;
        for (int start = 0; start + smaller.size() <= bigger.size(); ++start)
        {
            List<E> sublist = bigger.subList(start, start + smaller.size());
            if (sublist.equals(bigger)) { return true; }
        }
        return false;
    }

    public static <E> List<E> sort(Collection<E> c, Comparator<E> r)
    {
        List<E> list = new ArrayList<E>(c);
        Collections.sort(list, r);
        return list;
    }
  
    public static <K, V extends Comparable<V>> void addToValueSortedSet(Map<K, SortedSet<V>> map, K key, V value)
    {
        SortedSet<V> values = map.get(key);
        if (values == null)
        {
            values = new TreeSet<V>();
            map.put(key, values);
        }
        values.add(value);
    }

    public static <K, V> void addToValueList(Map<K, List<V>> map, K key, V value)
    {
        List<V> valueList = map.get(key);
        if (valueList == null)
        {
            valueList = new ArrayList<V>();
            map.put(key, valueList);
        }
        valueList.add(value);
    }

    public static <K, V> void addToValueCollection(Map<K, Collection<V>> map, K key, V value, CollectionFactory<V> cf)
    {
        Collection<V> valueList = map.get(key);
        if (valueList == null)
        {
            valueList = cf.newCollection();
            map.put(key, valueList);
        }
        valueList.add(value);
    }

    public static <K, V> List<V> getValueList(Map<K, List<V>> map, K key)
    {
        List<V> valueList = map.get(key);
        if (valueList == null) return Collections.emptyList();
        return valueList;
    }

    public static <K, V> Set<V> getValueSet(Map<K, Set<V>> map, K key)
    {
        Set<V> valueSet = map.get(key);
        if (valueSet == null) return Collections.emptySet();
        return valueSet;
    }

    public static <T> List<T> makeList(T... args)
    {
        return new ArrayList<T>(Arrays.asList(args));
    }

    public static <T> void quicksort(T[] array, Comparator<? super T> c)
    {

        quicksort(array, 0, array.length - 1, c);

    }

    public static <T> void quicksort(T[] array, int left0, int right0, Comparator<? super T> c)
    {

        int left, right;
        T pivot, temp;
        left = left0;
        right = right0 + 1;

        final int pivotIndex = (left0 + right0) / 2;
        pivot = array[pivotIndex];
        temp = array[left0];
        array[left0] = pivot;
        array[pivotIndex] = temp;

        do
        {

            do
                left++;
            while (left <= right0 && c.compare(array[left], pivot) < 0);

            do
                right--;
            while (c.compare(array[right], pivot) > 0);

            if (left < right)
            {
                temp = array[left];
                array[left] = array[right];
                array[right] = temp;
            }

        } while (left <= right);

        temp = array[left0];
        array[left0] = array[right];
        array[right] = temp;

        if (left0 < right) quicksort(array, left0, right, c);
        if (left < right0) quicksort(array, left, right0, c);

    }

    public static <T> List<T> doubletonList(T t1, T t2)
    {
        return new DoubletonList(t1, t2);
    }

    public static class MutableSingletonSet<E> extends AbstractSet<E> implements Serializable
    {
        private final class MutableSingletonSetIterator implements Iterator<E>
        {
            private boolean hasNext = true;

            public boolean hasNext()
            {
                return hasNext;
            }

            public E next()
            {
                if (hasNext)
                {
                    hasNext = false;
                    return element;
                }
                throw new NoSuchElementException();
            }

            public void remove()
            {
                throw new UnsupportedOperationException();
            }

            public void reset()
            {
                hasNext = true;

            }
        }

        // use serialVersionUID from JDK 1.2.2 for interoperability
        private static final long serialVersionUID = 3193687207550431679L;

        private E element;

        private final MutableSingletonSetIterator iter;

        public MutableSingletonSet(E o)
        {
            element = o;
            this.iter = new MutableSingletonSetIterator();
        }

        public void set(E o)
        {
            element = o;
        }

        @Override
        public Iterator<E> iterator()
        {
            iter.reset();
            return iter;
        }

        @Override
        public int size()
        {
            return 1;
        }

        @Override
        public boolean contains(Object o)
        {
            return eq(o, element);
        }
    }

    private static class DoubletonList<E> extends AbstractList<E> implements RandomAccess, Serializable
    {

        /**
         * 
         */
        private static final long serialVersionUID = -8444118491195689776L;

        private final E element1;

        private final E element2;

        DoubletonList(E e1, E e2)
        {
            element1 = e1;
            element2 = e2;
        }

        @Override
        public int size()
        {
            return 2;
        }

        @Override
        public boolean contains(Object obj)
        {
            return eq(obj, element1) || eq(obj, element2);
        }

        @Override
        public E get(int index)
        {
            if (index == 0) return element1;
            if (index == 1) return element2;

            throw new IndexOutOfBoundsException("Index: " + index + ", Size: 2");
        }
    }

    private static boolean eq(Object o1, Object o2)
    {
        return (o1 == null ? o2 == null : o1.equals(o2));
    }

    public static <K, V, V2> Map<V, V2> getOrCreateMap(Map<K, Map<V, V2>> map, K key)
    {
        Map<V, V2> r = map.get(key);
        if (r == null) map.put(key, r = new HashMap<V, V2>());
        return r;
    }

}
