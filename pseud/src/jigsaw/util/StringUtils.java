package jigsaw.util;

import java.util.*;

public class StringUtils {

  public static String join(Collection<String> s, String delimiter) {
    if (s.isEmpty()) return "";
    Iterator<String> iter = s.iterator();
    StringBuffer buffer = new StringBuffer(iter.next());
    while (iter.hasNext()) {
      buffer.append(delimiter);
      buffer.append(iter.next());
    }
    return buffer.toString();
  }
  
  public static String uncapitalize(String s) {
    if (Character.isUpperCase(s.charAt(0)))
      return Character.toLowerCase(s.charAt(0))+s.substring(1);
    else 
      return new String(s);    
  }

  public static String[] rsplit(String s, String delim, int times) {
    List<String> stack = new ArrayList<>();
    for(int i=0; i<times; i++) {
      int pos = s.lastIndexOf(delim);
      stack.add(s.substring(pos + delim.length()));
      s = s.substring(0, pos);
    }
    stack.add(s);
    Collections.reverse(stack);
    return stack.toArray(new String[0]);
  }

  public static void main(String[] args) {
    for(String s: rsplit("more#__#[a,b] than#__#[c,d]", "#__#", 1)) {
      System.out.println(s);
    }
    for(String s: rsplit("more#__#[a,b] than#__#[c,d]", "#__#", 2)) {
      System.out.println(s);
    }
  }
}
