package com.eonmux.neuralfree.server.tools;

/**
 * The glob matcher the file and grep searches filter their entries with (llama.cpp's {@code glob_match}):
 * {@code **} matches any characters across separators, {@code *} any characters within one path segment,
 * {@code ?} one character that is not a separator, and {@code [...]} a character class with {@code a-z} ranges
 * and a leading {@code !} negation. A pattern with no {@code /} matches a path's file name at any depth; one
 * with separators matches the whole relative path, auto-prefixed with a recursive-star directory level unless
 * it is already a bare recursive star, starts with one plus a separator, or is anchored by a leading slash.
 */
public final class ToolGlob {

    private ToolGlob() {
    }

    /** Whether {@code pattern} matches the '/'-separated relative path {@code rel} (llama.cpp's path_glob_match). */
    public static boolean match(String pattern, String rel) {
        if (!pattern.contains("/")) {
            return matches(pattern, fileName(rel));
        }
        if (pattern.equals("**") || pattern.startsWith("**/") || pattern.startsWith("/")) {
            return matches(pattern, rel);
        }
        return matches("**/" + pattern, rel);
    }

    /** The part of {@code path} after its last separator. */
    private static String fileName(String path) {
        int at = path.lastIndexOf('/');
        return at < 0 ? path : path.substring(at + 1);
    }

    /** The recursive glob match: {@code p} against {@code s} at the offsets. */
    private static boolean matches(String p, String s) {
        return match(p, 0, s, 0, 0);
    }

    /** Backtracking depth guard, well above what a sane pattern needs. */
    private static final int MAX_DEPTH = 64;

    private static boolean match(String p, int pi, String s, int si, int depth) {
        if (depth >= MAX_DEPTH) {
            return false;
        }
        while (pi < p.length()) {
            char c = p.charAt(pi);
            if (c == '*' && pi + 1 < p.length() && p.charAt(pi + 1) == '*') {
                int next = pi + 2;
                if (next < p.length() && p.charAt(next) == '/') {
                    // "**/" also matches zero segments, so "**/x" matches "x"
                    if (match(p, next + 1, s, si, depth + 1)) {
                        return true;
                    }
                }
                for (int at = si; at <= s.length(); at++) {
                    if (match(p, next, s, at, depth + 1)) {
                        return true;
                    }
                }
                return false;
            }
            if (c == '*') {
                int next = pi + 1;
                int at = si;
                while (true) {
                    if (match(p, next, s, at, depth + 1)) {
                        return true;
                    }
                    if (at >= s.length() || s.charAt(at) == '/') {
                        return false;
                    }
                    at++;
                }
            }
            if (c == '?') {
                if (si >= s.length() || s.charAt(si) == '/') {
                    return false;
                }
                pi++;
                si++;
                continue;
            }
            if (c == '[') {
                int end = classEnd(p, pi);
                if (end < 0) {
                    // no closing bracket: the bracket is a literal
                    if (si >= s.length() || s.charAt(si) != '[') {
                        return false;
                    }
                    pi++;
                    si++;
                    continue;
                }
                if (si >= s.length() || !classMatch(s.charAt(si), p, pi + 1, end)) {
                    return false;
                }
                pi = end + 1;
                si++;
                continue;
            }
            if (si >= s.length() || s.charAt(si) != c) {
                return false;
            }
            pi++;
            si++;
        }
        return si == s.length();
    }

    /** The index of the closing {@code ]} of the class opening at {@code open}, or -1 without one. */
    private static int classEnd(String p, int open) {
        int at = open + 1;
        if (at < p.length() && (p.charAt(at) == ']' || p.charAt(at) == '-')) {
            at++;
        }
        while (at < p.length() && p.charAt(at) != ']') {
            at++;
        }
        return at < p.length() ? at : -1;
    }

    /** Whether {@code c} matches the class between the brackets at {@code from} (inclusive) and {@code end}. */
    private static boolean classMatch(char c, String p, int from, int end) {
        int at = from;
        boolean negated = false;
        if (at < end && p.charAt(at) == '!') {
            negated = true;
            at++;
        }
        boolean hit = false;
        while (at < end) {
            if (at + 2 < end && p.charAt(at + 1) == '-' && p.charAt(at + 2) != ']') {
                if (c >= p.charAt(at) && c <= p.charAt(at + 2)) {
                    hit = true;
                    break;
                }
                at += 3;
                continue;
            }
            if (p.charAt(at) == c) {
                hit = true;
                break;
            }
            at++;
        }
        return hit != negated;
    }
}
