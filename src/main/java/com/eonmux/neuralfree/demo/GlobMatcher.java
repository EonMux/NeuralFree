package com.eonmux.neuralfree.demo;

/**
 * Shell glob matching over {@code /}-separated relative paths, the matcher llama.cpp's {@code /glob} command
 * uses (common/common.cpp {@code glob_match}): {@code **} crosses directory separators, {@code *} and {@code ?}
 * do not, and {@code [abc]}, {@code [a-z]} and {@code [!x]} are character classes.
 */
final class GlobMatcher {

    /** The recursion depth past which a pattern stops matching (common.h COMMON_RECURSIVE_LIMIT). */
    private static final int RECURSIVE_LIMIT = 128;

    private GlobMatcher() {
    }

    /** Whether {@code text} matches {@code pattern}. */
    static boolean matches(String pattern, String text) {
        return matches(pattern, 0, text, 0, 0);
    }

    private static boolean matches(String p, int pi, String s, int si, int depth) {
        if (depth >= RECURSIVE_LIMIT) {
            return false;
        }
        while (pi < p.length()) {
            char c = p.charAt(pi);
            if (c == '*' && pi + 1 < p.length() && p.charAt(pi + 1) == '*') {
                int next = pi + 2;
                for (int k = si; k <= s.length(); k++) {
                    if (matches(p, next, s, k, depth + 1)) {
                        return true;
                    }
                }
                return false;
            }
            if (c == '*') {
                int next = pi + 1;
                int k = si;
                for (; k < s.length() && s.charAt(k) != '/'; k++) {
                    if (matches(p, next, s, k, depth + 1)) {
                        return true;
                    }
                }
                return matches(p, next, s, k, depth + 1);
            }
            if (c == '?' && si < s.length() && s.charAt(si) != '/') {
                pi++;
                si++;
                continue;
            }
            if (c == '[' && si < s.length()) {
                int close = classEnd(p, pi);
                if (close > 0) {
                    int matched = classMatch(p, pi + 1, close, s.charAt(si));
                    if (matched < 0) {
                        return false;
                    }
                    pi = matched;
                    si++;
                    continue;
                }
            }
            if (si >= s.length() || s.charAt(si) != c) {
                return false;
            }
            pi++;
            si++;
        }
        return si == s.length();
    }

    /** The index after the {@code ]} of the class starting at {@code open}, or -1 without a closed class. */
    private static int classEnd(String p, int open) {
        int i = open + 1;
        if (i < p.length() && (p.charAt(i) == ']' || p.charAt(i) == '-')) {
            i++;
        }
        while (i < p.length() && p.charAt(i) != ']') {
            i++;
        }
        return i < p.length() ? i : -1;
    }

    /**
     * Whether {@code c} is in the class between {@code start} and the {@code ]} at {@code close}
     * ({@code glob_class_match}): a leading {@code !} negates, a {@code ]} or {@code -} right after it is literal,
     * and {@code a-z} is a range. Returns the index after the {@code ]} on a match, or -1.
     */
    private static int classMatch(String p, int start, int close, char c) {
        int i = start;
        boolean negate = false;
        if (p.charAt(i) == '!') {
            negate = true;
            i++;
        }
        boolean hit = false;
        if (p.charAt(i) == ']' || p.charAt(i) == '-') {
            if (i < close && p.charAt(i) == c) {
                return negate ? -1 : close + 1;
            }
            i++;
        }
        while (i < close && !hit) {
            if (i + 2 < close && p.charAt(i + 1) == '-' && p.charAt(i + 2) != ']') {
                hit = p.charAt(i) <= c && c <= p.charAt(i + 2);
                i += 3;
            } else {
                hit = p.charAt(i) == c;
                i++;
            }
        }
        return hit != negate ? close + 1 : -1;
    }
}
