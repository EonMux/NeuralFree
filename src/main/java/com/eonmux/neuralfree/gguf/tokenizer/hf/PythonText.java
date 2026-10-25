package com.eonmux.neuralfree.gguf.tokenizer.hf;

import java.util.ArrayList;
import java.util.List;

/**
 * The Python text semantics the tokenizer file readers reproduce: universal newlines, {@code readlines()},
 * {@code str.isspace()}, {@code str.strip()} and {@code str.split(None, maxsplit)}.
 */
final class PythonText {

    private PythonText() {
    }

    /** {@code text} with {@code \r\n} and lone {@code \r} replaced by {@code \n} (Python universal newlines). */
    static String translateNewlines(String text) {
        if (text.indexOf('\r') < 0) {
            return text;
        }
        return text.replace("\r\n", "\n").replace('\r', '\n');
    }

    /** The lines of newline-translated {@code text} without terminators, as Python {@code readlines()} yields them. */
    static List<String> lines(String text) {
        String t = translateNewlines(text);
        List<String> lines = new ArrayList<>();
        int start = 0;
        while (start < t.length()) {
            int nl = t.indexOf('\n', start);
            if (nl < 0) {
                lines.add(t.substring(start));
                break;
            }
            lines.add(t.substring(start, nl));
            start = nl + 1;
        }
        return lines;
    }

    /** Python {@code str.isspace()} for one code point. */
    static boolean isSpace(int cp) {
        return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680
                || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F
                || cp == 0x3000;
    }

    /** Python {@code str.strip()}. */
    static String strip(String s) {
        int start = 0;
        int end = s.length();
        while (start < end && isSpace(s.codePointAt(start))) {
            start += Character.charCount(s.codePointAt(start));
        }
        while (end > start && isSpace(s.codePointBefore(end))) {
            end -= Character.charCount(s.codePointBefore(end));
        }
        return s.substring(start, end);
    }

    /**
     * Python {@code str.split(None, maxsplit)}: runs of whitespace separate parts; after {@code maxSplit} parts the
     * rest of the string, without its leading whitespace, is the last part.
     */
    static List<String> split(String s, int maxSplit) {
        List<String> parts = new ArrayList<>();
        int i = 0;
        int n = s.length();
        while (i < n) {
            while (i < n && isSpace(s.codePointAt(i))) {
                i += Character.charCount(s.codePointAt(i));
            }
            if (i >= n) {
                break;
            }
            if (parts.size() == maxSplit) {
                parts.add(s.substring(i));
                break;
            }
            int start = i;
            while (i < n && !isSpace(s.codePointAt(i))) {
                i += Character.charCount(s.codePointAt(i));
            }
            parts.add(s.substring(start, i));
        }
        return parts;
    }
}
