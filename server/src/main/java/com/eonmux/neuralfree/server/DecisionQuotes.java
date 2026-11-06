package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * The quoted spellings of a decision family: the token texts an input must not let the tokenizer read as special
 * tokens (llama.cpp's {@code init_quoted_spellings}, tools/server/server-decision.cpp). h2o-lightning quotes them
 * with JSON escapes ({@code h2o_quote}: {@code <} to {@code \\u003c}, {@code >} to {@code \\u003e}), so they stay
 * text; the lfm3 d1 families break them with a zero-width space after their first character ({@code d1_break},
 * repeated until no spelling matches any more). Immutable.
 */
final class DecisionQuotes {

    /** The zero-width space that breaks a spelling in the lfm3 d1 input. */
    private static final String ZWSP = "\u200b";

    /** The spellings that end a quoted text, longest first. */
    private final List<String> spellings;
    /** Their distinct first characters, in spelling order. */
    private final String first;

    private DecisionQuotes(List<String> spellings, String first) {
        this.spellings = spellings;
        this.first = first;
    }

    /**
     * The texts of the tokens of {@code vocab} that carry one of {@code attrs} and at least {@code minSize}
     * characters, longest first (llama.cpp's {@code init_quoted_spellings}).
     */
    static DecisionQuotes of(TokenizerVocab vocab, int attrs, int minSize) {
        List<String> spellings = new ArrayList<>();
        for (int id = 0; id < vocab.size(); id++) {
            if ((vocab.attr(id) & attrs) != 0) {
                String text = vocab.text(id);
                if (text.length() >= minSize) {
                    spellings.add(text);
                }
            }
        }
        spellings.sort(Comparator.comparingInt((String s) -> s.length()).reversed());
        StringBuilder first = new StringBuilder();
        for (String spelling : spellings) {
            if (first.indexOf(String.valueOf(spelling.charAt(0))) < 0) {
                first.append(spelling.charAt(0));
            }
        }
        return new DecisionQuotes(List.copyOf(spellings), first.toString());
    }

    /**
     * Quotes the spellings written in {@code text} with JSON escapes, so they stay text: a {@code <} becomes
     * {@code \\u003c} and a {@code >} a {@code \\u003e} inside each match (llama.cpp's {@code h2o_quote}).
     */
    String quote(String text) {
        StringBuilder out = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); ) {
            String spelling = matchAt(text, i);
            if (spelling == null) {
                out.append(text.charAt(i++));
                continue;
            }
            out.append(spelling.replace("<", "\\u003c").replace(">", "\\u003e"));
            i += spelling.length();
        }
        return out.toString();
    }

    /**
     * Breaks the spellings written in {@code text} with a zero-width space after each match's first character,
     * again for a match that overlapped one broken before it (llama.cpp's {@code d1_break}).
     */
    String breakTokens(String text) {
        String cur = text;
        for (boolean changed = true; changed; ) {
            changed = false;
            StringBuilder out = new StringBuilder(cur.length() + 16);
            for (int i = 0; i < cur.length(); ) {
                String spelling = matchAt(cur, i);
                if (spelling == null) {
                    out.append(cur.charAt(i++));
                    continue;
                }
                int n = utf8LeadBytes(spelling.charAt(0));
                out.append(spelling, 0, n).append(ZWSP).append(spelling.substring(n));
                i += spelling.length();
                changed = true;
            }
            cur = out.toString();
        }
        return cur;
    }

    /** The longest spelling at {@code i}, or null when none matches. */
    private String matchAt(String text, int i) {
        if (first.indexOf(text.charAt(i)) < 0) {
            return null;
        }
        for (String spelling : spellings) {
            if (text.startsWith(spelling, i)) {
                return spelling;
            }
        }
        return null;
    }

    /** The UTF-8 byte length of the character with the lead unit {@code c}. */
    private static int utf8LeadBytes(char c) {
        return c < 0x80 ? 1 : c < 0x800 ? 2 : Character.isHighSurrogate(c) ? 4 : 3;
    }
}
