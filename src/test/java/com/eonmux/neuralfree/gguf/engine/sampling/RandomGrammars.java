package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GbnfParser;
import com.eonmux.neuralfree.gguf.grammar.GrammarException;

import java.nio.charset.StandardCharsets;
import java.util.Random;

/**
 * Random GBNF grammars and a vocabulary whose pieces stress the grammar runtime: single and multi-character ASCII
 * pieces, whole and split two- and three-byte characters ({@code é}, {@code 叫}), pieces ending inside a character,
 * malformed bytes, an empty piece, a piece starting on the terminating 0 and the end-of-generation id 0. The
 * grammars combine literals, character classes and their negations, ranges, any-character, alternations,
 * repetitions, rule references with non-left recursion, token references and token and character budgets.
 */
public final class RandomGrammars {

    private static final byte[][] PIECES = {
            {},                                           // 0: end of generation
            bytes("a"), bytes("b"), bytes("c"), bytes("x"),
            bytes("ab"), bytes("abc"), bytes("ca"), bytes("bb"), bytes("cab"),
            bytes("é"), {(byte) 0xC3}, {(byte) 0xA9},     // whole, lead, tail
            bytes("叫"), {(byte) 0xE5, (byte) 0x8F}, {(byte) 0xAB}, {(byte) 0xE5}, {(byte) 0x8F, (byte) 0xAB},
            bytes("aé"), {(byte) 0xC3, (byte) 0xA9, (byte) 0xE5}, {(byte) 0xAB, 'a'}, bytes("é叫"),
            {(byte) 0x80}, {(byte) 0xFF, 'a'}, {}, {0, 'a'}, bytes("\""), bytes(" "), bytes("a\""), bytes("{"),
            bytes("}"), bytes("ac"), bytes("叫b"), bytes("xé"),
    };

    private static final String[] LITERALS = {"a", "b", "c", "ab", "é", "叫", "aé", "叫b", "\\\"", " ", "x"};
    private static final String[] CLASSES = {"[abc]", "[a-c]", "[^a]", "[é叫]", "[^叫]", "[a-cé]", "[^\"]", "."};

    private RandomGrammars() {
    }

    private static byte[] bytes(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    /** The stress vocabulary: id 0 is the end-of-generation token. */
    public static final SamplerVocab VOCAB = new SamplerVocab() {
        @Override public int vocabSize() { return PIECES.length; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return PIECES[id].clone(); }
        @Override public String pieceText(int id) { return new String(PIECES[id], StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            for (int id = 1; id < PIECES.length; id++) {
                if (new String(PIECES[id], StandardCharsets.UTF_8).equals(text)) {
                    return new int[]{id};
                }
            }
            return new int[0];
        }
    };

    /** A random grammar that parses, drawn from {@code rnd}. */
    public static Grammar next(Random rnd) {
        while (true) {
            String source = source(rnd);
            try {
                return GbnfParser.parse(source);
            } catch (GrammarException e) {
                // a draw the parser refuses (an empty repetition, a recursion it rejects): draw again
            }
        }
    }

    /** A random GBNF source: a root and up to three helper rules referenced acyclically or behind a literal. */
    static String source(Random rnd) {
        int rules = 1 + rnd.nextInt(3);
        StringBuilder out = new StringBuilder();
        for (int r = 0; r < rules; r++) {
            String name = r == 0 ? "root" : "r" + r;
            String attribute = r > 0 && rnd.nextInt(4) == 0
                    ? (rnd.nextBoolean() ? "[max_tokens=" + (1 + rnd.nextInt(3)) + "]"
                    : "[max_chars=" + (1 + rnd.nextInt(4)) + "]") : "";
            out.append(name).append(attribute).append(" ::= ").append(expr(rnd, r, rules, 0)).append('\n');
        }
        return out.toString();
    }

    private static String expr(Random rnd, int rule, int rules, int depth) {
        int alternates = 1 + rnd.nextInt(depth == 0 ? 3 : 2);
        StringBuilder out = new StringBuilder();
        for (int a = 0; a < alternates; a++) {
            if (a > 0) {
                out.append(" | ");
            }
            int items = 1 + rnd.nextInt(3);
            for (int i = 0; i < items; i++) {
                if (i > 0) {
                    out.append(' ');
                }
                out.append(item(rnd, rule, rules, depth, i == 0));
            }
        }
        return out.toString();
    }

    private static String item(Random rnd, int rule, int rules, int depth, boolean first) {
        String atom;
        int pick = rnd.nextInt(10);
        if (pick < 3) {
            atom = "\"" + LITERALS[rnd.nextInt(LITERALS.length)] + "\"";
        } else if (pick < 6) {
            atom = CLASSES[rnd.nextInt(CLASSES.length)];
        } else if (pick == 6 && rule + 1 < rules) {
            atom = "r" + (rule + 1 + rnd.nextInt(rules - rule - 1));
        } else if (pick == 6 && !first && rule > 0) {
            atom = "r" + rule; // recursion behind an earlier item, never left recursion
        } else if (pick == 7 && depth < 2) {
            atom = "(" + expr(rnd, rule, rules, depth + 1) + ")";
        } else if (pick == 8) {
            int id = 1 + rnd.nextInt(PIECES.length - 1);
            atom = rnd.nextInt(3) == 0 ? "!<>" : (rnd.nextBoolean() ? "<[" + id + "]>" : "!<[" + id + "]>");
        } else {
            atom = "\"" + LITERALS[rnd.nextInt(LITERALS.length)] + "\"";
        }
        switch (rnd.nextInt(6)) {
            case 0: return atom + "*";
            case 1: return atom + "+";
            case 2: return atom + "?";
            case 3: return atom + "{0,2}";
            default: return atom;
        }
    }
}
