package com.eonmux.neuralfree.jmh;

import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A Qwen-sized vocabulary for the sampler and grammar benchmarks: {@value #SIZE} distinct pieces drawn from a
 * fixed-seed generator in a byte-level vocabulary's mix (words with and without a leading space or capital,
 * digit runs, JSON and code punctuation, whitespace runs, non-ASCII words), with the JSON pieces the benchmarks
 * step through first and one end-of-generation token last. Immutable.
 */
public final class SyntheticVocab implements SamplerVocab {

    /** Qwen3's vocabulary size. */
    public static final int SIZE = 151_936;

    private static final String[] FIXED = {"{", "\"", "name", "\":", " \""};
    private static final String[] PUNCT = {"{", "}", "[", "]", ",", ":", "\"", "\":", "\",", " \"", "{\"", "\"}",
        "\n", "\n\n", "  ", "    ", "(", ")", ";", ".", "->", "==", "!=", "//", "/*", "*/", "\t", "\\n", "'", "`"};
    private static final String LETTERS = "etaoinshrdlcumwfgypbvkjxqz";

    private final byte[][] pieces;
    private final String[] texts;
    private final int eos;

    private SyntheticVocab(List<String> list) {
        pieces = new byte[list.size()][];
        texts = new String[list.size()];
        for (int i = 0; i < list.size(); i++) {
            texts[i] = list.get(i);
            pieces[i] = list.get(i).getBytes(StandardCharsets.UTF_8);
        }
        eos = list.size() - 1;
    }

    /** The vocabulary of {@code seed}. */
    public static SyntheticVocab of(long seed) {
        Set<String> set = new LinkedHashSet<>();
        for (String f : FIXED) {
            set.add(f);
        }
        for (int b = 0x20; b < 0x7F; b++) {
            set.add(String.valueOf((char) b));
        }
        SplittableRandom random = new SplittableRandom(seed);
        while (set.size() < SIZE - 1) {
            set.add(piece(random));
        }
        List<String> list = new ArrayList<>(set);
        list.add("");
        return new SyntheticVocab(list);
    }

    private static String piece(SplittableRandom random) {
        int roll = random.nextInt(100);
        if (roll < 5) {
            return PUNCT[random.nextInt(PUNCT.length)] + PUNCT[random.nextInt(PUNCT.length)];
        }
        if (roll < 9) {
            StringBuilder d = new StringBuilder();
            int n = 1 + random.nextInt(3);
            for (int i = 0; i < n; i++) {
                d.append((char) ('0' + random.nextInt(10)));
            }
            return d.toString();
        }
        if (roll < 11) {
            int cp = 0x4E00 + random.nextInt(0x5000);
            return (random.nextBoolean() ? " " : "") + new String(Character.toChars(cp));
        }
        StringBuilder w = new StringBuilder();
        if (random.nextInt(10) < 6) {
            w.append(' ');
        }
        int len = 1 + Math.min(random.nextInt(6), random.nextInt(12));
        for (int i = 0; i < len; i++) {
            char c = LETTERS.charAt(Math.min(random.nextInt(LETTERS.length()), random.nextInt(LETTERS.length())));
            w.append(i == 0 && random.nextInt(8) == 0 ? Character.toUpperCase(c) : c);
        }
        if (random.nextInt(12) == 0) {
            w.append(PUNCT[random.nextInt(PUNCT.length)]);
        }
        return w.toString();
    }

    @Override
    public int vocabSize() {
        return pieces.length;
    }

    @Override
    public boolean isEog(int id) {
        return id == eos;
    }

    @Override
    public int eotToken() {
        return -1;
    }

    @Override
    public int eosToken() {
        return eos;
    }

    @Override
    public byte[] piece(int id) {
        return pieces[id];
    }

    @Override
    public String pieceText(int id) {
        return texts[id];
    }

    @Override
    public int[] tokenize(String text) {
        throw new UnsupportedOperationException("the synthetic vocabulary does not tokenize");
    }
}
