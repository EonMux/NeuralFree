package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertEquals;

/**
 * {@link SpecialTokens#partition} against the llama.cpp {@code tokenizer_st_partition} loop it short-cuts: the
 * reference below splits every text fragment once per partition token, longest first, as llama.cpp does, while the
 * partition skips the tokens its {@link SpecialTextIndex} finds absent from the phase's input. Over random
 * vocabularies with overlapping and duplicate special texts, control, user-defined and preserved tokens, the
 * {@code lstrip}, {@code rstrip}, {@code single_word} and {@code normalized} attributes and a lowercasing normalizer,
 * and random texts woven from those texts, both give the same fragments with and without special-token parsing and
 * normalization.
 */
public class SpecialTokensPartitionDifferentialTest {

    private static final String[] TEXTS = {"<a>", "<ab>", "<abq>", "ab", "bc", "BCD", "abc", "<|x|>", "<|x|>y", "Ab",
            "d d", "aa", "a"};
    private static final String[] FILLER = {"x", "y", " ", "\n", "Z", "é", "叫", "a", "b", "c", "D", "<", ">", "|"};

    @Test
    public void partitionMatchesTheSequentialSplit() {
        Random rnd = new Random(11);
        for (int v = 0; v < 120; v++) {
            List<String> tokens = new ArrayList<>(List.of("x", "y", "z"));
            List<Object> types = new ArrayList<>(List.of(1, 1, 1));
            List<Object> attrs = new ArrayList<>(List.of(0, 0, 0));
            for (String text : TEXTS) {
                if (rnd.nextInt(4) == 0) {
                    continue;
                }
                tokens.add(text);
                types.add(rnd.nextBoolean() ? 3 : 4);
                int attr = 0;
                attr |= rnd.nextInt(4) == 0 ? TokenizerVocab.ATTR_LSTRIP : 0;
                attr |= rnd.nextInt(4) == 0 ? TokenizerVocab.ATTR_RSTRIP : 0;
                attr |= rnd.nextInt(4) == 0 ? TokenizerVocab.ATTR_SINGLE_WORD : 0;
                attr |= rnd.nextInt(3) == 0 ? TokenizerVocab.ATTR_NORMALIZED : 0;
                attrs.add(attr);
            }
            Map<String, Object> md = new HashMap<>();
            md.put(TokenizerVocab.KEY_MODEL, "gpt2");
            md.put(TokenizerVocab.KEY_PRE, "gpt-2");
            md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
            md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
            md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
            md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs);
            if (rnd.nextBoolean()) {
                md.put(GgufKeys.NF_TOKENIZER_NORMALIZER, Normalizer.lowercase().toCanonicalJson());
            }
            TokenizerVocab vocab = TokenizerVocab.fromMetadata(md);
            SpecialTokens special = SpecialTokens.of(vocab, rnd.nextBoolean() ? List.of("y") : List.of());
            for (int t = 0; t < 60; t++) {
                String text = text(rnd);
                for (int mode = 0; mode < 4; mode++) {
                    boolean parseSpecial = (mode & 1) != 0;
                    boolean normalize = (mode & 2) != 0;
                    assertEquals("vocab " + v + " text '" + text + "' mode " + mode,
                            render(reference(special, vocab, text, parseSpecial, normalize)),
                            render(special.partition(text, parseSpecial, normalize)));
                }
            }
        }
    }

    @Test
    public void aTextFoundOnlyAsTheSuffixOfAnUnfinishedLongerMatchStillSplits() {
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(List.of("x", "<abq>", "ab")));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, new ArrayList<Object>(List.of(1, 4, 4)));
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        SpecialTokens special = SpecialTokens.of(TokenizerVocab.fromMetadata(md));
        assertEquals(List.of("'<'", "#2", "'>'"), render(special.partition("<ab>", true)));
    }

    private static String text(Random rnd) {
        StringBuilder out = new StringBuilder();
        int parts = rnd.nextInt(12);
        for (int i = 0; i < parts; i++) {
            String part = rnd.nextBoolean() ? TEXTS[rnd.nextInt(TEXTS.length)] : FILLER[rnd.nextInt(FILLER.length)];
            out.append(rnd.nextInt(5) == 0 ? part.toUpperCase() : part);
        }
        return out.toString();
    }

    private static List<String> render(List<SpecialTokens.Fragment> fragments) {
        List<String> out = new ArrayList<>();
        for (SpecialTokens.Fragment f : fragments) {
            out.add(f.isToken() ? "#" + f.token() : "'" + f.text() + "'");
        }
        return out;
    }

    // ------------------------------------------------------------------ the sequential split, every token per pass

    private static List<SpecialTokens.Fragment> reference(SpecialTokens special, TokenizerVocab vocab, String text,
                                                          boolean parseSpecial, boolean normalize) {
        List<SpecialTokens.Fragment> fragments = new ArrayList<>();
        if (text.isEmpty()) {
            return fragments;
        }
        boolean normalizedTokens = false;
        for (int id : special.partitionOrder()) {
            normalizedTokens |= special.hasAttr(id, TokenizerVocab.ATTR_NORMALIZED);
        }
        fragments.add(SpecialTokens.Fragment.text(text));
        fragments = split(special, vocab, fragments, parseSpecial, false, false);
        if (normalize && !vocab.normalizer().isIdentity()) {
            List<SpecialTokens.Fragment> out = new ArrayList<>();
            for (SpecialTokens.Fragment f : fragments) {
                if (f.isToken()) {
                    out.add(f);
                } else {
                    String s = vocab.normalizer().normalize(f.text());
                    if (!s.isEmpty()) {
                        out.add(SpecialTokens.Fragment.text(s));
                    }
                }
            }
            fragments = out;
        }
        if (normalizedTokens) {
            List<SpecialTokens.Fragment> detached = new ArrayList<>();
            for (SpecialTokens.Fragment f : fragments) {
                detached.add(f.isToken() ? f : SpecialTokens.Fragment.text(f.text()));
            }
            fragments = split(special, vocab, detached, parseSpecial, true, normalize);
        }
        return fragments;
    }

    private static List<SpecialTokens.Fragment> split(SpecialTokens special, TokenizerVocab vocab,
                                                      List<SpecialTokens.Fragment> fragments, boolean parseSpecial,
                                                      boolean normalizedPhase, boolean normalize) {
        boolean normalizedText = normalizedPhase && normalize && !vocab.normalizer().isIdentity();
        Set<Integer> preserved = special.preservedIds();
        List<SpecialTokens.Fragment> current = fragments;
        for (int id : special.partitionOrder()) {
            int attr = special.attr(id);
            if (!parseSpecial && ((attr & (TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_UNKNOWN)) != 0
                    || preserved.contains(id))) {
                continue;
            }
            if (((attr & TokenizerVocab.ATTR_NORMALIZED) != 0) != normalizedPhase) {
                continue;
            }
            String text = normalizedText ? vocab.normalizer().normalize(vocab.text(id)) : vocab.text(id);
            if (text.isEmpty()) {
                continue;
            }
            List<SpecialTokens.Fragment> next = new ArrayList<>();
            for (SpecialTokens.Fragment f : current) {
                if (f.isToken()) {
                    next.add(f);
                } else {
                    splitFragment(f, text, id, attr, next);
                }
            }
            current = next;
        }
        return current;
    }

    private static void splitFragment(SpecialTokens.Fragment f, String special, int id, int attr,
                                      List<SpecialTokens.Fragment> out) {
        String s = f.text();
        boolean lstrip = (attr & TokenizerVocab.ATTR_LSTRIP) != 0;
        boolean rstrip = (attr & TokenizerVocab.ATTR_RSTRIP) != 0;
        boolean singleWord = (attr & TokenizerVocab.ATTR_SINGLE_WORD) != 0;
        int pos = 0;
        int from = 0;
        while (true) {
            int match = s.indexOf(special, from);
            if (match < 0) {
                break;
            }
            if (singleWord && (isWordChar(f.codePointBefore(match))
                    || isWordChar(f.codePointAt(match + special.length())))) {
                from = match + special.length();
                continue;
            }
            int leftEnd = match;
            if (lstrip) {
                while (leftEnd > pos && " \t\n\u000B\f\r".indexOf(s.charAt(leftEnd - 1)) >= 0) {
                    leftEnd--;
                }
            }
            if (leftEnd > pos) {
                out.add(f.slice(pos, leftEnd));
            }
            out.add(SpecialTokens.Fragment.token(id));
            pos = match + special.length();
            if (rstrip) {
                while (pos < s.length() && " \t\n\u000B\f\r".indexOf(s.charAt(pos)) >= 0) {
                    pos++;
                }
            }
            from = pos;
        }
        if (pos < s.length()) {
            out.add(pos == 0 ? f : f.slice(pos, s.length()));
        }
    }

    private static boolean isWordChar(int cp) {
        if (cp == SpecialTokens.Fragment.NONE) {
            return false;
        }
        if (Character.isAlphabetic(cp) || cp == 0x200C || cp == 0x200D) {
            return true;
        }
        int type = Character.getType(cp);
        return type == Character.NON_SPACING_MARK || type == Character.ENCLOSING_MARK
                || type == Character.COMBINING_SPACING_MARK || type == Character.DECIMAL_DIGIT_NUMBER
                || type == Character.CONNECTOR_PUNCTUATION;
    }
}
