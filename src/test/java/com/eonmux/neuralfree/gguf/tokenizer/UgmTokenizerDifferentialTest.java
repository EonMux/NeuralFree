package com.eonmux.neuralfree.gguf.tokenizer;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * {@link UgmTokenizer}'s array tries ({@link ByteTrie}) against {@link LegacyUgmTokenizer}'s substring lookups:
 * over random Unigram vocabularies with multi-byte pieces, pieces that prefix one another, user-defined, unused and
 * control tokens, random scores and whitespace flags, both segment random texts into the same ids, or
 * fail alike. Also {@link ByteTrie} itself against a map of its keys, a repeated key keeping its later value.
 */
public class UgmTokenizerDifferentialTest {

    private static final String SP = "▁";
    private static final String[] PIECES = {"a", "b", "c", "ab", SP, SP + "a", "é", "叫", "😀", " ", "x"};

    @Test
    public void segmentationMatchesTheSubstringLookup() {
        Random rnd = new Random(5);
        int compared = 0;
        for (int v = 0; v < 150; v++) {
            List<Object> tokens = new ArrayList<>(List.of("<pad>", "</s>", "<unk>"));
            List<Object> types = new ArrayList<>(List.of(3, 3, 2));
            List<Object> scores = new ArrayList<>(List.of(0f, 0f, 0f));
            int count = 4 + rnd.nextInt(40);
            for (int i = 0; i < count; i++) {
                StringBuilder text = new StringBuilder();
                int parts = 1 + rnd.nextInt(4);
                for (int p = 0; p < parts; p++) {
                    text.append(PIECES[rnd.nextInt(PIECES.length)]);
                }
                if (tokens.contains(text.toString())) {
                    continue;
                }
                tokens.add(text.toString());
                int roll = rnd.nextInt(10);
                types.add(roll < 7 ? 1 : roll == 7 ? 4 : roll == 8 ? 5 : 3);
                scores.add(-rnd.nextFloat() * 10f);
            }
            Map<String, Object> md = new HashMap<>();
            md.put(TokenizerVocab.KEY_MODEL, "t5");
            md.put(TokenizerVocab.KEY_TOKENS, tokens);
            md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
            md.put(TokenizerVocab.KEY_SCORES, scores);
            md.put(TokenizerVocab.KEY_ADD_SPACE_PREFIX, rnd.nextBoolean());
            md.put(TokenizerVocab.KEY_REMOVE_EXTRA_WS, rnd.nextBoolean());
            TokenizerVocab vocab = TokenizerVocab.fromMetadata(md);
            SpecialTokens special = SpecialTokens.of(vocab);
            UgmTokenizer trie = new UgmTokenizer(vocab, special);
            LegacyUgmTokenizer legacy = new LegacyUgmTokenizer(vocab, special);
            for (int t = 0; t < 40; t++) {
                StringBuilder text = new StringBuilder();
                int parts = rnd.nextInt(14);
                for (int p = 0; p < parts; p++) {
                    text.append(rnd.nextInt(6) == 0 ? (String) tokens.get(rnd.nextInt(tokens.size()))
                            : PIECES[rnd.nextInt(PIECES.length)]);
                }
                if (rnd.nextInt(20) == 0) {
                    text.append('\uD800');
                }
                boolean parseSpecial = rnd.nextBoolean();
                String where = "vocab " + v + " text '" + text + "'";
                assertEquals(where, outcome(legacy, text.toString(), parseSpecial),
                        outcome(trie, text.toString(), parseSpecial));
                compared++;
            }
        }
        assertEquals(6000, compared);
    }

    @Test
    public void trieWalksMatchAMapOfItsKeys() {
        Random rnd = new Random(9);
        for (int round = 0; round < 200; round++) {
            int count = rnd.nextInt(30);
            byte[][] keys = new byte[count][];
            int[] values = new int[count];
            Map<String, Integer> expected = new HashMap<>();
            for (int i = 0; i < count; i++) {
                byte[] key = new byte[rnd.nextInt(5)];
                for (int b = 0; b < key.length; b++) {
                    key[b] = (byte) (rnd.nextBoolean() ? 'a' + rnd.nextInt(3) : 0x7E + rnd.nextInt(4));
                }
                keys[i] = key;
                values[i] = rnd.nextInt(1000);
                expected.put(Arrays.toString(key), values[i]);
            }
            ByteTrie trie = new ByteTrie(keys, values);
            for (int q = 0; q < 50; q++) {
                byte[] s = new byte[rnd.nextInt(7)];
                for (int b = 0; b < s.length; b++) {
                    s[b] = (byte) (rnd.nextBoolean() ? 'a' + rnd.nextInt(3) : 0x7E + rnd.nextInt(4));
                }
                int node = ByteTrie.ROOT;
                int depth = 0;
                for (int len = 1; len <= s.length; len++) {
                    node = trie.child(node, s[len - 1]);
                    byte[] prefix = Arrays.copyOf(s, len);
                    boolean anyKeyHasPrefix = false;
                    for (byte[] key : keys) {
                        anyKeyHasPrefix |= key.length >= len && Arrays.equals(Arrays.copyOf(key, len), prefix);
                    }
                    assertEquals(anyKeyHasPrefix, node != ByteTrie.NONE);
                    if (node == ByteTrie.NONE) {
                        break;
                    }
                    depth = len;
                    Integer value = expected.get(Arrays.toString(prefix));
                    assertEquals(value == null ? ByteTrie.NONE : value, trie.value(node));
                }
                assertEquals(depth, trie.pathDepth(s, 0));
            }
        }
    }

    private static String outcome(TokenizerAlgorithm tokenizer, String text, boolean parseSpecial) {
        try {
            return Arrays.toString(tokenizer.encode(text, false, parseSpecial));
        } catch (RuntimeException e) {
            return e.getClass().getSimpleName() + ": " + e.getMessage();
        }
    }
}
