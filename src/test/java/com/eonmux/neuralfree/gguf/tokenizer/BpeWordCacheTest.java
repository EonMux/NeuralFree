package com.eonmux.neuralfree.gguf.tokenizer;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * The word cache of {@link BpeTokenizer}: a tokenizer that has encoded other text, past the cache's capacity and
 * from several threads at once, encodes every text exactly as a fresh tokenizer does, over a byte-level vocabulary
 * with random merges, and over the Hugging Face model overlay with its subword prefix.
 */
public class BpeWordCacheTest {

    private static final String ALPHABET = "abcde";

    @Test
    public void aWarmTokenizerEncodesAsAFreshOne() {
        Random rnd = new Random(3);
        TokenizerVocab vocab = vocab(rnd, false);
        BpeTokenizer shared = new BpeTokenizer(vocab, SpecialTokens.of(vocab));
        for (int t = 0; t < 300; t++) {
            String text = text(rnd, 3, 40);
            int[] fresh = new BpeTokenizer(vocab, SpecialTokens.of(vocab)).encode(text, false, false);
            assertArrayEquals(text, fresh, shared.encode(text, false, false));
            assertArrayEquals(text, fresh, shared.encode(text, false, false));
        }
    }

    @Test
    public void theModelOverlayCachesItsWordsAlike() {
        Random rnd = new Random(4);
        TokenizerVocab vocab = vocab(rnd, true);
        BpeTokenizer shared = new BpeTokenizer(vocab, SpecialTokens.of(vocab));
        for (int t = 0; t < 200; t++) {
            String text = text(rnd, 3, 30);
            int[] fresh = new BpeTokenizer(vocab, SpecialTokens.of(vocab)).encode(text, false, false);
            assertArrayEquals(text, fresh, shared.encode(text, false, false));
        }
    }

    @Test
    public void moreDistinctWordsThanTheCacheHoldsStayExact() {
        Random rnd = new Random(5);
        TokenizerVocab vocab = vocab(rnd, false);
        BpeTokenizer shared = new BpeTokenizer(vocab, SpecialTokens.of(vocab));
        Set<String> words = new LinkedHashSet<>();
        while (words.size() < 25_000) {
            words.add(word(rnd, 7));
        }
        List<String> all = new ArrayList<>(words);
        BpeTokenizer reference = new BpeTokenizer(vocab, SpecialTokens.of(vocab));
        for (int from = 0; from < all.size(); from += 500) {
            String text = String.join(" ", all.subList(from, Math.min(all.size(), from + 500)));
            assertArrayEquals(reference.encode(text, false, false), shared.encode(text, false, false));
        }
        for (int t = 0; t < 50; t++) {
            String text = String.join(" ", all.get(rnd.nextInt(all.size())), all.get(rnd.nextInt(all.size())));
            assertArrayEquals(text, new BpeTokenizer(vocab, SpecialTokens.of(vocab)).encode(text, false, false),
                    shared.encode(text, false, false));
        }
    }

    @Test
    public void concurrentEncodesAgreeWithSingleThreadedOnes() throws Exception {
        Random rnd = new Random(6);
        TokenizerVocab vocab = vocab(rnd, false);
        List<String> texts = new ArrayList<>();
        List<int[]> expected = new ArrayList<>();
        for (int t = 0; t < 400; t++) {
            String text = text(rnd, 1, 60);
            texts.add(text);
            expected.add(new BpeTokenizer(vocab, SpecialTokens.of(vocab)).encode(text, false, false));
        }
        BpeTokenizer shared = new BpeTokenizer(vocab, SpecialTokens.of(vocab));
        ExecutorService pool = Executors.newFixedThreadPool(8);
        try {
            List<Future<Boolean>> runs = new ArrayList<>();
            for (int w = 0; w < 8; w++) {
                int seed = w;
                runs.add(pool.submit(() -> {
                    Random order = new Random(seed);
                    for (int i = 0; i < 2_000; i++) {
                        int k = order.nextInt(texts.size());
                        assertArrayEquals(texts.get(k), expected.get(k), shared.encode(texts.get(k), false, false));
                    }
                    return true;
                }));
            }
            for (Future<Boolean> run : runs) {
                assertTrue(run.get());
            }
        } finally {
            pool.shutdownNow();
        }
    }

    /**
     * A gpt-2 vocabulary of the byte-level alphabet and the space marker plus random merges over them; with
     * {@code model}, the Hugging Face model overlay with a {@code ##} subword prefix instead.
     */
    private static TokenizerVocab vocab(Random rnd, boolean model) {
        String prefix = model ? "##" : "";
        List<String> tokens = new ArrayList<>();
        Set<String> known = new LinkedHashSet<>();
        for (char c : (ALPHABET + "Ġ").toCharArray()) {
            for (String t : model ? List.of(String.valueOf(c), prefix + c) : List.of(String.valueOf(c))) {
                if (known.add(t)) {
                    tokens.add(t);
                }
            }
        }
        List<String> merges = new ArrayList<>();
        Set<String> mergeSet = new LinkedHashSet<>();
        for (int m = 0; m < 150; m++) {
            String left = tokens.get(rnd.nextInt(tokens.size()));
            String right = tokens.get(rnd.nextInt(tokens.size()));
            if (model && !right.startsWith(prefix)) {
                right = prefix + right;
                if (!known.contains(right)) {
                    continue;
                }
            }
            String merged = left + (model ? right.substring(prefix.length()) : right);
            if (merged.length() > 6 || !mergeSet.add(left + " " + right)) {
                continue;
            }
            merges.add(left + " " + right);
            if (known.add(merged)) {
                tokens.add(merged);
            }
        }
        List<Object> types = new ArrayList<>();
        for (int i = 0; i < tokens.size(); i++) {
            types.add(1);
        }
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>(merges));
        if (model) {
            md.put(com.eonmux.neuralfree.gguf.hparams.GgufKeys.NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX, prefix);
        }
        return TokenizerVocab.fromMetadata(md);
    }

    private static String text(Random rnd, int minWords, int maxWords) {
        int n = minWords + rnd.nextInt(maxWords - minWords + 1);
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                out.append(rnd.nextInt(8) == 0 ? "  " : " ");
            }
            out.append(word(rnd, 1 + rnd.nextInt(9)));
        }
        return out.toString();
    }

    private static String word(Random rnd, int length) {
        StringBuilder w = new StringBuilder();
        for (int i = 0; i < length; i++) {
            w.append(ALPHABET.charAt(rnd.nextInt(ALPHABET.length())));
        }
        return w.toString();
    }
}
