package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.jmh.SyntheticText;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * BPE encode of a 4 KB text ({@link BpeTokenizer#encode}): the Qwen2 pre-tokenizer split, the GPT-2 byte encoding
 * and the rank-ordered merges, over a byte-level vocabulary whose merges are learned at setup from a 400 KB
 * {@link SyntheticText} corpus, so the merge table has the shape of a real one without a model download.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(1)
public class BpeEncodeBenchmark {

    /** Most merges learned at setup; training stops earlier once no pair occurs twice in the corpus. */
    @Param({"8000"})
    public int merges;

    /** Characters of the encoded text. */
    @Param({"4096"})
    public int chars;

    private BpeTokenizer tokenizer;
    private String text;

    @Setup
    public void setUp() {
        List<String> base = byteTokens();
        BpeTokenizer byteLevel = tokenizer(base, new ArrayList<>());
        List<String> learned = learnMerges(byteLevel.words(SyntheticText.of(1, 400_000)), merges);
        Set<String> tokens = new LinkedHashSet<>(base);
        for (String m : learned) {
            tokens.add(m.replace(" ", ""));
        }
        tokenizer = tokenizer(new ArrayList<>(tokens), learned);
        text = SyntheticText.of(2, chars);
        int[] ids = tokenizer.encode(text, false, false);
        if (ids.length == 0 || !tokenizer.decode(ids).equals(text)) {
            throw new IllegalStateException("the synthetic vocabulary does not round-trip the benchmark text");
        }
    }

    @Benchmark
    public int[] encode() {
        return tokenizer.encode(text, false, false);
    }

    /** The 256 GPT-2 byte-level symbols, in byte order, then the end-of-text control token. */
    private static List<String> byteTokens() {
        List<String> tokens = new ArrayList<>();
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            tokens.add(String.valueOf((char) (printable ? b : 256 + extra++)));
        }
        tokens.add("<|endoftext|>");
        return tokens;
    }

    private static BpeTokenizer tokenizer(List<String> tokens, List<String> mergeList) {
        List<Object> types = new ArrayList<>();
        for (String t : tokens) {
            types.add(t.equals("<|endoftext|>") ? 3 : 1);
        }
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "qwen2");
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put("tokenizer.ggml.merges", new ArrayList<Object>(mergeList));
        md.put("tokenizer.ggml.eos_token_id", tokens.indexOf("<|endoftext|>"));
        TokenizerVocab vocab = TokenizerVocab.fromMetadata(md);
        return new BpeTokenizer(vocab, SpecialTokens.of(vocab));
    }

    /**
     * Classic BPE training over the byte-encoded words: the most frequent adjacent pair (ties by its text) merges
     * next, until {@code count} merges or no pair occurs twice.
     */
    private static List<String> learnMerges(List<String> words, int count) {
        Map<String, Integer> freq = new LinkedHashMap<>();
        for (String w : words) {
            freq.merge(w, 1, Integer::sum);
        }
        List<List<String>> symbols = new ArrayList<>();
        int[] weight = new int[freq.size()];
        int at = 0;
        for (Map.Entry<String, Integer> e : freq.entrySet()) {
            List<String> s = new ArrayList<>();
            e.getKey().codePoints().forEach(cp -> s.add(new String(Character.toChars(cp))));
            symbols.add(s);
            weight[at++] = e.getValue();
        }
        List<String> merges = new ArrayList<>();
        while (merges.size() < count) {
            Map<String, Integer> pairs = new HashMap<>();
            for (int i = 0; i < symbols.size(); i++) {
                List<String> s = symbols.get(i);
                for (int j = 0; j + 1 < s.size(); j++) {
                    pairs.merge(s.get(j) + " " + s.get(j + 1), weight[i], Integer::sum);
                }
            }
            String best = null;
            int bestCount = 1;
            for (Map.Entry<String, Integer> p : pairs.entrySet()) {
                int c = p.getValue();
                if (c > bestCount || (c == bestCount && best != null && p.getKey().compareTo(best) < 0)) {
                    best = p.getKey();
                    bestCount = c;
                }
            }
            if (best == null) {
                break;
            }
            merges.add(best);
            int split = best.indexOf(' ');
            String left = best.substring(0, split);
            String right = best.substring(split + 1);
            String joined = left + right;
            for (List<String> s : symbols) {
                for (int j = 0; j + 1 < s.size(); j++) {
                    if (s.get(j).equals(left) && s.get(j + 1).equals(right)) {
                        s.set(j, joined);
                        s.remove(j + 1);
                    }
                }
            }
        }
        return merges;
    }
}
