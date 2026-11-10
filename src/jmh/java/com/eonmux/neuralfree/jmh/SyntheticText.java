package com.eonmux.neuralfree.jmh;

import java.util.SplittableRandom;

/**
 * Deterministic text for the host benchmarks: English-like prose with numbers, punctuation, code and JSON fragments
 * and a share of non-ASCII words, drawn from a fixed-seed generator so every run measures the same bytes.
 */
public final class SyntheticText {

    private static final String[] WORDS = {
        "the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "with", "was", "on", "be", "by", "this",
        "are", "from", "or", "an", "which", "model", "token", "layer", "attention", "weight", "matrix", "vector",
        "decode", "prefill", "sequence", "batch", "kernel", "device", "memory", "buffer", "value", "key", "query",
        "output", "input", "function", "return", "string", "number", "object", "array", "request", "response",
        "server", "client", "latency", "throughput", "generation", "sampling", "temperature", "probability",
        "distribution", "grammar", "schema", "tokenizer", "vocabulary", "embedding", "projection", "normalization",
        "Transformer", "GPU", "CPU", "Java", "llama", "Qwen", "inference", "benchmark", "measurement",
        "über", "naïve", "café", "日本語", "中文", "привет", "données", "größe", "ñandú"
    };

    private static final String[] FRAGMENTS = {
        "{\"name\": \"value\", \"count\": 42}", "for (int i = 0; i < n; i++) {", "return x * y + z;",
        "https://example.com/path?query=1", "x86_64-linux-gnu", "2026-01-01T00:00:00Z", "[1, 2, 3, 4]",
        "if (a != null && b > 0)", "    ", "\n\n", "(see section 3.2)", "e.g.", "i.e.", "--flag=value"
    };

    private SyntheticText() {
    }

    /** {@code chars} characters (at least) of text from {@code seed}. */
    public static String of(long seed, int chars) {
        SplittableRandom random = new SplittableRandom(seed);
        StringBuilder out = new StringBuilder(chars + 64);
        boolean sentenceStart = true;
        while (out.length() < chars) {
            int roll = random.nextInt(100);
            if (roll < 4) {
                out.append(' ').append(FRAGMENTS[random.nextInt(FRAGMENTS.length)]);
            } else if (roll < 10) {
                out.append(' ').append(random.nextInt(100_000));
            } else {
                String w = WORDS[random.nextInt(WORDS.length)];
                if (sentenceStart && Character.isLowerCase(w.charAt(0))) {
                    w = Character.toUpperCase(w.charAt(0)) + w.substring(1);
                }
                out.append(out.length() == 0 ? "" : " ").append(w);
                sentenceStart = false;
            }
            int end = random.nextInt(100);
            if (end < 7) {
                out.append(end < 5 ? "." : end < 6 ? "?" : "!");
                sentenceStart = true;
                if (random.nextInt(8) == 0) {
                    out.append('\n');
                }
            } else if (end < 13) {
                out.append(',');
            }
        }
        return out.toString();
    }
}
