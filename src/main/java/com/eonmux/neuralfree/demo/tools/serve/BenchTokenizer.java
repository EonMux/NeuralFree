package com.eonmux.neuralfree.demo.tools.serve;

/**
 * The tokenizer the text datasets count their prompts with: the benchmark client either loads one
 * locally over NeuralFree's own tokenizer ({@code --tokenizer}) or implements this over the server's
 * {@code POST /tokenize} endpoint ({@code prompt} form, special tokens added as vLLM's
 * {@code tokenizer(prompt)} adds them); the tests inject counting stubs.
 */
public interface BenchTokenizer {

    /** The token ids of {@code text}, specials included, as the server's tokenizer produces them. */
    int[] tokenize(String text);

    /**
     * The text of a token sequence, as the reference's {@code tokenizer.decode} renders it. The
     * datasets that must send text where the reference sends decoded tokens need it; the default
     * refuses, naming the tokenizer, so a counting stub never pretends to decode.
     *
     * @throws UnsupportedOperationException when this tokenizer cannot decode
     */
    default String detokenize(int[] tokens) {
        throw new UnsupportedOperationException("this benchmark tokenizer cannot decode token ids");
    }

    /** The vocabulary size, when the tokenizer knows it; zero when it does not. */
    default int vocabSize() {
        return 0;
    }

    /** The ids this tokenizer never wants in a synthesized prompt (its special tokens). */
    default int[] specialIds() {
        return new int[0];
    }

    /**
     * The model's chat template, when this tokenizer can render one; null when it carries none.
     * The datasets the reference renders client-side use it ({@link BenchChatTemplate}).
     */
    default BenchChatTemplate chatTemplate() {
        return null;
    }
}
