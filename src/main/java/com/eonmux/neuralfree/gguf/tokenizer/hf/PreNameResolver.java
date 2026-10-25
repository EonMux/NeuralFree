package com.eonmux.neuralfree.gguf.tokenizer.hf;

import java.util.function.Function;

/**
 * Resolves the llama.cpp pre-tokenizer name ({@code tokenizer.ggml.pre}) of a Hugging Face tokenizer, the step the
 * converter performs with {@code get_vocab_base_pre} (LC/conversion/base.py:1803-2151).
 */
@FunctionalInterface
public interface PreNameResolver {

    /**
     * The pre-tokenizer name, or null when none matches.
     *
     * @param tj                 the checkpoint's {@code tokenizer.json}, or null when the tokenizer has none (a
     *                           tiktoken or SentencePiece vocabulary)
     * @param encodeWithSpecials encodes a text with the candidate vocabulary as {@code tokenizer.encode} does by
     *                           default: special tokens recognized and the post-processor's special ids added
     */
    String resolve(HfTokenizerJson tj, Function<String, int[]> encodeWithSpecials);
}
