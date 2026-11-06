package com.eonmux.neuralfree.gguf.grammar;

/**
 * The model vocabulary a GBNF {@code <token>} reference resolves through ({@code parse_token} in llama.cpp's
 * {@code llama-grammar.cpp}): the reference's text, angle brackets included, must tokenize to exactly one token.
 */
@FunctionalInterface
public interface GrammarVocab {

    /**
     * llama_tokenize(text, add_special = false, parse_special = true): {@code text}'s token ids, a special token's
     * text ({@code <|channel|>}) matching that token.
     */
    int[] tokenize(String text);
}
