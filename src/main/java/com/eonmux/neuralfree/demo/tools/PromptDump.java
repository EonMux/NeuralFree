package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;

import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Objects;

/**
 * The {@code --verbose-prompt} token dump of llama.cpp's embedding and retrieval examples: per prompt its text, its
 * token count and one {@code %6d -> 'piece'} line per token, the piece rendered as {@code common_token_to_piece}
 * renders it ({@link GGUFTokenizer#pieceWithSpecial}: special tokens show their text). Immutable; one instance
 * serves every prompt of a run.
 */
final class PromptDump {

    private final GGUFTokenizer tokenizer;

    PromptDump(GGUFTokenizer tokenizer) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
    }

    /** Appends the block of prompt {@code index}: llama.cpp's {@code main: prompt %d: '%s'} lines. */
    void append(StringBuilder out, int index, String text, int[] tokens) {
        out.append("main: prompt ").append(index).append(": '").append(text).append("'\n");
        out.append("main: number of tokens in prompt = ").append(tokens.length).append('\n');
        for (int id : tokens) {
            out.append(String.format(Locale.ROOT, "%6d -> '%s'\n", id, piece(id)));
        }
        out.append("\n\n");
    }

    /** The token's piece with special tokens rendered as their text ({@code token_to_piece(special = true)}). */
    String piece(int id) {
        if (id < 0 || id >= tokenizer.vocabSize()) {
            throw new IllegalArgumentException("token " + id + " is outside the vocabulary of "
                    + tokenizer.vocabSize());
        }
        return new String(tokenizer.pieceWithSpecial(id), StandardCharsets.UTF_8);
    }
}
