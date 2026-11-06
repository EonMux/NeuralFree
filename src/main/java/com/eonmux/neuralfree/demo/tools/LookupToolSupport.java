package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * The pieces the lookup cache tools share: a model's tokenizer and training context, and a prompt given as text
 * or as a file. The cache files themselves are read and written by
 * {@link com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile}; only the tokenizer metadata is read here, so the
 * tools do not load model weights.
 */
final class LookupToolSupport {

    /** A model's tokenizer and its training context ({@code n_ctx_train}, 0 when the metadata has none). */
    static final class TokenizerModel {

        private final GGUFTokenizer tokenizer;
        private final int contextLength;

        private TokenizerModel(GGUFTokenizer tokenizer, int contextLength) {
            this.tokenizer = tokenizer;
            this.contextLength = contextLength;
        }

        /** Tokenizes text with special tokens parsed and the vocabulary's BOS/EOS policy, as llama.cpp's
         * {@code common_tokenize(..., true, true)} does. */
        int[] encode(String text) {
            return tokenizer.encode(text, true, true);
        }

        /** The model's training context, or 0 when its metadata carries no usable context length. */
        int contextLength() {
            return contextLength;
        }
    }

    private LookupToolSupport() {
    }

    /**
     * Reads the tokenizer and training context of the model at {@code model}.
     *
     * @throws IOException when the model cannot be opened or its tokenizer metadata is invalid
     */
    static TokenizerModel open(Path model) throws IOException {
        try (ModelSource source = ModelSources.open(model.toString())) {
            return new TokenizerModel(new GGUFTokenizer(source), contextLength(source));
        }
    }

    /** The {@code <architecture>.context_length} metadata value, or 0 when it is absent or not a number. */
    private static int contextLength(ModelSource source) {
        Object arch = source.getMetadata().get(GgufKeys.GENERAL_ARCHITECTURE);
        if (!(arch instanceof String)) {
            return 0;
        }
        Object value = source.getMetadata().get(String.format(Locale.ROOT, GgufKeys.CONTEXT_LENGTH, arch));
        return value instanceof Number ? ((Number) value).intValue() : 0;
    }

    /**
     * The text of the prompt file {@code file}, with one trailing newline dropped, as llama.cpp's
     * {@code read_file} does.
     *
     * @throws IOException when the file cannot be read
     */
    static String readPrompt(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        int end = bytes.length;
        if (end > 0 && bytes[end - 1] == '\n') {
            end--;
        }
        return new String(bytes, 0, end, StandardCharsets.UTF_8);
    }
}
