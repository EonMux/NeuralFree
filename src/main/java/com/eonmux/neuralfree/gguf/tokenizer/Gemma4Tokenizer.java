package com.eonmux.neuralfree.gguf.tokenizer;

/**
 * Gemma 4 tokenizer ({@code tokenizer.ggml.model = "gemma4"}).
 *
 * <p>llama.cpp loads this model as a BPE vocabulary with the {@code gemma4} pre-tokenizer
 * forced (src/llama-vocab.cpp:2368-2408): spaces are escaped to U+2581, text is split only into
 * newline and non-newline runs, runs are merged by rank over raw UTF-8 code points (no GPT-2
 * byte encoding), an all-newline run that is itself a token is emitted whole, unmatched symbols
 * fall back to {@code <0xXX>} byte tokens, and BOS is always added
 * (src/llama-vocab.cpp:535-544, 679-685, 2931-2936).</p>
 */
public final class Gemma4Tokenizer implements TokenizerAlgorithm {

    private final BpeTokenizer bpe;

    public Gemma4Tokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || !"gemma4".equals(vocab.model())) {
            throw new IllegalArgumentException("Gemma4Tokenizer requires tokenizer model 'gemma4', got '"
                    + (vocab != null ? vocab.model() : null) + "'");
        }
        this.bpe = new BpeTokenizer(vocab, special, PreTokenizerRegistry.forName("gemma4"));
    }

    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        return bpe.encode(text, addSpecial, parseSpecial);
    }

    @Override
    public String decode(int[] ids) {
        return bpe.decode(ids);
    }

    @Override
    public byte[] piece(int id) {
        return bpe.piece(id);
    }
}
