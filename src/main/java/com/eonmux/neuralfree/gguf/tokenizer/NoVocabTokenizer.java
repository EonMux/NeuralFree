package com.eonmux.neuralfree.gguf.tokenizer;

/**
 * Tokenizer for models without a vocabulary ({@code tokenizer.ggml.model} = none or no_vocab,
 * llama.cpp {@code LLAMA_VOCAB_TYPE_NONE}): the vocabulary holds {@code <arch>.vocab_size} dummy
 * tokens and text cannot be tokenized.
 *
 * <p>Such models consume token ids or embeddings supplied by the caller. As in llama.cpp,
 * tokenizing text is an error ({@code llama_vocab::impl::tokenize} aborts for this type,
 * src/llama-vocab.cpp:3963-3964) and detokenizing yields no text ({@code detokenize} returns 0,
 * src/llama-vocab.cpp:4107-4109); every token therefore renders as an empty piece.</p>
 */
public final class NoVocabTokenizer implements TokenizerAlgorithm {

    private static final byte[] EMPTY = new byte[0];

    private final TokenizerVocab vocab;

    public NoVocabTokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.NONE) {
            throw new IllegalArgumentException("NoVocabTokenizer requires a NONE vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
    }

    /** Always fails: a model without a vocabulary cannot tokenize text. */
    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        throw new UnsupportedOperationException("Model '" + vocab.architecture() + "' has no vocabulary ("
                + TokenizerVocab.KEY_MODEL + " = '" + vocab.model() + "', " + vocab.size()
                + " dummy tokens); it cannot tokenize text and takes token ids or embeddings directly");
    }

    /** Empty: tokens of a model without a vocabulary carry no text. */
    @Override
    public String decode(int[] ids) {
        if (ids == null) throw new IllegalArgumentException("token ids are null");
        return "";
    }

    /** Empty for every id: tokens of a model without a vocabulary carry no text. */
    @Override
    public byte[] piece(int id) {
        return EMPTY.clone();
    }
}
