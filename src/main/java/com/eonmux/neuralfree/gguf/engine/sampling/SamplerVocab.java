package com.eonmux.neuralfree.gguf.engine.sampling;

/**
 * The tokenizer surface the llama.cpp samplers need of the model's vocabulary (the llama_vocab handle of
 * llama.cpp's chain construction): {@link DryStage} tokenizes its sequence breakers and reads token pieces at
 * construction, {@link InfillStage} reads token pieces and the end-of-generation ids per apply. The engine hands one
 * built on the model's tokenizer in through {@code Sampler.create(..., vocab)}.
 */
public interface SamplerVocab {

    /** llama_vocab_n_tokens. */
    int vocabSize();

    /** llama_vocab::is_eog: the resolved end-of-generation set, FIM_PAD/REP/SEP included. */
    boolean isEog(int id);

    /** token_eot(), or negative when the vocabulary defines none (llama.cpp's LLAMA_TOKEN_NULL). */
    int eotToken();

    /** token_eos(), or negative (LLAMA_TOKEN_NULL) when the vocabulary defines none. */
    int eosToken();

    /**
     * llama_vocab::token_to_piece(id, lstrip = 0, special = false): the raw bytes the token renders to; UNKNOWN and
     * CONTROL ids render to no bytes.
     */
    byte[] piece(int id);

    /** llama_vocab::detokenize({id}, special = true): the single token's text, its special text included. */
    String pieceText(int id);

    /** llama_tokenize(text, add_special = false, parse_special = false). */
    int[] tokenize(String text);
}
