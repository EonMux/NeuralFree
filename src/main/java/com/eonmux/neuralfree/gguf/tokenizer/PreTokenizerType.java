package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.List;

/**
 * llama.cpp {@code llama_vocab_pre_type} (src/llama-vocab.h:10-74): the 63 BPE pre-tokenizer
 * types with their {@code llm_tokenizer_bpe} regex sequence and byte-encoding flag
 * (src/llama-vocab.cpp:280-594). {@link #code()} is the llama.cpp enum value. CHATGLM3, LLAMA4 and
 * PIXTRAL have no case in the llama.cpp constructor and therefore use the default sequence; no
 * {@code tokenizer.ggml.pre} name maps to them.
 */
public enum PreTokenizerType {
    DEFAULT(0, PreTokenizerPatterns.DEFAULT, true),
    LLAMA3(1, PreTokenizerPatterns.LLAMA3, true),
    DEEPSEEK_LLM(2, PreTokenizerPatterns.DEEPSEEK_LLM, true),
    DEEPSEEK_CODER(3, PreTokenizerPatterns.DEEPSEEK_CODER, true),
    FALCON(4, PreTokenizerPatterns.FALCON, true),
    MPT(5, PreTokenizerPatterns.GPT2, true),
    STARCODER(6, PreTokenizerPatterns.STARCODER, true),
    GPT2(7, PreTokenizerPatterns.GPT2, true),
    REFACT(8, PreTokenizerPatterns.STARCODER, true),
    COMMAND_R(9, PreTokenizerPatterns.STARCODER, true),
    STABLELM2(10, PreTokenizerPatterns.QWEN2, true),
    QWEN2(11, PreTokenizerPatterns.QWEN2, true),
    OLMO(12, PreTokenizerPatterns.GPT2, true),
    DBRX(13, PreTokenizerPatterns.LLAMA3, true),
    SMAUG(14, PreTokenizerPatterns.LLAMA3, true),
    PORO(15, PreTokenizerPatterns.PORO, true),
    CHATGLM3(16, PreTokenizerPatterns.DEFAULT, true),
    CHATGLM4(17, PreTokenizerPatterns.LLAMA3, true),
    VIKING(18, PreTokenizerPatterns.VIKING, true),
    JAIS(19, PreTokenizerPatterns.GPT2, true),
    TEKKEN(20, PreTokenizerPatterns.TEKKEN, true),
    SMOLLM(21, PreTokenizerPatterns.STARCODER, true),
    CODESHELL(22, PreTokenizerPatterns.STARCODER, true),
    BLOOM(23, PreTokenizerPatterns.PORO, true),
    GPT3_FINNISH(24, PreTokenizerPatterns.PORO, true),
    EXAONE(25, PreTokenizerPatterns.STARCODER, true),
    CHAMELEON(26, PreTokenizerPatterns.CHAMELEON, true),
    MINERVA(27, PreTokenizerPatterns.STARCODER, true),
    DEEPSEEK3_LLM(28, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    GPT4O(29, PreTokenizerPatterns.GPT4O, true),
    SUPERBPE(30, PreTokenizerPatterns.SUPERBPE, true),
    TRILLION(31, PreTokenizerPatterns.GPT2, true),
    BAILINGMOE(32, PreTokenizerPatterns.BAILINGMOE, true),
    LLAMA4(33, PreTokenizerPatterns.DEFAULT, true),
    PIXTRAL(34, PreTokenizerPatterns.DEFAULT, true),
    SEED_CODER(35, PreTokenizerPatterns.SEED_CODER, true),
    HUNYUAN(36, PreTokenizerPatterns.QWEN2, true),
    KIMI_K2(37, PreTokenizerPatterns.KIMI_K2, true),
    HUNYUAN_DENSE(38, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    GROK_2(39, PreTokenizerPatterns.QWEN2, true),
    GRANITE_DOCLING(40, PreTokenizerPatterns.GPT2, true),
    MINIMAX_M2(41, PreTokenizerPatterns.GPT4O, true),
    AFMOE(42, PreTokenizerPatterns.AFMOE, true),
    SOLAR_OPEN(43, PreTokenizerPatterns.QWEN2, true),
    YOUTU(44, PreTokenizerPatterns.YOUTU, true),
    EXAONE_MOE(45, PreTokenizerPatterns.EXAONE_MOE, true),
    QWEN35(46, PreTokenizerPatterns.QWEN35, true),
    TINY_AYA(47, PreTokenizerPatterns.TINY_AYA, true),
    JOYAI_LLM(48, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    JAIS2(49, PreTokenizerPatterns.JAIS2, true),
    GEMMA4(50, PreTokenizerPatterns.NEWLINE_RUNS, false),
    SARVAM_MOE(51, PreTokenizerPatterns.NEWLINE_RUNS, false),
    MINICPM5(52, PreTokenizerPatterns.MINICPM5, true),
    WHITESPACE(53, PreTokenizerPatterns.WHITESPACE, false),
    GRANITE_EMB_MULTI(54, PreTokenizerPatterns.GPT4O, true),
    MELLUM2(55, PreTokenizerPatterns.STARCODER, true),
    LAGUNA(56, PreTokenizerPatterns.LAGUNA, true),
    HY_V4(57, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    SPARK2_5(58, PreTokenizerPatterns.SPARK2_5, true),
    UFAKZEKA(59, PreTokenizerPatterns.UFAKZEKA, true),
    MMBERT(60, PreTokenizerPatterns.NEWLINE_RUNS, false),
    K2_HORIZON(61, PreTokenizerPatterns.K2_HORIZON, true),
    FLINT(62, PreTokenizerPatterns.FLINT, true);

    private final int code;
    private final List<String> regexes;
    private final boolean byteEncode;

    PreTokenizerType(int code, List<String> regexes, boolean byteEncode) {
        this.code = code;
        this.regexes = regexes;
        this.byteEncode = byteEncode;
    }

    /** The {@code llama_vocab_pre_type} value. */
    public int code() { return code; }

    /** The ordered {@code regex_exprs}, as llama.cpp writes them. */
    public List<String> regexes() { return regexes; }

    /** GPT-2 byte encoding of the split words; false for SPM-style BPE on raw UTF-8. */
    public boolean byteEncode() { return byteEncode; }

    /**
     * Words made only of newlines that are vocabulary tokens skip merging (the Gemma 4 fix,
     * src/llama-vocab.cpp:679-685).
     */
    public boolean wholeNewlineWords() { return this == GEMMA4 || this == MMBERT; }

    /** Words are re-split so that every U+2581 starts a word (mmBERT metaspace, src/llama-vocab.cpp:645-665). */
    public boolean metaspaceWords() { return this == MMBERT; }

    /** {@code add_bos} is forced on after the GGUF override (src/llama-vocab.cpp:2931-2937). */
    public boolean forceAddBos() { return this == GEMMA4; }
}
