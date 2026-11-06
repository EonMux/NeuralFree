package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.Collections;
import java.util.List;

/**
 * llama.cpp {@code llama_vocab_pre_type} (src/llama-vocab.h:10-78): the 67 BPE pre-tokenizer
 * types with their {@code llm_tokenizer_bpe} regex sequence and byte-encoding flag
 * (src/llama-vocab.cpp:281-608), plus the Hugging Face form of the sequences llama.cpp adapted
 * ({@link #hfRegexes()}). {@link #code()} is the llama.cpp enum value. CHATGLM3, LLAMA4 and
 * PIXTRAL have no case in the llama.cpp constructor and therefore use the default sequence; no
 * {@code tokenizer.ggml.pre} name maps to them.
 */
public enum PreTokenizerType {
    DEFAULT(0, PreTokenizerPatterns.DEFAULT, true),
    LLAMA3(1, PreTokenizerPatterns.LLAMA3, PreTokenizerPatterns.LLAMA3_HF, true),
    DEEPSEEK_LLM(2, PreTokenizerPatterns.DEEPSEEK_LLM, true),
    DEEPSEEK_CODER(3, PreTokenizerPatterns.DEEPSEEK_CODER, true),
    FALCON(4, PreTokenizerPatterns.FALCON, true),
    MPT(5, PreTokenizerPatterns.GPT2, true),
    STARCODER(6, PreTokenizerPatterns.STARCODER, true),
    GPT2(7, PreTokenizerPatterns.GPT2, true),
    REFACT(8, PreTokenizerPatterns.STARCODER, true),
    COMMAND_R(9, PreTokenizerPatterns.STARCODER, true),
    STABLELM2(10, PreTokenizerPatterns.QWEN2, PreTokenizerPatterns.QWEN2_HF, true),
    QWEN2(11, PreTokenizerPatterns.QWEN2, PreTokenizerPatterns.QWEN2_HF, true),
    OLMO(12, PreTokenizerPatterns.GPT2, true),
    DBRX(13, PreTokenizerPatterns.LLAMA3, true),
    SMAUG(14, PreTokenizerPatterns.LLAMA3, true),
    PORO(15, PreTokenizerPatterns.PORO, true),
    CHATGLM3(16, PreTokenizerPatterns.DEFAULT, true),
    CHATGLM4(17, PreTokenizerPatterns.LLAMA3, true),
    VIKING(18, PreTokenizerPatterns.VIKING, true),
    JAIS(19, PreTokenizerPatterns.GPT2, true),
    TEKKEN(20, PreTokenizerPatterns.TEKKEN, PreTokenizerPatterns.TEKKEN_HF, true),
    SMOLLM(21, PreTokenizerPatterns.STARCODER, true),
    CODESHELL(22, PreTokenizerPatterns.STARCODER, true),
    BLOOM(23, PreTokenizerPatterns.PORO, true),
    GPT3_FINNISH(24, PreTokenizerPatterns.PORO, true),
    EXAONE(25, PreTokenizerPatterns.STARCODER, true),
    CHAMELEON(26, PreTokenizerPatterns.CHAMELEON, true),
    MINERVA(27, PreTokenizerPatterns.STARCODER, true),
    DEEPSEEK3_LLM(28, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    GPT4O(29, PreTokenizerPatterns.GPT4O, PreTokenizerPatterns.GPT4O_HF, true),
    SUPERBPE(30, PreTokenizerPatterns.SUPERBPE, true),
    TRILLION(31, PreTokenizerPatterns.GPT2, true),
    BAILINGMOE(32, PreTokenizerPatterns.BAILINGMOE, PreTokenizerPatterns.BAILINGMOE_HF, true),
    LLAMA4(33, PreTokenizerPatterns.DEFAULT, true),
    PIXTRAL(34, PreTokenizerPatterns.DEFAULT, true),
    SEED_CODER(35, PreTokenizerPatterns.SEED_CODER, PreTokenizerPatterns.SEED_CODER_HF, true),
    HUNYUAN(36, PreTokenizerPatterns.QWEN2, PreTokenizerPatterns.QWEN2_HF, true),
    KIMI_K2(37, PreTokenizerPatterns.KIMI_K2, true),
    HUNYUAN_DENSE(38, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    GROK_2(39, PreTokenizerPatterns.QWEN2, PreTokenizerPatterns.QWEN2_HF, true),
    GRANITE_DOCLING(40, PreTokenizerPatterns.GPT2, true),
    MINIMAX_M2(41, PreTokenizerPatterns.GPT4O, PreTokenizerPatterns.GPT4O_HF, true),
    AFMOE(42, PreTokenizerPatterns.AFMOE, true),
    SOLAR_OPEN(43, PreTokenizerPatterns.QWEN2, PreTokenizerPatterns.QWEN2_HF, true),
    YOUTU(44, PreTokenizerPatterns.YOUTU, true),
    EXAONE_MOE(45, PreTokenizerPatterns.EXAONE_MOE, PreTokenizerPatterns.EXAONE_MOE_HF, true),
    QWEN35(46, PreTokenizerPatterns.QWEN35, PreTokenizerPatterns.QWEN35_HF, true),
    TINY_AYA(47, PreTokenizerPatterns.TINY_AYA, PreTokenizerPatterns.TINY_AYA_HF, true),
    JOYAI_LLM(48, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    JAIS2(49, PreTokenizerPatterns.JAIS2, PreTokenizerPatterns.JAIS2_HF, true),
    GEMMA4(50, PreTokenizerPatterns.NEWLINE_RUNS, false),
    SARVAM_MOE(51, PreTokenizerPatterns.NEWLINE_RUNS, false),
    MINICPM5(52, PreTokenizerPatterns.MINICPM5, PreTokenizerPatterns.MINICPM5_HF, true),
    WHITESPACE(53, PreTokenizerPatterns.WHITESPACE, false),
    GRANITE_EMB_MULTI(54, PreTokenizerPatterns.GPT4O, true),
    MELLUM2(55, PreTokenizerPatterns.STARCODER, true),
    LAGUNA(56, PreTokenizerPatterns.LAGUNA, true),
    HY_V4(57, PreTokenizerPatterns.DEEPSEEK3_LLM, true),
    SPARK2_5(58, PreTokenizerPatterns.SPARK2_5, true),
    UFAKZEKA(59, PreTokenizerPatterns.UFAKZEKA, true),
    MMBERT(60, PreTokenizerPatterns.NEWLINE_RUNS, false),
    K2_HORIZON(61, PreTokenizerPatterns.K2_HORIZON, true),
    FLINT(62, PreTokenizerPatterns.FLINT, true),
    DOMYN_SMALL(63, PreTokenizerPatterns.NEWLINE_RUNS, false),
    NEMOTRON_3_5_EMBD(64, PreTokenizerPatterns.TEKKEN, PreTokenizerPatterns.TEKKEN_HF, true),
    SOPRANO(65, PreTokenizerPatterns.SOPRANO, false),
    TEKKEN_NUM3(66, PreTokenizerPatterns.TEKKEN_NUM3, true);

    private final int code;
    private final List<String> regexes;
    private final List<String> hfRegexes;
    private final boolean byteEncode;

    PreTokenizerType(int code, List<String> regexes, boolean byteEncode) {
        this(code, regexes, Collections.emptyList(), byteEncode);
    }

    PreTokenizerType(int code, List<String> regexes, List<String> hfRegexes, boolean byteEncode) {
        this.code = code;
        this.regexes = regexes;
        this.hfRegexes = hfRegexes;
        this.byteEncode = byteEncode;
    }

    /** The {@code llama_vocab_pre_type} value. */
    public int code() { return code; }

    /** The ordered {@code regex_exprs}, as llama.cpp writes them. */
    public List<String> regexes() { return regexes; }

    /**
     * The same sequence as the Hugging Face tokenizer.json writes it, for the types whose
     * {@link #regexes()} llama.cpp adapted only in regex dialect (its "original regex from
     * tokenizer.json" comments); empty for every other type. These strings are
     * {@link PreTokenizerSpec.Dialect#ONIGURUMA} text.
     */
    public List<String> hfRegexes() { return hfRegexes; }

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

    /**
     * The text a raw fragment is split from, after whitespace escaping
     * ({@code llm_tokenizer_bpe_session::tokenize}, src/llama-vocab.cpp:653-680): NEMOTRON_3_5_EMBD
     * drops the U+200C after each {@code [} and {@code <}; SOPRANO collapses each whitespace run to
     * one space and lowercases every other code point, U+0130 becoming {@code i} followed by U+0307
     * as Unicode full lowercase maps it; every other type splits the text as it is.
     */
    public String normalizeFragment(String text) {
        if (this == NEMOTRON_3_5_EMBD) return text.replace("[‌", "[").replace("<‌", "<");
        if (this != SOPRANO) return text;
        StringBuilder sb = new StringBuilder(text.length());
        boolean space = false;
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (TokenizerVocab.isWhitespace(cp)) {
                if (!space) sb.append(' ');
                space = true;
            } else {
                if (cp == 0x0130) {
                    sb.append("i̇");
                } else {
                    sb.appendCodePoint(UnicodeFlags.toLower(cp));
                }
                space = false;
            }
        }
        return sb.toString();
    }

    /** A merged symbol that is not a token becomes the unknown token, not byte tokens (src/llama-vocab.cpp:800-801). */
    public boolean unknownSymbolsAsUnk() { return this == SOPRANO; }

    /** NORMAL tokens render their raw text, without byte or whitespace decoding (src/llama-vocab.cpp:4142-4144). */
    public boolean rawNormalPieces() { return this == SOPRANO; }

    /** The EOS token always ends generation (src/llama-vocab.cpp:3325-3327). */
    public boolean eosIsEog() { return this == SOPRANO; }
}
