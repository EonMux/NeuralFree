package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The BPE pre-tokenizer regex sequences of llama.cpp {@code llm_tokenizer_bpe}
 * (src/llama-vocab.cpp:280-594), transcribed character for character: each constant holds the
 * exact {@code regex_exprs} strings of one or more {@code LLAMA_VOCAB_PRE_TYPE_*} values, in order.
 *
 * <p>The strings keep llama.cpp's regex text because llama.cpp selects its hand-written splitters
 * by exact text, and because category escapes are rewritten for the collapsed text before
 * matching; {@link UnicodeRegexSplit} applies both rules. Non-ASCII literals are written as
 * Unicode escapes.</p>
 */
public final class PreTokenizerPatterns {

    private PreTokenizerPatterns() {}

    /** {@code FLINT}. */
    public static final List<String> FLINT = seq(
            "\\p{N}{1,3}",
            "[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:"
                + "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt"
                + "}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM"
                + "]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code LLAMA3}, {@code DBRX}, {@code SMAUG}, {@code CHATGLM4}. */
    public static final List<String> LLAMA3 = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                + "N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code JAIS2}. */
    public static final List<String> JAIS2 = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                + "N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128"
                + "}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?"
                + "!\\S)|\\s{1}");

    /** {@code DEEPSEEK_LLM}. */
    public static final List<String> DEEPSEEK_LLM = seq(
            "[\r\n]",
            "\\s?[A-Za-z\u00B5\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u01BA\u01BC-\u01BF\u01C4-\u0293\u0295-"
                + "\u02AF\u0370-\u0373\u0376\u0377\u037B-\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1"
                + "\u03A3-\u03F5\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u10A0-\u10C5\u13A0-\u13F5\u13F8-\u13FD"
                + "\u1C90-\u1CBA\u1CBD-\u1CBF\u1D00-\u1D2B\u1D6B-\u1D77\u1D79-\u1D9A\u1E00-\u1F15\u1F18-\u1F1D"
                + "\u1F20-\u1F45\u1F48-\u1F4D\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-"
                + "\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FF4"
                + "\u1FF6-\u1FFC\u2102\u2107\u210A-\u2113\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D"
                + "\u212F-\u2134\u2139\u213C-\u213F\u2145-\u2149\u214E\u2183\u2184\u2C00-\u2C7B\u2C7E-\u2CE4"
                + "\u2CEB-\u2CEE\u2CF2\u2CF3\uA640-\uA66D\uA680-\uA69B\uA722-\uA76F\uA771-\uA787\uA78B-\uA78E"
                + "\uAB70-\uABBF\uFB00-\uFB06\uFB13-\uFB17\uFF21-\uFF3A\uFF41-\uFF5A\uD801\uDC00-\uD801\uDC4F"
                + "\uD801\uDCB0-\uD801\uDCD3\uD801\uDCD8-\uD801\uDCFB\uD803\uDC80-\uD803\uDCB2\uD803\uDCC0-"
                + "\uD803\uDCF2\uD806\uDCA0-\uD806\uDCDF\uD83A\uDD00-\uD83A\uDD43]+",
            "\\s?[!-/:-~\uFF01-\uFF0F\uFF1A-\uFF5E\u2018-\u201F\u3000-\u3002]+",
            "\\s+$",
            "[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+",
            "\\p{N}+");

    /** {@code DEEPSEEK3_LLM}, {@code HUNYUAN_DENSE}, {@code JOYAI_LLM}, {@code HY_V4}. */
    public static final List<String> DEEPSEEK3_LLM = seq(
            "\\p{N}{1,3}",
            "[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M"
                + "}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+");

    /** {@code SPARK2_5}. */
    public static final List<String> SPARK2_5 = seq(
            "\\p{N}{1,3}",
            "[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M"
                + "}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
            "\\p{N}");

    /** {@code YOUTU}. */
    public static final List<String> YOUTU = seq(
            "[\uAC00-\uD7A3\u3131-\u318E]+|[\uFF01\u2026\u201C\u201D\u2018\u2019\u2014\uFF1A\uFF1B\uFF0C"
                + "\u3001-\u303F\uFE30-\uFE4F]+|[\u3105-\u312F]+|[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:"
                + "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt"
                + "}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM"
                + "]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code DEEPSEEK_CODER}. */
    public static final List<String> DEEPSEEK_CODER = seq(
            "[\r\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            "[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+",
            "\\p{N}");

    /** {@code FALCON}. */
    public static final List<String> FALCON = seq(
            "[\\p{P}\\$\\+<=>\\^~\\|`]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "[0-9][0-9][0-9]");

    /**
     * {@code STARCODER}, {@code REFACT}, {@code COMMAND_R}, {@code SMOLLM}, {@code CODESHELL},
     * {@code EXAONE}, {@code MINERVA}, {@code MELLUM2}.
     */
    public static final List<String> STARCODER = seq(
            "\\p{N}",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)");

    /** {@code GPT2}, {@code MPT}, {@code OLMO}, {@code JAIS}, {@code TRILLION}, {@code GRANITE_DOCLING}. */
    public static final List<String> GPT2 = seq(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)");

    /** {@code STABLELM2}, {@code QWEN2}, {@code HUNYUAN}, {@code SOLAR_OPEN}, {@code GROK_2}. */
    public static final List<String> QWEN2 = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                + "N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code QWEN35}. */
    public static final List<String> QWEN35 = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M"
                + "}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code PORO}, {@code BLOOM}, {@code GPT3_FINNISH}. */
    public static final List<String> PORO = seq(
            " ?[^(\\s|.,!?\u2026\u3002\uFF0C\u3001\u0964\u06D4\u060C)]+");

    /** {@code VIKING}. */
    public static final List<String> VIKING = seq(
            " ?[^(\\s|.,!?\u2026\u3002\uFF0C\u3001\u0964\u06D4\u060C)]+",
            "\\p{N}");

    /** {@code TEKKEN}. */
    public static final List<String> TEKKEN = seq(
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p"
                + "{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\"
                + "p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code CHAMELEON}. */
    public static final List<String> CHAMELEON = seq(
            "<sentinel:[0-9]+>",
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
            "([\\t\\n]|    |  )",
            "\\p{N}",
            "[\\p{P}!-/:-@\\[-`{-~]",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)");

    /** {@code GPT4O}, {@code MINIMAX_M2}, {@code GRANITE_EMB_MULTI}. */
    public static final List<String> GPT4O = seq(
            "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[t"
                + "T]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a"
                + "-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
                + "?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code TINY_AYA}. */
    public static final List<String> TINY_AYA = seq(
            "\\d{1,3}(?=(?:\\d{3})*\\b)",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:"
                + "'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt"
                + "}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM"
                + "]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\"
                + "s+");

    /** {@code KIMI_K2}: the trigger of llama.cpp's Kimi-K2 splitter. */
    public static final List<String> KIMI_K2 = seq(
            "\\p{Han}+");

    /** {@code SUPERBPE}. */
    public static final List<String> SUPERBPE = seq(
            "\\p{N}+",
            "(?=(\\d{3})+(?!\\d))");

    /** {@code BAILINGMOE}. */
    public static final List<String> BAILINGMOE = seq(
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L"
                + "}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+");

    /** {@code SEED_CODER}. */
    public static final List<String> SEED_CODER = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                + "N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code UFAKZEKA}. */
    public static final List<String> UFAKZEKA = seq(
            "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S"
                + ")|\\s+");

    /** {@code AFMOE}. */
    public static final List<String> AFMOE = seq(
            "\\p{AFMoE_digits}",
            "[\u4E00-\u9FFF\u3400-\u4DBF\u8C48-\uFAFF\u3040-\u309F\u30A0-\u30FF\uFF65-\uFF9F\u2F00-\u2FDF"
                + "\u0E40-\u0E7F\u0E80-\u0EFF\u1780-\u17FF\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF\uAC00-\uD7AF"
                + "\u1100-\u11FF]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p"
                + "{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code LAGUNA}. */
    public static final List<String> LAGUNA = seq(
            "[^\\n]+|[\\n]+",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                + "N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code EXAONE_MOE}. */
    public static final List<String> EXAONE_MOE = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p"
                + "{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\"
                + "s+");

    /** {@code GEMMA4}, {@code SARVAM_MOE}, {@code MMBERT}: newline and non-newline runs. */
    public static final List<String> NEWLINE_RUNS = seq(
            "[^\\n]+|[\\n]+");

    /** {@code MINICPM5}. */
    public static final List<String> MINICPM5 = seq(
            "\\p{N}{1,3}",
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                + "N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    /** {@code K2_HORIZON}. */
    public static final List<String> K2_HORIZON = seq(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\"
                + "p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|"
                + "\\s+");

    /** {@code WHITESPACE}. */
    public static final List<String> WHITESPACE = seq(
            "\\S+");

    /** {@code DEFAULT}, and the types without a case of their own (CHATGLM3, LLAMA4, PIXTRAL). */
    public static final List<String> DEFAULT = seq(
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "\\p{N}+",
            "[0-9][0-9][0-9]");

    private static List<String> seq(String... regexes) {
        return Collections.unmodifiableList(Arrays.asList(regexes));
    }
}
