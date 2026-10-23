package com.eonmux.neuralfree.gguf.arch;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Model architectures, one constant per llama.cpp {@code LLM_ARCH_NAMES} entry in the same order
 * (llama.cpp src/llama-arch.cpp:8-172), each carrying the string stored in
 * {@code general.architecture}.
 *
 * <p>The per-architecture behaviour lives in {@link ArchRegistry}; this enum holds only the name and
 * the intrinsic llama.cpp classifications ({@code llm_arch_is_recurrent}, {@code llm_arch_is_hybrid},
 * {@code llm_arch_is_diffusion}, llama-arch.cpp).
 */
public enum Arch {
    CLIP("clip"),
    LLAMA("llama"),
    LLAMA4("llama4"),
    DECI("deci"),
    FALCON("falcon"),
    GROK("grok"),
    GPT2("gpt2"),
    GPTJ("gptj"),
    GPTNEOX("gptneox"),
    MPT("mpt"),
    BAICHUAN("baichuan"),
    STARCODER("starcoder"),
    REFACT("refact"),
    BERT("bert"),
    MODERN_BERT("modern-bert"),
    NOMIC_BERT("nomic-bert"),
    NOMIC_BERT_MOE("nomic-bert-moe"),
    NEO_BERT("neo-bert"),
    JINA_BERT_V2("jina-bert-v2"),
    JINA_BERT_V3("jina-bert-v3"),
    EUROBERT("eurobert"),
    BLOOM("bloom"),
    STABLELM("stablelm"),
    QWEN("qwen"),
    QWEN2("qwen2"),
    QWEN2MOE("qwen2moe"),
    QWEN2VL("qwen2vl"),
    SENSENOVA_U1("sensenova_u1"),
    QWEN3("qwen3"),
    FLINT("flint"),
    QWEN3MOE("qwen3moe"),
    QWEN3NEXT("qwen3next"),
    QWEN3VL("qwen3vl"),
    QWEN3VLMOE("qwen3vlmoe"),
    QWEN35("qwen35"),
    QWEN35MOE("qwen35moe"),
    CLEF("clef"),
    VLLM_DECISION2("vllm-decision2"),
    QWEN4EXP("qwen4exp"),
    PHI2("phi2"),
    PHI3("phi3"),
    PHIMOE("phimoe"),
    PLAMO("plamo"),
    PLAMO2("plamo2"),
    PLAMO3("plamo3"),
    CODESHELL("codeshell"),
    ORION("orion"),
    INTERNLM2("internlm2"),
    MINICPM("minicpm"),
    MINICPM3("minicpm3"),
    GEMMA("gemma"),
    GEMMA2("gemma2"),
    GEMMA3("gemma3"),
    GEMMA3N("gemma3n"),
    GEMMA4("gemma4"),
    GEMMA4_ASSISTANT("gemma4-assistant"),
    GEMMA_EMBEDDING("gemma-embedding"),
    GEMMA_EMBEDDING2("gemma-embedding2"),
    STARCODER2("starcoder2"),
    MAMBA("mamba"),
    MAMBA2("mamba2"),
    MAPLE("maple"),
    JAMBA("jamba"),
    FALCON_H1("falcon-h1"),
    XVERSE("xverse"),
    COMMAND_R("command-r"),
    COHERE2("cohere2"),
    COHERE2MOE("cohere2moe"),
    DBRX("dbrx"),
    OLMO("olmo"),
    OLMO2("olmo2"),
    OLMOE("olmoe"),
    MUSE_GLIMMER("muse-glimmer"),
    MUSE_SPARK("muse-spark"),
    OPENELM("openelm"),
    ARCTIC("arctic"),
    DEEPSEEK("deepseek"),
    DEEPSEEK2("deepseek2"),
    DEEPSEEK2OCR("deepseek2-ocr"),
    DEEPSEEK32("deepseek32"),
    DEEPSEEK4("deepseek4"),
    CHATGLM("chatglm"),
    GLM4("glm4"),
    GLM4_MOE("glm4moe"),
    GLM_DSA("glm-dsa"),
    BITNET("bitnet"),
    T5("t5"),
    T5ENCODER("t5encoder"),
    JAIS("jais"),
    JAIS2("jais2"),
    NEMOTRON("nemotron"),
    NEMOTRON_H("nemotron_h"),
    NEMOTRON_H_MOE("nemotron_h_moe"),
    EXAONE("exaone"),
    EXAONE4("exaone4"),
    EXAONE_MOE("exaone-moe"),
    RWKV6("rwkv6"),
    RWKV6QWEN2("rwkv6qwen2"),
    RWKV7("rwkv7"),
    ARWKV7("arwkv7"),
    GRANITE("granite"),
    GRANITE_MOE("granitemoe"),
    GRANITE_HYBRID("granitehybrid"),
    GRANITE_SWITCH("graniteswitch"),
    GRANITE_SWA("granite_swa"),
    CHAMELEON("chameleon"),
    WAVTOKENIZER_DEC("wavtokenizer-dec"),
    PLM("plm"),
    BAILINGMOE("bailingmoe"),
    BAILINGMOE2("bailingmoe2"),
    BAILINGMOE3("bailingmoe3"),
    DOTS1("dots1"),
    DOTS3NOTE("dots3note"),
    ARCEE("arcee"),
    AFMOE("afmoe"),
    LAGUNA("laguna"),
    ERNIE4_5("ernie4_5"),
    ERNIE4_5_MOE("ernie4_5-moe"),
    HUNYUAN_MOE("hunyuan-moe"),
    HUNYUAN_DENSE("hunyuan-dense"),
    HUNYUAN_VL("hunyuan_vl"),
    HY_V3("hy_v3"),
    HY_V4("hy_v4"),
    SMOLLM3("smollm3"),
    OPENAI_MOE("gpt-oss"),
    LFM2("lfm2"),
    LFM2MOE("lfm2moe"),
    DREAM("dream"),
    SMALLTHINKER("smallthinker"),
    LLADA("llada"),
    LLADA_MOE("llada-moe"),
    SEED_OSS("seed_oss"),
    GROVEMOE("grovemoe"),
    APERTUS("apertus"),
    MINIMAX_01("minimax-01"),
    HRM_TEXT("hrm_text"),
    MINIMAX_M2("minimax-m2"),
    MINIMAX_M3("minimax-m3"),
    COGVLM("cogvlm"),
    RND1("rnd1"),
    PANGU_EMBED("pangu-embedded"),
    MISTRAL3("mistral3"),
    EAGLE3("eagle3"),
    DFLASH("dflash"),
    MISTRAL4("mistral4"),
    PADDLEOCR("paddleocr"),
    MIMO2("mimo2"),
    STEP35("step35"),
    SPARK2_5("spark2_5"),
    LLAMA_EMBED("llama-embed"),
    MAINCODER("maincoder"),
    MAION_CODER("maion-coder"),
    KIMI_LINEAR("kimi-linear"),
    KIMI_K3("kimi-k3"),
    GLM5_NEXT("glm5-next"),
    TALKIE("talkie"),
    MELLUM("mellum"),
    NANBEIGE("nanbeige"),
    QWEN3TTS("qwen3tts"),
    POCKETTTS("pockettts"),
    K2_HORIZON("k2-horizon"),
    BEAM("beam"),
    KOLIBRI1("kolibri1"),
    UNKNOWN("(unknown)");

    private static final Map<String, Arch> BY_GGUF_NAME;

    static {
        Map<String, Arch> m = new HashMap<>();
        for (Arch a : values()) {
            m.put(a.ggufName, a);
        }
        BY_GGUF_NAME = Collections.unmodifiableMap(m);
    }

    private final String ggufName;

    Arch(String ggufName) {
        this.ggufName = ggufName;
    }

    /** The {@code general.architecture} string, also the prefix of arch-scoped metadata keys. */
    public String ggufName() {
        return ggufName;
    }

    /**
     * Exact-match lookup of a {@code general.architecture} value (llama.cpp {@code llm_arch_from_string}).
     *
     * @return the architecture, or {@link #UNKNOWN} when the name is null or not a llama.cpp architecture
     */
    public static Arch fromGgufName(String name) {
        if (name == null) {
            return UNKNOWN;
        }
        Arch a = BY_GGUF_NAME.get(name);
        return a != null ? a : UNKNOWN;
    }

    /** Every layer is recurrent (llama.cpp {@code llm_arch_is_recurrent}). */
    public boolean isRecurrent() {
        switch (this) {
            case MAMBA:
            case MAMBA2:
            case RWKV6:
            case RWKV6QWEN2:
            case RWKV7:
            case ARWKV7:
                return true;
            default:
                return false;
        }
    }

    /** Mixes attention and recurrent layers (llama.cpp {@code llm_arch_is_hybrid}). */
    public boolean isHybrid() {
        switch (this) {
            case JAMBA:
            case FALCON_H1:
            case PLAMO2:
            case GRANITE_HYBRID:
            case LFM2:
            case LFM2MOE:
            case NEMOTRON_H:
            case NEMOTRON_H_MOE:
            case QWEN3NEXT:
            case KIMI_LINEAR:
            case BAILINGMOE3:
            case KIMI_K3:
            case GLM5_NEXT:
            case QWEN35:
            case QWEN35MOE:
            case QWEN4EXP:
            case DEEPSEEK4:
            case MINIMAX_01:
                return true;
            default:
                return false;
        }
    }

    /** Masked-diffusion language model (llama.cpp {@code llm_arch_is_diffusion}). */
    public boolean isDiffusion() {
        switch (this) {
            case DREAM:
            case LLADA:
            case LLADA_MOE:
            case RND1:
                return true;
            default:
                return false;
        }
    }
}
