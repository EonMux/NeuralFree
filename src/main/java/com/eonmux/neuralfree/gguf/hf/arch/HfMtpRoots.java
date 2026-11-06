package com.eonmux.neuralfree.gguf.hf.arch;

/**
 * The shared root tensors the converters' {@code mtp_only} filters keep beside the NextN / MTP blocks: the token
 * embedding, the final norm and the output head the draft reads from its target. The names are the filtered Hugging
 * Face names (after the class filters and the inherited renames), one set per converter family.
 */
final class HfMtpRoots {

    /** DeepSeek-V2-deriving classes (deepseek.py, glm.py, dots3.py, mellum.py, mimo.py, step3.py, hunyuan.py). */
    static final String[] COMMON = {
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"};

    /** Qwen's MTP mixin keeps both the {@code model.}-prefixed and the flat names (qwen.py:325-357). */
    static final String[] QWEN = {
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
            "embed_tokens.weight", "norm.weight"};

    /** {@code BailingMoeV3Model} names its embedding {@code model.word_embeddings.weight} (bailingmoe3.py:126-129). */
    static final String[] BAILING = {
            "model.word_embeddings.weight", "model.norm.weight", "lm_head.weight"};

    /** {@code DeepseekV4Model} uses the native root names (deepseek.py:616-620). */
    static final String[] DEEPSEEK_V4 = {
            "embed.weight", "norm.weight", "head.weight", "head.scale"};

    /** {@code NemotronHModel}'s shared roots, with the NVFP4 scale sidecars of its head (nemotron.py:522-540). */
    static final String[] NEMOTRON = {
            "backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight",
            "lm_head.weight_scale", "lm_head.weight_scale_2", "lm_head.weight_scale_inv",
            "lm_head.input_scale", "lm_head.input_global_scale", "lm_head.weight_global_scale",
            "lm_head.weight_packed"};

    private HfMtpRoots() {
    }
}
