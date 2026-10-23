package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;

/**
 * The deepseek2 graph's YaRN-corrected attention scale, shared by its MLA layers and deepseek2-ocr's plain
 * multi-head attention (models/deepseek2.cpp:427-437, 484-486).
 */
final class DeepseekScale {

    private DeepseekScale() {
    }

    /**
     * {@code mscale^2 / sqrt(headK)} with the DeepSeek mscale of the context's YaRN attention factor
     * (deepseek2.cpp:427-437, llama-context.cpp:257-306). The context frequency scale is 1 for scaling type NONE
     * and {@code yarn_ext_factor} defaults by scaling type when unset; both formulas come from {@link RopeMath}.
     * {@code attention.scale} plays no part.
     *
     * @param headK {@code n_embd_head_k_mla()}: the decompressed MLA key width, or the plain head width without MLA
     *              (llama-hparams.cpp:313-315)
     */
    static float yarnKqScale(ModelHparams hp, int headK) {
        if (headK <= 0) {
            throw new IllegalStateException(hp.arch().ggufName() + ": kq_scale needs a positive key width, got "
                    + headK);
        }
        RopeScalingType scaling = hp.ropeScalingType();
        float freqScale = scaling == RopeScalingType.NONE ? 1.0f : hp.ropeFreqScale();
        float extFactor = hp.yarnExtFactor() >= 0f ? hp.yarnExtFactor() : RopeMath.defaultExtFactor(scaling);
        float attnFactor = RopeMath.yarnAttnFactor(hp.yarnAttnFactor(), extFactor, freqScale, hp.yarnLogMul(),
                hp.arch() == Arch.DEEPSEEK2, hp.ropeAttnFactor());
        float mscale = RopeMath.deepseekKqMscale(attnFactor, freqScale, hp.yarnLogMul());
        return mscale * mscale / (float) Math.sqrt(headK);
    }
}
