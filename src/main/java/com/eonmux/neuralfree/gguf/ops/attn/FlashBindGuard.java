package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;

/**
 * The binding-layer guard of the flash-attention shapes: an attention geometry that no GPU flash kernel serves
 * ({@link FlashAttnTiling#geometryRefusal} — head counts and dims the kernels refuse for the shape's own sake, or a
 * decode tile the device's shared budget cannot stage) is a device limitation, so the mixers throw a
 * {@link GpuModelUnsupportedException} naming the layer and the dims while their blocks bind, and the model factory
 * falls back to the CPU engine with that reason. The factories' own argument checks stay in {@link FlashAttnTiling}
 * and keep throwing {@code IllegalArgumentException} there: a failure that is a caller's bug (a non-positive key
 * range or shared budget) is an engine defect and fails the load, as does every other error the planning raises.
 * Every bind site that plans flash attention ({@link GpuAttentionMixer}, {@code GpuMlaMixer},
 * {@code GpuCrossAttentionMixer}, the drafter cache attention) calls {@link #requireServable} before it plans.
 */
public final class FlashBindGuard {

    private FlashBindGuard() {
    }

    /**
     * Refuses an attention geometry the GPU flash kernels cannot serve, naming {@code where} (the layer or stage
     * that binds) and the offending dims; a servable geometry returns normally.
     *
     * @param kernels the device's LLM kernels, for the shared-memory budget the decode tiles must fit
     * @throws GpuModelUnsupportedException when {@link FlashAttnTiling#geometryRefusal} names the shape
     */
    public static void requireServable(LlmKernels kernels, String where, int nHead, int nHeadKv, int dK, int dV) {
        String refusal = FlashAttnTiling.geometryRefusal(nHead, nHeadKv, dK, dV, kernels.faSmemFloats());
        if (refusal != null) {
            throw new GpuModelUnsupportedException(where + ": " + refusal);
        }
    }
}
