package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams of Inkling (vLLM's {@code InklingForCausalLM}): a NoPE decoder whose local layers attend a
 * {@code STANDARD} sliding window, whose attention adds per-query relative logits and whose K/V streams pass through
 * short convolutions. The metadata carries the model-level relative table ({@code inkling.relative_extent},
 * {@code inkling.relative_dim}), the convolution width ({@code inkling.sconv_kernel_size}), the optional per-token
 * query log scaling ({@code inkling.log_scaling_n_floor}, {@code inkling.log_scaling_alpha}) and the usual
 * per-layer head counts and head dims, with {@code attention.key_length_swa} for the local layers.
 */
public final class InklingHooks {

    private InklingHooks() {
    }

    /**
     * Reads the Inkling keys into {@code b}.
     *
     * @throws GgufFormatException when a geometry is missing or outside the range the graph can carry
     */
    public static void apply(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        int relExtent = kv.u32Required(GgufKeys.INKLING_RELATIVE_EXTENT);
        if (relExtent <= 0) {
            throw invalid(kv, GgufKeys.INKLING_RELATIVE_EXTENT, "must be positive, got " + relExtent);
        }
        b.inklingRelExtent(relExtent);
        int dRel = kv.u32Required(GgufKeys.INKLING_RELATIVE_DIM);
        if (dRel <= 0) {
            throw invalid(kv, GgufKeys.INKLING_RELATIVE_DIM, "must be positive, got " + dRel);
        }
        b.inklingRelDim(dRel);
        int convWidth = kv.u32Required(GgufKeys.INKLING_SCONV_KERNEL);
        if (convWidth < 2) {
            throw invalid(kv, GgufKeys.INKLING_SCONV_KERNEL, "must be at least 2, got " + convWidth);
        }
        b.inklingSconvKernel(convWidth);
        int floor = kv.u32(GgufKeys.INKLING_LOG_SCALING_FLOOR, 0);
        if (floor < 0) {
            throw invalid(kv, GgufKeys.INKLING_LOG_SCALING_FLOOR, "must be non-negative, got " + floor);
        }
        b.inklingLogScalingFloor(floor);
        float alpha = kv.f32(GgufKeys.INKLING_LOG_SCALING_ALPHA, 0.1f);
        if (Float.isNaN(alpha) || Float.isInfinite(alpha)) {
            throw invalid(kv, GgufKeys.INKLING_LOG_SCALING_ALPHA, "must be finite, got " + alpha);
        }
        b.inklingLogScalingAlpha(alpha);
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, 0.0f));

        boolean[] local = localLayers(b, kv);
        b.isSwaArray(local);
        int window = kv.u32(GgufKeys.ATTENTION_SLIDING_WINDOW, 0);
        boolean anyLocal = false;
        for (boolean flag : local) {
            anyLocal |= flag;
        }
        if (anyLocal) {
            if (window < 1) {
                throw invalid(kv, GgufKeys.ATTENTION_SLIDING_WINDOW, "local layers need a positive window, got "
                        + window);
            }
            b.swaType(SwaType.STANDARD);
            b.nSwa(window);
        } else {
            b.swaType(SwaType.NONE);
            b.nSwa(0);
        }
        checkHeadDims(b, kv);
        // The muP width multiplier divides the final logits (vLLM InklingLogitsProcessor); 0 when absent.
        b.logitScale(kv.f32(GgufKeys.LOGIT_SCALE, 0.0f));
    }

    /**
     * The local-layer flags: {@code attention.sliding_window_pattern}, required here because a model without local
     * layers still states the array (every entry false).
     */
    private static boolean[] localLayers(ModelHparamsBuilder b, GgufKv kv) {
        boolean[] flags = kv.boolArray(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN);
        if (flags == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN)
                    + "' is missing");
        }
        boolean[] local = new boolean[b.nLayerAll()];
        System.arraycopy(flags, 0, local, 0, Math.min(flags.length, local.length));
        for (int il = b.nLayer(); il < local.length; il++) {
            local[il] = false;
        }
        return local;
    }

    /**
     * Head geometry of every layer: the query and KV head counts must exist, and a local layer's head dim
     * ({@code attention.key_length_swa}) must match its value head dim.
     */
    private static void checkHeadDims(ModelHparamsBuilder b, GgufKv kv) {
        for (int il = 0; il < b.nLayerAll(); il++) {
            int nHead = b.nHead(il);
            int nHeadKv = b.nHeadKv(il);
            if (nHead <= 0 || nHeadKv <= 0) {
                throw new GgufFormatException(kv.key(GgufKeys.ATTENTION_HEAD_COUNT) + ": layer " + il + " has "
                        + nHead + " query heads and " + nHeadKv + " KV heads");
            }
            if (nHead % nHeadKv != 0) {
                throw new GgufFormatException(kv.key(GgufKeys.ATTENTION_HEAD_COUNT) + ": layer " + il + " has "
                        + nHead + " query heads, not a multiple of its " + nHeadKv + " KV heads");
            }
            int headK = b.isSwa(il) ? b.headKSwa() : b.headKFull();
            int headV = b.isSwa(il) ? b.headVSwa() : b.headVFull();
            if (headK <= 0 || headV != headK) {
                throw new GgufFormatException(kv.key(GgufKeys.ATTENTION_KEY_LENGTH) + ": layer " + il + " has head"
                        + " dims " + headK + "/" + headV + "; the per-head K norm needs one width");
            }
        }
    }

    private static GgufFormatException invalid(GgufKv kv, String key, String message) {
        return new GgufFormatException("metadata key '" + kv.key(key) + "': " + message);
    }
}
