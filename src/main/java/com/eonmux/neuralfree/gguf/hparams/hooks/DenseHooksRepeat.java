package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams hooks of the repeat-block architectures, whose trunk runs a contiguous block {@code [n_input, n_input +
 * n_recurrent)} a configured number of times: Dory and IQuestLoopCoder. The repeat geometry comes from the
 * {@code recurrent_block.*} keys of the conversion, and {@code mlp_layer_pattern} marks the blocks that carry an MLP
 * without attention.
 */
public final class DenseHooksRepeat {

    private DenseHooksRepeat() {
    }

    /**
     * Dory ({@code dory.py}): the per-layer attention type (sliding, full or MLP-only) with its shared sliding window,
     * the RMS epsilon of every justnorm and norm, the repeat geometry, the slerp alpha scale and the always-on
     * justnorm stream combination.
     */
    public static void dory(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        readRepeatBlock(b, kv);
        b.repeatJustNorm(requiredBool(kv, GgufKeys.RECURRENT_BLOCK_JUSTNORM));
        b.repeatJustNormEps(kv.f32(GgufKeys.RECURRENT_BLOCK_JUSTNORM_EPS, 1e-6f));
        if (!(kv.f32Required(GgufKeys.ATTENTION_SCALE) > 0f)) {
            throw new GgufFormatException("dory: '" + kv.key(GgufKeys.ATTENTION_SCALE) + "' must be positive");
        }
        b.attentionScale(kv.f32Required(GgufKeys.ATTENTION_SCALE));
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBase());
        b.ropeFreqScaleSwa(b.ropeFreqScale());
        float slerp = kv.f32Required(GgufKeys.SLERP_ALPHA_SCALE);
        if (!(slerp > 0f) || Float.isInfinite(slerp)) {
            throw new GgufFormatException("dory: '" + kv.key(GgufKeys.SLERP_ALPHA_SCALE)
                    + "' must be a positive finite number, got " + slerp);
        }
        b.slerpAlphaScale(slerp);
        boolean[] swa = requiredFlags(b, kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN);
        boolean[] mlp = requiredFlags(b, kv, GgufKeys.MLP_LAYER_PATTERN);
        for (int il = 0; il < b.nLayer(); il++) {
            if (swa[il] && mlp[il]) {
                throw new GgufFormatException("dory: layer " + il + " is both a sliding-window attention block and an"
                        + " MLP-only block");
            }
        }
        boolean anySwa = any(swa);
        if (anySwa) {
            b.swaType(SwaType.STANDARD);
            b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));
            if (b.nSwa() <= 0) {
                throw new GgufFormatException("dory: '" + kv.key(GgufKeys.ATTENTION_SLIDING_WINDOW)
                        + "' must be positive for a sliding-window block");
            }
        } else {
            b.swaType(SwaType.NONE);
        }
        b.isSwaArray(swa);
        b.isMlpArray(mlp);
    }

    /**
     * IQuestLoopCoder ({@code iquest_loopcoder.py}): a Llama block repeated {@code loop_num} times, where every pass
     * after the first gates a read-only attention over the first pass's KV against a sliding-window attention over
     * its own per-pass cache. The stream is carried unchanged between passes ({@code LoopCombine.NONE}).
     */
    public static void iquestLoopCoder(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        readRepeatBlock(b, kv);
        if (kv.has(GgufKeys.RECURRENT_BLOCK_JUSTNORM) && kv.bool(GgufKeys.RECURRENT_BLOCK_JUSTNORM, false)) {
            throw new GgufFormatException("iquest-loopcoder: '" + kv.key(GgufKeys.RECURRENT_BLOCK_JUSTNORM)
                    + "' must be false; the graph carries the stream between passes unchanged");
        }
        b.repeatJustNorm(false);
        if (!kv.bool(GgufKeys.RECURRENT_BLOCK_DUAL_ATTENTION, false)) {
            throw new GgufFormatException("iquest-loopcoder: '" + kv.key(GgufKeys.RECURRENT_BLOCK_DUAL_ATTENTION)
                    + "' must be true; the gated attention of the later passes is the architecture's graph");
        }
        int window = kv.u32Required(GgufKeys.RECURRENT_BLOCK_LOCAL_WINDOW);
        if (window <= 0) {
            throw new GgufFormatException("iquest-loopcoder: '" + kv.key(GgufKeys.RECURRENT_BLOCK_LOCAL_WINDOW)
                    + "' must be positive, got " + window);
        }
        float scale = kv.f32Required(GgufKeys.ATTENTION_SCALE);
        if (!(scale > 0f) || Float.isInfinite(scale)) {
            throw new GgufFormatException("iquest-loopcoder: '" + kv.key(GgufKeys.ATTENTION_SCALE)
                    + "' must be positive and finite, got " + scale);
        }
        b.attentionScale(scale);
        b.repeatDualAttention(true);
        b.repeatLocalWindow(window);
        b.swaType(SwaType.NONE);
        b.isSwaArray(new boolean[b.nLayerAll()]);
        b.isMlpArray(new boolean[b.nLayerAll()]);
    }

    /**
     * Reads the repeat geometry, checking that the three ranges cover the physical trunk.
     *
     * @throws GgufFormatException when a count is negative, the passes are fewer than one, the ranges do not cover the
     *                             trunk, or a recurrent range is empty
     */
    static void readRepeatBlock(ModelHparamsBuilder b, GgufKv kv) {
        int loops = kv.u32Required(GgufKeys.RECURRENT_BLOCK_LOOPS);
        int input = kv.u32Required(GgufKeys.RECURRENT_BLOCK_INPUT);
        int recurrent = kv.u32Required(GgufKeys.RECURRENT_BLOCK_RECURRENT);
        int output = kv.u32Required(GgufKeys.RECURRENT_BLOCK_OUTPUT);
        if (loops < 1) {
            throw new GgufFormatException("'" + kv.key(GgufKeys.RECURRENT_BLOCK_LOOPS) + "' must be at least 1, got "
                    + loops);
        }
        if (recurrent < 1) {
            throw new GgufFormatException("'" + kv.key(GgufKeys.RECURRENT_BLOCK_RECURRENT) + "' must be positive, got "
                    + recurrent);
        }
        if ((long) input + recurrent + output != b.nLayer()) {
            throw new GgufFormatException("the repeat ranges " + input + " + " + recurrent + " + " + output
                    + " do not cover the " + b.nLayer() + " physical layers");
        }
        long slots = (long) input + (long) recurrent * loops + output;
        if (slots > ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException("the repeat schedule has " + slots + " slots, more than the maximum of "
                    + ModelHparams.MAX_LAYERS);
        }
        b.repeatLoops(loops);
        b.repeatInputLayers(input);
        b.repeatRecurrentLayers(recurrent);
        b.repeatOutputLayers(output);
    }

    /** A required per-layer flag array padded with false to the trunk depth. */
    private static boolean[] requiredFlags(ModelHparamsBuilder b, GgufKv kv, String keyFmt) {
        boolean[] flags = kv.boolArray(keyFmt);
        if (flags == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(keyFmt) + "' is missing");
        }
        if (flags.length > ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException("metadata key '" + kv.key(keyFmt) + "' has " + flags.length
                    + " entries, more than the maximum of " + ModelHparams.MAX_LAYERS + " layers");
        }
        if (flags.length < b.nLayer()) {
            throw new GgufFormatException("metadata key '" + kv.key(keyFmt) + "' has " + flags.length
                    + " entries, fewer than the " + b.nLayer() + " layers");
        }
        boolean[] out = new boolean[b.nLayerAll()];
        System.arraycopy(flags, 0, out, 0, b.nLayer());
        return out;
    }

    private static boolean requiredBool(GgufKv kv, String keyFmt) {
        if (!kv.has(keyFmt)) {
            throw new GgufFormatException("required metadata key '" + kv.key(keyFmt) + "' is missing");
        }
        return kv.bool(keyFmt, false);
    }

    private static boolean any(boolean[] flags) {
        for (boolean f : flags) {
            if (f) {
                return true;
            }
        }
        return false;
    }
}
