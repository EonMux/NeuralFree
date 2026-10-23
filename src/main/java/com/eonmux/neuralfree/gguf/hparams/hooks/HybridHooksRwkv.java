package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams hooks of the RWKV architectures: rwkv6, rwkv6qwen2, rwkv7 and arwkv7. Every layer is recurrent
 * ({@link com.eonmux.neuralfree.gguf.arch.Arch#isRecurrent()}). The token-shift state holds
 * {@code token_shift_count} rows of {@code n_embd}: two (time mix and channel mix) for the RWKV channel-mix
 * FFN graphs, one for the Qwen-FFN graphs, which their graphs assert (rwkv6.cpp:96, rwkv7.cpp:126,
 * rwkv6qwen2.cpp:85, arwkv7.cpp:121).
 *
 * <p>Whether an rwkv6qwen2 layer is a QRWKV layer is a tensor fact ({@code time_mix_first} absent,
 * rwkv6-base.cpp:50), resolved into {@code RecurrentSpec.qrwkv()} by {@code RecurrentResolver}.
 */
public final class HybridHooksRwkv {

    private HybridHooksRwkv() {
    }

    /** rwkv6.cpp:3-10. */
    public static void rwkv6(ModelHparamsBuilder b, GgufKv kv) {
        readRwkv6(b, kv);
        requireTokenShift(b, 2, "rwkv6.cpp:96");
    }

    /** rwkv6qwen2.cpp:3-10. */
    public static void rwkv6qwen2(ModelHparamsBuilder b, GgufKv kv) {
        readRwkv6(b, kv);
        requireTokenShift(b, 1, "rwkv6qwen2.cpp:85");
    }

    /** rwkv7.cpp:3-11. */
    public static void rwkv7(ModelHparamsBuilder b, GgufKv kv) {
        readRwkv7(b, kv);
        requireTokenShift(b, 2, "rwkv7.cpp:126");
    }

    /** arwkv7.cpp:3-11. */
    public static void arwkv7(ModelHparamsBuilder b, GgufKv kv) {
        readRwkv7(b, kv);
        requireTokenShift(b, 1, "arwkv7.cpp:121");
    }

    /** rwkv6.cpp:4-10 and rwkv6qwen2.cpp:4-10 (identical). */
    private static void readRwkv6(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_EPS, b.normEps()));              // rwkv6.cpp:4
        b.normRmsEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, b.normRmsEps()));    // rwkv6.cpp:5
        b.wkvHeadSize(kv.u32Required(GgufKeys.WKV_HEAD_SIZE));                         // rwkv6.cpp:6
        b.timeMixExtraDim(kv.u32Required(GgufKeys.TIME_MIX_EXTRA_DIM));                // rwkv6.cpp:7
        b.timeDecayExtraDim(kv.u32Required(GgufKeys.TIME_DECAY_EXTRA_DIM));            // rwkv6.cpp:8
        b.rescaleEveryN(kv.u32(GgufKeys.RESCALE_EVERY_N_LAYERS, b.rescaleEveryN()));   // rwkv6.cpp:9
        b.tokenShiftCount(kv.u32(GgufKeys.TOKEN_SHIFT_COUNT, b.tokenShiftCount()));    // rwkv6.cpp:10
        requireWkvHeads(b, kv);
        requireGroupedKvHeads(b);
    }

    /** rwkv7.cpp:4-11 and arwkv7.cpp:4-11 (identical). */
    private static void readRwkv7(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_EPS, b.normEps()));              // rwkv7.cpp:4
        b.normRmsEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, b.normRmsEps()));    // rwkv7.cpp:5
        b.wkvHeadSize(kv.u32Required(GgufKeys.WKV_HEAD_SIZE));                         // rwkv7.cpp:6
        b.nLoraDecay(kv.u32Required(GgufKeys.ATTENTION_DECAY_LORA_RANK));              // rwkv7.cpp:7
        b.nLoraIclr(kv.u32Required(GgufKeys.ATTENTION_ICLR_LORA_RANK));                // rwkv7.cpp:8
        b.nLoraValueResMix(kv.u32Required(GgufKeys.ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK));  // rwkv7.cpp:9
        b.nLoraGate(kv.u32(GgufKeys.ATTENTION_GATE_LORA_RANK, b.nLoraGate()));         // rwkv7.cpp:10
        b.tokenShiftCount(kv.u32(GgufKeys.TOKEN_SHIFT_COUNT, b.tokenShiftCount()));    // rwkv7.cpp:11
        requireWkvHeads(b, kv);
    }

    /** The WKV mixers split {@code n_embd} into {@code n_embd / wkv_head_size} heads (rwkv6-base.cpp:42-43). */
    private static void requireWkvHeads(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.requireNonZero(kv, GgufKeys.WKV_HEAD_SIZE, b.wkvHeadSize());
        if (b.nEmbd() % b.wkvHeadSize() != 0) {
            throw HybridHooks.invalid(b, "embedding length " + b.nEmbd() + " is not a multiple of wkv.head_size "
                    + b.wkvHeadSize());
        }
    }

    /**
     * RWKV6 layers with grouped K/V heads repeat each K/V head over {@code n_head / n_head_kv} WKV heads,
     * which must divide evenly (rwkv6-base.cpp:110-111).
     */
    private static void requireGroupedKvHeads(ModelHparamsBuilder b) {
        int nHead = b.nEmbd() / b.wkvHeadSize();
        for (int il = 0; il < b.nLayer(); il++) {
            int nHeadKv = b.nHeadKv(il);
            if (nHeadKv != 0 && nHeadKv != nHead && nHead % nHeadKv != 0) {
                throw HybridHooks.invalid(b, "layer " + il + " has " + nHeadKv + " KV heads, which do not divide the "
                        + nHead + " WKV heads (rwkv6-base.cpp:111)");
            }
        }
    }

    private static void requireTokenShift(ModelHparamsBuilder b, int expected, String where) {
        if (b.tokenShiftCount() != expected) {
            throw HybridHooks.invalid(b, "token_shift_count must be " + expected + ", got " + b.tokenShiftCount()
                    + " (" + where + ")");
        }
    }
}
