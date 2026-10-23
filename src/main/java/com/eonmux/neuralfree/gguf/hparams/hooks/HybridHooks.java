package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Arrays;

/**
 * Shared readers for the recurrent and hybrid architecture hooks ({@link HybridHooksSsm},
 * {@link HybridHooksRwkv}, {@link HybridHooksLfm}, {@link HybridHooksGdn}, {@link HybridHooksKda},
 * {@link HybridHooksLightning}), referenced from {@link com.eonmux.neuralfree.gguf.arch.ArchTableHybrid}.
 * Each helper transcribes a pattern that several llama.cpp {@code load_arch_hparams} bodies repeat; the
 * hooks cite the lines they reproduce.
 */
public final class HybridHooks {

    private HybridHooks() {
    }

    /**
     * Reads the required SSM keys {@code ssm.conv_kernel}, {@code ssm.inner_size}, {@code ssm.state_size} and
     * {@code ssm.time_step_rank}, plus {@code ssm.group_count} when {@code withGroups}
     * (mamba.cpp:4-7, mamba2.cpp:4-8).
     *
     * @throws GgufFormatException when a key is missing
     */
    static void readSsm(ModelHparamsBuilder b, GgufKv kv, boolean withGroups) {
        b.ssmDConv(kv.u32Required(GgufKeys.SSM_CONV_KERNEL));
        b.ssmDInner(kv.u32Required(GgufKeys.SSM_INNER_SIZE));
        b.ssmDState(kv.u32Required(GgufKeys.SSM_STATE_SIZE));
        b.ssmDtRank(kv.u32Required(GgufKeys.SSM_TIME_STEP_RANK));
        if (withGroups) {
            b.ssmNGroup(kv.u32Required(GgufKeys.SSM_GROUP_COUNT));
        }
    }

    /** Reads the required {@code attention.layer_norm_rms_epsilon}. */
    static void readRmsEps(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));
    }

    /**
     * Marks trunk layers without KV heads as recurrent: {@code is_recr_impl[i] = n_head_kv(i) == 0} for
     * {@code i < n_layer()} (jamba.cpp:12, plamo2.cpp:19, granite-hybrid.cpp:24, lfm2.cpp:15, lfm2moe.cpp:13,
     * kimi-linear.cpp:18, kimi-k3.cpp:32, bailingmoe3.cpp:37). NextN blocks keep their default.
     */
    static void markRecurrentByKvHeads(ModelHparamsBuilder b) {
        for (int il = 0; il < b.nLayer(); il++) {
            b.isRecurrent(il, b.nHeadKv(il) == 0);
        }
    }

    /**
     * Recurrent layers from {@code attention.recurrent_layers} (per-layer bool array or a broadcast scalar of
     * {@code n_layer_all} entries), else every trunk layer except each {@code full_attention_interval}-th is
     * recurrent: {@code is_recr[i] = i < n_layer() && (i + 1) % interval != 0} (qwen3next.cpp:17-23,
     * qwen35.cpp:17-23, qwen35moe.cpp:20-26, qwen4exp.cpp:149-157, minimax-01.cpp:12-18). The interval
     * applied is stored in {@link ModelHparamsBuilder#fullAttentionInterval()}.
     *
     * @param defaultInterval interval used when {@code full_attention_interval} is absent
     * @throws GgufFormatException when the array length differs from the block count or the interval is zero
     */
    static void readRecurrentLayers(ModelHparamsBuilder b, GgufKv kv, int defaultInterval) {
        boolean[] flags = readLayerFlags(kv, GgufKeys.ATTENTION_RECURRENT_LAYERS, b.nLayerAll());
        if (flags != null) {
            b.isRecurrentArray(flags);
            return;
        }
        int interval = kv.u32(GgufKeys.FULL_ATTENTION_INTERVAL, defaultInterval);
        if (interval == 0) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.FULL_ATTENTION_INTERVAL)
                    + "' must be greater than zero");
        }
        boolean[] recr = new boolean[b.nLayerAll()];
        for (int il = 0; il < recr.length; il++) {
            recr[il] = il < b.nLayer() && (il + 1) % interval != 0;
        }
        b.isRecurrentArray(recr);
        b.fullAttentionInterval(interval);
    }

    /**
     * Per-layer flags as llama.cpp {@code get_key_or_arr} reads a bool array (llama-model-loader.cpp:460-496):
     * an array must hold exactly {@code n} entries, a scalar is broadcast.
     *
     * @return the flags, or null when the key is absent
     * @throws GgufFormatException when the array length differs from {@code n}
     */
    static boolean[] readLayerFlags(GgufKv kv, String keyFmt, int n) {
        if (!kv.has(keyFmt)) {
            return null;
        }
        if (kv.isArray(keyFmt)) {
            int len = kv.arrayLength(keyFmt);
            if (len != n) {
                throw new GgufFormatException("metadata key '" + kv.key(keyFmt) + "' has wrong array length; expected "
                        + n + ", got " + len);
            }
            return kv.boolArray(keyFmt);
        }
        boolean v = kv.bool(keyFmt, false);
        boolean[] out = new boolean[n];
        Arrays.fill(out, v);
        return out;
    }

    /**
     * Router gating from a required {@code expert_gating_func}; a missing key or the value {@code NONE (0)},
     * which the llama.cpp MoE graph cannot route with, is rejected.
     *
     * @throws GgufFormatException when the key is missing, zero or unknown
     */
    static GatingFunc requiredGating(GgufKv kv) {
        if (!kv.has(GgufKeys.EXPERT_GATING_FUNC)) {
            throw new GgufFormatException("required metadata key '" + kv.key(GgufKeys.EXPERT_GATING_FUNC)
                    + "' is missing");
        }
        GatingFunc g = HparamsSupport.readGating(kv, null);
        if (g == null) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.EXPERT_GATING_FUNC)
                    + "' must name a gating function, got NONE (0)");
        }
        return g;
    }

    /**
     * Reads the required MLA head sizes {@code attention.key_length_mla} / {@code value_length_mla} and
     * {@code attention.kv_lora_rank} (kimi-linear.cpp:6-8, kimi-k3.cpp:18-21, bailingmoe3.cpp:8-10,
     * glm5-next.cpp:10-13); callers read {@code attention.q_lora_rank} with their own requiredness.
     */
    static void readMla(ModelHparamsBuilder b, GgufKv kv) {
        b.headKMlaImpl(kv.u32Required(GgufKeys.ATTENTION_KEY_LENGTH_MLA));
        b.headVMlaImpl(kv.u32Required(GgufKeys.ATTENTION_VALUE_LENGTH_MLA));
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));
    }

    /**
     * Rejects a zero value of a size key the graph divides by or sizes buffers with.
     *
     * @throws GgufFormatException when {@code value == 0}
     */
    static void requireNonZero(GgufKv kv, String keyFmt, int value) {
        if (value == 0) {
            throw new GgufFormatException("metadata key '" + kv.key(keyFmt) + "' must be greater than zero");
        }
    }

    /**
     * Requires equal key and value head sizes, which the attention graphs of several hybrids assert
     * ({@code n_embd_head_v() == n_embd_head_k()}).
     *
     * @param where the asserting llama.cpp source line
     * @throws GgufFormatException when the sizes differ
     */
    static void requireEqualHeadSizes(ModelHparamsBuilder b, String where) {
        if (b.headKFull() != b.headVFull()) {
            throw invalid(b, "attention.key_length " + b.headKFull() + " and attention.value_length " + b.headVFull()
                    + " must be equal (" + where + ")");
        }
    }

    /**
     * Builds the exception for an inconsistent hparams combination of the builder's architecture.
     */
    static GgufFormatException invalid(ModelHparamsBuilder b, String message) {
        return new GgufFormatException(b.arch().ggufName() + ": " + message);
    }
}
