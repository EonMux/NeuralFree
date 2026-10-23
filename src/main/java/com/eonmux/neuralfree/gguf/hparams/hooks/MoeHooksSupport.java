package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.IntPredicate;

/**
 * Key-reading primitives shared by the MoE, MLA and DSA hooks ({@link MoeHooks}, {@link MoeHooksMla},
 * {@link MoeHooksSwa}). Each helper reproduces one llama.cpp loader call
 * ({@code get_key}, {@code get_arr}, {@code get_arr_n}, {@code get_key_or_arr}, src/llama-model-loader.cpp)
 * with its required/optional semantics and its treatment of absent keys.
 */
final class MoeHooksSupport {

    private MoeHooksSupport() {
    }

    /** {@code get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, f_norm_rms_eps)}, required. */
    static void rmsEps(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));
    }

    /**
     * {@code get_key(LLM_KV_EXPERT_GATING_FUNC, expert_gating_func)} with the key required. A present
     * {@code NONE (0)} value leaves llama.cpp without a router function, so it is rejected.
     *
     * @throws GgufFormatException when the key is missing, unknown or zero
     */
    static GatingFunc requiredGating(GgufKv kv) {
        if (!kv.has(GgufKeys.EXPERT_GATING_FUNC)) {
            throw new GgufFormatException("required metadata key '" + kv.key(GgufKeys.EXPERT_GATING_FUNC)
                    + "' is missing");
        }
        GatingFunc g = HparamsSupport.readGating(kv, null);
        if (g == null) {
            throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.EXPERT_GATING_FUNC)
                    + "' must name a gating function, got none (0)");
        }
        return g;
    }

    /** Optional gating key; absent or {@code NONE (0)} yields {@code dflt}. */
    static GatingFunc gating(GgufKv kv, GatingFunc dflt) {
        return HparamsSupport.readGating(kv, dflt);
    }

    /**
     * Required boolean key ({@code get_key} on a bool member).
     *
     * @throws GgufFormatException when the key is missing
     */
    static boolean boolRequired(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        return kv.bool(key, false);
    }

    /**
     * Copies the full-attention RoPE base and scale to the SWA layers, then reads the optional
     * {@code rope.freq_base_swa} over the base (the {@code rope_freq_base_train_swa = rope_freq_base_train;
     * rope_freq_scale_train_swa = rope_freq_scale_train; get_key(LLM_KV_ROPE_FREQ_BASE_SWA, ..., false)}
     * sequence).
     */
    static void swaRopeFromFull(ModelHparamsBuilder b, GgufKv kv) {
        b.ropeFreqBaseSwa(b.ropeFreqBase());
        b.ropeFreqScaleSwa(b.ropeFreqScale());
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBase());
    }

    /**
     * {@code get_arr(key, std::array<uint32_t, LLAMA_MAX_LAYERS>)} overlay of a per-layer flag array: the
     * array's entries replace the first {@code length} values of {@code current} and later layers keep
     * theirs; BOOL and integer element types are accepted (nonzero is true).
     *
     * @return the merged flags, or {@code current} unchanged when the key is absent and optional
     * @throws GgufFormatException when the key is required and missing, or longer than
     *                             {@link ModelHparams#MAX_LAYERS}
     */
    static boolean[] overlayFlags(GgufKv kv, String key, boolean[] current, boolean required) {
        boolean[] flags = kv.boolArray(key);
        if (flags == null) {
            if (kv.has(key)) {
                throw new GgufFormatException("metadata key '" + kv.key(key) + "' must be an array");
            }
            if (required) {
                throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
            }
            return current;
        }
        if (flags.length > ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException("metadata key '" + kv.key(key) + "' has " + flags.length
                    + " entries, more than the maximum of " + ModelHparams.MAX_LAYERS + " layers");
        }
        boolean[] out = current.clone();
        System.arraycopy(flags, 0, out, 0, Math.min(flags.length, out.length));
        return out;
    }

    /**
     * {@code get_key_or_arr(key, arr, n, required)} into an unsigned per-layer flag array: an array must have
     * exactly {@code n} entries, a scalar is broadcast to the first {@code n} layers, and layers at or after
     * {@code n} keep their values.
     *
     * @return the merged flags, or {@code current} unchanged when the key is absent and optional
     */
    static boolean[] keyOrArrFlags(GgufKv kv, String key, int n, boolean[] current, boolean required) {
        if (!kv.has(key)) {
            if (required) {
                throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
            }
            return current;
        }
        int[] fallback = new int[n];
        int[] values = kv.keyOrArr(key, n, fallback);
        boolean[] out = current.clone();
        for (int il = 0; il < n && il < out.length; il++) {
            out[il] = values[il] != 0;
        }
        return out;
    }

    /**
     * {@code get_key_or_arr(key, float array, n_layer_all, required)}: scalar broadcast or exact-length array.
     *
     * @return the per-layer values, or null when the key is absent and optional
     * @throws GgufFormatException when the key is required and missing
     */
    static float[] keyOrArrF32(ModelHparamsBuilder b, GgufKv kv, String key, boolean required) {
        if (!kv.has(key)) {
            if (required) {
                throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
            }
            return null;
        }
        return kv.keyOrArrF32(key, b.nLayerAll(), 0f);
    }

    /** Per-layer flags over all {@code nLayerAll} blocks from a layer predicate. */
    static boolean[] layersWhere(ModelHparamsBuilder b, IntPredicate when) {
        boolean[] out = new boolean[b.nLayerAll()];
        for (int il = 0; il < out.length; il++) {
            out[il] = when.test(il);
        }
        return out;
    }

    /**
     * Rejects a zero divisor that llama.cpp would use in a {@code %} expression (undefined behaviour there).
     *
     * @throws GgufFormatException when {@code value} is not positive
     */
    static int requirePositive(GgufKv kv, String key, int value) {
        if (value <= 0) {
            throw new GgufFormatException("metadata key '" + kv.key(key) + "' must be positive, got " + value);
        }
        return value;
    }

    /**
     * Optional {@code deepstack_mapping} ({@code get_arr} into a -1 filled per-layer array): the deepstack
     * layer count is the number of distinct non-negative indices, and no index may exceed that count
     * (models/granite-swa.cpp:22-40).
     *
     * @throws GgufFormatException for an index above the count or an array longer than
     *                             {@link ModelHparams#MAX_LAYERS}
     */
    static void readDeepstackMapping(ModelHparamsBuilder b, GgufKv kv) {
        String key = GgufKeys.DEEPSTACK_MAPPING;
        int[] mapping = kv.i32Array(key);
        Set<Integer> unique = new TreeSet<>();
        if (mapping != null) {
            if (mapping.length > ModelHparams.MAX_LAYERS) {
                throw new GgufFormatException("metadata key '" + kv.key(key) + "' has " + mapping.length
                        + " entries, more than the maximum of " + ModelHparams.MAX_LAYERS + " layers");
            }
            int[] perLayer = new int[b.nLayerAll()];
            Arrays.fill(perLayer, -1);
            System.arraycopy(mapping, 0, perLayer, 0, Math.min(mapping.length, perLayer.length));
            b.deepstackMappingArray(perLayer);
            for (int v : mapping) {
                if (v >= 0) {
                    unique.add(v);
                }
            }
        }
        b.nDeepstackLayers(unique.size());
        for (int v : unique) {
            if (v > unique.size()) {
                throw new GgufFormatException("Invalid deepstack index: " + v + " > " + unique.size());
            }
        }
    }

    /**
     * The {@code n_expert > 0} / {@code n_expert_used > 0} checks that llama.cpp {@code load_arch_tensors}
     * performs before creating routed-expert tensors.
     *
     * @throws GgufFormatException when the expert count or the used-expert count is zero
     */
    static void requireExperts(ModelHparamsBuilder b, String arch) {
        if (b.nExpert() == 0 || b.nExpertUsed(0) == 0) {
            throw new GgufFormatException(arch + ": n_expert and n_expert_used must be > 0 (got " + b.nExpert()
                    + " and " + b.nExpertUsed(0) + ")");
        }
    }

    /**
     * {@link #requireExperts} for loaders that reach the expert branch only for layers in
     * {@code [n_layer_dense_lead, loopEnd)}.
     */
    static void requireExpertsAfterLead(ModelHparamsBuilder b, String arch, int loopEnd) {
        if (b.nLayerDenseLead() < loopEnd) {
            requireExperts(b, arch);
        }
    }
}
