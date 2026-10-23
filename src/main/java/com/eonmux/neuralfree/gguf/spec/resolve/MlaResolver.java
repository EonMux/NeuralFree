package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Resolves multi-head latent attention (deepseek2 family, deepseek32, glm-dsa, kimi-linear, kimi-k3, bailingmoe3,
 * glm5-next, dots3note, hy-v4, minicpm3, plm) into an {@link MlaSpec}. A layer uses MLA when it carries
 * {@code attn_kv_a_mqa} ({@code [n_embd, kv_lora_rank + n_rot]}, deepseek2.cpp:106).
 *
 * <p>Shapes: {@code head_k_rope = n_rot}, {@code head_k_nope = n_embd_head_k_mla - n_rot}, {@code head_v =
 * n_embd_head_v_mla} (deepseek2.cpp:69-74); dots3note SWA and NextN layers use the {@code *_swa} MLA sizes
 * (dots3note.cpp:75-85). The query is low-rank when {@code attn_q_a} exists ({@code q_lora_rank} = its rows).
 *
 * <p>Absorbed {@code attn_k_b}/{@code attn_v_b} allow the latent cache of {@code [kv_lora | rope]} rows; the
 * {@link MemoryConfig#mlaCache()} mode picks it ({@code AUTO}: latent when absorbed). {@code attn_kv_b} only
 * allows the decompressed per-head cache.
 *
 * <p>{@code kq_scale}: the deepseek2 graph family pre-scales by the YaRN {@code mscale^2} (deepseek2.cpp:427-437 with
 * the context's {@code yarn_attn_factor}, llama-context.cpp:257-306); the others use {@code 1/sqrt(head_k)}. Immutable.
 */
public final class MlaResolver {

    private final ModelHparams hp;
    private final TensorCatalog tc;
    private final MemoryConfig.MlaCacheMode cacheMode;

    public MlaResolver(ModelHparams hp, TensorCatalog tc, MemoryConfig memory) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
        this.cacheMode = Objects.requireNonNull(memory, "memory").mlaCache();
    }

    /** True when main-stack layer {@code il} runs MLA attention. */
    public boolean isMlaLayer(int il) {
        checkLayer(il);
        return tc.hasLayer(il, TensorNames.weight(TensorNames.ATTN_KV_A_MQA));
    }

    /**
     * MLA facet of layer {@code il}.
     *
     * @throws IllegalArgumentException when the layer has no {@code attn_kv_a_mqa}
     * @throws MissingTensorException   when neither absorbed nor {@code attn_kv_b} weights exist
     * @throws IllegalStateException    for inconsistent latent shapes, a latent cache requested without absorbed
     *                                  weights, or an architecture without a known {@code kq_scale} rule
     */
    public MlaSpec resolve(int il) {
        if (!isMlaLayer(il)) {
            throw new IllegalArgumentException(arch() + ": layer " + il + " has no MLA attention");
        }
        boolean swaDims = hp.arch() == Arch.DOTS3NOTE && (il >= hp.nLayer() || hp.isSwa(il));
        int kvLora = swaDims ? hp.nLoraKvSwa() : hp.nLoraKv();
        int headKMla = swaDims ? hp.headKMlaSwa() : hp.headKMla();
        int headVMla = swaDims ? hp.headVMlaSwa() : hp.headVMla();
        int nRot = hp.nRot(0);                                   // hparams.n_rot(): the layer-0 rotary dims

        TensorRef kvA = tc.require(il, TensorNames.weight(TensorNames.ATTN_KV_A_MQA));
        if (kvA.ne(1) != (long) kvLora + nRot) {
            throw new IllegalStateException(arch() + ": " + kvA.name() + " has " + kvA.ne(1)
                    + " rows, expected kv_lora_rank " + kvLora + " + n_rot " + nRot);
        }
        boolean absorbed = tc.hasLayer(il, TensorNames.weight(TensorNames.ATTN_K_B))
                && tc.hasLayer(il, TensorNames.weight(TensorNames.ATTN_V_B));
        if (!absorbed) {
            tc.require(il, TensorNames.weight(TensorNames.ATTN_KV_B));
        }
        TensorRef qA = tc.layer(il, TensorNames.weight(TensorNames.ATTN_Q_A));
        return MlaSpec.builder()
                .nHead(hp.nHead(il))
                .qLoraRank(qA != null ? FfnResolver.toInt(qA.ne(1), il) : 0)
                .kvLoraRank(kvLora)
                .headKNope(headKMla - nRot)
                .headKRope(nRot)
                .headV(headVMla)
                .absorbed(absorbed)
                .latentCache(latentCache(il, absorbed))
                .ropeApplied(hp.ropeType() != RopeType.NONE && nRot > 0)
                .kqScale(kqScale(headKMla))
                .build();
    }

    private boolean latentCache(int il, boolean absorbed) {
        switch (cacheMode) {
            case LATENT:
                if (!absorbed) {
                    throw new IllegalStateException(arch() + ": layer " + il + " MLA latent cache requested but the "
                            + "model has no absorbed attn_k_b/attn_v_b weights");
                }
                return true;
            case DECOMPRESSED:
                return false;
            default:
                return absorbed;
        }
    }

    /** Attention score scale of the MLA graph of this architecture. */
    float kqScale(int headKMla) {
        switch (hp.arch()) {
            case DEEPSEEK2:
            case MISTRAL4:
            case DEEPSEEK32:
            case GLM_DSA:
                return DeepseekScale.yarnKqScale(hp, headKMla);
            case KIMI_LINEAR:
            case KIMI_K3:
            case GLM5_NEXT:
            case DOTS3NOTE:
            case HY_V4:
            case BAILINGMOE3:
                return 1.0f / (float) Math.sqrt(headKMla);
            case MINICPM3:
            case PLM:
                return 1.0f / (float) Math.sqrt(hp.headK(0));          // minicpm3.cpp:68, plm.cpp:49: n_embd_head_k()
            default:
                throw new IllegalStateException(arch() + ": no MLA kq_scale rule for this architecture");
        }
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }

    private String arch() {
        return hp.arch().ggufName();
    }
}
