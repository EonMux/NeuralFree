package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A qwen4exp target (1 trunk block + its NextN block, all layers full attention) and its MTP-only sidecar
 * files (qwen4exp.cpp:267-300): the shared sidecar holds only the MTP block, with and without the file's own
 * {@code token_embd}. The trunk QSA compress ratio and the MTP entry of {@code attention.compress_ratios} are
 * parameters, so tests can exercise the MTP ratio inheritance (qwen4exp.cpp:141-156). In-memory
 * ({@link GgufKv#ofMap}, {@link TensorCatalog#of}).
 */
public final class Qwen4ExpSidecars {

    private static final String ARCH = "qwen4exp";
    /** Trunk blocks, NextN blocks. */
    private static final int TRUNK = 1;
    private static final int NEXTN = 1;
    /** Trunk QSA compress ratio; {@code top_k} 8 divides it. */
    private static final int RATIO = 4;

    private Qwen4ExpSidecars() {
    }

    /** Metadata of the target and of its sidecar: one model family, differing only in the MTP ratio. */
    public static Map<String, Object> hparams(int trunkRatio, int mtpRatio) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", ARCH);
        m.put("general.name", "synthetic-qwen4exp");
        m.put(ARCH + ".vocab_size", 64);
        m.put(ARCH + ".context_length", 256);
        m.put(ARCH + ".embedding_length", 64);
        m.put(ARCH + ".block_count", TRUNK + NEXTN);
        m.put(ARCH + ".nextn_predict_layers", NEXTN);
        m.put(ARCH + ".feed_forward_length", 128);
        m.put(ARCH + ".attention.head_count", 8);
        m.put(ARCH + ".attention.head_count_kv", 8);
        m.put(ARCH + ".attention.key_length", 16);
        m.put(ARCH + ".attention.value_length", 16);
        m.put(ARCH + ".attention.layer_norm_rms_epsilon", 1e-6f);
        m.put(ARCH + ".attention.recurrent_layers", new int[] {0, 0});
        m.put(ARCH + ".rope.dimension_count", 16);
        m.put(ARCH + ".rope.dimension_sections", new int[] {2, 2, 2, 2});
        m.put(ARCH + ".rope.freq_base", 10000.0f);
        m.put(ARCH + ".expert_count", 4);
        m.put(ARCH + ".expert_used_count", 2);
        m.put(ARCH + ".expert_feed_forward_length", 32);
        m.put(ARCH + ".expert_shared_feed_forward_length", 32);
        m.put(ARCH + ".ssm.conv_kernel", 4);
        m.put(ARCH + ".ssm.inner_size", 64);
        m.put(ARCH + ".ssm.state_size", 16);
        m.put(ARCH + ".ssm.time_step_rank", 4);
        m.put(ARCH + ".ssm.group_count", 2);
        m.put(ARCH + ".hyper_connection.count", 4);
        m.put(ARCH + ".hyper_connection.low_rank", 8);
        m.put(ARCH + ".attention.indexer.head_count", 4);
        m.put(ARCH + ".attention.indexer.key_length", 32);
        m.put(ARCH + ".attention.indexer.top_k", 8);
        m.put(ARCH + ".attention.compress_ratios", new int[] {trunkRatio, mtpRatio});
        m.put("tokenizer.ggml.model", "no_vocab");
        return m;
    }

    /** The target: the trunk block, the NextN block, the embeddings, the HC head and the LM head. */
    public static List<GGUFTensorInfo> targetTensors() {
        List<GGUFTensorInfo> ts = blocks(0, 1);
        ts.add(t("token_embd.weight", 64, 64));
        ts.add(t("output_hc_norm.weight", 64, 4));
        ts.add(t("output_hc_down.weight", 256, 8));
        ts.add(t("output_hc_up.weight", 8, 256));
        ts.add(t("output.weight", 64, 64));
        return ts;
    }

    /** The shared MTP-only sidecar: the NextN block, no embeddings, no HC head, no LM head. */
    public static List<GGUFTensorInfo> sharedSidecarTensors() {
        return blocks(1);
    }

    /** The shared MTP-only sidecar that carries its own {@code token_embd} but no LM head. */
    public static List<GGUFTensorInfo> sidecarWithEmbeddingTensors() {
        List<GGUFTensorInfo> ts = blocks(1);
        ts.add(t("token_embd.weight", 64, 64));
        return ts;
    }

    /** The MTP-only sidecar with its own embeddings and LM head. */
    public static List<GGUFTensorInfo> fullSidecarTensors() {
        List<GGUFTensorInfo> ts = blocks(1);
        ts.add(t("token_embd.weight", 64, 64));
        ts.add(t("output.weight", 64, 64));
        return ts;
    }

    /** The hparams of {@code hparams(trunkRatio, mtpRatio)}. */
    public static ModelHparams loadHparams(int trunkRatio, int mtpRatio) {
        return HparamsLoader.load(GgufKv.ofMap(hparams(trunkRatio, mtpRatio)), ArchRegistry.get(Arch.QWEN4EXP));
    }

    /** The spec of {@code targetTensors()}. */
    public static ModelSpec buildTarget() {
        return ModelSpecBuilder.build(GgufKv.ofMap(hparams(RATIO, RATIO)), TensorCatalog.of(ARCH, targetTensors()));
    }

    /**
     * The MTP drafter spec of a sidecar over {@code target}.
     *
     * @param trunkRatio the trunk QSA ratio the sidecar's metadata gives
     * @param mtpRatio   the MTP entry of its {@code attention.compress_ratios}, 0 when the file gives none
     * @param tensors    the sidecar's tensors ({@link #sharedSidecarTensors()} and friends)
     */
    public static ModelSpec buildSidecar(ModelSpec target, int trunkRatio, int mtpRatio,
                                         List<GGUFTensorInfo> tensors) {
        return ModelSpecBuilder.buildMtpDraft(GgufKv.ofMap(hparams(trunkRatio, mtpRatio)),
                TensorCatalog.of(ARCH, tensors), MemoryConfig.defaults(), target);
    }

    /** The stored tensors of the blocks {@code il}, the NextN block with its fusion and gated head. */
    private static List<GGUFTensorInfo> blocks(int... ils) {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        for (int il : ils) {
            String p = "blk." + il + ".";
            ts.add(t(p + "hc_attn_norm.weight", 64, 4));
            ts.add(t(p + "hc_attn_down.weight", 256, 8));
            ts.add(t(p + "hc_attn_up.weight", 8, 256));
            ts.add(t(p + "hc_attn_inject.weight", 256, 4));
            ts.add(t(p + "hc_ffn_norm.weight", 64, 4));
            ts.add(t(p + "hc_ffn_down.weight", 256, 8));
            ts.add(t(p + "hc_ffn_up.weight", 8, 256));
            ts.add(t(p + "hc_ffn_inject.weight", 256, 4));
            ts.add(t(p + "attn_q.weight", 64, 256));
            ts.add(t(p + "attn_k.weight", 64, 128));
            ts.add(t(p + "attn_v.weight", 64, 128));
            ts.add(t(p + "attn_output.weight", 128, 64));
            ts.add(t(p + "attn_q_norm.weight", 16));
            ts.add(t(p + "attn_k_norm.weight", 16));
            ts.add(t(p + "indexer.q_proj.weight", 64, 128));
            ts.add(t(p + "indexer.k_proj.weight", 64, 32));
            ts.add(t(p + "indexer.q_norm.weight", 32));
            ts.add(t(p + "indexer.k_norm.weight", 32));
            ts.add(t(p + "ffn_gate_inp.weight", 64, 4));
            ts.add(t(p + "ffn_gate_exps.weight", 64, 32, 4));
            ts.add(t(p + "ffn_up_exps.weight", 64, 32, 4));
            ts.add(t(p + "ffn_down_exps.weight", 32, 64, 4));
            ts.add(t(p + "ffn_gate_inp_shexp.weight", 64));
            ts.add(t(p + "ffn_gate_shexp.weight", 64, 32));
            ts.add(t(p + "ffn_up_shexp.weight", 64, 32));
            ts.add(t(p + "ffn_down_shexp.weight", 32, 64));
            if (il >= TRUNK) {
                ts.add(t(p + "nextn.eh_proj.weight", 128, 64));
                ts.add(t(p + "nextn.enorm.weight", 64));
                ts.add(t(p + "nextn.hnorm.weight", 64, 4));
                ts.add(t(p + "nextn.hc_head_norm.weight", 64, 4));
                ts.add(t(p + "nextn.hc_head_down.weight", 256, 8));
                ts.add(t(p + "nextn.hc_head_up.weight", 8, 256));
            }
        }
        return ts;
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }
}
