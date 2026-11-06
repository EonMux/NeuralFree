package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig;
import com.eonmux.neuralfree.gguf.hf.config.OnlineQuantConfig.LayerKind;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The eligibility and the producer of online weight quantization: {@code --quant-online} and the
 * checkpoint's own online {@code quant_method} quantize float matmul weights at load, with llama-quantize's
 * eligibility rules (LC/src/llama-quant.cpp:302-421 {@code tensor_allows_quantization}, including its
 * architecture-gated exclusions and the {@code quantize_output_tensor} choice) and the quantizers of
 * {@link OnlineQuantizers} — the llama-quantize block types row by row, and vLLM's online fp8, int8, mxfp8,
 * mxfp4 and nvfp4 methods through {@link OnlineQuantEncoders}. The vLLM methods quantize the modules vLLM does: a
 * shorthand by {@link #vllmQuantizes} (never the embeddings or LM heads, and only the routed experts for the
 * MoE-only methods), a quantization config by the caller's per-layer choice ({@link #quantize}); the members of a
 * fused projection share their tensor-scoped scales ({@link #quantizeFused}). The registry is the one seam to
 * extend; {@code OnlineQuant.NONE} requantizes nothing.
 */
public final class RequantPolicy {

    /** Requantization keeps nothing below rank 2 ({@code ggml_n_dims(tensor) < 2}, llama-quant.cpp:307). */
    private static final int MIN_RANK = 2;

    /**
     * The names requantization keeps away from, in the order of llama-quant.cpp:315-418: the norm tensors, the
     * expert gating and routing tensors, the Engram gate, the very small projections (altup, laurel,
     * per_layer_model_proj), the Mamba, Kimi and LFM3 convolutions, MiniMax's tiny indexer, RWKV's small time-mix
     * weights (:388-402), T5's relative position bias (:405) and the multimodal position, patch and audio
     * codebook tensors (:408-418).
     */
    private static final String[] FORBIDDEN = {
        "_norm.weight", "ffn_gate_inp.weight", "ffn_gate_tid2eid.weight", "engram_gate.weight", "altup", "laurel",
        "per_layer_model_proj", "ssm_conv1d", "shortconv.conv.weight", "conv2d_spatial.weight", "conv_dw",
        "indexer.k_proj.weight", "indexer.q_proj.weight", "time_mix_first.weight", "time_mix_w0.weight",
        "time_mix_w1.weight", "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight",
        "time_mix_v2.weight", "time_mix_a0.weight", "time_mix_a1.weight", "time_mix_a2.weight",
        "time_mix_g1.weight", "time_mix_g2.weight", "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
        "time_mix_lerp_fused.weight", "attn_rel_b.weight", ".position_embd", "sam.pos_embd", "sam.neck.",
        "sam.net_", ".rel_pos", ".patch_embd", ".patch_merger", "a.rvq.codebook", "mm.a.code_embd",
    };

    /**
     * The names llama-quant.cpp:336-341 keeps unquantized on a LIMITE model only: its very small gates and dense
     * connection mixers.
     */
    private static final String[] FORBIDDEN_LIMITE = {"attn_gate.weight", "attn_ve_gate.weight", "mudd_"};

    /**
     * The names llama-quant.cpp:364-380 keeps unquantized on a GLM5-next model only: its hc, indexer and MLA/SSM
     * projection weights.
     */
    private static final String[] FORBIDDEN_GLM5_NEXT = {
        "hc_", "indexer.attn_q_b", "indexer.attn_k", "indexer.proj", "indexer_compressor_gate",
        "indexer_compressor_ape", "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight",
        "ssm_beta.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight",
    };

    /** The name prefixes llama-quant.cpp:356-358 keeps away from: LFM3-Audio's encoder convs and codec FSQ. */
    private static final String[] FORBIDDEN_PREFIX = {"a.conv1d.", "a.gen.wav.quant."};

    /** The exact names llama-quant.cpp compares, :344-345 (BERT positions and token types). */
    private static final String[] FORBIDDEN_EXACT = {"pos_embd.weight", "token_types.weight"};

    /**
     * The GGUF names of the embeddings and LM heads, which vLLM's online methods leave unquantized: they are
     * {@code VocabParallelEmbedding} / {@code ParallelLMHead} modules, neither a linear layer nor routed experts.
     * Matched as name suffixes.
     */
    private static final String[] VLLM_UNQUANTIZED = {"token_embd.weight", "nextn.embed_tokens.weight",
        "nextn.shared_head_head.weight"};

    /**
     * The GGUF names of the linear layers vLLM builds without a quantization config, so online quantization never
     * reaches them: the shared-expert gate of Qwen2-MoE and Qwen3-Next ({@code ReplicatedLinear(...,
     * quant_config=None)}). Matched as name suffixes.
     */
    private static final String[] VLLM_NO_QUANT_CONFIG = {"ffn_gate_inp_shexp.weight"};

    /** The GGUF name suffix of every routed-experts weight ({@code ffn_gate_exps}, {@code ffn_gate_up_exps}, ...). */
    private static final String ROUTED_EXPERTS_SUFFIX = "_exps.weight";

    /**
     * The forbidden names vLLM's online methods quantize all the same, being weights of linear layers built with
     * the quantization config: MLA's {@code kv_a_proj_with_mqa} (alone or in {@code fused_qkv_a_proj}) and the
     * {@code kv_b_proj} the converter splits into {@code attn_k_b} and {@code attn_v_b} (deepseek_v2.py), and
     * Qwen3.5's {@code in_proj_b} in {@code in_proj_ba} (qwen_gdn_linear_attn.py:867-889).
     */
    private static final String[] VLLM_QUANTIZED = {"attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight",
        "ssm_beta.weight"};

    private RequantPolicy() {
    }

    /**
     * The vLLM layer class of the GGUF tensor {@code ggufName}: {@link LayerKind#MOE} for the routed experts,
     * {@link LayerKind#EMBEDDING} for the embeddings and LM heads, null for the linear layers vLLM builds without a
     * quantization config, and {@link LayerKind#LINEAR} for every other tensor.
     */
    public static LayerKind layerKind(String ggufName) {
        Objects.requireNonNull(ggufName, "ggufName");
        if (ggufName.endsWith(ROUTED_EXPERTS_SUFFIX)) {
            return LayerKind.MOE;
        }
        if (ggufName.equals("output.weight") || endsWithAny(ggufName, VLLM_UNQUANTIZED)) {
            return LayerKind.EMBEDDING;
        }
        return endsWithAny(ggufName, VLLM_NO_QUANT_CONFIG) ? null : LayerKind.LINEAR;
    }

    private static boolean endsWithAny(String name, String[] suffixes) {
        for (String s : suffixes) {
            if (name.endsWith(s)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Whether the vLLM online shorthand {@code target} quantizes the GGUF tensor {@code ggufName}: the routed
     * experts for every method, the other linear weights for the methods with a {@code linear} spec
     * ({@link OnlineQuantConfig#shorthand}), never an embedding or LM head. The llama-quantize targets are not
     * vLLM methods and answer true; {@link #eligible} decides for them. {@link OnlineQuant#ONLINE} quantizes by its
     * configuration and answers false.
     */
    static boolean vllmQuantizes(String ggufName, OnlineQuant target) {
        Objects.requireNonNull(ggufName, "ggufName");
        Objects.requireNonNull(target, "target");
        if (!target.isVllm()) {
            return true;
        }
        OnlineQuantConfig shorthand = OnlineQuantConfig.shorthand(target);
        LayerKind kind = layerKind(ggufName);
        return shorthand != null && kind != null && shorthand.quantizes(kind);
    }

    /**
     * Whether the GGUF tensor {@code ggufName}, currently stored as {@code current} and target at {@code target},
     * is requantized online (llama-quant.cpp:302-421 over the current type and the plan's rank): its type is float
     * (F32, F16 or BF16), its rank is at least 2, {@code ne[0]} is a whole number of blocks of {@code target}, and
     * its name passes the ported name rules of model architecture {@code arch}: it ends with {@code weight}, is
     * none of the forbidden names, and, unless {@code quantizeOutputTensor}, is not the LM head.
     */
    public static boolean eligible(String ggufName, Arch arch, boolean quantizeOutputTensor, GGMLType current,
                                    long[] ne, GGMLType target) {
        return eligible(ggufName, arch, quantizeOutputTensor, current, ne, target, false);
    }

    /**
     * {@link #eligible(String, Arch, boolean, GGMLType, long[], GGMLType)}, with the forbidden names vLLM's online
     * methods quantize ({@link #VLLM_QUANTIZED}) allowed when {@code vllm}.
     */
    private static boolean eligible(String ggufName, Arch arch, boolean quantizeOutputTensor, GGMLType current,
                                    long[] ne, GGMLType target, boolean vllm) {
        if (current != GGMLType.F32 && current != GGMLType.F16 && current != GGMLType.BF16) {
            return false;
        }
        if (ne == null || ne.length < MIN_RANK || target == null) {
            return false;
        }
        if (target.getBlockSize() <= 0 || ne[0] % target.getBlockSize() != 0) {
            return false;
        }
        return nameEligible(ggufName, arch, quantizeOutputTensor, vllm);
    }

    /** The name rules of the llama-quantize targets ({@link #nameEligible(String, Arch, boolean)}). */
    static boolean nameEligible(String ggufName, Arch arch, boolean quantizeOutputTensor) {
        return nameEligible(ggufName, arch, quantizeOutputTensor, false);
    }

    /**
     * The name rules in one shot: the name ends with {@code weight} (llama-quant.cpp:312), is not
     * {@code output.weight} unless {@code quantizeOutputTensor} (:317), is none of the exact names, starts with
     * none of the forbidden prefixes, holds none of the forbidden spellings (:315-418), is not part of the
     * architecture-gated exclusions (LIMITE's gates :336-341, GLM5-next's projections :364-380, the decision head of
     * vllm-decision2 :382-385). With {@code vllm}, the forbidden spellings exclude the weights vLLM's online
     * methods quantize ({@link #VLLM_QUANTIZED}).
     */
    static boolean nameEligible(String ggufName, Arch arch, boolean quantizeOutputTensor, boolean vllm) {
        Objects.requireNonNull(arch, "arch");
        if (ggufName == null || !ggufName.endsWith("weight")) {
            return false;
        }
        if (!quantizeOutputTensor && ggufName.equals("output.weight")) {
            return false;
        }
        if (arch == Arch.VLLM_DECISION2 && ggufName.startsWith("decision.")) {
            return false;
        }
        if (vllm && endsWithAny(ggufName, VLLM_QUANTIZED)) {
            return true;
        }
        if (arch == Arch.LIMITE && containsAny(ggufName, FORBIDDEN_LIMITE)) {
            return false;
        }
        if (arch == Arch.GLM5_NEXT && containsAny(ggufName, FORBIDDEN_GLM5_NEXT)) {
            return false;
        }
        for (String forbidden : FORBIDDEN_EXACT) {
            if (ggufName.equals(forbidden)) {
                return false;
            }
        }
        for (String forbidden : FORBIDDEN_PREFIX) {
            if (ggufName.startsWith(forbidden)) {
                return false;
            }
        }
        return !containsAny(ggufName, FORBIDDEN);
    }

    private static boolean containsAny(String name, String[] parts) {
        for (String part : parts) {
            if (name.contains(part)) {
                return true;
            }
        }
        return false;
    }

    /**
     * The online requantization step of the Hugging Face planning pipeline: {@code src} requantized, or {@code src} itself when the
     * option carries no quantizer ({@code NONE}), {@code src} is not a plain float source (an already-quantized
     * checkpoint's tensors are never requantized), the planned type {@code current} is not float, the tensor is
     * not {@link #eligible(String, Arch, boolean, GGMLType, long[], GGMLType)}, or a vLLM online shorthand leaves
     * the tensor unquantized ({@link #vllmQuantizes}).
     *
     * @throws IllegalArgumentException for {@link OnlineQuant#ONLINE}, which selects per layer
     */
    public static RowSource replace(String ggufName, Arch arch, boolean quantizeOutputTensor, RowSource src,
                                    GGMLType current, OnlineQuant target) {
        Objects.requireNonNull(src, "src");
        if (target == null || target == OnlineQuant.NONE || !src.format().isFloat()) {
            return src;
        }
        requireConcrete(target);
        if (!vllmQuantizes(ggufName, target)) {
            return src;
        }
        return quantize(ggufName, arch, quantizeOutputTensor, src, current, target);
    }

    /**
     * {@code src} quantized into {@code target}, a layer the caller selected, or {@code src} itself when it is not
     * a plain float source or the tensor is not
     * {@link #eligible(String, Arch, boolean, RowSource, GGMLType, OnlineQuant)}. A tensor-scoped target scans
     * {@code src} alone ({@link #quantizeFused} scans a fused group). MLA's {@code attn_k_b} under a row-local
     * method is vLLM's dequantized {@code W_UK} ({@link MlaKeyRows}): its stored rows run down the per-row scales of
     * vLLM's {@code kv_b}.
     *
     * @throws IllegalArgumentException for {@link OnlineQuant#NONE} and {@link OnlineQuant#ONLINE}, and for a
     *                                  tensor-scoped target on a member of {@code kv_b}, whose scales span both
     *                                  members ({@link #quantizeFused})
     */
    public static RowSource quantize(String ggufName, Arch arch, boolean quantizeOutputTensor, RowSource src,
                                     GGMLType current, OnlineQuant target) {
        Objects.requireNonNull(src, "src");
        requireConcrete(target);
        if (!eligible(ggufName, arch, quantizeOutputTensor, src, current, target)) {
            return src;
        }
        if (OnlineQuantizers.scanning(target) != null) {
            return quantizeFused(Collections.singletonList(ggufName), Collections.singletonList(src), target).get(0);
        }
        OnlineQuantizer q = OnlineQuantizers.get(target);
        return FusedProjections.transposed(ggufName) ? MlaKeyRows.rowLocal(ggufName, src, q)
                : QuantizedRows.rowLocal(ggufName, src, q);
    }

    /**
     * Whether the tensor {@code ggufName} planned from {@code src} as {@code current} quantizes into
     * {@code target}: {@code src} is a plain float source and the tensor is
     * {@link #eligible(String, Arch, boolean, GGMLType, long[], GGMLType) eligible} for the storage type of vLLM's
     * weight rows ({@link OnlineQuantizer#typeFor}, {@link ScanningQuantizer#typeFor}: for MLA's transposed
     * {@code attn_k_b}, the rows of {@code kv_lora} values), with the names vLLM's methods quantize allowed for
     * them.
     */
    public static boolean eligible(String ggufName, Arch arch, boolean quantizeOutputTensor, RowSource src,
                                   GGMLType current, OnlineQuant target) {
        Objects.requireNonNull(src, "src");
        requireConcrete(target);
        if (!src.format().isFloat()) {
            return false;
        }
        long[] ne = PlannedTensor.neOf(src.shape());
        if (FusedProjections.transposed(ggufName)) {
            if (ne.length != 3) {
                return false;
            }
            ne = new long[]{ne[1], ne[0], ne[2]};
        }
        if (ne.length < MIN_RANK || ne[0] <= 0 || ne[0] > Integer.MAX_VALUE) {
            return false;
        }
        ScanningQuantizer scanning = OnlineQuantizers.scanning(target);
        int n = (int) ne[0];
        GGMLType type = scanning != null ? scanning.typeFor(n) : OnlineQuantizers.get(target).typeFor(n);
        return eligible(ggufName, arch, quantizeOutputTensor, current, ne, type, target.isVllm());
    }

    /**
     * The members of one fused projection quantized with one set of tensor-scoped scales, as vLLM quantizes its
     * fused weight ({@code QKVParallelLinear}, {@code MergedColumnParallelLinear}, the experts' {@code w13},
     * MLA's {@code fused_qkv_a_proj} and {@code kv_b_proj}, the GDN {@code in_proj_qkvz} and {@code in_proj_ba}):
     * the members, in fusion order, form one virtual tensor per expert ({@link FusedLayout}, by the
     * {@link FusedProjections#layoutOf layout} of the group and the {@link #layerKind layer kind} of its first
     * member), scanned together, so a per-tensor scale is the maximum over every member and per-128x128-block tiles
     * run over vLLM's rows: the checkpoint's row order of a member the converter reordered, the 128-row padding of
     * each routed-experts shard, and the columns of {@code kv_b} that {@code attn_k_b} stores as rows. Each returned
     * source writes its own rows from the shared scales: as the scanner's row type, a transposed member as the
     * scanner's {@link ScanningQuantizer#columnTypeFor column type}, or as vLLM's dequantized {@code W_UK}
     * ({@link MlaKeyRows}) when there is none. Every member is
     * {@link #eligible(String, RowSource, GGMLType, OnlineQuant) eligible}, which the caller checks.
     *
     * @throws IllegalArgumentException when {@code target} is not tensor-scoped, or the members differ in row
     *                                  length or expert count or do not form the group's layout
     * @throws UncheckedIOException     when a member's stored floats cannot be read
     */
    public static List<RowSource> quantizeFused(List<String> names, List<RowSource> srcs, OnlineQuant target) {
        Objects.requireNonNull(names, "names");
        Objects.requireNonNull(srcs, "srcs");
        ScanningQuantizer prototype = OnlineQuantizers.scanning(target);
        if (prototype == null) {
            throw new IllegalArgumentException(target + " has no tensor-scoped scales to share");
        }
        if (names.isEmpty() || names.size() != srcs.size()) {
            throw new IllegalArgumentException("a fused group needs one name per source, got " + names + " for "
                    + srcs.size() + " sources");
        }
        FusedLayout layout = FusedLayout.of(names, srcs, FusedProjections.layoutOf(names.get(0)),
                layerKind(names.get(0)) == LayerKind.MOE);
        ScanningQuantizer s = prototype.open(layout.rows, layout.inner, layout.experts);
        for (int m = 0; m < srcs.size(); m++) {
            scan(s, names.get(m), srcs.get(m), layout.placements.get(m), layout.inner);
        }
        s.freeze();
        List<RowSource> out = new ArrayList<>(srcs.size());
        for (int m = 0; m < srcs.size(); m++) {
            out.add(member(s, names.get(m), srcs.get(m), layout.placements.get(m)));
        }
        return out;
    }

    /** Feeds every stored float of member {@code src} at {@code at} to the scan {@code s}. */
    private static void scan(ScanningQuantizer s, String name, RowSource src, FusedLayout.Placement at, int inner) {
        int n = FusedLayout.innerOf(name, src.shape());
        try {
            FloatRows.consume(name, src, n, 0, Math.toIntExact(src.rows()), (row, x, off) -> {
                if (at.transposed()) {
                    s.scanColumn(at.row(row), at.column(row), x, off, n);
                } else {
                    s.scanRow(at.row(row), x, off, inner);
                }
            });
        } catch (IOException e) {
            throw new UncheckedIOException(name + ": the quantization scan could not read the stored floats", e);
        }
    }

    /** Member {@code src} at {@code at}, quantized from the frozen scan {@code s}. */
    private static RowSource member(ScanningQuantizer s, String name, RowSource src, FusedLayout.Placement at) {
        if (!at.transposed()) {
            return QuantizedRows.scanned(name, src, s, at, null);
        }
        GGMLType column = s.columnTypeFor(FusedLayout.innerOf(name, src.shape()), at.columnAlign());
        return column != null ? QuantizedRows.scanned(name, src, s, at, column)
                : MlaKeyRows.scanned(name, src, s, at);
    }

    private static void requireConcrete(OnlineQuant target) {
        Objects.requireNonNull(target, "target");
        if (target == OnlineQuant.NONE || target == OnlineQuant.ONLINE) {
            throw new IllegalArgumentException(target + " is not a quantizer; the online method selects one per"
                    + " layer");
        }
    }
}
