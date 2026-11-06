package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hugging Face conversion of {@code Zamba2ForCausalLM} (transformers/models/zamba2/modeling_zamba2.py, vLLM
 * vllm/model_executor/models/zamba2.py): Mamba-2 layers interleaved with hybrid layers whose shared attention/MLP
 * blocks are stored once per bare block and reused by several depths.
 *
 * <p>Tensor names: the Mamba-2 block of layer {@code i} is {@code model.layers.i.mamba.*} (or
 * {@code model.layers.i.mamba_decoder.mamba.*} in the hybrid layers of a vLLM-style checkpoint), its input norm
 * {@code model.layers.i.input_layernorm.weight} (or {@code mamba_decoder.input_layernorm.weight}). The shared block
 * {@code b} of a hybrid layer is {@code model.layers.i.shared_transformer.*} and is written once as
 * {@code shared_blk.b.*}; the per-use projection is {@code model.layers.i.linear.weight}. Adapters live under
 * {@code linear_{q,k,v}_adapter_list.j.{0,1}.weight} and {@code gate_up_proj_adapter_list.j.{0,1}.weight}, where
 * {@code j} is the hybrid ordinal of the use (the list index), and become
 * {@code blk.<layer(j)>.attn_{q,k,v}_adapter_{a,b}.weight} / {@code blk.<layer(j)>.ffn_gate_up_adapter_{a,b}.weight};
 * the tied copies a checkpoint repeats under the other layers of the block are dropped.</p>
 */
public final class HfZamba2Hooks {

    private static final String SHARED_PREFIX = "shared_transformer.";
    private static final Pattern LAYER = Pattern.compile("^model\\.layers\\.(\\d+)\\.(.+)$");
    private static final Pattern ATTENTION_ADAPTER = Pattern.compile(
            "self_attn\\.linear_([qkv])_adapter_list\\.(\\d+)\\.([01])\\.weight");
    private static final Pattern MLP_ADAPTER = Pattern.compile(
            "feed_forward\\.gate_up_proj_adapter_list\\.(\\d+)\\.([01])\\.weight");
    private static final String STATE_LAYOUT = HfZamba2Hooks.class.getName() + ".layout";
    private static final String STATE_WRITTEN = HfZamba2Hooks.class.getName() + ".written";
    private static final String STATE_TOK_EMBD = HfZamba2Hooks.class.getName() + ".tokEmbd";
    private static final String TOKEN_EMBD = "token_embd.weight";
    private static final String OUTPUT = "output.weight";
    private static final String OUTPUT_NORM = "output_norm.weight";

    private HfZamba2Hooks() {
    }

    // ------------------------------------------------------------------ architecture

    /** {@link Arch#ZAMBA2} for every {@code Zamba2ForCausalLM} checkpoint. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.ZAMBA2;
    }

    /**
     * Refuses the configurations the engine's shared block cannot express: a biased Mamba-2 projection
     * ({@code add_bias_linear}) and a shared MLP activation other than GELU (the block always runs
     * {@code GELU_ERF(gate) * up}, vLLM {@code GeluAndMul}).
     *
     * @throws GgufFormatException naming the unsupported setting
     */
    public static HfConfig fixup(HfConfig cfg) {
        if (cfg.findBool(false, "add_bias_linear")) {
            throw new GgufFormatException(cfg.source() + ": add_bias_linear is unsupported; the Mamba-2 in/out"
                    + " projections must have no bias (vLLM Zamba2MambaDecoderLayer use_bias)");
        }
        String act = cfg.findStringOr("gelu", "hidden_act", "activation_function");
        if (!"gelu".equals(act)) {
            throw new GgufFormatException(cfg.source() + ": hidden_act '" + act + "' is unsupported; the shared"
                    + " transformer block's gated MLP is GELU_ERF (vLLM GeluAndMul)");
        }
        return cfg;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code Zamba2Model.set_gguf_parameters}: the Mamba-2 sizes, the shared attention's {@code 2 * hidden_size}
     * input with whole-head RoPE, and the shared-block keys of {@link GgufKeys#SHARED_BLOCKS_COUNT} and friends.
     *
     * @throws GgufFormatException when the configuration contradicts the shared-block geometry
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        Layout layout = layout(ctx);
        int nLayer = layout.hybrid.length;
        long hidden = cfg.requireInt("hidden_size");
        long heads = cfg.requireInt("num_attention_heads");
        long headsKv = cfg.findInt((int) heads, "num_key_value_heads");
        if (hidden <= 0 || heads <= 0 || 2 * hidden % heads != 0) {
            throw new GgufFormatException(cfg.source() + ": attention_hidden_size " + (2 * hidden)
                    + " is not divisible by num_attention_heads " + heads);
        }
        long headDim = 2 * hidden / heads;
        if (headDim % 2 != 0) {
            throw new GgufFormatException(cfg.source() + ": attention_head_dim " + headDim + " must be even; the"
                    + " shared attention scales by (head_dim / 2) ** -0.5");
        }
        if (headsKv != heads) {
            throw new GgufFormatException(cfg.source() + ": num_key_value_heads " + headsKv
                    + " must equal num_attention_heads " + heads + "; the shared attention's q/k/v adapters all"
                    + " write attention_hidden_size values (vLLM Zamba2Attention)");
        }
        long intermediate = cfg.findInt((int) (4 * hidden), "intermediate_size");
        long dInner = innerSize(cfg);
        long nHeads = cfg.findInt(0, "n_mamba_heads");
        long dState = cfg.findInt(64, "mamba_d_state");
        long dConv = cfg.findInt(4, "mamba_d_conv");
        long nGroups = cfg.findInt(1, "mamba_ngroups", "n_groups");
        if (nHeads <= 0 || dInner % nHeads != 0) {
            throw new GgufFormatException(cfg.source() + ": mamba_expand * hidden_size " + dInner
                    + " is not a whole number of mamba heads " + nHeads);
        }
        if (nGroups <= 0 || dInner % nGroups != 0 || dState <= 0 || dConv <= 0) {
            throw new GgufFormatException(cfg.source() + ": mamba_d_state " + dState + ", mamba_d_conv " + dConv
                    + " and mamba_ngroups " + nGroups + " do not fit the SSM inner size " + dInner);
        }
        int memBlocks = cfg.findInt(1, "num_mem_blocks");
        int rank = cfg.findInt(128, "adapter_rank");
        boolean attentionAdapter = cfg.findBool(false, "use_shared_attention_adapter");
        boolean memoryRope = cfg.findBool(false, "use_mem_rope");
        boolean longContext = cfg.findBool(false, "use_long_context");
        if (layout.hybridCount == 0) {
            throw new GgufFormatException(cfg.source() + ": layers_block_type marks no hybrid layer; a Zamba2"
                    + " model needs at least one shared transformer block");
        }
        if (memBlocks < 1 || memBlocks > layout.hybridCount) {
            throw new GgufFormatException(cfg.source() + ": num_mem_blocks " + memBlocks + " must be in [1, "
                    + layout.hybridCount + "], the number of hybrid layers");
        }
        if (rank < 1) {
            throw new GgufFormatException(cfg.source() + ": adapter_rank " + rank + " must be positive");
        }
        double eps = cfg.findFloat(1e-5, "rms_norm_eps", "layer_norm_epsilon");
        double theta = cfg.findFloat(10000.0, "rope_theta");
        if (memoryRope && longContext) {
            theta = theta * Math.pow(8.0, headDim / (headDim - 2.0));
        }
        long context = cfg.findInt(4096, "max_position_embeddings");
        if (longContext) {
            context = 16384;
        }
        kv.set(GgufKeys.CONTEXT_LENGTH, context);
        kv.set(GgufKeys.EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, intermediate);
        kv.set(GgufKeys.BLOCK_COUNT, (long) nLayer);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, headsKv);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, eps);
        kv.set(GgufKeys.SSM_CONV_KERNEL, dConv);
        kv.set(GgufKeys.SSM_INNER_SIZE, dInner);
        kv.set(GgufKeys.SSM_STATE_SIZE, dState);
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, nHeads);
        kv.set(GgufKeys.SSM_GROUP_COUNT, nGroups);
        if (memoryRope) {
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, headDim);
            kv.set(GgufKeys.ROPE_FREQ_BASE, theta);
        }
        kv.set(GgufKeys.SHARED_BLOCKS_COUNT, (long) memBlocks);
        kv.set(GgufKeys.SHARED_BLOCKS_ADAPTER_RANK, (long) rank);
        kv.set(GgufKeys.SHARED_BLOCKS_ATTENTION_ADAPTER, attentionAdapter);
        kv.set(GgufKeys.SHARED_BLOCKS_MEMORY_ROPE, memoryRope);
        kv.set(GgufKeys.SHARED_BLOCKS_HYBRID_LAYERS, layout.hybrid);
    }

    /**
     * The Mamba-family vocabulary chain of {@code Mamba2Model} (mamba.py:128-141), unpadded: a Zamba2 checkpoint's
     * tokenizer already matches its embedding table.
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfSsmHooks.ssmVocab(ctx, pre, 1, false);
    }

    // ------------------------------------------------------------------ tensors

    /**
     * Plans every Zamba2 tensor under its GGUF name: the model-level embedding, output norm and (untied) head, the
     * per-layer Mamba-2 block and shared-linear projection, the shared blocks (once per bare block) and the per-use
     * adapters. Tensors of other modules (the rotary embedding's buffers) are ignored.
     *
     * @throws IOException when a tied-output comparison cannot read a tensor
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if ("model.embed_tokens.weight".equals(hfName)) {
            addUnique(ctx, TOKEN_EMBD, src, out, hfName);
            ctx.state(STATE_TOK_EMBD, TokEmbd.class, TokEmbd::new).source = src;
            return;
        }
        if ("model.final_layernorm.weight".equals(hfName)) {
            addUnique(ctx, OUTPUT_NORM, src, out, hfName);
            return;
        }
        if ("lm_head.weight".equals(hfName)) {
            TokEmbd tok = ctx.state(STATE_TOK_EMBD, TokEmbd.class, TokEmbd::new);
            if (tok.source != null && HfSsmHooks.equalValues(tok.source, src)) {
                out.drop(hfName, "equals token_embd byte for byte (Zamba2Config.tie_word_embeddings)");
                return;
            }
            addUnique(ctx, OUTPUT, src, out, hfName);
            return;
        }
        Matcher m = LAYER.matcher(hfName);
        if (!m.matches()) {
            return;
        }
        int il = Integer.parseInt(m.group(1));
        String rest = m.group(2);
        String gguf = layerTensor(ctx, il, rest);
        if (gguf == null) {
            return;
        }
        addUnique(ctx, gguf, transform(ctx, rest, src), out, hfName);
    }

    /** The GGUF name of a layer tensor, or null when the converter drops it. */
    private static String layerTensor(HfLoadContext ctx, int il, String rest) {
        String norm = TensorNames.layerName(il, TensorNames.weight(TensorNames.ATTN_NORM));
        if ("input_layernorm.weight".equals(rest) || "mamba_decoder.input_layernorm.weight".equals(rest)) {
            return norm;
        }
        if ("linear.weight".equals(rest)) {
            return TensorNames.layerName(il, TensorNames.weight("shared_linear"));
        }
        if (rest.startsWith("mamba_decoder.mamba.")) {
            return mambaTensor(il, rest.substring("mamba_decoder.mamba.".length()));
        }
        if (rest.startsWith("mamba.")) {
            return mambaTensor(il, rest.substring("mamba.".length()));
        }
        if (rest.startsWith(SHARED_PREFIX)) {
            return sharedTensor(ctx, il, rest.substring(SHARED_PREFIX.length()));
        }
        return null;
    }

    private static String mambaTensor(int il, String tail) {
        switch (tail) {
            case "in_proj.weight":
                return TensorNames.layerName(il, TensorNames.weight(TensorNames.SSM_IN));
            case "conv1d.weight":
                return TensorNames.layerName(il, TensorNames.weight(TensorNames.SSM_CONV1D));
            case "conv1d.bias":
                return TensorNames.layerName(il, TensorNames.SSM_CONV1D + ".bias");
            case "dt_bias":
                return TensorNames.layerName(il, TensorNames.SSM_DT + ".bias");
            case "A_log":
                return TensorNames.layerName(il, TensorNames.SSM_A);
            case "D":
                return TensorNames.layerName(il, TensorNames.SSM_D);
            case "norm.weight":
                return TensorNames.layerName(il, TensorNames.weight(TensorNames.SSM_NORM));
            case "out_proj.weight":
                return TensorNames.layerName(il, TensorNames.weight(TensorNames.SSM_OUT));
            default:
                return null;
        }
    }

    private static String sharedTensor(HfLoadContext ctx, int il, String tail) {
        Layout layout = layout(ctx);
        int block = il < layout.block.length ? layout.block[il] : -1;
        if (block < 0) {
            return null;
        }
        Matcher attn = ATTENTION_ADAPTER.matcher(tail);
        if (attn.matches()) {
            int use = layout.hybridLayer(Integer.parseInt(attn.group(2)));
            return TensorNames.layerName(use, TensorNames.weight("attn_" + attn.group(1) + "_adapter_"
                    + ("0".equals(attn.group(3)) ? "a" : "b")));
        }
        Matcher mlp = MLP_ADAPTER.matcher(tail);
        if (mlp.matches()) {
            int use = layout.hybridLayer(Integer.parseInt(mlp.group(1)));
            return TensorNames.layerName(use, TensorNames.weight("ffn_gate_up_adapter_"
                    + ("0".equals(mlp.group(2)) ? "a" : "b")));
        }
        String prefix = TensorNames.STACK_SHARED_BLK + "." + block + ".";
        switch (tail) {
            case "input_layernorm.weight":
                return prefix + TensorNames.weight(TensorNames.ATTN_NORM);
            case "self_attn.q_proj.weight":
                return prefix + TensorNames.weight(TensorNames.ATTN_Q);
            case "self_attn.k_proj.weight":
                return prefix + TensorNames.weight(TensorNames.ATTN_K);
            case "self_attn.v_proj.weight":
                return prefix + TensorNames.weight(TensorNames.ATTN_V);
            case "self_attn.o_proj.weight":
                return prefix + TensorNames.weight(TensorNames.ATTN_OUT);
            case "pre_ff_layernorm.weight":
                return prefix + "attn_pre_ff_norm.weight";
            case "feed_forward.gate_up_proj.weight":
                return prefix + "ffn_gate_up.weight";
            case "feed_forward.down_proj.weight":
                return prefix + "ffn_down.weight";
            default:
                return null;
        }
    }

    /** The Mamba-2 value transforms of {@code Mamba2Model.modify_tensors} (mamba.py:181-200). */
    private static RowSource transform(HfLoadContext ctx, String rest, RowSource src) {
        if (rest.endsWith("conv1d.weight")) {
            return ElementTransforms.squeezeConv(src);
        }
        if (rest.endsWith("A_log")) {
            return ElementTransforms.negExp(src);
        }
        if (rest.endsWith(".D")) {
            return HfSsmHooks.reshapeAppendOne(src);
        }
        if (rest.endsWith("mamba.norm.weight") || rest.endsWith("mamba_decoder.mamba.norm.weight")) {
            return ElementTransforms.reshape(src, HfSsmHooks.groupShape(ctx.config(), innerSize(ctx.config()),
                    "mamba_ngroups", "n_groups"));
        }
        return src;
    }

    /** The Mamba-2 inner width: {@code mamba_expand * hidden_size} (Zamba2Config.mamba_expand). */
    private static long innerSize(HfConfig cfg) {
        return cfg.findInt(2, "mamba_expand") * cfg.requireInt("hidden_size");
    }

    /** Adds {@code gguf} once: a tied duplicate of an already planned tensor is dropped. */
    private static void addUnique(HfLoadContext ctx, String gguf, RowSource src, TensorSink out, String hfName) {
        Written written = ctx.state(STATE_WRITTEN, Written.class, Written::new);
        if (!written.names.add(gguf)) {
            out.drop(hfName, "tied duplicate of " + gguf + " (Zamba2Model shares the block across hybrid layers)");
            return;
        }
        out.add(gguf, src);
    }

    // ------------------------------------------------------------------ layout

    /** The hybrid flag of every layer and the block and hybrid ordinal of every layer. */
    private static Layout layout(HfLoadContext ctx) {
        return ctx.state(STATE_LAYOUT, Layout.class, () -> Layout.of(ctx.config()));
    }

    private static final class Layout {
        private final boolean[] hybrid;
        private final int[] block;
        private final int[] hybridLayer;
        private final int hybridCount;

        private Layout(boolean[] hybrid, int[] block, int[] hybridLayer, int hybridCount) {
            this.hybrid = hybrid;
            this.block = block;
            this.hybridLayer = hybridLayer;
            this.hybridCount = hybridCount;
        }

        private static Layout of(HfConfig cfg) {
            List<String> types = HfMoeSwaHooksB.stringList(cfg, "layers_block_type");
            if (types.isEmpty()) {
                throw new GgufFormatException(cfg.source() + ": required key 'layers_block_type' is missing;"
                        + " a Zamba2 checkpoint lists its 'mamba' and 'hybrid' layers");
            }
            int n = types.size();
            boolean[] hybrid = new boolean[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                String t = types.get(i);
                if ("hybrid".equals(t)) {
                    hybrid[i] = true;
                    count++;
                } else if (!"mamba".equals(t)) {
                    throw new GgufFormatException(cfg.source() + ": layers_block_type[" + i + "] = '" + t
                            + "' is neither 'mamba' nor 'hybrid'");
                }
            }
            int memBlocks = cfg.findInt(1, "num_mem_blocks");
            if (memBlocks < 1) {
                throw new GgufFormatException(cfg.source() + ": num_mem_blocks " + memBlocks + " must be positive");
            }
            int[] block = new int[n];
            int[] hybridLayer = new int[count];
            int ordinal = 0;
            for (int i = 0; i < n; i++) {
                if (hybrid[i]) {
                    block[i] = ordinal % memBlocks;
                    hybridLayer[ordinal] = i;
                    ordinal++;
                } else {
                    block[i] = -1;
                }
            }
            return new Layout(hybrid, block, hybridLayer, count);
        }

        private int hybridLayer(int ordinal) {
            if (ordinal < 0 || ordinal >= hybridLayer.length) {
                throw new GgufFormatException("adapter list index " + ordinal + " is not a hybrid layer ordinal"
                        + " (0.." + (hybridLayer.length - 1) + ")");
            }
            return hybridLayer[ordinal];
        }
    }

    private static final class Written {
        private final Set<String> names = new HashSet<>();
    }

    private static final class TokEmbd {
        private RowSource source;
    }
}
