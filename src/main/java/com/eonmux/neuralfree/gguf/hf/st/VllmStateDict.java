package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.VllmWeightsMapper;
import com.eonmux.neuralfree.gguf.hf.names.HfArchNames;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Reverses vLLM's packed module state dict into the Hugging Face tensor names the converter reads.
 *
 * <p>A vLLM model stores a checkpoint under its own module names (vLLM {@code ShardedStateLoader.save_model},
 * {@code TensorSerializer.write_module}). Most names match the Hugging Face checkpoint, but vLLM packs what the
 * model implementation fuses:</p>
 * <ul>
 *   <li>{@code QKVParallelLinear}: one {@code qkv_proj} whose output dimension is q, then k, then v, split with the
 *       configuration's head counts;</li>
 *   <li>{@code MergedColumnParallelLinear}: {@code gate_up_proj} (and shared-expert variants), split into its
 *       stacked shards;</li>
 *   <li>{@code FusedMoE}: {@code w13_weight} (gate then up per expert) and {@code w2_weight} (down per expert),
 *       split per expert under the naming of the checkpoint's converter class
 *       ({@link VllmExpertLayouts}).</li>
 * </ul>
 * <p>Which fused modules split is read from the architecture's {@link VllmWeightsMapper} (vLLM's own
 * {@code hf_to_vllm_mapper} table): a {@code qkv_proj}/{@code gate_up_proj} that no stacked entry fuses is a
 * Hugging Face fused tensor the converter already handles and passes through. The architecture's GGUF tensor list
 * also decides it: a converter that keeps the fused query/key/value tensor (no {@code ATTN_Q}) or the fused gate/up
 * tensor (no {@code FFN_GATE}, Phi-3's fused-up gating) reads it unchanged. A packed module of a family this class
 * does not cover is refused by name rather than guessed at. Quantized FusedMoE siblings ({@code w13_weight_scale}
 * and the like) are refused by name: their packing is a quantization scheme NeuralFree does not convert here.</p>
 * <p>Every other name passes through unchanged. Immutable.</p>
 */
public final class VllmStateDict {

    private static final Logger LOG = LogManager.getLogger(VllmStateDict.class);

    /** The marker vLLM writes into a tensorized module state dict. */
    public static final String TENSORIZED_MARKER = ".vllm_tensorized_marker";

    private static final String QKV_SUFFIX = ".qkv_proj";
    private static final String GATE_UP_SUFFIX = ".gate_up_proj";
    private static final String W13 = ".w13_weight";
    private static final String W13_BIAS = ".w13_bias";
    private static final String W2 = ".w2_weight";
    private static final String W2_BIAS = ".w2_bias";
    private static final String WEIGHT = ".weight";
    private static final String BIAS = ".bias";
    private static final String LM_HEAD = "lm_head.weight";
    private static final String EMBED_TOKENS = "model.embed_tokens.weight";

    private VllmStateDict() {
    }

    /**
     * The Hugging Face view of a vLLM module state dict.
     *
     * @param stored the stored tensors, in checkpoint order
     * @param config the checkpoint's configuration, which decides the fused splits
     * @throws IOException when a packed module cannot be split for this architecture, or a packed family is not
     *                     covered
     */
    public static List<StoredTensor> toHf(List<StoredTensor> stored, HfConfig config) throws IOException {
        Objects.requireNonNull(stored, "stored");
        Objects.requireNonNull(config, "config");
        VllmWeightsMapper mapper = VllmWeightsMapper.of(config.architecture());
        List<StoredTensor> out = new ArrayList<>(stored.size());
        for (StoredTensor t : stored) {
            String name = t.name();
            if (name.equals(TENSORIZED_MARKER) || name.endsWith(TENSORIZED_MARKER)) {
                LOG.debug("dropping vLLM's tensorized marker '{}'", name);
                continue;
            }
            if (isQuantizedExpertSibling(name)) {
                throw unsupported(name, "quantized FusedMoE tensors (" + family(name)
                        + ") are not converted from a vLLM state dict");
            }
            if (isConv1d(config) && isConv1dName(name)) {
                throw unsupported(name, "GPT-2's vLLM state dict holds Conv1D weights transposed against the"
                        + " Hugging Face checkpoint, a layout NeuralFree does not reverse");
            }
            String tail = tail(name, QKV_SUFFIX);
            if (tail != null) {
                List<StoredTensor> split = splitFused(t, QKV_SUFFIX, tail, mapper, config, "qkv");
                if (split != null) {
                    out.addAll(split);
                    continue;
                }
            }
            tail = tail(name, GATE_UP_SUFFIX);
            if (tail != null) {
                List<StoredTensor> split = splitFused(t, GATE_UP_SUFFIX, tail, mapper, config, "gate_up");
                if (split != null) {
                    out.addAll(split);
                    continue;
                }
            }
            if (name.endsWith(W13) || name.endsWith(W13_BIAS) || name.endsWith(W2) || name.endsWith(W2_BIAS)) {
                out.addAll(splitExperts(t, config));
                continue;
            }
            out.add(t);
        }
        addTiedEmbedding(out, config);
        return out;
    }

    // ------------------------------------------------------------------ tied embeddings

    /**
     * Restores the tied token-embedding tensor vLLM's state dict drops.
     *
     * <p>vLLM ties {@code lm_head.weight} to the token embedding and {@code ShardedStateLoader.save_model} filters
     * tensors sharing storage, keeping the lexicographically smaller name: a tied checkpoint carries
     * {@code lm_head.weight} but not {@code model.embed_tokens.weight}. With {@code tie_word_embeddings} the two are
     * the same tensor, so the missing name reads the head's rows.</p>
     */
    private static void addTiedEmbedding(List<StoredTensor> out, HfConfig config) {
        if (!config.findBool(false, "tie_word_embeddings")) {
            return;
        }
        StoredTensor head = null;
        boolean embedded = false;
        for (StoredTensor t : out) {
            if (t.name().equals(LM_HEAD)) {
                head = t;
            } else if (t.name().equals(EMBED_TOKENS)) {
                embedded = true;
            }
        }
        if (head != null && !embedded) {
            out.add(StoredTensor.of(EMBED_TOKENS, head.source()));
        }
    }

    /** Whether {@code name} is a GPT-2 Conv1D projection, stored transposed by vLLM. */
    private static boolean isConv1dName(String name) {
        for (String suffix : new String[]{".c_attn.weight", ".c_attn.bias", ".c_proj.weight", ".c_proj.bias",
                ".c_fc.weight", ".c_fc.bias"}) {
            if (name.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    /** Whether the checkpoint is a GPT-2 family model, whose Hugging Face weights use Conv1D. */
    private static boolean isConv1d(HfConfig config) {
        String arch = config.architecture();
        return arch != null && arch.startsWith("GPT2");
    }

    // ------------------------------------------------------------------ fused linear projections

    /**
     * The tail ({@code .weight} or {@code .bias}) when {@code name} is a fused module {@code suffix} parameter, or
     * null.
     */
    private static String tail(String name, String suffix) {
        if (name.endsWith(suffix + WEIGHT)) {
            return WEIGHT;
        }
        return name.endsWith(suffix + BIAS) ? BIAS : null;
    }

    /**
     * The stacked shards of a fused {@code qkv_proj}/{@code gate_up_proj}, or null when the architecture's mapper
     * does not fuse that module (the converter's Hugging Face checkpoint stores it fused already).
     */
    private static List<StoredTensor> splitFused(StoredTensor tensor, String suffix, String tail,
                                                 VllmWeightsMapper mapper, HfConfig config, String kind)
            throws IOException {
        String name = tensor.name();
        String module = name.substring(0, name.length() - tail.length());
        if (kind.equals("qkv") && !splitsQkv(config)) {
            return null;
        }
        if (kind.equals("gate_up") && !splitsGateUp(config)) {
            return null;
        }
        if (mapper.unresolved()) {
            throw unsupported(name, "vLLM builds this architecture's weight mapper from code ("
                    + mapper.unresolvedReason() + ")");
        }
        List<String> shards = mapper.stackedShards(module);
        if (shards.isEmpty()) {
            return null;
        }
        long[] shape = tensor.shape();
        List<long[]> sizes = new ArrayList<>(shards.size());
        if (kind.equals("qkv")) {
            if (shards.size() != 3) {
                throw unsupported(name, "the vLLM mapper fuses " + shards.size() + " projections into "
                        + module + ", not q/k/v");
            }
            long[] qkv = qkvRows(config, name);
            sizes.add(new long[]{qkv[0]});
            sizes.add(new long[]{qkv[1]});
            sizes.add(new long[]{qkv[2]});
        } else {
            if (shards.size() != 2) {
                throw unsupported(name, "the vLLM mapper fuses " + shards.size() + " projections into " + module
                        + ", not gate/up");
            }
            long rows = shape[0];
            if (rows <= 0 || rows % 2 != 0) {
                throw unsupported(name, "the fused gate/up output dimension " + rows + " is not two equal halves");
            }
            sizes.add(new long[]{rows / 2});
            sizes.add(new long[]{rows / 2});
        }
        long total = 0;
        for (long[] s : sizes) {
            total = Math.addExact(total, s[0]);
        }
        if (total != shape[0]) {
            throw unsupported(name, "the fused output dimension " + shape[0] + " does not hold the projections "
                    + total + " rows of the configuration's head counts");
        }
        List<StoredTensor> out = new ArrayList<>(shards.size());
        long from = 0;
        for (int i = 0; i < shards.size(); i++) {
            long count = sizes.get(i)[0];
            RowSource part = RowTransforms.sliceRows(tensor.source(), from, count);
            out.add(StoredTensor.of(shards.get(i) + tail, part));
            from += count;
        }
        return out;
    }

    /**
     * Whether the converter reads the fused query/key/value shards, i.e. whether the architecture's GGUF tensor list
     * has {@code ATTN_Q} rather than only the fused {@code ATTN_QKV}. An architecture whose converter keeps the fused
     * tensor (Phi-3's fused {@code qkv_proj}) reads it through unchanged.
     */
    private static boolean splitsQkv(HfConfig config) {
        Set<String> tensors = archTensors(config);
        return tensors == null || tensors.contains("ATTN_Q");
    }

    /**
     * Whether the converter reads separate gate and up projections: its GGUF tensor list has {@code FFN_GATE}. A
     * fused-up architecture (Phi-3's {@code FUSED_UP} gating) stores the fused {@code gate_up_proj} the graph splits
     * itself and reads it through unchanged.
     */
    private static boolean splitsGateUp(HfConfig config) {
        Set<String> tensors = archTensors(config);
        return tensors == null || tensors.contains("FFN_GATE");
    }

    /** The GGUF tensor kinds of the configured architecture, or null when it has no static converter arch. */
    private static Set<String> archTensors(HfConfig config) {
        Optional<HfArchNames.Row> row = HfArchNames.row(config.architecture(), HfArchEntry.Kind.TEXT);
        if (!row.isPresent() || row.get().arch() == null) {
            return null;
        }
        return HfNameTables.archTensors(row.get().arch().ggufName());
    }

    /**
     * The q, k and v row counts of vLLM's packed {@code QKVParallelLinear}: {@code num_attention_heads} query heads,
     * {@code num_key_value_heads} key and value heads, each {@code head_dim} wide.
     */
    static long[] qkvRows(HfConfig config, String name) throws IOException {
        int heads = config.findInt(-1, "num_attention_heads", "n_heads", "n_head");
        if (heads <= 0) {
            throw unsupported(name, "the configuration has no positive num_attention_heads");
        }
        int kvHeads = config.findInt(heads, "num_key_value_heads", "n_kv_heads", "num_kv_heads");
        if (kvHeads <= 0 || kvHeads > heads) {
            throw unsupported(name, "the configuration's key/value head count " + kvHeads + " is outside [1, "
                    + heads + "]");
        }
        int headDim = config.findInt(0, "head_dim");
        if (headDim <= 0) {
            int hidden = config.findInt(-1, "hidden_size", "d_model", "n_embd");
            if (hidden <= 0 || hidden % heads != 0) {
                throw unsupported(name, "the configuration has neither head_dim nor a hidden_size divisible by "
                        + heads + " attention heads");
            }
            headDim = hidden / heads;
        }
        return new long[]{(long) heads * headDim, (long) kvHeads * headDim, (long) kvHeads * headDim};
    }

    // ------------------------------------------------------------------ FusedMoE

    /** Whether {@code name} is a quantization sibling of a packed expert weight or bias. */
    private static boolean isQuantizedExpertSibling(String name) {
        for (String stem : new String[]{W13, W13_BIAS, W2, W2_BIAS}) {
            if (name.contains(stem + "_")) {
                return true;
            }
        }
        return false;
    }

    /** The packed parameter a refusal is about, for its message. */
    private static String family(String name) {
        for (String stem : new String[]{W13_BIAS, W13, W2_BIAS, W2}) {
            int at = name.indexOf(stem);
            if (at >= 0) {
                return name.substring(at + 1);
            }
        }
        return name;
    }

    private static List<StoredTensor> splitExperts(StoredTensor tensor, HfConfig config) throws IOException {
        String name = tensor.name();
        String stem;
        boolean gateUp;
        String tail;
        boolean bias;
        if (name.endsWith(W13)) {
            stem = W13;
            gateUp = true;
            tail = WEIGHT;
            bias = false;
        } else if (name.endsWith(W13_BIAS)) {
            stem = W13_BIAS;
            gateUp = true;
            tail = BIAS;
            bias = true;
        } else if (name.endsWith(W2)) {
            stem = W2;
            gateUp = false;
            tail = WEIGHT;
            bias = false;
        } else {
            stem = W2_BIAS;
            gateUp = false;
            tail = BIAS;
            bias = true;
        }
        String prefix = name.substring(0, name.length() - stem.length());
        VllmExpertLayouts.Layout layout = VllmExpertLayouts.of(config).orElseThrow(() -> unsupported(name,
                "the converter class of architecture '" + config.architecture() + "' has no expert layout")
                );
        if (layout.kind() == VllmExpertLayouts.Kind.UNSUPPORTED) {
            throw unsupported(name, layout.refusal() + " (architecture '" + config.architecture() + "')");
        }
        long[] shape = tensor.shape();
        if (layout.kind() == VllmExpertLayouts.Kind.FUSED) {
            if (bias) {
                throw unsupported(name, "the fused expert bias of architecture '" + config.architecture()
                        + "' has no Hugging Face equivalent");
            }
            String suffix = gateUp ? layout.gateUpSuffix() : layout.downSuffix();
            RowSource source = layout.transpose() ? ExpertTransforms.transposeLast2(tensor.source())
                    : tensor.source();
            // the converter's fused stacks carry no `.weight`: its hook transposes and splits them itself
            return java.util.Collections.singletonList(StoredTensor.of(prefix + suffix, source));
        }
        int experts = config.findInt(-1, "num_experts", "num_local_experts", "n_routed_experts",
                "moe_num_experts");
        if (shape.length < 2 || shape.length > 3) {
            throw unsupported(name, "a packed expert tensor must be [experts, width] or [experts, width, hidden],"
                    + " got " + Arrays.toString(shape));
        }
        long expertCount = shape[0];
        if (experts >= 0 && experts != expertCount) {
            throw unsupported(name, "the configuration declares " + experts + " experts but the tensor holds "
                    + expertCount);
        }
        if (expertCount <= 0 || expertCount > Integer.MAX_VALUE) {
            throw unsupported(name, "the expert count " + expertCount + " is outside [1, 2^31)");
        }
        int e = (int) expertCount;
        List<StoredTensor> out = new ArrayList<>(gateUp ? e * 2 : e);
        if (gateUp) {
            if (shape[1] <= 0 || shape[1] % 2 != 0) {
                throw unsupported(name, "the packed gate/up width " + shape[1] + " is not two equal halves");
            }
            long half = shape[1] / 2;
            for (int x = 0; x < e; x++) {
                RowSource gate = squeeze(sliceWidth(tensor.source(), x, 0, half));
                RowSource up = squeeze(sliceWidth(tensor.source(), x, half, half));
                out.add(StoredTensor.of(prefix + "." + x + "." + layout.gate() + tail, gate));
                out.add(StoredTensor.of(prefix + "." + x + "." + layout.up() + tail, up));
            }
        } else {
            for (int x = 0; x < e; x++) {
                RowSource down = squeeze(RowTransforms.sliceDim(tensor.source(), 0, x, 1));
                out.add(StoredTensor.of(prefix + "." + x + "." + layout.down() + tail, down));
            }
        }
        return out;
    }

    /** One expert's half-width block of a packed stack: dim 0 narrowed to expert {@code x}, then dim 1. */
    private static RowSource sliceWidth(RowSource source, int x, long from, long count) {
        return RowTransforms.sliceDim(RowTransforms.sliceDim(source, 0, x, 1), 1, from, count);
    }

    /** The expert dimension of a one-expert view removed. */
    private static RowSource squeeze(RowSource one) {
        return ElementTransforms.squeezeDim(one, 0);
    }

    private static IOException unsupported(String name, String reason) {
        return new IOException("Invalid vLLM checkpoint: tensor '" + name + "' cannot be mapped to a Hugging Face"
                + " tensor: " + reason);
    }
}
