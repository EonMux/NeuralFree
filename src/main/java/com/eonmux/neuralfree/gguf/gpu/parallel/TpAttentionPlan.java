package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.ops.attn.QkvProjection;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The attention half of a tensor-parallel split ({@code neuralfree.split.mode=tensor}, vLLM's tensor parallelism):
 * every eligible trunk layer's query, key and value projections are column-parallel (rank {@code r} holds the
 * {@code n_head / ranks} query heads and the {@code n_head_kv / ranks} key/value heads starting at the same head
 * index) and its output projection is row-parallel (the matching column block of {@code attn_output}), so each rank
 * attends over its own KV cache rows and a device-to-device all-reduce sums the partial branch outputs.
 *
 * <p>A layer is eligible when it is standard softmax attention over a cache on the main trunk, with a head count
 * and a KV head count that divide into the ranks (both are refused by name otherwise, which also covers GQA/MQA
 * caches with fewer KV heads than ranks), a per-head Q/K norm (the whole-projection norms normalize across heads and
 * cannot be sharded), no MLA, MSA, DSV4, shared-KV, value-expert, value-embedding, low-rank-Q or linear-replace
 * graph, no output value scale, and no PRISM input transform on the output projection (that transform is built for
 * the full attention width). A layer whose graphs fail a check is refused by name rather than silently left whole,
 * because a whole attention layer would need a full cache on the main device beside the per-rank shards. The plan is
 * immutable.</p>
 *
 * <p>{@link #context} builds the context-parallel plan ({@code neuralfree.split.mode=context}, vLLM's decode context
 * parallelism): every eligible layer keeps all of its query and KV heads on every rank, and the ranks divide the
 * context positions between them ({@link #contextStart}/{@link #contextEnd}), so each rank attends over one context
 * block of its own full cache and the merged log-sum-exp combines the ranks. Head divisibility is not required, a
 * layer that is ineligible keeps its whole-device mixer and cache on the main device, and every tensor a rank needs
 * uploads whole. The block partition and eligibility are the same on every rank.</p>
 */
public final class TpAttentionPlan {

    /** How one attention tensor moves to a rank. */
    public enum Shard {
        /** The rank's slice of the output neurons (Q/K/V/gate rows, or their biases). */
        ROWS,
        /** The rank's segments of a fused {@code attn_qkv} product ({@code [Q | K | V]} rows). */
        FUSED_ROWS,
        /** The rank's heads of a per-head Q/K norm weight or bias. */
        NORM_HEADS,
        /** A per-head vector of one value per head (sinks, gate bias). */
        PER_HEAD,
        /** The rank's column block of the output projection. */
        OUT_COLUMNS,
        /** The whole output-width bias, uploaded to rank 0 only because the all-reduce sums the branch outputs. */
        OUT_BIAS,
        /** The whole tensor on every rank (a Q/K norm row the heads share). */
        WHOLE
    }

    /** The split attention of one layer. */
    public static final class Layer {
        private final String stack;
        private final int weightLayer;
        private final int il;
        private final int nHead;
        private final int nHeadKv;
        private final int headDimK;
        private final int headDimV;
        private final int shardHeads;
        private final int shardKvHeads;
        private final int qProjWidth;
        private final int kDim;
        private final int vDim;
        private final int outDim;
        private final boolean fused;
        private final Map<String, Shard> tensors;

        Layer(String stack, int weightLayer, int il, int nHead, int nHeadKv, int headDimK, int headDimV,
              int shardHeads, int shardKvHeads, int qProjWidth, int kDim, int vDim, int outDim, boolean fused,
              Map<String, Shard> tensors) {
            this.stack = stack;
            this.weightLayer = weightLayer;
            this.il = il;
            this.nHead = nHead;
            this.nHeadKv = nHeadKv;
            this.headDimK = headDimK;
            this.headDimV = headDimV;
            this.shardHeads = shardHeads;
            this.shardKvHeads = shardKvHeads;
            this.qProjWidth = qProjWidth;
            this.kDim = kDim;
            this.vDim = vDim;
            this.outDim = outDim;
            this.fused = fused;
            this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
        }

        public String stack() { return stack; }
        public int weightLayer() { return weightLayer; }
        public int il() { return il; }
        public int nHead() { return nHead; }
        public int nHeadKv() { return nHeadKv; }
        public int headDimK() { return headDimK; }
        public int headDimV() { return headDimV; }
        /** Query heads each rank holds. */
        public int shardHeads() { return shardHeads; }
        /** Key/value heads each rank holds. */
        public int shardKvHeads() { return shardKvHeads; }
        /** Width of the query projection's output; {@code 2 * n_head * head_dim_k} with an interleaved gate. */
        public int qProjWidth() { return qProjWidth; }
        public int kDim() { return kDim; }
        public int vDim() { return vDim; }
        public int outDim() { return outDim; }
        /** True when Q, K and V come from one fused {@code attn_qkv} projection. */
        public boolean fused() { return fused; }
        /** First query head rank {@code rank} holds. */
        public int headStart(int rank) { return rank * shardHeads; }
        /** First KV head rank {@code rank} holds. */
        public int kvHeadStart(int rank) { return rank * shardKvHeads; }
        /** First output column rank {@code rank} holds. */
        public int outColumnStart(int rank) { return rank * (outDim / (nHead / shardHeads)); }
        /** Output columns each rank holds. */
        public int outColumns() { return (int) ((long) shardHeads * headDimV); }

        /** The tensor suffixes this layer splits, in model order. */
        public Map<String, Shard> tensors() { return tensors; }
        /** How a tensor whose suffix is {@code suffix} moves to a rank, or null when it is not split. */
        public Shard shardOf(String suffix) { return tensors.get(suffix); }
        /** The full tensor name of a suffix. */
        public String tensor(String suffix) {
            return TensorNames.layerName(stack, weightLayer, suffix);
        }

        /** The layer's attention spec with this rank's heads. */
        public AttentionSpec shardSpec(AttentionSpec of, int rank) {
            return of.toBuilder().nHead(shardHeads).nHeadKv(shardKvHeads).build();
        }

        @Override
        public String toString() {
            return stack + "." + il + " (" + nHead + " query / " + nHeadKv + " KV heads, " + shardHeads + " / "
                    + shardKvHeads + " per rank)";
        }
    }

    private final int ranks;
    private final List<Layer> layers;
    private final Map<String, Layer> byLayer;
    private final Map<String, Layer> byTensor;
    private final boolean context;
    /** The layers a context split leaves whole on the main device, with the reason, in model order. */
    private final List<String> skipped;

    private TpAttentionPlan(int ranks, List<Layer> layers, Map<String, Layer> byLayer, Map<String, Layer> byTensor,
                            boolean context, List<String> skipped) {
        this.ranks = ranks;
        this.layers = layers;
        this.byLayer = byLayer;
        this.byTensor = byTensor;
        this.context = context;
        this.skipped = Collections.unmodifiableList(new ArrayList<>(skipped));
    }

    /** Number of ranks the heads divide over. */
    public int ranks() { return ranks; }

    /**
     * True when the plan is context parallelism rather than vLLM's tensor parallelism: every rank holds every head
     * and its attention covers one context block, and the ranks' partial softmaxes merge through
     * {@code LLM_AttnCpMerge}. False for a head-sharded tensor split.
     */
    public boolean contextParallel() { return context; }

    /** First context position rank {@code rank} attends to over a context of {@code nCtx} positions. */
    public int contextStart(int rank, int nCtx) {
        checkContextRank(rank);
        return (int) ((long) rank * nCtx / ranks);
    }

    /** One past the last context position rank {@code rank} attends to. */
    public int contextEnd(int rank, int nCtx) {
        checkContextRank(rank);
        return (int) ((long) (rank + 1) * nCtx / ranks);
    }

    private void checkContextRank(int rank) {
        if (!context) {
            throw new IllegalStateException("the plan is not a context-parallel split");
        }
        if (rank < 0 || rank >= ranks) {
            throw new IllegalArgumentException("rank " + rank + " is outside the " + ranks + " rank(s)");
        }
    }

    /** The split layers, in model order (unmodifiable). */
    public List<Layer> layers() { return Collections.unmodifiableList(layers); }

    /**
     * The attention layers a context split leaves whole on the main device, each as {@code stack.il: reason}, in
     * model order (unmodifiable and empty for a tensor split).
     */
    public List<String> skipped() { return skipped; }

    /** The split attention of a layer, or null when the layer's attention stays whole. */
    public Layer layerOf(LayerSpec layer) {
        return byLayer.get(layerKey(layer));
    }

    /** The split attention a tensor belongs to, or null when the tensor is not split. */
    public Layer layerOf(String tensorName) {
        return byTensor.get(tensorName);
    }

    /** True when a tensor uploads as per-rank shards. */
    public boolean splits(String tensorName) {
        return byTensor.containsKey(tensorName);
    }

    /** True when a layer's attention is sharded. */
    public boolean splits(LayerSpec layer) {
        return byLayer.containsKey(layerKey(layer));
    }

    /**
     * Plans the attention split of {@code spec}'s trunk over {@code ranks} ranks.
     *
     * @throws UnsupportedOperationException when no attention layer is eligible; the message names the first
     *                                       disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks or null arguments
     */
    public static TpAttentionPlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader) {
        return plan(spec, ranks, uploader, false);
    }

    /**
     * Plans context parallelism over {@code ranks}: every eligible trunk layer keeps every head on every rank and
     * the ranks divide the context positions between them, so the per-rank attention covers one context block and
     * the merged log-sum-exp combines them. A layer whose attention cannot run this way keeps its whole-device mixer
     * on the main device and is left out of the plan, so a model with some ineligible layers still runs; those
     * layers' caches live on the main device only.
     *
     * @throws UnsupportedOperationException when no attention layer is eligible; the message names the first
     *                                       disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks or null arguments
     */
    public static TpAttentionPlan context(ModelSpec spec, int ranks, GpuWeightUploader uploader) {
        return plan(spec, ranks, uploader, true);
    }

    private static TpAttentionPlan plan(ModelSpec spec, int ranks, GpuWeightUploader uploader, boolean context) {
        if (ranks < 2) {
            throw new IllegalArgumentException((context ? "a context split" : "a tensor split")
                    + " needs at least two ranks, got " + ranks);
        }
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(uploader, "uploader");
        List<Layer> layers = new ArrayList<>();
        Map<String, Layer> byLayer = new LinkedHashMap<>();
        List<String> skipped = new ArrayList<>();
        String firstDisqualifier = null;
        for (LayerSpec l : spec.layers()) {
            if (!bearsAttention(l)) {
                // a recurrent, convolutional or memoryless mixer keeps its whole-device block on the main device
                continue;
            }
            String reason = ineligible(l);
            if (reason == null && context) {
                reason = contextIneligible(l);
            }
            if (reason == null && !context) {
                reason = divides(l, ranks);
            }
            if (reason == null && !context) {
                try {
                    reason = transforms(l, uploader);
                } catch (UnsupportedOperationException e) {
                    reason = e.getMessage();
                }
            }
            if (reason != null) {
                String named = describe(l) + ": " + reason;
                if (firstDisqualifier == null) {
                    firstDisqualifier = named;
                }
                if (context) {
                    // the layer keeps its whole-device mixer and cache on the main device
                    skipped.add(named);
                    continue;
                }
                // a whole attention layer would need a full cache on the main device beside the per-rank shards
                throw new UnsupportedOperationException("split-mode tensor cannot split the attention of "
                        + spec.arch().ggufName() + ": " + named);
            }
            if (byLayer.containsKey(layerKey(l))) {
                // a loop slot reuses its physical layer's tensors; one split serves every slot
                continue;
            }
            Layer layer = layer(l, ranks, uploader, spec.arch(), context);
            if (context) {
                String blocked = contextTransform(layer, uploader);
                if (blocked != null) {
                    if (firstDisqualifier == null) {
                        firstDisqualifier = describe(l) + ": " + blocked;
                    }
                    skipped.add(describe(l) + ": " + blocked);
                    continue;
                }
            }
            layers.add(layer);
            byLayer.put(layerKey(l), layer);
        }
        if (layers.isEmpty()) {
            throw new UnsupportedOperationException((context
                    ? "split-mode context divides the context positions between the ranks, but every "
                    : "split-mode tensor shards the attention projections over the ranks, but every ")
                    + spec.arch().ggufName() + " layer is ineligible"
                    + (firstDisqualifier == null ? "" : "; the first disqualifier is " + firstDisqualifier));
        }
        return new TpAttentionPlan(ranks, layers, byLayer, byTensor(layers), context, skipped);
    }

    private static String layerKey(LayerSpec l) {
        return l.stack() + ":" + l.weightLayer();
    }

    private static String describe(LayerSpec l) {
        return l.stack() + "." + l.il();
    }

    /** True when the layer's mixer reads or writes an attention cache, so it must join the split or be refused. */
    private static boolean bearsAttention(LayerSpec l) {
        switch (l.mixerKind()) {
            case ATTENTION:
            case MLA:
            case DSA_MLA:
            case MSA:
            case DSV4:
            case CROSS_ATTENTION_T5:
                return true;
            default:
                return false;
        }
    }

    /** Why a layer's attention cannot run as a context split, or null when it can. */
    private static String contextIneligible(LayerSpec l) {
        AttentionSpec a = l.attention();
        if (a == null) {
            return "the layer has no attention spec";
        }
        if (a.sinks() || a.sinkTokens() > 0) {
            /* Every rank would add the shared sink logit to its own denominator, counting it once per rank. */
            return "attention sinks are counted once per rank by the log-sum-exp merge; the layer keeps its"
                    + " whole-device attention";
        }
        if (a.dual() || a.dualChunk() != null) {
            return "dual-chunk or dual attention does not expose one partial softmax per context block";
        }
        if (a.effectiveSwaType() != SwaType.NONE) {
            /* vLLM refuses DCP over a sliding window: the window's rows are not a fixed context block of the
             * sequence, so a rank shard would have to follow every eviction across the ranks. */
            return "sliding-window attention has no fixed context block to shard; the layer keeps its whole-device"
                    + " attention";
        }
        return null;
    }

    /** Why an input transform blocks the layer from the context split, or null when none does. */
    private static String contextTransform(Layer layer, GpuWeightUploader uploader) {
        InputTransformLoader.Result transforms = uploader.transforms();
        if (transforms == null || transforms.isEmpty()) {
            return null;
        }
        for (String suffix : layer.tensors().keySet()) {
            InputTransform t = transforms.forward(layer.tensor(suffix));
            if (t != null && !t.isIdentity()) {
                return "the PRISM input transform of " + layer.tensor(suffix) + " is attached to the resident"
                        + " upload, which a context split does not reproduce";
            }
        }
        return null;
    }

    /** Why a layer's attention cannot shard, or null when it can. */
    private static String ineligible(LayerSpec l) {
        if (l.mla() != null || l.mixerKind() == MixerKind.MLA || l.mixerKind() == MixerKind.DSA_MLA) {
            return "MLA attention keeps its latent cache and head projection whole; it is not tensor-split";
        }
        if (l.mixerKind() != MixerKind.ATTENTION) {
            if (l.mixerKind() == MixerKind.NONE) {
                return null;
            }
            return l.mixerKind() + " has no column-parallel attention split";
        }
        AttentionSpec a = l.attention();
        if (a == null) {
            return "the layer has no attention spec";
        }
        if (a.linearReplace()) {
            return "the linear replacement has no attention heads to split";
        }
        if (!a.usesMemory()) {
            return "only cached (causal) attention splits; the layer attends to the current batch alone";
        }
        if (a.sharesKv(l.il())) {
            return "the layer reads layer " + a.kvOwnerLayer() + "'s KV cache; a shared cache is not split";
        }
        if (a.valueExperts() != null) {
            return "routed value experts are not tensor-split";
        }
        if (a.valueEmbeddings()) {
            return "value embeddings are looked up by token id and are not tensor-split";
        }
        if (a.qLatentDim() > 0) {
            return "the low-rank query projection is not tensor-split";
        }
        if (a.xsaEps() > 0f) {
            return "exclusive self attention is not tensor-split";
        }
        if (a.valueScale() != 0f) {
            return "f_attn_value_scale scales the summed branch output and is not tensor-split";
        }
        QkNormKind norm = a.qkNorm();
        if (norm == QkNormKind.RMS_FULL || norm == QkNormKind.LN_FULL) {
            return "the " + norm + " Q/K norm normalizes across every head and cannot be split";
        }
        if (a.gate() == AttnGateKind.Q_INTERLEAVED && a.headDimK() != a.headDimV()) {
            return "a Q-interleaved gate needs equal Q and V head dims";
        }
        if (!TensorNames.STACK_BLK.equals(l.stack())) {
            return "only the main trunk splits; the " + l.stack() + " stack keeps its attention whole";
        }
        if (l.rope() != null && l.rope().positionsPerToken() > 1) {
            return "several positions per token (M-RoPE) are not tensor-split";
        }
        return null;
    }

    /** Why the head counts cannot divide over the ranks, or null when they do. */
    private static String divides(LayerSpec l, int ranks) {
        AttentionSpec a = l.attention();
        if (a.nHead() % ranks != 0) {
            return "n_head " + a.nHead() + " does not divide into " + ranks + " rank(s)";
        }
        if (a.nHeadKv() % ranks != 0) {
            return "n_head_kv " + a.nHeadKv() + " does not divide into " + ranks + " rank(s)"
                    + (a.nHeadKv() < ranks ? " (GQA/MQA has fewer KV heads than ranks)" : "");
        }
        return null;
    }

    /** Why a PRISM/Hadamard transform blocks the split, or null when none does. */
    private static String transforms(LayerSpec l, GpuWeightUploader uploader) {
        InputTransformLoader.Result transforms = uploader.transforms();
        if (transforms == null || transforms.isEmpty()) {
            return null;
        }
        String out = TensorNames.layerName(l.stack(), l.weightLayer(), TensorNames.weight(TensorNames.ATTN_OUT));
        InputTransform t = transforms.forward(out);
        if (t != null && !t.isIdentity()) {
            return "the PRISM input transform of " + out + " is built for the full attention width";
        }
        if (transforms.hadamard(out) != null) {
            return "the Hadamard rotation of " + out + " is built for the full attention width";
        }
        return null;
    }

    /** The split attention of one eligible layer. */
    private static Layer layer(LayerSpec l, int ranks, GpuWeightUploader uploader,
                               com.eonmux.neuralfree.gguf.arch.Arch arch, boolean context) {
        AttentionSpec a = l.attention();
        QkvProjection proj = QkvProjection.of(l, suffix -> plan(l, suffix, uploader) != null, arch);
        Map<String, Shard> tensors = new LinkedHashMap<>();
        boolean fused = proj.source() == QkvProjection.Source.FUSED;
        if (fused) {
            requireRows(l, proj.qkvWeight(), proj.fusedWidth(), ranks, uploader, context);
            put(tensors, proj.qkvWeight(), Shard.FUSED_ROWS, uploader, l);
            put(tensors, proj.qkvBias(), Shard.FUSED_ROWS, uploader, l);
            put(tensors, proj.qBias(), Shard.ROWS, uploader, l);
            put(tensors, proj.kBias(), Shard.ROWS, uploader, l);
            put(tensors, proj.vBias(), Shard.ROWS, uploader, l);
        } else if (proj.source() == QkvProjection.Source.SEPARATE) {
            requireRows(l, proj.qWeight(), proj.qProjWidth(), ranks, uploader, context);
            put(tensors, proj.qWeight(), Shard.ROWS, uploader, l);
            put(tensors, proj.qBias(), Shard.ROWS, uploader, l);
            if (proj.projectsKv()) {
                requireRows(l, proj.kWeight(), proj.kDim(), ranks, uploader, context);
                put(tensors, proj.kWeight(), Shard.ROWS, uploader, l);
                put(tensors, proj.kBias(), Shard.ROWS, uploader, l);
                if (!proj.vFromK()) {
                    requireRows(l, proj.vWeight(), proj.vDim(), ranks, uploader, context);
                    put(tensors, proj.vWeight(), Shard.ROWS, uploader, l);
                    put(tensors, proj.vBias(), Shard.ROWS, uploader, l);
                }
            }
        }
        normShard(l, TensorNames.weight(TensorNames.ATTN_Q_NORM), a.qkNorm(), a.nHead(), a.headDimK(), ranks,
                uploader, tensors, context);
        normShard(l, TensorNames.bias(TensorNames.ATTN_Q_NORM), a.qkNorm(), a.nHead(), a.headDimK(), ranks,
                uploader, tensors, context);
        if (proj.projectsKv()) {
            normShard(l, TensorNames.weight(TensorNames.ATTN_K_NORM), a.qkNorm(), a.nHeadKv(), a.headDimK(), ranks,
                    uploader, tensors, context);
            normShard(l, TensorNames.bias(TensorNames.ATTN_K_NORM), a.qkNorm(), a.nHeadKv(), a.headDimK(), ranks,
                    uploader, tensors, context);
        }
        if (proj.gateWeight() != null) {
            int gateWidth = gateWidth(a, proj);
            requireRows(l, proj.gateWeight(), gateWidth, ranks, uploader, context);
            put(tensors, proj.gateWeight(), Shard.ROWS, uploader, l);
            put(tensors, proj.gateBias(), Shard.PER_HEAD, uploader, l);
        }
        if (a.sinks()) {
            put(tensors, TensorNames.weight(TensorNames.ATTN_SINKS), Shard.PER_HEAD, uploader, l);
        }
        GpuWeightUploader.TensorPlan out = plan(l, proj.outWeight(), uploader);
        if (out == null || out.ne().length < 2 || out.ne()[0] != proj.outDim()) {
            throw new UnsupportedOperationException(describe(l) + ": " + proj.outWeight() + " is not the ["
                    + proj.outDim() + " x n_embd] output projection");
        }
        if (!context) {
            int outColumns = (a.nHead() / ranks) * a.headDimV();
            if (!out.needsDecode()) {
                long block = out.deviceType().getBlockSize();
                if (outColumns % block != 0) {
                    throw new UnsupportedOperationException(describe(l) + ": the rank's " + outColumns
                            + " output columns of " + out.name() + " are not whole " + out.deviceType()
                            + " blocks of " + block + " element(s)");
                }
            }
        }
        tensors.put(proj.outWeight(), Shard.OUT_COLUMNS);
        put(tensors, proj.outBias(), Shard.OUT_BIAS, uploader, l);
        if (context) {
            /* Every rank runs the whole attention, so every tensor of the layer uploads whole to every rank. */
            Map<String, Shard> whole = new LinkedHashMap<>();
            for (String suffix : tensors.keySet()) {
                whole.put(suffix, Shard.WHOLE);
            }
            tensors = whole;
        }
        return new Layer(l.stack(), l.weightLayer(), l.il(), a.nHead(), a.nHeadKv(), a.headDimK(), a.headDimV(),
                context ? a.nHead() : a.nHead() / ranks, context ? a.nHeadKv() : a.nHeadKv() / ranks,
                proj.qProjWidth(), proj.kDim(), proj.vDim(), proj.outDim(), fused, tensors);
    }

    /** The gate width of {@code layer}'s output gate. */
    private static int gateWidth(AttentionSpec a, QkvProjection proj) {
        switch (a.gate()) {
            case ELEMENTWISE_SIGMOID:
            case ELEMENTWISE_SOFTPLUS:
            case SOFTPLUS_BASE2:
                return proj.outDim();
            case PER_HEAD_SIGMOID:
            case PER_HEAD_SIGMOID_BIAS:
            case PER_HEAD_SOFTPLUS:
                return a.nHead();
            default:
                return a.nHead();
        }
    }

    /** Adds a per-head Q/K norm tensor when the file has it and it tiles head rows. */
    private static void normShard(LayerSpec l, String suffix, QkNormKind kind, int heads, int headDim, int ranks,
                                  GpuWeightUploader uploader, Map<String, Shard> tensors, boolean context) {
        GpuWeightUploader.TensorPlan p = plan(l, suffix, uploader);
        if (p == null) {
            return;
        }
        if (context) {
            /* Every rank binds the whole norm; its shape is the model's own. */
            tensors.put(suffix, Shard.WHOLE);
            return;
        }
        if (p.deviceType() != GGMLType.F32) {
            throw new UnsupportedOperationException(describe(l) + ": " + p.name() + " is " + p.deviceType()
                    + "; a split Q/K norm needs an F32 tensor");
        }
        long total = p.info().numElements();
        if (total == heads) {
            tensors.put(suffix, Shard.PER_HEAD);
            return;
        }
        if (p.ne().length >= 2 && p.ne()[1] == heads && total == (long) heads * headDim) {
            tensors.put(suffix, Shard.NORM_HEADS);
            return;
        }
        if (total == headDim) {
            // one row shared by every head: identical on every rank
            tensors.put(suffix, Shard.WHOLE);
            return;
        }
        throw new UnsupportedOperationException(describe(l) + ": " + p.name() + " holds " + total
                + " values, which does not tile " + heads + " heads of " + headDim);
    }

    private static void requireRows(LayerSpec l, String suffix, long rows, int ranks,
                                    GpuWeightUploader uploader, boolean context) {
        GpuWeightUploader.TensorPlan p = plan(l, suffix, uploader);
        if (p == null) {
            throw new UnsupportedOperationException(describe(l) + ": the model has no tensor "
                    + TensorNames.layerName(l.stack(), l.weightLayer(), suffix));
        }
        if (p.ne().length < 2 || p.ne()[1] != rows) {
            throw new UnsupportedOperationException(describe(l) + ": " + p.name() + " has "
                    + (p.ne().length < 2 ? "no rows" : p.ne()[1] + " rows") + ", expected " + rows);
        }
        if (!context && rows % ranks != 0) {
            throw new UnsupportedOperationException(describe(l) + ": " + p.name() + " has " + rows
                    + " rows, which does not divide into " + ranks + " rank(s)");
        }
    }

    private static void put(Map<String, Shard> tensors, String suffix, Shard shard, GpuWeightUploader uploader,
                            LayerSpec l) {
        if (suffix != null && plan(l, suffix, uploader) != null) {
            tensors.put(suffix, shard);
        }
    }

    private static GpuWeightUploader.TensorPlan plan(LayerSpec l, String suffix, GpuWeightUploader uploader) {
        return uploader.plan(TensorNames.layerName(l.stack(), l.weightLayer(), suffix));
    }

    private static Map<String, Layer> byTensor(List<Layer> layers) {
        Map<String, Layer> out = new LinkedHashMap<>();
        for (Layer layer : layers) {
            for (String suffix : layer.tensors.keySet()) {
                out.put(layer.tensor(suffix), layer);
            }
        }
        return Collections.unmodifiableMap(out);
    }
}
