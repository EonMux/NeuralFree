package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Reassembles one whole model from the per-rank state dicts of vLLM's tensor- and pipeline-parallel workers.
 *
 * <p>vLLM's {@code ShardedStateLoader} and its sharded {@code tensorizer} format save each worker's own state dict:
 * the tensors a column-parallel linear owns along its output dimension, a row-parallel linear along its input
 * dimension, a vocabulary-parallel embedding or head as its own row block (padded to the vocabulary's padded size),
 * and replicated parameters once per worker. A fused {@code QKVParallelLinear}/{@code MergedColumnParallelLinear}
 * packs its shards per rank (q then k then v, gate then up), so each rank's parts are separated before the parts are
 * concatenated and joined again; {@code FusedMoE}'s {@code w13} packs gate then up along dim 1 the same way, while
 * its {@code w2} is row-parallel along the last dimension. Vocabulary blocks are concatenated and trimmed to the
 * configured vocabulary size, and replicated parameters are taken once after checking that every worker holds the
 * same bytes. Pipeline stages hold disjoint parameters, so their results are merged; a name that repeats across
 * stages must be identical.</p>
 *
 * <p>A parameter whose name this class does not recognize is taken as replicated, and unequal copies are refused by
 * name: an unknown sharded parameter is never silently misread. Quantized FusedMoE siblings are refused by name.</p>
 */
public final class VllmShardAssembly {

    /** How a vLLM layer type partitions a parameter. */
    enum Shard {
        /** A fused q/k/v projection, packed per rank as q, then k, then v. */
        QKV,
        /** A fused gate/up projection, packed per rank as gate, then up. */
        GATE_UP,
        COLUMN,
        ROW,
        VOCAB,
        EXPERT_W13,
        EXPERT_W2,
        REPLICATED
    }

    private static final int COMPARE_CHUNK_BYTES = 4 << 20;

    /** One worker's tensors: its pipeline stage, its tensor-parallel rank and the tensors by name. */
    public static final class Rank {
        private final int pipeline;
        private final int tensor;
        private final LinkedHashMap<String, StoredTensor> tensors;

        /**
         * @param pipeline the pipeline-parallel rank
         * @param tensor   the tensor-parallel rank
         * @param tensors  the rank's tensors in checkpoint order
         * @throws IllegalArgumentException when a rank is negative or a name appears twice
         */
        public Rank(int pipeline, int tensor, Map<String, StoredTensor> tensors) {
            if (pipeline < 0 || tensor < 0) {
                throw new IllegalArgumentException("a worker rank cannot be negative: pp=" + pipeline + ", tp="
                        + tensor);
            }
            this.pipeline = pipeline;
            this.tensor = tensor;
            this.tensors = new LinkedHashMap<>();
            for (Map.Entry<String, StoredTensor> e : Objects.requireNonNull(tensors, "tensors").entrySet()) {
                if (this.tensors.put(e.getKey(), e.getValue()) != null) {
                    throw new IllegalArgumentException("tensor '" + e.getKey() + "' appears twice in rank pp="
                            + pipeline + ", tp=" + tensor);
                }
            }
        }

        public int pipeline() {
            return pipeline;
        }

        public int tensor() {
            return tensor;
        }

        public LinkedHashMap<String, StoredTensor> tensors() {
            return tensors;
        }
    }

    private VllmShardAssembly() {
    }

    /**
     * The whole model of {@code ranks}.
     *
     * @param description what is being assembled, for messages
     * @param config      the checkpoint configuration, which fixes the vocabulary size
     * @throws IOException when the ranks do not form complete pipeline/tensor grids, a worker misses a tensor its
     *                     peers hold, a replicated tensor differs between workers, or a vocabulary block is short
     */
    public static List<StoredTensor> assemble(List<Rank> ranks, HfConfig config, String description)
            throws IOException {
        Objects.requireNonNull(ranks, "ranks");
        Objects.requireNonNull(config, "config");
        if (ranks.isEmpty()) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': it holds no worker state dict");
        }
        Map<Integer, Map<Integer, Rank>> grid = grid(ranks, description);
        int pipelineStages = grid.size();
        LinkedHashMap<String, StoredTensor> out = new LinkedHashMap<>();
        for (int pp = 0; pp < pipelineStages; pp++) {
            List<Rank> stage = new ArrayList<>(grid.get(pp).values());
            stage.sort((a, b) -> Integer.compare(a.tensor(), b.tensor()));
            Map<String, StoredTensor> merged = assembleStage(stage, config, description, pp);
            for (Map.Entry<String, StoredTensor> e : merged.entrySet()) {
                StoredTensor prev = out.putIfAbsent(e.getKey(), e.getValue());
                if (prev != null) {
                    if (!sameBytes(prev.source(), e.getValue().source())) {
                        throw new IOException("Invalid vLLM checkpoint '" + description + "': pipeline stages hold"
                                + " different copies of tensor '" + e.getKey() + "'");
                    }
                }
            }
        }
        return new ArrayList<>(out.values());
    }

    /** The ranks by pipeline stage and tensor-parallel rank; every grid is complete and the same size. */
    private static Map<Integer, Map<Integer, Rank>> grid(List<Rank> ranks, String description) throws IOException {
        Map<Integer, Map<Integer, Rank>> grid = new java.util.TreeMap<>();
        for (Rank r : ranks) {
            Map<Integer, Rank> stage = grid.computeIfAbsent(r.pipeline(), k -> new java.util.TreeMap<>());
            if (stage.put(r.tensor(), r) != null) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': rank pp=" + r.pipeline()
                        + ", tp=" + r.tensor() + " appears twice");
            }
        }
        int stages = grid.size();
        int width = -1;
        for (int pp = 0; pp < stages; pp++) {
            Map<Integer, Rank> stage = grid.get(pp);
            if (stage == null) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': pipeline stage " + pp
                        + " is missing (found stages 0.." + (stages - 1) + ")");
            }
            if (width == -1) {
                width = stage.size();
            }
            if (stage.size() != width) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': pipeline stage " + pp
                        + " holds " + stage.size() + " tensor-parallel ranks, not " + width);
            }
            for (int tp = 0; tp < width; tp++) {
                if (!stage.containsKey(tp)) {
                    throw new IOException("Invalid vLLM checkpoint '" + description + "': tensor-parallel rank "
                            + tp + " of pipeline stage " + pp + " is missing");
                }
            }
        }
        return grid;
    }

    /** One pipeline stage's tensors, reassembled across its tensor-parallel ranks. */
    private static Map<String, StoredTensor> assembleStage(List<Rank> stage, HfConfig config, String description,
                                                           int pp) throws IOException {
        LinkedHashMap<String, StoredTensor> out = new LinkedHashMap<>();
        List<String> names = new ArrayList<>(stage.get(0).tensors().keySet());
        for (int i = 1; i < stage.size(); i++) {
            Rank rank = stage.get(i);
            if (rank.tensors().size() != names.size() || !rank.tensors().keySet().containsAll(names)) {
                for (String name : names) {
                    if (!rank.tensors().containsKey(name)) {
                        throw new IOException("Invalid vLLM checkpoint '" + description + "': tensor '" + name
                                + "' is missing from rank pp=" + pp + ", tp=" + rank.tensor());
                    }
                }
                for (String name : rank.tensors().keySet()) {
                    if (!names.contains(name)) {
                        throw new IOException("Invalid vLLM checkpoint '" + description + "': rank pp=" + pp
                                + ", tp=" + rank.tensor() + " holds tensor '" + name + "', which the other ranks"
                                + " do not");
                    }
                }
            }
        }
        for (String name : names) {
            if (isQuantizedExpertSibling(name)) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': tensor '" + name
                        + "' is a quantized FusedMoE sibling, whose packing is not converted");
            }
            List<RowSource> sources = new ArrayList<>(stage.size());
            for (Rank r : stage) {
                StoredTensor t = r.tensors().get(name);
                if (t == null) {
                    throw new IOException("Invalid vLLM checkpoint '" + description + "': tensor '" + name
                            + "' is missing from rank pp=" + pp + ", tp=" + r.tensor());
                }
                sources.add(t.source());
            }
            RowSource reassembled = reassemble(name, sources, config, description);
            out.put(name, StoredTensor.of(name, reassembled));
        }
        return out;
    }

    private static RowSource reassemble(String name, List<RowSource> sources, HfConfig config, String description)
            throws IOException {
        RowSource first = sources.get(0);
        Shard shard = classify(name);
        if (sources.size() == 1) {
            return shard == Shard.VOCAB ? trimVocab(name, first, config, description) : first;
        }
        switch (shard) {
            case QKV:
                return reassembleQkv(name, sources, config, description);
            case GATE_UP:
                return reassembleGateUp(name, sources, description);
            case COLUMN:
                return RowTransforms.concatDim(sources, 0);
            case ROW:
                if (first.shape().length < 2) {
                    throw new IOException("Invalid vLLM checkpoint '" + description + "': row-parallel tensor '"
                            + name + "' is rank " + first.shape().length);
                }
                return RowTransforms.concatDim(sources, 1);
            case VOCAB:
                return trimVocab(name, RowTransforms.concatDim(sources, 0), config, description);
            case EXPERT_W13:
                return reassembleExpertGateUp(name, sources, config, description);
            case EXPERT_W2:
                checkExpertCount(name, first.shape(), config, description);
                return RowTransforms.concatDim(sources, first.shape().length - 1);
            default:
                return requireIdentical(name, sources, description);
        }
    }

    /**
     * A fused q/k/v projection: every rank holds its own q rows, then its k rows, then its v rows, so the whole
     * tensor is each part concatenated across the ranks before q, k and v are joined.
     *
     * <p>vLLM's {@code QKVParallelLinear} may replicate key/value heads when there are fewer of them than
     * tensor-parallel ranks ({@code num_kv_head_replicas}); then only one rank of each replica group holds the head
     * and the others repeat it. A fused module that vLLM does not shard (its rows are the whole projection on every
     * rank) is taken once after checking the ranks agree.</p>
     */
    private static RowSource reassembleQkv(String name, List<RowSource> sources, HfConfig config, String description)
            throws IOException {
        RowSource first = sources.get(0);
        long[] shape = first.shape();
        if (shape.length < 1 || shape.length > 2) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': fused tensor '" + name
                    + "' is rank " + shape.length + ", not 1 or 2");
        }
        long[] rows = VllmStateDict.qkvRows(config, name);
        if (rows[1] != rows[2]) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': fused tensor '" + name
                    + "' has different key and value widths (" + rows[1] + " and " + rows[2] + "), which this"
                    + " reassembly does not cover");
        }
        long full = Math.addExact(Math.addExact(rows[0], rows[1]), rows[2]);
        long fused = shape[0];
        if (fused == full) {
            return requireIdentical(name, sources, description);
        }
        int tp = sources.size();
        if (rows[0] % tp != 0) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': the " + rows[0] + " query rows of '"
                    + name + "' do not divide across " + tp + " tensor-parallel ranks");
        }
        long query = rows[0] / tp;
        boolean replicatedKv;
        long kv;
        if (rows[1] % tp == 0 && fused == query + 2 * (rows[1] / tp)) {
            kv = rows[1] / tp;
            replicatedKv = false;
        } else {
            long local = fused - query;
            if (local <= 0 || local % 2 != 0) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': fused tensor '" + name
                        + "' holds " + fused + " rows, not the " + full + " of the configured q, k and v heads,"
                        + " nor a tensor-parallel shard of them");
            }
            kv = local / 2;
            long kvHeads = rows[1] / kv;
            if (rows[1] % kv != 0 || kvHeads <= 0 || kvHeads > tp || tp % kvHeads != 0) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': fused tensor '" + name
                        + "' holds " + fused + " rows, which is neither an even q/k/v partition nor a replica of"
                        + " the configured key/value heads");
            }
            replicatedKv = true;
        }
        for (RowSource s : sources) {
            if (s.shape().length != shape.length || s.shape()[0] != fused) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': the tensor-parallel shards of '"
                        + name + "' have different shapes");
            }
        }
        long replicas = replicatedKv ? tp / (rows[1] / kv) : 1;
        List<RowSource> queries = new ArrayList<>(tp);
        List<RowSource> keys = new ArrayList<>();
        List<RowSource> values = new ArrayList<>();
        for (int i = 0; i < tp; i++) {
            RowSource s = sources.get(i);
            queries.add(RowTransforms.sliceDim(s, 0, 0, query));
            if (!replicatedKv || i % replicas == 0) {
                keys.add(RowTransforms.sliceDim(s, 0, query, kv));
                values.add(RowTransforms.sliceDim(s, 0, query + kv, kv));
            }
        }
        return RowTransforms.concatDim(Arrays.asList(RowTransforms.concatDim(queries, 0),
                RowTransforms.concatDim(keys, 0), RowTransforms.concatDim(values, 0)), 0);
    }

    /**
     * A fused gate/up projection: every rank holds its own gate rows followed by its own up rows. The full tensor is
     * the gate halves joined, then the up halves joined. A module vLLM does not shard holds the whole projection on
     * every rank and is taken once.
     */
    private static RowSource reassembleGateUp(String name, List<RowSource> sources, String description)
            throws IOException {
        RowSource first = sources.get(0);
        long[] shape = first.shape();
        if (shape.length < 1 || shape.length > 2) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': fused tensor '" + name
                    + "' is rank " + shape.length + ", not 1 or 2");
        }
        if (identical(sources)) {
            return first;
        }
        long rows = shape[0];
        if (rows <= 0 || rows % 2 != 0) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': the fused gate/up output dimension "
                    + rows + " of '" + name + "' is not two equal halves");
        }
        long half = rows / 2;
        List<RowSource> gates = new ArrayList<>(sources.size());
        List<RowSource> ups = new ArrayList<>(sources.size());
        for (RowSource s : sources) {
            if (s.shape().length != shape.length || s.shape()[0] != rows) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': the tensor-parallel shards of '"
                        + name + "' have different shapes");
            }
            gates.add(RowTransforms.sliceDim(s, 0, 0, half));
            ups.add(RowTransforms.sliceDim(s, 0, half, half));
        }
        return RowTransforms.concatDim(Arrays.asList(RowTransforms.concatDim(gates, 0),
                RowTransforms.concatDim(ups, 0)), 0);
    }

    /**
     * Packed {@code FusedMoE.w13}: every rank holds its gate rows followed by its up rows along dim 1, so the whole
     * stack is the gate halves joined, then the up halves joined.
     */
    private static RowSource reassembleExpertGateUp(String name, List<RowSource> sources, HfConfig config,
                                                    String description) throws IOException {
        RowSource first = sources.get(0);
        long[] shape = first.shape();
        if (shape.length < 2 || shape.length > 3) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': packed expert tensor '" + name
                    + "' is rank " + shape.length);
        }
        checkExpertCount(name, shape, config, description);
        long width = shape[1];
        if (width <= 0 || width % 2 != 0) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': the packed gate/up width " + width
                    + " of '" + name + "' is not two equal halves");
        }
        long half = width / 2;
        List<RowSource> gates = new ArrayList<>(sources.size());
        List<RowSource> ups = new ArrayList<>(sources.size());
        for (RowSource s : sources) {
            long[] other = s.shape();
            if (other.length != shape.length || other[0] != shape[0] || other[1] != width) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': the tensor-parallel shards of '"
                        + name + "' have different shapes");
            }
            gates.add(RowTransforms.sliceDim(s, 1, 0, half));
            ups.add(RowTransforms.sliceDim(s, 1, half, half));
        }
        return RowTransforms.concatDim(Arrays.asList(RowTransforms.concatDim(gates, 1),
                RowTransforms.concatDim(ups, 1)), 1);
    }

    /** Refuses an expert-parallel shard: fewer experts than the configuration declares cannot be reassembled. */
    private static void checkExpertCount(String name, long[] shape, HfConfig config, String description)
            throws IOException {
        if (shape.length < 1) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': packed expert tensor '" + name
                    + "' is a scalar");
        }
        int experts = config.findInt(-1, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts");
        if (experts >= 0 && shape[0] != experts) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': packed expert tensor '" + name
                    + "' holds " + shape[0] + " experts, the configuration declares " + experts
                    + "; an expert-parallel shard cannot be reassembled from tensor-parallel ranks");
        }
    }

    /** The first source after checking that every rank holds the same bytes. */
    private static RowSource requireIdentical(String name, List<RowSource> sources, String description)
            throws IOException {
        RowSource first = sources.get(0);
        for (RowSource s : sources) {
            if (!sameBytes(first, s)) {
                throw new IOException("Invalid vLLM checkpoint '" + description + "': tensor '" + name
                        + "' is replicated, but the tensor-parallel ranks hold different bytes");
            }
        }
        return first;
    }

    /** Whether every source holds the same bytes. */
    private static boolean identical(List<RowSource> sources) throws IOException {
        RowSource first = sources.get(0);
        for (int i = 1; i < sources.size(); i++) {
            if (!sameBytes(first, sources.get(i))) {
                return false;
            }
        }
        return true;
    }

    /** Trims a vocabulary block to the configured vocabulary size, refusing a block that is too short. */
    private static RowSource trimVocab(String name, RowSource source, HfConfig config, String description)
            throws IOException {
        int vocab = config.findInt(-1, "vocab_size");
        if (vocab <= 0) {
            return source;
        }
        long rows = source.shape()[0];
        if (rows == vocab) {
            return source;
        }
        if (rows < vocab) {
            throw new IOException("Invalid vLLM checkpoint '" + description + "': vocabulary tensor '" + name
                    + "' holds " + rows + " rows, fewer than the configured vocab_size " + vocab);
        }
        return RowTransforms.truncateRows(source, vocab);
    }

    /** How {@code name}'s vLLM layer type partitions it. */
    static Shard classify(String name) {
        Objects.requireNonNull(name, "name");
        if (isVocab(name)) {
            return Shard.VOCAB;
        }
        if (name.endsWith(".w13_weight") || name.endsWith(".w13_bias")) {
            return Shard.EXPERT_W13;
        }
        if (name.endsWith(".w2_weight")) {
            return Shard.EXPERT_W2;
        }
        if (isQkv(name)) {
            return Shard.QKV;
        }
        if (isGateUp(name)) {
            return Shard.GATE_UP;
        }
        if (isColumn(name)) {
            return Shard.COLUMN;
        }
        if (isRow(name)) {
            return Shard.ROW;
        }
        return Shard.REPLICATED;
    }

    private static boolean isVocab(String name) {
        return name.endsWith(".embed_tokens.weight") || name.endsWith(".lm_head.weight")
                || name.endsWith(".wte.weight") || name.endsWith(".word_embeddings.weight")
                || name.equals("lm_head.weight") || name.equals("embed_tokens.weight");
    }

    /** A fused q/k/v projection ({@code QKVParallelLinear}, or a merged column linear of the same three shards). */
    private static boolean isQkv(String name) {
        return name.endsWith(".qkv_proj.weight") || name.endsWith(".qkv_proj.bias");
    }

    /** A fused gate/up projection ({@code MergedColumnParallelLinear}). */
    private static boolean isGateUp(String name) {
        return name.endsWith(".gate_up_proj.weight") || name.endsWith(".gate_up_proj.bias");
    }

    private static boolean isColumn(String name) {
        for (String suffix : new String[]{
                ".q_proj.weight", ".q_proj.bias", ".k_proj.weight", ".k_proj.bias", ".v_proj.weight",
                ".v_proj.bias", ".gate_proj.weight", ".gate_proj.bias", ".up_proj.weight", ".up_proj.bias",
                ".w1.weight", ".w3.weight"}) {
            if (name.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isRow(String name) {
        for (String suffix : new String[]{
                ".o_proj.weight", ".down_proj.weight", ".out_proj.weight", ".linear_proj.weight",
                ".dense_output.weight", ".shared_expert.down_proj.weight"}) {
            if (name.endsWith(suffix)) {
                return true;
            }
        }
        return false;
    }

    private static boolean isQuantizedExpertSibling(String name) {
        return name.contains(".w13_weight_") || name.contains(".w2_weight_");
    }

    /** Whether two row sources hold the same format, shape and bytes. */
    static boolean sameBytes(RowSource a, RowSource b) throws IOException {
        if (!a.format().equals(b.format()) || !Arrays.equals(a.shape(), b.shape())) {
            return false;
        }
        long rows = a.rows();
        if (rows != b.rows() || a.rowBytes() != b.rowBytes()) {
            return false;
        }
        int perChunk = (int) Math.max(1, Math.min(Integer.MAX_VALUE, COMPARE_CHUNK_BYTES
                / Math.max(1L, a.rowBytes())));
        for (long first = 0; first < rows; first += perChunk) {
            int count = (int) Math.min(perChunk, rows - first);
            ByteBuffer x = read(a, first, count);
            ByteBuffer y = read(b, first, count);
            if (!x.equals(y)) {
                return false;
            }
        }
        return true;
    }

    private static ByteBuffer read(RowSource source, long first, int count) throws IOException {
        long bytes = Math.multiplyExact(source.rowBytes(), (long) count);
        if (bytes > Integer.MAX_VALUE - 8) {
            throw new IOException("tensor " + source.describe() + ": " + count + " rows exceed one buffer");
        }
        ByteBuffer buffer = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        source.readRows(first, count, buffer);
        buffer.flip();
        return buffer;
    }

    /** The reassembly orders, for messages. */
    public static List<String> names() {
        List<String> out = new ArrayList<>();
        for (Shard s : Shard.values()) {
            out.add(s.name());
        }
        return Collections.unmodifiableList(out);
    }
}
