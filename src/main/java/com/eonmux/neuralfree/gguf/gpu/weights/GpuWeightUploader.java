package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.PrismHadamardLoader;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * Uploads the tensors of a GGUF model to the device and serves them as {@link GpuWeight}s.
 *
 * <p><b>Per-tensor device type.</b> Every tensor is classified by how the forward path consumes it
 * ({@link TensorUse}) and gets its own device type ({@link TensorPlan#deviceType()}): a matmul, lookup or expert
 * weight keeps its stored type when the backend has the kernels for it ({@link TypePolicy}, checked with
 * {@code LlmKernels.has}) and is otherwise dequantized to F32 on the CPU; tensors consumed by element-wise or
 * special kernels (norm weights, biases, scales, convolution and recurrent parameters, LoRA factors, ...) are
 * always F32; integer tables are uploaded as stored. Decisions are made per tensor, never per layer or per weight
 * index, so models whose quantization varies across layers get each tensor's own kernels.</p>
 *
 * <p><b>Row shards.</b> A tensor larger than {@code min(maxAllocSize, maxStorageBufferRange)} is split into row
 * shards of balanced size, each in its own buffer ({@link GpuWeight#sharded}); tensors of rank 3 or more split
 * only on {@code ne[1]} (expert) boundaries. A tensor is never dropped for its size.</p>
 *
 * <p><b>Experts.</b> Expert tensors ({@code *_exps.weight}, {@code *_chexps.weight}) are uploaded whole and stay
 * resident, sharded on expert boundaries.</p>
 *
 * <p><b>Per-tensor transforms.</b> A scalar {@code <base>.scale} tensor is the output scale {@code w_s} of
 * {@code <base>.weight} (llama-model.cpp:1724-1790, applied at llama-graph.cpp:1643-1645) and is
 * attached with {@link GpuWeight#withScale}; the scale tensor itself stays resident as F32. PRISM transforms
 * ({@link PrismHadamardLoader.Result}) are attached with {@link GpuWeight#withHadamard} and
 * {@link GpuWeight#withInverseHadamard}.</p>
 *
 * <p>Uploads run on the given queue in order and copy their host data at enqueue; every upload call ends with the
 * queue finished. Not thread-safe. The uploader owns every buffer it allocates; {@link #close()} releases them.</p>
 */
public final class GpuWeightUploader implements GpuWeights, AutoCloseable {

    private static final Logger LOGGER = LogManager.getLogger(GpuWeightUploader.class);

    /** Default host staging size per upload chunk. */
    public static final int DEFAULT_STAGING_BYTES = 64 * 1024 * 1024;

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final String archName;
    private final Config config;
    private final long maxBufferBytes;
    private final Map<String, TensorPlan> plans;
    private final Map<String, GpuWeight> resident = new LinkedHashMap<>();
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final RowCopier copier;
    private long residentBytes;
    private boolean closed;

    /**
     * Plans every tensor of the model; nothing is uploaded until {@link #uploadResident} or {@link #upload}.
     *
     * @param backend  device the tensors are allocated on
     * @param queue    queue the uploads run on
     * @param tensors  every tensor of the model
     * @param store    tensor data of the model
     * @param archName {@code general.architecture}, used in messages
     * @param config   type policy, expert placement and transforms
     * @throws IllegalStateException when a tensor's shape is invalid for its type; a tensor that cannot be uploaded
     *                               (see {@link TensorPlan#requireUploadable()}) fails only when it is uploaded
     */
    public GpuWeightUploader(ComputeBackend backend, ComputeQueue queue, List<GGUFTensorInfo> tensors,
                             WeightStore store, String archName, Config config) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.queue = Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(store, "store");
        this.archName = archName == null || archName.isEmpty() ? "unknown" : archName;
        this.config = Objects.requireNonNull(config, "config");
        Objects.requireNonNull(tensors, "tensors");
        long deviceMax = deviceMaxBufferBytes(backend);
        this.maxBufferBytes = config.maxBufferBytes > 0 ? Math.min(config.maxBufferBytes, deviceMax) : deviceMax;
        this.plans = planAll(tensors);
        this.copier = new RowCopier(store, config.stagingBytes);
    }

    /** A configuration builder with the given type policy. */
    public static Config.Builder config(TypePolicy policy) {
        return new Config.Builder(policy);
    }

    /**
     * The largest single buffer this uploader allocates on {@code backend}: {@code min(maxAllocSize,
     * maxStorageBufferRange)} over the limits the device reports, or {@link Long#MAX_VALUE} when it reports none.
     */
    public static long deviceMaxBufferBytes(ComputeBackend backend) {
        return WeightShards.maxBufferBytes(backend.capabilities());
    }

    // ------------------------------------------------------------------ GpuWeights

    @Override
    public GpuWeight find(String name) {
        return resident.get(name);
    }

    @Override
    public String archName() {
        return archName;
    }

    // ------------------------------------------------------------------ planning

    /** The upload plan of a tensor, or null when the model has no tensor of that name. */
    public TensorPlan plan(String name) {
        return plans.get(name);
    }

    /** Every tensor plan, in model order. Unmodifiable. */
    public Map<String, TensorPlan> plans() {
        return Collections.unmodifiableMap(plans);
    }

    /** The plans of the tensors {@code selected} accepts, in model order. Unmodifiable. */
    public List<TensorPlan> plans(Predicate<GGUFTensorInfo> selected) {
        Objects.requireNonNull(selected, "selected");
        List<TensorPlan> out = new ArrayList<>();
        for (TensorPlan p : plans.values()) {
            if (selected.test(p.info())) {
                out.add(p);
            }
        }
        return Collections.unmodifiableList(out);
    }

    /** Device bytes of the tensors {@code selected} accepts, as {@link #uploadResident} allocates them. */
    public long plannedBytes(Predicate<GGUFTensorInfo> selected) {
        long bytes = 0;
        for (TensorPlan p : plans(selected)) {
            bytes = Math.addExact(bytes, p.deviceBytes());
        }
        return bytes;
    }

    /**
     * The device type of a tensor: its stored type, or F32 when it is dequantized on upload.
     *
     * @throws IllegalArgumentException when the model has no tensor of that name
     */
    public GGMLType deviceType(String name) {
        return requirePlan(name).deviceType();
    }

    /** Largest single buffer this uploader allocates. */
    public long maxBufferBytes() {
        return maxBufferBytes;
    }

    // ------------------------------------------------------------------ upload

    /**
     * Uploads every tensor that is not deferred, then finishes the queue.
     *
     * @param deferred tensors to leave for a later {@link #upload(String)} or {@code uploadResident} call, for another
     *                 owner (streamed layers), or off the device ({@link WeightPlacement})
     * @throws IllegalStateException         when an allocation or upload fails; the message names the tensor and sizes
     * @throws UnsupportedOperationException when a tensor to upload cannot be uploaded
     *                                       ({@link TensorPlan#requireUploadable()})
     */
    public void uploadResident(Predicate<GGUFTensorInfo> deferred) {
        Objects.requireNonNull(deferred, "deferred");
        ensureOpen();
        int count = 0;
        int dequantized = 0;
        int sharded = 0;
        long bytes = 0;
        for (TensorPlan p : plans.values()) {
            if (resident.containsKey(p.name()) || deferred.test(p.info())) {
                continue;
            }
            GpuWeight w = uploadPlan(p);
            count++;
            bytes += p.deviceBytes();
            if (p.needsDecode()) {
                dequantized++;
            }
            if (!w.isSingleShard()) {
                sharded++;
            }
        }
        finishQueue();
        attachScales();
        copier.release();
        LOGGER.info("{}: uploaded {} tensors ({} MB), {} dequantized to F32, {} split into row shards"
                        + " (largest buffer {} MB)", archName, count, bytes >> 20, dequantized, sharded,
                maxBufferBytes == Long.MAX_VALUE ? "unbounded" : String.valueOf(maxBufferBytes >> 20));
    }

    /**
     * Uploads one tensor now (a no-op when it is already resident), then finishes the queue.
     *
     * @return the resident tensor
     * @throws IllegalArgumentException when the model has no such tensor
     * @throws IllegalStateException    when the allocation or upload fails
     */
    public GpuWeight upload(String name) {
        ensureOpen();
        TensorPlan p = requirePlan(name);
        GpuWeight w = resident.get(name);
        if (w == null) {
            uploadPlan(p);
            finishQueue();
            attachScales();
            copier.release();
            w = resident.get(name);
        }
        return w;
    }

    /** Device bytes of the resident tensors. */
    public long residentBytes() {
        return residentBytes;
    }

    /** Number of resident tensors. */
    public int residentCount() {
        return resident.size();
    }

    // ------------------------------------------------------------------ lifecycle

    /** Releases every buffer this uploader allocated and the host staging. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = Closeables.closeAll(owned, null);
        owned.clear();
        resident.clear();
        residentBytes = 0;
        copier.release();
        if (failure != null) {
            throw failure;
        }
    }

    // ------------------------------------------------------------------ internals

    private Map<String, TensorPlan> planAll(List<GGUFTensorInfo> tensors) {
        Map<String, GGUFTensorInfo> byName = new LinkedHashMap<>();
        for (GGUFTensorInfo t : tensors) {
            byName.put(t.name(), t);
        }
        boolean tiedOutput = !byName.containsKey("output.weight") && !byName.containsKey("lm_head.weight");
        Map<String, TensorPlan> out = new LinkedHashMap<>();
        for (GGUFTensorInfo t : byName.values()) {
            TensorUse use = TensorUse.classify(t, tiedOutput);
            GGMLType deviceType = deviceTypeFor(t, use);
            out.put(t.name(), TensorPlan.create(t, use, deviceType, maxBufferBytes));
        }
        return out;
    }

    private GGMLType deviceTypeFor(GGUFTensorInfo t, TensorUse use) {
        GGMLType type = t.type();
        switch (use) {
            case INTEGER:
                return type;
            case ELEMENTWISE:
                return GGMLType.F32;
            case LOOKUP_MATMUL:
                return type == GGMLType.F32 || (config.policy.isNative(type, TensorUse.LOOKUP)
                        && config.policy.isNative(type, TensorUse.MATMUL)) ? type : GGMLType.F32;
            default:
                return type == GGMLType.F32 || config.policy.isNative(type, use) ? type : GGMLType.F32;
        }
    }

    private TensorPlan requirePlan(String name) {
        TensorPlan p = plans.get(name);
        if (p == null) {
            throw new IllegalArgumentException(archName + " has no tensor " + name);
        }
        return p;
    }

    /** Allocates and fills every shard of one tensor; on failure releases its buffers. */
    private GpuWeight uploadPlan(TensorPlan p) {
        p.requireUploadable();
        long[] shardRows = p.shardRows();
        List<GpuBuffer> buffers = new ArrayList<>(shardRows.length);
        try {
            long row = 0;
            for (long rows : shardRows) {
                long bytes = rows * p.deviceRowBytes();
                GpuBuffer b = allocate(p, bytes);
                buffers.add(b);
                try {
                    copier.copyRows(p, row, rows, b, 0L, queue);
                } catch (UncheckedIOException e) {
                    throw e;
                } catch (RuntimeException e) {
                    throw new IllegalStateException("upload of tensor " + p.name() + " rows [" + row + ", "
                            + (row + rows) + ") to " + backend.capabilities().getDeviceName() + " failed: "
                            + e.getMessage(), e);
                }
                row += rows;
            }
        } catch (RuntimeException e) {
            for (GpuBuffer b : buffers) {
                b.close();
            }
            throw e;
        }
        owned.addAll(buffers);
        GpuWeight w = buffers.size() == 1
                ? GpuWeight.of(p.name(), p.deviceType(), p.ne(), buffers.get(0))
                : GpuWeight.sharded(p.name(), p.deviceType(), p.ne(), buffers, shardRows);
        w = withPrism(w);
        resident.put(p.name(), w);
        residentBytes += p.deviceBytes();
        if (buffers.size() > 1) {
            LOGGER.info("{}: {} ({} {}, {} MB) split into {} row shards of at most {} MB", archName, p.name(),
                    p.sourceType(), Arrays.toString(p.ne()), p.deviceBytes() >> 20, buffers.size(),
                    maxBufferBytes >> 20);
        }
        return w;
    }

    private GpuBuffer allocate(TensorPlan p, long bytes) {
        try {
            return backend.allocateBuffer(bytes);
        } catch (RuntimeException e) {
            throw new IllegalStateException("cannot allocate " + bytes + " bytes for tensor " + p.name() + " ("
                    + p.deviceType() + ", " + backend.getAllocatedBytes() + " bytes already allocated on "
                    + backend.capabilities().getDeviceName() + "): " + e.getMessage(), e);
        }
    }

    private GpuWeight withPrism(GpuWeight w) {
        PrismHadamardLoader.Result prism = config.prism;
        if (prism == null || prism.isEmpty()) {
            return w;
        }
        HadamardSpec forward = prism.forward().get(w.name());
        if (forward != null) {
            w = w.withHadamard(forward);
        }
        HadamardSpec inverse = prism.inverse().get(w.name());
        if (inverse != null) {
            w = w.withInverseHadamard(inverse);
        }
        return w;
    }

    /** Attaches each resident scalar {@code <base>.scale} to its resident non-expert {@code <base>.weight}. */
    private void attachScales() {
        for (Map.Entry<String, GpuWeight> e : new ArrayList<>(resident.entrySet())) {
            GpuWeight w = e.getValue();
            String name = e.getKey();
            if (!name.endsWith(".weight") || w.hasScale()) {
                continue;
            }
            TensorPlan wp = plans.get(name);
            if (wp == null || wp.use() == TensorUse.EXPERT) {
                continue;
            }
            TensorPlan sp = plans.get(name.substring(0, name.length() - ".weight".length()) + ".scale");
            if (sp == null || sp.info().numElements() != 1) {
                continue;
            }
            float s = copier.readScalar(sp);
            if (!Float.isFinite(s)) {
                throw new IllegalStateException("tensor " + sp.name() + " holds a non-finite output scale " + s);
            }
            resident.put(name, w.withScale(s));
        }
    }

    private void finishQueue() {
        try {
            queue.finish();
            copier.synced(queue);
        } catch (RuntimeException e) {
            throw new IllegalStateException("weight upload to " + backend.capabilities().getDeviceName()
                    + " failed: " + e.getMessage(), e);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("weight uploader is closed");
        }
    }

    // ================================================================== nested types

    /** How the forward path consumes a tensor; decides its device type. */
    public enum TensorUse {
        /** Projection weight read by {@code {T}_MatVec} / batch matmul kernels. */
        MATMUL,
        /** Row lookup table read by {@code {T}_EmbeddingLookup} kernels. */
        LOOKUP,
        /** The token embedding of a model whose output head is tied to it: looked up and multiplied. */
        LOOKUP_MATMUL,
        /**
         * Position or token-type table ({@code position_embd}, {@code token_types}) whose rows are added to the
         * token embedding; a row lookup ({@code GET_ROWS}) like {@link #LOOKUP}.
         */
        ADDED_LOOKUP,
        /** Expert tensor {@code [cols, rows, n_expert]} read per expert. */
        EXPERT,
        /** Consumed by element-wise or special kernels: always F32. */
        ELEMENTWISE,
        /** Integer table: uploaded as stored. */
        INTEGER;

        /**
         * Names (after the {@code blk.N.} / {@code enc.blk.N.} / {@code dec.blk.N.} / {@code posnet.N.} /
         * {@code convnext.N.} prefix and the {@code .weight}/{@code .bias} suffix) of tensors that ggml consumes
         * with an op other than {@code MUL_MAT}/{@code MUL_MAT_ID}/{@code GET_ROWS} (llama-arch.cpp:817-1144,
         * {@code LLM_TENSOR_INFOS}), plus the relative-position bias and compressor position tables
         * ({@code attn_rel_b}, {@code *_compressor_ape}) that ggml reads with {@code GET_ROWS}; NeuralFree's kernels
         * for all of them read F32.
         */
        private static final Set<String> ELEMENTWISE_NAMES = new HashSet<>(Arrays.asList(
                "altup_correct_scale", "altup_router_norm", "attn_compressor_ape", "attn_compressor_norm",
                "attn_conv_base", "attn_k_norm", "attn_kv_a_norm", "attn_norm", "attn_norm_2", "attn_output_norm",
                "attn_q_a_norm", "attn_q_norm", "attn_rel_b", "attn_res_score", "attn_resid_alpha", "attn_resid_beta",
                "attn_sinks", "attn_sub_norm", "channel_mix_lerp_k", "channel_mix_lerp_r", "cls.norm", "conv1",
                "conv1d", "conv2", "cross_attn_norm", "cross_attn_norm_kv", "cross_attn_rel_b", "dec.output_norm",
                "decision.field_norm", "decision.hidden_norm", "decision.option_norm", "decision.option_summary_norm",
                "decision.query_norm", "decision.scales", "dw", "enc.output_norm", "exp_probs_b", "exp_probs_b_vl",
                "ffn.act", "ffn_conv_base", "ffn_norm", "ffn_norm_exps", "ffn_post_exp_norm", "ffn_pre_exp_norm",
                "ffn_res_score", "ffn_resid_alpha", "ffn_resid_beta", "ffn_routed_norm", "ffn_sub_norm", "gamma",
                "hc_attn_base", "hc_attn_norm", "hc_attn_scale", "hc_ffn_base", "hc_ffn_norm", "hc_ffn_scale",
                "hrm.z_l_init", "indexer.k_norm", "indexer.q_norm", "indexer_compressor_ape",
                "indexer_compressor_norm", "laurel_post_norm", "layer_output_norm", "layer_output_scale",
                "lilicorr.anchor_norm", "lilicorr.attention_bias", "lilicorr.output_norm", "lilicorr.rank_embedding",
                "lilicorr.slot_embedding", "nextn.enorm", "nextn.hc_head_norm", "nextn.hnorm",
                "nextn.shared_head_norm", "norm", "norm1", "norm2", "output_hc_base", "output_hc_norm",
                "output_hc_scale", "output_norm", "output_res_score", "per_layer_proj_norm", "ple_conv1d",
                "ple_norm_conv", "ple_norm_key", "ple_norm_query", "post_attention_norm", "post_ffw_norm",
                "post_ffw_norm_1", "post_ffw_norm_2", "post_norm", "pre_ffw_norm_2", "rope_factors_long",
                "rope_factors_short", "rope_freqs", "shortconv.conv", "ssm_a", "ssm_b_norm", "ssm_c_norm",
                "ssm_conv1d", "ssm_conv1d_k", "ssm_conv1d_q", "ssm_conv1d_v", "ssm_d", "ssm_dt_norm", "ssm_norm",
                "time_mix_a0", "time_mix_decay", "time_mix_first", "time_mix_k_a", "time_mix_k_k",
                "time_mix_lerp_fused", "time_mix_lerp_g", "time_mix_lerp_k", "time_mix_lerp_r", "time_mix_lerp_v",
                "time_mix_lerp_w", "time_mix_lerp_x", "time_mix_ln", "time_mix_r_k", "time_mix_v0", "time_mix_w0",
                "token_embd_norm"));

        /** Token row lookup tables ({@code GET_ROWS} in llama-arch.cpp {@code LLM_TENSOR_INFOS}). */
        private static final Set<String> LOOKUP_NAMES = new HashSet<>(Arrays.asList(
                "token_embd.weight", "tok_embeddings.weight", "per_layer_token_embd.weight",
                "nextn.embed_tokens.weight"));

        /** Lookup tables whose rows are added to the token embedding (also {@code GET_ROWS}). */
        private static final Set<String> ADDED_LOOKUP_NAMES = new HashSet<>(Arrays.asList(
                "position_embd.weight", "token_types.weight"));

        private static final Pattern PREFIX =
                Pattern.compile("^(?:(?:enc|dec)\\.)?(?:blk|posnet|convnext)\\.\\d+\\.");

        /** Classifies one tensor; {@code tiedOutput} when the model has no separate output head. */
        static TensorUse classify(GGUFTensorInfo t, boolean tiedOutput) {
            GGMLType type = t.type();
            if (type == GGMLType.I8 || type == GGMLType.I16 || type == GGMLType.I32 || type == GGMLType.I64) {
                return INTEGER;
            }
            String name = t.name();
            String local = PREFIX.matcher(name).replaceFirst("");
            if (WeightShards.isExpert(local, t.rank())) {
                return EXPERT;
            }
            if (LOOKUP_NAMES.contains(local)) {
                boolean tokenEmbd = local.equals("token_embd.weight") || local.equals("tok_embeddings.weight");
                return tokenEmbd && tiedOutput && local.equals(name) ? LOOKUP_MATMUL : LOOKUP;
            }
            if (ADDED_LOOKUP_NAMES.contains(local)) {
                return ADDED_LOOKUP;
            }
            if (t.rank() <= 1 || local.endsWith(".bias") || local.endsWith(".scale")
                    || local.endsWith(".lora_a") || local.endsWith(".lora_b")) {
                return ELEMENTWISE;
            }
            String base = local.endsWith(".weight") ? local.substring(0, local.length() - ".weight".length()) : local;
            return ELEMENTWISE_NAMES.contains(base) ? ELEMENTWISE : MATMUL;
        }
    }

    /** Whether a backend has kernels for a weight type in a given use. */
    @FunctionalInterface
    public interface TypePolicy {
        /** True when tensors of {@code type} in {@code use} are consumed as stored (no CPU dequantization). */
        boolean isNative(GGMLType type, TensorUse use);

        /**
         * The policy of the engine's GPU path: a matmul type needs {@code GpuOps} support and its
         * {@code {T}_MatVec} kernel, a lookup type its embedding-lookup kernel, an expert type its
         * {@code {T}_MatVecId} kernel.
         */
        static TypePolicy forKernels(LlmKernels kernels) {
            Objects.requireNonNull(kernels, "kernels");
            return (type, use) -> {
                switch (use) {
                    case MATMUL:
                        return GpuOps.supportsWeightType(type)
                                && kernels.has(matVecKernel(type));
                    case LOOKUP:
                    case LOOKUP_MATMUL:
                    case ADDED_LOOKUP:
                        return GpuOps.supportsWeightType(type)
                                && kernels.has(lookupKernel(type));
                    case EXPERT:
                        return MoeKernels.supports(type)
                                && kernels.has(MoeKernels.matVecId(type));
                    default:
                        return false;
                }
            };
        }

        /** {@code {T}_MatVec} kernel name of a weight type ({@code Float_MatVec} for F32). */
        static String matVecKernel(GGMLType type) {
            return type == GGMLType.F32 ? "Float_MatVec" : type.name() + "_MatVec";
        }

        /** Single-token embedding-lookup kernel name of a weight type. */
        static String lookupKernel(GGMLType type) {
            switch (type) {
                case F32:
                    return "LLM_EmbeddingLookup";
                case Q8_0:
                    return "LLM_Q8_0_EmbeddingLookup";
                case MXFP4:
                    return "LLM_MXFP4_EmbeddingLookup";
                default:
                    return type.name() + "_EmbeddingLookup";
            }
        }
    }

    /** Immutable uploader configuration. */
    public static final class Config {
        private final TypePolicy policy;
        private final PrismHadamardLoader.Result prism;
        private final int stagingBytes;
        private final long maxBufferBytes;

        private Config(Builder b) {
            this.policy = b.policy;
            this.prism = b.prism;
            this.stagingBytes = b.stagingBytes;
            this.maxBufferBytes = b.maxBufferBytes;
        }

        public TypePolicy policy() {
            return policy;
        }

        /** PRISM transforms, or null. */
        public PrismHadamardLoader.Result prism() {
            return prism;
        }

        public int stagingBytes() {
            return stagingBytes;
        }

        /** Largest buffer override in bytes; 0 uses the device limit. */
        public long maxBufferBytes() {
            return maxBufferBytes;
        }

        /** Builder of {@link Config}. */
        public static final class Builder {
            private final TypePolicy policy;
            private PrismHadamardLoader.Result prism;
            private int stagingBytes = DEFAULT_STAGING_BYTES;
            private long maxBufferBytes;

            Builder(TypePolicy policy) {
                this.policy = Objects.requireNonNull(policy, "policy");
            }

            /** PRISM transforms to attach; null or empty for none. */
            public Builder prism(PrismHadamardLoader.Result v) {
                this.prism = v;
                return this;
            }

            /** Host staging per upload chunk, at least 1 MiB. */
            public Builder stagingBytes(int v) {
                if (v < (1 << 20)) {
                    throw new IllegalArgumentException("staging must be at least 1 MiB, got " + v);
                }
                this.stagingBytes = v;
                return this;
            }

            /** Caps every buffer below the device limit; 0 (the default) uses the device limit. */
            public Builder maxBufferBytes(long v) {
                if (v < 0) {
                    throw new IllegalArgumentException("maxBufferBytes must be non-negative, got " + v);
                }
                this.maxBufferBytes = v;
                return this;
            }

            public Config build() {
                return new Config(this);
            }
        }
    }

    /** Upload plan of one tensor: use, device type and row shards. Immutable. */
    public static final class TensorPlan {
        private final GGUFTensorInfo info;
        private final TensorUse use;
        private final GGMLType deviceType;
        private final long[] ne;
        private final long rows;
        private final long cols;
        private final long sourceRowBytes;
        private final long deviceRowBytes;
        private final long[] shardRows;
        private final String problem;

        private TensorPlan(GGUFTensorInfo info, TensorUse use, GGMLType deviceType, long[] ne, long rows, long cols,
                           long sourceRowBytes, long deviceRowBytes, long[] shardRows, String problem) {
            this.info = info;
            this.use = use;
            this.deviceType = deviceType;
            this.ne = ne;
            this.rows = rows;
            this.cols = cols;
            this.sourceRowBytes = sourceRowBytes;
            this.deviceRowBytes = deviceRowBytes;
            this.shardRows = shardRows;
            this.problem = problem;
        }

        static TensorPlan create(GGUFTensorInfo info, TensorUse use, GGMLType deviceType, long maxBufferBytes) {
            GGMLType src = info.type();
            long[] ne = info.dims().length == 0 ? new long[]{1} : info.dims().clone();
            for (long d : ne) {
                if (d <= 0) {
                    throw new IllegalStateException("tensor " + info.name() + " has a non-positive dimension "
                            + Arrays.toString(ne));
                }
            }
            if (src.getBlockSize() <= 0 || src.getTypeSize() <= 0 || ne[0] % src.getBlockSize() != 0) {
                throw new IllegalStateException("tensor " + info.name() + " (" + src + ") has row width " + ne[0]
                        + ", not a whole number of " + src + " blocks");
            }
            long cols = ne[0];
            long rows = 1;
            for (int i = 1; i < ne.length; i++) {
                rows = Math.multiplyExact(rows, ne[i]);
            }
            long srcRowBytes = src.getBytesForElements(cols);
            long devRowBytes = deviceType.getBytesForElements(cols);
            long unitRows = ne.length >= 3 ? ne[1] : 1;
            String problem = null;
            long[] shards = null;
            if (deviceType != src && !CpuBlockDecoders.supports(src)) {
                problem = "tensor " + info.name() + " has type " + src
                        + ", which the backend has no kernels for and the CPU cannot dequantize";
            } else {
                long unitBytes = Math.multiplyExact(unitRows, devRowBytes);
                if (unitBytes > maxBufferBytes) {
                    problem = "tensor " + info.name() + ": " + (unitRows > 1 ? "one expert" : "one row") + " of "
                            + unitBytes + " bytes exceeds the device's largest buffer of " + maxBufferBytes
                            + " bytes";
                } else {
                    shards = WeightShards.shardRows(rows, unitRows, unitBytes, maxBufferBytes);
                }
            }
            return new TensorPlan(info, use, deviceType, ne, rows, cols, srcRowBytes, devRowBytes, shards, problem);
        }

        /**
         * Throws when this tensor cannot be uploaded: its type needs a dequantization the CPU decoders lack, or one
         * row (one expert for rank-3 tensors) exceeds the device's largest buffer.
         */
        public void requireUploadable() {
            if (problem != null) {
                throw new UnsupportedOperationException(problem);
            }
        }

        /** Why the tensor cannot be uploaded, or null. */
        public String problem() {
            return problem;
        }

        public String name() {
            return info.name();
        }

        public GGUFTensorInfo info() {
            return info;
        }

        public TensorUse use() {
            return use;
        }

        /** Stored type in the GGUF file. */
        public GGMLType sourceType() {
            return info.type();
        }

        /** Type of the device copy: the stored type, or F32 when dequantized on upload. */
        public GGMLType deviceType() {
            return deviceType;
        }

        /** True when the device copy is decoded to F32 on the CPU. */
        public boolean needsDecode() {
            return deviceType != info.type();
        }

        /** ggml shape (copy). */
        public long[] ne() {
            return ne.clone();
        }

        public long rows() {
            return rows;
        }

        public long cols() {
            return cols;
        }

        public long sourceRowBytes() {
            return sourceRowBytes;
        }

        public long deviceRowBytes() {
            return deviceRowBytes;
        }

        /** Device bytes of the whole tensor. */
        public long deviceBytes() {
            return rows * deviceRowBytes;
        }

        /** Rows per shard (copy); one entry when the tensor fits one buffer, empty when it cannot be uploaded. */
        public long[] shardRows() {
            return shardRows == null ? new long[0] : shardRows.clone();
        }

        public int shardCount() {
            return shardRows == null ? 0 : shardRows.length;
        }

        @Override
        public String toString() {
            return info.name() + " " + info.type() + (needsDecode() ? "->" + deviceType : "") + " "
                    + Arrays.toString(ne) + " " + use + " shards=" + shardCount();
        }
    }

    /**
     * Copies row ranges of a tensor from the weight store into device buffers through reusable host staging,
     * decoding to F32 when the plan says so.
     *
     * <p>A backend may keep a host copy of every pending write until its queue is finished (OpenCL retains each
     * write's staging copy until {@code finish()}). The copier therefore finishes a queue once the writes it enqueued
     * there since that queue's last finish reach {@link #IN_FLIGHT_CHUNKS} staging chunks, which bounds the host
     * memory held by pending uploads independently of the model size.</p>
     */
    private static final class RowCopier {
        /** Staging chunks a queue may hold in pending writes before the copier finishes it. */
        static final int IN_FLIGHT_CHUNKS = 8;

        private final WeightStore store;
        private final int stagingBytes;
        private final long inFlightLimit;
        private final Map<ComputeQueue, long[]> pending = new IdentityHashMap<>();
        private ByteBuffer staging;
        private float[] floats;

        RowCopier(WeightStore store, int stagingBytes) {
            this.store = store;
            this.stagingBytes = stagingBytes;
            this.inFlightLimit = (long) IN_FLIGHT_CHUNKS * stagingBytes;
        }

        /** Declares that {@code q} has been finished: none of its writes are pending any more. */
        void synced(ComputeQueue q) {
            pending.remove(q);
        }

        /** Accounts {@code bytes} of writes enqueued on {@code q}; finishes {@code q} when the window is full. */
        private void enqueued(ComputeQueue q, long bytes) {
            long[] held = pending.computeIfAbsent(q, k -> new long[1]);
            held[0] += bytes;
            if (held[0] >= inFlightLimit) {
                q.finish();
                held[0] = 0;
            }
        }

        /**
         * Enqueues the upload of rows {@code [firstRow, firstRow + nRows)} of {@code p} to {@code dst} at byte
         * {@code dstOffset}. The host staging is reusable when this returns (writes copy at enqueue).
         */
        void copyRows(TensorPlan p, long firstRow, long nRows, GpuBuffer dst, long dstOffset, ComputeQueue q) {
            if (p.needsDecode()) {
                copyDecoded(p, firstRow, nRows, dst, dstOffset, q);
            } else {
                copyRaw(p, firstRow * p.sourceRowBytes(), nRows * p.sourceRowBytes(), dst, dstOffset, q);
            }
        }

        private void copyRaw(TensorPlan p, long srcOffset, long bytes, GpuBuffer dst, long dstOffset,
                             ComputeQueue q) {
            ByteBuffer buf = staging(stagingBytes);
            long done = 0;
            while (done < bytes) {
                int n = (int) Math.min(bytes - done, buf.capacity());
                buf.clear();
                read(p, srcOffset + done, n, buf);
                closeEvent(q.enqueueWriteBuffer(dst, dstOffset + done, n, buf));
                enqueued(q, n);
                done += n;
            }
        }

        private void copyDecoded(TensorPlan p, long firstRow, long nRows, GpuBuffer dst, long dstOffset,
                                 ComputeQueue q) {
            long srcRow = p.sourceRowBytes();
            long cols = p.cols();
            long rowsPerChunk = Math.max(1, Math.min(stagingBytes / srcRow, stagingBytes / (cols * 4L)));
            long chunkFloats = rowsPerChunk * cols;
            if (rowsPerChunk * srcRow > Integer.MAX_VALUE || chunkFloats > Integer.MAX_VALUE - 8) {
                throw new IllegalStateException("tensor " + p.name() + ": one row of " + cols
                        + " elements exceeds the host staging");
            }
            ByteBuffer buf = staging((int) Math.max(stagingBytes, rowsPerChunk * srcRow));
            float[] out = floats((int) chunkFloats);
            long done = 0;
            while (done < nRows) {
                long r = Math.min(rowsPerChunk, nRows - done);
                int n = (int) (r * srcRow);
                buf.clear();
                read(p, (firstRow + done) * srcRow, n, buf);
                buf.flip();
                CpuBlockDecoders.decodeElements(p.sourceType(), buf, 0L, out, 0,
                        r * cols);
                closeEvent(q.enqueueWriteBuffer(dst, dstOffset + done * cols * 4L, r * cols * 4L, out));
                enqueued(q, r * cols * 4L);
                done += r;
            }
        }

        /** Decodes the single element of a one-element tensor. */
        float readScalar(TensorPlan p) {
            int n = (int) p.sourceRowBytes();
            ByteBuffer buf = ByteBuffer.allocate(n);
            byte[] raw = new byte[n];
            try {
                store.readTensor(p.info().offset(), n, raw);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read tensor " + p.name(), e);
            }
            buf.put(raw).flip();
            float[] v = new float[(int) Math.max(1, p.cols())];
            CpuBlockDecoders.decodeElements(p.sourceType(), buf, 0L, v, 0,
                    p.cols());
            return v[0];
        }

        private void read(TensorPlan p, long byteOffset, int n, ByteBuffer dst) {
            try {
                store.readTensorDirect(p.info().offset() + byteOffset, n, dst);
            } catch (IOException e) {
                throw new UncheckedIOException("cannot read " + n + " bytes of tensor " + p.name() + " at byte "
                        + byteOffset, e);
            }
        }

        private ByteBuffer staging(int capacity) {
            if (staging == null || staging.capacity() < capacity) {
                staging = ByteBuffer.allocateDirect(capacity).order(ByteOrder.LITTLE_ENDIAN);
            }
            return staging;
        }

        private float[] floats(int n) {
            if (floats == null || floats.length < n) {
                floats = new float[n];
            }
            return floats;
        }

        /** Drops the host staging and the pending-write accounting (writes copy their host data at enqueue). */
        void release() {
            staging = null;
            floats = null;
            pending.clear();
        }

        private static void closeEvent(GpuEvent e) {
            if (e != null) {
                e.close();
            }
        }
    }
}
