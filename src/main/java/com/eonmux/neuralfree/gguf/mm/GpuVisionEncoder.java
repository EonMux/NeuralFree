package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.ops.vision.VisionKernels;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * GPU executor of one vision encoder and its projector on an OpenCL, CUDA or Vulkan backend through
 * {@link GpuOps} and the LLM kernel library. It runs the same family graphs as {@link CpuVisionEncoder}: both
 * dispatch through {@link VisionFamilies}, and each graph is written once against {@link VisionOps}, run here by
 * {@link GpuVisionOps} and its tower {@link GpuVitTower}. So every runnable projector family runs on the device:
 * the patch embedding as a GEMM over the {@link ClipPatches} matrix; the ViT blocks with layer or RMS norms, fused
 * or separate Q/K/V, Q/K norms, the qwen vision m-rope or the 2D RoPE of {@code build_rope_2d}, bidirectional flash
 * attention (one launch per key window for windowed layers), dense or routed-expert FFNs and layer scales; every
 * projector, its reorders, convs, resampler and QFormer attentions; and the convolutional encoders (the yasa2
 * ConvNeXt, the gemma3nv MobileNetV5 and the DeepSeek-OCR SAM backbone) over the depthwise-conv, average-pool and
 * relative-position-attention kernels of {@code gguf.ops.vision.VisionKernels}. One {@link #encode} call uploads
 * the patch matrix and the host-built tables (positions, reorder indices), enqueues the graph and reads the
 * embedding rows back. Host rows the graphs assemble (class and position rows, framing rows, the MobileNetV5
 * fusion adapter's feature concat) are uploaded per call, and the small reductions the graphs run on the host
 * (the squeeze-excite and the response norm's channel means) read the device rows back first. Work buffers are
 * scratch temporaries reused call after call. {@link #unsupportedReason} names what keeps a model off a device
 * (a head width the flash kernels cannot take, an attention shape without a tiled plan, an expert layout the MoE
 * kernels cannot route); callers fall back to the CPU encoder then. Not thread-safe; a closed encoder refuses
 * calls.
 */
public final class GpuVisionEncoder implements VisionEncoder {

    /**
     * Kernels the encoder launches directly, besides the projections and helpers of {@link GpuOps} and the
     * {@code {T}_BatchMatMulId} expert products of the weight types a MoE tower stores.
     */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            FlowKernels.GATHER_ROWS, GpuOps.BATCH_BIAS_ADD, RopeKernels.ROPE_EXT, AttnKernels.flashAttn(KvDType.F32),
            NormKernels.ROW_NORM, NormKernels.MUL_ROW_SCALE, ActKernels.UNARY, ActKernels.GLU, MoeKernels.ROUTER,
            MoeKernels.WEIGHTED_SUM, VisionKernels.CONV2D_DEPTHWISE, VisionKernels.POOL_2D_AVG,
            VisionKernels.REL_POS_ATTN));

    private final ClipVisionModel model;
    private final GpuOps ops;
    private final GpuVisionOps graph;
    private final GpuProjectionDeltas deltas;
    private final String device;
    private final List<AutoCloseable> owned = new ArrayList<>();
    private LoraAttachment loras;
    /** The open one-caller LoRA session ({@link #openLoraSession}), or null. */
    private LoraAttachment sessionLoras;
    private boolean closed;

    private GpuVisionEncoder(ClipVisionModel model, GpuOps ops, GpuWeights weights, String device) {
        this.model = Objects.requireNonNull(model, "model");
        this.ops = Objects.requireNonNull(ops, "ops");
        this.device = Objects.requireNonNull(device, "device");
        int smem = ops.kernels().faSmemFloats();
        int units = Math.max(1, ops.kernels().caps().getComputeUnits());
        String reason = unsupportedReason(model, smem, units);
        if (reason != null) {
            throw new UnsupportedOperationException(reason);
        }
        NormKernels.checkReduceLocal(ops.local());
        this.deltas = new GpuProjectionDeltas();
        this.graph = new GpuVisionOps(model, ops, new GpuVisionWeights(model, ops, weights, deltas), smem, units);
        prewarmKernels();
    }

    /**
     * The LoRA deltas of this encoder's tower and connector products: an attachment over them
     * ({@link #attachLoras}) applies the adapters' vision pairs to every encoded row.
     */
    public GpuProjectionDeltas projectionDeltas() {
        return deltas;
    }

    @Override
    public LoraAttachment attachLoras(List<LoraAttachment.Binding> bindings) {
        Objects.requireNonNull(bindings, "bindings");
        if (loras != null) {
            throw new IllegalStateException("the vision encoder already runs LoRA adapters (" + loras + ")");
        }
        if (!LoraAttachment.adapts(bindings, LoraAdapter.Side.VISION)) {
            return null;
        }
        loras = LoraAttachment.gpu(bindings, ops, deltas, LoraAdapter.Side.VISION);
        return loras;
    }

    @Override
    public int openLoraSession(List<LoraAttachment.Binding> bindings) {
        Objects.requireNonNull(bindings, "bindings");
        ensureOpen();
        if (sessionLoras != null) {
            throw new IllegalStateException("the vision encoder already runs a LoRA session (" + sessionLoras
                    + "); it serves one caller's session at a time");
        }
        sessionLoras = LoraAttachment.session(bindings, ops, deltas, LoraAdapter.Side.VISION);
        return LORA_SESSION_SLOT;
    }

    @Override
    public void closeLoraSession(int slot) {
        if (sessionLoras == null) {
            throw new IllegalStateException("the vision encoder runs no LoRA session to close");
        }
        if (slot != LORA_SESSION_SLOT) {
            throw new IllegalStateException("the vision encoder's LoRA session runs in adapter slot "
                    + LORA_SESSION_SLOT + ", not " + slot);
        }
        sessionLoras.close();
        sessionLoras = null;
    }

    /**
     * Binds an encoder over device weights already resident on {@code ops}' backend; the weights, ops and scratch
     * stay owned by the caller.
     *
     * @param device device name {@link #device()} reports
     * @throws UnsupportedOperationException with {@link #unsupportedReason} when the device cannot run the model
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is not resident
     */
    public static GpuVisionEncoder bind(ClipVisionModel model, GpuOps ops, GpuWeights weights, String device) {
        return new GpuVisionEncoder(model, ops, weights, device);
    }

    /**
     * Uploads the mmproj tensors to {@code kernels}' backend and binds an encoder that owns its device state: the
     * placeholders, the scratch, the ops, the uploaded weights and, last, {@code backendOwner} (the backend's
     * release when the caller hands it over; null to keep the backend). On failure everything acquired here is
     * released, {@code backendOwner} included.
     *
     * @param tensors    every tensor of the mmproj file
     * @param store      tensor data of the file; read during this call only
     * @param transforms input transforms of the file's weights
     * @throws UnsupportedOperationException with {@link #unsupportedReason} when the device cannot run the model
     * @throws IllegalStateException         when a weight upload fails
     */
    public static GpuVisionEncoder open(ClipVisionModel model, LlmKernels kernels, String device,
                                        List<GGUFTensorInfo> tensors, WeightStore store,
                                        InputTransformLoader.Result transforms, AutoCloseable backendOwner) {
        Objects.requireNonNull(kernels, "kernels");
        List<AutoCloseable> acquired = new ArrayList<>();
        if (backendOwner != null) {
            acquired.add(backendOwner);
        }
        try {
            ComputeBackend backend = kernels.backend();
            ComputeQueue q = backend.computeQueue();
            GpuPlaceholders ph = GpuPlaceholders.allocate(backend, q);
            acquired.add(ph);
            GpuScratch sc = GpuScratch.builder(backend, 1).build();
            acquired.add(sc);
            GpuOps o = new GpuOps(q, kernels, sc, ph);
            acquired.add(o);
            GpuWeightUploader uploader = new GpuWeightUploader(backend, q, tensors, store, "clip",
                    GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(kernels))
                            .transforms(transforms).build());
            acquired.add(uploader);
            uploader.uploadResident(t -> false);
            WeightKernelPrewarm.prewarm(uploader.plans(t -> true), o, kernels);
            GpuVisionEncoder enc = new GpuVisionEncoder(model, o, uploader, device);
            enc.owned.addAll(acquired);
            return enc;
        } catch (RuntimeException | Error e) {
            RuntimeException failure = closeReversed(acquired);
            if (failure != null) {
                e.addSuppressed(failure);
            }
            throw e;
        }
    }

    /**
     * Why {@code model} cannot run on the GPU encoder of a device with {@code smemFloats} flash shared floats and
     * {@code computeUnits} compute units, or null when it can: a head width the flash kernels cannot take, heads
     * that do not span {@code n_embd} for Q, K and V, attention sinks, an attention of the graph (the tower, the
     * MiniCPM-V resampler, the Granite QFormer) without a tiled flash plan, or a MoE block beyond the router's
     * expert limits or with an expert type the id-matmul kernels lack. A blockless embedder has no attention; the
     * convolutional encoders (the yasa2 ConvNeXt, the DeepSeek-OCR SAM backbone, MobileNetV5) run their device
     * kernels on every backend the flash kernels run on.
     */
    public static String unsupportedReason(ClipVisionModel model, int smemFloats, int computeUnits) {
        Objects.requireNonNull(model, "model");
        String type = model.hparams().projectorType().typeName();
        String tower = model.layers().isEmpty() ? null : towerReason(model, type);
        if (tower != null) {
            return tower;
        }
        for (int[] shape : attentionShapes(model)) {
            FlashAttnTiling p;
            try {
                p = GpuVisionOps.attentionPlan(shape[0], shape[1], shape[2], Math.max(1, model.nPos()), smemFloats,
                        computeUnits);
            } catch (IllegalArgumentException e) {
                return "projector type " + type + ": " + e.getMessage();
            }
            if (p.plan() != FlashAttnTiling.Plan.TILED) {
                return "projector type " + type + ": no tiled flash-attention plan for " + shape[0] + " heads of"
                        + " width " + shape[2] + " (" + smemFloats + " shared floats): " + p;
            }
        }
        return moeReason(model, type);
    }

    /** Whether the GPU encoder can run {@code model} with these kernels ({@link #unsupportedReason}). */
    public static String unsupportedReason(ClipVisionModel model, LlmKernels kernels) {
        Objects.requireNonNull(kernels, "kernels");
        return unsupportedReason(model, kernels.faSmemFloats(), Math.max(1, kernels.caps().getComputeUnits()));
    }

    /** Why the blocks of {@code model} cannot run on the GPU tower, or null. */
    private static String towerReason(ClipVisionModel model, String type) {
        ClipHparams h = model.hparams();
        int dh = model.dHead();
        if (dh <= 0 || dh % 4 != 0) {
            return "projector type " + type + ": head width " + dh + " is not a multiple of 4, which the flash"
                    + " attention kernels need";
        }
        for (ClipLayer l : model.layers()) {
            if (l.dHead() * h.nHead() != h.nEmbd()) {
                return "projector type " + type + ": the GPU tower reads Q rows n_embd (" + h.nEmbd()
                        + ") wide, the blocks use " + h.nHead() + " query and " + h.nHeadKv() + " key/value heads"
                        + " of " + l.dHead();
            }
            if (l.attnSinks() != null) {
                return "projector type " + type + ": the GPU flash attention has no attention sinks ("
                        + l.attnSinks().name() + ")";
            }
        }
        return null;
    }

    /** The {@code {nHead, nHeadKv, dHead}} of every attention the model's graph launches. */
    private static List<int[]> attentionShapes(ClipVisionModel model) {
        ClipHparams h = model.hparams();
        List<int[]> shapes = new ArrayList<>();
        if (!model.layers().isEmpty()) {
            shapes.add(new int[]{h.nHead(), h.nHeadKv(), model.dHead()});
        }
        if (ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER.equals(model.projectorGraph())) {
            int heads = model.outputDim() / ClipProjectorTensors.RESAMPLER_HEAD_DIM;
            shapes.add(new int[]{heads, heads, ClipProjectorTensors.RESAMPLER_HEAD_DIM});
        }
        if (ProjectorEntry.ENCODER_GRANITE4_VIT.equals(model.encoderGraph())) {
            int heads = h.nEmbd() / Granite4VisionGraph.HEAD_DIM;
            shapes.add(new int[]{heads, heads, Granite4VisionGraph.HEAD_DIM});
        }
        return shapes;
    }

    /** Why a MoE block of {@code model} cannot route on the device, or null. */
    private static String moeReason(ClipVisionModel model, String type) {
        int k = model.hparams().nExpertUsed();
        for (int il = 0; il < model.layers().size(); il++) {
            ClipLayer l = model.layers().get(il);
            if (!l.moe()) {
                continue;
            }
            if (l.nExpert() > MoeKernels.MAX_EXPERTS || Math.min(k, l.nExpert()) > MoeKernels.MAX_K) {
                return "projector type " + type + ": block " + il + " routes " + Math.min(k, l.nExpert()) + " of "
                        + l.nExpert() + " experts, beyond the GPU router's " + MoeKernels.MAX_K + " of "
                        + MoeKernels.MAX_EXPERTS;
            }
            for (TensorRef t : new TensorRef[]{l.moeGateWeight(), l.moeUpWeight(), l.moeDownWeight()}) {
                if (!MoeKernels.supports(t.type())) {
                    return "projector type " + type + ": expert tensor " + t.name() + " is " + t.type()
                            + ", which has no GPU id-matmul kernel";
                }
            }
        }
        return null;
    }

    /** Prepares each kernel launched through {@link GpuOps#enqueue} at its work-group size. */
    private void prewarmKernels() {
        long[] row = {ops.local()};
        ops.prewarm(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.GATHER_LOCAL});
        ops.prewarm(GpuOps.BATCH_BIAS_ADD, row);
        ops.prewarm(NormKernels.MUL_ROW_SCALE, row);
        ops.prewarm(RopeKernels.ROPE_EXT, row);
        ops.prewarm(AttnKernels.flashAttn(KvDType.F32), FlashAttnTiling.prefillLocal());
    }

    // ------------------------------------------------------------------ VisionEncoder

    @Override
    public float[][] encode(ClipImage image) {
        ensureOpen();
        Objects.requireNonNull(image, "image");
        graph.begin();
        return VisionFamilies.encode(graph, image);
    }

    @Override
    public float[][] encodeAnyres(ClipImage overview, List<ClipImage> tiles, int gridX, int gridY, int origWidth,
                                  int origHeight) {
        ensureOpen();
        graph.begin();
        return VisionFamilies.encodeAnyres(graph, overview, tiles, gridX, gridY, origWidth, origHeight);
    }

    /**
     * The model's shared ViT tower ({@link VisionOps#tower}, layer norms) over {@code n} input rows the caller
     * assembled, read back to the host: the pre-norm, every block under {@code opts} and the post-norm. The
     * towers of a CLIP dual encoder run here, their heads and projections on the host.
     *
     * @throws IllegalArgumentException when {@code rows} holds other than {@code n} rows of {@code n_embd}
     */
    float[] tower(float[] rows, int n, VisionOps.TowerOptions opts) {
        ensureOpen();
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(opts, "opts");
        int width = model.hparams().nEmbd();
        if (n <= 0 || rows.length != (long) n * width) {
            throw new IllegalArgumentException("the tower reads " + n + " rows of " + width + ", got "
                    + rows.length + " values");
        }
        graph.begin();
        VisionOps.Block x = graph.upload(rows, width);
        return graph.download(graph.tower(x, n, NormKind.LAYER, opts));
    }

    @Override
    public ClipVisionModel model() {
        return model;
    }

    @Override
    public String device() {
        return device;
    }

    /** Ends use of the encoder and releases the device state it owns (see {@link #open}). Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        if (sessionLoras != null) {
            try {
                sessionLoras.close();
            } catch (RuntimeException e) {
                failure = e;
            }
            sessionLoras = null;
        }
        if (loras != null) {
            try {
                loras.close();
            } catch (RuntimeException e) {
                if (failure != null) {
                    failure.addSuppressed(e);
                } else {
                    failure = e;
                }
            }
            loras = null;
        }
        RuntimeException released = closeReversed(owned);
        owned.clear();
        if (failure != null) {
            if (released != null) {
                failure.addSuppressed(released);
            }
            throw failure;
        }
        if (released != null) {
            throw released;
        }
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "GpuVisionEncoder{%s on %s, out=%d}", model.hparams().projectorType(),
                device, model.outputDim());
    }

    private static RuntimeException closeReversed(List<AutoCloseable> closeables) {
        List<AutoCloseable> reversed = new ArrayList<>(closeables);
        Collections.reverse(reversed);
        return Closeables.closeAll(reversed, null);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the GPU vision encoder is closed");
        }
    }
}
