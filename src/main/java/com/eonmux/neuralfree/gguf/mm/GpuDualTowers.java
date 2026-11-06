package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.flow.WeightKernelPrewarm;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.utils.Closeables;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * The GPU executor of a CLIP dual encoder's towers on an OpenCL, CUDA or Vulkan backend: both towers' blocks and
 * norms run through {@link GpuVisionEncoder#tower} (the shared vision tower, {@link GpuVitTower}) over one upload of
 * the file's block weights, the text tower under the causal mask of a {@link ClipFamily#CLIP} file. The rows the
 * towers read are assembled on the host, and the pooled row's head, projection or attention pooling head runs on
 * the host as well, so the token embedding, the projections and the heads stay off the device. Owns its device
 * state (placeholders, scratch, ops, the uploaded weights and the backend's release). Not thread-safe; one call at
 * a time. Package-private executor of {@link ClipDualEncoder}.
 */
final class GpuDualTowers implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(GpuDualTowers.class);

    private final GpuVisionEncoder text;
    private final GpuVisionEncoder vision;
    private final boolean causalText;
    private final String device;
    private final List<AutoCloseable> owned;
    private boolean closed;

    private GpuDualTowers(GpuVisionEncoder text, GpuVisionEncoder vision, boolean causalText, String device,
                          List<AutoCloseable> owned) {
        this.text = text;
        this.vision = vision;
        this.causalText = causalText;
        this.device = device;
        this.owned = owned;
    }

    /**
     * The towers on the device {@code options} choose, as an mmproj's vision encoder chooses it
     * ({@link MmprojModel#openVisionEncoder(MmprojOptions)}): null (the host towers) when offload is off, when the
     * AUTO mode finds no GPU that runs the LLM kernels, or when the device cannot run a tower (the reason logged).
     *
     * @throws IllegalArgumentException    when the selector names no device or one that cannot run the LLM
     *                                     kernels, or the options name a rank group of two or more devices (the
     *                                     towers run whole on one device)
     * @throws NoBackendAvailableException when an explicit backend mode (not AUTO) has no usable device
     * @throws IOException                 when the upload mapping cannot be opened
     */
    static GpuDualTowers select(ClipVisionModel text, ClipVisionModel vision, boolean causalText,
                                MmprojOptions options, ModelSource source, InputTransformLoader.Result transforms)
            throws IOException {
        Objects.requireNonNull(options, "options");
        if (options.devices().size() > 1) {
            throw new IllegalArgumentException("the CLIP dual encoder's towers run whole on one device; "
                    + MmprojOptions.PROP_DEVICES + " names " + options.devices().size() + " ranks, which only a"
                    + " vision encoder of a generation model spreads images over (" + MmprojOptions.PROP_TP_MODE
                    + "=" + MmEncoderTpMode.DATA + "); unset it or keep " + MmprojOptions.PROP_TP_MODE + "="
                    + MmEncoderTpMode.WEIGHTS);
        }
        if (!options.offload()) {
            return null;
        }
        HWAcceleration acceleration = HWAcceleration.forMode(options.backend().toAccelerationMode());
        AutoCloseable release = acceleration::shutdown;
        GpuDevices.Selected target;
        LlmKernels kernels;
        WeightStore store;
        try {
            target = MmprojModel.gpuTarget(acceleration, options);
            if (target == null) {
                LOGGER.info("CLIP dual encoder: no GPU backend can run the LLM kernels; the towers run on the CPU");
                acceleration.shutdown();
                return null;
            }
            kernels = LlmKernels.open(target.backend());
            String reason = unsupportedReason(text, vision, kernels);
            if (reason != null) {
                LOGGER.warn("CLIP dual encoder on {}: {}; the towers run on the CPU", target.name(), reason);
                acceleration.shutdown();
                return null;
            }
            store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD);
        } catch (IOException | RuntimeException | Error e) {
            acceleration.shutdown();
            throw e;
        }
        try {
            GpuDualTowers towers = open(text, vision, causalText, kernels, target.name(), source.getTensors(), store,
                    transforms, release);
            LOGGER.info("CLIP dual encoder: towers on {}", target.name());
            return towers;
        } finally {
            try {
                store.close();
            } catch (Exception e) {
                LOGGER.warn("releasing the CLIP dual encoder upload mapping failed: {}", e.getMessage());
            }
        }
    }

    /**
     * Why the device of {@code kernels} cannot run the towers ({@link GpuVisionEncoder#unsupportedReason}), or null
     * when it can; a null tower is not checked.
     */
    static String unsupportedReason(ClipVisionModel text, ClipVisionModel vision, LlmKernels kernels) {
        Objects.requireNonNull(kernels, "kernels");
        String reason = text == null ? null : GpuVisionEncoder.unsupportedReason(text, kernels);
        if (reason != null) {
            return "the text tower: " + reason;
        }
        reason = vision == null ? null : GpuVisionEncoder.unsupportedReason(vision, kernels);
        return reason == null ? null : "the vision tower: " + reason;
    }

    /**
     * Uploads the block weights of the towers to {@code kernels}' backend and binds both towers. On failure
     * everything acquired here is released, {@code backendOwner} included.
     *
     * @param text         the text tower, or null
     * @param vision       the vision tower, or null
     * @param causalText   whether the text tower attends causally (CLIP) or bidirectionally (SigLIP)
     * @param tensors      every tensor of the two-tower file
     * @param store        tensor data of the file; read during this call only
     * @param backendOwner the backend's release, or null to keep the backend
     * @throws UnsupportedOperationException when the device cannot run a tower ({@link #unsupportedReason})
     * @throws IllegalStateException         when a weight upload fails
     */
    static GpuDualTowers open(ClipVisionModel text, ClipVisionModel vision, boolean causalText, LlmKernels kernels,
                              String device, List<GGUFTensorInfo> tensors, WeightStore store,
                              InputTransformLoader.Result transforms, AutoCloseable backendOwner) {
        Objects.requireNonNull(kernels, "kernels");
        Objects.requireNonNull(device, "device");
        if (text == null && vision == null) {
            throw new IllegalArgumentException("a dual encoder needs a text or a vision tower");
        }
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
            Predicate<GGUFTensorInfo> hostOnly = t -> hostOnly(t.name());
            uploader.uploadResident(hostOnly);
            WeightKernelPrewarm.prewarm(uploader.plans(hostOnly.negate()), o, kernels);
            GpuVisionEncoder t = text == null ? null : GpuVisionEncoder.bind(text, o, uploader, device);
            if (t != null) {
                acquired.add(t);
            }
            GpuVisionEncoder v = vision == null ? null : GpuVisionEncoder.bind(vision, o, uploader, device);
            if (v != null) {
                acquired.add(v);
            }
            return new GpuDualTowers(t, v, causalText, device, acquired);
        } catch (RuntimeException | Error e) {
            RuntimeException failure = closeReversed(acquired);
            if (failure != null) {
                e.addSuppressed(failure);
            }
            throw e;
        }
    }

    /**
     * The tensors the host reads and the device never does: the token embedding, the projections, the text and
     * attention pooling heads, and SigLIP's logit scale and bias.
     */
    static boolean hostOnly(String name) {
        return name.equals(ClipTextTower.TOKEN_EMBD) || name.equals(ClipTextTower.PROJECTION)
                || name.equals(ClipImageTower.PROJECTION) || name.startsWith(ClipTextTower.HEAD + ".")
                || name.startsWith("v.head.") || name.equals(ClipDualEncoder.LOGIT_SCALE)
                || name.equals(ClipDualEncoder.LOGIT_BIAS);
    }

    /** The device name the towers run on. */
    String device() {
        return device;
    }

    /** The text tower's blocks and final norm over {@code n} assembled rows. */
    float[] text(float[] rows, int n) {
        ensureOpen();
        if (text == null) {
            throw new IllegalStateException("the GPU towers hold no text tower");
        }
        VisionOps.TowerOptions opts = causalText ? VisionOps.TowerOptions.NONE.withCausal()
                : VisionOps.TowerOptions.NONE;
        return text.tower(rows, n, opts);
    }

    /** The vision tower's pre-norm, blocks and post-norm over {@code n} assembled rows. */
    float[] vision(float[] rows, int n) {
        ensureOpen();
        if (vision == null) {
            throw new IllegalStateException("the GPU towers hold no vision tower");
        }
        return vision.tower(rows, n, VisionOps.TowerOptions.NONE);
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the GPU dual-encoder towers are closed");
        }
    }

    /** Releases the device state, the backend's release last. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = closeReversed(owned);
        owned.clear();
        if (failure != null) {
            throw failure;
        }
    }

    private static RuntimeException closeReversed(List<AutoCloseable> closeables) {
        List<AutoCloseable> reversed = new ArrayList<>(closeables);
        Collections.reverse(reversed);
        return Closeables.closeAll(reversed, null);
    }

    @Override
    public String toString() {
        return "GpuDualTowers{text=" + (text != null) + ", vision=" + (vision != null) + " on " + device + "}";
    }
}
