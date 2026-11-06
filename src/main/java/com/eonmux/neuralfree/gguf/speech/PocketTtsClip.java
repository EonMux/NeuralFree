package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.io.IOException;
import java.util.Objects;

/**
 * An opened pockettts generator file: the clip GGUF that carries the flow-matching decoder and the streaming mimi
 * decoder beside the backbone, and, when the file declares one, the mimi speaker encoder of voice cloning. It is
 * the pockettts counterpart of {@link Qwen3TtsClip}: llama.cpp loads it as the mmproj ({@code --mmproj}) and selects
 * the generator from the file's {@code clip.gen.audio.projector_type} ({@code pockettts_gen}).
 *
 * <p>Opening reads the metadata and the tensor table, resolves the file's {@link PocketTtsFlow} (GEN_CODE),
 * {@link PocketTtsMimiDecoder} (GEN_WAV), optional {@link PocketTtsSpeakerEncoder} and {@link PocketTtsPack}, and
 * binds the host weights once; the graphs share them. Immutable over the file; close it to release the mmap and the
 * decoded vectors.
 */
public final class PocketTtsClip implements AutoCloseable {

    /** The {@code general.architecture} of a clip generator file. */
    public static final String ARCH = "clip";

    private final String path;
    private final ModelSource source;
    private final CpuWeights weights;
    private final PocketTtsFlow flow;
    private final PocketTtsMimiDecoder decoder;
    private final PocketTtsPack pack;
    private final PocketTtsSpeakerEncoder speakerEncoder;
    private final InputTransformLoader.Result transforms;
    private GpuWeightUploader deviceUploader;
    private WeightStore deviceStore;
    private SpeechDeviceWeights deviceWeights;
    private boolean closed;

    private PocketTtsClip(String path, ModelSource source, CpuWeights weights,
                          InputTransformLoader.Result transforms, PocketTtsFlow flow,
                          PocketTtsMimiDecoder decoder, PocketTtsPack pack,
                          PocketTtsSpeakerEncoder speakerEncoder) {
        this.path = path;
        this.source = source;
        this.weights = weights;
        this.transforms = transforms;
        this.flow = flow;
        this.decoder = decoder;
        this.pack = pack;
        this.speakerEncoder = speakerEncoder;
    }

    /**
     * Opens the pockettts generator file at {@code path} on the host.
     *
     * @throws IllegalArgumentException when the file is not a pockettts generator
     * @throws GgufFormatException      when a key or a tensor differs from the reference graph's
     * @throws IOException when the file cannot be read
     */
    public static PocketTtsClip open(String path) throws IOException {
        return open(path, ModelSourceOptions.fromSystemProperties());
    }

    /** {@link #open(String)} with explicit load {@code options}. */
    public static PocketTtsClip open(String path, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        ModelSource source = ModelSources.open(path, options);
        CpuWeights weights = null;
        try {
            GgufKv kv = GgufKv.of(source);
            requirePocketTts(path, kv);
            TensorCatalog tc = TensorCatalog.of(source);
            InputTransformLoader.Result transforms = InputTransformLoader.load(source, kv, tc,
                    ActivationQuant.WEIGHT_ONLY);
            weights = CpuWeights.open(source, tc, transforms);
            PocketTtsFlow flow = PocketTtsFlow.open(kv, tc, weights);
            PocketTtsMimiDecoder decoder = PocketTtsMimiDecoder.open(kv, tc, weights);
            if (flow.latent() != decoder.latent()) {
                throw new GgufFormatException("pockettts: the flow network"
                        + " produces " + flow.latent() + "-value latents, the mimi decoder takes "
                        + decoder.latent());
            }
            PocketTtsSpeakerEncoder speaker = null;
            if (kv.bool(GgufKeys.CLIP_HAS_AUDIO_ENCODER, false)) {
                speaker = PocketTtsSpeakerEncoder.open(kv, tc, weights);
            }
            return new PocketTtsClip(path, source, weights, transforms, flow, decoder, PocketTtsPack.of(kv),
                    speaker);
        } catch (IOException | RuntimeException | Error e) {
            closeQuietly(weights, source, e);
            throw e;
        }
    }

    private static void requirePocketTts(String path, GgufKv kv) {
        if (!ARCH.equals(kv.archName())) {
            throw new IllegalArgumentException("'" + path + "' is not a pockettts generator: general.architecture is"
                    + " '" + kv.archName() + "', the clip generator files use '" + ARCH + "'");
        }
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!PocketTtsFlow.PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException("'" + path + "' is not a pockettts generator: "
                    + GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is " + (type == null ? "absent" : "'" + type + "'")
                    + ", expected '" + PocketTtsFlow.PROJECTOR_TYPE + "'");
        }
    }

    /** The flow-matching decoder graph (GEN_CODE). */
    public PocketTtsFlow flow() {
        return flow;
    }

    /** The streaming mimi decoder graph (GEN_WAV). */
    public PocketTtsMimiDecoder decoder() {
        return decoder;
    }

    /** The pack settings of the checkpoint variant. */
    public PocketTtsPack pack() {
        return pack;
    }

    /** The speaker encoder of voice cloning, or null when the file carries none. */
    public PocketTtsSpeakerEncoder speakerEncoder() {
        return speakerEncoder;
    }

    /**
     * Binds the generator's weights to the device of {@code ctx}, once: the first call uploads every tensor the flow
     * network and the mimi decoder read, on the context's queue and with its kernel library, and later calls return
     * the same view. The device binding is released with the clip.
     *
     * @throws IOException              when the file cannot be reopened for the weight upload
     * @throws UnsupportedOperationException when a tensor's type cannot be uploaded to this device
     */
    SpeechDeviceWeights bindDevice(GpuForwardContext ctx) throws IOException {
        Objects.requireNonNull(ctx, "ctx");
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("the pockettts generator file '" + path + "' is closed");
            }
            if (deviceWeights != null) {
                return deviceWeights;
            }
            WeightStore store = null;
            GpuWeightUploader uploader = null;
            try {
                store = source.createMappedStore();
                uploader = new GpuWeightUploader(ctx.backend(), ctx.queue(), source.getTensors(), store, ARCH,
                        GpuWeightUploader.config(GpuWeightUploader.TypePolicy.forKernels(ctx.kernels()))
                                .transforms(transforms).build());
                SpeechDeviceWeights device = new SpeechDeviceWeights("pockettts", ctx.ops(), uploader);
                this.deviceStore = store;
                this.deviceUploader = uploader;
                this.deviceWeights = device;
                return device;
            } catch (IOException | RuntimeException | Error e) {
                closeQuietly(uploader, store, e);
                throw e;
            }
        }
    }

    /** The file this generator opened. */
    public String path() {
        return path;
    }

    /** Removes the open file and weights; idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        IOException failure = null;
        for (AutoCloseable c : new AutoCloseable[]{deviceWeights, deviceUploader, deviceStore}) {
            if (c == null) {
                continue;
            }
            try {
                c.close();
            } catch (Exception e) {
                failure = failure == null ? (e instanceof IOException ? (IOException) e
                        : new IOException("closing '" + path + "' failed", e)) : failure;
            }
        }
        try {
            weights.close();
        } catch (IOException e) {
            failure = e;
        }
        try {
            source.close();
        } catch (IOException e) {
            failure = failure == null ? e : failure;
        }
        if (failure != null) {
            throw new IllegalStateException("closing '" + path + "' failed", failure);
        }
    }

    private static void closeQuietly(AutoCloseable first, AutoCloseable second, Throwable failure) {
        for (AutoCloseable c : new AutoCloseable[]{first, second}) {
            if (c == null) {
                continue;
            }
            try {
                c.close();
            } catch (Exception closeFailure) {
                failure.addSuppressed(closeFailure);
            }
        }
    }

    @Override
    public String toString() {
        return "PocketTtsClip{" + path + ", " + flow + ", " + decoder + ", " + pack
                + (speakerEncoder == null ? "" : ", " + speakerEncoder) + '}';
    }
}
