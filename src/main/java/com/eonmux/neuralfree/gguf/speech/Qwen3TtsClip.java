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
 * An opened qwen3tts generator file: the clip GGUF that carries the code predictor and the code2wav DAC decoder
 * beside the backbone, and, when the file declares one, the ECAPA-TDNN speaker encoder of voice cloning. It is the
 * qwen3tts counterpart of {@link com.eonmux.neuralfree.gguf.mm.SopranoVocoder}: llama.cpp loads it as the mmproj
 * ({@code --mmproj}) and the server selects its generator from the file's {@code clip.gen.audio.projector_type}
 * ({@code qwen3tts_gen}).
 *
 * <p>Opening reads the metadata and the tensor table, resolves the file's architecture
 * ({@link Qwen3TtsCodePredictor}, {@link Qwen3TtsCode2Wav} and, with {@code clip.has_audio_encoder}, a
 * {@link Qwen3TtsSpeakerEncoder}) and binds the host weights once; the graphs share them. A GPU engine binds the
 * same tensors to its device once, lazily, through {@link #bindDevice}. The file's geometry is immutable; close the
 * clip to release the mmap, the decoded vectors and any device binding.
 */
public final class Qwen3TtsClip implements AutoCloseable {

    /** The {@code general.architecture} of a clip generator file. */
    public static final String ARCH = "clip";

    private final String path;
    private final ModelSource source;
    private final CpuWeights weights;
    private final InputTransformLoader.Result transforms;
    private final Qwen3TtsCodePredictor predictor;
    private final Qwen3TtsCode2Wav code2Wav;
    private final Qwen3TtsSpeakerEncoder speakerEncoder;
    private GpuWeightUploader deviceUploader;
    private WeightStore deviceStore;
    private SpeechDeviceWeights deviceWeights;
    private boolean closed;

    private Qwen3TtsClip(String path, ModelSource source, CpuWeights weights,
                         InputTransformLoader.Result transforms, Qwen3TtsCodePredictor predictor,
                         Qwen3TtsCode2Wav code2Wav, Qwen3TtsSpeakerEncoder speakerEncoder) {
        this.path = path;
        this.source = source;
        this.weights = weights;
        this.transforms = transforms;
        this.predictor = predictor;
        this.code2Wav = code2Wav;
        this.speakerEncoder = speakerEncoder;
    }

    /**
     * Opens the qwen3tts generator file at {@code path} on the host.
     *
     * @throws IllegalArgumentException when the file is not a qwen3tts code2wav generator
     * @throws GgufFormatException      when a key or a tensor differs from the loader's
     * @throws IOException              when the file cannot be read
     */
    public static Qwen3TtsClip open(String path) throws IOException {
        return open(path, ModelSourceOptions.fromSystemProperties());
    }

    /** {@link #open(String)} with explicit load {@code options}. */
    public static Qwen3TtsClip open(String path, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        ModelSource source = ModelSources.open(path, options);
        CpuWeights weights = null;
        try {
            GgufKv kv = GgufKv.of(source);
            requireQwen3Tts(path, kv);
            TensorCatalog tc = TensorCatalog.of(source);
            InputTransformLoader.Result transforms = InputTransformLoader.load(source, kv, tc,
                    ActivationQuant.WEIGHT_ONLY);
            weights = CpuWeights.open(source, tc, transforms);
            Qwen3TtsCodePredictor predictor = Qwen3TtsCodePredictor.open(kv, tc, weights);
            Qwen3TtsCode2Wav code2Wav = Qwen3TtsCode2Wav.open(kv, tc, weights);
            if (predictor.codesPerFrame() != code2Wav.codesPerFrame()) {
                throw new GgufFormatException("qwen3tts: the code predictor produces " + predictor.codesPerFrame()
                        + " codes per frame, the code2wav quantizer takes " + code2Wav.codesPerFrame());
            }
            Qwen3TtsSpeakerEncoder speaker = null;
            if (kv.bool(GgufKeys.CLIP_HAS_AUDIO_ENCODER, false)) {
                speaker = Qwen3TtsSpeakerEncoder.open(kv, tc, weights);
            }
            return new Qwen3TtsClip(path, source, weights, transforms, predictor, code2Wav, speaker);
        } catch (IOException | RuntimeException | Error e) {
            closeQuietly(weights, source, e);
            throw e;
        }
    }

    private static void requireQwen3Tts(String path, GgufKv kv) {
        if (!ARCH.equals(kv.archName())) {
            throw new IllegalArgumentException("'" + path + "' is not a qwen3tts generator: general.architecture is '"
                    + kv.archName() + "', the clip generator files use '" + ARCH + "'");
        }
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!Qwen3TtsCodePredictor.PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException("'" + path + "' is not a qwen3tts generator: "
                    + GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is " + (type == null ? "absent" : "'" + type + "'")
                    + ", expected '" + Qwen3TtsCodePredictor.PROJECTOR_TYPE + "'");
        }
    }

    /** The code predictor graph. */
    public Qwen3TtsCodePredictor predictor() {
        return predictor;
    }

    /** The code2wav decoder graph. */
    public Qwen3TtsCode2Wav code2Wav() {
        return code2Wav;
    }

    /** The speaker encoder of voice cloning, or null when the file carries none. */
    public Qwen3TtsSpeakerEncoder speakerEncoder() {
        return speakerEncoder;
    }

    /**
     * Binds the generator's weights to the device of {@code ctx}, once: the first call uploads every tensor the
     * predictor and the decoder read, on the context's queue and with its kernel library, and later calls return
     * the same view. The device binding is released with the clip.
     *
     * @throws IOException              when the file cannot be reopened for the weight upload
     * @throws UnsupportedOperationException when a tensor's type cannot be uploaded to this device
     */
    SpeechDeviceWeights bindDevice(GpuForwardContext ctx) throws IOException {
        Objects.requireNonNull(ctx, "ctx");
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("the qwen3tts generator file '" + path + "' is closed");
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
                SpeechDeviceWeights device = new SpeechDeviceWeights("qwen3tts", ctx.ops(), uploader);
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
        for (AutoCloseable c : new AutoCloseable[]{deviceWeights, deviceUploader, deviceStore, weights, source}) {
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
        return "Qwen3TtsClip{" + path + ", " + predictor + ", " + code2Wav
                + (speakerEncoder == null ? "" : ", " + speakerEncoder) + '}';
    }
}
