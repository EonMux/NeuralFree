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
 * An opened lfm3a generator file: the clip GGUF that carries the MaskGIT depthformer and the liquid-codec decoder
 * beside the LFM3-Audio backbone. It is the lfm3a counterpart of {@link Qwen3TtsClip}: llama.cpp loads it as the
 * mmproj ({@code --mmproj}) and selects the generator from the file's
 * {@code clip.gen.audio.projector_type} ({@code lfm3a_gen}).
 *
 * <p>Opening reads the metadata and the tensor table, resolves the file's two graphs
 * ({@link Lfm3aDepthformer} and {@link Lfm3aCodec}) and binds the host weights once; the graphs share them.
 * The depthformer's codebook count and vocabulary must match the codec's FSQ table, since every frame the
 * depthformer draws is decoded by the codec. The speech-code sampling defaults are the LFM3 model card's
 * (clip.cpp:2180-2183), overridable by the file's {@code clip.gen.audio.sampling.*} keys. Immutable over the
 * file; close it to release the mmap and the decoded vectors.
 */
public final class Lfm3aGenClip implements AutoCloseable {

    /** The {@code general.architecture} of a clip generator file. */
    public static final String ARCH = "clip";

    /** The model-card speech sampling temperature the reference reads when the file omits the key. */
    public static final float DEFAULT_TEMP = 0.4f;

    /** The default top-k, off: every codebook entry is a candidate. */
    public static final int DEFAULT_TOP_K = 0;

    /** The default code repetition penalty. */
    public static final float DEFAULT_PENALTY_REPEAT = 1.2f;

    /** The default repetition window in frames. */
    public static final int DEFAULT_PENALTY_LAST_N = 64;

    private final String path;
    private final ModelSource source;
    private final CpuWeights weights;
    private final Lfm3aDepthformer depthformer;
    private final Lfm3aCodec codec;
    private final float temp;
    private final int topK;
    private final float penaltyRepeat;
    private final int penaltyLastN;
    private final InputTransformLoader.Result transforms;
    private GpuWeightUploader deviceUploader;
    private WeightStore deviceStore;
    private SpeechDeviceWeights deviceWeights;
    private boolean closed;

    private Lfm3aGenClip(String path, ModelSource source, CpuWeights weights,
                         InputTransformLoader.Result transforms, Lfm3aDepthformer depthformer,
                         Lfm3aCodec codec, float temp, int topK, float penaltyRepeat, int penaltyLastN) {
        this.path = path;
        this.source = source;
        this.weights = weights;
        this.transforms = transforms;
        this.depthformer = depthformer;
        this.codec = codec;
        this.temp = temp;
        this.topK = topK;
        this.penaltyRepeat = penaltyRepeat;
        this.penaltyLastN = penaltyLastN;
    }

    /**
     * Opens the lfm3a generator file at {@code path} on the host.
     *
     * @throws IllegalArgumentException when the file is not an lfm3a generator
     * @throws GgufFormatException      when a key or a tensor differs from the loader's
     * @throws IOException              when the file cannot be read
     */
    public static Lfm3aGenClip open(String path) throws IOException {
        return open(path, ModelSourceOptions.fromSystemProperties());
    }

    /** {@link #open(String)} with explicit load {@code options}. */
    public static Lfm3aGenClip open(String path, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        ModelSource source = ModelSources.open(path, options);
        CpuWeights weights = null;
        try {
            GgufKv kv = GgufKv.of(source);
            requireLfm3a(path, kv);
            TensorCatalog tc = TensorCatalog.of(source);
            InputTransformLoader.Result transforms = InputTransformLoader.load(source, kv, tc,
                    ActivationQuant.WEIGHT_ONLY);
            weights = CpuWeights.open(source, tc, transforms);
            Lfm3aDepthformer depthformer = Lfm3aDepthformer.open(kv, tc, weights);
            Lfm3aCodec codec = Lfm3aCodec.open(kv, tc, weights, depthformer.nCodebooks());
            if (codec.vocab() + 1 != depthformer.vocab()) {
                // the FSQ table indexes the codebook only: the end code has no waveform (lfm3.py:476-477,
                // clip.cpp:6235), so its row is absent and the codec's table is one row short of the vocabulary
                throw new GgufFormatException("lfm3a: the depthformer draws codes from a vocabulary of "
                        + depthformer.vocab() + ", the codec's FSQ table holds " + codec.vocab()
                        + " rows; the table must cover the codebook, one row short of the vocabulary");
            }
            // the sampling defaults of mtmd_gen_inp_default, overridable by the model's clip.gen.audio keys
            float temp = kv.f32(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_TEMP, DEFAULT_TEMP);
            int topK = kv.u32(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_TOP_K, DEFAULT_TOP_K);
            float penaltyRepeat = kv.f32(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_PENALTY_REPEAT, DEFAULT_PENALTY_REPEAT);
            int penaltyLastN = kv.u32(GgufKeys.CLIP_GEN_AUDIO_SAMPLING_PENALTY_LAST_N, DEFAULT_PENALTY_LAST_N);
            if (!(temp >= 0f) || Float.isInfinite(temp)) {
                throw new GgufFormatException("lfm3a: " + GgufKeys.CLIP_GEN_AUDIO_SAMPLING_TEMP
                        + " must be a non-negative finite temperature, got " + temp);
            }
            if (topK < 0) {
                throw new GgufFormatException("lfm3a: " + GgufKeys.CLIP_GEN_AUDIO_SAMPLING_TOP_K
                        + " must not be negative, got " + topK);
            }
            if (!(penaltyRepeat >= 0f) || Float.isInfinite(penaltyRepeat)) {
                throw new GgufFormatException("lfm3a: " + GgufKeys.CLIP_GEN_AUDIO_SAMPLING_PENALTY_REPEAT
                        + " must be a non-negative finite penalty, got " + penaltyRepeat);
            }
            if (penaltyLastN < 0) {
                throw new GgufFormatException("lfm3a: " + GgufKeys.CLIP_GEN_AUDIO_SAMPLING_PENALTY_LAST_N
                        + " must not be negative, got " + penaltyLastN);
            }
            return new Lfm3aGenClip(path, source, weights, transforms, depthformer, codec, temp, topK,
                    penaltyRepeat, penaltyLastN);
        } catch (IOException | RuntimeException | Error e) {
            closeQuietly(weights, source, e);
            throw e;
        }
    }

    private static void requireLfm3a(String path, GgufKv kv) {
        if (!ARCH.equals(kv.archName())) {
            throw new IllegalArgumentException("'" + path + "' is not an lfm3a generator: general.architecture is '"
                    + kv.archName() + "', the clip generator files use '" + ARCH + "'");
        }
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!Lfm3aDepthformer.PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException("'" + path + "' is not an lfm3a generator: "
                    + GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is " + (type == null ? "absent" : "'" + type + "'")
                    + ", expected '" + Lfm3aDepthformer.PROJECTOR_TYPE + "'");
        }
    }

    /** The MaskGIT depthformer graph. */
    public Lfm3aDepthformer depthformer() {
        return depthformer;
    }

    /** The liquid-codec decoder graph; sessions over it carry the sliding-window states. */
    public Lfm3aCodec codec() {
        return codec;
    }

    /** The codebook vocabulary of the speech codes, end code included. */
    public int vocab() {
        return depthformer.vocab();
    }

    /** The codes the codec FSQ table can decode; the end code {@code vocab() - 1} has no waveform. */
    public int fsqVocab() {
        return codec.vocab();
    }

    /** The speech sampling temperature ({@code clip.gen.audio.sampling.temp} of the file). */
    public float temp() {
        return temp;
    }

    /** The speech sampling top-k, 0 for off. */
    public int topK() {
        return topK;
    }

    /** The code repetition penalty. */
    public float penaltyRepeat() {
        return penaltyRepeat;
    }

    /** The repetition window in frames. */
    public int penaltyLastN() {
        return penaltyLastN;
    }

    /**
     * Binds the generator's weights to the device of {@code ctx}, once: the first call uploads every tensor the
     * depthformer and the liquid codec read, on the context's queue and with its kernel library, and later calls
     * return the same view. The device binding is released with the clip.
     *
     * @throws IOException              when the file cannot be reopened for the weight upload
     * @throws UnsupportedOperationException when a tensor's type cannot be uploaded to this device
     */
    SpeechDeviceWeights bindDevice(GpuForwardContext ctx) throws IOException {
        Objects.requireNonNull(ctx, "ctx");
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("the lfm3a generator file '" + path + "' is closed");
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
                SpeechDeviceWeights device = new SpeechDeviceWeights("lfm3a", ctx.ops(), uploader);
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
        return "Lfm3aGenClip{" + path + ", " + depthformer + ", " + codec + '}';
    }
}
