package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.StreamingIstft;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.ops.audio.CpuVocosDecoder;
import com.eonmux.neuralfree.gguf.ops.audio.VocosDecoder;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;

import java.io.File;
import java.io.IOException;
import java.util.Objects;

/**
 * The soprano speech decoder (llama.cpp {@code PROJECTOR_TYPE_SOPRANO}, tools/mtmd/models/soprano.cpp): a GGUF
 * whose {@code clip.gen.audio.projector_type} is {@code soprano} holds the Vocos head
 * ({@link VocosDecoder}) that maps normalized hidden-state frames to polar STFT spectra, and
 * {@link #decode} runs the shared streaming iSTFT ({@code mtmd_audio_streaming_istft}) over them. The audio
 * constants are the loader's (clip.cpp:2224-2229): {@value VocosDecoder#SAMPLE_RATE} Hz, one frame per
 * {@value VocosDecoder#HOP_LENGTH} samples; {@code n_fft} comes from the output head width
 * (soprano.cpp:63-64).
 *
 * <p>{@link #decode} follows {@code clip_graph_soprano::decode_spectrum} (soprano.cpp:62-78): one iSTFT frame
 * per spectrum row, then the head trimming of {@code hop / 2} samples on top of the shared iSTFT's own head
 * trim, and the tail cut to {@code (frames - 1) * hop}. A call refuses fewer than
 * {@value VocosDecoder#MIN_FRAMES} frames (mtmd.cpp:2435). Weights stay on the host in their stored types; the
 * head runs on the CPU. Not thread-safe; close it to release the file and the weights.
 */
public final class SopranoVocoder implements AutoCloseable {

    /** The {@code clip.gen.audio.projector_type} value of this family. */
    public static final String PROJECTOR_TYPE = "soprano";
    /** The {@code general.architecture} of a clip generator file. */
    public static final String ARCH = "clip";
    /** Hparams keys of the generator audio section (clip.cpp:1405-1412 with the {@code gen.audio} prefix). */
    public static final String BLOCK_COUNT_KEY = GgufKeys.CLIP_GEN_AUDIO_BLOCK_COUNT;
    public static final String PROJECTION_DIM_KEY = GgufKeys.CLIP_GEN_AUDIO_PROJECTION_DIM;
    public static final String EPSILON_KEY = GgufKeys.CLIP_GEN_AUDIO_ATTENTION_LAYER_NORM_EPSILON;

    private final String path;
    private final ModelSource source;
    private final CpuWeights weights;
    private final VocosDecoder dec;
    private final CpuVocosDecoder executor;
    private final StreamingIstft istft;
    private boolean closed;

    private SopranoVocoder(String path, ModelSource source, CpuWeights weights, VocosDecoder dec,
                           CpuVocosDecoder executor) {
        this.path = path;
        this.source = source;
        this.weights = weights;
        this.dec = dec;
        this.executor = executor;
        this.istft = new StreamingIstft(dec.nFft(), VocosDecoder.HOP_LENGTH);
    }

    /**
     * Opens the soprano decoder file at {@code path} on the host.
     *
     * @throws IllegalArgumentException when the file is not a soprano generator file
     * @throws GgufFormatException      when a key or a tensor shape differs from the loader's
     * @throws IOException              when the file cannot be read
     */
    public static SopranoVocoder open(String path) throws IOException {
        return open(path, ModelSourceOptions.fromSystemProperties());
    }

    /** {@link #open(String)} with explicit load {@code options}. */
    public static SopranoVocoder open(String path, ModelSourceOptions options) throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        ModelSource source = ModelSources.open(path, options);
        CpuWeights weights = null;
        try {
            GgufKv kv = GgufKv.of(source);
            requireSoprano(path, kv);
            int blocks = kv.u32Required(BLOCK_COUNT_KEY);
            float eps = kv.f32Required(EPSILON_KEY);
            TensorCatalog tc = TensorCatalog.of(source);
            InputTransformLoader.Result transforms = InputTransformLoader.load(source, kv, tc,
                    ActivationQuant.WEIGHT_ONLY);
            weights = CpuWeights.open(source, tc, transforms);
            VocosDecoder dec = VocosDecoder.fromTensors(tc, blocks, eps);
            int declared = kv.u32(PROJECTION_DIM_KEY, 0);
            if (declared > 0 && declared != dec.nInp()) {
                throw new GgufFormatException("soprano: " + PROJECTION_DIM_KEY + " is " + declared
                        + " but the input projection takes " + dec.nInp());
            }
            return new SopranoVocoder(path, source, weights, dec, CpuVocosDecoder.bind(dec, weights));
        } catch (IOException | RuntimeException | Error e) {
            closeQuietly(weights, source, e);
            throw e;
        }
    }

    /** The geometry of the opened file. */
    public VocosDecoder decoder() {
        return dec;
    }

    /**
     * Spectrum frames of {@code features} (polar head included) without the PCM conversion: the executor's
     * {@code [frames][2 * nBins]} interleaved {@code (real, imaginary)} rows.
     *
     * @throws IllegalArgumentException when the batch is empty, too small, or of another width
     * @throws IllegalStateException    when closed
     */
    public float[][] spectrum(float[][] features) {
        ensureOpen();
        return executor.spectrum(features);
    }

    /**
     * PCM of one feature batch: the spectra through the streaming iSTFT, with the trimming of
     * {@code decode_spectrum} — {@code hop / 2} more head samples than the shared iSTFT drops, and a tail of
     * exactly {@code (frames - 1) * hop} samples (soprano.cpp:75-77).
     *
     * @throws IllegalArgumentException when the batch is empty, too small, or of another width
     * @throws IllegalStateException    when closed
     */
    public AudioClip decode(float[][] features) {
        ensureOpen();
        float[][] spectra = spectrum(features);
        int frames = spectra.length;
        int hop = VocosDecoder.HOP_LENGTH;
        istft.reset();
        float[] all = new float[frames * hop + (dec.nFft() - hop)];
        int n = 0;
        for (int f = 0; f < frames; f++) {
            for (float v : istft.processFrame(spectra[f])) {
                all[n++] = v;
            }
        }
        for (float v : istft.flush()) {
            all[n++] = v;
        }
        int head = hop / 2;
        int kept = Math.min((frames - 1) * hop, Math.max(n - head, 0));
        float[] trimmed = new float[kept];
        System.arraycopy(all, head, trimmed, 0, kept);
        return new AudioClip(trimmed, VocosDecoder.SAMPLE_RATE);
    }

    private static void requireSoprano(String path, GgufKv kv) {
        if (!ARCH.equals(kv.archName())) {
            throw new IllegalArgumentException("'" + path + "' is not a soprano decoder: general.architecture is '"
                    + kv.archName() + "', the clip generator files use '" + ARCH + "'");
        }
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException("'" + path + "' is not a soprano decoder: "
                    + GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is " + (type == null ? "absent" : "'" + type + "'")
                    + ", expected '" + PROJECTOR_TYPE + "'");
        }
    }

    /** The file this decoder opened. */
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
            } catch (RuntimeException | Error e) {
                if (failure != e) {
                    failure.addSuppressed(e);
                }
            } catch (Exception e) {
                failure.addSuppressed(e);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the soprano decoder is closed");
        }
    }

    @Override
    public String toString() {
        return "SopranoVocoder{" + new File(path).getName() + ", " + dec + (closed ? ", closed" : "") + "}";
    }
}
