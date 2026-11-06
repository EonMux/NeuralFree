package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;

/**
 * Validated geometry of the soprano Vocos head (llama.cpp tools/mtmd/models/soprano.cpp, the
 * {@code clip_vocos} tensors): normalized hidden-state frames of width {@link #nInp()} map to polar STFT frames
 * of {@link #nFrameVals()} values, which {@link CpuVocosDecoder} turns into interleaved {@code (real,
 * imaginary)} spectra and a streaming iSTFT turns into PCM.
 *
 * <ol>
 *   <li>bilinear upsample of every channel along time, {@code align_corners} with the factor
 *       {@value #UPSCALE}: {@code n_frames -> upscale * (n_frames - 1) + 1};</li>
 *   <li>the input projection, a same-padding {@code conv1d} when {@code input_w} is 3-D
 *       ({@code [K, n_inp, C]}) and a pointwise matrix product otherwise ({@code [n_inp, C]}), plus bias and a
 *       LayerNorm over channels per frame;</li>
 *   <li>the Vocos blocks: depthwise same-padding {@code conv1d} ({@code dwconv}, kernel {@link #DW_K}),
 *       LayerNorm over channels, {@code pw1}, GELU (erf form, {@code ggml_gelu_erf}), {@code pw2}, the
 *       per-channel layer scale {@code gamma}, residual;</li>
 *   <li>a final LayerNorm and the output projection plus bias: {@code n_frame_vals} values per frame — the
 *       first {@code n_bins = n_frame_vals / 2} log-magnitudes, then {@code n_bins} phases (soprano.cpp:45-50),
 *       {@code n_fft = n_frame_vals - 2} (soprano.cpp:64).</li>
 * </ol>
 *
 * <p>Tensor names are the clip generator names (tools/mtmd/clip-impl.h:320-323, 271-276); shapes follow the
 * loader (clip.cpp:3662-3684) in ggml dimension order, {@code ne[0]} first. Immutable.
 */
public final class VocosDecoder {

    /** Time upscale of the input projection stage (models.h:438, the SopranoDecoder constant). */
    public static final int UPSCALE = 4;
    /** Kernel width of the block depthwise convolutions (dwconv). */
    public static final int DW_K = 7;
    /** Soprano refuses feature batches below this frame count (mtmd.cpp:2435). */
    public static final int MIN_FRAMES = 2;

    /** Soprano decoder audio constants (tools/mtmd/clip.cpp:2224-2229): the model speaks at 32 kHz. */
    public static final int SAMPLE_RATE = 32000;
    /** Soprano decoder audio constants (tools/mtmd/clip.cpp:2224-2229): 512 samples per frame. */
    public static final int HOP_LENGTH = 512;

    /** Tensor base names of the head (clip-impl.h {@code TN_A_GEN_WAV_INPUT} and friends). */
    public static final String INPUT = "a.gen.wav.input";
    public static final String NORM = "a.gen.wav.norm";
    public static final String OUTPUT_NORM = "a.gen.wav.output_norm";
    public static final String OUTPUT = "a.gen.wav.output";

    private final int nInp;
    private final int width;
    private final int nFf;
    private final int nFrameVals;
    private final int convK;
    private final int blocks;
    private final float eps;

    private VocosDecoder(int nInp, int width, int nFf, int nFrameVals, int convK, int blocks, float eps) {
        this.nInp = nInp;
        this.width = width;
        this.nFf = nFf;
        this.nFrameVals = nFrameVals;
        this.convK = convK;
        this.blocks = blocks;
        this.eps = eps;
    }

    /**
     * Geometry of a resolved soprano file: {@code blocks} Vocos blocks, the LayerNorm epsilon of the
     * {@code clip.gen.audio.attention.layer_norm_epsilon} key, every width fixed by the tensors.
     *
     * @throws IllegalArgumentException when {@code blocks} or {@code eps} is invalid
     * @throws GgufFormatException      when a tensor is missing or a shape differs from the loader's
     */
    public static VocosDecoder fromTensors(TensorCatalog tc, int blocks, float eps) {
        Objects.requireNonNull(tc, "tensors");
        if (blocks <= 0) {
            throw new GgufFormatException("soprano: the Vocos block count must be positive, got " + blocks);
        }
        if (!(eps >= 0f) || Float.isInfinite(eps)) {
            throw new GgufFormatException("soprano: layer norm epsilon must be finite and non-negative, got " + eps);
        }
        TensorRef input = tc.require(weight(INPUT));
        int nInp;
        int convK;
        if (input.rank() >= 3) {
            convK = dim(input, 0);
            nInp = dim(input, 1);
        } else {
            convK = 0;
            nInp = dim(input, 0);
        }
        int width = dim(input, input.rank() - 1);
        vector(tc, bias(INPUT), width);
        vector(tc, weight(NORM), width);
        vector(tc, bias(NORM), width);

        int nFf = -1;
        for (int il = 0; il < blocks; il++) {
            int layerFf = checkBlock(tc, il, width);
            if (nFf >= 0 && layerFf != nFf) {
                throw new GgufFormatException("soprano: Vocos block " + il + " has " + layerFf
                        + " hidden units, block 0 has " + nFf + " (llama.cpp uses one pw width)");
            }
            nFf = layerFf;
        }
        vector(tc, weight(OUTPUT_NORM), width);
        vector(tc, bias(OUTPUT_NORM), width);
        TensorRef output = tc.require(weight(OUTPUT));
        int nFrameVals = dim(output, 1);
        shape(output, width, nFrameVals);
        vector(tc, bias(OUTPUT), nFrameVals);
        if (nFrameVals % 2 != 0 || nFrameVals < 4) {
            throw new GgufFormatException("soprano: the output head holds " + nFrameVals
                    + " values per frame, an even count of at least two magnitude and two phase bins is needed"
                    + " (soprano.cpp:46-47)");
        }
        return new VocosDecoder(nInp, width, nFf, nFrameVals, convK, blocks, eps);
    }

    private static int checkBlock(TensorCatalog tc, int il, int c) {
        shape(tc.require(dwconv(il, "weight")), DW_K, 1, c);
        vector(tc, dwconv(il, "bias"), c);
        vector(tc, block(il, "norm", "weight"), c);
        vector(tc, block(il, "norm", "bias"), c);
        TensorRef pw1 = tc.require(block(il, "pw1", "weight"));
        int ff = dim(pw1, 1);
        shape(pw1, c, ff);
        vector(tc, block(il, "pw1", "bias"), ff);
        shape(tc.require(block(il, "pw2", "weight")), ff, c);
        vector(tc, block(il, "pw2", "bias"), c);
        vector(tc, "a.gen.wav.up.blk." + il + ".gamma", c);
        return ff;
    }

    /** {@code <base>.weight}. */
    public static String weight(String base) {
        return base + ".weight";
    }

    /** {@code <base>.bias}. */
    public static String bias(String base) {
        return base + ".bias";
    }

    /** {@code a.gen.wav.up.blk.<il>.<part>.<kind>} of the block stack. */
    public static String block(int il, String part, String kind) {
        return "a.gen.wav.up.blk." + il + "." + part + "." + kind;
    }

    /** {@code a.gen.wav.up.blk.<il>.dwconv.<kind>}. */
    public static String dwconv(int il, String kind) {
        return "a.gen.wav.up.blk." + il + ".dwconv." + kind;
    }

    // ------------------------------------------------------------------ calls

    /**
     * Validates one feature batch: at least {@value #MIN_FRAMES} frames (mtmd.cpp:2435 refuses fewer, the head
     * is a generator over whole utterances) and the input width of the file.
     *
     * @throws IllegalArgumentException otherwise
     */
    public void checkFeatures(int frames, int width) {
        if (frames < MIN_FRAMES) {
            throw new IllegalArgumentException("soprano needs at least " + MIN_FRAMES + " frames of features, got "
                    + frames + " (mtmd.cpp:2435)");
        }
        if (width != nInp) {
            throw new IllegalArgumentException("soprano features are " + width + " wide, the model takes " + nInp);
        }
    }

    /**
     * Validates the capacity of one decode call: every {@code [frames][width]} intermediate inside the 32-bit
     * element range.
     *
     * @throws IllegalArgumentException otherwise
     */
    public void checkCapacity(int frames) {
        if (frames <= 0) {
            throw new IllegalArgumentException("frame count must be positive, got " + frames);
        }
        long upsampled = upsampledFrames(frames);
        long widest = Math.max(Math.max(width, nFf), Math.max(nInp, nFrameVals));
        if (widest * Math.max(frames, upsampled) > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("soprano capacity " + frames + " frames x width " + widest
                    + " exceeds the 32-bit element range");
        }
    }

    /** Frames after the bilinear upsample: {@code upscale * (frames - 1) + 1} (soprano.cpp:8). */
    public int upsampledFrames(int frames) {
        return UPSCALE * (frames - 1) + 1;
    }

    /** Input width of the feature frames ({@code clip.gen.audio.projection_dim}). */
    public int nInp() {
        return nInp;
    }

    /** Vocos width {@code C}: the depthwise convolutions, the norms and the pointwise projections. */
    public int width() {
        return width;
    }

    /** Pointwise hidden width ({@code pw1} rows). */
    public int nFf() {
        return nFf;
    }

    /** Values per output frame: {@code 2 * n_bins}, log-magnitudes then phases. */
    public int nFrameVals() {
        return nFrameVals;
    }

    /** Spectrum bins per frame ({@code n_frame_vals / 2}). */
    public int nBins() {
        return nFrameVals / 2;
    }

    /** Transform size of the spectra ({@code n_frame_vals - 2}, soprano.cpp:64). */
    public int nFft() {
        return nFrameVals - 2;
    }

    /** Kernel width of the input projection; 0 when the projection is a pointwise matrix product. */
    public int convK() {
        return convK;
    }

    /** Vocos block count ({@code clip.gen.audio.block_count}). */
    public int blocks() {
        return blocks;
    }

    /** LayerNorm epsilon ({@code clip.gen.audio.attention.layer_norm_epsilon}). */
    public float eps() {
        return eps;
    }

    @Override
    public String toString() {
        return "VocosDecoder{inp=" + nInp + ", C=" + width + ", ff=" + nFf + ", frameVals=" + nFrameVals
                + ", fft=" + nFft() + ", convK=" + convK + ", blocks=" + blocks + ", eps=" + eps + "}";
    }

    // ------------------------------------------------------------------ validation helpers

    private static void vector(TensorCatalog tc, String name, int c) {
        shape(tc.require(name), c);
    }

    /** {@code ne[i]} of a tensor as a positive int. */
    private static int dim(TensorRef r, int i) {
        long v = r.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("soprano: tensor " + r.name() + " has dimension " + i + " = " + v);
        }
        return (int) v;
    }

    /** The llama.cpp loader's exact shape check: listed dims match, every later dim is 1. */
    private static void shape(TensorRef r, long... ne) {
        boolean ok = true;
        for (int i = ne.length; i < r.rank(); i++) {
            ok = r.ne(i) == 1;
        }
        for (int i = 0; i < ne.length && ok; i++) {
            ok = r.ne(i) == ne[i];
        }
        if (!ok) {
            StringBuilder want = new StringBuilder("{");
            for (int i = 0; i < ne.length; i++) {
                want.append(i == 0 ? "" : ", ").append(ne[i]);
            }
            throw new GgufFormatException("soprano: tensor " + r.name() + " has shape " + r
                    + ", the loader expects " + want.append('}'));
        }
    }
}
