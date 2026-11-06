package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The lfm3a liquid-codec decoder geometry of llama.cpp's {@code clip_graph_lfm3a_gen::build_gen_wav}
 * (tools/mtmd/models/lfm3a-gen.cpp:172-251): a batch of codec frames becomes polar STFT frames, which the host
 * turns into PCM through the shared {@link com.eonmux.neuralfree.gguf.audio.StreamingIstft}.
 *
 * <ol>
 *   <li>the FSQ level table (every codebook shares it) maps the codes to level vectors, one projection over the
 *       frame's stacked code vectors and the in-projection width the decoder transformer to
 *       {@code n_embd_dec};</li>
 *   <li>nearest upsampling repeats every codec frame over {@code frame_stack} decoder positions, whose positions
 *       continue across calls from the state's position slot;</li>
 *   <li>eight LFM2 decoder layers — short convolutions with persisted taps or sliding-window gated attention with
 *       persisted key/value prefixes, both with the LFM3 sandwich norms — run under a causal window mask over
 *       [cached window | this batch];</li>
 *   <li>the output norm and projection produce per position {@code 2 * n_bins} values: the log magnitude, then
 *       the phase.</li>
 * </ol>
 *
 * <p>State slots of the reference's {@code list_lfm3a_gen_state_slots}: the position counter and, per layer, the
 * conv taps or the key/value window. Tensors use the clip generator names of tools/mtmd/clip-impl.h in ggml
 * dimension order. The reference hardcodes the geometry (8 layers of 512 values, window 30, frame stack 6, 24 kHz,
 * n_fft 1280, hop 320, rope base 1e6); the transformer head counts come from the tensors ({@code q_norm} width).
 * {@code clip.gen.wav.window}, {@code clip.gen.wav.frame_stack}, {@code clip.gen.wav.n_fft} and
 * {@code clip.gen.wav.hop_length} override the hardcoded values when a file declares a different geometry.
 */
public final class Lfm3aCodec {

    /** The sample rate of the decoded waveform (clip.cpp:2174, lfm3a 24 kHz). */
    public static final int SAMPLE_RATE = 24000;
    /** The reference decoder's sliding window in decoder positions (clip.cpp:2171, decoder_window). */
    public static final int DEFAULT_WINDOW = 30;
    /** The reference frame stack, STFT frames per codec frame (clip.cpp:2172, stft_frame_stack). */
    public static final int DEFAULT_FRAME_STACK = 6;
    /** The reference transform size (clip.cpp:2175). */
    public static final int DEFAULT_N_FFT = 1280;
    /** The reference samples between frames (clip.cpp:2177). */
    public static final int DEFAULT_HOP = 320;
    /** The reference decoder epsilon (clip.cpp:2169). */
    public static final float EPS = 1e-5f;
    /** The reference decoder RoPE base (clip.cpp:2170). */
    public static final float ROPE_THETA = 1000000.0f;

    /** Optional geometry keys of an override (llama.cpp hardcodes the values). */
    public static final String WINDOW_KEY = "clip.gen.wav.window";
    public static final String FRAME_STACK_KEY = "clip.gen.wav.frame_stack";
    public static final String N_FFT_KEY = "clip.gen.wav.n_fft";
    public static final String HOP_KEY = "clip.gen.wav.hop_length";

    /** Tensor base names of tools/mtmd/clip-impl.h:261-270. */
    public static final String QUANT_FIRST_CB = "a.gen.wav.quant.first.codebook";
    public static final String QUANT_FIRST_OUT = "a.gen.wav.quant.first.out_proj";
    public static final String TFM_IN_PROJ = "a.gen.wav.tfm.in_proj";
    public static final String TFM_OUT_PROJ = "a.gen.wav.tfm.out_proj";
    public static final String TFM_OUTPUT_NORM = "a.gen.wav.tfm.output_norm";

    /**
     * What the speech pipeline decodes through: one stateful liquid-codec stream. Implemented by the host
     * {@link Lfm3aCodecSession} and by the device executor of a GPU engine.
     */
    public interface Decoder extends AutoCloseable {
        /** {@link Lfm3aCodecSession#decodePolar(int[])}. */
        float[] decodePolar(int[] codes);

        /** {@link Lfm3aCodecSession#decodeClip(int[])}. */
        com.eonmux.neuralfree.gguf.audio.AudioClip decodeClip(int[] codes);

        /** {@link Lfm3aCodecSession#flushClip()}. */
        com.eonmux.neuralfree.gguf.audio.AudioClip flushClip();

        /** The decoder positions consumed so far. */
        int position();

        /** Back to a cold stream. */
        void reset();

        /** Releases the stream's buffers; idempotent. */
        @Override
        void close();
    }

    /** The GGUF name of one tensor of decoder layer {@code index}. */
    static String layerTensor(int index, String part) {
        return "a.gen.wav.tfm.blk." + index + "." + part;
    }

    /** Layer tensor suffixes ({@code a.gen.wav.tfm.blk.<il>.<part>}). */
    static final String LN1 = "ln1.weight";
    static final String LN2 = "ln2.weight";
    static final String ATTN_Q = "attn_q.weight";
    static final String ATTN_Q_NORM = "attn_q_norm.weight";
    static final String ATTN_K = "attn_k.weight";
    static final String ATTN_K_NORM = "attn_k_norm.weight";
    static final String ATTN_V = "attn_v.weight";
    static final String ATTN_OUT = "attn_out.weight";
    static final String ATTN_GATE = "attn_gate.weight";
    static final String ATTN_POST_NORM = "attn_post_norm.weight";
    static final String FFN_POST_NORM = "ffn_post_norm.weight";
    static final String FFN_GATE = "ffn_gate.weight";
    static final String FFN_UP = "ffn_up.weight";
    static final String FFN_DOWN = "ffn_down.weight";
    static final String CONV_PW1 = "conv_pw1.weight";
    static final String CONV_DW = "conv_dw.weight";
    static final String CONV_PW2 = "conv_pw2.weight";

    /** One decoder layer: a short conv block or a gated attention block, both with sandwich norms. */
    static final class Layer {
        final float[] ln1;
        final float[] ln2;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;
        final float[] attnPostNorm;
        final float[] ffPostNorm;
        // conv block
        final CpuTensorRef convPw1;
        final CpuTensorRef convDw;
        final CpuTensorRef convPw2;
        final int taps;
        // attention block
        final CpuTensorRef q;
        final float[] qNorm;
        final CpuTensorRef k;
        final float[] kNorm;
        final CpuTensorRef v;
        final CpuTensorRef out;
        final CpuTensorRef attnGate;
        final int nHead;
        final int nHeadKv;
        final int dHead;

        Layer(float[] ln1, float[] ln2, CpuTensorRef gate, CpuTensorRef up, CpuTensorRef down,
              float[] attnPostNorm, float[] ffPostNorm, CpuTensorRef convPw1, CpuTensorRef convDw,
              CpuTensorRef convPw2, int taps, CpuTensorRef q, float[] qNorm, CpuTensorRef k, float[] kNorm,
              CpuTensorRef v, CpuTensorRef out, CpuTensorRef attnGate, int nHead, int nHeadKv, int dHead) {
            this.ln1 = ln1;
            this.ln2 = ln2;
            this.gate = gate;
            this.up = up;
            this.down = down;
            this.attnPostNorm = attnPostNorm;
            this.ffPostNorm = ffPostNorm;
            this.convPw1 = convPw1;
            this.convDw = convDw;
            this.convPw2 = convPw2;
            this.taps = taps;
            this.q = q;
            this.qNorm = qNorm;
            this.k = k;
            this.kNorm = kNorm;
            this.v = v;
            this.out = out;
            this.attnGate = attnGate;
            this.nHead = nHead;
            this.nHeadKv = nHeadKv;
            this.dHead = dHead;
        }

        /** True when the layer runs the short convolution. */
        boolean isConv() {
            return convDw != null;
        }
    }

    private final int nCodebooks;
    private final int nLvl;
    private final int vocab;
    private final int nEmbdDec;
    private final int frameStack;
    private final int window;
    private final int nFft;
    private final int hop;
    private final CpuWeights weights;
    private final CpuTensorRef quantFirstCb;
    private final CpuTensorRef quantFirstOut;
    private final float[] quantFirstOutB;
    private final CpuTensorRef tfmIn;
    private final float[] tfmInB;
    private final CpuTensorRef tfmOut;
    private final float[] tfmOutB;
    private final float[] tfmOutNorm;
    private final Layer[] layers;

    private Lfm3aCodec(int nCodebooks, int nLvl, int vocab, int nEmbdDec, int frameStack, int window, int nFft,
                       int hop, CpuWeights weights, CpuTensorRef quantFirstCb, CpuTensorRef quantFirstOut,
                       float[] quantFirstOutB, CpuTensorRef tfmIn, float[] tfmInB, CpuTensorRef tfmOut,
                       float[] tfmOutB, float[] tfmOutNorm, Layer[] layers) {
        this.nCodebooks = nCodebooks;
        this.nLvl = nLvl;
        this.vocab = vocab;
        this.nEmbdDec = nEmbdDec;
        this.frameStack = frameStack;
        this.window = window;
        this.nFft = nFft;
        this.hop = hop;
        this.weights = weights;
        this.quantFirstCb = quantFirstCb;
        this.quantFirstOut = quantFirstOut;
        this.quantFirstOutB = quantFirstOutB;
        this.tfmIn = tfmIn;
        this.tfmInB = tfmInB;
        this.tfmOut = tfmOut;
        this.tfmOutB = tfmOutB;
        this.tfmOutNorm = tfmOutNorm;
        this.layers = layers;
    }

    /**
     * Resolves the codec decoder of an opened clip generator file.
     *
     * @param nCodebooks the depthformer's codebook count, whose frames this decoder takes
     * @throws IllegalArgumentException when the file is not an lfm3a codec decoder
     * @throws GgufFormatException      when a key or a tensor shape differs from the loader's
     */
    public static Lfm3aCodec open(GgufKv kv, TensorCatalog tc, CpuWeights weights, int nCodebooks) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!Lfm3aDepthformer.PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is "
                    + (type == null ? "absent" : "'" + type + "'") + ", an lfm3a codec decoder declares '"
                    + Lfm3aDepthformer.PROJECTOR_TYPE + "'");
        }
        if (nCodebooks <= 0) {
            throw new IllegalArgumentException("nCodebooks must be positive, got " + nCodebooks);
        }
        int window = positiveOverride(kv, WINDOW_KEY, DEFAULT_WINDOW, "window");
        int frameStack = positiveOverride(kv, FRAME_STACK_KEY, DEFAULT_FRAME_STACK, "frame_stack");
        int nFft = positiveOverride(kv, N_FFT_KEY, DEFAULT_N_FFT, "n_fft");
        int hop = positiveOverride(kv, HOP_KEY, DEFAULT_HOP, "hop_length");
        if (hop > nFft) {
            throw new GgufFormatException("lfm3a codec: " + HOP_KEY + " " + hop + " exceeds " + N_FFT_KEY + " "
                    + nFft);
        }

        TensorRef cbRef = tc.require(QUANT_FIRST_CB + ".weight");
        int nLvl = dim(cbRef, 0);
        int vocab = dim(cbRef, 1);
        shape(cbRef, nLvl, vocab);
        if (nLvl <= 0) {
            throw new GgufFormatException("lfm3a codec: " + QUANT_FIRST_CB + ".weight holds no level table");
        }
        TensorRef firstOutRef = tc.require(QUANT_FIRST_OUT + ".weight");
        int nEmbdDec = dim(firstOutRef, 1);
        shape(firstOutRef, (long) nLvl * nCodebooks, nEmbdDec);
        CpuTensorRef quantFirstOut = weights.get(QUANT_FIRST_OUT + ".weight");
        float[] quantFirstOutB = weights.floats(QUANT_FIRST_OUT + ".bias");
        checkShape(quantFirstOut, (long) nLvl * nCodebooks, nEmbdDec);
        vector(quantFirstOutB, QUANT_FIRST_OUT + ".bias", nEmbdDec);

        CpuTensorRef tfmIn = weights.get(TFM_IN_PROJ + ".weight");
        float[] tfmInB = weights.floats(TFM_IN_PROJ + ".bias");
        checkShape(tfmIn, nEmbdDec, nEmbdDec);
        vector(tfmInB, TFM_IN_PROJ + ".bias", nEmbdDec);
        TensorRef tfmOutRef = tc.require(TFM_OUT_PROJ + ".weight");
        shape(tfmOutRef, nEmbdDec, nFft + 2);
        CpuTensorRef tfmOut = weights.get(TFM_OUT_PROJ + ".weight");
        float[] tfmOutB = weights.floats(TFM_OUT_PROJ + ".bias");
        checkShape(tfmOut, nEmbdDec, nFft + 2);
        vector(tfmOutB, TFM_OUT_PROJ + ".bias", nFft + 2);
        float[] tfmOutNorm = weights.floats(TFM_OUTPUT_NORM + ".weight");
        vector(tfmOutNorm, TFM_OUTPUT_NORM + ".weight", nEmbdDec);

        int nLayer = Qwen3TtsCode2Wav.countIndexed(tc, "a.gen.wav.tfm.blk.");
        if (nLayer == 0) {
            throw new GgufFormatException("lfm3a codec: the file holds no a.gen.wav.tfm.blk.* layer");
        }
        Layer[] layers = new Layer[nLayer];
        Integer expHead = null;
        Integer expHeadKv = null;
        Integer expDHead = null;
        for (int il = 0; il < nLayer; il++) {
            Layer l = layer(weights, il, nEmbdDec);
            if (!l.isConv()) {
                if (expDHead == null) {
                    expDHead = l.dHead;
                    expHead = l.nHead;
                    expHeadKv = l.nHeadKv;
                } else if (expDHead != l.dHead || expHead != l.nHead || expHeadKv != l.nHeadKv) {
                    throw new GgufFormatException("lfm3a codec: the attention layers disagree on the head geometry"
                            + " (first " + expHead + "/" + expHeadKv + " of " + expDHead + ", layer " + il + " "
                            + l.nHead + "/" + l.nHeadKv + " of " + l.dHead + ")");
                }
                if (l.dHead % 2 != 0) {
                    throw new GgufFormatException("lfm3a codec: head dim " + l.dHead + " must be even for RoPE");
                }
            }
            layers[il] = l;
        }
        return new Lfm3aCodec(nCodebooks, nLvl, vocab, nEmbdDec, frameStack, window, nFft, hop, weights,
                weights.get(QUANT_FIRST_CB + ".weight"), quantFirstOut, quantFirstOutB, tfmIn, tfmInB, tfmOut,
                tfmOutB, tfmOutNorm, layers);
    }

    private static Layer layer(CpuWeights weights, int il, int nEmbdDec) {
        String prefix = "a.gen.wav.tfm.blk." + il + ".";
        float[] ln1 = weights.floats(prefix + LN1);
        float[] ln2 = weights.floats(prefix + LN2);
        CpuTensorRef gate = weights.get(prefix + FFN_GATE);
        CpuTensorRef up = weights.get(prefix + FFN_UP);
        CpuTensorRef down = weights.get(prefix + FFN_DOWN);
        float[] attnPostNorm = weights.floatsOrNull(prefix + ATTN_POST_NORM);
        float[] ffPostNorm = weights.floatsOrNull(prefix + FFN_POST_NORM);
        vector(ln1, prefix + LN1, nEmbdDec);
        vector(ln2, prefix + LN2, nEmbdDec);
        if (attnPostNorm != null) {
            vector(attnPostNorm, prefix + ATTN_POST_NORM, nEmbdDec);
        }
        if (ffPostNorm != null) {
            vector(ffPostNorm, prefix + FFN_POST_NORM, nEmbdDec);
        }
        int ff = (int) gate.rows();
        checkShape(gate, nEmbdDec, ff);
        checkShape(up, nEmbdDec, ff);
        checkShape(down, ff, nEmbdDec);

        CpuTensorRef convPw1 = weights.getOrNull(prefix + CONV_PW1);
        CpuTensorRef convDw = weights.getOrNull(prefix + CONV_DW);
        CpuTensorRef convPw2 = weights.getOrNull(prefix + CONV_PW2);
        CpuTensorRef q = weights.getOrNull(prefix + ATTN_Q);
        float[] qNorm = weights.floatsOrNull(prefix + ATTN_Q_NORM);
        CpuTensorRef k = weights.getOrNull(prefix + ATTN_K);
        float[] kNorm = weights.floatsOrNull(prefix + ATTN_K_NORM);
        CpuTensorRef v = weights.getOrNull(prefix + ATTN_V);
        CpuTensorRef out = weights.getOrNull(prefix + ATTN_OUT);
        CpuTensorRef attnGate = weights.getOrNull(prefix + ATTN_GATE);
        boolean isConv = convPw1 != null && convDw != null && convPw2 != null;
        boolean isAttn = q != null && k != null && v != null && out != null && qNorm != null && kNorm != null
                && attnGate != null;
        if (isConv == isAttn) {
            throw new GgufFormatException("lfm3a codec: layer " + il + " is neither a conv nor an attention block"
                    + " (clip.cpp requires exactly one of the two tensor sets)");
        }
        int taps = -1;
        if (isConv) {
            taps = (int) convDw.cols();
            if (taps < 2) {
                throw new GgufFormatException("lfm3a codec: " + prefix + CONV_DW + " needs at least two taps, got "
                        + taps);
            }
            checkShape(convPw1, nEmbdDec, 3L * nEmbdDec);
            checkShape(convDw, taps, nEmbdDec);
            checkShape(convPw2, nEmbdDec, nEmbdDec);
            return new Layer(ln1, ln2, gate, up, down, attnPostNorm, ffPostNorm, convPw1, convDw, convPw2, taps,
                    null, null, null, null, null, null, null, 0, 0, 0);
        }
        int dHead = qNorm.length;
        if (dHead <= 0 || q.rows() % dHead != 0 || k.rows() % dHead != 0 || v.rows() % dHead != 0) {
            throw new GgufFormatException("lfm3a codec: " + prefix + ATTN_Q_NORM + " holds " + dHead
                    + " values, not a divisor of the q/k/v widths");
        }
        int nHead = (int) q.rows() / dHead;
        int nHeadKv = (int) k.rows() / dHead;
        if (nHead <= 0 || nHeadKv <= 0 || nHead % nHeadKv != 0) {
            throw new GgufFormatException("lfm3a codec: heads " + nHead + "/" + nHeadKv
                    + " must divide the decoder width " + nEmbdDec + " and each other");
        }
        vector(qNorm, prefix + ATTN_Q_NORM, dHead);
        vector(kNorm, prefix + ATTN_K_NORM, dHead);
        checkShape(q, nEmbdDec, (long) nHead * dHead);
        checkShape(k, nEmbdDec, (long) nHeadKv * dHead);
        checkShape(v, nEmbdDec, (long) nHeadKv * dHead);
        checkShape(out, (long) nHead * dHead, nEmbdDec);
        checkShape(attnGate, nEmbdDec, nHead);
        return new Layer(ln1, ln2, gate, up, down, attnPostNorm, ffPostNorm, null, null, null, -1, q, qNorm, k,
                kNorm, v, out, attnGate, nHead, nHeadKv, dHead);
    }

    // ------------------------------------------------------------------ accessors

    /** Codebook frames the decoder takes, the depthformer's codebook count. */
    public int nCodebooks() {
        return nCodebooks;
    }

    /** Level-table rows per codebook of the FSQ quantizer. */
    public int nLvl() {
        return nLvl;
    }

    /** Code vocabulary rows of the FSQ level table. */
    public int vocab() {
        return vocab;
    }

    /** Decoder transformer width ({@code 512} in the reference). */
    public int nEmbdDec() {
        return nEmbdDec;
    }

    /** STFT frames per codec frame ({@code stft_frame_stack}, 6 in the reference). */
    public int frameStack() {
        return frameStack;
    }

    /** The sliding attention window and persisted prefix in decoder positions ({@code decoder_window}, 30). */
    public int window() {
        return window;
    }

    /** The STFT transform size ({@code 1280} in the reference). */
    public int nFft() {
        return nFft;
    }

    /** Samples between STFT frames ({@code 320} in the reference). */
    public int hop() {
        return hop;
    }

    /** Spectrum bins per STFT frame. */
    public int nBins() {
        return nFft / 2 + 1;
    }

    /** The waveform's sample rate. */
    public int sampleRate() {
        return SAMPLE_RATE;
    }

    /** The persisted state slot names, in the reference's {@code list_lfm3a_gen_state_slots} order. */
    public List<String> stateSlots() {
        List<String> names = new ArrayList<>();
        names.add("pos");
        for (int il = 0; il < layers.length; il++) {
            if (layers[il].isConv()) {
                names.add("conv_" + il);
            } else {
                names.add("k_" + il);
                names.add("v_" + il);
            }
        }
        return names;
    }

    /** A fresh decoder state: position 0, zeroed conv taps and attention windows, a cold ISTFT. */
    public Lfm3aCodecSession newSession() {
        return new Lfm3aCodecSession(this);
    }

    /** A fresh decoder over {@code device}, or a host {@link #newSession()} when {@code device} is null. */
    public Decoder newDecoder(SpeechDeviceWeights device) {
        return device == null ? newSession() : new GpuLfm3aCodec(this, device);
    }

    CpuWeights weights() {
        return weights;
    }

    Layer[] layers() {
        return layers;
    }

    CpuTensorRef quantFirstCbWeight() {
        return quantFirstCb;
    }

    CpuTensorRef quantFirstOutWeight() {
        return quantFirstOut;
    }

    float[] quantFirstOutBiasValues() {
        return quantFirstOutB;
    }

    CpuTensorRef tfmInWeight() {
        return tfmIn;
    }

    float[] tfmInBiasValues() {
        return tfmInB;
    }

    CpuTensorRef tfmOutWeight() {
        return tfmOut;
    }

    float[] tfmOutBiasValues() {
        return tfmOutB;
    }

    float[] tfmOutNormValues() {
        return tfmOutNorm;
    }

    private static int positiveOverride(GgufKv kv, String key, int dflt, String what) {
        int v = kv.u32(key, dflt);
        if (v <= 0) {
            throw new GgufFormatException("lfm3a codec: " + key + " must be positive, got " + v);
        }
        return v;
    }

    private static int dim(TensorRef ref, int i) {
        long v = ref.ne(i);
        if (v <= 0 || v > Integer.MAX_VALUE) {
            throw new GgufFormatException("lfm3a codec: tensor " + ref.name() + " dimension " + i + " is " + v);
        }
        return (int) v;
    }

    private static void shape(TensorRef ref, long... ne) {
        boolean ok = ref.rank() >= ne.length;
        for (int i = 0; i < ne.length && ok; i++) {
            ok = ref.ne(i) == ne[i];
        }
        for (int i = ne.length; i < ref.rank() && ok; i++) {
            ok = ref.ne(i) == 1;
        }
        if (!ok) {
            throw new GgufFormatException("lfm3a codec: tensor " + ref.name() + " has shape " + ref + ", expected "
                    + java.util.Arrays.toString(ne) + " (ggml order)");
        }
    }

    private static void checkShape(CpuTensorRef ref, long cols, long rows) {
        if (ref.cols() != cols || ref.rows() != rows) {
            throw new GgufFormatException("lfm3a codec: tensor " + ref + " is " + ref.rows() + " x " + ref.cols()
                    + ", expected " + rows + " x " + cols);
        }
    }

    private static void vector(float[] v, String name, int n) {
        if (v == null || v.length != n) {
            throw new GgufFormatException("lfm3a codec: " + name + " must hold " + n + " floats, got "
                    + (v == null ? "none" : v.length));
        }
    }

    @Override
    public String toString() {
        return "Lfm3aCodec{embd=" + nEmbdDec + ", layers=" + layers.length + ", frame_stack=" + frameStack
                + ", window=" + window + ", n_fft=" + nFft + ", hop=" + hop + "}";
    }
}
