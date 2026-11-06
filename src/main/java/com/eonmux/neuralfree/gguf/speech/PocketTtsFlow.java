package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.Random;

/**
 * The pockettts flow-matching decoder of llama.cpp's {@code clip_graph_pockettts_gen} GEN_CODE graph
 * (tools/mtmd/models/pockettts-gen.cpp:12-176): a backbone hidden state becomes one continuous latent of the next
 * frame and an end-of-speech score.
 *
 * <p>Every frame is a stochastic ordinary differential equation integrated from Gaussian noise, whose standard
 * deviation is {@code sqrt(temperature)}; the reference's one integration step evaluates the velocity field between
 * {@code s = 0} and {@code t = 1}. The velocity network pairs two timestep embedders (the first on {@code s}, the
 * second on {@code t}, averaged), adds the projected backbone state, and runs the gated AdaLN residual blocks and
 * the AdaLN final layer. Its time embedder's closing "RMSNorm" divides the input (not the centered value) by the
 * unbiased standard deviation, as the reference's {@code _rms_norm} does. The same backbone state feeds the
 * end-of-speech probe; a score above -4 ends the speech, llama.cpp's default threshold
 * ({@code hparams.gen_eos_threshold}).
 *
 * <p>Tensors use the clip generator names of tools/mtmd/clip-impl.h:295-309 in ggml dimension order. Immutable over
 * shared host weights; {@link #decode} draws its noise from the caller's generator and holds no state.
 */
public final class PocketTtsFlow {

    /** The {@code clip.gen.audio.projector_type} of a pockettts generator file. */
    public static final String PROJECTOR_TYPE = "pockettts_gen";

    /**
     * What the speech pipeline draws through: the flow network's velocity field, the latent-to-backbone projection
     * and the end-of-speech probe. Implemented by the host graph itself and by the device executor of a GPU engine.
     */
    public interface Engine extends AutoCloseable {
        /** {@link PocketTtsFlow#decode(float[], float, Random)}. */
        float[] decode(float[] cond, float temperature, Random rng);

        /** {@link PocketTtsFlow#project(float[])}. */
        float[] project(float[] row);

        /** {@link PocketTtsFlow#isEndOfSpeech(float[])}. */
        boolean isEndOfSpeech(float[] cond);

        /** Releases any per-engine device state; the host graph holds none. */
        @Override
        default void close() {
        }
    }

    /** The integration steps of {@code hparams.flow_n_step} (clip.cpp:2221). */
    public static final int STEPS = 1;
    /** The end-of-speech threshold of {@code hparams.gen_eos_threshold} (clip.cpp:2222). */
    public static final float EOS_THRESHOLD = -4.0f;
    /** The noise temperature of {@code Config.default_temperature} (clip.cpp:6121). */
    public static final float DEFAULT_TEMPERATURE = 0.7f;

    /** LayerNorm epsilon of the AdaLN blocks ({@code build_norm(..., 1e-6f)}). */
    static final float BLOCK_EPS = 1e-6f;
    /** Epsilon of the time embedder's RMSNorm ({@code _rms_norm} in mlp.py). */
    static final float TIME_EPS = 1e-5f;

    /** Tensor base names of tools/mtmd/clip-impl.h:295-309. */
    public static final String INPUT_PROJ = "a.gen.flow.input_proj";
    public static final String COND_EMBD = "a.gen.flow.cond_embd";
    public static final String TIME = "a.gen.flow.time.";
    public static final String BLOCK = "a.gen.flow.blk.";
    public static final String FINAL_ADA = "a.gen.flow.final.ada";
    public static final String FINAL_PROJ = "a.gen.flow.final.proj";
    public static final String OUT_EOS = "a.gen.out_eos";
    public static final String INPUT_LINEAR = "a.gen.input_linear";

    /** One timestep embedder: {@code cos/sin(t * freqs)} -> up -> SiLU -> down -> RMSNorm. */
    static final class TimeEmbedder {
        final float[] freqs;
        final CpuTensorRef up;
        final float[] upBias;
        final CpuTensorRef down;
        final float[] downBias;
        final float[] norm;

        TimeEmbedder(float[] freqs, CpuTensorRef up, float[] upBias, CpuTensorRef down, float[] downBias,
                     float[] norm) {
            this.freqs = freqs;
            this.up = up;
            this.upBias = upBias;
            this.down = down;
            this.downBias = downBias;
            this.norm = norm;
        }
    }

    /** One AdaLN residual block: in_ln -> modulate -> up -> SiLU -> down, gated residual. */
    static final class Block {
        final float[] normW;
        final float[] normB;
        final CpuTensorRef up;
        final float[] upBias;
        final CpuTensorRef down;
        final float[] downBias;
        final CpuTensorRef ada;
        final float[] adaBias;

        Block(float[] normW, float[] normB, CpuTensorRef up, float[] upBias, CpuTensorRef down, float[] downBias,
              CpuTensorRef ada, float[] adaBias) {
            this.normW = normW;
            this.normB = normB;
            this.up = up;
            this.upBias = upBias;
            this.down = down;
            this.downBias = downBias;
            this.ada = ada;
            this.adaBias = adaBias;
        }
    }

    private final int latent;
    private final int condWidth;
    private final int hidden;
    private final float[] inputProjBias;
    private final CpuTensorRef inputProj;
    private final CpuTensorRef condEmbd;
    private final float[] condEmbdBias;
    private final TimeEmbedder[] times;
    private final Block[] blocks;
    private final CpuTensorRef finalAda;
    private final float[] finalAdaBias;
    private final CpuTensorRef finalProj;
    private final float[] finalProjBias;
    private final float[] outEos;
    private final float[] outEosBias;
    private final CpuTensorRef inputLinear;
    private final CpuMatVec matVec;

    private PocketTtsFlow(Builder b) {
        this.latent = b.latent;
        this.condWidth = b.condWidth;
        this.hidden = b.hidden;
        this.inputProj = b.inputProj;
        this.inputProjBias = b.inputProjBias;
        this.condEmbd = b.condEmbd;
        this.condEmbdBias = b.condEmbdBias;
        this.times = b.times;
        this.blocks = b.blocks;
        this.finalAda = b.finalAda;
        this.finalAdaBias = b.finalAdaBias;
        this.finalProj = b.finalProj;
        this.finalProjBias = b.finalProjBias;
        this.outEos = b.outEos;
        this.outEosBias = b.outEosBias;
        this.inputLinear = b.inputLinear;
        this.matVec = b.matVec;
    }

    /**
     * Resolves the flow network of an opened pockettts generator file.
     *
     * @throws IllegalArgumentException when the file is another generator family
     * @throws GgufFormatException      when a tensor is missing or a shape differs from the reference graph's
     */
    public static PocketTtsFlow open(GgufKv kv, TensorCatalog tc, CpuWeights weights) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(tc, "tc");
        Objects.requireNonNull(weights, "weights");
        String type = kv.str(com.eonmux.neuralfree.gguf.hparams.GgufKeys.CLIP_GEN_AUDIO_PROJECTOR_TYPE, null);
        if (!PROJECTOR_TYPE.equals(type)) {
            throw new IllegalArgumentException(com.eonmux.neuralfree.gguf.hparams.GgufKeys
                    .CLIP_GEN_AUDIO_PROJECTOR_TYPE + " is " + (type == null ? "absent" : "'" + type + "'")
                    + ", a pockettts generator declares '" + PROJECTOR_TYPE + "'");
        }
        Builder b = new Builder();
        b.matVec = weights.matVec();
        TensorRef inputProjRef = tc.require(INPUT_PROJ + ".weight");
        b.latent = PocketTtsTensors.dim(inputProjRef, 0);
        b.hidden = PocketTtsTensors.dim(inputProjRef, 1);
        if (b.hidden < 2) {
            throw new GgufFormatException("pockettts flow: the velocity network is " + b.hidden
                    + " wide; its time embedder's unbiased RMSNorm needs at least 2 values");
        }
        b.inputProj = weights.get(INPUT_PROJ + ".weight");
        b.inputProjBias = weights.floats(INPUT_PROJ + ".bias");
        PocketTtsTensors.vector(b.inputProjBias, b.hidden, INPUT_PROJ + ".bias");

        TensorRef condRef = tc.require(COND_EMBD + ".weight");
        b.condWidth = PocketTtsTensors.dim(condRef, 0);
        if (PocketTtsTensors.dim(condRef, 1) != b.hidden) {
            throw new GgufFormatException("pockettts flow: " + COND_EMBD + ".weight writes "
                    + PocketTtsTensors.dim(condRef, 1) + " values, the blocks run " + b.hidden);
        }
        b.condEmbd = weights.get(COND_EMBD + ".weight");
        b.condEmbdBias = weights.floats(COND_EMBD + ".bias");
        PocketTtsTensors.vector(b.condEmbdBias, b.hidden, COND_EMBD + ".bias");

        b.times = new TimeEmbedder[2];
        for (int i = 0; i < b.times.length; i++) {
            b.times[i] = timeEmbedder(tc, weights, i, b.hidden);
        }

        int nBlock = PocketTtsTensors.countIndexed(tc, BLOCK);
        if (nBlock == 0) {
            throw new GgufFormatException("pockettts flow: the file holds no " + BLOCK + "* AdaLN block");
        }
        b.blocks = new Block[nBlock];
        for (int il = 0; il < nBlock; il++) {
            b.blocks[il] = block(tc, weights, il, b.hidden);
        }

        TensorRef finalAdaRef = tc.require(FINAL_ADA + ".weight");
        if (PocketTtsTensors.dim(finalAdaRef, 0) != b.hidden
                || PocketTtsTensors.dim(finalAdaRef, 1) != 2 * b.hidden) {
            throw new GgufFormatException("pockettts flow: " + FINAL_ADA + ".weight must map " + b.hidden
                    + " values onto " + 2 * b.hidden);
        }
        b.finalAda = weights.get(FINAL_ADA + ".weight");
        b.finalAdaBias = weights.floats(FINAL_ADA + ".bias");
        PocketTtsTensors.vector(b.finalAdaBias, 2 * b.hidden, FINAL_ADA + ".bias");
        TensorRef finalProjRef = tc.require(FINAL_PROJ + ".weight");
        PocketTtsTensors.shape(finalProjRef, b.hidden, b.latent);
        b.finalProj = weights.get(FINAL_PROJ + ".weight");
        b.finalProjBias = weights.floats(FINAL_PROJ + ".bias");
        PocketTtsTensors.vector(b.finalProjBias, b.latent, FINAL_PROJ + ".bias");

        TensorRef eosRef = tc.require(OUT_EOS + ".weight");
        PocketTtsTensors.shape(eosRef, b.condWidth, 1);
        b.outEos = weights.floats(OUT_EOS + ".weight");
        b.outEosBias = weights.floats(OUT_EOS + ".bias");
        PocketTtsTensors.vector(b.outEos, b.condWidth, OUT_EOS + ".weight");
        PocketTtsTensors.vector(b.outEosBias, 1, OUT_EOS + ".bias");

        TensorRef linearRef = tc.require(INPUT_LINEAR + ".weight");
        PocketTtsTensors.shape(linearRef, b.latent, b.condWidth);
        b.inputLinear = weights.get(INPUT_LINEAR + ".weight");
        return new PocketTtsFlow(b);
    }

    private static TimeEmbedder timeEmbedder(TensorCatalog tc, CpuWeights weights, int index, int hidden) {
        String base = TIME + index + ".";
        TensorRef freqRef = tc.require(base + "freqs");
        if (freqRef.rank() != 1) {
            throw new GgufFormatException("pockettts flow: " + base + "freqs must be one-dimensional");
        }
        int freq = PocketTtsTensors.dim(freqRef, 0);
        float[] freqs = weights.floats(base + "freqs");
        PocketTtsTensors.vector(freqs, freq, base + "freqs");
        TensorRef upRef = tc.require(base + "up.weight");
        PocketTtsTensors.shape(upRef, 2L * freq, hidden);
        TensorRef downRef = tc.require(base + "down.weight");
        PocketTtsTensors.shape(downRef, hidden, hidden);
        float[] upBias = weights.floats(base + "up.bias");
        PocketTtsTensors.vector(upBias, hidden, base + "up.bias");
        float[] downBias = weights.floats(base + "down.bias");
        PocketTtsTensors.vector(downBias, hidden, base + "down.bias");
        float[] norm = weights.floats(base + "norm");
        PocketTtsTensors.vector(norm, hidden, base + "norm");
        return new TimeEmbedder(freqs, weights.get(base + "up.weight"), upBias, weights.get(base + "down.weight"),
                downBias, norm);
    }

    private static Block block(TensorCatalog tc, CpuWeights weights, int il, int hidden) {
        String base = BLOCK + il + ".";
        TensorRef normRef = tc.require(base + "norm.weight");
        PocketTtsTensors.shape(normRef, hidden);
        TensorRef upRef = tc.require(base + "up.weight");
        PocketTtsTensors.shape(upRef, hidden, hidden);
        TensorRef downRef = tc.require(base + "down.weight");
        PocketTtsTensors.shape(downRef, hidden, hidden);
        TensorRef adaRef = tc.require(base + "ada.weight");
        PocketTtsTensors.shape(adaRef, hidden, 3L * hidden);
        float[] normW = weights.floats(base + "norm.weight");
        float[] normB = weights.floats(base + "norm.bias");
        PocketTtsTensors.vector(normW, hidden, base + "norm.weight");
        PocketTtsTensors.vector(normB, hidden, base + "norm.bias");
        float[] upBias = weights.floats(base + "up.bias");
        float[] downBias = weights.floats(base + "down.bias");
        float[] adaBias = weights.floats(base + "ada.bias");
        PocketTtsTensors.vector(upBias, hidden, base + "up.bias");
        PocketTtsTensors.vector(downBias, hidden, base + "down.bias");
        PocketTtsTensors.vector(adaBias, 3 * hidden, base + "ada.bias");
        return new Block(normW, normB, weights.get(base + "up.weight"), upBias, weights.get(base + "down.weight"),
                downBias, weights.get(base + "ada.weight"), adaBias);
    }

    /** The latent width ({@code flow_lm.input_linear}'s input). */
    public int latent() {
        return latent;
    }

    /** The backbone hidden width the flow network conditions on. */
    public int condWidth() {
        return condWidth;
    }

    /** The velocity network's hidden width. */
    public int hidden() {
        return hidden;
    }

    /** The number of AdaLN residual blocks. */
    public int blocks() {
        return blocks.length;
    }

    /** A fresh execution engine over {@code device}, or the host graph when {@code device} is null. */
    public Engine newEngine(SpeechDeviceWeights device) {
        if (device != null) {
            return new GpuPocketTtsFlow(this, device);
        }
        return new Engine() {
            @Override
            public float[] decode(float[] cond, float temperature, Random rng) {
                return PocketTtsFlow.this.decode(cond, temperature, rng);
            }

            @Override
            public float[] project(float[] row) {
                return PocketTtsFlow.this.project(row);
            }

            @Override
            public boolean isEndOfSpeech(float[] cond) {
                return PocketTtsFlow.this.isEndOfSpeech(cond);
            }
        };
    }

    /** The GGUF name of one tensor of timestep embedder {@code index}. */
    static String timeTensor(int index, String part) {
        return TIME + index + "." + part;
    }

    /** The GGUF name of one tensor of AdaLN block {@code index}. */
    static String blockTensor(int index, String part) {
        return BLOCK + index + "." + part;
    }

    CpuTensorRef inputProjWeight() {
        return inputProj;
    }

    float[] inputProjBias() {
        return inputProjBias;
    }

    CpuTensorRef condEmbdWeight() {
        return condEmbd;
    }

    float[] condEmbdBias() {
        return condEmbdBias;
    }

    TimeEmbedder[] timeEmbedders() {
        return times;
    }

    Block[] blockTensors() {
        return blocks;
    }

    CpuTensorRef finalAdaWeight() {
        return finalAda;
    }

    float[] finalAdaBias() {
        return finalAdaBias;
    }

    CpuTensorRef finalProjWeight() {
        return finalProj;
    }

    float[] finalProjBias() {
        return finalProjBias;
    }

    float[] outEosWeight() {
        return outEos;
    }

    float[] outEosBias() {
        return outEosBias;
    }

    CpuTensorRef inputLinearWeight() {
        return inputLinear;
    }

    /**
     * The end-of-speech score of a backbone hidden state: {@code out_eos @ cond + bias}.
     *
     * @param cond the backbone hidden state ({@link #condWidth()} floats)
     * @throws IllegalArgumentException when the row is another width
     */
    public float eosScore(float[] cond) {
        requireCond(cond);
        double acc = outEosBias[0];
        for (int i = 0; i < condWidth; i++) {
            acc += (double) outEos[i] * cond[i];
        }
        return (float) acc;
    }

    /** True when the backbone hidden state ends the speech ({@link #EOS_THRESHOLD}). */
    public boolean isEndOfSpeech(float[] cond) {
        return eosScore(cond) > EOS_THRESHOLD;
    }

    /**
     * Integrates the flow from Gaussian noise of standard deviation {@code sqrt(temperature)}.
     *
     * @param cond        the backbone hidden state ({@link #condWidth()} floats)
     * @param temperature the noise temperature; 0 makes the draw deterministic
     * @param rng         the noise source
     * @return the frame's latent ({@link #latent()} floats)
     * @throws IllegalArgumentException when the row is another width or the temperature is negative or not finite
     */
    public float[] decode(float[] cond, float temperature, Random rng) {
        Objects.requireNonNull(rng, "rng");
        requireCond(cond);
        if (!(temperature >= 0f) || Float.isInfinite(temperature)) {
            throw new IllegalArgumentException("the flow temperature must be finite and not negative, got "
                    + temperature);
        }
        double std = Math.sqrt(temperature);
        float[] x = new float[latent];
        for (int i = 0; i < latent; i++) {
            x[i] = (float) (rng.nextGaussian() * std);
        }
        for (int step = 0; step < STEPS; step++) {
            float s = (float) step / STEPS;
            float t = (float) (step + 1) / STEPS;
            float[] v = velocity(cond, x, s, t);
            for (int i = 0; i < latent; i++) {
                x[i] += v[i] / STEPS;
            }
        }
        return x;
    }

    /**
     * The backbone input row of a latent: {@code input_linear @ latent}.
     *
     * @param row the latent ({@link #latent()} floats)
     * @return the projected row ({@link #condWidth()} floats)
     * @throws IllegalArgumentException when the row is another width
     */
    public float[] project(float[] row) {
        Objects.requireNonNull(row, "row");
        if (row.length != latent) {
            throw new IllegalArgumentException("a latent row holds " + row.length + " values, the flow takes "
                    + latent);
        }
        float[] out = new float[condWidth];
        matVec.matmul(inputLinear, row, 0, latent, out, 0, condWidth, 1);
        return out;
    }

    // ------------------------------------------------------------------ the velocity field

    /** One velocity evaluation {@code v(cond, s, t, x)} (pockettts-gen.cpp:45-98). */
    private float[] velocity(float[] cond, float[] x, float s, float t) {
        float[] cur = project(inputProj, x, inputProjBias);
        float[] ts = timeEmbed(times[0], s);
        float[] other = timeEmbed(times[1], t);
        for (int i = 0; i < hidden; i++) {
            ts[i] = (ts[i] + other[i]) / times.length;
        }
        float[] c = project(condEmbd, cond, condEmbdBias);
        float[] y = new float[hidden];
        for (int i = 0; i < hidden; i++) {
            y[i] = ts[i] + c[i];
        }
        for (Block block : blocks) {
            float[] mod = project(block.ada, silu(y), block.adaBias);
            float[] h = layerNorm(cur, block.normW, block.normB, BLOCK_EPS);
            for (int i = 0; i < hidden; i++) {
                h[i] = h[i] * (1f + mod[hidden + i]) + mod[i];
            }
            h = project(block.up, h, block.upBias);
            for (int i = 0; i < hidden; i++) {
                h[i] = silu(h[i]);
            }
            h = project(block.down, h, block.downBias);
            for (int i = 0; i < hidden; i++) {
                cur[i] += mod[2 * hidden + i] * h[i];
            }
        }
        float[] mod = project(finalAda, silu(y), finalAdaBias);
        cur = layerNorm(cur, null, null, BLOCK_EPS);
        for (int i = 0; i < hidden; i++) {
            cur[i] = cur[i] * (1f + mod[hidden + i]) + mod[i];
        }
        return project(finalProj, cur, finalProjBias);
    }

    /** {@code cos/sin(t * freqs)}, the up projection, SiLU, the down projection and the reference's RMSNorm. */
    private float[] timeEmbed(TimeEmbedder te, float t) {
        int freq = te.freqs.length;
        float[] emb = new float[2 * freq];
        for (int i = 0; i < freq; i++) {
            double arg = (double) te.freqs[i] * t;
            emb[i] = (float) Math.cos(arg);
            emb[freq + i] = (float) Math.sin(arg);
        }
        float[] cur = project(te.up, emb, te.upBias);
        for (int i = 0; i < cur.length; i++) {
            cur[i] = silu(cur[i]);
        }
        cur = project(te.down, cur, te.downBias);
        return rmsNorm(cur, te.norm);
    }

    /**
     * The reference's {@code _rms_norm}: the input (not the centered value) over the square root of the unbiased
     * variance, rescaled by the learned alpha.
     */
    private static float[] rmsNorm(float[] x, float[] weight) {
        int n = x.length;
        double mean = 0.0;
        for (float v : x) {
            mean += v;
        }
        mean /= n;
        double var = 0.0;
        for (float v : x) {
            double d = v - mean;
            var += d * d;
        }
        var /= n;
        double denom = Math.sqrt(var * n / (double) (n - 1) + TIME_EPS);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = (float) (x[i] / denom) * weight[i];
        }
        return out;
    }

    private float[] project(CpuTensorRef w, float[] x, float[] bias) {
        int in = (int) w.cols();
        int out = (int) w.rows();
        float[] y = new float[out];
        matVec.matmul(w, x, 0, in, y, 0, out, 1);
        for (int i = 0; i < out; i++) {
            y[i] += bias[i];
        }
        return y;
    }

    /** A LayerNorm over the channels, with the affine pair when both weights are present. */
    private static float[] layerNorm(float[] x, float[] weight, float[] bias, float eps) {
        int n = x.length;
        double mean = 0.0;
        for (float v : x) {
            mean += v;
        }
        mean /= n;
        double var = 0.0;
        for (float v : x) {
            double d = v - mean;
            var += d * d;
        }
        var /= n;
        double scale = 1.0 / Math.sqrt(var + eps);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            float v = (float) ((x[i] - mean) * scale);
            out[i] = weight == null ? v : v * weight[i] + bias[i];
        }
        return out;
    }

    private static float[] silu(float[] x) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = silu(x[i]);
        }
        return out;
    }

    private static float silu(float x) {
        return (float) (x / (1.0 + Math.exp(-x)));
    }

    private void requireCond(float[] cond) {
        Objects.requireNonNull(cond, "cond");
        if (cond.length != condWidth) {
            throw new IllegalArgumentException("the backbone hidden state holds " + cond.length + " values, the flow"
                    + " network takes " + condWidth);
        }
    }

    private static final class Builder {
        int latent;
        int condWidth;
        int hidden;
        CpuTensorRef inputProj;
        float[] inputProjBias;
        CpuTensorRef condEmbd;
        float[] condEmbdBias;
        TimeEmbedder[] times;
        Block[] blocks;
        CpuTensorRef finalAda;
        float[] finalAdaBias;
        CpuTensorRef finalProj;
        float[] finalProjBias;
        float[] outEos;
        float[] outEosBias;
        CpuTensorRef inputLinear;
        CpuMatVec matVec;
    }

    @Override
    public String toString() {
        return "PocketTtsFlow{latent=" + latent + ", cond=" + condWidth + ", hidden=" + hidden + ", blocks="
                + blocks.length + '}';
    }
}
