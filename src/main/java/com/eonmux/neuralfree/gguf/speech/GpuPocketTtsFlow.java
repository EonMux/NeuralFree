package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.audio.SpeechKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Device executor of {@link PocketTtsFlow}: the same velocity field, latent projection and end-of-speech probe as
 * the host graph, run as kernels on the engine's queue over the clip weights the {@link SpeechDeviceWeights}
 * uploaded.
 *
 * <p>Every frame integrates the ordinary differential equation of the reference in place: the timestep embedders
 * are projections and a learned unbiased-variance normalization ({@link SpeechKernels#RMS_NORM_UNBIASED}), the
 * gated AdaLN blocks use LayerNorm and {@link SpeechKernels#ADA_IN_MODULATE}, and the per-channel tanh gate scales
 * the residual ({@link SpeechKernels#ROW_SCALE_ADD}). The Gaussian noise draw, the latent read-back and the
 * end-of-speech comparison stay on the host, so both paths draw the same uniforms.
 *
 * <p>Owns its activation buffers; not thread-safe.
 */
final class GpuPocketTtsFlow implements PocketTtsFlow.Engine {

    private final PocketTtsFlow geo;
    private final SpeechDeviceWeights device;
    private final GpuOps ops;
    private final int latent;
    private final int condWidth;
    private final int hidden;
    private final int nStep;
    private final float blockEps;
    private final float timeEps;

    private final GpuWeight inputProj;
    private final GpuBuffer inputProjBias;
    private final GpuWeight condEmbd;
    private final GpuBuffer condEmbdBias;
    private final TimeWeights[] times;
    private final BlockWeights[] blocks;
    private final GpuWeight finalAda;
    private final GpuBuffer finalAdaBias;
    private final GpuWeight finalProj;
    private final GpuBuffer finalProjBias;
    private final GpuWeight outEos;
    private final GpuWeight inputLinear;
    private final float outEosBias;

    private final GpuBuffer cond;
    private final GpuBuffer x;
    private final GpuBuffer velocity;
    private final GpuBuffer ts;
    private final GpuBuffer tsOther;
    private final GpuBuffer condProj;
    private final GpuBuffer y;
    private final GpuBuffer siluY;
    private GpuBuffer cur;
    private GpuBuffer curNext;
    private final GpuBuffer normed;
    private final GpuBuffer h;
    private final GpuBuffer hUp;
    private final GpuBuffer hDown;
    private final GpuBuffer mod;
    private final GpuBuffer timeIn;
    private final GpuBuffer latentIn;
    private final GpuBuffer eosOut;
    private final GpuBuffer projOut;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final float[] latentHost;
    private final float[] projHost;
    private final float[] eosHost = new float[1];
    private boolean closed;

    /** Device tensors of one timestep embedder. */
    private static final class TimeWeights {
        final GpuWeight up;
        final GpuBuffer upBias;
        final GpuWeight down;
        final GpuBuffer downBias;
        final GpuWeight norm;

        TimeWeights(GpuWeight up, GpuBuffer upBias, GpuWeight down, GpuBuffer downBias, GpuWeight norm) {
            this.up = up;
            this.upBias = upBias;
            this.down = down;
            this.downBias = downBias;
            this.norm = norm;
        }
    }

    /** Device tensors of one AdaLN residual block. */
    private static final class BlockWeights {
        final GpuWeight norm;
        final GpuWeight normBias;
        final GpuWeight up;
        final GpuBuffer upBias;
        final GpuWeight down;
        final GpuBuffer downBias;
        final GpuWeight ada;
        final GpuBuffer adaBias;

        BlockWeights(GpuWeight norm, GpuWeight normBias, GpuWeight up, GpuBuffer upBias, GpuWeight down,
                     GpuBuffer downBias, GpuWeight ada, GpuBuffer adaBias) {
            this.norm = norm;
            this.normBias = normBias;
            this.up = up;
            this.upBias = upBias;
            this.down = down;
            this.downBias = downBias;
            this.ada = ada;
            this.adaBias = adaBias;
        }
    }

    GpuPocketTtsFlow(PocketTtsFlow geo, SpeechDeviceWeights device) {
        this.geo = Objects.requireNonNull(geo, "geo");
        this.device = Objects.requireNonNull(device, "device");
        this.ops = device.ops();
        this.latent = geo.latent();
        this.condWidth = geo.condWidth();
        this.hidden = geo.hidden();
        this.nStep = PocketTtsFlow.STEPS;
        this.blockEps = PocketTtsFlow.BLOCK_EPS;
        this.timeEps = PocketTtsFlow.TIME_EPS;

        this.inputProj = device.weight(PocketTtsFlow.INPUT_PROJ + ".weight");
        requireMatrix(inputProj, hidden, latent, PocketTtsFlow.INPUT_PROJ);
        this.inputProjBias = device.f32(PocketTtsFlow.INPUT_PROJ + ".bias", geo.inputProjBias());
        this.condEmbd = device.weight(PocketTtsFlow.COND_EMBD + ".weight");
        requireMatrix(condEmbd, hidden, condWidth, PocketTtsFlow.COND_EMBD);
        this.condEmbdBias = device.f32(PocketTtsFlow.COND_EMBD + ".bias", geo.condEmbdBias());

        PocketTtsFlow.TimeEmbedder[] sourceTimes = geo.timeEmbedders();
        this.times = new TimeWeights[sourceTimes.length];
        int maxFreq = 0;
        for (int i = 0; i < sourceTimes.length; i++) {
            PocketTtsFlow.TimeEmbedder t = sourceTimes[i];
            maxFreq = Math.max(maxFreq, t.freqs.length);
            GpuWeight up = device.weight(PocketTtsFlow.timeTensor(i, "up.weight"));
            requireMatrix(up, hidden, 2L * t.freqs.length, PocketTtsFlow.timeTensor(i, "up.weight"));
            GpuWeight down = device.weight(PocketTtsFlow.timeTensor(i, "down.weight"));
            requireMatrix(down, hidden, hidden, PocketTtsFlow.timeTensor(i, "down.weight"));
            device.f32(PocketTtsFlow.timeTensor(i, "freqs"), t.freqs);
            times[i] = new TimeWeights(up, device.f32(PocketTtsFlow.timeTensor(i, "up.bias"), t.upBias),
                    down, device.f32(PocketTtsFlow.timeTensor(i, "down.bias"), t.downBias),
                    device.f32Weight(PocketTtsFlow.timeTensor(i, "norm"), t.norm));
        }

        PocketTtsFlow.Block[] sourceBlocks = geo.blockTensors();
        this.blocks = new BlockWeights[sourceBlocks.length];
        for (int b = 0; b < sourceBlocks.length; b++) {
            PocketTtsFlow.Block src = sourceBlocks[b];
            GpuWeight up = device.weight(PocketTtsFlow.blockTensor(b, "up.weight"));
            GpuWeight down = device.weight(PocketTtsFlow.blockTensor(b, "down.weight"));
            GpuWeight ada = device.weight(PocketTtsFlow.blockTensor(b, "ada.weight"));
            requireMatrix(up, hidden, hidden, PocketTtsFlow.blockTensor(b, "up.weight"));
            requireMatrix(down, hidden, hidden, PocketTtsFlow.blockTensor(b, "down.weight"));
            requireMatrix(ada, 3L * hidden, hidden, PocketTtsFlow.blockTensor(b, "ada.weight"));
            blocks[b] = new BlockWeights(
                    device.f32Weight(PocketTtsFlow.blockTensor(b, "norm.weight"), src.normW),
                    device.f32Weight(PocketTtsFlow.blockTensor(b, "norm.bias"), src.normB),
                    up, device.f32(PocketTtsFlow.blockTensor(b, "up.bias"), src.upBias),
                    down, device.f32(PocketTtsFlow.blockTensor(b, "down.bias"), src.downBias),
                    ada, device.f32(PocketTtsFlow.blockTensor(b, "ada.bias"), src.adaBias));
        }

        this.finalAda = device.weight(PocketTtsFlow.FINAL_ADA + ".weight");
        requireMatrix(finalAda, 2L * hidden, hidden, PocketTtsFlow.FINAL_ADA);
        this.finalAdaBias = device.f32(PocketTtsFlow.FINAL_ADA + ".bias", geo.finalAdaBias());
        this.finalProj = device.weight(PocketTtsFlow.FINAL_PROJ + ".weight");
        requireMatrix(finalProj, latent, hidden, PocketTtsFlow.FINAL_PROJ);
        this.finalProjBias = device.f32(PocketTtsFlow.FINAL_PROJ + ".bias", geo.finalProjBias());
        this.outEos = device.weight(PocketTtsFlow.OUT_EOS + ".weight");
        this.inputLinear = device.weight(PocketTtsFlow.INPUT_LINEAR + ".weight");
        requireMatrix(inputLinear, condWidth, latent, PocketTtsFlow.INPUT_LINEAR);
        this.outEosBias = geo.outEosBias()[0];

        this.cond = own("cond", condWidth);
        this.x = own("x", latent);
        this.velocity = own("velocity", latent);
        this.ts = own("ts", hidden);
        this.tsOther = own("ts.other", hidden);
        this.condProj = own("cond.proj", hidden);
        this.y = own("y", hidden);
        this.siluY = own("y.silu", hidden);
        this.cur = own("cur", hidden);
        this.curNext = own("cur.next", hidden);
        this.normed = own("normed", hidden);
        this.h = own("h", hidden);
        this.hUp = own("h.up", hidden);
        this.hDown = own("h.down", hidden);
        this.mod = own("mod", 3L * hidden);
        this.timeIn = own("time.in", Math.max(2L * maxFreq, 1));
        this.latentIn = own("latent.in", latent);
        this.eosOut = own("eos.out", 1);
        this.projOut = own("proj.out", condWidth);
        this.latentHost = new float[latent];
        this.projHost = new float[condWidth];
    }

    private GpuBuffer own(String name, long floats) {
        GpuBuffer buffer = device.allocate("flow." + name, floats);
        owned.add(buffer);
        return buffer;
    }

    private static void requireMatrix(GpuWeight w, long rows, long cols, String what) {
        if (w.rows() != rows || w.cols() != cols) {
            throw new IllegalArgumentException("pockettts flow tensor " + what + " is " + w.rows() + " x "
                    + w.cols() + " on the device, the graph needs " + rows + " x " + cols);
        }
    }

    @Override
    public float[] decode(float[] condRow, float temperature, Random rng) {
        ensureOpen();
        requireCond(condRow);
        Objects.requireNonNull(rng, "rng");
        if (!(temperature >= 0f) || Float.isInfinite(temperature)) {
            throw new IllegalArgumentException("the flow temperature must be finite and not negative, got "
                    + temperature);
        }
        double std = Math.sqrt(temperature);
        for (int i = 0; i < latent; i++) {
            latentHost[i] = (float) (rng.nextGaussian() * std);
        }
        device.writeFloats(cond, condRow);
        device.writeFloats(x, latentHost);
        for (int step = 0; step < nStep; step++) {
            float s = (float) step / nStep;
            float t = (float) (step + 1) / nStep;
            velocity(s, t);
            ops.scaleAdd(velocity, 0, 1.0f / nStep, x, 0, latent);
        }
        device.readFloats(x, latentHost, latent);
        return latentHost.clone();
    }

    /** One velocity evaluation, left in {@link #velocity}. */
    private void velocity(float s, float t) {
        timeEmbed(0, s, ts);
        for (int i = 1; i < times.length; i++) {
            timeEmbed(i, t, tsOther);
            ops.add(ts, 0, tsOther, 0, ts, 0, hidden);
        }
        if (times.length > 1) {
            ops.scale(ts, 0, hidden, 1f / times.length);
        }
        apply(condEmbd, cond, condEmbdBias, condProj);
        ops.add(ts, 0, condProj, 0, y, 0, hidden);
        ops.unary(ActOp.SILU, ActParams.NONE, y, 0, siluY, 0, hidden);
        apply(inputProj, x, inputProjBias, cur);

        for (BlockWeights block : blocks) {
            apply(block.ada, siluY, block.adaBias, mod);
            ops.norm(NormKind.LAYER, block.norm, block.normBias, blockEps, cur, 0, normed, 0, 1, hidden);
            ops.enqueue(SpeechKernels.ADA_IN_MODULATE, new long[]{SpeechKernels.roundUp(hidden)},
                    new long[]{SpeechKernels.LOCAL}, 1, hidden, normed, mod, h);
            apply(block.up, h, block.upBias, hUp);
            ops.unary(ActOp.SILU, ActParams.NONE, hUp, 0, hUp, 0, hidden);
            apply(block.down, hUp, block.downBias, hDown);
            ops.enqueue(SpeechKernels.ROW_SCALE_ADD, new long[]{SpeechKernels.roundUp(hidden)},
                    new long[]{SpeechKernels.LOCAL}, 1, hidden, 2 * hidden, cur, hDown, mod, curNext);
            swapCur();
        }

        apply(finalAda, siluY, finalAdaBias, mod);
        ops.norm(NormKind.LAYER, null, null, blockEps, cur, 0, normed, 0, 1, hidden);
        ops.enqueue(SpeechKernels.ADA_IN_MODULATE, new long[]{SpeechKernels.roundUp(hidden)},
                new long[]{SpeechKernels.LOCAL}, 1, hidden, normed, mod, h);
        apply(finalProj, h, finalProjBias, velocity);
    }

    /** {@code mod = w @ x + bias}, one row. */
    private void apply(GpuWeight w, GpuBuffer in, GpuBuffer bias, GpuBuffer out) {
        ops.matvec(w, in, 0, out, 0, 1);
        ops.enqueue(GpuOps.BATCH_BIAS_ADD, ActKernels.global1d(w.rows(), ops.local()),
                new long[]{ops.local()}, (int) w.rows(), 1, bias, out);
    }

    /** One timestep embedder's output into {@code out}. */
    private void timeEmbed(int index, float t, GpuBuffer out) {
        PocketTtsFlow.TimeEmbedder source = geo.timeEmbedders()[index];
        int freq = source.freqs.length;
        float[] emb = new float[2 * freq];
        for (int i = 0; i < freq; i++) {
            double arg = (double) source.freqs[i] * t;
            emb[i] = (float) Math.cos(arg);
            emb[freq + i] = (float) Math.sin(arg);
        }
        device.writeFloats(timeIn, emb);
        TimeWeights tw = times[index];
        apply(tw.up, timeIn, tw.upBias, out);
        ops.unary(ActOp.SILU, ActParams.NONE, out, 0, out, 0, hidden);
        apply(tw.down, out, tw.downBias, hUp);
        ops.enqueue(SpeechKernels.RMS_NORM_UNBIASED, new long[]{SpeechKernels.roundUp(hidden)},
                new long[]{SpeechKernels.LOCAL}, hidden, timeEps, hUp, tw.norm.shard(0), out);
    }

    /** Makes the block's result the current stream and recycles the previous one. */
    private void swapCur() {
        GpuBuffer t = cur;
        cur = curNext;
        curNext = t;
    }

    @Override
    public float[] project(float[] row) {
        ensureOpen();
        Objects.requireNonNull(row, "row");
        if (row.length != latent) {
            throw new IllegalArgumentException("a latent row holds " + row.length + " values, the flow takes "
                    + latent);
        }
        device.writeFloats(latentIn, row);
        ops.matvec(inputLinear, latentIn, 0, projOut, 0, 1);
        device.readFloats(projOut, projHost, condWidth);
        return projHost.clone();
    }

    @Override
    public boolean isEndOfSpeech(float[] condRow) {
        ensureOpen();
        requireCond(condRow);
        device.writeFloats(cond, condRow);
        ops.matvec(outEos, cond, 0, eosOut, 0, 1);
        device.readFloats(eosOut, eosHost, 1);
        return eosHost[0] + outEosBias > PocketTtsFlow.EOS_THRESHOLD;
    }

    private void requireCond(float[] condRow) {
        Objects.requireNonNull(condRow, "cond");
        if (condRow.length != condWidth) {
            throw new IllegalArgumentException("the backbone hidden state holds " + condRow.length + " values, the"
                    + " flow network takes " + condWidth);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the pockettts GPU flow engine is closed");
        }
    }

    /** Releases this engine's activation buffers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (GpuBuffer buffer : owned) {
            try {
                buffer.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
            }
        }
        owned.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuPocketTtsFlow{" + geo + '}';
    }
}
