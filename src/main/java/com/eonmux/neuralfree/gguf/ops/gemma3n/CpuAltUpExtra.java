package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.PleLayout;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuAltUpExtra}: Gemma3n AltUp ({@code LayerSpec.Extras.ALTUP}) with the same stream layout
 * ({@code n_altup} blocks of {@code ubatch*n_embd} in the scratch temporary {@link GpuAltUpExtra#TEMP_STREAMS}),
 * the same call sequence and the {@link CpuAltUp} twins of the AltUp kernels. The streams and per-layer inputs carry
 * data from one block call to the next under names only the Gemma3n extras use. Not thread-safe.
 */
public final class CpuAltUpExtra implements CpuLayerExtra {

    private CpuMatVec matVec;
    private int il;
    private int nAltup;
    private int act;
    private int nEmbd;
    private int ubatch;
    private int stride;
    private float eps;
    private boolean first;
    private boolean last;
    private CpuTensorRef router;
    private float[] routerNorm;
    private CpuTensorRef predictCoef;
    private CpuTensorRef correctCoef;
    /** Twin of the {@code GpuAltUpExtra} router and coefficient LoRA deltas. */
    private CpuProjectionDeltas.Site routerDeltas;
    private CpuProjectionDeltas.Site predictDeltas;
    private CpuProjectionDeltas.Site correctDeltas;
    private float[] correctScale;
    private CpuTensorRef altupProj;
    private CpuTensorRef unembdProj;
    private float[] streams;
    private float[] routerIn;
    private float[] modalities;
    private float[] coefs;
    private float[] firstPred;
    private float[] proj;
    private float[] merge;
    private float[] inputs;
    private CpuPleExtra.PerLayerInputs producer;
    private CpuPleExtra.LayerGate gate;

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (matVec != null) {
            throw new IllegalStateException("AltUp extra is already bound");
        }
        ModelSpec spec = ctx.spec();
        ModelHparams hp = spec.hparams();
        String arch = spec.arch().ggufName();
        int a = hp.nAltup();
        if (a < 2 || a > Gemma3nKernels.MAX_ALTUP) {
            throw new IllegalArgumentException(arch + ": AltUp needs 2 to " + Gemma3nKernels.MAX_ALTUP
                    + " streams, got " + a);
        }
        Gemma3nKernels.checkActive(hp.iAltupAct(), a);
        GpuPleExtra.requireFirstLayer(spec, LayerSpec.Extras.ALTUP);
        PleLayout layout = PleLayout.resolve(spec);
        layout.requireLayerTensors(spec, layer);
        int e = layout.nEmbd();
        if (layer.residual().kind() != ResidualKind.STANDARD || layer.residual().streamWidth() != e) {
            throw new IllegalArgumentException(arch + " layer " + layer.il() + ": AltUp needs a STANDARD residual of"
                    + " n_embd " + e + ", got " + layer.residual());
        }
        CpuWeights weights = ctx.weights();
        CpuScratch scratch = ctx.scratch();
        int u = scratch.ubatch();
        this.il = layer.il();
        this.nAltup = a;
        this.act = hp.iAltupAct();
        this.nEmbd = e;
        this.ubatch = u;
        this.stride = Math.multiplyExact(u, e);
        this.eps = hp.normRmsEps();
        this.first = il == 0;
        this.last = il == spec.nLayer() - 1;
        this.router = CpuPleExtra.matrix(weights, layer.tensorName(TensorNames.weight(TensorNames.ALTUP_ROUTER)), e,
                a);
        this.routerNorm = CpuPleExtra.vector(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_ROUTER_NORM)), e);
        this.predictCoef = CpuPleExtra.matrix(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_PREDICT_COEF)), a, (long) a * a);
        this.correctCoef = CpuPleExtra.matrix(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_CORRECT_COEF)), a, a);
        CpuProjectionDeltas deltas = ctx.projectionDeltas();
        this.routerDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ALTUP_ROUTER), router, ctx.matVec());
        this.predictDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ALTUP_PREDICT_COEF), predictCoef,
                ctx.matVec());
        this.correctDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ALTUP_CORRECT_COEF), correctCoef,
                ctx.matVec());
        this.correctScale = CpuPleExtra.vector(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_CORRECT_SCALE)), e);
        this.streams = scratch.temp(GpuAltUpExtra.TEMP_STREAMS, Math.multiplyExact(a, stride));
        this.routerIn = scratch.temp(GpuAltUpExtra.TEMP_ROUTER, stride);
        this.modalities = scratch.temp(GpuAltUpExtra.TEMP_MODALITIES, u * a);
        this.coefs = scratch.temp(GpuAltUpExtra.TEMP_COEFS, u * a * a);
        this.firstPred = scratch.temp(GpuAltUpExtra.TEMP_FIRST, stride);
        this.inputs = CpuPleExtra.PerLayerInputs.array(scratch, layout);
        this.gate = new CpuPleExtra.LayerGate(ctx, layer, layout);
        if (first) {
            this.producer = new CpuPleExtra.PerLayerInputs(ctx, layout);
            this.altupProj = CpuPleExtra.matrix(weights, TensorNames.weight(TensorNames.ALTUP_PROJ), e,
                    (long) (a - 1) * e);
            this.proj = scratch.temp(GpuAltUpExtra.TEMP_PROJ, Math.multiplyExact(stride, a - 1));
        }
        if (last) {
            this.unembdProj = CpuPleExtra.matrix(weights, TensorNames.weight(TensorNames.ALTUP_UNEMBD_PROJ), e,
                    (long) (a - 1) * e);
            this.proj = scratch.temp(GpuAltUpExtra.TEMP_PROJ, Math.multiplyExact(stride, a - 1));
            this.merge = scratch.temp(GpuAltUpExtra.TEMP_MERGE, Math.multiplyExact(a, stride));
        }
        this.matVec = ctx.matVec();
    }

    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        checkStep(n, x);
        int len = n * nEmbd;
        if (first) {
            producer.compute(n, x);
            System.arraycopy(x, 0, streams, 0, len);
            int w = (nAltup - 1) * nEmbd;
            matVec.matmul(altupProj, x, 0, proj, 0, n);
            for (int j = 0; j < nAltup - 1; j++) {
                copyRows(proj, j * nEmbd, w, streams, (j + 1) * stride, nEmbd, n);
            }
            CpuAltUp.magnitudeMatch(n, nEmbd, nAltup - 1, stride, x, 0, streams, stride);
        }
        modalities(n, streams, act * stride);
        matVec.matmul(predictCoef, modalities, 0, coefs, 0, n);
        predictDeltas.applyRows(n, modalities, 0, coefs, 0);
        CpuAltUp.predict(n, nEmbd, nAltup, stride, coefs, 0, streams, 0, streams, 0);
        System.arraycopy(streams, act * stride, x, 0, len);
    }

    @Override
    public void afterBlock(int pos0, int n, float[] x) {
        checkStep(n, x);
        int len = n * nEmbd;
        modalities(n, x, 0);
        matVec.matmul(correctCoef, modalities, 0, coefs, 0, n);
        correctDeltas.applyRows(n, modalities, 0, coefs, 0);
        CpuAltUp.correct(n, nEmbd, nAltup, act, stride, coefs, 0, x, 0, streams, 0, streams, 0);
        System.arraycopy(streams, act * stride, firstPred, 0, len);
        NormMath.mulRowScale(n, nEmbd, correctScale, 0, nEmbd, firstPred, 0);
        float[] out = gate.apply(n, firstPred, 0, inputs);
        for (int j = 1; j < nAltup; j++) {
            CpuOps.add(streams, j * stride, out, 0, streams, j * stride, len);
        }
        if (last) {
            mergeStreams(n, x);
        } else {
            System.arraycopy(streams, act * stride, x, 0, len);
        }
    }

    /** Router modalities {@code tanh(altup_router . (rms_norm(in) * altup_router_norm / n_embd))} (gemma3n.cpp:398-410). */
    private void modalities(int n, float[] in, int inOff) {
        CpuOps.norm(NormKind.RMS, eps, in, inOff, routerIn, 0, n, nEmbd, routerNorm, null);
        CpuOps.scale(routerIn, 0, n * nEmbd, CpuAltUp.routerInputScale(nEmbd));
        matVec.matmul(router, routerIn, 0, modalities, 0, n);
        routerDeltas.applyRows(n, routerIn, 0, modalities, 0);
        CpuOps.unary(ActOp.TANH, ActParams.NONE, modalities, 0, modalities, 0, n * nAltup);
    }

    /** The final stream merge into {@code x} (gemma3n.cpp:266-287). */
    private void mergeStreams(int n, float[] x) {
        int w = (nAltup - 1) * nEmbd;
        for (int j = 1; j < nAltup; j++) {
            matVec.matmul(unembdProj, streams, j * stride, proj, 0, n);
            copyRows(proj, (j - 1) * nEmbd, w, merge, j * stride, nEmbd, n);
        }
        CpuAltUp.magnitudeMatch(n, nEmbd, nAltup - 1, stride, streams, act * stride, merge, stride);
        System.arraycopy(streams, 0, merge, 0, n * nEmbd);
        CpuAltUp.streamsMean(n, nEmbd, nAltup, stride, merge, 0, x, 0);
    }

    /** {@code rows} rows of {@code cols} from {@code src[srcOff + r*srcStride]} to dense rows at {@code dst[dstOff]}. */
    private static void copyRows(float[] src, int srcOff, int srcStride, float[] dst, int dstOff, int cols, int rows) {
        for (int r = 0; r < rows; r++) {
            System.arraycopy(src, srcOff + r * srcStride, dst, dstOff + r * cols, cols);
        }
    }

    private void checkStep(int n, float[] x) {
        if (matVec == null) {
            throw new IllegalStateException("AltUp extra is not bound");
        }
        GpuPleExtra.checkTokens(n, ubatch, "AltUp");
        CpuPleExtra.requireLength(x, (long) n * nEmbd, "residual stream");
    }

    @Override
    public String toString() {
        return "CpuAltUpExtra{" + (matVec == null ? "unbound" : "layer " + il + ", altup=" + nAltup + ", act=" + act
                + (first ? ", init" : "") + (last ? ", merge" : "")) + '}';
    }

    /**
     * CPU twin of {@link GpuAltUpExtra.Laurel} ({@code LayerSpec.Extras.LAUREL}):
     * {@code x = ((branch + x) + laurel(n)) * 1/sqrt(2)} with {@code n} the {@code attn_norm} output the executor
     * leaves in {@link CpuScratch#NORMED}.
     */
    public static final class Laurel implements CpuLayerExtra {

        private CpuMatVec matVec;
        private int il;
        private int nEmbd;
        private int ubatch;
        private float eps;
        private CpuTensorRef left;
        private CpuTensorRef right;
        /** LoRA deltas of {@code laurel_l} / {@code laurel_r} ({@code build_lora_mm}, gemma3n.cpp:379-380). */
        private CpuProjectionDeltas.Site leftDeltas;
        private CpuProjectionDeltas.Site rightDeltas;
        private float[] postNorm;
        private float[] normed;
        private float[] low;
        private float[] out;

        @Override
        public void bind(CpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            if (matVec != null) {
                throw new IllegalStateException("Laurel extra is already bound");
            }
            ModelSpec spec = ctx.spec();
            String arch = spec.arch().ggufName();
            int e = spec.hparams().nEmbd();
            int rank = spec.hparams().laurelRank();
            if (rank < 1) {
                throw new IllegalArgumentException(arch + ": Laurel needs a positive rank, got " + rank);
            }
            if (layer.residual().kind() != ResidualKind.STANDARD || layer.residual().streamWidth() != e) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": Laurel needs a STANDARD residual"
                        + " of n_embd " + e + ", got " + layer.residual());
            }
            if (!layer.norm().attnNorm()) {
                throw new IllegalArgumentException(arch + " layer " + layer.il() + ": Laurel reads the attn_norm"
                        + " output, and the layer has no attn_norm");
            }
            CpuWeights weights = ctx.weights();
            CpuScratch scratch = ctx.scratch();
            this.left = CpuPleExtra.matrix(weights, layer.tensorName(TensorNames.weight(TensorNames.LAUREL_L)), e,
                    rank);
            this.right = CpuPleExtra.matrix(weights, layer.tensorName(TensorNames.weight(TensorNames.LAUREL_R)),
                    rank, e);
            this.leftDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.LAUREL_L), left,
                    ctx.matVec());
            this.rightDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.LAUREL_R), right,
                    ctx.matVec());
            this.postNorm = CpuPleExtra.vector(weights,
                    layer.tensorName(TensorNames.weight(TensorNames.LAUREL_POST_NORM)), e);
            this.il = layer.il();
            this.nEmbd = e;
            this.ubatch = scratch.ubatch();
            this.eps = spec.hparams().normRmsEps();
            this.normed = scratch.slot(CpuScratch.NORMED);
            this.low = scratch.temp(GpuAltUpExtra.Laurel.TEMP_LOW, ubatch * rank);
            this.out = scratch.temp(GpuAltUpExtra.Laurel.TEMP_OUT, ubatch * e);
            this.matVec = ctx.matVec();
        }

        @Override
        public boolean afterMixer(int pos0, int n, float[] x, float[] branch) {
            if (matVec == null) {
                throw new IllegalStateException("Laurel extra is not bound");
            }
            GpuPleExtra.checkTokens(n, ubatch, "Laurel");
            int len = n * nEmbd;
            CpuPleExtra.requireLength(x, len, "residual stream");
            CpuPleExtra.requireLength(branch, len, "attention branch");
            matVec.matmul(left, normed, 0, low, 0, n);
            leftDeltas.applyRows(n, normed, 0, low, 0);
            matVec.matmul(right, low, 0, out, 0, n);
            rightDeltas.applyRows(n, low, 0, out, 0);
            CpuOps.norm(NormKind.RMS, eps, out, 0, out, 0, n, nEmbd, postNorm, null);
            CpuOps.add(out, 0, normed, 0, out, 0, len);
            CpuOps.add(branch, 0, x, 0, x, 0, len);
            CpuOps.add(x, 0, out, 0, x, 0, len);
            CpuOps.scale(x, 0, len, CpuAltUp.ATTN_LAUREL_SCALE);
            return true;
        }

        @Override
        public String toString() {
            return "CpuAltUpExtra.Laurel{" + (matVec == null ? "unbound" : "layer " + il) + '}';
        }
    }
}
