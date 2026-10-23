package com.eonmux.neuralfree.gguf.ops.gemma3n;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuLayerExtra;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.gemma3n.Gemma3nBlocks.PleLayout;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Gemma3n AltUp on the GPU ({@code LayerSpec.Extras.ALTUP}, llama.cpp src/models/gemma3n.cpp). The {@code n_altup}
 * streams of the step live in the scratch temporary {@link #TEMP_STREAMS} ({@code n_altup} blocks of
 * {@code ubatch*n_embd}, stream stride {@code S = ubatch*n_embd}, Gemma3nKernels layout); the residual stream
 * {@code x} carries the active stream through each layer's attention and FFN.
 *
 * <p>{@link #beforeBlock}: layer 0 first computes the per-layer inputs ({@link GpuPleExtra.PerLayerInputs}) and
 * the streams from the embedding output: stream 0 = x, streams {@code 1..n_altup-1} = {@code altup_proj} slices of x
 * magnitude-matched to x (gemma3n.cpp:111-126). Every layer then predicts (gemma3n.cpp:398-437): router
 * {@code tanh(altup_router . (rms_norm(stream[act]) * altup_router_norm / n_embd))}, coefficients
 * {@code altup_predict_coef . modalities}, {@code LLM_AltUp_Predict} in place, and copies the active prediction into
 * {@code x} (gemma3n.cpp:132-135).</p>
 *
 * <p>{@link #afterBlock} ({@code x} = attention, Laurel and FFN output {@code attn_ffw_laurel_gated}): router on
 * {@code x}, {@code altup_correct_coef}, {@code LLM_AltUp_Correct} in place (the kernel adds the 1); then the
 * per-layer embedding of the active corrected stream, scaled by {@code altup_correct_scale} before
 * {@code per_layer_inp_gate} (gemma3n.cpp:229-249), is added to streams {@code 1..n_altup-1} (gemma3n.cpp:251-258);
 * {@code x} becomes the active corrected stream. The last layer instead merges the streams
 * (gemma3n.cpp:266-287): {@code altup_unembd_proj} slices of streams {@code 1..}, magnitude-matched to the active
 * stream, averaged with stream 0 into {@code x}.</p>
 *
 * <p>The streams and per-layer inputs carry data from one block call to the next; their temporary names are used
 * by the Gemma3n extras only, so nothing else writes them between those calls. The AltUp kernels are prepared at
 * bind with sticky arguments; a step re-sets the token count and work size only. Not thread-safe.</p>
 */
public final class GpuAltUpExtra implements GpuLayerExtra {

    /** Scratch temporary: the AltUp streams, {@code n_altup * ubatch * n_embd}. */
    static final String TEMP_STREAMS = "gemma3n.altup.streams";
    /** Scratch temporary: router input, {@code ubatch * n_embd}. */
    static final String TEMP_ROUTER = "gemma3n.altup.router";
    /** Scratch temporary: router modalities, {@code ubatch * n_altup}. */
    static final String TEMP_MODALITIES = "gemma3n.altup.modalities";
    /** Scratch temporary: predict or correct coefficients, {@code ubatch * n_altup^2}. */
    static final String TEMP_COEFS = "gemma3n.altup.coefs";
    /** Scratch temporary: the scaled active corrected stream, {@code ubatch * n_embd} from element 0. */
    static final String TEMP_FIRST = "gemma3n.altup.first";
    /** Scratch temporary: {@code altup_proj} / {@code altup_unembd_proj} output, {@code ubatch * (n_altup-1) * n_embd}. */
    static final String TEMP_PROJ = "gemma3n.altup.proj";
    /** Scratch temporary: stream 0 then the unembedded streams of the merge, {@code n_altup * ubatch * n_embd}. */
    static final String TEMP_MERGE = "gemma3n.altup.merge";

    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private GpuOps ops;
    private int il;
    private int nAltup;
    private int act;
    private int nEmbd;
    private int ubatch;
    private int stride;
    private int elemLocal;
    private int rowLocal;
    private boolean first;
    private boolean last;
    private GpuWeight router;
    private GpuWeight routerNorm;
    private GpuWeight predictCoef;
    private GpuWeight correctCoef;
    /**
     * LoRA deltas of the router and the coefficient projections ({@code build_lora_mm}, gemma3n.cpp:421, 449; the
     * router is a plain {@code ggml_mul_mat} there, :410). The model-level {@code altup_proj} /
     * {@code altup_unembd_proj} stacks take none, as in llama.cpp (:121, 276).
     */
    private GpuProjectionDeltas.Site routerDeltas;
    private GpuProjectionDeltas.Site predictDeltas;
    private GpuProjectionDeltas.Site correctDeltas;
    private GpuWeight altupProj;
    private GpuWeight unembdProj;
    private float eps;
    private ScratchSlot streams;
    private ScratchSlot routerIn;
    private ScratchSlot modalities;
    private ScratchSlot coefs;
    private ScratchSlot firstPred;
    private ScratchSlot proj;
    private ScratchSlot merge;
    private ScratchSlot inputs;
    private GpuPleExtra.PerLayerInputs producer;
    private GpuPleExtra.LayerGate gate;
    private PreparedDispatch predict;
    private PreparedDispatch correct;
    private PreparedDispatch rowScale;
    private PreparedDispatch initMagnitude;
    private PreparedDispatch mergeMagnitude;
    private PreparedDispatch mean;
    private GpuBuffer boundX;
    private int boundXOff;

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (ops != null) {
            throw new IllegalStateException("AltUp extra is already bound");
        }
        PleLayout layout = checkLayer(ctx.spec(), layer);
        GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
        bindShape(ctx, layer, layout);
        GpuWeight correctScale = bindWeights(ctx, weights, layer);
        bindScratch(ctx, weights, layer, layout);
        try {
            prepareLayer(correctScale);
            if (first) {
                prepareFirst(ctx, weights, layout);
            }
            if (last) {
                prepareLast(ctx.scratch(), weights);
            }
        } catch (RuntimeException ex) {
            close();
            this.ops = null;
            throw ex;
        }
    }

    /**
     * The PLE layout of the model, after checking the AltUp stream count, the active stream, the first-layer
     * placement, the layer's tensors and its residual.
     */
    private static PleLayout checkLayer(ModelSpec spec, LayerSpec layer) {
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
        return layout;
    }

    /** Stream count and width, the layer's position in the stack and the work-group sizes. */
    private void bindShape(GpuForwardContext ctx, LayerSpec layer, PleLayout layout) {
        ModelSpec spec = ctx.spec();
        ModelHparams hp = spec.hparams();
        this.il = layer.il();
        this.nAltup = hp.nAltup();
        this.act = hp.iAltupAct();
        this.nEmbd = layout.nEmbd();
        this.ubatch = ctx.scratch().ubatch();
        this.stride = Math.toIntExact((long) ubatch * nEmbd);
        this.eps = hp.normRmsEps();
        this.first = il == 0;
        this.last = il == spec.nLayer() - 1;
        this.ops = ctx.ops();
        this.elemLocal = ops.local();
        this.rowLocal = Gemma3nKernels.rowLocal(ops.local());
    }

    /** The router, its norm and the predict / correct coefficients with their LoRA sites; returns the correct scale. */
    private GpuWeight bindWeights(GpuForwardContext ctx, GpuWeights weights, LayerSpec layer) {
        int a = nAltup;
        int e = nEmbd;
        this.router = GpuPleExtra.resident(weights, layer.tensorName(TensorNames.weight(TensorNames.ALTUP_ROUTER)),
                e, a);
        this.routerNorm = GpuPleExtra.residentVector(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_ROUTER_NORM)), e);
        this.predictCoef = GpuPleExtra.resident(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_PREDICT_COEF)), a, (long) a * a);
        this.correctCoef = GpuPleExtra.resident(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_CORRECT_COEF)), a, a);
        GpuWeight correctScale = GpuPleExtra.residentVector(weights,
                layer.tensorName(TensorNames.weight(TensorNames.ALTUP_CORRECT_SCALE)), e);
        GpuProjectionDeltas deltas = ctx.projectionDeltas();
        this.routerDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ALTUP_ROUTER), router);
        this.predictDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ALTUP_PREDICT_COEF), predictCoef);
        this.correctDeltas = deltas.serve(layer, TensorNames.weight(TensorNames.ALTUP_CORRECT_COEF), correctCoef);
        return correctScale;
    }

    /** The stream, router and coefficient temporaries, the per-layer inputs, the layer gate and the bound X slot. */
    private void bindScratch(GpuForwardContext ctx, GpuWeights weights, LayerSpec layer, PleLayout layout) {
        GpuScratch scratch = ctx.scratch();
        int a = nAltup;
        this.streams = scratch.temp(TEMP_STREAMS, (long) a * stride);
        this.routerIn = scratch.temp(TEMP_ROUTER, stride);
        this.modalities = scratch.temp(TEMP_MODALITIES, (long) ubatch * a);
        this.coefs = scratch.temp(TEMP_COEFS, (long) ubatch * a * a);
        this.firstPred = scratch.temp(TEMP_FIRST, stride);
        this.inputs = GpuPleExtra.PerLayerInputs.slot(scratch, layout);
        this.gate = new GpuPleExtra.LayerGate(ctx, weights, layer, layout);
        ScratchSlot x = ctx.scratch().slot(ScratchSlots.X);
        this.boundX = x.buffer();
        this.boundXOff = x.offset();
    }

    /** First layer: the per-layer input producer, {@code altup_proj} and the initial magnitude match. */
    private void prepareFirst(GpuForwardContext ctx, GpuWeights weights, PleLayout layout) {
        int a = nAltup;
        this.producer = new GpuPleExtra.PerLayerInputs(ctx, weights, layout);
        this.altupProj = GpuPleExtra.resident(weights, TensorNames.weight(TensorNames.ALTUP_PROJ), nEmbd,
                (long) (a - 1) * nEmbd);
        this.proj = ctx.scratch().temp(TEMP_PROJ, (long) stride * (a - 1));
        this.initMagnitude = prepare(Gemma3nKernels.MAGNITUDE_MATCH, Gemma3nKernels.rowGlobal(
                (long) (a - 1) * ubatch, rowLocal), rowLocal, Gemma3nKernels.magnitudeMatchArgs(ubatch, nEmbd,
                a - 1, stride, boundX, boundXOff, streams.buffer(), streams.offset() + stride));
    }

    /** Last layer: {@code altup_unembd_proj}, the merge magnitude match and the stream mean back into X. */
    private void prepareLast(GpuScratch scratch, GpuWeights weights) {
        int a = nAltup;
        int e = nEmbd;
        this.unembdProj = GpuPleExtra.resident(weights, TensorNames.weight(TensorNames.ALTUP_UNEMBD_PROJ), e,
                (long) (a - 1) * e);
        this.proj = scratch.temp(TEMP_PROJ, (long) stride * (a - 1));
        this.merge = scratch.temp(TEMP_MERGE, (long) a * stride);
        this.mergeMagnitude = prepare(Gemma3nKernels.MAGNITUDE_MATCH, Gemma3nKernels.rowGlobal(
                (long) (a - 1) * ubatch, rowLocal), rowLocal, Gemma3nKernels.magnitudeMatchArgs(ubatch, e,
                a - 1, stride, streams.buffer(), streams.offset() + act * stride, merge.buffer(),
                merge.offset() + stride));
        this.mean = prepare(Gemma3nKernels.STREAMS_MEAN, Gemma3nKernels.elementwiseGlobal(ubatch, e,
                elemLocal), elemLocal, Gemma3nKernels.streamsMeanArgs(ubatch, e, a, stride, merge.buffer(),
                merge.offset(), boundX, boundXOff));
    }

    private void prepareLayer(GpuWeight correctScale) {
        int a = nAltup;
        this.predict = prepare(Gemma3nKernels.ALTUP_PREDICT, Gemma3nKernels.elementwiseGlobal(ubatch, nEmbd,
                elemLocal), elemLocal, Gemma3nKernels.predictArgs(ubatch, nEmbd, a, stride, coefs.buffer(),
                coefs.offset(), streams.buffer(), streams.offset(), streams.buffer(), streams.offset()));
        this.correct = prepare(Gemma3nKernels.ALTUP_CORRECT, Gemma3nKernels.elementwiseGlobal(ubatch, nEmbd,
                elemLocal), elemLocal, Gemma3nKernels.correctArgs(ubatch, nEmbd, a, act, stride, coefs.buffer(),
                coefs.offset(), boundX, boundXOff, streams.buffer(), streams.offset(), streams.buffer(),
                streams.offset()));
        if (firstPred.offset() != 0) {
            throw new IllegalStateException("AltUp temporary " + TEMP_FIRST + " must start at element 0 of its buffer");
        }
        this.rowScale = prepare(NormKernels.MUL_ROW_SCALE, ActKernels.global1d((long) stride, elemLocal), elemLocal,
                NormKernels.mulRowScaleArgs(ubatch, nEmbd, correctScale.shard(0), nEmbd, firstPred.buffer()));
    }

    private PreparedDispatch prepare(String kernel, long[] global, int local, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, new long[]{local}, args);
        prepared.add(d);
        return d;
    }

    @Override
    public void beforeBlock(int pos0, int nTokens, ScratchSlot x) {
        checkStep(nTokens, x);
        int n = nTokens * nEmbd;
        GpuBuffer s = streams.buffer();
        int s0 = streams.offset();
        if (first) {
            producer.compute(nTokens, x);
            ops.copy(x.buffer(), x.offset(), s, s0, n);
            int w = (nAltup - 1) * nEmbd;
            ops.matmul(altupProj, x.buffer(), x.offset(), proj.buffer(), proj.offset(), nTokens);
            for (int j = 0; j < nAltup - 1; j++) {
                ops.copy2d(proj.buffer(), proj.offset() + j * nEmbd, w, s, s0 + (j + 1) * stride, nEmbd, nTokens,
                        nEmbd);
            }
            run(initMagnitude, Gemma3nKernels.ARG_N_TOKENS, nTokens,
                    Gemma3nKernels.rowGlobal((long) (nAltup - 1) * nTokens, rowLocal));
        }
        modalities(nTokens, s, s0 + act * stride);
        ops.matmul(predictCoef, modalities.buffer(), modalities.offset(), coefs.buffer(), coefs.offset(), nTokens);
        predictDeltas.applyRows(nTokens, modalities.buffer(), modalities.offset(), coefs.buffer(), coefs.offset());
        run(predict, Gemma3nKernels.ARG_N_TOKENS, nTokens,
                Gemma3nKernels.elementwiseGlobal(nTokens, nEmbd, elemLocal));
        ops.copy(s, s0 + act * stride, x.buffer(), x.offset(), n);
    }

    @Override
    public void afterBlock(int pos0, int nTokens, ScratchSlot x) {
        checkStep(nTokens, x);
        int n = nTokens * nEmbd;
        GpuBuffer s = streams.buffer();
        int s0 = streams.offset();
        modalities(nTokens, x.buffer(), x.offset());
        ops.matmul(correctCoef, modalities.buffer(), modalities.offset(), coefs.buffer(), coefs.offset(), nTokens);
        correctDeltas.applyRows(nTokens, modalities.buffer(), modalities.offset(), coefs.buffer(), coefs.offset());
        run(correct, Gemma3nKernels.ARG_N_TOKENS, nTokens,
                Gemma3nKernels.elementwiseGlobal(nTokens, nEmbd, elemLocal));
        ops.copy(s, s0 + act * stride, firstPred.buffer(), firstPred.offset(), n);
        run(rowScale, NormKernels.ARG_ROW_SCALE_N_ROWS, nTokens, ActKernels.global1d((long) n, elemLocal));
        ScratchSlot out = gate.apply(nTokens, firstPred.buffer(), firstPred.offset(), inputs);
        for (int j = 1; j < nAltup; j++) {
            int at = s0 + j * stride;
            ops.add(s, at, out.buffer(), out.offset(), s, at, n);
        }
        if (last) {
            mergeStreams(nTokens, x);
        } else {
            ops.copy(s, s0 + act * stride, x.buffer(), x.offset(), n);
        }
    }

    /** Router modalities {@code tanh(altup_router . (rms_norm(in) * altup_router_norm / n_embd))} (gemma3n.cpp:398-410). */
    private void modalities(int nTokens, GpuBuffer in, int inOff) {
        GpuBuffer r = routerIn.buffer();
        int rOff = routerIn.offset();
        ops.norm(NormKind.RMS, routerNorm, null, eps, in, inOff, r, rOff, nTokens, nEmbd);
        ops.scale(r, rOff, nTokens * nEmbd, CpuAltUp.routerInputScale(nEmbd));
        ops.matmul(router, r, rOff, modalities.buffer(), modalities.offset(), nTokens);
        routerDeltas.applyRows(nTokens, r, rOff, modalities.buffer(), modalities.offset());
        ops.unary(ActOp.TANH, ActParams.NONE, modalities.buffer(), modalities.offset(), modalities.buffer(),
                modalities.offset(), nTokens * nAltup);
    }

    /** The final stream merge into {@code x} (gemma3n.cpp:266-287). */
    private void mergeStreams(int nTokens, ScratchSlot x) {
        int n = nTokens * nEmbd;
        GpuBuffer s = streams.buffer();
        int s0 = streams.offset();
        GpuBuffer m = merge.buffer();
        int m0 = merge.offset();
        int w = (nAltup - 1) * nEmbd;
        for (int j = 1; j < nAltup; j++) {
            ops.matmul(unembdProj, s, s0 + j * stride, proj.buffer(), proj.offset(), nTokens);
            ops.copy2d(proj.buffer(), proj.offset() + (j - 1) * nEmbd, w, m, m0 + j * stride, nEmbd, nTokens, nEmbd);
        }
        run(mergeMagnitude, Gemma3nKernels.ARG_N_TOKENS, nTokens,
                Gemma3nKernels.rowGlobal((long) (nAltup - 1) * nTokens, rowLocal));
        ops.copy(s, s0, m, m0, n);
        run(mean, Gemma3nKernels.ARG_N_TOKENS, nTokens,
                Gemma3nKernels.elementwiseGlobal(nTokens, nEmbd, elemLocal));
    }

    /** Re-sets a prepared dispatch's token-count argument {@code nArg} and work size, then launches it. */
    private static void run(PreparedDispatch d, int nArg, int nTokens, long[] global) {
        d.setInt(nArg, nTokens);
        d.setGlobal(global);
        d.dispatch();
    }

    private void checkStep(int nTokens, ScratchSlot x) {
        if (ops == null) {
            throw new IllegalStateException("AltUp extra is not bound");
        }
        GpuPleExtra.checkTokens(nTokens, ubatch, "AltUp");
        Objects.requireNonNull(x, "x").require((long) nTokens * nEmbd);
        if (x.buffer() != boundX || x.offset() != boundXOff) {
            throw new IllegalStateException("AltUp layer " + il + " was bound to the residual slot "
                    + ScratchSlots.X + "; the executor passed another view " + x);
        }
    }

    @Override
    public void close() {
        RuntimeException failure = Closeables.closeAll(prepared, null);
        prepared.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuAltUpExtra{" + (ops == null ? "unbound" : "layer " + il + ", altup=" + nAltup + ", act=" + act
                + (first ? ", init" : "") + (last ? ", merge" : "")) + '}';
    }

    /**
     * Gemma3n Laurel on the GPU ({@code LayerSpec.Extras.LAUREL}, gemma3n.cpp:139-141, 199-204 and 375-384):
     * {@code afterMixer} folds the attention branch into {@code x} as
     * {@code x = ((branch + x) + laurel(n)) * 1/sqrt(2)} with {@code n} the layer's {@code attn_norm} output, which the
     * executor leaves in {@link ScratchSlots#NORMED} through the mixer, and
     * {@code laurel(n) = rms_norm(laurel_r . (laurel_l . n)) * laurel_post_norm + n}. Dispatches per step: two
     * projections, {@code LLM_RowNorm}, three adds ({@code LLM_Axpby_F32}) and the scale.
     */
    public static final class Laurel implements GpuLayerExtra {

        /** Scratch temporary: low-rank Laurel activations, {@code ubatch * laurel_rank}. */
        static final String TEMP_LOW = "gemma3n.laurel.low";
        /** Scratch temporary: Laurel output, {@code ubatch * n_embd}. */
        static final String TEMP_OUT = "gemma3n.laurel.out";

        private GpuOps ops;
        private int il;
        private int nEmbd;
        private int ubatch;
        private float eps;
        private GpuWeight left;
        private GpuWeight right;
        /** LoRA deltas of {@code laurel_l} / {@code laurel_r} ({@code build_lora_mm}, gemma3n.cpp:379-380). */
        private GpuProjectionDeltas.Site leftDeltas;
        private GpuProjectionDeltas.Site rightDeltas;
        private GpuWeight postNorm;
        private ScratchSlot normed;
        private ScratchSlot low;
        private ScratchSlot out;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            Objects.requireNonNull(ctx, "ctx");
            Objects.requireNonNull(layer, "layer");
            if (ops != null) {
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
            GpuWeights weights = ctx.weightSource().weightsFor(layer.weightLayer());
            this.left = GpuPleExtra.resident(weights, layer.tensorName(TensorNames.weight(TensorNames.LAUREL_L)), e,
                    rank);
            this.right = GpuPleExtra.resident(weights, layer.tensorName(TensorNames.weight(TensorNames.LAUREL_R)),
                    rank, e);
            this.postNorm = GpuPleExtra.residentVector(weights,
                    layer.tensorName(TensorNames.weight(TensorNames.LAUREL_POST_NORM)), e);
            this.leftDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.LAUREL_L), left);
            this.rightDeltas = ctx.projectionDeltas().serve(layer, TensorNames.weight(TensorNames.LAUREL_R), right);
            GpuScratch scratch = ctx.scratch();
            this.il = layer.il();
            this.nEmbd = e;
            this.ubatch = scratch.ubatch();
            this.eps = spec.hparams().normRmsEps();
            this.normed = scratch.slot(ScratchSlots.NORMED);
            this.low = scratch.temp(TEMP_LOW, (long) ubatch * rank);
            this.out = scratch.temp(TEMP_OUT, (long) ubatch * e);
            this.ops = ctx.ops();
        }

        @Override
        public boolean afterMixer(int pos0, int nTokens, ScratchSlot x, ScratchSlot branch) {
            if (ops == null) {
                throw new IllegalStateException("Laurel extra is not bound");
            }
            GpuPleExtra.checkTokens(nTokens, ubatch, "Laurel");
            int n = nTokens * nEmbd;
            Objects.requireNonNull(x, "x").require(n);
            Objects.requireNonNull(branch, "branch").require(n);
            GpuBuffer o = out.buffer();
            int oOff = out.offset();
            ops.matmul(left, normed.buffer(), normed.offset(), low.buffer(), low.offset(), nTokens);
            leftDeltas.applyRows(nTokens, normed.buffer(), normed.offset(), low.buffer(), low.offset());
            ops.matmul(right, low.buffer(), low.offset(), o, oOff, nTokens);
            rightDeltas.applyRows(nTokens, low.buffer(), low.offset(), o, oOff);
            ops.norm(NormKind.RMS, postNorm, null, eps, o, oOff, o, oOff, nTokens, nEmbd);
            ops.add(o, oOff, normed.buffer(), normed.offset(), o, oOff, n);
            ops.add(branch.buffer(), branch.offset(), x.buffer(), x.offset(), x.buffer(), x.offset(), n);
            ops.add(x.buffer(), x.offset(), o, oOff, x.buffer(), x.offset(), n);
            ops.scale(x.buffer(), x.offset(), n, CpuAltUp.ATTN_LAUREL_SCALE);
            return true;
        }

        @Override
        public String toString() {
            return "GpuAltUpExtra.Laurel{" + (ops == null ? "unbound" : "layer " + il) + '}';
        }
    }
}
