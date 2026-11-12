package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.tensor.NormOps;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The GPU trainer: the same forward pass and cross-entropy loss as {@link TrainableModel}, evaluated and
 * differentiated through the project's tensor GPU stack.
 *
 * <p>The tensor binding, the {@link FinetunePlan} and the host float arrays are the ones {@link TrainableModel}
 * builds, so a parameter's name, shape and master values are identical on both trainers and the GGUF writer needs no
 * device awareness. This class owns one device {@link Tensor} per parameter ({@link DeviceParameters}) and expresses
 * the model with the differentiable {@link TensorOps} and {@link NormOps} operations; those dispatch to the project's
 * {@code train} kernel library on a GPU backend (matmul, row norms, softmax, the SiLU/GELU/ReLU family, row gather
 * and scatter, elementwise arithmetic) and to the CPU reference when a kernel is unavailable. The backward pass is the
 * autograd backward of that graph, computed on the same backend.</p>
 *
 * <p><b>Residency.</b> The device tensors are the authoritative values and gradients: a window computes with them,
 * its backward pass leaves the gradients on the device, and the optimizer {@link #newUpdate} provides steps them
 * there with its moments on the device ({@link DeviceFinetuneUpdate}), so a run moves no weight or gradient between
 * host and device per window. The host arrays of {@link #parameters()} are a mirror read back when accessed; a host
 * optimizer (the fallback when the moments do not fit the device) writes them and they are uploaded before the next
 * window.</p>
 *
 * <p>Each window is recorded as one {@link ComputationGraph} step; closing the step releases its activations. The
 * loss and the argmax count are read after the backward pass is enqueued, so the device runs the whole step without
 * waiting for the host in between. Every {@link
 * TrainableModel.StepResult} and every gradient computes the formulas of {@link TrainMath}: the residual scaling,
 * sandwich norms, RoPE, grouped-query causal (and sliding-window) attention with its soft and logit soft-caps, dense
 * and routed-expert feed-forward blocks with their gating, the embedding scale and the tied output head are the
 * plan's, so a model trains to the same result on either trainer.</p>
 *
 * <p>The GPU family this trainer represents: the {@link TensorOps} activation set. GELU's exact (erf) and "quick"
 * forms have no train kernel and are refused by name; every other dense, MoE, sliding-window, sandwich-norm and
 * soft-cap geometry the plan accepts is supported. MoE experts are evaluated densely (every expert of every token,
 * weighted by the routing probabilities), which is the same function as the sparse evaluation and is what the
 * autograd graph can weight and differentiate without a gather over expert matrices.</p>
 *
 * <p>Not thread-safe; one window at a time. The backend is owned by the caller and is not closed here.</p>
 */
public final class GpuTrainableModel implements Trainer {

    private final TrainableModel host;
    private final ComputeBackend backend;
    private final int nCtx;
    private final int nEmbd;
    private final int nVocab;
    private final FinetunePlan plan;
    private final DeviceParameters parameters;
    /** The device updates handed out, closed with the trainer if their optimizer was not closed first. */
    private final List<DeviceFinetuneUpdate> updates = new ArrayList<>();
    private final Map<String, Tensor> cachedConstants = new HashMap<>();
    private final Map<Integer, RopeFreqTable> ropeTables = new HashMap<>();
    private boolean closed;

    private GpuTrainableModel(TrainableModel host, ComputeBackend backend) {
        this.host = host;
        this.backend = Objects.requireNonNull(backend, "backend");
        this.plan = host.plan();
        this.nCtx = host.nCtx();
        this.nEmbd = host.nEmbd();
        this.nVocab = host.nVocab();
        checkRepresentable(plan);
        this.parameters = DeviceParameters.upload(backend, host.parameters());
    }

    /**
     * A GPU trainer over {@code model}, running windows of {@code nCtx} tokens on {@code backend}.
     *
     * @throws UnsupportedOperationException when the model or an activation is outside what this trainer runs
     * @throws IllegalArgumentException      when the window cannot fit or a tensor shape contradicts the plan
     */
    public static GpuTrainableModel open(GGUFModel model, int nCtx, ComputeBackend backend) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(backend, "backend");
        TrainableModel host = TrainableModel.open(model, nCtx);
        try {
            return new GpuTrainableModel(host, backend);
        } catch (RuntimeException e) {
            host.close();
            throw e;
        }
    }

    /** Whether the plan uses only operations the tensor GPU stack provides. */
    private static void checkRepresentable(FinetunePlan plan) {
        for (FinetunePlan.Layer layer : plan.layers()) {
            if (layer.glu() != null) {
                requireGlu(layer.glu(), "layer " + layer.il());
            }
            if (layer.act() != null) {
                requireAct(layer.act(), "layer " + layer.il());
            }
            FinetunePlan.Moe moe = layer.moe();
            if (moe != null) {
                if (moe.glu() != null) {
                    requireGlu(moe.glu(), "layer " + layer.il() + " experts");
                }
                if (moe.act() != null) {
                    requireAct(moe.act(), "layer " + layer.il() + " experts");
                }
            }
        }
    }

    private static void requireGlu(GluOp op, String where) {
        if (op != GluOp.SWIGLU && op != GluOp.GEGLU_TANH && op != GluOp.REGLU) {
            throw new UnsupportedOperationException("the GPU finetune trainer runs SWIGLU, GEGLU_TANH and REGLU"
                    + " gates; " + where + " runs " + op + " (train on the CPU with -ngl 0)");
        }
    }

    private static void requireAct(ActOp op, String where) {
        if (op != ActOp.SILU && op != ActOp.GELU_TANH && op != ActOp.RELU) {
            throw new UnsupportedOperationException("the GPU finetune trainer runs SILU, GELU_TANH and RELU"
                    + " activations; " + where + " runs " + op + " (train on the CPU with -ngl 0)");
        }
    }

    // ------------------------------------------------------------------ parameters

    private Tensor deviceTensor(String name) {
        return parameters.tensor(name);
    }

    // ------------------------------------------------------------------ accessors

    @Override
    public FinetunePlan plan() {
        return plan;
    }

    @Override
    public int nCtx() {
        return nCtx;
    }

    @Override
    public int nEmbd() {
        return nEmbd;
    }

    @Override
    public int nVocab() {
        return nVocab;
    }

    @Override
    public List<TrainableModel.Parameter> parameters() {
        return parameters.parameters();
    }

    @Override
    public TrainableModel.Parameter parameter(String name) {
        return parameters.parameter(name);
    }

    // ------------------------------------------------------------------ steps

    @Override
    public TrainableModel.StepResult forward(int[] tokens, int[] labels) {
        return run(tokens, labels, false);
    }

    @Override
    public TrainableModel.StepResult step(int[] tokens, int[] labels) {
        return run(tokens, labels, true);
    }

    @Override
    public void zeroGrad() {
        parameters.zeroGrad();
    }

    /**
     * The device update ({@link DeviceFinetuneUpdate}) when its memory plan places AdamW's moments on the device or
     * the optimizer is SGD; otherwise the host update over the mirrored parameter arrays.
     */
    @Override
    public FinetuneOptimizer.Update newUpdate(FinetuneOptimizer.Kind kind, float weightDecay) {
        DeviceFinetuneUpdate device = DeviceFinetuneUpdate.open(parameters, backend, kind, weightDecay);
        if (device == null) {
            return FinetuneOptimizer.hostUpdate(parameters(), kind, weightDecay);
        }
        updates.add(device);
        return device;
    }

    /** The bytes of parameter values and gradients moved between host and device since the trainer opened. */
    long hostTransferBytes() {
        return parameters.transferredBytes();
    }

    @Override
    public float[][] logits(int[] tokens) {
        checkWindow(tokens);
        parameters.syncDevice();
        Ctx ctx = new Ctx();
        try (ComputationGraph.StepScope scope = ComputationGraph.current().step()) {
            Tensor rows = head(tokens.length, blockStack(tokens.length, tokens, ctx));
            return toRows(rows, tokens.length);
        } finally {
            ctx.closeTransient();
        }
    }

    /**
     * One window, training (records a graph and backpropagates, accumulating into the device gradients) or
     * evaluating under the same graph shape. The labels go up with the tokens, before the first kernel of the window:
     * an upload after an index-validating kernel (the embedding gather) would first drain the compute queue to check
     * its index errors ({@code TrainGpu.beforeHostRead}). The loss and the predictions are read once the whole step
     * is enqueued, the window's one host synchronization.
     */
    private TrainableModel.StepResult run(int[] tokens, int[] labels, boolean training) {
        checkWindow(tokens, labels);
        parameters.syncDevice();
        int n = tokens.length;
        Ctx ctx = new Ctx();
        try (ComputationGraph.StepScope scope = ComputationGraph.current().step()) {
            Tensor labelIds = ctx.ints(labels, n, 1);
            Tensor stream = blockStack(n, tokens, ctx);
            Tensor logits = head(n, stream);
            Tensor lossTensor = lossTensor(logits, labelIds);
            try (Tensor predicted = TensorOps.argmax(logits, 1)) {
                if (training) {
                    try {
                        lossTensor.backward();
                    } finally {
                        parameters.deviceGradientsChanged();
                    }
                }
                float loss = lossTensor.toFloatArray()[0];
                return new TrainableModel.StepResult(loss, correct(predicted, labels, n), n);
            }
        } finally {
            ctx.closeTransient();
        }
    }

    private void checkWindow(int[] tokens) {
        checkWindow(tokens, tokens);
    }

    private void checkWindow(int[] tokens, int[] labels) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(labels, "labels");
        if (tokens.length != labels.length) {
            throw new IllegalArgumentException("the window holds " + tokens.length + " tokens and " + labels.length
                    + " labels");
        }
        if (tokens.length < 1 || tokens.length > nCtx) {
            throw new IllegalArgumentException("the window holds " + tokens.length + " tokens, outside [1, " + nCtx
                    + "]");
        }
        for (int t = 0; t < tokens.length; t++) {
            if (tokens[t] < 0 || tokens[t] >= nVocab) {
                throw new IllegalArgumentException("token " + tokens[t] + " at position " + t + " is outside the "
                        + nVocab + "-token vocabulary");
            }
            if (labels[t] < 0 || labels[t] >= nVocab) {
                throw new IllegalArgumentException("label " + labels[t] + " at position " + t + " is outside the "
                        + nVocab + "-token vocabulary");
            }
        }
    }

    // ------------------------------------------------------------------ graph

    /** The embedding, every trunk block, and the final normed stream. */
    private Tensor blockStack(int n, int[] tokens, Ctx ctx) {
        Tensor idx = ctx.ints(tokens, n);
        Tensor stream = TensorOps.indexSelectRows(deviceTensor(plan.tokenEmbd()), idx, false);
        float embdScale = plan.embdScale();
        if (embdScale != 1.0f) {
            stream = TensorOps.mul(stream, embdScale);
        }
        for (int il = 0; il < plan.nLayer(); il++) {
            stream = block(plan.layer(il), stream, n, ctx);
        }
        return stream;
    }

    /** The output norm and the LM head, with the final soft-cap. */
    private Tensor head(int n, Tensor stream) {
        Tensor normed = norm(plan.outputNorm(), stream, nEmbd);
        Tensor logits = linear(normed, plan.outputWeight(), plan.outputBias());
        float cap = plan.finalSoftcap();
        if (cap > 0f) {
            logits = TensorOps.mul(TensorOps.tanh(TensorOps.mul(logits, 1f / cap)), cap);
        }
        return logits;
    }

    /** One block: pre-norm attention, its residual, the FFN's norm and branch, and the second residual. */
    private Tensor block(FinetunePlan.Layer l, Tensor x, int n, Ctx ctx) {
        float scale = l.residualScale();
        Tensor h = norm(l.attnNorm(), x, nEmbd);
        Tensor q = reshape3(linear(h, l.attnQ(), l.attnQBias()), n, l.nHead(), l.headDimK());
        Tensor k = reshape3(linear(h, l.attnK(), l.attnKBias()), n, l.nHeadKv(), l.headDimK());
        Tensor v = reshape3(linear(h, l.attnV(), l.attnVBias()), n, l.nHeadKv(), l.headDimV());
        boolean afterRope = l.qkNorm() != null && l.qkNorm().afterRope();
        if (l.qkNorm() != null && !afterRope) {
            q = qkNorm(l.qkNorm().query(), q, n, l.nHead(), l.headDimK());
            k = qkNorm(l.qkNorm().key(), k, n, l.nHeadKv(), l.headDimK());
        }
        q = rope(l, q, n, l.nHead(), l.headDimK());
        k = rope(l, k, n, l.nHeadKv(), l.headDimK());
        if (l.qkNorm() != null && afterRope) {
            q = qkNorm(l.qkNorm().query(), q, n, l.nHead(), l.headDimK());
            k = qkNorm(l.qkNorm().key(), k, n, l.nHeadKv(), l.headDimK());
        }
        Tensor attn = attention(l, q, k, v, n);
        Tensor proj = linear(attn, l.attnOut(), l.attnOutBias());
        Tensor attnBranch = l.postAttnNorm() != null ? norm(l.postAttnNorm(), proj, nEmbd) : proj;
        Tensor x2 = TensorOps.add(x, scaled(attnBranch, scale));
        Tensor ffnIn = l.ffnNorm() != null ? norm(l.ffnNorm(), x2, nEmbd) : h;
        Tensor ffnOut = l.moe() != null ? moe(l.moe(), ffnIn, n, ctx) : denseFfn(l, ffnIn);
        Tensor ffnBranch = l.postFfnNorm() != null ? norm(l.postFfnNorm(), ffnOut, nEmbd) : ffnOut;
        return TensorOps.add(x2, scaled(ffnBranch, scale));
    }

    private static Tensor scaled(Tensor x, float scale) {
        return scale == 1f ? x : TensorOps.mul(x, scale);
    }

    /** {@code y = x W^T + b}, the row-major GGUF matrix convention. */
    private Tensor linear(Tensor x, String weight, String bias) {
        return TensorOps.linear(x, deviceTensor(weight), bias == null ? null : deviceTensor(bias));
    }

    /**
     * The norm as {@link TrainMath#normForward} computes it: RMS or layer normalization over each row of
     * {@code rowLen} values, scaled by the weight the plan binds and shifted by its bias. The tensor stack's norm
     * operations take both operands, so a missing weight or bias is a constant of ones or zeros; an RMS norm has no
     * bias operand, so a bound bias is added after it.
     */
    private Tensor norm(FinetunePlan.Norm norm, Tensor x, int rowLen) {
        Tensor w = norm.weight() != null ? deviceTensor(norm.weight()) : constantRow(1f, rowLen);
        if (norm.kind() == NormKind.RMS) {
            Tensor normed = NormOps.rmsNorm(x, w, norm.eps());
            return norm.bias() != null
                    ? TensorOps.add(normed, deviceTensor(norm.bias()).reshape(1, rowLen)) : normed;
        }
        Tensor b = norm.bias() != null ? deviceTensor(norm.bias()) : constantRow(0f, rowLen);
        return NormOps.layerNorm(x, rowLen, w, b, norm.eps());
    }

    /** A cached constant row of {@code value}: the operand a norm with no weight or bias tensor reads. */
    private Tensor constantRow(float value, int rowLen) {
        String key = (value == 1f ? "ones:" : "zeros:") + rowLen;
        Tensor cached = cachedConstants.get(key);
        if (cached != null) {
            return cached;
        }
        float[] values = new float[rowLen];
        java.util.Arrays.fill(values, value);
        Tensor constant = constant(values, rowLen);
        cachedConstants.put(key, constant);
        return constant;
    }

    private Tensor qkNorm(FinetunePlan.Norm norm, Tensor x, int n, int heads, int headDim) {
        Tensor flat = x.reshape(n * heads, headDim);
        return norm(norm, flat, headDim).reshape(n, heads, headDim);
    }

    private static Tensor reshape3(Tensor flat, int n, int heads, int headDim) {
        return flat.reshape(n, heads, headDim);
    }

    // ------------------------------------------------------------------ attention

    /** Causal (or sliding-window) grouped-query attention with the layer's scale and soft-cap. */
    private Tensor attention(FinetunePlan.Layer l, Tensor q, Tensor k, Tensor v, int n) {
        int nHead = l.nHead();
        int nHeadKv = l.nHeadKv();
        Tensor qT = q.permute(1, 0, 2);
        Tensor kT = k.permute(1, 0, 2);
        Tensor vT = v.permute(1, 0, 2);
        if (nHeadKv != nHead) {
            kT = expandHeads(kT, nHeadKv, nHead, l.headDimK(), n);
            vT = expandHeads(vT, nHeadKv, nHead, l.headDimV(), n);
        }
        Tensor scores = TensorOps.bmm(qT, kT.transpose(0, 2, 1));
        scores = TensorOps.mul(scores, l.scoreScale());
        float softcap = l.attnSoftcap();
        if (softcap > 0f) {
            scores = TensorOps.mul(TensorOps.tanh(TensorOps.mul(scores, 1f / softcap)), softcap);
        }
        Tensor mask = mask(n, l.swaWindow());
        if (mask != null) {
            scores = TensorOps.add(scores, mask);
        }
        Tensor probs = TensorOps.softmax(scores, 2);
        return TensorOps.bmm(probs, vT).permute(1, 0, 2).reshape(n, nHead * l.headDimV());
    }

    /** Copies each KV head to the {@code group} query heads that read it. */
    private Tensor expandHeads(Tensor x, int nHeadKv, int nHead, int headDim, int n) {
        Tensor table = x.reshape(nHeadKv, n * headDim);
        Tensor gathered = TensorOps.indexSelectRows(table, headIndex(nHeadKv, nHead), false);
        return gathered.reshape(nHead, n, headDim);
    }

    /** The KV head each query head reads, {@code h / (nHead / nHeadKv)}, cached per grouping. */
    private Tensor headIndex(int nHeadKv, int nHead) {
        String key = "heads:" + nHeadKv + ":" + nHead;
        Tensor cached = cachedConstants.get(key);
        if (cached != null) {
            return cached;
        }
        int group = nHead / nHeadKv;
        int[] index = new int[nHead];
        for (int h = 0; h < nHead; h++) {
            index[h] = h / group;
        }
        Tensor table = intConstant(index, nHead);
        cachedConstants.put(key, table);
        return table;
    }

    /** The additive causal mask of a window, cached per length. */
    private Tensor mask(int n, int window) {
        String key = "mask:" + window + ":" + n;
        Tensor cached = cachedConstants.get(key);
        if (cached != null) {
            return cached;
        }
        float[] values = new float[n * n];
        for (int t = 0; t < n; t++) {
            int first = window > 0 ? Math.max(0, t - window + 1) : 0;
            for (int j = 0; j < n; j++) {
                values[t * n + j] = j >= first && j <= t ? 0f : Float.NEGATIVE_INFINITY;
            }
        }
        Tensor mask = constant(values, 1, n, n);
        cachedConstants.put(key, mask);
        return mask;
    }

    // ------------------------------------------------------------------ rope

    private Tensor rope(FinetunePlan.Layer l, Tensor x, int n, int heads, int headDim) {
        RopeFreqTable table = ropeTable(l);
        if (table == null) {
            return x;
        }
        String key = "rope:" + l.il() + ":" + n;
        Tensor cos = cachedConstants.get(key + ":cos");
        Tensor sin = cachedConstants.get(key + ":sin");
        Tensor partner = cachedConstants.get(key + ":" + heads + ":partner");
        if (cos == null) {
            float[] cosValues = new float[n * headDim];
            float[] sinValues = new float[n * headDim];
            int span = table.pairSpan();
            float[] cs = new float[2 * table.pairs()];
            for (int t = 0; t < n; t++) {
                table.cosSin(t, cs);
                int row = t * headDim;
                for (int d = 0; d < headDim; d++) {
                    cosValues[row + d] = 1f;
                }
                for (int i = 0; i < table.pairs(); i++) {
                    int a = table.pairFirst(i);
                    int b = a + span;
                    cosValues[row + a] = cs[2 * i];
                    cosValues[row + b] = cs[2 * i];
                    sinValues[row + a] = -cs[2 * i + 1];
                    sinValues[row + b] = cs[2 * i + 1];
                }
            }
            cos = constant(cosValues, n, 1, headDim);
            sin = constant(sinValues, n, 1, headDim);
            cachedConstants.put(key + ":cos", cos);
            cachedConstants.put(key + ":sin", sin);
        }
        if (partner == null) {
            int[] partnerValues = new int[n * heads * headDim];
            int span = table.pairSpan();
            for (int t = 0; t < n; t++) {
                for (int h = 0; h < heads; h++) {
                    int base = (t * heads + h) * headDim;
                    for (int d = 0; d < headDim; d++) {
                        partnerValues[base + d] = d;
                    }
                    for (int i = 0; i < table.pairs(); i++) {
                        int a = table.pairFirst(i);
                        int b = a + span;
                        partnerValues[base + a] = b;
                        partnerValues[base + b] = a;
                    }
                }
            }
            partner = intConstant(partnerValues, n, heads, headDim);
            cachedConstants.put(key + ":" + heads + ":partner", partner);
        }
        Tensor swapped = TensorOps.gather(x, 2, partner);
        return TensorOps.add(TensorOps.mul(x, cos), TensorOps.mul(swapped, sin));
    }

    private RopeFreqTable ropeTable(FinetunePlan.Layer l) {
        if (ropeTables.containsKey(l.il())) {
            return ropeTables.get(l.il());
        }
        RopeSpec spec = l.rope();
        RopeFreqTable table = null;
        if (spec != null && spec.type() != RopeType.NONE) {
            CpuWeights weights = host.source().cpuWeights();
            float[] factors = spec.freqFactors() != null ? weights.floats(spec.freqFactors()) : null;
            table = RopeFreqTable.of(spec, factors);
        }
        ropeTables.put(l.il(), table);
        return table;
    }

    // ------------------------------------------------------------------ feed-forward

    private Tensor denseFfn(FinetunePlan.Layer l, Tensor x) {
        Tensor up = linear(x, l.ffnUp(), l.ffnUpBias());
        Tensor act;
        if (l.glu() != null) {
            Tensor gate = linear(x, l.ffnGate(), l.ffnGateBias());
            act = glu(l.glu(), gate, up);
        } else {
            act = activation(l.act(), up);
        }
        return linear(act, l.ffnDown(), l.ffnDownBias());
    }

    /**
     * The routed FFN. Every expert is evaluated on every token and the outputs are combined by the routing weights,
     * the top-k selection being a constant mask so the softmax's gradient keeps the selected probabilities' path.
     */
    private Tensor moe(FinetunePlan.Moe m, Tensor x, int n, Ctx ctx) {
        int experts = m.nExpert();
        int used = m.nUsed();
        int ff = m.nFfExp();
        Tensor routerLogits = linear(x, m.router(), m.routerBias());
        Tensor probs = TensorOps.softmax(routerLogits, 1);
        float[] hostProbs = probs.toFloatArray();
        int[] selected = select(hostProbs, n, experts, used);
        float[] mask = new float[n * experts];
        for (int s : selected) {
            mask[s] = 1f;
        }
        Tensor selection = ctx.temp(mask, n, experts);
        Tensor weights = TensorOps.mul(probs, selection);
        if (m.normWeights()) {
            Tensor denom = TensorOps.clamp(TensorOps.sum(weights, 1), MoeSpec.NORM_SUM_MIN, Float.MAX_VALUE);
            weights = TensorOps.div(weights, denom.reshape(n, 1));
        }
        float scale = m.weightsScale() == 0f || m.weightsScale() == 1f ? 1f : m.weightsScale();
        weights = TensorOps.mul(weights, scale);
        Tensor gateExps = m.gateExps() != null ? deviceTensor(m.gateExps()).reshape(experts, ff, nEmbd) : null;
        Tensor upExps = deviceTensor(m.upExps()).reshape(experts, ff, nEmbd);
        Tensor downExps = deviceTensor(m.downExps()).reshape(experts, nEmbd, ff);
        Tensor out = null;
        for (int e = 0; e < experts; e++) {
            Tensor up = TensorOps.linear(x, sliceMatrix(upExps, e, ff, nEmbd), null);
            Tensor act;
            if (gateExps != null) {
                Tensor gate = TensorOps.linear(x, sliceMatrix(gateExps, e, ff, nEmbd), null);
                act = glu(m.glu(), gate, up);
            } else {
                act = activation(m.act(), up);
            }
            Tensor y = TensorOps.linear(act, sliceMatrix(downExps, e, nEmbd, ff), null);
            Tensor term = TensorOps.mul(y, weights.slice(1, e, e + 1));
            out = out == null ? term : TensorOps.add(out, term);
        }
        return out;
    }

    /** Expert {@code e}'s {@code rows x cols} matrix out of a stacked expert tensor. */
    private static Tensor sliceMatrix(Tensor stacked, int e, int rows, int cols) {
        return stacked.slice(0, e, e + 1).reshape(rows, cols);
    }

    /** The flat indices of the top-{@code used} probabilities of every token, the lowest index on ties. */
    private static int[] select(float[] probs, int n, int experts, int used) {
        int[] selected = new int[n * used];
        for (int t = 0; t < n; t++) {
            int base = t * experts;
            for (int s = 0; s < used; s++) {
                int best = -1;
                for (int e = 0; e < experts; e++) {
                    boolean taken = false;
                    for (int prior = 0; prior < s; prior++) {
                        if (selected[t * used + prior] - base == e) {
                            taken = true;
                            break;
                        }
                    }
                    if (!taken && (best < 0 || probs[base + e] > probs[base + best])) {
                        best = e;
                    }
                }
                selected[t * used + s] = t * experts + best;
            }
        }
        return selected;
    }

    private Tensor glu(GluOp op, Tensor gate, Tensor up) {
        switch (op) {
            case SWIGLU:
                return TensorOps.mul(TensorOps.silu(gate), up);
            case GEGLU_TANH:
                return TensorOps.mul(TensorOps.gelu(gate), up);
            case REGLU:
                return TensorOps.mul(TensorOps.relu(gate), up);
            default:
                throw new UnsupportedOperationException("the GPU finetune trainer does not run the " + op + " gate");
        }
    }

    private Tensor activation(ActOp op, Tensor x) {
        switch (op) {
            case SILU:
                return TensorOps.silu(x);
            case GELU_TANH:
                return TensorOps.gelu(x);
            case RELU:
                return TensorOps.relu(x);
            default:
                throw new UnsupportedOperationException("the GPU finetune trainer does not run the " + op
                        + " activation");
        }
    }

    // ------------------------------------------------------------------ loss and readings

    /** The mean negative log-likelihood of the label ids ({@code [n, 1]}) under the logits' softmax. */
    private static Tensor lossTensor(Tensor logits, Tensor labelIds) {
        Tensor logp = TensorOps.logSoftmax(logits, 1);
        Tensor picked = TensorOps.gather(logp, 1, labelIds);
        return TensorOps.neg(TensorOps.mean(picked));
    }

    /** The positions whose predicted token is the label. */
    private static int correct(Tensor predicted, int[] labels, int n) {
        int[] tokens = predicted.toIntArray();
        int correct = 0;
        for (int t = 0; t < n; t++) {
            if (tokens[t] == labels[t]) {
                correct++;
            }
        }
        return correct;
    }

    private float[][] toRows(Tensor logits, int n) {
        float[] flat = logits.toFloatArray();
        float[][] rows = new float[n][];
        for (int t = 0; t < n; t++) {
            rows[t] = new float[nVocab];
            System.arraycopy(flat, t * nVocab, rows[t], 0, nVocab);
        }
        return rows;
    }

    // ------------------------------------------------------------------ constants

    /** Builds a constant FLOAT32 tensor ({@code dims} override the flat shape) and tracks it for the call. */
    private Tensor constant(float[] values, int... dims) {
        Tensor flat = Tensor.fromArray(backend, values);
        Tensor shaped = dims.length == 0 ? flat : flat.reshape(dims);
        if (shaped != flat) {
            flat.close();
        }
        return shaped;
    }

    private Tensor intConstant(int[] values, int... dims) {
        Tensor flat = Tensor.fromArray(backend, values);
        Tensor shaped = dims.length == 0 ? flat : flat.reshape(dims);
        if (shaped != flat) {
            flat.close();
        }
        return shaped;
    }

    /** Per-call constants and the lifetime of the values the graph does not own. */
    private final class Ctx {
        private final List<Tensor> temporaries = new ArrayList<>();

        Tensor temp(float[] values, int... dims) {
            Tensor t = constant(values, dims);
            temporaries.add(t);
            return t;
        }

        Tensor ints(int[] values, int... dims) {
            Tensor t = intConstant(values, dims);
            temporaries.add(t);
            return t;
        }

        void closeTransient() {
            for (int i = temporaries.size() - 1; i >= 0; i--) {
                temporaries.get(i).close();
            }
            temporaries.clear();
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (DeviceFinetuneUpdate update : updates) {
            try {
                update.close();
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        updates.clear();
        try {
            parameters.close();
        } catch (RuntimeException e) {
            failure = e;
        }
        for (Tensor t : cachedConstants.values()) {
            try {
                t.close();
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        cachedConstants.clear();
        ropeTables.clear();
        host.close();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "GpuTrainableModel{" + plan.arch() + ", " + nEmbd + " wide, " + parameters.size() + " tensors on "
                + backend.backendType() + "}";
    }
}
