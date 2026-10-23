package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.block.CpuMixer;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.ops.encdec.T5RelativeBias;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.ops.rope.CpuRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntUnaryOperator;

/**
 * CPU twin of {@link GpuAttentionMixer}: the same order of operations (llama.cpp {@code build_qkv} +
 * {@code build_attn}, src/llama-graph.cpp:1807-1912, 2946-3100, 3115-3290) on {@link CpuScratch} arrays with the
 * kernel-mirroring CPU math ({@link NormMath}, {@link CpuRope}, {@link AttnExtMath}, {@link CpuHadamard},
 * {@link CpuAttentionCore}), the shared Q/K norm geometry ({@link QkNormGeometry}) and the MoVA value projection
 * ({@link CpuValueExperts}).
 *
 * <p>Positions come from the {@link CpuScratch#POS} int slot, which the embedding stage fills per step
 * ({@code pos[s * n + t]}, section 0 = the token positions). With a cache, K and V are stored at
 * {@link KvLayout#rowOf} of each position after the executor's {@link CpuMemory#beginStep}, and attention reads the
 * cache rows with their stored positions ({@link CpuMemory#positionOfRow}); a shared-KV layer reads its owner's
 * {@link KvAccessor}. Without a cache, attention reads the batch's K and V with {@code kv_pos = q_pos}.</p>
 *
 * <p>Every projection adds the {@link CpuProjectionDeltas} registered on it (LoRA) right after its base product and
 * before its bias, as {@link GpuAttentionMixer} does.</p>
 *
 * <p>Working arrays are the {@code Q}, {@code K}, {@code V} and {@code ATTN_OUT} slots plus {@code attn.*}
 * temporaries reserved at bind for the ubatch; a step allocates no per-token arrays. Not thread-safe.</p>
 */
public final class CpuAttentionMixer implements CpuMixer {

    private static final String T_QKV = "attn.qkv";
    private static final String T_QGATE = "attn.qgate";
    private static final String T_GATE = "attn.gate";
    private static final String T_SHIFT = "attn.shift";

    private CpuForwardContext ctx;
    private CpuScratch scratch;
    private CpuMatVec matVec;
    private LayerSpec layer;
    private AttentionSpec spec;
    private QkvProjection proj;
    private int ubatch;
    private boolean bound;
    private boolean closed;

    private CpuTensorRef wq;
    private CpuTensorRef wk;
    private CpuTensorRef wv;
    private CpuTensorRef wqkv;
    private CpuTensorRef wgate;
    private CpuTensorRef wo;
    private float[] qkvBias;
    private float[] qBias;
    private float[] kBias;
    private float[] vBias;
    private float[] gateBias;
    private float[] outBias;
    private ActParams clamp;

    private CpuProjectionDeltas.Site qSite;
    private CpuProjectionDeltas.Site kSite;
    private CpuProjectionDeltas.Site vSite;
    private CpuProjectionDeltas.Site qkvSite;
    private CpuProjectionDeltas.Site gateSite;
    private CpuProjectionDeltas.Site outSite;
    private CpuValueExperts valueExperts;

    private NormStep qNorm;
    private NormStep kNorm;
    private NormStep vNorm;
    private NormStep subNorm;
    private NormStep outHeadNorm;
    private RopeFreqTable ropeTable;
    private RopeFreqTable shiftTable;
    private int gateWidth;
    private boolean singlePosition;

    private KvAccessor kv;
    private IntUnaryOperator rowPos;
    private CpuQuantizedKvRotation rotation;
    private int shiftRotK;
    private CpuAttentionCore.Params params;

    /** An unbound mixer; {@link #bind} reads the layer's attention spec. */
    public CpuAttentionMixer() {
    }

    // ================================================================== bind

    /**
     * @throws IllegalArgumentException      when the layer has no attention spec or a tensor has the wrong shape
     * @throws IllegalStateException         when the layer uses a cache but the context has no memory
     * @throws UnsupportedOperationException for a latent cache or value experts on a fused projection
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (bound) {
            throw new IllegalStateException("attention mixer of layer " + this.layer.il() + " is already bound");
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.scratch = ctx.scratch();
        this.matVec = ctx.matVec();
        this.ubatch = scratch.ubatch();
        CpuWeights w = ctx.weights();
        this.proj = QkvProjection.of(layer, s -> w.layerOrNull(layer, s) != null, ctx.spec().arch());
        this.spec = layer.attention();
        if (proj.source() == QkvProjection.Source.NONE) {
            wo = w.layer(layer, proj.outWeight());
            outSite = serveDeltas(proj.outWeight(), wo);
        } else {
            bindAttention(w);
        }
        bound = true;
    }

    private void bindAttention(CpuWeights w) {
        requireSlot(CpuScratch.Q, (long) ubatch * spec.qDim());
        requireSlot(CpuScratch.ATTN_OUT, (long) ubatch * spec.outDim());
        if (proj.projectsKv()) {
            requireSlot(CpuScratch.K, (long) ubatch * spec.kDim());
            requireSlot(CpuScratch.V, (long) ubatch * spec.vDim());
        }
        if ((long) scratch.ints(CpuScratch.POS).length < 4L * ubatch) {
            throw new IllegalStateException("scratch slot " + CpuScratch.POS + " holds fewer than 4 x " + ubatch
                    + " positions");
        }
        singlePosition = layer.rope() == null || layer.rope().positionsPerToken() == 1;
        bindProjections(w);
        bindNorms(w);
        bindRope(w);
        if (spec.usesMemory()) {
            if (!ctx.hasMemory()) {
                throw new IllegalStateException("layer " + layer.il() + " attends through a KV cache but the model "
                        + ctx.spec().arch().ggufName() + " has no memory");
            }
            CpuMemory memory = ctx.memory();
            kv = memory.kv(layer.il());
            if (kv == null) {
                throw new IllegalStateException("layer " + layer.il() + " has no KV cache in the memory plan");
            }
            checkCache(kv.layout());
            final int group = kv.group();
            rowPos = r -> memory.positionOfRow(group, r);
            RotationPlan rot = memory.plan().layer(kv.layer()).rotation();
            rotation = CpuQuantizedKvRotation.of(rot, spec.qDim(), proj.projectsKv() ? spec.kDim() : 0,
                    proj.projectsKv() ? spec.vDim() : 0, spec.outDim());
            shiftRotK = kv.layout().dtypeK().quantized() && rot.rotateK() ? rot.nRotK() : 0;
        }
        params = CpuAttentionCore.Params.of(spec, sinks(w), relBias(w));
        bindOutput(w);
    }

    private void checkCache(KvLayout layout) {
        if (layout.vAliasesK()) {
            throw new UnsupportedOperationException("layer " + layer.il() + " has a latent (MLA) cache; the softmax"
                    + " attention mixer reads separate K and V rows");
        }
        if (layout.widthK() != spec.kDim() || layout.widthV() != spec.vDim()) {
            throw new IllegalArgumentException("layer " + layer.il() + ": cache rows of K " + layout.widthK() + " / V "
                    + layout.widthV() + " do not hold " + spec.nHeadKv() + " heads of (" + spec.headDimK() + ", "
                    + spec.headDimV() + ")");
        }
    }

    private void bindProjections(CpuWeights w) {
        int qw = proj.qProjWidth();
        if (proj.source() == QkvProjection.Source.FUSED) {
            wqkv = w.layer(layer, proj.qkvWeight());
            QkvProjection.checkShape(wqkv.toString(), wqkv.cols(), wqkv.rows(), wqkv.cols(), proj.fusedWidth());
            qkvSite = serveDeltas(proj.qkvWeight(), wqkv);
            qSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_Q), wqkv);
            kSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_K), wqkv);
            vSite = serveDeltas(TensorNames.weight(TensorNames.ATTN_V), wqkv);
            scratch.temp(T_QKV, ubatch * proj.fusedWidth());
            qkvBias = vector(w, proj.qkvBias(), proj.fusedWidth());
        } else {
            wq = w.layer(layer, proj.qWeight());
            QkvProjection.checkShape(proj.qWeight(), wq.cols(), wq.rows(), wq.cols(), qw);
            qSite = serveDeltas(proj.qWeight(), wq);
            if (proj.projectsKv()) {
                wk = w.layer(layer, proj.kWeight());
                QkvProjection.checkShape(proj.kWeight(), wk.cols(), wk.rows(), wq.cols(), spec.kDim());
                kSite = serveDeltas(proj.kWeight(), wk);
                if (proj.valueExperts()) {
                    float[] bias = vector(w, proj.vGateBias(), spec.valueExperts().nExpert());
                    valueExperts = new CpuValueExperts(ctx, layer, proj, spec.valueExperts(), bias);
                    if (valueExperts.inputWidth() != wq.cols()) {
                        throw new IllegalArgumentException("layer " + layer.il() + ": value router "
                                + proj.vGateWeight() + " reads " + valueExperts.inputWidth()
                                + " values, the Q projection " + wq.cols());
                    }
                } else if (!proj.vFromK()) {
                    wv = w.layer(layer, proj.vWeight());
                    QkvProjection.checkShape(proj.vWeight(), wv.cols(), wv.rows(), wq.cols(), spec.vDim());
                    vSite = serveDeltas(proj.vWeight(), wv);
                }
            }
        }
        qBias = vector(w, proj.qBias(), qw);
        if (proj.projectsKv()) {
            kBias = vector(w, proj.kBias(), spec.kDim());
            vBias = vector(w, proj.vBias(), spec.vDim());
        }
        if (spec.clampKqv() > 0f) {
            clamp = ActParams.builder().clamp(-spec.clampKqv(), spec.clampKqv()).build();
        }
        if (proj.interleavedGate()) {
            scratch.temp(T_QGATE, ubatch * qw);
            gateWidth = spec.qDim();
            scratch.temp(T_GATE, ubatch * gateWidth);
        } else if (proj.gateWeight() != null) {
            wgate = w.layer(layer, proj.gateWeight());
            gateWidth = AttnExtMath.gateWidth(spec.gate(), spec.nHead(), spec.headDimV());
            QkvProjection.checkShape(proj.gateWeight(), wgate.cols(), wgate.rows(), wgate.cols(), gateWidth);
            gateSite = serveDeltas(proj.gateWeight(), wgate);
            scratch.temp(T_GATE, ubatch * gateWidth);
        }
    }

    private void bindNorms(CpuWeights w) {
        QkNormKind kind = spec.qkNorm();
        if (kind != QkNormKind.NONE) {
            boolean hasQ = w.layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_Q_NORM)) != null;
            boolean hasK = w.layerOrNull(layer, TensorNames.weight(TensorNames.ATTN_K_NORM)) != null;
            if (QkNormGeometry.applies(kind, ctx.spec().arch(), false, hasQ, hasK)) {
                qNorm = qkNorm(w, geometry(kind, spec.nHead()), TensorNames.ATTN_Q_NORM);
            }
            if (proj.projectsKv() && QkNormGeometry.applies(kind, ctx.spec().arch(), true, hasQ, hasK)) {
                kNorm = qkNorm(w, geometry(kind, spec.nHeadKv()), TensorNames.ATTN_K_NORM);
            }
        }
        if (spec.vNorm() && proj.projectsKv()) {
            vNorm = new NormStep(NormKind.RMS, rmsEps(), spec.nHeadKv(), spec.headDimV(), null, null, 1);
        }
    }

    private QkNormGeometry geometry(QkNormKind kind, int nHeads) {
        return QkNormGeometry.of(kind, spec.headDimK(), nHeads, hparams().normEps(), rmsEps(), layer.il());
    }

    /**
     * Q or K norm in geometry {@code g}, loading the tensors as {@code GpuAttentionMixer.addQkNorm} does: a weight of
     * one scalar per head expands to {@code n_heads} rows, the bias is loaded only when the graph applies it.
     */
    private NormStep qkNorm(CpuWeights w, QkNormGeometry g, String base) {
        int rowLen = g.rowLen();
        CpuTensorRef weightRef = w.layerOrNull(layer, TensorNames.weight(base));
        CpuTensorRef biasRef = spec.qkNormBias() ? w.layerOrNull(layer, TensorNames.bias(base)) : null;
        float[] weight = weightRef != null ? w.layerFloatsOrNull(layer, TensorNames.weight(base)) : null;
        float[] bias = biasRef != null ? w.layerFloatsOrNull(layer, TensorNames.bias(base)) : null;
        int wRows = 1;
        boolean perHeadScalar = weightRef != null && g.perHeadScalar(weightRef.cols(), weightRef.rows());
        if (perHeadScalar) {
            wRows = g.nHeads();
            weight = broadcastPerHead(weight, rowLen);
            if (bias != null && g.perHeadScalar(biasRef.cols(), biasRef.rows())) {
                bias = broadcastPerHead(bias, rowLen);
            }
        } else if (weight != null) {
            wRows = g.weightRows(TensorNames.weight(base), weight.length);
        }
        if (bias != null) {
            g.checkBias(TensorNames.bias(base), bias.length, wRows);
        }
        return new NormStep(g.normKind(), g.eps(), g.rowsPerToken(), rowLen, weight, bias, wRows);
    }

    /** {@code [scalars.length][rowLen]} with row {@code h} filled with {@code scalars[h]} (ggml broadcast multiply). */
    private static float[] broadcastPerHead(float[] scalars, int rowLen) {
        float[] rows = new float[scalars.length * rowLen];
        for (int h = 0; h < scalars.length; h++) {
            Arrays.fill(rows, h * rowLen, (h + 1) * rowLen, scalars[h]);
        }
        return rows;
    }

    private void bindRope(CpuWeights w) {
        RopeSpec rope = layer.rope();
        if (rope == null || spec.noRope() || rope.type() == RopeType.NONE) {
            return;
        }
        float[] ff = rope.freqFactors() != null ? w.floats(rope.freqFactors()) : null;
        ropeTable = RopeFreqTable.of(rope, ff);
        if (spec.usesMemory() && !spec.sharesKv(layer.il()) && GpuContextShift.refusal(ctx.spec(), layer) == null) {
            shiftTable = RopeFreqTable.forShift(rope, ff);
        }
    }

    private void bindOutput(CpuWeights w) {
        if (proj.outHeadNorm()) {
            outHeadNorm = new NormStep(NormKind.RMS, rmsEps(), spec.nHead(), spec.headDimV(), null, null, 1);
        }
        if (spec.gate() != AttnGateKind.NONE) {
            gateBias = vector(w, proj.gateBias(), spec.nHead());
        }
        if (proj.subNormWeight() != null) {
            NormKind kind = layer.norm().kind();
            float eps = kind == NormKind.LAYER ? hparams().normEps() : rmsEps();
            float[] sw = vector(w, proj.subNormWeight(), spec.outDim());
            float[] sb = vector(w, proj.subNormBias(), spec.outDim());
            subNorm = new NormStep(kind, eps, 1, spec.outDim(), sw, sb, 1);
        }
        wo = w.layer(layer, proj.outWeight());
        QkvProjection.checkShape(proj.outWeight(), wo.cols(), wo.rows(), spec.outDim(), wo.rows());
        outSite = serveDeltas(proj.outWeight(), wo);
        outBias = vector(w, proj.outBias(), wo.rows());
    }

    private float[] sinks(CpuWeights w) {
        if (!spec.sinks()) {
            return null;
        }
        return vector(w, TensorNames.weight(TensorNames.ATTN_SINKS), spec.nHead());
    }

    /**
     * The T5 relative-position bias table of the stack's bias layer ({@link T5RelativeBias#host}: F32
     * {@code [n_buckets][n_head]}), or null without a T5 bias.
     */
    private float[] relBias(CpuWeights w) {
        return T5RelativeBias.uses(spec) ? T5RelativeBias.host(w, layer, spec) : null;
    }

    /** A layer vector ({@code suffix}, null for none) decoded to F32 and checked to hold {@code n} values. */
    private float[] vector(CpuWeights w, String suffix, long n) {
        if (suffix == null) {
            return null;
        }
        float[] v = w.layerFloatsOrNull(layer, suffix);
        if (v == null) {
            throw new IllegalStateException("layer " + layer.il() + ": tensor " + suffix + " is missing");
        }
        QkvProjection.checkLength(suffix, v.length, n);
        return v;
    }

    private void requireSlot(String name, long n) {
        if (scratch.slot(name).length < n) {
            throw new IllegalStateException("scratch slot " + name + " holds " + scratch.slot(name).length
                    + " floats, " + n + " are needed");
        }
    }

    private ModelHparams hparams() {
        return ctx.spec().hparams();
    }

    private float rmsEps() {
        return hparams().normRmsEps();
    }

    // ================================================================== steps

    @Override
    public void decode(int pos, float[] in, float[] out) {
        run(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int n, float[] in, float[] out) {
        run(pos0, n, in, out);
    }

    private void run(int pos0, int n, float[] in, float[] out) {
        ensureBound();
        Objects.requireNonNull(in, "in");
        Objects.requireNonNull(out, "out");
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("layer " + layer.il() + ": step of " + n + " tokens outside [1, "
                    + ubatch + "]");
        }
        if (pos0 < 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": negative position " + pos0);
        }
        if (proj.source() == QkvProjection.Source.NONE) {
            matVec.matmul(wo, in, 0, out, 0, n);
            addDeltas(outSite, n, in, (int) wo.cols(), out, 0, (int) wo.rows());
            return;
        }
        float[] q = scratch.slot(CpuScratch.Q);
        float[] k = proj.projectsKv() ? scratch.slot(CpuScratch.K) : null;
        float[] v = proj.projectsKv() ? scratch.slot(CpuScratch.V) : null;
        float[] attn = scratch.slot(CpuScratch.ATTN_OUT);
        float[] gate = gateWidth > 0 ? scratch.temp(T_GATE, ubatch * gateWidth) : null;
        int[] pos = scratch.ints(CpuScratch.POS);
        if (singlePosition && pos[0] != pos0) {
            throw new IllegalStateException("layer " + layer.il() + " runs position " + pos0 + " but the "
                    + CpuScratch.POS + " slot starts at " + pos[0]);
        }
        project(n, in, q, k, v, gate);
        if (!spec.qkNormAfterRope()) {
            normalizeQk(n, q, k);
        }
        if (vNorm != null) {
            vNorm.apply(n, v);
        }
        if (ropeTable != null) {
            RopeSpec rope = layer.rope();
            CpuRope.apply(ropeTable, q, 0, n, spec.nHead(), spec.headDimK(), spec.qDim(), rope.nOffs(), pos, 0, n);
            if (k != null) {
                CpuRope.apply(ropeTable, k, 0, n, spec.nHeadKv(), spec.headDimK(), spec.kDim(), rope.nOffs(), pos, 0,
                        n);
            }
        }
        if (spec.qkNormAfterRope()) {
            normalizeQk(n, q, k);
        }
        if (spec.temperature().enabled()) {
            AttnExtMath.applyTemperature(spec.temperature(), n, spec.nHead(), spec.headDimK(), pos, 0, q, 0,
                    spec.qDim());
        }
        if (rotation != null) {
            rotation.beforeStore(n, q, k, v);
        }
        if (kv != null) {
            if (k != null) {
                store(n, pos, k, v);
            }
            int keys = Math.min(kv.layout().rows(), pos0 + n);
            CpuAttentionCore.attendRows(params, n, q, 0, spec.qDim(), pos, 0, kv, rowPos, 0, keys, attn, 0,
                    spec.outDim());
        } else {
            CpuAttentionCore.DenseKv batch = CpuAttentionCore.DenseKv.of(k, 0, spec.kDim(), spec.headDimK(), v, 0,
                    spec.vDim(), spec.headDimV(), spec.nHeadKv(), spec.headDimK(), spec.headDimV(), pos, 0);
            CpuAttentionCore.attendRows(params, n, q, 0, spec.qDim(), pos, 0, batch, 0, n, attn, 0, spec.outDim());
        }
        if (rotation != null) {
            rotation.afterAttention(n, attn);
        }
        if (outHeadNorm != null) {
            outHeadNorm.apply(n, attn);
        }
        if (gate != null) {
            AttnExtMath.applyOutGate(spec.gate(), n, spec.nHead(), spec.headDimV(), attn, 0, spec.outDim(), gate, 0,
                    gateWidth, gateBias, 0);
        }
        if (subNorm != null) {
            subNorm.apply(n, attn);
        }
        matVec.matmul(wo, attn, 0, out, 0, n);
        addDeltas(outSite, n, attn, spec.outDim(), out, 0, (int) wo.rows());
        if (outBias != null) {
            CpuOps.addRowBias(out, 0, n, (int) wo.rows(), outBias, 0);
        }
        if (spec.valueScale() != 0f) {
            CpuOps.scale(out, 0, n * (int) wo.rows(), spec.valueScale());
        }
    }

    private void project(int n, float[] in, float[] q, float[] k, float[] v, float[] gate) {
        int qw = proj.qProjWidth();
        float[] qProj = proj.interleavedGate() ? scratch.temp(T_QGATE, ubatch * qw) : q;
        if (wqkv != null) {
            int fw = proj.fusedWidth();
            float[] qkv = scratch.temp(T_QKV, ubatch * fw);
            int cols = (int) wqkv.cols();
            matVec.matmul(wqkv, in, 0, qkv, 0, n);
            addDeltas(qkvSite, n, in, cols, qkv, 0, fw);
            addDeltas(qSite, n, in, cols, qkv, proj.qOffset(), fw);
            addDeltas(kSite, n, in, cols, qkv, proj.kOffset(), fw);
            addDeltas(vSite, n, in, cols, qkv, proj.vOffset(), fw);
            if (qkvBias != null) {
                CpuOps.addRowBias(qkv, 0, n, fw, qkvBias, 0);
            }
            for (int t = 0; t < n; t++) {
                int row = t * fw;
                System.arraycopy(qkv, row + proj.qOffset(), qProj, t * qw, qw);
                if (k != null) {
                    System.arraycopy(qkv, row + proj.kOffset(), k, t * spec.kDim(), spec.kDim());
                    System.arraycopy(qkv, row + proj.vOffset(), v, t * spec.vDim(), spec.vDim());
                }
            }
        } else {
            int cols = (int) wq.cols();
            matVec.matmul(wq, in, 0, qProj, 0, n);
            addDeltas(qSite, n, in, cols, qProj, 0, qw);
            if (wk != null) {
                matVec.matmul(wk, in, 0, k, 0, n);
                addDeltas(kSite, n, in, cols, k, 0, spec.kDim());
                if (wv != null) {
                    matVec.matmul(wv, in, 0, v, 0, n);
                    addDeltas(vSite, n, in, cols, v, 0, spec.vDim());
                } else if (valueExperts != null) {
                    valueExperts.run(n, in, v);
                } else {
                    System.arraycopy(k, 0, v, 0, n * spec.kDim());
                }
            }
        }
        if (qBias != null) {
            CpuOps.addRowBias(qProj, 0, n, qw, qBias, 0);
        }
        if (kBias != null) {
            CpuOps.addRowBias(k, 0, n, spec.kDim(), kBias, 0);
        }
        if (vBias != null) {
            CpuOps.addRowBias(v, 0, n, spec.vDim(), vBias, 0);
        }
        if (clamp != null) {
            CpuOps.unary(ActOp.CLAMP, clamp, qProj, 0, qProj, 0, n * qw);
            if (k != null) {
                CpuOps.unary(ActOp.CLAMP, clamp, k, 0, k, 0, n * spec.kDim());
                CpuOps.unary(ActOp.CLAMP, clamp, v, 0, v, 0, n * spec.vDim());
            }
        }
        if (proj.interleavedGate()) {
            int hd = spec.headDimK();
            for (int r = 0; r < n * spec.nHead(); r++) {
                System.arraycopy(qProj, r * 2 * hd, q, r * hd, hd);
                System.arraycopy(qProj, r * 2 * hd + hd, gate, r * hd, hd);
            }
        } else if (wgate != null) {
            matVec.matmul(wgate, in, 0, gate, 0, n);
            addDeltas(gateSite, n, in, (int) wgate.cols(), gate, 0, gateWidth);
        }
    }

    private static void addDeltas(CpuProjectionDeltas.Site site, int n, float[] x, int xStride, float[] y, int yOff,
                                  int yStride) {
        if (site != null && !site.isEmpty()) {
            site.apply(n, x, 0, xStride, y, yOff, yStride);
        }
    }

    /** The delta site of projection {@code weight}; a matvec that rotates its input (PRISM) refuses deltas. */
    private CpuProjectionDeltas.Site serveDeltas(String weight, CpuTensorRef w) {
        String refusal = matVec.transformsOf(w).input() != null ? weight + " of layer " + layer.il() + " rotates its"
                + " input (PRISM Hadamard) inside the matvec; a LoRA delta would need the rotated rows" : null;
        return ctx.projectionDeltas().serve(layer, weight, refusal);
    }

    private void normalizeQk(int n, float[] q, float[] k) {
        if (qNorm != null) {
            qNorm.apply(n, q);
        }
        if (kNorm != null) {
            kNorm.apply(n, k);
        }
    }

    private void store(int n, int[] pos, float[] k, float[] v) {
        KvLayout layout = kv.layout();
        for (int t = 0; t < n; t++) {
            int row = layout.rowOf(pos[t]);
            kv.writeK(row, k, t * spec.kDim());
            kv.writeV(row, v, t * spec.vDim());
        }
    }

    // ================================================================== context shift

    /**
     * Applies the RoPE delta of a context shift to this layer's own K rows: the executor has already moved the rows
     * with {@link HeapModelMemory#shiftPositions}; every row is read, rotated by RoPE at its delta (inside the
     * Hadamard rotation of a rotated block-type cache, as {@code LLM_KV_RopeShift_<KV>} does) and written back
     * (llama.cpp {@code build_rope_shift}, src/llama-kv-cache.cpp:1945-2070). Rows with delta 0 are rotated too, as
     * llama.cpp shifts every cell with its per-cell shift (llama-kv-cache.cpp:1545-1556). Only the shifted dims of
     * each head are stored again: the rotated dims, widened for a block type to the kernel's work window
     * ({@link RopeKernels#shiftWindowStart}), so the other blocks keep their stored bits as in the GPU kernel and in
     * llama.cpp, which requantizes only the {@code n_rot} view of each head (llama-kv-cache.cpp:2058-2066).
     *
     * @throws UnsupportedOperationException for a layer without a cache, a model llama.cpp cannot shift
     *                                       ({@link GpuContextShift#refusal}), or a memory without shift deltas
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureBound();
        if (nKeep < 0 || nDiscard <= 0) {
            throw new IllegalArgumentException("context shift needs nKeep >= 0 and nDiscard > 0, got " + nKeep + "/"
                    + nDiscard);
        }
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        if (proj.source() == QkvProjection.Source.NONE || shiftTable == null || !GpuContextShift.shifts(layer)) {
            return;
        }
        int[] delta = ((HeapModelMemory) ctx.memory()).shiftDelta()[kv.group()];
        shiftRows(delta);
    }

    /**
     * Null when {@link #shiftContext} can run: a layer without its own projections, or a cached layer of a model
     * llama.cpp can shift over a memory that records shift deltas.
     */
    @Override
    public String shiftRefusal() {
        ensureBound();
        if (proj.source() == QkvProjection.Source.NONE) {
            return null;
        }
        if (!spec.usesMemory()) {
            return "layer " + layer.il() + " attends only to the current batch; it has no context to shift";
        }
        String refusal = GpuContextShift.refusal(ctx.spec(), layer);
        if (refusal != null) {
            return refusal;
        }
        CpuMemory memory = ctx.memory();
        if (shiftTable != null && GpuContextShift.shifts(layer) && !(memory instanceof HeapModelMemory)) {
            return "context shift needs HeapModelMemory deltas, the memory is " + memory.getClass().getSimpleName();
        }
        return null;
    }

    private void shiftRows(int[] delta) {
        RopeSpec rope = layer.rope();
        int hd = spec.headDimK();
        int width = spec.kDim();
        int rows = Math.min(delta.length, kv.layout().rows());
        float[] row = scratch.temp(T_SHIFT, width);
        int[] d = scratch.intTemp(T_SHIFT, 1);
        HadamardSpec h = shiftRotK > 1 ? HadamardSpec.rotation(shiftRotK) : null;
        boolean blocks = kv.layout().dtypeK().quantized();
        int w0 = blocks ? RopeKernels.shiftWindowStart(rope.nOffs(), shiftRotK) : rope.nOffs();
        int w1 = blocks ? RopeKernels.shiftWindowEnd(rope.nDims(), rope.nOffs(), shiftRotK)
                : rope.nOffs() + rope.nDims();
        for (int r = 0; r < rows; r++) {
            for (int head = 0; head < spec.nHeadKv(); head++) {
                kv.readK(r, head, row, head * hd);
            }
            if (h != null) {
                windowHadamard(h, row, hd, w0, w1);
            }
            d[0] = delta[r];
            CpuRope.apply(shiftTable, row, 0, 1, spec.nHeadKv(), hd, width, rope.nOffs(), d, 0, 1);
            if (h != null) {
                windowHadamard(h, row, hd, w0, w1);
            }
            for (int head = 0; head < spec.nHeadKv(); head++) {
                kv.writeKSlice(r, head * hd + w0, w1 - w0, row, head * hd + w0);
            }
        }
    }

    /** The self-inverse Hadamard rotation of the {@code [w0, w1)} window of every head, in {@code rotK} chunks. */
    private void windowHadamard(HadamardSpec h, float[] row, int hd, int w0, int w1) {
        for (int head = 0; head < spec.nHeadKv(); head++) {
            CpuHadamard.applyInPlace(h, row, head * hd + w0, 1, w1 - w0);
        }
    }

    // ================================================================== lifecycle

    private void ensureBound() {
        if (closed) {
            throw new IllegalStateException("attention mixer is closed");
        }
        if (!bound) {
            throw new IllegalStateException("attention mixer is not bound");
        }
    }

    /** The layer this mixer is bound to; null before {@link #bind}. */
    public LayerSpec layer() {
        return layer;
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuAttentionMixer{" + (layer == null ? "unbound" : "layer " + layer.il() + ", " + proj) + '}';
    }

    /** Row norm over the dense per-token rows of one projection ({@link NormMath#rowNorm}, in place). */
    private static final class NormStep {
        private final NormKind kind;
        private final float eps;
        private final int rowsPerToken;
        private final int rowLen;
        private final float[] w;
        private final float[] b;
        private final int wRows;

        private NormStep(NormKind kind, float eps, int rowsPerToken, int rowLen, float[] w, float[] b, int wRows) {
            this.kind = kind;
            this.eps = eps;
            this.rowsPerToken = rowsPerToken;
            this.rowLen = rowLen;
            this.w = w;
            this.b = b;
            this.wRows = wRows;
        }

        private void apply(int n, float[] x) {
            NormMath.rowNorm(kind, eps, n * rowsPerToken, rowLen, rowLen, wRows, x, 0, w, b, x, 0);
        }
    }
}
