package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Arrays;
import java.util.Random;

/**
 * Gated multi-head cross-attention between a query sequence and a key/value sequence (vision-language fusion).
 * Every step is a {@link TensorOps} composition ({@link ScaledDotProductAttention}), so the fusion runs on the device
 * when the train kernels are available on the inputs' backend; intermediates are released before returning.
 *
 * <p>Keys are layer-normed by {@code keyValueNorm}; values by the same norm when {@code valueDim == keyDim}, else by
 * their own {@code valueNorm}. The gate is {@code Linear(queryDim + hiddenDim, hiddenDim)} over the original query
 * concatenated with the projected attention output; its sigmoid scales that output.</p>
 *
 * <p><b>Attention-output dropout</b> (the dropout PyTorch's {@code TransformerDecoderLayer} applies to its
 * cross-attention block output): in training mode the projected attention output passes through the registered
 * {@link Dropout} child {@value #ATTENTION_DROPOUT} with rate {@code dropout} before the gate and the output norm. Its
 * keep decisions are host draws, one per element in row-major order, through {@code CheckpointFunction.replayable};
 * {@link #initialize} seeds them from its {@code Random} after every weight when {@code dropout > 0}, so equal seeds
 * give equal masks on every backend. In evaluation mode, or with {@code dropout == 0}, the output passes unchanged and
 * the fusion is identical to that of a module without dropout. A training-mode dropout records {@code MaskScaleGrad},
 * so a dropped element passes no gradient.</p>
 */
public class CrossAttentionFusion extends Module {

    /** Registered name of the attention-output dropout child module. */
    public static final String ATTENTION_DROPOUT = "attentionDropout";

    private final int queryDim;
    private final int keyDim;
    private final int valueDim;
    private final int hiddenDim;
    private final int numHeads;
    private final float dropout;
    private final boolean gated;
    private final Dropout attentionDropout;

    private Linear qProj;
    private Linear kProj;
    private Linear vProj;
    private Linear outProj;
    private Linear gate;
    private LayerNorm queryNorm;
    private LayerNorm keyValueNorm;
    private LayerNorm valueNorm;
    private LayerNorm outputNorm;

    private boolean initialized = false;

    public CrossAttentionFusion(int queryDim, int keyDim, int valueDim, int hiddenDim, int numHeads, float dropout, boolean gated) {
        if (queryDim <= 0) throw new IllegalArgumentException("queryDim must be positive");
        if (keyDim <= 0) throw new IllegalArgumentException("keyDim must be positive");
        if (valueDim <= 0) throw new IllegalArgumentException("valueDim must be positive");
        if (hiddenDim <= 0) throw new IllegalArgumentException("hiddenDim must be positive");
        if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be positive");
        if (!(dropout >= 0 && dropout < 1)) throw new IllegalArgumentException("dropout must be in [0, 1): " + dropout);
        ScaledDotProductAttention.requireDivisible(hiddenDim, numHeads);

        this.queryDim = queryDim;
        this.keyDim = keyDim;
        this.valueDim = valueDim;
        this.hiddenDim = hiddenDim;
        this.numHeads = numHeads;
        this.dropout = dropout;
        this.gated = gated;
        this.attentionDropout = registerModule(ATTENTION_DROPOUT, new Dropout(dropout));
    }

    public CrossAttentionFusion(int queryDim, int keyDim, int valueDim, int hiddenDim, int numHeads) {
        this(queryDim, keyDim, valueDim, hiddenDim, numHeads, 0.1f, true);
    }

    /**
     * Creates the parameters on {@code backend}, drawing every weight from {@code random} (a new one when null), then,
     * when {@code dropout > 0}, the seed of the attention-output dropout masks.
     */
    public void initialize(ComputeBackend backend, Random random) {
        if (initialized) return;

        Random r = random != null ? random : new Random();

        qProj = registerModule("qProj", linear(backend, queryDim, hiddenDim, r));
        kProj = registerModule("kProj", linear(backend, keyDim, hiddenDim, r));
        vProj = registerModule("vProj", linear(backend, valueDim, hiddenDim, r));
        outProj = registerModule("outProj", linear(backend, hiddenDim, hiddenDim, r));

        if (gated) {
            gate = registerModule("gate", linear(backend, queryDim + hiddenDim, hiddenDim, r));
        }

        queryNorm = new LayerNorm(new int[]{queryDim});
        keyValueNorm = new LayerNorm(new int[]{keyDim});
        outputNorm = new LayerNorm(new int[]{hiddenDim});

        queryNorm.initialize(backend);
        keyValueNorm.initialize(backend);
        outputNorm.initialize(backend);

        registerModule("queryNorm", queryNorm);
        registerModule("keyValueNorm", keyValueNorm);
        registerModule("outputNorm", outputNorm);

        if (valueDim != keyDim) {
            valueNorm = new LayerNorm(new int[]{valueDim});
            valueNorm.initialize(backend);
            registerModule("valueNorm", valueNorm);
        }

        if (dropout > 0) {
            attentionDropout.setSeed(r.nextLong());
        }
        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("CrossAttentionFusion requires query, key, and value tensors. Use fuse(query, keyValue) method.");
    }

    public Tensor fuse(Tensor query, Tensor keyValue) {
        return forward(query, keyValue, keyValue, null);
    }

    public Tensor forward(Tensor query, Tensor key, Tensor value) {
        return forward(query, key, value, null);
    }

    /**
     * Fuses {@code [batch, queryLen, queryDim]} queries with {@code [batch, keyLen, keyDim]} keys and
     * {@code [batch, keyLen, valueDim]} values ({@code value} may be {@code key} itself when the widths agree):
     * layer-normed projections, multi-head attention whose masked keys get exactly 0 weight (a fully masked row
     * attends to nothing), the output projection, the attention-output dropout in training mode, the optional gate
     * and the output norm. Returns a new
     * {@code [batch, queryLen, hiddenDim]} tensor.
     *
     * @param attentionMask null, or FLOAT32 key flags in row-major {@code [batch, keyLen]} order; 0 masks the key
     * @throws IllegalArgumentException when a shape does not match
     */
    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor attentionMask) {
        if (!initialized) {
            throw new IllegalStateException("CrossAttentionFusion not initialized. Call initialize() first.");
        }
        if (query == null || key == null || value == null) {
            throw new IllegalArgumentException("query, key, and value tensors cannot be null");
        }
        if (query.shape().rank() != 3) {
            throw new IllegalArgumentException("query must be [batch, queryLen, queryDim], got " + query.shape());
        }

        Tensor attentionOutput;
        try (Tensor normalizedQuery = queryNorm.forward(query);
             Tensor normalizedKey = keyValueNorm.forward(key);
             Tensor normalizedValue = value == key ? null : normalizeValue(value);
             Tensor q = qProj.forward(normalizedQuery);
             Tensor k = kProj.forward(normalizedKey);
             Tensor v = vProj.forward(normalizedValue != null ? normalizedValue : normalizedKey);
             Tensor attended = ScaledDotProductAttention.attend(q, k, v, numHeads, attentionMask, true)) {
            attentionOutput = dropAttentionOutput(outProj.forward(attended));
        }
        try (Tensor fused = attentionOutput;
             Tensor gatedOutput = gated ? applyGate(query, fused) : null) {
            return outputNorm.forward(gatedOutput != null ? gatedOutput : fused);
        }
    }

    /**
     * {@code projected} after the attention-output dropout. Takes ownership of {@code projected}: returns it itself
     * when the dropout does not apply, otherwise the dropout's new tensor, closing {@code projected}.
     */
    private Tensor dropAttentionOutput(Tensor projected) {
        Tensor dropped;
        try {
            dropped = attentionDropout.forward(projected);
        } catch (RuntimeException e) {
            projected.close();
            throw e;
        }
        if (dropped != projected) {
            projected.close();
        }
        return dropped;
    }

    /** {@code value} layer-normed by its own norm, or by the key norm when the widths agree. */
    private Tensor normalizeValue(Tensor value) {
        return (valueNorm != null ? valueNorm : keyValueNorm).forward(value);
    }

    /** {@code sigmoid(gate([query, fused])) * fused}, with the query and fused features concatenated per position. */
    private Tensor applyGate(Tensor query, Tensor fused) {
        try (Tensor gateInput = TensorOps.cat(Arrays.asList(query, fused), 2);
             Tensor gateLogits = gate.forward(gateInput);
             Tensor gateValues = TensorOps.sigmoid(gateLogits)) {
            return TensorOps.mul(gateValues, fused);
        }
    }

    private static Linear linear(ComputeBackend backend, int in, int out, Random random) {
        Linear layer = new Linear(in, out);
        layer.initialize(backend, random);
        return layer;
    }

    public int queryDim() { return queryDim; }
    public int keyDim() { return keyDim; }
    public int valueDim() { return valueDim; }
    public int hiddenDim() { return hiddenDim; }
    public int numHeads() { return numHeads; }
    public float dropout() { return dropout; }
    public boolean gated() { return gated; }
    public boolean isInitialized() { return initialized; }

    @Override
    public String toString() {
        return String.format("CrossAttentionFusion(queryDim=%d, keyDim=%d, valueDim=%d, hiddenDim=%d, numHeads=%d, dropout=%.2f, gated=%s)",
                queryDim, keyDim, valueDim, hiddenDim, numHeads, dropout, gated);
    }
}
