package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;
import java.util.Random;

/**
 * Multi-head attention with learned query, key, value and output projections ({@code dModel -> dModel}), computed as a
 * composition of {@link TensorOps} calls on the inputs' backend (see {@link #forward(Tensor, Tensor, Tensor, Tensor)}).
 */
public final class MultiHeadAttention extends Module {
    private final int dModel;
    private final int numHeads;
    private final int headDim;
    private final float dropoutRate;
    private final boolean causal;
    private final boolean batchFirst;
    
    private Linear qProj;
    private Linear kProj;
    private Linear vProj;
    private Linear outProj;
    private Dropout attnDropout;
    
    private boolean initialized = false;
    
    public MultiHeadAttention(int dModel, int numHeads) {
        this(dModel, numHeads, 0.0f, false, true);
    }
    
    public MultiHeadAttention(int dModel, int numHeads, float dropoutRate, boolean causal) {
        this(dModel, numHeads, dropoutRate, causal, true);
    }
    
    public MultiHeadAttention(int dModel, int numHeads, float dropoutRate, boolean causal, boolean batchFirst) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException("dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ")");
        }
        if (dropoutRate < 0.0f || dropoutRate >= 1.0f) {
            throw new IllegalArgumentException("dropoutRate must be in [0.0, 1.0): " + dropoutRate);
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        this.dropoutRate = dropoutRate;
        this.causal = causal;
        this.batchFirst = batchFirst;
    }
    
    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }
    
    /**
     * Creates the four Xavier-initialized projections on {@code backend} from {@code random}, and the attention
     * dropout when the rate is positive. Runs under {@link ComputationGraph#noGrad()} and records nothing.
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            createModules(backend, random);
        }
        initialized = true;
    }
    
    private void createModules(ComputeBackend backend, Random random) {
        this.qProj = new Linear(dModel, dModel);
        this.kProj = new Linear(dModel, dModel);
        this.vProj = new Linear(dModel, dModel);
        this.outProj = new Linear(dModel, dModel);
        
        qProj.initializeXavier(backend, random);
        kProj.initializeXavier(backend, random);
        vProj.initializeXavier(backend, random);
        outProj.initializeXavier(backend, random);
        
        registerModule("q_proj", qProj);
        registerModule("k_proj", kProj);
        registerModule("v_proj", vProj);
        registerModule("out_proj", outProj);
        
        if (dropoutRate > 0.0f) {
            this.attnDropout = new Dropout(dropoutRate);
            registerModule("attn_dropout", attnDropout);
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        return forward(input, input, input, null);
    }
    
    public Tensor forward(Tensor query, Tensor key, Tensor value) {
        return forward(query, key, value, null);
    }
    
    /**
     * Attention of {@code query} over {@code key} and {@code value}, each {@code [batch, seq, dModel]} when batch-first,
     * {@code [seq, batch, dModel]} otherwise, or {@code [seq, dModel]} for a batch of one. The result has the query's
     * layout with a batch dim ({@code [batch, seq, dModel]} or {@code [seq, batch, dModel]}).
     *
     * <p>Heads are split by reshape, permute and {@code contiguous}, attended by the {@link FlashAttentionOp}
     * composition (scaled {@code bmm} scores, the optional additive {@code mask}, causal {@code -inf} masking at key
     * positions after the query position, {@link TensorOps#maskedSoftmax} so masked weights are exactly 0, then
     * {@code bmm} with the values), and merged back by permute and {@code contiguous}. In training mode with a
     * positive dropout rate the attention weights pass through this module's dropout before the value product.
     * Everything stays on the tensors' backend.</p>
     *
     * <p>Every step records (the projections, the reshape and permute views, {@code contiguous}, {@code bmm},
     * {@link TensorOps#maskedSoftmax} and the dropout module), so in grad mode the result is a recorded node output and
     * a backward pass from a loss on it delivers gradients to the four projections' weights and biases, to the inputs
     * that require grad and to a mask that requires grad. The intermediates are closed here; the gradient functions
     * keep their own references to what they saved.</p>
     *
     * @param mask null, or a FLOAT32 additive bias that broadcasts to {@code [batch, numHeads, seq, kvSeq]};
     *             {@code -inf} entries exclude a key (a padding mask), finite entries bias the scores
     * @throws IllegalArgumentException when a tensor's element count does not match its batch, sequence and
     *                                  {@code dModel}, or the mask is invalid
     */
    public Tensor forward(Tensor query, Tensor key, Tensor value, Tensor mask) {
        ensureInitialized();
        Objects.requireNonNull(query, "Query cannot be null");
        Objects.requireNonNull(key, "Key cannot be null");
        Objects.requireNonNull(value, "Value cannot be null");
        
        Shape qShape = query.shape();
        if (qShape.rank() < 2 || qShape.rank() > 4) {
            throw new IllegalArgumentException("Query must be 2D, 3D or 4D, got rank " + qShape.rank());
        }
        
        int batchSize, seqLen;
        
        if (qShape.rank() == 2) {
            batchSize = 1;
            seqLen = qShape.dim(0);
        } else if (batchFirst) {
            batchSize = qShape.dim(0);
            seqLen = qShape.dim(1);
        } else {
            batchSize = qShape.dim(qShape.rank() - 2);
            seqLen = qShape.dim(0);
        }
        
        int kvSeqLen;
        Shape kShape = key.shape();
        if (kShape.rank() == 2) {
            kvSeqLen = kShape.dim(0);
        } else if (batchFirst) {
            kvSeqLen = kShape.dim(1);
        } else {
            kvSeqLen = kShape.dim(0);
        }
        
        requireTokens("Query", query, batchSize, seqLen);
        requireTokens("Key", key, batchSize, kvSeqLen);
        requireTokens("Value", value, batchSize, kvSeqLen);
        
        float scale = 1.0f / (float) Math.sqrt(headDim);
        
        try (Tensor q = splitHeads(qProj, query, batchSize, seqLen);
             Tensor k = splitHeads(kProj, key, batchSize, kvSeqLen);
             Tensor v = splitHeads(vProj, value, batchSize, kvSeqLen);
             Tensor weights = attentionWeights(q, k, scale, mask);
             Tensor context = FlashAttentionOp.weightedValues(weights, v);
             Tensor merged = mergeHeads(context, batchSize, seqLen)) {
            return outProj.forward(merged);
        }
    }
    
    /** {@code x} projected by {@code proj} and split into heads: a new contiguous {@code [batch, heads, seq, headDim]}. */
    private Tensor splitHeads(Linear proj, Tensor x, int batchSize, int seqLen) {
        Shape split = batchFirst ? Shape.of(batchSize, seqLen, numHeads, headDim)
            : Shape.of(seqLen, batchSize, numHeads, headDim);
        int[] toHeads = batchFirst ? new int[] {0, 2, 1, 3} : new int[] {1, 2, 0, 3};
        try (Tensor projected = proj.forward(x);
             Tensor tokens = projected.reshape(split);
             Tensor heads = tokens.permute(toHeads)) {
            return FlashAttentionOp.contiguousOwned(heads);
        }
    }
    
    /** {@code [batch, heads, seq, headDim]} merged into {@code [batch, seq, dModel]} or {@code [seq, batch, dModel]}. */
    private Tensor mergeHeads(Tensor context, int batchSize, int seqLen) {
        int[] toTokens = batchFirst ? new int[] {0, 2, 1, 3} : new int[] {2, 0, 1, 3};
        Shape merged = batchFirst ? Shape.of(batchSize, seqLen, dModel) : Shape.of(seqLen, batchSize, dModel);
        try (Tensor tokens = context.permute(toTokens);
             Tensor dense = FlashAttentionOp.contiguousOwned(tokens)) {
            return dense.reshape(merged);
        }
    }
    
    /** Masked softmax of the scaled, masked scores, with attention dropout applied in training mode. */
    private Tensor attentionWeights(Tensor q, Tensor k, float scale, Tensor mask) {
        Tensor weights;
        try (Tensor scores = FlashAttentionOp.maskedScores(q, k, scale, causal, mask)) {
            weights = TensorOps.maskedSoftmax(scores, 3);
        }
        if (attnDropout == null || !isTraining()) {
            return weights;
        }
        return TransformerSteps.then(weights, null, attnDropout::forward);
    }
    
    private void requireTokens(String role, Tensor t, int batchSize, int seqLen) {
        long expected = (long) batchSize * seqLen * dModel;
        if (seqLen <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException(role + " batch and sequence length must be positive, got shape "
                + t.shape());
        }
        if (t.numElements() != expected) {
            throw new IllegalArgumentException(role + " shape " + t.shape() + " does not hold batch " + batchSize
                + " x sequence " + seqLen + " x dModel " + dModel + " elements");
        }
    }
    
    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("MultiHeadAttention not initialized. Call initialize(backend) first.");
        }
    }
    
    public int dModel() {
        return dModel;
    }
    
    public int numHeads() {
        return numHeads;
    }
    
    public int headDim() {
        return headDim;
    }
    
    public float dropoutRate() {
        return dropoutRate;
    }
    
    public boolean isCausal() {
        return causal;
    }
    
    public Linear qProj() {
        ensureInitialized();
        return qProj;
    }
    
    public Linear kProj() {
        ensureInitialized();
        return kProj;
    }
    
    public Linear vProj() {
        ensureInitialized();
        return vProj;
    }
    
    public Linear outProj() {
        ensureInitialized();
        return outProj;
    }
    
    @Override
    public String toString() {
        return "MultiHeadAttention{" +
            "dModel=" + dModel +
            ", numHeads=" + numHeads +
            ", headDim=" + headDim +
            ", dropoutRate=" + dropoutRate +
            ", causal=" + causal +
            ", initialized=" + initialized +
            '}';
    }
}
