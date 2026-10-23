package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.activations.GELU;
import com.eonmux.neuralfree.nn.activations.GeGLU;
import com.eonmux.neuralfree.nn.activations.GLU;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.nn.norm.RMSNorm;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Objects;
import java.util.Random;

/**
 * One transformer encoder block: self-attention and a feed-forward network ({@code fc1}, GELU, GLU or GeGLU,
 * {@code fc2}), each wrapped in a residual connection, with layer or RMS normalization before each sub-block
 * ({@code preNorm}) or after each residual sum. In training mode with a positive dropout rate the attention output
 * and the feed-forward hidden activations pass through dropout.
 *
 * <p>The forward pass is a chain of recording sub-modules and {@code TensorOps.add} residuals, so in grad mode the
 * result is a recorded node output and a backward pass from a loss on it reaches every parameter of the block. The
 * intermediates are closed as the chain consumes them; the caller's input never is.</p>
 */
public final class TransformerEncoderLayer extends Module {
    public enum NormType {
        LAYER_NORM,
        RMS_NORM
    }
    
    public enum FFNType {
        MLP,
        GLU,
        GEGLU
    }
    
    private final int dModel;
    private final int numHeads;
    private final int dFF;
    private final float dropoutRate;
    private final boolean preNorm;
    private final NormType normType;
    private final FFNType ffnType;
    private final boolean causal;
    
    private MultiHeadAttention selfAttn;
    private Module norm1;
    private Module norm2;
    private Linear fc1;
    private Linear fc2;
    private Module activation;
    private Dropout attnDropout;
    private Dropout ffnDropout;
    
    private boolean initialized = false;
    
    public TransformerEncoderLayer(int dModel, int numHeads) {
        this(dModel, numHeads, dModel * 4, 0.1f);
    }
    
    public TransformerEncoderLayer(int dModel, int numHeads, int dFF, float dropoutRate) {
        this(dModel, numHeads, dFF, dropoutRate, true, NormType.LAYER_NORM, FFNType.MLP, false);
    }
    
    public TransformerEncoderLayer(int dModel, int numHeads, int dFF, float dropoutRate,
                                    boolean preNorm, NormType normType, FFNType ffnType, boolean causal) {
        if (dModel <= 0) {
            throw new IllegalArgumentException("dModel must be positive: " + dModel);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (dFF <= 0) {
            throw new IllegalArgumentException("dFF must be positive: " + dFF);
        }
        if (dropoutRate < 0.0f || dropoutRate >= 1.0f) {
            throw new IllegalArgumentException("dropoutRate must be in [0.0, 1.0): " + dropoutRate);
        }
        if (ffnType == FFNType.GLU || ffnType == FFNType.GEGLU) {
            if (dFF % 2 != 0) {
                throw new IllegalArgumentException("dFF must be even for GLU/GeGLU: " + dFF);
            }
        }
        
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.dropoutRate = dropoutRate;
        this.preNorm = preNorm;
        this.normType = normType;
        this.ffnType = ffnType;
        this.causal = causal;
    }
    
    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }
    
    /**
     * Creates and initializes the sub-modules on {@code backend} from {@code random}. Runs under
     * {@link ComputationGraph#noGrad()} and records nothing.
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
        this.selfAttn = new MultiHeadAttention(dModel, numHeads, dropoutRate, causal);
        selfAttn.initialize(backend, random);
        registerModule("self_attn", selfAttn);
        
        if (normType == NormType.LAYER_NORM) {
            LayerNorm ln1 = new LayerNorm(dModel, 1e-5f, true);
            ln1.initialize(backend);
            this.norm1 = registerModule("norm1", ln1);
            
            LayerNorm ln2 = new LayerNorm(dModel, 1e-5f, true);
            ln2.initialize(backend);
            this.norm2 = registerModule("norm2", ln2);
        } else {
            RMSNorm rn1 = new RMSNorm(dModel, 1e-8f, true);
            rn1.initialize(backend);
            this.norm1 = registerModule("norm1", rn1);
            
            RMSNorm rn2 = new RMSNorm(dModel, 1e-8f, true);
            rn2.initialize(backend);
            this.norm2 = registerModule("norm2", rn2);
        }
        
        int fc1OutDim;
        switch (ffnType) {
            case GLU:
                fc1OutDim = dFF;
                this.activation = new GLU(-1);
                break;
            case GEGLU:
                fc1OutDim = dFF;
                this.activation = new GeGLU(-1);
                break;
            case MLP:
            default:
                fc1OutDim = dFF;
                this.activation = new GELU();
                break;
        }
        
        this.fc1 = new Linear(dModel, fc1OutDim);
        fc1.initialize(backend, random);
        registerModule("fc1", fc1);
        
        int fc2InDim = (ffnType == FFNType.GLU || ffnType == FFNType.GEGLU) ? dFF / 2 : dFF;
        this.fc2 = new Linear(fc2InDim, dModel);
        fc2.initialize(backend, random);
        registerModule("fc2", fc2);
        
        if (dropoutRate > 0.0f) {
            this.attnDropout = new Dropout(dropoutRate);
            this.ffnDropout = new Dropout(dropoutRate);
            registerModule("attn_dropout", attnDropout);
            registerModule("ffn_dropout", ffnDropout);
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null);
    }
    
    /**
     * The block applied to {@code input} ({@code [seq, dModel]} or {@code [batch, seq, dModel]}) with the optional
     * additive attention {@code mask}: {@code x + A(N1(x))} then {@code y + F(N2(y))} with pre-norm, or
     * {@code N1(x + A(x))} then {@code N2(y + F(y))} with post-norm. A new tensor; {@code input} is not closed.
     */
    public Tensor forward(Tensor input, Tensor mask) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape shape = input.shape();
        int rank = shape.rank();
        
        if (rank != 2 && rank != 3) {
            throw new IllegalArgumentException("Input must be 2D [seq_len, d_model] or 3D [batch, seq_len, d_model], got rank " + rank);
        }
        
        if (preNorm) {
            Tensor attended;
            try (Tensor normalized = norm1.forward(input)) {
                attended = TransformerSteps.residual(input, attentionBranch(normalized, mask));
            }
            return TransformerSteps.then(attended, null, x -> {
                try (Tensor normalized = norm2.forward(x)) {
                    return TransformerSteps.residual(x, applyFFN(normalized));
                }
            });
        }
        Tensor attended;
        try (Tensor sum = TransformerSteps.residual(input, attentionBranch(input, mask))) {
            attended = norm1.forward(sum);
        }
        return TransformerSteps.then(attended, null, x -> {
            try (Tensor sum = TransformerSteps.residual(x, applyFFN(x))) {
                return norm2.forward(sum);
            }
        });
    }
    
    /** The self-attention of {@code h} with its dropout in training mode: a new tensor. */
    private Tensor attentionBranch(Tensor h, Tensor mask) {
        Tensor attnOut = selfAttn.forward(h, h, h, mask);
        return attnDropout == null ? attnOut : TransformerSteps.then(attnOut, null, attnDropout::forward);
    }
    
    /** {@code fc2(dropout(activation(fc1(x))))}: a new tensor; {@code x} is not closed. */
    private Tensor applyFFN(Tensor x) {
        Tensor h = fc1.forward(x);
        h = TransformerSteps.then(h, x, activation::forward);
        if (ffnDropout != null) {
            h = TransformerSteps.then(h, x, ffnDropout::forward);
        }
        return TransformerSteps.then(h, x, fc2::forward);
    }
    
    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("TransformerEncoderLayer not initialized. Call initialize(backend) first.");
        }
    }
    
    public int dModel() {
        return dModel;
    }
    
    public int numHeads() {
        return numHeads;
    }
    
    public int dFF() {
        return dFF;
    }
    
    public float dropoutRate() {
        return dropoutRate;
    }
    
    public boolean isPreNorm() {
        return preNorm;
    }
    
    public NormType normType() {
        return normType;
    }
    
    public FFNType ffnType() {
        return ffnType;
    }
    
    public boolean isCausal() {
        return causal;
    }
    
    public MultiHeadAttention selfAttention() {
        ensureInitialized();
        return selfAttn;
    }
    
    @Override
    public String toString() {
        return "TransformerEncoderLayer{" +
            "dModel=" + dModel +
            ", numHeads=" + numHeads +
            ", dFF=" + dFF +
            ", dropoutRate=" + dropoutRate +
            ", preNorm=" + preNorm +
            ", normType=" + normType +
            ", ffnType=" + ffnType +
            ", causal=" + causal +
            ", initialized=" + initialized +
            '}';
    }
}
