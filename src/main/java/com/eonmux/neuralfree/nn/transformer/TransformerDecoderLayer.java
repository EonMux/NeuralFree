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
 * One transformer decoder block: causal self-attention, cross-attention over an encoder {@code memory} (skipped when
 * the memory is null) and a feed-forward network ({@code fc1}, GELU, GLU or GeGLU, {@code fc2}), each wrapped in a
 * residual connection, with layer or RMS normalization before each sub-block ({@code preNorm}) or after each residual
 * sum. In training mode with a positive dropout rate the attention outputs and the feed-forward hidden activations
 * pass through dropout.
 *
 * <p>The forward pass is a chain of recording sub-modules and {@code TensorOps.add} residuals, so in grad mode the
 * result is a recorded node output and a backward pass from a loss on it reaches every parameter of the block, and
 * the memory when it requires grad. The intermediates are closed as the chain consumes them; the caller's input and
 * memory never are.</p>
 */
public final class TransformerDecoderLayer extends Module {
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
    
    private MultiHeadAttention selfAttn;
    private MultiHeadAttention crossAttn;
    private Module norm1;
    private Module norm2;
    private Module norm3;
    private Linear fc1;
    private Linear fc2;
    private Module activation;
    private Dropout attnDropout;
    private Dropout crossDropout;
    private Dropout ffnDropout;
    
    private boolean initialized = false;
    
    public TransformerDecoderLayer(int dModel, int numHeads) {
        this(dModel, numHeads, dModel * 4, 0.1f);
    }
    
    public TransformerDecoderLayer(int dModel, int numHeads, int dFF, float dropoutRate) {
        this(dModel, numHeads, dFF, dropoutRate, true, NormType.LAYER_NORM, FFNType.MLP);
    }
    
    public TransformerDecoderLayer(int dModel, int numHeads, int dFF, float dropoutRate,
                                    boolean preNorm, NormType normType, FFNType ffnType) {
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
        this.selfAttn = new MultiHeadAttention(dModel, numHeads, dropoutRate, true);
        selfAttn.initialize(backend, random);
        registerModule("self_attn", selfAttn);
        
        this.crossAttn = new MultiHeadAttention(dModel, numHeads, dropoutRate, false);
        crossAttn.initialize(backend, random);
        registerModule("cross_attn", crossAttn);
        
        if (normType == NormType.LAYER_NORM) {
            LayerNorm ln1 = new LayerNorm(dModel, 1e-5f, true);
            ln1.initialize(backend);
            this.norm1 = registerModule("norm1", ln1);
            
            LayerNorm ln2 = new LayerNorm(dModel, 1e-5f, true);
            ln2.initialize(backend);
            this.norm2 = registerModule("norm2", ln2);
            
            LayerNorm ln3 = new LayerNorm(dModel, 1e-5f, true);
            ln3.initialize(backend);
            this.norm3 = registerModule("norm3", ln3);
        } else {
            RMSNorm rn1 = new RMSNorm(dModel, 1e-8f, true);
            rn1.initialize(backend);
            this.norm1 = registerModule("norm1", rn1);
            
            RMSNorm rn2 = new RMSNorm(dModel, 1e-8f, true);
            rn2.initialize(backend);
            this.norm2 = registerModule("norm2", rn2);
            
            RMSNorm rn3 = new RMSNorm(dModel, 1e-8f, true);
            rn3.initialize(backend);
            this.norm3 = registerModule("norm3", rn3);
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
            this.crossDropout = new Dropout(dropoutRate);
            this.ffnDropout = new Dropout(dropoutRate);
            registerModule("attn_dropout", attnDropout);
            registerModule("cross_dropout", crossDropout);
            registerModule("ffn_dropout", ffnDropout);
        }
    }
    
    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null, null, null, null);
    }
    
    public Tensor forward(Tensor input, Tensor memory) {
        return forward(input, memory, null, null, null);
    }
    
    public Tensor forward(Tensor input, Tensor memory, Tensor selfMask, Tensor memoryMask) {
        return forward(input, memory, selfMask, memoryMask, null);
    }
    
    /**
     * The block applied to {@code input} ({@code [seq, dModel]} or {@code [batch, seq, dModel]}): with pre-norm
     * {@code x + S(N1(x))}, then {@code y + C(N2(y), memory)} when {@code memory} is given, then {@code z + F(N3(z))};
     * with post-norm {@code N1(x + S(x))}, {@code N2(y + C(y, memory))}, {@code N3(z + F(z))}. {@code selfMask} and
     * {@code memoryMask} are optional additive attention masks; {@code position} is accepted for API compatibility and
     * not used. A new tensor; {@code input} and {@code memory} are not closed.
     */
    public Tensor forward(Tensor input, Tensor memory, Tensor selfMask, Tensor memoryMask, Tensor position) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape shape = input.shape();
        int rank = shape.rank();
        
        if (rank != 2 && rank != 3) {
            throw new IllegalArgumentException("Input must be 2D [seq_len, d_model] or 3D [batch, seq_len, d_model], got rank " + rank);
        }
        
        Tensor x = preNorm ? preNormSelfAttention(input, selfMask) : postNormSelfAttention(input, selfMask);
        if (memory != null) {
            x = TransformerSteps.then(x, null, y -> preNorm
                ? preNormCrossAttention(y, memory, memoryMask) : postNormCrossAttention(y, memory, memoryMask));
        }
        return TransformerSteps.then(x, null, preNorm ? this::preNormFeedForward : this::postNormFeedForward);
    }
    
    private Tensor preNormSelfAttention(Tensor x, Tensor selfMask) {
        try (Tensor normalized = norm1.forward(x)) {
            return TransformerSteps.residual(x, selfAttentionBranch(normalized, selfMask));
        }
    }
    
    private Tensor postNormSelfAttention(Tensor x, Tensor selfMask) {
        try (Tensor sum = TransformerSteps.residual(x, selfAttentionBranch(x, selfMask))) {
            return norm1.forward(sum);
        }
    }
    
    private Tensor preNormCrossAttention(Tensor x, Tensor memory, Tensor memoryMask) {
        try (Tensor normalized = norm2.forward(x)) {
            return TransformerSteps.residual(x, crossAttentionBranch(normalized, memory, memoryMask));
        }
    }
    
    private Tensor postNormCrossAttention(Tensor x, Tensor memory, Tensor memoryMask) {
        try (Tensor sum = TransformerSteps.residual(x, crossAttentionBranch(x, memory, memoryMask))) {
            return norm2.forward(sum);
        }
    }
    
    private Tensor preNormFeedForward(Tensor x) {
        try (Tensor normalized = norm3.forward(x)) {
            return TransformerSteps.residual(x, applyFFN(normalized));
        }
    }
    
    private Tensor postNormFeedForward(Tensor x) {
        try (Tensor sum = TransformerSteps.residual(x, applyFFN(x))) {
            return norm3.forward(sum);
        }
    }
    
    /** The causal self-attention of {@code h} with its dropout in training mode: a new tensor. */
    private Tensor selfAttentionBranch(Tensor h, Tensor selfMask) {
        Tensor out = selfAttn.forward(h, h, h, selfMask);
        return attnDropout == null ? out : TransformerSteps.then(out, null, attnDropout::forward);
    }
    
    /** The attention of {@code h} over {@code memory} with its dropout in training mode: a new tensor. */
    private Tensor crossAttentionBranch(Tensor h, Tensor memory, Tensor memoryMask) {
        Tensor out = crossAttn.forward(h, memory, memory, memoryMask);
        return crossDropout == null ? out : TransformerSteps.then(out, null, crossDropout::forward);
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
            throw new IllegalStateException("TransformerDecoderLayer not initialized. Call initialize(backend) first.");
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
    
    public MultiHeadAttention selfAttention() {
        ensureInitialized();
        return selfAttn;
    }
    
    public MultiHeadAttention crossAttention() {
        ensureInitialized();
        return crossAttn;
    }
    
    @Override
    public String toString() {
        return "TransformerDecoderLayer{" +
            "dModel=" + dModel +
            ", numHeads=" + numHeads +
            ", dFF=" + dFF +
            ", dropoutRate=" + dropoutRate +
            ", preNorm=" + preNorm +
            ", normType=" + normType +
            ", ffnType=" + ffnType +
            ", initialized=" + initialized +
            '}';
    }
}
