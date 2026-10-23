package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.nn.norm.RMSNorm;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * A stack of {@link TransformerDecoderLayer}s, with an optional position encoding applied to the input first and, with
 * pre-norm, a final normalization. The forward pass chains recording modules, so in grad mode a backward pass from a
 * loss on the result delivers gradients to every layer's parameters, to a {@link LearnedPE} table used as the
 * position encoding, which is a registered child ({@link #setPositionalEncoding}), and to the memory when it requires
 * grad. Each intermediate is closed once the next step has consumed it; the caller's input and memory never are.
 */
public final class TransformerDecoder extends Module {
    private final int numLayers;
    private final int dModel;
    private final int numHeads;
    private final int dFF;
    private final float dropoutRate;
    private final boolean preNorm;
    private TransformerDecoderLayer.NormType normType;
    private TransformerDecoderLayer.FFNType ffnType;
    
    private List<TransformerDecoderLayer> layers;
    private Module finalNorm;
    private PositionalEncoding posEncoding;

    /** Child name of a position encoding that is a module ({@link #setPositionalEncoding}). */
    private static final String POSITIONAL_ENCODING = "positional_encoding";
    
    private boolean initialized = false;
    
    public TransformerDecoder(int numLayers, int dModel, int numHeads) {
        this(numLayers, dModel, numHeads, dModel * 4, 0.1f);
    }
    
    public TransformerDecoder(int numLayers, int dModel, int numHeads, int dFF, float dropoutRate) {
        this(numLayers, dModel, numHeads, dFF, dropoutRate, true, 
             TransformerDecoderLayer.NormType.LAYER_NORM, 
             TransformerDecoderLayer.FFNType.MLP);
    }
    
    public TransformerDecoder(int numLayers, int dModel, int numHeads, int dFF, float dropoutRate,
                               boolean preNorm, TransformerDecoderLayer.NormType normType,
                               TransformerDecoderLayer.FFNType ffnType) {
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
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
        
        this.numLayers = numLayers;
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dFF = dFF;
        this.dropoutRate = dropoutRate;
        this.preNorm = preNorm;
        this.normType = normType;
        this.ffnType = ffnType;
        this.layers = new ArrayList<>();
    }
    
    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }
    
    /**
     * Creates and initializes the layers (and the final norm with pre-norm) on {@code backend} from {@code random}.
     * Runs under {@link ComputationGraph#noGrad()} and records nothing.
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
        for (int i = 0; i < numLayers; i++) {
            TransformerDecoderLayer layer = new TransformerDecoderLayer(
                dModel, numHeads, dFF, dropoutRate, preNorm, normType, ffnType
            );
            layer.initialize(backend, random);
            layers.add(layer);
            registerModule("layer_" + i, layer);
        }
        
        if (preNorm) {
            if (normType == TransformerDecoderLayer.NormType.LAYER_NORM) {
                LayerNorm ln = new LayerNorm(dModel, 1e-5f, true);
                ln.initialize(backend);
                this.finalNorm = registerModule("final_norm", ln);
            } else {
                RMSNorm rn = new RMSNorm(dModel, 1e-8f, true);
                rn.initialize(backend);
                this.finalNorm = registerModule("final_norm", rn);
            }
        }
    }
    
    /**
     * Sets the position encoding applied to the input before the first layer; null removes it.
     *
     * <p>An encoding that is a {@link Module} ({@link LearnedPE}) becomes this stack's child
     * {@code "positional_encoding"} ({@link Module#replaceModule}): its parameters are in {@link #parameters()}, so an
     * optimizer built from them trains the table, its state-dict keys are {@code "positional_encoding." + key}
     * ({@code "positional_encoding.pe_embedding"} for a {@link LearnedPE}), and {@link #to}, {@link #train(boolean)},
     * {@link #zeroGrad()}, {@link #setRequiresGrad(boolean)} and {@link #close()} reach it. The child is appended after
     * the children registered so far when first set (after the layers and the final norm when set after
     * {@code initialize}) and keeps that position when another module encoding replaces it. Replacing or removing a
     * module encoding, or setting an encoding that is not a module (sinusoidal, rotary, ALiBi: no parameters, never
     * registered), unregisters the previous one without closing it: the caller owns it again.</p>
     *
     * @throws IllegalStateException when this stack, or an encoding that is a module, is closed
     */
    public void setPositionalEncoding(PositionalEncoding pe) {
        replaceModule(POSITIONAL_ENCODING, pe instanceof Module ? (Module) pe : null);
        this.posEncoding = pe;
    }
    
    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null, null, null, null, 0);
    }
    
    public Tensor forward(Tensor input, Tensor memory) {
        return forward(input, memory, null, null, null, 0);
    }
    
    public Tensor forward(Tensor input, Tensor memory, Tensor selfMask, Tensor memoryMask) {
        return forward(input, memory, selfMask, memoryMask, null, 0);
    }
    
    public Tensor forward(Tensor input, Tensor memory, Tensor selfMask, Tensor memoryMask, 
                          Tensor position, int posOffset) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        
        Shape shape = input.shape();
        int rank = shape.rank();
        
        if (rank != 2 && rank != 3) {
            throw new IllegalArgumentException("Input must be 2D [seq_len, d_model] or 3D [batch, seq_len, d_model], got rank " + rank);
        }
        
        Tensor x = posEncoding != null ? posEncoding.apply(input, posOffset) : input;
        for (TransformerDecoderLayer layer : layers) {
            x = TransformerSteps.then(x, input, h -> layer.forward(h, memory, selfMask, memoryMask, position));
        }
        if (preNorm && finalNorm != null) {
            x = TransformerSteps.then(x, input, finalNorm::forward);
        }
        return x;
    }
    
    public List<TransformerDecoderLayer> layers() {
        ensureInitialized();
        return new ArrayList<>(layers);
    }
    
    public TransformerDecoderLayer getLayer(int index) {
        ensureInitialized();
        if (index < 0 || index >= layers.size()) {
            throw new IndexOutOfBoundsException("Layer index out of bounds: " + index);
        }
        return layers.get(index);
    }
    
    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("TransformerDecoder not initialized. Call initialize(backend) first.");
        }
    }
    
    public int numLayers() {
        return numLayers;
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
    
    public TransformerDecoderLayer.NormType normType() {
        return normType;
    }
    
    public TransformerDecoderLayer.FFNType ffnType() {
        return ffnType;
    }
    
    public PositionalEncoding getPositionalEncoding() {
        return posEncoding;
    }
    
    @Override
    public String toString() {
        return "TransformerDecoder{" +
            "numLayers=" + numLayers +
            ", dModel=" + dModel +
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
