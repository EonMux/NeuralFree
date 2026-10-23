package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * A stack of selective state-space layers ({@link SSMLayer}) over {@code [batch, seqLen, inputSize]} input, each
 * mapping its input to {@code [batch, seqLen, hiddenSize]}. With {@code numLayers > 1} and a non-zero dropout rate,
 * dropout (training mode only) is applied to the output of every layer except the last, as in {@link RNN}.
 *
 * <p>The forward pass is composed of recorded operations ({@link TensorOps}, Tensor views and the child modules), so
 * in grad mode the output carries the graph back to every parameter, the state matrix {@code a_log} and the output
 * scale {@code d} included.</p>
 */
public final class StateSpaceModel extends Module {
    private final int inputSize;
    private final int hiddenSize;
    private final int stateDim;
    private final int numLayers;
    private final boolean useBias;
    private final float dropout;
    
    private final List<SSMLayer> layers;
    private final List<Dropout> dropoutLayers;
    
    private boolean initialized = false;

    public StateSpaceModel(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, hiddenSize, 1, true, 0.0f);
    }

    public StateSpaceModel(int inputSize, int hiddenSize, int stateDim) {
        this(inputSize, hiddenSize, stateDim, 1, true, 0.0f);
    }

    public StateSpaceModel(int inputSize, int hiddenSize, int stateDim, int numLayers, 
                          boolean useBias, float dropout) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        if (stateDim <= 0) {
            throw new IllegalArgumentException("stateDim must be positive: " + stateDim);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (dropout < 0.0f || dropout >= 1.0f) {
            throw new IllegalArgumentException("dropout must be in [0, 1), got: " + dropout);
        }
        
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.stateDim = stateDim;
        this.numLayers = numLayers;
        this.useBias = useBias;
        this.dropout = dropout;
        this.layers = new ArrayList<>();
        this.dropoutLayers = new ArrayList<>();
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (initialized) {
            throw new IllegalStateException("StateSpaceModel is already initialized");
        }
        
        int layerInputSize = inputSize;
        for (int layer = 0; layer < numLayers; layer++) {
            SSMLayer ssmLayer = new SSMLayer(layerInputSize, hiddenSize, stateDim, useBias);
            ssmLayer.initialize(backend, random);
            layers.add(ssmLayer);
            registerModule("layer_" + layer, ssmLayer);
            layerInputSize = hiddenSize;
        }
        
        if (dropout > 0 && numLayers > 1) {
            for (int layer = 0; layer < numLayers - 1; layer++) {
                Dropout dropoutLayer = new Dropout(dropout);
                dropoutLayers.add(dropoutLayer);
                registerModule("dropout_" + layer, dropoutLayer);
            }
        }
        
        initialized = true;
    }

    /**
     * Runs every layer over {@code input} of shape {@code [batch, seqLen, inputSize]} and returns a new
     * {@code [batch, seqLen, hiddenSize]} tensor. Dropout (training mode only) is applied between layers.
     * Intermediate tensors are closed; {@code input} is not.
     *
     * @throws IllegalArgumentException when {@code input} is not rank 3 with last dimension {@code inputSize}, or has
     *                                  an empty batch or sequence
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        RNN.Sequences.batchAndLength(input, inputSize, "StateSpaceModel");
        
        Tensor output = input;
        try {
            for (int layer = 0; layer < layers.size(); layer++) {
                output = RNN.Sequences.advance(output, layers.get(layer).forward(output), input);
                if (layer < dropoutLayers.size()) {
                    output = RNN.Sequences.advance(output, dropoutLayers.get(layer).forward(output), input);
                }
            }
            return output;
        } catch (RuntimeException e) {
            RNN.Sequences.release(output, input);
            throw e;
        }
    }

    private void ensureInitialized() {
        if (!initialized || layers.isEmpty()) {
            throw new IllegalStateException("StateSpaceModel not initialized. Call initialize(backend) first.");
        }
    }

    public int inputSize() {
        return inputSize;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int stateDim() {
        return stateDim;
    }

    public int numLayers() {
        return numLayers;
    }

    public float dropoutRate() {
        return dropout;
    }

    @Override
    public String toString() {
        return "StateSpaceModel{" +
            "inputSize=" + inputSize +
            ", hiddenSize=" + hiddenSize +
            ", stateDim=" + stateDim +
            ", numLayers=" + numLayers +
            ", dropout=" + dropout +
            ", initialized=" + initialized +
            '}';
    }

    /**
     * One selective state-space layer over {@code [batch, seqLen, inputSize]} input, with state size
     * {@code stateDim} and output size {@code hiddenSize}. With {@code a = exp(a_log)} and {@code d} the learned
     * output scale, every step {@code t} computes from the projections of its input row {@code x_t}:
     * <pre>
     *   u_t     = tanh(input_proj(x_t))
     *   delta_t = sigmoid(delta_proj(u_t)),  b_t = b_proj(u_t),  c_t = c_proj(u_t)
     *   aBar_t  = exp(delta_t * a)
     *   bBar_t  = (|a| &lt; 1e-8 ? delta_t : (aBar_t - 1) / a) * b_t
     *   s_t     = aBar_t * s_{t-1} + bBar_t * xs_t           (s_{-1} = 0)
     *   y_t     = sum_s(c_t * s_t) * d                        ([hiddenSize])
     * </pre>
     * where {@code xs_t} is {@code x_t} truncated or zero-padded to {@code stateDim} features. Everything except the
     * recurrence for {@code s_t} is computed for all steps at once; the recurrence is a Java loop over device
     * tensors whose states are stacked ({@link TensorOps#stack}) into one {@code [batch, seqLen, stateDim]} tensor.
     * Every step is a recorded operation, so the gradient of the output reaches the four projections' parameters,
     * {@code a_log} (through {@code a} in {@code aBar} and in the ratio) and {@code d}.
     */
    private static final class SSMLayer extends Module {
        private static final float ZERO_A_THRESHOLD = 1e-8f;

        private final int inputSize;
        private final int hiddenSize;
        private final int stateDim;
        private final boolean useBias;
        
        private Linear inputProjection;
        private Linear deltaProjection;
        private Linear bProjection;
        private Linear cProjection;
        
        private Parameter aLogParam;
        private Parameter dParam;
        
        private boolean layerInitialized = false;

        SSMLayer(int inputSize, int hiddenSize, int stateDim, boolean useBias) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.stateDim = stateDim;
            this.useBias = useBias;
        }

        void initialize(ComputeBackend backend, Random random) {
            Objects.requireNonNull(backend, "Backend cannot be null");
            
            inputProjection = new Linear(inputSize, hiddenSize, useBias);
            inputProjection.initialize(backend, random);
            registerModule("input_proj", inputProjection);
            
            deltaProjection = new Linear(hiddenSize, stateDim, useBias);
            deltaProjection.initialize(backend, random);
            registerModule("delta_proj", deltaProjection);
            
            bProjection = new Linear(hiddenSize, stateDim, useBias);
            bProjection.initialize(backend, random);
            registerModule("b_proj", bProjection);
            
            cProjection = new Linear(hiddenSize, stateDim, useBias);
            cProjection.initialize(backend, random);
            registerModule("c_proj", cProjection);
            
            float[] aInit = new float[stateDim];
            for (int i = 0; i < stateDim; i++) {
                aInit[i] = (float) Math.log(0.5 * (1.0 + random.nextGaussian() * 0.1));
            }
            aLogParam = registerParameter("a_log", new Parameter(Tensor.fromArray(backend, aInit), "a_log"));
            
            float[] dInit = new float[hiddenSize];
            Arrays.fill(dInit, 1.0f);
            dParam = registerParameter("d", new Parameter(Tensor.fromArray(backend, dInit), "d"));
            
            layerInitialized = true;
        }

        /**
         * The layer output {@code [batch, seqLen, hiddenSize]} for {@code input} {@code [batch, seqLen, inputSize]};
         * every intermediate is closed and {@code input} stays open.
         */
        @Override
        public Tensor forward(Tensor input) {
            if (!layerInitialized) {
                throw new IllegalStateException("SSMLayer not initialized");
            }
            Objects.requireNonNull(input, "Input cannot be null");
            int seqLen = RNN.Sequences.batchAndLength(input, inputSize, "SSMLayer")[1];
            
            try (Tensor projectedLinear = inputProjection.forward(input);
                 Tensor projected = TensorOps.tanh(projectedLinear);
                 Tensor deltaLinear = deltaProjection.forward(projected);
                 Tensor delta = TensorOps.sigmoid(deltaLinear);
                 Tensor bValues = bProjection.forward(projected);
                 Tensor cValues = cProjection.forward(projected);
                 Tensor a = TensorOps.exp(aLogParam.tensor());
                 Tensor deltaA = TensorOps.mul(delta, a);
                 Tensor aBar = TensorOps.exp(deltaA);
                 Tensor bBar = discretizeB(delta, a, aBar, bValues);
                 Tensor stateInput = stateInput(input);
                 Tensor drive = TensorOps.mul(bBar, stateInput);
                 Tensor states = scan(aBar, drive, seqLen);
                 Tensor readout = TensorOps.mul(cValues, states);
                 Tensor y = TensorOps.sum(readout, 2);
                 Tensor yColumn = y.unsqueeze(2)) {
                return TensorOps.mul(yColumn, dParam.tensor());
            }
        }

        /**
         * {@code bBar = factor * b} with {@code factor = |a| < 1e-8 ? delta : (aBar - 1) / a} per state, selected
         * element-wise on the device from the broadcast {@code [stateDim]} condition. The condition is evaluated as
         * {@code (|a| - 1e-8) < 0}: a correctly rounded difference of two floats is zero only when they are equal and
         * otherwise keeps the sign of the exact difference, so it selects exactly the states with {@code |a| < 1e-8}
         * (a NaN compares false and takes the ratio, as in the direct comparison) without uploading a threshold.
         *
         * <p>The condition is a constant of the step: it is computed from a detached handle of {@code a} and records
         * nothing. The ratio divides by {@code a} only for the states that select it and by 1 for the others, so a
         * state whose {@code a} underflows to zero gives a finite ratio and no NaN gradient through the unselected
         * branch; the selected values are the same as with a plain division.</p>
         */
        private static Tensor discretizeB(Tensor delta, Tensor a, Tensor aBar, Tensor bValues) {
            try (Tensor nearZero = nearZeroMask(a);
                 Tensor one = Tensor.ones(a.backend(), a.shape());
                 Tensor safeA = TensorOps.where(nearZero, one, a);
                 Tensor nearZeroWide = nearZero.expand(delta.shape());
                 Tensor growth = TensorOps.add(aBar, -1.0f);
                 Tensor ratio = TensorOps.div(growth, safeA);
                 Tensor factor = TensorOps.where(nearZeroWide, delta, ratio)) {
                return TensorOps.mul(factor, bValues);
            }
        }

        /** The FLOAT32 {@code [stateDim]} mask of {@code (|a| - 1e-8) < 0} (1 or 0), computed without recording. */
        private static Tensor nearZeroMask(Tensor a) {
            try (Tensor constant = a.detach();
                 Tensor absA = TensorOps.abs(constant);
                 Tensor margin = TensorOps.add(absA, -ZERO_A_THRESHOLD);
                 Tensor zero = Tensor.zeros(a.backend(), Shape.scalar());
                 Tensor mask = TensorOps.lt(margin, zero)) {
                return mask.to(DType.FLOAT32);
            }
        }

        /** The input features driving the state: {@code input} truncated or zero-padded to {@code stateDim}. */
        private Tensor stateInput(Tensor input) {
            if (stateDim <= inputSize) {
                return input.slice(2, 0, stateDim);
            }
            Shape shape = input.shape();
            try (Tensor padding = Tensor.zeros(input.backend(), Shape.of(shape.dim(0), shape.dim(1),
                    stateDim - inputSize))) {
                return TensorOps.cat(Arrays.asList(input, padding), 2);
            }
        }

        /**
         * {@code s_t = aBar_t * s_{t-1} + drive_t} from {@code s_{-1} = 0}, stacked to
         * {@code [batch, seqLen, stateDim]}.
         */
        private Tensor scan(Tensor aBar, Tensor drive, int seqLen) {
            int batchSize = aBar.shape().dim(0);
            Tensor[] steps = new Tensor[seqLen];
            try (Tensor initial = Tensor.zeros(aBar.backend(), Shape.matrix(batchSize, stateDim))) {
                Tensor state = initial;
                for (int t = 0; t < seqLen; t++) {
                    try (Tensor decay = RNN.Sequences.timeStep(aBar, t);
                         Tensor stepDrive = RNN.Sequences.timeStep(drive, t);
                         Tensor carried = TensorOps.mul(decay, state)) {
                        steps[t] = TensorOps.add(carried, stepDrive);
                    }
                    state = steps[t];
                }
                return RNN.Sequences.stackSteps(steps);
            } catch (RuntimeException e) {
                RNN.Sequences.closeAll(steps);
                throw e;
            }
        }
    }
}
