package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.AdagradParams;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Adagrad extends Optimizer {
    private final float eps;
    private final float weightDecay;
    private final float initialAccumulatorValue;
    private final List<Tensor> accumulatorBuffers;

    public Adagrad(List<Parameter> params, float lr) {
        this(params, lr, 1e-10f, 0.0f, 0.0f);
    }

    public Adagrad(List<Parameter> params, float lr, float eps, float weightDecay, float initialAccumulatorValue) {
        super(params, createDefaults(lr, eps, weightDecay, initialAccumulatorValue));
        
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive, got: " + eps);
        }
        if (weightDecay < 0) {
            throw new IllegalArgumentException("Weight decay must be non-negative, got: " + weightDecay);
        }
        if (initialAccumulatorValue < 0) {
            throw new IllegalArgumentException("Initial accumulator value must be non-negative, got: " + initialAccumulatorValue);
        }
        
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.initialAccumulatorValue = initialAccumulatorValue;
        this.accumulatorBuffers = new ArrayList<>();
        
        for (Parameter param : params) {
            accumulatorBuffers.add(initialAccumulator(param, initialAccumulatorValue));
        }
    }

    /** A tensor of the parameter's shape holding {@code value} in every element, on the parameter's backend. */
    private static Tensor initialAccumulator(Parameter param, float value) {
        if (value == 0.0f) {
            return Tensor.zeros(param.backend(), param.shape());
        }
        float[] initialData = new float[(int) param.numElements()];
        Arrays.fill(initialData, value);
        Tensor flat = Tensor.fromArray(param.backend(), initialData);
        try {
            return flat.reshape(param.shape());
        } finally {
            flat.close();
        }
    }

    private static Map<String, Object> createDefaults(float lr, float eps, float weightDecay, float initialAccumulatorValue) {
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("lr", lr);
        defaults.put("eps", eps);
        defaults.put("weight_decay", weightDecay);
        defaults.put("initial_accumulator_value", initialAccumulatorValue);
        return defaults;
    }

    @Override
    public void step() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            if (params.isEmpty()) {
                return;
            }
            
            float lr = getLearningRate();
            
            incrementStep();
            
            for (int i = 0; i < params.size(); i++) {
                Parameter param = params.get(i);
                Tensor grad = param.grad();
            
                if (grad == null) {
                    continue;
                }
            
                Tensor acc = accumulatorBuffers.get(i);
                stepParameter(param, grad, acc, lr);
            }
        }
    }

    /**
     * One Adagrad step on a parameter: on the device with the accumulator kept there
     * ({@link GpuOptimizerOps#adagradStep}), else the CPU loop below, whose results are stored back into the
     * parameter's and the accumulator's elements.
     */
    private void stepParameter(Parameter param, Tensor grad, Tensor acc, float lr) {
        if (GpuOptimizerOps.adagradStep(param.tensor(), grad, acc, new AdagradParams(lr, eps, weightDecay))) {
            return;
        }

        float[] paramData = param.tensor().toFloatArray();
        float[] gradData = grad.toFloatArray();
        float[] accData = acc.toFloatArray();
        
        for (int i = 0; i < paramData.length; i++) {
            float g = gradData[i];
            
            if (weightDecay != 0) {
                g += weightDecay * paramData[i];
            }
            
            accData[i] += g * g;
            
            paramData[i] -= lr * g / ((float) Math.sqrt(accData[i]) + eps);
        }
        
        GpuOptimizerOps.storeFloats(param.tensor(), paramData);
        GpuOptimizerOps.storeFloats(acc, accData);
    }

    public float eps() {
        return eps;
    }

    public float weightDecay() {
        return weightDecay;
    }

    public float initialAccumulatorValue() {
        return initialAccumulatorValue;
    }

    /** The squared-gradient sums ({@code accumulator_i}). */
    @Override
    protected Map<String, List<Tensor>> stateBuffers() {
        Map<String, List<Tensor>> buffers = new LinkedHashMap<>();
        buffers.put("accumulator_", accumulatorBuffers);
        return buffers;
    }

    @Override
    public void close() {
        if (!closed) {
            for (Tensor acc : accumulatorBuffers) {
                acc.close();
            }
            accumulatorBuffers.clear();
            
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "Adagrad{closed=true}";
        }
        return "Adagrad{" +
               "lr=" + getLearningRate() +
               ", eps=" + eps +
               ", weightDecay=" + weightDecay +
               ", initialAccumulatorValue=" + initialAccumulatorValue +
               ", params=" + params.size() +
               ", stepCount=" + stepCount +
               '}';
    }
}
