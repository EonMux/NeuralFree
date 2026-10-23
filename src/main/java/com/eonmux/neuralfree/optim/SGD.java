package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.SgdParams;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SGD extends Optimizer {
    private final float momentum;
    private final float weightDecay;
    private final boolean nesterov;
    private final List<Tensor> velocityBuffers;

    public SGD(List<Parameter> params, float lr) {
        this(params, lr, 0.0f, 0.0f, false);
    }

    public SGD(List<Parameter> params, float lr, float momentum, float weightDecay, boolean nesterov) {
        super(params, createDefaults(lr, momentum, weightDecay, nesterov));
        
        if (momentum < 0) {
            throw new IllegalArgumentException("Momentum must be non-negative, got: " + momentum);
        }
        if (weightDecay < 0) {
            throw new IllegalArgumentException("Weight decay must be non-negative, got: " + weightDecay);
        }
        
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;
        this.velocityBuffers = new ArrayList<>();
        
        if (momentum > 0) {
            for (Parameter param : params) {
                Tensor velocity = Tensor.zeros(param.backend(), param.shape());
                velocityBuffers.add(velocity);
            }
        }
    }

    private static Map<String, Object> createDefaults(float lr, float momentum, float weightDecay, boolean nesterov) {
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("lr", lr);
        defaults.put("momentum", momentum);
        defaults.put("weight_decay", weightDecay);
        defaults.put("nesterov", nesterov);
        return defaults;
    }

    @Override
    public void step() {
        ensureOpen();
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            float lr = getLearningRate();
            
            if (params.isEmpty()) {
                return;
            }
            
            for (int i = 0; i < params.size(); i++) {
                Parameter param = params.get(i);
                Tensor grad = param.grad();
            
                if (grad == null) {
                    continue;
                }
            
                Tensor velocity = momentum > 0 ? velocityBuffers.get(i) : null;
                stepParameter(param, grad, velocity, lr);
            }
            
            incrementStep();
        }
    }

    /**
     * One SGD step on a parameter: on the device with the velocity kept there ({@link GpuOptimizerOps#sgdStep}),
     * else the CPU loops below, whose results are stored back into the parameter's and the velocity's elements.
     * {@code velocity} is null exactly when momentum is 0.
     */
    private void stepParameter(Parameter param, Tensor grad, Tensor velocity, float lr) {
        SgdParams step = new SgdParams(lr, momentum, weightDecay, nesterov);
        if (GpuOptimizerOps.sgdStep(param.tensor(), grad, velocity, step)) {
            return;
        }
        if (velocity != null) {
            stepWithMomentum(param, grad, velocity, lr);
        } else {
            stepWithoutMomentum(param, grad, lr);
        }
    }

    private void stepWithoutMomentum(Parameter param, Tensor grad, float lr) {
        float[] paramData = param.tensor().toFloatArray();
        float[] gradData = grad.toFloatArray();
        
        for (int i = 0; i < paramData.length; i++) {
            float g = gradData[i];
            
            if (weightDecay != 0) {
                g += weightDecay * paramData[i];
            }
            
            paramData[i] -= lr * g;
        }
        
        GpuOptimizerOps.storeFloats(param.tensor(), paramData);
    }

    private void stepWithMomentum(Parameter param, Tensor grad, Tensor velocity, float lr) {
        float[] paramData = param.tensor().toFloatArray();
        float[] gradData = grad.toFloatArray();
        float[] velocityData = velocity.toFloatArray();
        
        for (int i = 0; i < paramData.length; i++) {
            float g = gradData[i];
            
            if (weightDecay != 0) {
                g += weightDecay * paramData[i];
            }
            
            velocityData[i] = momentum * velocityData[i] + g;
            
            if (nesterov) {
                paramData[i] -= lr * (momentum * velocityData[i] + g);
            } else {
                paramData[i] -= lr * velocityData[i];
            }
        }
        
        GpuOptimizerOps.storeFloats(param.tensor(), paramData);
        GpuOptimizerOps.storeFloats(velocity, velocityData);
    }

    public float momentum() {
        return momentum;
    }

    public float weightDecay() {
        return weightDecay;
    }

    public boolean nesterov() {
        return nesterov;
    }

    /** The momentum buffers ({@code velocity_i}), kept only when momentum is positive. */
    @Override
    protected Map<String, List<Tensor>> stateBuffers() {
        Map<String, List<Tensor>> buffers = new LinkedHashMap<>();
        buffers.put("velocity_", velocityBuffers);
        return buffers;
    }

    @Override
    public void close() {
        if (!closed) {
            for (Tensor velocity : velocityBuffers) {
                velocity.close();
            }
            velocityBuffers.clear();
            
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "SGD{closed=true}";
        }
        return "SGD{" +
               "lr=" + getLearningRate() +
               ", momentum=" + momentum +
               ", weightDecay=" + weightDecay +
               ", nesterov=" + nesterov +
               ", params=" + params.size() +
               ", stepCount=" + stepCount +
               '}';
    }
}
