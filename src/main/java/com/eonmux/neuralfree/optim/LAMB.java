package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.LambParams;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class LAMB extends Optimizer {
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final boolean biasCorrection;
    private final List<Tensor> momentBuffers;
    private final List<Tensor> velocityBuffers;

    public LAMB(List<Parameter> params, float lr) {
        this(params, lr, 0.9f, 0.999f, 1e-6f, 0.0f, true);
    }

    public LAMB(List<Parameter> params, float lr, float beta1, float beta2, float eps, float weightDecay, boolean biasCorrection) {
        super(params, createDefaults(lr, beta1, beta2, eps, weightDecay, biasCorrection));
        
        if (beta1 < 0 || beta1 >= 1) {
            throw new IllegalArgumentException("beta1 must be in [0, 1), got: " + beta1);
        }
        if (beta2 < 0 || beta2 >= 1) {
            throw new IllegalArgumentException("beta2 must be in [0, 1), got: " + beta2);
        }
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive, got: " + eps);
        }
        if (weightDecay < 0) {
            throw new IllegalArgumentException("Weight decay must be non-negative, got: " + weightDecay);
        }
        
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.biasCorrection = biasCorrection;
        this.momentBuffers = new ArrayList<>();
        this.velocityBuffers = new ArrayList<>();
        
        for (Parameter param : params) {
            Tensor m = Tensor.zeros(param.backend(), param.shape());
            Tensor v = Tensor.zeros(param.backend(), param.shape());
            momentBuffers.add(m);
            velocityBuffers.add(v);
        }
    }

    private static Map<String, Object> createDefaults(float lr, float beta1, float beta2, float eps, float weightDecay, boolean biasCorrection) {
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("lr", lr);
        defaults.put("beta1", beta1);
        defaults.put("beta2", beta2);
        defaults.put("eps", eps);
        defaults.put("weight_decay", weightDecay);
        defaults.put("bias_correction", biasCorrection);
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
            
            float biasCorrection1 = 1.0f;
            float biasCorrection2 = 1.0f;
            
            if (biasCorrection) {
                biasCorrection1 = 1.0f - (float) Math.pow(beta1, stepCount);
                biasCorrection2 = 1.0f - (float) Math.pow(beta2, stepCount);
            }
            
            for (int i = 0; i < params.size(); i++) {
                Parameter param = params.get(i);
                Tensor grad = param.grad();
            
                if (grad == null) {
                    continue;
                }
            
                Tensor m = momentBuffers.get(i);
                Tensor v = velocityBuffers.get(i);
            
                stepParameter(param, grad, m, v, lr, biasCorrection1, biasCorrection2);
            }
        }
    }

    /**
     * One LAMB step on a parameter: on the device with the moments kept there ({@link GpuOptimizerOps#lambStep}; the
     * bias corrections are 1 when bias correction is off, which leaves the moments uncorrected exactly as the CPU loop
     * does), else the CPU loop below, whose results are stored back into the parameter's and the moments' elements.
     */
    private void stepParameter(Parameter param, Tensor grad, Tensor m, Tensor v,
                               float lr, float biasCorrection1, float biasCorrection2) {
        LambParams step = new LambParams(lr, beta1, beta2, eps, weightDecay, biasCorrection1, biasCorrection2);
        if (GpuOptimizerOps.lambStep(param.tensor(), grad, m, v, step)) {
            return;
        }

        float[] paramData = param.tensor().toFloatArray();
        float[] gradData = grad.toFloatArray();
        float[] mData = m.toFloatArray();
        float[] vData = v.toFloatArray();
        
        float weightNorm = 0.0f;
        float updateNorm = 0.0f;
        
        float[] r = new float[paramData.length];
        
        for (int i = 0; i < paramData.length; i++) {
            float g = gradData[i];
            
            mData[i] = beta1 * mData[i] + (1.0f - beta1) * g;
            vData[i] = beta2 * vData[i] + (1.0f - beta2) * g * g;
            
            float mHat = mData[i];
            float vHat = vData[i];
            
            if (biasCorrection) {
                mHat = mData[i] / biasCorrection1;
                vHat = vData[i] / biasCorrection2;
            }
            
            r[i] = mHat / ((float) Math.sqrt(vHat) + eps);
            
            if (weightDecay != 0) {
                r[i] += weightDecay * paramData[i];
            }
            
            weightNorm += paramData[i] * paramData[i];
            updateNorm += r[i] * r[i];
        }
        
        weightNorm = (float) Math.sqrt(weightNorm);
        updateNorm = (float) Math.sqrt(updateNorm);
        
        float trustRatio = 1.0f;
        if (weightNorm > 0 && updateNorm > 0) {
            trustRatio = weightNorm / (updateNorm + eps);
        }
        
        for (int i = 0; i < paramData.length; i++) {
            paramData[i] -= lr * trustRatio * r[i];
        }
        
        GpuOptimizerOps.storeFloats(param.tensor(), paramData);
        GpuOptimizerOps.storeFloats(m, mData);
        GpuOptimizerOps.storeFloats(v, vData);
    }

    public float beta1() {
        return beta1;
    }

    public float beta2() {
        return beta2;
    }

    public float eps() {
        return eps;
    }

    public float weightDecay() {
        return weightDecay;
    }

    public boolean biasCorrection() {
        return biasCorrection;
    }

    /** The first moments ({@code moment_i}) and second moments ({@code velocity_i}). */
    @Override
    protected Map<String, List<Tensor>> stateBuffers() {
        Map<String, List<Tensor>> buffers = new LinkedHashMap<>();
        buffers.put("moment_", momentBuffers);
        buffers.put("velocity_", velocityBuffers);
        return buffers;
    }

    @Override
    public void close() {
        if (!closed) {
            for (Tensor m : momentBuffers) {
                m.close();
            }
            momentBuffers.clear();
            
            for (Tensor v : velocityBuffers) {
                v.close();
            }
            velocityBuffers.clear();
            
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "LAMB{closed=true}";
        }
        return "LAMB{" +
               "lr=" + getLearningRate() +
               ", beta1=" + beta1 +
               ", beta2=" + beta2 +
               ", eps=" + eps +
               ", weightDecay=" + weightDecay +
               ", biasCorrection=" + biasCorrection +
               ", params=" + params.size() +
               ", stepCount=" + stepCount +
               '}';
    }
}
