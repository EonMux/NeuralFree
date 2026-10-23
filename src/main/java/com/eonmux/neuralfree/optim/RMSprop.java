package com.eonmux.neuralfree.optim;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps.RmspropParams;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class RMSprop extends Optimizer {
    private final float alpha;
    private final float eps;
    private final float weightDecay;
    private final boolean centered;
    private final float momentum;
    private final List<Tensor> squaredGradBuffers;
    private final List<Tensor> momentumBuffers;
    private final List<Tensor> avgGradBuffers;

    public RMSprop(List<Parameter> params, float lr) {
        this(params, lr, 0.9f, 1e-8f, 0.0f, false, 0.0f);
    }

    public RMSprop(List<Parameter> params, float lr, float alpha, float eps, float weightDecay, boolean centered, float momentum) {
        super(params, createDefaults(lr, alpha, eps, weightDecay, centered, momentum));
        
        if (alpha < 0 || alpha >= 1) {
            throw new IllegalArgumentException("alpha (smoothing constant) must be in [0, 1), got: " + alpha);
        }
        if (eps <= 0) {
            throw new IllegalArgumentException("eps must be positive, got: " + eps);
        }
        if (weightDecay < 0) {
            throw new IllegalArgumentException("Weight decay must be non-negative, got: " + weightDecay);
        }
        if (momentum < 0) {
            throw new IllegalArgumentException("Momentum must be non-negative, got: " + momentum);
        }
        
        this.alpha = alpha;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.centered = centered;
        this.momentum = momentum;
        this.squaredGradBuffers = new ArrayList<>();
        this.momentumBuffers = new ArrayList<>();
        this.avgGradBuffers = new ArrayList<>();
        
        for (Parameter param : params) {
            Tensor v = Tensor.zeros(param.backend(), param.shape());
            squaredGradBuffers.add(v);
            
            if (momentum > 0) {
                Tensor buf = Tensor.zeros(param.backend(), param.shape());
                momentumBuffers.add(buf);
            } else {
                momentumBuffers.add(null);
            }
            
            if (centered) {
                Tensor avg = Tensor.zeros(param.backend(), param.shape());
                avgGradBuffers.add(avg);
            } else {
                avgGradBuffers.add(null);
            }
        }
    }

    private static Map<String, Object> createDefaults(float lr, float alpha, float eps, float weightDecay, boolean centered, float momentum) {
        Map<String, Object> defaults = new HashMap<>();
        defaults.put("lr", lr);
        defaults.put("alpha", alpha);
        defaults.put("eps", eps);
        defaults.put("weight_decay", weightDecay);
        defaults.put("centered", centered);
        defaults.put("momentum", momentum);
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
            
                Tensor v = squaredGradBuffers.get(i);
                Tensor momentumBuf = momentumBuffers.get(i);
                Tensor avgGrad = avgGradBuffers.get(i);
            
                stepParameter(param, grad, v, momentumBuf, avgGrad, lr);
            }
        }
    }

    /**
     * One RMSprop step on a parameter: on the device with the squared-gradient average, the momentum buffer (null
     * unless momentum is positive) and the gradient average (null unless centered) kept there
     * ({@link GpuOptimizerOps#rmspropStep}), else the CPU loops below, whose results are stored back into the
     * tensors' elements.
     */
    private void stepParameter(Parameter param, Tensor grad, Tensor v, Tensor momentumBuf, Tensor avgGrad, float lr) {
        RmspropParams step = new RmspropParams(lr, alpha, eps, weightDecay, momentum, centered);
        if (GpuOptimizerOps.rmspropStep(param.tensor(), grad, v, momentumBuf, avgGrad, step)) {
            return;
        }
        if (centered) {
            stepCentered(param, grad, v, avgGrad, momentumBuf, lr);
        } else {
            stepPlain(param, grad, v, momentumBuf, lr);
        }
    }

    private void stepPlain(Parameter param, Tensor grad, Tensor v, Tensor momentumBuf, float lr) {
        float[] paramData = param.tensor().toFloatArray();
        float[] gradData = grad.toFloatArray();
        float[] vData = v.toFloatArray();
        float[] momentumData = momentumBuf != null ? momentumBuf.toFloatArray() : null;
        
        for (int i = 0; i < paramData.length; i++) {
            float g = gradData[i];
            
            if (weightDecay != 0) {
                g += weightDecay * paramData[i];
            }
            
            vData[i] = alpha * vData[i] + (1.0f - alpha) * g * g;
            
            float update = g / ((float) Math.sqrt(vData[i]) + eps);
            
            if (momentum > 0 && momentumData != null) {
                momentumData[i] = momentum * momentumData[i] + update;
                paramData[i] -= lr * momentumData[i];
            } else {
                paramData[i] -= lr * update;
            }
        }
        
        GpuOptimizerOps.storeFloats(param.tensor(), paramData);
        GpuOptimizerOps.storeFloats(v, vData);
        if (momentumBuf != null) {
            GpuOptimizerOps.storeFloats(momentumBuf, momentumData);
        }
    }

    private void stepCentered(Parameter param, Tensor grad, Tensor v, Tensor avgGrad, Tensor momentumBuf, float lr) {
        float[] paramData = param.tensor().toFloatArray();
        float[] gradData = grad.toFloatArray();
        float[] vData = v.toFloatArray();
        float[] avgGradData = avgGrad.toFloatArray();
        float[] momentumData = momentumBuf != null ? momentumBuf.toFloatArray() : null;
        
        for (int i = 0; i < paramData.length; i++) {
            float g = gradData[i];
            
            if (weightDecay != 0) {
                g += weightDecay * paramData[i];
            }
            
            avgGradData[i] = alpha * avgGradData[i] + (1.0f - alpha) * g;
            vData[i] = alpha * vData[i] + (1.0f - alpha) * g * g;
            
            float vHat = vData[i] - avgGradData[i] * avgGradData[i];
            float update = g / ((float) Math.sqrt(vHat) + eps);
            
            if (momentum > 0 && momentumData != null) {
                momentumData[i] = momentum * momentumData[i] + update;
                paramData[i] -= lr * momentumData[i];
            } else {
                paramData[i] -= lr * update;
            }
        }
        
        GpuOptimizerOps.storeFloats(param.tensor(), paramData);
        GpuOptimizerOps.storeFloats(v, vData);
        GpuOptimizerOps.storeFloats(avgGrad, avgGradData);
        if (momentumBuf != null) {
            GpuOptimizerOps.storeFloats(momentumBuf, momentumData);
        }
    }

    public float alpha() {
        return alpha;
    }

    public float eps() {
        return eps;
    }

    public float weightDecay() {
        return weightDecay;
    }

    public boolean centered() {
        return centered;
    }

    public float momentum() {
        return momentum;
    }

    /**
     * The squared-gradient averages ({@code squared_grad_i}), the momentum buffers ({@code momentum_buf_i}, only when
     * momentum is positive) and the gradient averages ({@code avg_grad_i}, only when centered).
     */
    @Override
    protected Map<String, List<Tensor>> stateBuffers() {
        Map<String, List<Tensor>> buffers = new LinkedHashMap<>();
        buffers.put("squared_grad_", squaredGradBuffers);
        buffers.put("momentum_buf_", momentumBuffers);
        buffers.put("avg_grad_", avgGradBuffers);
        return buffers;
    }

    @Override
    public void close() {
        if (!closed) {
            for (Tensor v : squaredGradBuffers) {
                v.close();
            }
            squaredGradBuffers.clear();
            
            for (Tensor m : momentumBuffers) {
                if (m != null) {
                    m.close();
                }
            }
            momentumBuffers.clear();
            
            for (Tensor a : avgGradBuffers) {
                if (a != null) {
                    a.close();
                }
            }
            avgGradBuffers.clear();
            
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "RMSprop{closed=true}";
        }
        return "RMSprop{" +
               "lr=" + getLearningRate() +
               ", alpha=" + alpha +
               ", eps=" + eps +
               ", weightDecay=" + weightDecay +
               ", centered=" + centered +
               ", momentum=" + momentum +
               ", params=" + params.size() +
               ", stepCount=" + stepCount +
               '}';
    }
}
