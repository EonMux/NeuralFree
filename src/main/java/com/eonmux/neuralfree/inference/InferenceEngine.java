package com.eonmux.neuralfree.inference;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.transformer.KVCache;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Objects;

public final class InferenceEngine implements AutoCloseable {
    private final Module model;
    private final ComputeBackend backend;
    private final InferenceConfig config;
    private final KVCache kvCache;
    
    private volatile boolean closed = false;
    private boolean wasTraining;
    
    public InferenceEngine(Module model, ComputeBackend backend) {
        this(model, backend, InferenceConfig.defaults());
    }
    
    public InferenceEngine(Module model, ComputeBackend backend, InferenceConfig config) {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(config, "Config cannot be null");
        
        this.model = model;
        this.backend = backend;
        this.config = config;
        
        this.wasTraining = model.isTraining();
        
        model.eval();
        
        if (config.enableKvCache() && config.kvCacheConfig() != null) {
            KVCacheConfig kvConfig = config.kvCacheConfig();
            this.kvCache = new KVCache(
                kvConfig.numLayers(),
                kvConfig.numHeads(),
                kvConfig.headDim(),
                kvConfig.maxSeqLen(),
                kvConfig.batchSize()
            );
            this.kvCache.initialize(backend);
        } else {
            this.kvCache = null;
        }
    }
    
    public Tensor infer(Tensor input) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");
        
        ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
        try {
            return model.forward(input);
        } finally {
            noGrad.close();
        }
    }
    
    public Tensor inferWithKvCache(Tensor input) {
        ensureOpen();
        Objects.requireNonNull(input, "Input cannot be null");
        
        if (kvCache == null) {
            return infer(input);
        }
        
        ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
        try {
            return model.forward(input);
        } finally {
            noGrad.close();
        }
    }
    
    public Tensor[] inferBatch(Tensor[] inputs) {
        ensureOpen();
        Objects.requireNonNull(inputs, "Inputs cannot be null");
        
        Tensor[] outputs = new Tensor[inputs.length];
        ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
        try {
            for (int i = 0; i < inputs.length; i++) {
                outputs[i] = model.forward(inputs[i]);
            }
            return outputs;
        } finally {
            noGrad.close();
        }
    }
    
    public void resetKvCache() {
        ensureOpen();
        if (kvCache != null) {
            kvCache.clear();
        }
    }
    
    public void resizeKvCache(int newMaxSeqLen) {
        ensureOpen();
        if (kvCache != null) {
            kvCache.resize(newMaxSeqLen);
        }
    }
    
    public KVCache kvCache() {
        ensureOpen();
        return kvCache;
    }
    
    public Module model() {
        ensureOpen();
        return model;
    }
    
    public ComputeBackend backend() {
        return backend;
    }
    
    public InferenceConfig config() {
        return config;
    }
    
    public int kvCacheSeqLength(int layerIdx) {
        ensureOpen();
        return kvCache != null ? kvCache.getSeqLength(layerIdx) : 0;
    }
    
    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("InferenceEngine has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            
            if (kvCache != null) {
                kvCache.close();
            }
            
            if (wasTraining) {
                model.train();
            }
        }
    }
    
    @Override
    public String toString() {
        if (closed) {
            return "InferenceEngine{closed=true}";
        }
        
        return "InferenceEngine{" +
            "model=" + model.moduleName() +
            ", kvCacheEnabled=" + (kvCache != null) +
            ", config=" + config +
            '}';
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static final class Builder {
        private Module model;
        private ComputeBackend backend;
        private InferenceConfig config = InferenceConfig.defaults();
        
        public Builder model(Module model) {
            this.model = model;
            return this;
        }
        
        public Builder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public Builder config(InferenceConfig config) {
            this.config = config;
            return this;
        }
        
        public Builder enableKvCache(int numLayers, int numHeads, int headDim, int maxSeqLen, int batchSize) {
            this.config = new InferenceConfig(
                true,
                new KVCacheConfig(numLayers, numHeads, headDim, maxSeqLen, batchSize),
                config.enableQuantization(),
                config.quantizationConfig(),
                config.dtype()
            );
            return this;
        }
        
        public InferenceEngine build() {
            Objects.requireNonNull(model, "Model cannot be null");
            Objects.requireNonNull(backend, "Backend cannot be null");
            return new InferenceEngine(model, backend, config);
        }
    }
}
