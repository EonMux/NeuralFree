package com.eonmux.neuralfree.inference;

import com.eonmux.neuralfree.tensor.DType;

import java.util.Objects;

public final class InferenceConfig {
    private final boolean enableKvCache;
    private final KVCacheConfig kvCacheConfig;
    private final boolean enableQuantization;
    private final QuantizationConfig quantizationConfig;
    private final DType dtype;
    
    public InferenceConfig(boolean enableKvCache, KVCacheConfig kvCacheConfig,
                           boolean enableQuantization, QuantizationConfig quantizationConfig,
                           DType dtype) {
        this.enableKvCache = enableKvCache;
        this.kvCacheConfig = kvCacheConfig;
        this.enableQuantization = enableQuantization;
        this.quantizationConfig = quantizationConfig;
        this.dtype = dtype != null ? dtype : DType.FLOAT32;
    }
    
    public static InferenceConfig defaults() {
        return new InferenceConfig(false, null, false, null, DType.FLOAT32);
    }
    
    public boolean enableKvCache() {
        return enableKvCache;
    }
    
    public KVCacheConfig kvCacheConfig() {
        return kvCacheConfig;
    }
    
    public boolean enableQuantization() {
        return enableQuantization;
    }
    
    public QuantizationConfig quantizationConfig() {
        return quantizationConfig;
    }
    
    public DType dtype() {
        return dtype;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        InferenceConfig that = (InferenceConfig) o;
        return enableKvCache == that.enableKvCache &&
               enableQuantization == that.enableQuantization &&
               Objects.equals(kvCacheConfig, that.kvCacheConfig) &&
               Objects.equals(quantizationConfig, that.quantizationConfig) &&
               dtype == that.dtype;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(enableKvCache, kvCacheConfig, enableQuantization, quantizationConfig, dtype);
    }
    
    @Override
    public String toString() {
        return "InferenceConfig{" +
            "enableKvCache=" + enableKvCache +
            ", kvCacheConfig=" + kvCacheConfig +
            ", enableQuantization=" + enableQuantization +
            ", quantizationConfig=" + quantizationConfig +
            ", dtype=" + dtype +
            '}';
    }
    
    public static final class Builder {
        private boolean enableKvCache = false;
        private KVCacheConfig kvCacheConfig;
        private boolean enableQuantization = false;
        private QuantizationConfig quantizationConfig;
        private DType dtype = DType.FLOAT32;
        
        public Builder enableKvCache(boolean enable) {
            this.enableKvCache = enable;
            return this;
        }
        
        public Builder kvCacheConfig(KVCacheConfig config) {
            this.kvCacheConfig = config;
            this.enableKvCache = config != null;
            return this;
        }
        
        public Builder enableQuantization(boolean enable) {
            this.enableQuantization = enable;
            return this;
        }
        
        public Builder quantizationConfig(QuantizationConfig config) {
            this.quantizationConfig = config;
            this.enableQuantization = config != null;
            return this;
        }
        
        public Builder dtype(DType dtype) {
            this.dtype = dtype;
            return this;
        }
        
        public InferenceConfig build() {
            return new InferenceConfig(enableKvCache, kvCacheConfig, 
                enableQuantization, quantizationConfig, dtype);
        }
    }
}
