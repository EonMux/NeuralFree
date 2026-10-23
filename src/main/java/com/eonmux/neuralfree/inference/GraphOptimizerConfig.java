package com.eonmux.neuralfree.inference;

import java.util.Objects;

public final class GraphOptimizerConfig {
    private final boolean enableFusion;
    private final boolean enableConstantFolding;
    private final boolean enableDeadCodeElimination;
    private final boolean enableMemoryPlanning;
    
    public GraphOptimizerConfig(boolean enableFusion, boolean enableConstantFolding,
                                boolean enableDeadCodeElimination, boolean enableMemoryPlanning) {
        this.enableFusion = enableFusion;
        this.enableConstantFolding = enableConstantFolding;
        this.enableDeadCodeElimination = enableDeadCodeElimination;
        this.enableMemoryPlanning = enableMemoryPlanning;
    }
    
    public static GraphOptimizerConfig defaults() {
        return new GraphOptimizerConfig(true, true, true, true);
    }
    
    public static GraphOptimizerConfig fusionOnly() {
        return new GraphOptimizerConfig(true, false, false, false);
    }
    
    public static GraphOptimizerConfig minimal() {
        return new GraphOptimizerConfig(false, false, false, false);
    }
    
    public static GraphOptimizerConfig aggressive() {
        return new GraphOptimizerConfig(true, true, true, true);
    }
    
    public boolean enableFusion() {
        return enableFusion;
    }
    
    public boolean enableConstantFolding() {
        return enableConstantFolding;
    }
    
    public boolean enableDeadCodeElimination() {
        return enableDeadCodeElimination;
    }
    
    public boolean enableMemoryPlanning() {
        return enableMemoryPlanning;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GraphOptimizerConfig that = (GraphOptimizerConfig) o;
        return enableFusion == that.enableFusion &&
               enableConstantFolding == that.enableConstantFolding &&
               enableDeadCodeElimination == that.enableDeadCodeElimination &&
               enableMemoryPlanning == that.enableMemoryPlanning;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(enableFusion, enableConstantFolding, 
            enableDeadCodeElimination, enableMemoryPlanning);
    }
    
    @Override
    public String toString() {
        return "GraphOptimizerConfig{" +
            "enableFusion=" + enableFusion +
            ", enableConstantFolding=" + enableConstantFolding +
            ", enableDeadCodeElimination=" + enableDeadCodeElimination +
            ", enableMemoryPlanning=" + enableMemoryPlanning +
            '}';
    }
    
    public static final class Builder {
        private boolean enableFusion = true;
        private boolean enableConstantFolding = true;
        private boolean enableDeadCodeElimination = true;
        private boolean enableMemoryPlanning = true;
        
        public Builder enableFusion(boolean enable) {
            this.enableFusion = enable;
            return this;
        }
        
        public Builder enableConstantFolding(boolean enable) {
            this.enableConstantFolding = enable;
            return this;
        }
        
        public Builder enableDeadCodeElimination(boolean enable) {
            this.enableDeadCodeElimination = enable;
            return this;
        }
        
        public Builder enableMemoryPlanning(boolean enable) {
            this.enableMemoryPlanning = enable;
            return this;
        }
        
        public GraphOptimizerConfig build() {
            return new GraphOptimizerConfig(enableFusion, enableConstantFolding,
                enableDeadCodeElimination, enableMemoryPlanning);
        }
    }
}
