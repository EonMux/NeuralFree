package com.eonmux.neuralfree.inference;

import java.util.Objects;

public final class PagedAttentionConfig {
    private final int pageSize;
    private final int numPages;
    private final int numHeads;
    private final int headDim;
    private final int numLayers;
    
    public PagedAttentionConfig(int pageSize, int numPages, int numHeads, int headDim, int numLayers) {
        if (pageSize <= 0) {
            throw new IllegalArgumentException("pageSize must be positive: " + pageSize);
        }
        if (numPages <= 0) {
            throw new IllegalArgumentException("numPages must be positive: " + numPages);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (headDim <= 0) {
            throw new IllegalArgumentException("headDim must be positive: " + headDim);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        
        this.pageSize = pageSize;
        this.numPages = numPages;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.numLayers = numLayers;
    }
    
    public static PagedAttentionConfig defaults() {
        return new PagedAttentionConfig(16, 1024, 32, 128, 32);
    }
    
    public static PagedAttentionConfig small() {
        return new PagedAttentionConfig(8, 256, 16, 64, 12);
    }
    
    public static PagedAttentionConfig large() {
        return new PagedAttentionConfig(32, 4096, 32, 128, 32);
    }
    
    public int pageSize() {
        return pageSize;
    }
    
    public int numPages() {
        return numPages;
    }
    
    public int numHeads() {
        return numHeads;
    }
    
    public int headDim() {
        return headDim;
    }
    
    public int numLayers() {
        return numLayers;
    }
    
    public long maxMemoryBytes() {
        return (long) numPages * pageSize * numHeads * headDim * numLayers * 2L * 4L;
    }
    
    public int maxTotalTokens() {
        return numPages * pageSize;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PagedAttentionConfig that = (PagedAttentionConfig) o;
        return pageSize == that.pageSize &&
               numPages == that.numPages &&
               numHeads == that.numHeads &&
               headDim == that.headDim &&
               numLayers == that.numLayers;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(pageSize, numPages, numHeads, headDim, numLayers);
    }
    
    @Override
    public String toString() {
        return "PagedAttentionConfig{" +
            "pageSize=" + pageSize +
            ", numPages=" + numPages +
            ", numHeads=" + numHeads +
            ", headDim=" + headDim +
            ", numLayers=" + numLayers +
            ", maxTokens=" + maxTotalTokens() +
            '}';
    }
    
    public static final class Builder {
        private int pageSize = 16;
        private int numPages = 1024;
        private int numHeads = 32;
        private int headDim = 128;
        private int numLayers = 32;
        
        public Builder pageSize(int pageSize) {
            this.pageSize = pageSize;
            return this;
        }
        
        public Builder numPages(int numPages) {
            this.numPages = numPages;
            return this;
        }
        
        public Builder numHeads(int numHeads) {
            this.numHeads = numHeads;
            return this;
        }
        
        public Builder headDim(int headDim) {
            this.headDim = headDim;
            return this;
        }
        
        public Builder numLayers(int numLayers) {
            this.numLayers = numLayers;
            return this;
        }
        
        public PagedAttentionConfig build() {
            return new PagedAttentionConfig(pageSize, numPages, numHeads, headDim, numLayers);
        }
    }
}
