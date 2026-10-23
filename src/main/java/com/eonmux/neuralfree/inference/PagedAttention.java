package com.eonmux.neuralfree.inference;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class PagedAttention implements AutoCloseable {
    private final ComputeBackend backend;
    private final PagedAttentionConfig config;
    
    private final List<KVPage> pagePool;
    private final BitSet freePages;
    private final Map<Integer, List<Integer>> sequenceToPages;
    
    private volatile boolean closed = false;
    private int nextSequenceId = 0;
    
    public PagedAttention(ComputeBackend backend, PagedAttentionConfig config) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(config, "Config cannot be null");
        
        this.backend = backend;
        this.config = config;
        
        int numPages = config.numPages();
        this.pagePool = new ArrayList<>(numPages);
        this.freePages = new BitSet(numPages);
        this.freePages.set(0, numPages);
        this.sequenceToPages = new HashMap<>();
        
        initializePagePool();
    }
    
    private void initializePagePool() {
        int numPages = config.numPages();
        int pageSize = config.pageSize();
        int numHeads = config.numHeads();
        int headDim = config.headDim();
        
        for (int i = 0; i < numPages; i++) {
            KVPage page = new KVPage(
                i,
                pageSize,
                numHeads,
                headDim,
                backend
            );
            pagePool.add(page);
        }
    }
    
    public int allocateSequence(int initialTokens) {
        ensureOpen();
        
        if (initialTokens < 0) {
            throw new IllegalArgumentException("initialTokens must be non-negative: " + initialTokens);
        }
        
        int sequenceId = nextSequenceId++;
        int pagesNeeded = calculatePagesNeeded(initialTokens);
        
        if (pagesNeeded > freePages.cardinality()) {
            throw new IllegalStateException("Not enough free pages. Required: " + pagesNeeded + 
                ", Available: " + freePages.cardinality());
        }
        
        List<Integer> pages = new ArrayList<>();
        for (int i = 0; i < pagesNeeded; i++) {
            int freeIndex = freePages.nextSetBit(0);
            if (freeIndex < 0) {
                for (int pageId : pages) {
                    freePages.set(pageId);
                }
                throw new IllegalStateException("Not enough contiguous free pages");
            }
            
            freePages.clear(freeIndex);
            pages.add(freeIndex);
        }
        
        sequenceToPages.put(sequenceId, pages);
        return sequenceId;
    }
    
    public void appendTokens(int sequenceId, int numTokens) {
        ensureOpen();
        validateSequence(sequenceId);
        
        if (numTokens <= 0) {
            throw new IllegalArgumentException("numTokens must be positive: " + numTokens);
        }
        
        List<Integer> pages = sequenceToPages.get(sequenceId);
        KVPage lastPage = pagePool.get(pages.get(pages.size() - 1));
        
        int availableSpace = lastPage.availableSpace();
        
        if (numTokens > availableSpace) {
            int remainingTokens = numTokens - availableSpace;
            int additionalPages = calculatePagesNeeded(remainingTokens);
            
            if (additionalPages > freePages.cardinality()) {
                throw new IllegalStateException("Not enough free pages for append. Required: " + 
                    additionalPages + ", Available: " + freePages.cardinality());
            }
            
            for (int i = 0; i < additionalPages; i++) {
                int freeIndex = freePages.nextSetBit(0);
                if (freeIndex >= 0) {
                    freePages.clear(freeIndex);
                    pages.add(freeIndex);
                }
            }
        }
    }
    
    public Tensor getKeyCache(int sequenceId, int layerIdx) {
        ensureOpen();
        validateSequence(sequenceId);
        
        List<Integer> pages = sequenceToPages.get(sequenceId);
        int totalTokens = getTotalTokens(sequenceId);
        
        if (totalTokens == 0) {
            return null;
        }
        
        int numHeads = config.numHeads();
        int headDim = config.headDim();
        
        float[] allKeys = new float[totalTokens * numHeads * headDim];
        int offset = 0;
        
        for (int pageId : pages) {
            KVPage page = pagePool.get(pageId);
            float[] pageKeys = page.getKeys(layerIdx);
            int copyLen = Math.min(pageKeys.length, allKeys.length - offset);
            System.arraycopy(pageKeys, 0, allKeys, offset, copyLen);
            offset += copyLen;
        }
        
        return Tensor.fromArray(backend, allKeys).reshape(
            Shape.of(1, numHeads, totalTokens, headDim)
        );
    }
    
    public Tensor getValueCache(int sequenceId, int layerIdx) {
        ensureOpen();
        validateSequence(sequenceId);
        
        List<Integer> pages = sequenceToPages.get(sequenceId);
        int totalTokens = getTotalTokens(sequenceId);
        
        if (totalTokens == 0) {
            return null;
        }
        
        int numHeads = config.numHeads();
        int headDim = config.headDim();
        
        float[] allValues = new float[totalTokens * numHeads * headDim];
        int offset = 0;
        
        for (int pageId : pages) {
            KVPage page = pagePool.get(pageId);
            float[] pageValues = page.getValues(layerIdx);
            int copyLen = Math.min(pageValues.length, allValues.length - offset);
            System.arraycopy(pageValues, 0, allValues, offset, copyLen);
            offset += copyLen;
        }
        
        return Tensor.fromArray(backend, allValues).reshape(
            Shape.of(1, numHeads, totalTokens, headDim)
        );
    }
    
    public void updateKeyCache(int sequenceId, int layerIdx, Tensor keys) {
        ensureOpen();
        validateSequence(sequenceId);
        Objects.requireNonNull(keys, "Keys tensor cannot be null");
    }
    
    public void updateValueCache(int sequenceId, int layerIdx, Tensor values) {
        ensureOpen();
        validateSequence(sequenceId);
        Objects.requireNonNull(values, "Values tensor cannot be null");
    }
    
    public void releaseSequence(int sequenceId) {
        ensureOpen();
        validateSequence(sequenceId);
        
        List<Integer> pages = sequenceToPages.remove(sequenceId);
        if (pages != null) {
            for (int pageId : pages) {
                pagePool.get(pageId).clear();
                freePages.set(pageId);
            }
        }
    }
    
    public int getSequenceLength(int sequenceId) {
        ensureOpen();
        validateSequence(sequenceId);
        
        List<Integer> pages = sequenceToPages.get(sequenceId);
        if (pages == null || pages.isEmpty()) {
            return 0;
        }
        
        return getTotalTokens(sequenceId);
    }
    
    public int freePageCount() {
        return freePages.cardinality();
    }
    
    public int usedPageCount() {
        return config.numPages() - freePages.cardinality();
    }
    
    public double memoryUtilization() {
        return (double) usedPageCount() / config.numPages();
    }
    
    public int activeSequenceCount() {
        return sequenceToPages.size();
    }
    
    private int calculatePagesNeeded(int tokens) {
        int pageSize = config.pageSize();
        return (tokens + pageSize - 1) / pageSize;
    }
    
    private int getTotalTokens(int sequenceId) {
        List<Integer> pages = sequenceToPages.get(sequenceId);
        if (pages == null || pages.isEmpty()) {
            return 0;
        }
        
        return pages.size() * config.pageSize();
    }
    
    private void validateSequence(int sequenceId) {
        if (!sequenceToPages.containsKey(sequenceId)) {
            throw new IllegalArgumentException("Unknown sequence ID: " + sequenceId);
        }
    }
    
    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("PagedAttention has been closed");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            
            for (KVPage page : pagePool) {
                page.close();
            }
            pagePool.clear();
            sequenceToPages.clear();
        }
    }
    
    @Override
    public String toString() {
        if (closed) {
            return "PagedAttention{closed=true}";
        }
        
        return "PagedAttention{" +
            "numPages=" + config.numPages() +
            ", pageSize=" + config.pageSize() +
            ", freePages=" + freePageCount() +
            ", activeSequences=" + activeSequenceCount() +
            ", memoryUtil=" + String.format("%.1f%%", memoryUtilization() * 100) +
            '}';
    }
    
    private static final class KVPage {
        private final int pageId;
        private final int pageSize;
        private final int numHeads;
        private final int headDim;
        private final int numLayers;
        
        private final List<Tensor> keyTensors;
        private final List<Tensor> valueTensors;
        private int tokenCount;
        
        KVPage(int pageId, int pageSize, int numHeads, int headDim, ComputeBackend backend) {
            this.pageId = pageId;
            this.pageSize = pageSize;
            this.numHeads = numHeads;
            this.headDim = headDim;
            this.numLayers = 32;
            
            this.keyTensors = new ArrayList<>();
            this.valueTensors = new ArrayList<>();
            this.tokenCount = 0;
        }
        
        int pageId() { return pageId; }
        int availableSpace() { return pageSize - tokenCount; }
        int tokenCount() { return tokenCount; }
        
        float[] getKeys(int layerIdx) {
            return new float[pageSize * numHeads * headDim];
        }
        
        float[] getValues(int layerIdx) {
            return new float[pageSize * numHeads * headDim];
        }
        
        void clear() {
            tokenCount = 0;
            keyTensors.clear();
            valueTensors.clear();
        }
        
        void close() {
            clear();
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static final class Builder {
        private ComputeBackend backend;
        private PagedAttentionConfig config = PagedAttentionConfig.defaults();
        
        public Builder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }
        
        public Builder config(PagedAttentionConfig config) {
            this.config = config;
            return this;
        }
        
        public Builder pageSize(int pageSize) {
            this.config = new PagedAttentionConfig(
                pageSize, config.numPages(), config.numHeads(), 
                config.headDim(), config.numLayers()
            );
            return this;
        }
        
        public Builder numPages(int numPages) {
            this.config = new PagedAttentionConfig(
                config.pageSize(), numPages, config.numHeads(),
                config.headDim(), config.numLayers()
            );
            return this;
        }
        
        public PagedAttention build() {
            Objects.requireNonNull(backend, "Backend cannot be null");
            return new PagedAttention(backend, config);
        }
    }
}
