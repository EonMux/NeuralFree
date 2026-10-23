package com.eonmux.neuralfree.inference;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public final class ContinuousBatcher implements AutoCloseable {
    private final Module model;
    private final ContinuousBatcherConfig config;
    private final BlockingQueue<InferenceRequest> requestQueue;
    private final Map<String, InferenceRequest> activeRequests;
    private final PriorityQueue<InferenceRequest> priorityQueue;
    
    private final AtomicInteger requestIdCounter = new AtomicInteger(0);
    private final AtomicLong totalTokensGenerated = new AtomicLong(0);
    private final AtomicLong totalRequestsProcessed = new AtomicLong(0);
    
    private volatile boolean running = true;
    private volatile boolean closed = false;
    
    public ContinuousBatcher(Module model, ContinuousBatcherConfig config) {
        Objects.requireNonNull(model, "Model cannot be null");
        Objects.requireNonNull(config, "Config cannot be null");
        
        this.model = model;
        this.config = config;
        this.requestQueue = new LinkedBlockingQueue<>(config.maxQueueSize());
        this.activeRequests = new ConcurrentHashMap<>();
        this.priorityQueue = new PriorityQueue<>(
            Comparator.comparingInt(InferenceRequest::priority).reversed()
        );
    }
    
    public InferenceRequest submit(float[] inputTokens, int maxNewTokens) {
        return submit(inputTokens, maxNewTokens, 0);
    }
    
    public InferenceRequest submit(float[] inputTokens, int maxNewTokens, int priority) {
        ensureOpen();
        ensureRunning();
        Objects.requireNonNull(inputTokens, "Input tokens cannot be null");
        
        if (inputTokens.length == 0) {
            throw new IllegalArgumentException("Input tokens cannot be empty");
        }
        if (maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be positive: " + maxNewTokens);
        }
        
        String requestId = generateRequestId();
        InferenceRequest request = new InferenceRequest(
            requestId, inputTokens, maxNewTokens, priority, System.currentTimeMillis()
        );
        
        if (!requestQueue.offer(request)) {
            throw new IllegalStateException("Request queue is full (max: " + config.maxQueueSize() + ")");
        }
        
        return request;
    }
    
    public InferenceRequest submitAsync(float[] inputTokens, int maxNewTokens, int priority) {
        return submit(inputTokens, maxNewTokens, priority);
    }
    
    public BatchResult processBatch() {
        ensureOpen();
        ensureRunning();
        
        List<InferenceRequest> batch = new ArrayList<>();
        
        int batchSize = Math.min(config.maxBatchSize(), requestQueue.size());
        for (int i = 0; i < batchSize; i++) {
            InferenceRequest request = requestQueue.poll();
            if (request != null) {
                batch.add(request);
                activeRequests.put(request.requestId(), request);
            }
        }
        
        if (batch.isEmpty()) {
            return BatchResult.empty();
        }
        
        int maxInputLen = 0;
        int maxSeqLen = 0;
        for (InferenceRequest req : batch) {
            maxInputLen = Math.max(maxInputLen, req.inputTokens().length);
            maxSeqLen = Math.max(maxSeqLen, req.currentTokens().length + req.remainingTokens());
        }
        
        int[][] paddedInputs = padInputs(batch, maxInputLen);
        int[][] attentionMasks = createAttentionMasks(batch, maxInputLen);
        
        List<GenerationResult> results = new ArrayList<>();
        for (InferenceRequest request : batch) {
            try {
                float[] currentTokens = request.currentTokens();
                int remaining = request.remainingTokens();
                
                if (remaining > 0 && currentTokens.length < maxSeqLen) {
                    float[] newToken = generateNextToken(currentTokens);
                    request.appendToken(newToken[0]);
                    totalTokensGenerated.incrementAndGet();
                    
                    if (request.remainingTokens() == 0 || 
                        currentTokens.length >= config.maxSeqLen()) {
                        request.complete();
                        results.add(new GenerationResult(
                            request.requestId(),
                            request.outputTokens(),
                            true,
                            null
                        ));
                        activeRequests.remove(request.requestId());
                        totalRequestsProcessed.incrementAndGet();
                    }
                }
            } catch (Exception e) {
                request.fail(e.getMessage());
                results.add(new GenerationResult(
                    request.requestId(),
                    request.outputTokens(),
                    false,
                    e.getMessage()
                ));
                activeRequests.remove(request.requestId());
            }
        }
        
        return new BatchResult(results, batch.size());
    }
    
    private float[] generateNextToken(float[] currentTokens) {
        float nextToken = (float) (Math.random() * 100);
        return new float[] { nextToken };
    }
    
    private int[][] padInputs(List<InferenceRequest> batch, int maxLen) {
        int[][] padded = new int[batch.size()][maxLen];
        for (int i = 0; i < batch.size(); i++) {
            float[] input = batch.get(i).inputTokens();
            int len = Math.min(input.length, maxLen);
            for (int j = 0; j < len; j++) {
                padded[i][maxLen - len + j] = (int) input[j];
            }
        }
        return padded;
    }
    
    private int[][] createAttentionMasks(List<InferenceRequest> batch, int maxLen) {
        int[][] masks = new int[batch.size()][maxLen];
        for (int i = 0; i < batch.size(); i++) {
            float[] input = batch.get(i).inputTokens();
            int len = Math.min(input.length, maxLen);
            for (int j = maxLen - len; j < maxLen; j++) {
                masks[i][j] = 1;
            }
        }
        return masks;
    }
    
    public void processUntilEmpty() {
        ensureOpen();
        ensureRunning();
        
        while (!requestQueue.isEmpty() || !activeRequests.isEmpty()) {
            processBatch();
        }
    }
    
    public void cancel(String requestId) {
        ensureOpen();
        Objects.requireNonNull(requestId, "Request ID cannot be null");
        
        activeRequests.remove(requestId);
        requestQueue.removeIf(req -> req.requestId().equals(requestId));
    }
    
    public void cancelAll() {
        ensureOpen();
        requestQueue.clear();
        activeRequests.clear();
    }
    
    public int queueSize() {
        return requestQueue.size();
    }
    
    public int activeCount() {
        return activeRequests.size();
    }
    
    public long totalTokensGenerated() {
        return totalTokensGenerated.get();
    }
    
    public long totalRequestsProcessed() {
        return totalRequestsProcessed.get();
    }
    
    public void pause() {
        running = false;
    }
    
    public void resume() {
        ensureOpen();
        running = true;
    }
    
    public boolean isRunning() {
        return running && !closed;
    }
    
    private String generateRequestId() {
        return "req_" + requestIdCounter.incrementAndGet();
    }
    
    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ContinuousBatcher has been closed");
        }
    }
    
    private void ensureRunning() {
        if (!running) {
            throw new IllegalStateException("ContinuousBatcher is paused");
        }
    }
    
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            running = false;
            requestQueue.clear();
            activeRequests.clear();
        }
    }
    
    @Override
    public String toString() {
        return "ContinuousBatcher{" +
            "queueSize=" + requestQueue.size() +
            ", activeCount=" + activeRequests.size() +
            ", running=" + running +
            ", closed=" + closed +
            '}';
    }
    
    public static final class InferenceRequest {
        private final String requestId;
        private final float[] inputTokens;
        private final int maxNewTokens;
        private final int priority;
        private final long submitTime;
        
        private final List<Float> generatedTokens;
        private volatile boolean completed = false;
        private volatile boolean failed = false;
        private volatile String errorMessage;
        
        InferenceRequest(String requestId, float[] inputTokens, int maxNewTokens, 
                        int priority, long submitTime) {
            this.requestId = requestId;
            this.inputTokens = inputTokens.clone();
            this.maxNewTokens = maxNewTokens;
            this.priority = priority;
            this.submitTime = submitTime;
            this.generatedTokens = new ArrayList<>();
        }
        
        public String requestId() { return requestId; }
        public float[] inputTokens() { return inputTokens.clone(); }
        public int maxNewTokens() { return maxNewTokens; }
        public int priority() { return priority; }
        public long submitTime() { return submitTime; }
        
        public float[] currentTokens() {
            float[] current = new float[inputTokens.length + generatedTokens.size()];
            System.arraycopy(inputTokens, 0, current, 0, inputTokens.length);
            for (int i = 0; i < generatedTokens.size(); i++) {
                current[inputTokens.length + i] = generatedTokens.get(i);
            }
            return current;
        }
        
        public float[] outputTokens() {
            return currentTokens();
        }
        
        public int remainingTokens() {
            return maxNewTokens - generatedTokens.size();
        }
        
        public boolean isCompleted() { return completed; }
        public boolean isFailed() { return failed; }
        public String errorMessage() { return errorMessage; }
        
        void appendToken(float token) {
            if (!completed && !failed) {
                generatedTokens.add(token);
            }
        }
        
        void complete() {
            this.completed = true;
        }
        
        void fail(String error) {
            this.failed = true;
            this.errorMessage = error;
            this.completed = true;
        }
    }
    
    public static final class GenerationResult {
        private final String requestId;
        private final float[] tokens;
        private final boolean success;
        private final String error;
        
        GenerationResult(String requestId, float[] tokens, boolean success, String error) {
            this.requestId = requestId;
            this.tokens = tokens != null ? tokens.clone() : new float[0];
            this.success = success;
            this.error = error;
        }
        
        public String requestId() { return requestId; }
        public float[] tokens() { return tokens.clone(); }
        public boolean success() { return success; }
        public String error() { return error; }
    }
    
    public static final class BatchResult {
        private final List<GenerationResult> results;
        private final int batchSize;
        
        BatchResult(List<GenerationResult> results, int batchSize) {
            this.results = results != null ? new ArrayList<>(results) : new ArrayList<>();
            this.batchSize = batchSize;
        }
        
        static BatchResult empty() {
            return new BatchResult(null, 0);
        }
        
        public List<GenerationResult> results() { return new ArrayList<>(results); }
        public int batchSize() { return batchSize; }
        public boolean isEmpty() { return results.isEmpty(); }
    }
}
