package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.Observers;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;

import java.util.Objects;

/**
 * The {@link ForwardBackend} wrapper that drives the runtime expert rearrangement ({@link EplbRebalancer}): every
 * forward pass of the wrapped backend counts one step and lands on {@link EplbRebalancer#afterStep}, which
 * rearranges the experts on its schedule. Everything else — queries, memory mirroring, closing — goes to the
 * delegate untouched, and the memory and hidden tap see each call exactly as without the wrapper. One wrapper per
 * forward call, so an entry that the interface routes through a primitive counts once.
 */
public final class EplbStepBackend implements ForwardBackend {

    private final ForwardBackend delegate;
    private final EplbRebalancer rebalancer;

    public EplbStepBackend(ForwardBackend delegate, EplbRebalancer rebalancer) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.rebalancer = Objects.requireNonNull(rebalancer, "rebalancer");
    }

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        float[] out = delegate.prefill(tokens, pos0);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[] prefillEmbeddings(float[][] rows, int pos0) {
        float[] out = delegate.prefillEmbeddings(rows, pos0);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[] decode(int token, int pos) {
        float[] out = delegate.decode(token, pos);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        int out = delegate.decodeGreedyOnDevice(token, pos);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        float[][] out = delegate.forwardRows(tokens, pos0, outputRows);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        int[] out = delegate.forwardRowsGreedy(tokens, pos0, outputRows);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows) {
        RowTopK out = delegate.forwardRowsTopK(tokens, pos0, k, outputRows);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows, LogitPrefilter prefilter) {
        RowTopK out = delegate.forwardRowsTopK(tokens, pos0, k, outputRows, prefilter);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k) {
        RowTopK out = delegate.decodeTopKOnDevice(token, pos, k);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k, LogitPrefilter prefilter) {
        RowTopK out = delegate.decodeTopKOnDevice(token, pos, k, prefilter);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[][] forwardBatch(ForwardBatch batch) {
        float[][] out = delegate.forwardBatch(batch);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public int[] forwardBatchGreedy(ForwardBatch batch) {
        int[] out = delegate.forwardBatchGreedy(batch);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public RowTopK forwardBatchTopK(ForwardBatch batch, int k) {
        RowTopK out = delegate.forwardBatchTopK(batch, k);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public RowTopK forwardBatchTopK(ForwardBatch batch, int k, LogitPrefilter prefilter) {
        RowTopK out = delegate.forwardBatchTopK(batch, k, prefilter);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[][] hiddenBatch(ForwardBatch batch) {
        float[][] out = delegate.hiddenBatch(batch);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        float[][] out = delegate.forwardAll(tokens, outputRows);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        float[][] out = delegate.hiddenRows(tokens, tokenTypes);
        rebalancer.afterStep();
        return out;
    }

    @Override
    public int maxSequences() {
        return delegate.maxSequences();
    }

    @Override
    public String batchRefusal() {
        return delegate.batchRefusal();
    }

    @Override
    public String embeddingRowsRefusal() {
        return delegate.embeddingRowsRefusal();
    }

    @Override
    public SequenceEncoder sequenceEncoder() {
        return delegate.sequenceEncoder();
    }

    @Override
    public boolean batchTopKOnDevice() {
        return delegate.batchTopKOnDevice();
    }

    @Override
    public boolean batchPrefilterOnDevice() {
        return delegate.batchPrefilterOnDevice();
    }

    @Override
    public boolean topKOnDevice() {
        return delegate.topKOnDevice();
    }

    @Override
    public boolean topKPrefilterOnDevice() {
        return delegate.topKPrefilterOnDevice();
    }

    @Override
    public Observers observers() {
        return delegate.observers();
    }

    @Override
    public void reset() {
        delegate.reset();
    }

    @Override
    public boolean canRollback(int nTokens) {
        return delegate.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        delegate.rollback(nTokens);
    }

    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        delegate.shiftContext(nKeep, nDiscard);
    }

    @Override
    public String shiftSequenceRefusal() {
        return delegate.shiftSequenceRefusal();
    }

    @Override
    public void shiftSequence(int seq, int nKeep, int nDiscard) {
        delegate.shiftSequence(seq, nKeep, nDiscard);
    }

    @Override
    public String sequenceEditRefusal() {
        return delegate.sequenceEditRefusal();
    }

    @Override
    public void addSequence(int seq, int p0, int p1, int delta) {
        delegate.addSequence(seq, p0, p1, delta);
    }

    @Override
    public void divSequence(int seq, int p0, int p1, int d) {
        delegate.divSequence(seq, p0, p1, d);
    }

    @Override
    public ModelMemory memory() {
        return delegate.memory();
    }

    @Override
    public void setHiddenTap(HiddenTap tap) {
        delegate.setHiddenTap(tap);
    }

    @Override
    public void close() {
        delegate.close();
    }

    @Override
    public String toString() {
        return "EplbStepBackend{" + rebalancer + "}";
    }
}
