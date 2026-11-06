package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.head.GpuRowTopK;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.utils.Closeables;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** DFlash2's selector and greedy lattice walk on the device; reads only selected ids and optional probabilities. */
final class GpuDflashSelector implements AutoCloseable {
    private final DflashSpec spec;
    private final GpuOps ops;
    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final List<AutoCloseable> owned = new ArrayList<>();
    private final GpuWeight hidden;
    private final GpuBuffer prev;
    private final GpuBuffer next;
    private final GpuBuffer inverse;
    private final GpuBuffer logits;
    private final GpuBuffer gate;
    private final GpuBuffer scores;
    private final GpuBuffer ids;
    private final GpuBuffer probs;
    private final GpuRowTopK topK;

    GpuDflashSelector(DflashSpec spec, GpuOps ops, ComputeBackend backend, CpuWeights host) {
        this.spec = spec;
        this.ops = ops;
        this.backend = backend;
        this.queue = ops.queue();
        try {
            int rank = spec.selectorRank();
            this.prev = matrix(host, TensorNames.DFLASH_SELECTOR_PREV, spec.nVocab(), rank).shard(0);
            this.next = matrix(host, TensorNames.DFLASH_SELECTOR_NEXT, spec.nVocab(), rank).shard(0);
            this.hidden = matrix(host, TensorNames.DFLASH_SELECTOR_HIDDEN, rank, spec.nEmbd());
            this.inverse = inverse(host);
            this.logits = floats((long) spec.blockSize() * spec.nVocab());
            this.gate = floats((long) spec.blockSize() * rank);
            this.scores = floats(spec.selectorTopK());
            this.ids = floats(spec.blockSize());
            this.probs = floats(spec.blockSize());
            this.topK = GpuRowTopK.bind(ops, backend, logits, spec.blockSize(), spec.nVocab(), spec.selectorTopK());
            if (topK == null) {
                throw new UnsupportedOperationException("DFlash2 selector needs LLM_Row_TopK on " + backend.backendType());
            }
            owned.add(topK);
        } catch (RuntimeException e) {
            RuntimeException failure = Closeables.closeAll(owned, null);
            if (failure != null) e.addSuppressed(failure);
            throw e;
        }
    }

    /** Runs the scored rows of this block, including target-vocabulary ordering after the d2t scatter. */
    int[] select(GpuBuffer rawLogits, GpuBuffer tEmbd, int nRows, int anchor, float pMin) {
        int rows = Math.min(nRows, spec.graphBlockSize());
        if (rows <= 1) return new int[0];
        int k = spec.selectorTopK();
        enqueue("LLM_Dflash2Logits", (long) rows * spec.nVocab(), rows, spec.nVocabDraft(), spec.nVocab(),
                spec.logitScale(), spec.finalSoftcap(), inverse, rawLogits, logits);
        ops.matmul(hidden, tEmbd, 0, gate, 0, rows);
        topK.run(rows);
        for (int p = 1; p < rows; p++) {
            enqueue("LLM_Dflash2Score", k, p, k, spec.selectorRank(), anchor, topK.topVal(), topK.topIdx(), gate,
                    prev, next, ids, scores);
            enqueue("LLM_Dflash2Pick", 1, p, k, pMin > 0.0f ? 1 : 0, topK.topIdx(), scores, ids, probs);
        }
        byte[] raw = new byte[(rows - 1) * Integer.BYTES];
        float[] probability = pMin > 0.0f ? new float[rows - 1] : null;
        GpuEvent readIds = queue.enqueueReadBuffer(ids, Integer.BYTES, raw.length, raw);
        GpuEvent readProbs = probability == null ? null
                : queue.enqueueReadBuffer(probs, Float.BYTES, probability.length * 4L, probability);
        try {
            queue.finish();
        } finally {
            close(readIds);
            close(readProbs);
        }
        int[] result = new int[rows - 1];
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int n = 0;
        while (n < result.length && (probability == null || !(probability[n] < pMin))) {
            result[n] = bb.getInt(n * Integer.BYTES);
            n++;
        }
        return n == result.length ? result : Arrays.copyOf(result, n);
    }

    /** Bytes beside the backbone: F32 selector tables, target rows, inverse map, gates and readout buffers. */
    static long bytes(DflashSpec spec) {
        long b = spec.blockSize();
        long k = spec.selectorTopK();
        long r = spec.selectorRank();
        return Float.BYTES * (2L * spec.nVocab() * r + r * spec.nEmbd() + spec.nVocab()
                + b * (spec.nVocab() + r + 3 + 2 * k) + k);
    }

    private GpuBuffer inverse(CpuWeights host) {
        int[] map = spec.hasD2t() ? DflashDrafter.readD2t(host, spec.nVocab()) : null;
        ByteBuffer values = ByteBuffer.allocate(Math.multiplyExact(spec.nVocab(), Integer.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        int[] inverseMap = new int[spec.nVocab()];
        Arrays.fill(inverseMap, -1);
        for (int i = 0; i < spec.nVocabDraft(); i++) inverseMap[map == null ? i : map[i]] = i;
        for (int i : inverseMap) values.putInt(i);
        GpuBuffer buffer = floats(spec.nVocab());
        close(queue.enqueueWriteBuffer(buffer, 0, values.capacity(), values.array()));
        return buffer;
    }

    private GpuWeight matrix(CpuWeights host, String name, int rows, int cols) {
        DflashMatrix matrix = DflashMatrix.require(host, TensorNames.weight(name)).expect(rows, cols);
        GpuBuffer buffer = floats((long) rows * cols);
        float[] values = new float[Math.multiplyExact(rows, cols)];
        for (int i = 0; i < rows; i++) matrix.row(i, values, i * cols);
        close(queue.enqueueWriteBuffer(buffer, 0, values.length * 4L, values));
        return GpuWeight.of(name, GGMLType.F32, new long[]{cols, rows}, buffer);
    }

    private GpuBuffer floats(long n) {
        GpuBuffer buffer = backend.allocateBuffer(Math.multiplyExact(n, Float.BYTES));
        owned.add(buffer);
        return buffer;
    }

    private void enqueue(String name, long n, Object... args) {
        int local = ops.local();
        ops.enqueue(name, new long[]{(n + local - 1) / local * local}, new long[]{local}, args);
    }

    private static void close(GpuEvent event) { if (event != null) event.close(); }

    @Override
    public void close() {
        RuntimeException failure = Closeables.closeAll(owned, null);
        owned.clear();
        if (failure != null) throw failure;
    }
}
