package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.diag.GpuProfiler;
import com.eonmux.neuralfree.gguf.gpu.weights.LayerWeightSet;
import com.eonmux.neuralfree.gguf.gpu.weights.StreamingLayerWeightSource;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Decode benchmarks of one open {@link GGUFModelGPU}: a single observed decode under a {@link GpuProfiler}, the
 * per-position profile with unobserved throughput, and the transfer/decode comparison of a streamed model. The
 * profiler observes every device the model runs on through that device's own observer slot, so the report gives
 * each stage's device; the streaming comparison covers every device that streams. The bodies of the model's public
 * benchmark methods; they report on the model and leave its memory empty.
 */
final class GpuBenchmarks {

    private static final double MIB = 1024.0 * 1024.0;

    private final GGUFModelGPU model;

    GpuBenchmarks(GGUFModelGPU model) {
        this.model = model;
    }

    /**
     * Decodes {@code tokenId} at {@code position} under a {@link GpuProfiler} and prints the per-stage times; the
     * memory is reset and filled with {@code position} copies of the token first.
     *
     * @param quiet true to time without printing (warm-up)
     * @throws UnsupportedOperationException when the model has no decoder memory
     */
    void profileDecodeToken(int tokenId, int position, boolean quiet) {
        ForwardBackend f = model.requireDecoder("profileDecodeToken");
        if (position < 0 || position >= model.memoryPlan().nCtx()) {
            throw new IllegalArgumentException("position " + position + " is outside the context [0, "
                    + model.memoryPlan().nCtx() + ")");
        }
        f.reset();
        if (position > 0) {
            f.prefill(repeated(tokenId, position), 0);
        }
        GpuProfiler profiler = new GpuProfiler(model.computeQueue(0));
        for (int d = 0; d < model.deviceCount(); d++) {
            model.observerSlot(d).use(profiler.device(model.deviceName(d), model.computeQueue(d)));
        }
        try {
            f.decode(tokenId, position);
        } finally {
            for (int d = 0; d < model.deviceCount(); d++) {
                model.observerSlot(d).restore();
            }
        }
        if (!quiet) {
            GpuProfiler.Report r = profiler.report();
            System.out.print(r.format("decode at position " + position, r.nanos(GpuProfiler.STEP)));
        }
    }

    /**
     * Profiles the decode at several positions, then times unobserved decode throughput. Call it before
     * {@code generate}; it leaves the memory empty.
     */
    void runBenchmark(int tokenId) {
        System.out.println("\n=== GPU DECODE BENCHMARK ===");
        System.out.println("Model: " + model.name() + " (" + model.architecture() + "), " + model.layerCount()
                + " layers, embed " + model.embeddingLength() + ", heads " + model.headCount() + "/"
                + model.headCountKv() + ", vocab " + model.vocabSize());
        if (!model.hasDecoderMemory()) {
            System.out.println("  " + model.architecture() + " has no decoder memory; there is no decode step to"
                    + " time.");
            return;
        }
        for (int w = 0; w < 3 && w < model.memoryPlan().nCtx(); w++) {
            profileDecodeToken(tokenId, w, true);
        }
        for (int pos : new int[]{0, 10, 50, 100, 500}) {
            if (pos >= model.memoryPlan().nCtx()) {
                break;
            }
            profileDecodeToken(tokenId, pos, false);
            System.out.println();
        }
        int primed = Math.min(10, model.memoryPlan().nCtx() / 2);
        int timed = Math.min(20, model.memoryPlan().nCtx() - primed);
        System.out.println("=== THROUGHPUT (unobserved, " + timed + " tokens) ===");
        long ns = timeDecode(tokenId, primed, timed);
        System.out.printf(Locale.ROOT, "  %d tokens in %.1f ms = %.2f ms/token = %.1f tok/s%n", timed, ns / 1e6,
                ns / 1e6 / timed, timed * 1e9 / ns);
        System.out.println("========================================\n");
    }

    /**
     * Compares the per-token weight transfer of a streamed model with its streamed decode: the time to make every
     * streamed layer resident (transfer only) against the decode of {@code tokenId}. Models whose weights are
     * resident, or whose streamed layers all fit the slots, stream nothing per token and report that instead.
     */
    void benchmarkStreamingOverlap(int tokenId) {
        System.out.println("\n=== STREAMING OVERLAP BENCHMARK ===");
        System.out.println("Model: " + model.name() + " (" + model.architecture() + "), " + model.layerCount()
                + " layers");
        boolean streams = false;
        boolean perToken = false;
        for (int d = 0; d < model.deviceCount(); d++) {
            StreamingLayerWeightSource streaming = model.streamingOf(d);
            if (streaming == null) {
                continue;
            }
            streams = true;
            perToken |= streaming.depth() < streaming.streamedLayers();
            System.out.println("  " + model.deviceName(d) + ": " + streaming.streamedLayers()
                    + " streamed layers through " + streaming.depth() + " slots");
        }
        if (!streams) {
            System.out.println("  Layer weights are resident in VRAM; nothing streams per token.");
        } else if (!perToken) {
            System.out.println("  Every streamed layer fits its device's slots; after warm-up nothing streams per"
                    + " token.");
        } else if (!model.hasDecoderMemory() || model.memoryPlan().nCtx() < 4) {
            System.out.println("  " + model.architecture() + " has no decode step to time.");
        } else {
            overlap(tokenId);
        }
        System.out.println("========================================\n");
    }

    private void overlap(int tokenId) {
        long bytesPerToken = 0L;
        for (int d = 0; d < model.deviceCount(); d++) {
            bytesPerToken += streamedBytes(model.streamedSetsOf(d));
        }
        finishAll();
        long t0 = System.nanoTime();
        for (int d = 0; d < model.deviceCount(); d++) {
            StreamingLayerWeightSource streaming = model.streamingOf(d);
            for (LayerWeightSet s : model.streamedSetsOf(d)) {
                streaming.await(s.weightLayer());
            }
        }
        finishAll();
        long transferNs = System.nanoTime() - t0;
        int warm = Math.min(2, model.memoryPlan().nCtx() / 2);
        int timed = Math.min(10, model.memoryPlan().nCtx() - warm);
        long decodeNs = timeDecode(tokenId, warm, timed) / timed;
        double transferMs = transferNs / 1e6;
        double decodeMs = decodeNs / 1e6;
        System.out.printf(Locale.ROOT, "  %-30s %10.1f MB%n", "Streamed weights per token:", bytesPerToken / MIB);
        System.out.printf(Locale.ROOT, "  %-30s %10.3f ms  (%.2f GB/s)%n", "Transfer only (per token):",
                transferMs, transferNs > 0 ? bytesPerToken / (double) transferNs : 0.0);
        System.out.printf(Locale.ROOT, "  %-30s %10.3f ms  (%d tokens)%n",
                "Streamed decode (per token):", decodeMs, timed);
        System.out.printf(Locale.ROOT, "  %-30s %10.1f%%%n", "Transfer share of decode:",
                decodeNs > 0 ? 100.0 * transferNs / decodeNs : 0.0);
        System.out.printf(Locale.ROOT, "  %-30s %10.3f ms%n", "Decode beyond transfer:",
                Math.max(0.0, decodeMs - transferMs));
    }

    /** Fills {@code primed} positions, then returns the nanoseconds of {@code timed} decodes; leaves memory empty. */
    private long timeDecode(int tokenId, int primed, int timed) {
        ForwardBackend f = model.requireDecoder("benchmark");
        f.reset();
        try {
            if (primed > 0) {
                f.prefill(repeated(tokenId, primed), 0);
            }
            finishAll();
            long t0 = System.nanoTime();
            for (int i = 0; i < timed; i++) {
                f.decode(tokenId, primed + i);
            }
            finishAll();
            return Math.max(1L, System.nanoTime() - t0);
        } finally {
            f.reset();
        }
    }

    /** Waits for the queue of every device. */
    private void finishAll() {
        for (int d = 0; d < model.deviceCount(); d++) {
            model.computeQueue(d).finish();
        }
    }

    private static int[] repeated(int token, int n) {
        int[] t = new int[n];
        Arrays.fill(t, token);
        return t;
    }

    private static long streamedBytes(List<LayerWeightSet> sets) {
        long sum = 0;
        for (LayerWeightSet s : sets) {
            sum += s.uploadBytes();
        }
        return sum;
    }
}
