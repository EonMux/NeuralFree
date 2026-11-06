package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.diag.TensorTraceObserver;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;

import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * The stage trace shared by {@code nftool debug} and {@code nftool eval-callback}: the prompt's forward pass with one
 * {@link TensorTraceObserver} per forward device, each reading its own device's stages on its own queue, installed
 * for the call and removed afterwards.
 *
 * <p>The trace reads the GPU execution stages, the engine's counterpart of llama.cpp's graph-node callback
 * ({@code common_debug_cb_eval}); a model that opened on the CPU backend has no stage trace and is refused by name.
 * A model split over several devices reports every stage with its device's name.</p>
 */
final class ForwardTrace {

    private ForwardTrace() {
    }

    /**
     * Runs the prompt and prints the stages that pass the filters.
     *
     * @param steps how many forward steps to report, {@code 0} for every one
     * @throws IOException                   when the model has no GPU stage trace
     * @throws TensorTraceObserver.NonFiniteException when a stage sums to NaN and {@code abortOnNan} is set
     */
    static void trace(EngineModel model, int[] tokens, List<String> filters, boolean abortOnNan, int steps,
                      PrintStream out) throws IOException {
        if (!(model instanceof GGUFModelGPU)) {
            throw new IOException(noTrace(model));
        }
        GGUFModelGPU gpu = (GGUFModelGPU) model;
        int devices = gpu.deviceCount();
        List<TensorTraceObserver> observers = new ArrayList<>(devices);
        List<ExecutionObserver> previous = new ArrayList<>(devices);
        for (int d = 0; d < devices; d++) {
            observers.add(new TensorTraceObserver(gpu.computeQueue(d), out, filters, abortOnNan, steps,
                    gpu.deviceName(d)));
            previous.add(gpu.executionObserver(d));
        }
        try {
            install(gpu, observers);
            ForwardBackend backend = model.forwardBackend();
            backend.reset();
            backend.prefill(tokens, 0);
        } finally {
            restore(gpu, previous);
        }
    }

    /** The refusal of a model without a GPU stage trace, naming what to use instead. */
    static String noTrace(EngineModel model) {
        return "the per-op tensor trace reads the GPU execution stages, but " + model.architecture()
                + " opened on a backend without one (the CPU forward path has no stage trace); select a GPU backend"
                + " or use --save-logits to dump logits/embeddings";
    }

    private static void install(GGUFModelGPU gpu, List<TensorTraceObserver> observers) {
        for (int d = 0; d < observers.size(); d++) {
            gpu.setExecutionObserver(d, observers.get(d));
        }
    }

    private static void restore(GGUFModelGPU gpu, List<ExecutionObserver> previous) {
        for (int d = 0; d < previous.size(); d++) {
            gpu.setExecutionObserver(d, previous.get(d));
        }
    }
}
