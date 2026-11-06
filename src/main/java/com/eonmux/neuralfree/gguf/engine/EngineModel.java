package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.io.IOException;

/**
 * An {@link LLMModel} built on the spec-driven engine: it exposes its resolved {@link ModelSpec} and the
 * {@link ForwardBackend} that runs it, so drafters, the speculative decoder and the heads can drive the forward
 * passes directly.
 */
public interface EngineModel extends LLMModel {

    /** The backend running this model's forward passes. */
    ForwardBackend forwardBackend();

    /**
     * The host view of the model's weights: the token embedding table and the output norm that speech generation
     * reads beside the forward backend. Both engines of this package implement it over the model file; a device
     * engine that keeps no host view refuses by name.
     *
     * @throws UnsupportedOperationException when the model has no host access to its weights
     * @throws java.io.IOException          when a device engine's host view cannot be opened
     */
    default com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights cpuWeights() throws java.io.IOException {
        throw new UnsupportedOperationException(architecture() + " runs its weights on a device; its host weights"
                + " are not reachable");
    }

    /**
     * One row of the model's token embedding table ({@code spec().embedding().nEmbd()} floats): the prompt rows the
     * speech generator feeds the forward pass. The returned array belongs to the caller. The default reads the host
     * weights ({@link #cpuWeights()}); a device engine may override it with a device readback.
     *
     * @throws UnsupportedOperationException when the model has no token embedding table or no host access to it
     * @throws IllegalArgumentException      when {@code token} is outside the table
     */
    default float[] embeddingRow(int token) throws java.io.IOException {
        TensorRef table = spec().embedding().tokenEmbd();
        if (table == null) {
            throw new UnsupportedOperationException(architecture() + " has no token embedding table; speech"
                    + " generation reads one");
        }
        long rows = table.ne(1);
        if (token < 0 || token >= rows) {
            throw new IllegalArgumentException("token " + token + " is outside the " + rows + "-row embedding table");
        }
        int width = (int) table.ne(0);
        float[] row = new float[width];
        com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights weights = cpuWeights();
        weights.embedRow(weights.get(table), token, row, 0);
        return row;
    }

    /**
     * The output norm weights ({@code n_embd} floats) that the speech generator applies to the backbone's last hidden
     * row. The returned array belongs to the caller. The default reads the host weights ({@link #cpuWeights()}); a
     * device engine may override it.
     *
     * @throws UnsupportedOperationException when the model has no output norm or no host access to it
     */
    default float[] outputNormWeights() throws java.io.IOException {
        HeadSpec head = spec().head();
        if (head.outputNorm() == null) {
            throw new UnsupportedOperationException(architecture() + " has no output norm; speech generation"
                    + " normalizes the backbone's last hidden row with one");
        }
        return cpuWeights().floats(head.outputNorm()).clone();
    }

    /**
     * A quantized copy of this model's own projection matrices opened as a second forward backend over this
     * model's memory, the target view of two-level verification ({@code neuralfree.draft.tlv}): it verifies a
     * draft tree greedily and commits nothing, while this model's passes stay exact. The returned view owns only
     * what was built beside the model; the caller closes it with the session.
     *
     * @param bits the copy's bits per weight, 6 ({@code Q6_K} rows) or 8 ({@code Q8_0} rows)
     * @throws UnsupportedOperationException when the model provides no quantized view, or its weights and shape
     *                                       cannot serve one (naming {@code neuralfree.draft.tlv})
     * @throws IllegalArgumentException      naming {@code neuralfree.draft.tlv.bits} for a width that is not 6 or 8
     * @throws java.io.IOException          when the copy's bytes cannot be read
     */
    default QuantizedView openQuantizedView(int bits) throws java.io.IOException {
        throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " opens a quantized copy of the"
                + " target's own weights beside them, which " + architecture() + " does not provide; two-level"
                + " verification runs over a GGUF target");
    }

    /** The resolved model specification. */
    @Override
    ModelSpec spec();

    /**
     * The pooling operator of the model's pooling and classifier heads (vLLM's pooler over a batched step), or
     * null when the model has neither: the batch engine's pooling requests apply it to the hidden rows their
     * prefill computed. Used on the model's engine thread.
     */
    default PoolingOperator poolingOperator() {
        return null;
    }

    /** The support class of the resolved architecture. */
    @Override
    default SupportClass supportClass() {
        return spec().supportClass();
    }

    /**
     * Why this model fell back from a GPU load to the CPU engine ({@code LLMModelFactory.open}, the failure meaning
     * the device cannot serve the model), or null when it opened on the requested backend — every GPU engine, and a
     * CPU engine whose configuration never asked for one. The demos' startup lines and the server's {@code /props}
     * report it, so a silent half-speed fallback is visible with its cause.
     */
    default String gpuFallbackReason() {
        return null;
    }

    /**
     * The sampling defaults the model carries ({@code general.sampling.*}, {@code neuralfree.sampling.*}), which
     * generation applies to every setting the caller left unset unless
     * {@link EngineConfig#modelSamplingDefaults()} is off. This default has none.
     */
    default SamplingDefaults modelSamplingDefaults() {
        return SamplingDefaults.NONE;
    }

    /**
     * The registry of the model's per-request LoRA adapters (vLLM {@code /v1/load_lora_adapter}, llama.cpp
     * {@code /lora-adapters}): adapters loaded into it run only for the requests that select them
     * ({@code GenerationRequest.lora}), in the adapter slots of a batched forward. Owned by the model and used on
     * its engine thread. Null when the model serves no per-request adapters; this default has none.
     */
    default LoraRegistry loraRegistry() {
        return null;
    }

    /**
     * The model's live weight updates (vLLM's in-process weight transfer engine): a trainer writes new tensor
     * contents over the weights this model serves, without reloading it. Run the updater on the model's own thread,
     * between forward passes. This default refuses by name; the GGUF engines of one host or one device implement
     * it for the tensors they hold.
     *
     * @throws UnsupportedOperationException when the model takes no live weight updates, naming the architecture
     */
    default WeightUpdater weightUpdater() {
        throw new UnsupportedOperationException(architecture() + " takes no live weight updates (no in-process"
                + " weight transfer for it)");
    }

    /**
     * Writes the model's single-sequence session state to {@code out} ({@code SessionFile.save}, llama.cpp's
     * {@code llama_state_save_file}): the stored tokens and every KV group's rows and recurrent state, over either
     * backend, so a later {@link #loadSession} into a model of the same plan continues the sequence exactly. This
     * default refuses by name.
     *
     * @throws UnsupportedOperationException when the model carries no session state, or its memory is outside the
     *                                       session file's scope, naming it
     */
    default void saveSession(java.nio.file.Path out) throws IOException {
        throw new UnsupportedOperationException("session save refused: " + spec().arch().ggufName()
                + " carries no session state");
    }

    /**
     * Reads the session file at {@code in} into the model's memory, resetting it first ({@code SessionFile.load},
     * llama.cpp's {@code llama_state_load_file}). This default refuses by name.
     *
     * @throws UnsupportedOperationException as {@link #saveSession}
     * @throws IOException                   for an unreadable file, a foreign format or a plan mismatch
     */
    default void loadSession(java.nio.file.Path in) throws IOException {
        throw new UnsupportedOperationException("session load refused: " + spec().arch().ggufName()
                + " carries no session state");
    }
}
