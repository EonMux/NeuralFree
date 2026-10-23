package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Common interface for LLM model implementations (CPU and GPU).
 *
 * <p>The scalar shape getters ({@link #headCount()}, {@link #headCountKv()}, {@link #feedForwardLength()},
 * {@link #headDim()}, {@link #kvDim()}, {@link #rmsNormEps()}, {@link #ropeFreqBase()}) describe layer 0 of the
 * trunk; models with per-layer shapes expose the full description through {@link #spec()}.
 */
public interface LLMModel extends Closeable {

    int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP) throws IOException;

    int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                   TokenCallback callback) throws IOException;

    /**
     * Generate tokens with repetition penalty.
     * @param repetitionPenalty penalty factor for repeated tokens (1.0 = no penalty, 1.1 = default)
     * @param repeatWindow number of recent tokens to consider for penalty (0 = all generated tokens)
     */
    default int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                           float repetitionPenalty, int repeatWindow,
                           TokenCallback callback) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, callback);
    }

    /**
     * Generate tokens with full sampling parameter control matching llama.cpp.
     * Sampling chain order: penalties -> top_k -> top_p -> min_p -> temperature -> sample.
     *
     * @param repetitionPenalty penalty factor for repeated tokens (1.0 = disabled)
     * @param repeatWindow number of recent tokens to consider for penalty (0 = all)
     * @param minP minimum probability threshold relative to max (0.0 = disabled, 0.05 = default)
     * @param topK number of top candidates to keep (0 = disabled, 40 = default)
     */
    default int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                           float repetitionPenalty, int repeatWindow, float minP, int topK,
                           TokenCallback callback) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP,
                repetitionPenalty, repeatWindow, callback);
    }

    int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP) throws IOException;

    int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                             TokenCallback callback) throws IOException;

    /**
     * Generate with prompt and repetition penalty.
     */
    default int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                     float repetitionPenalty, int repeatWindow,
                                     TokenCallback callback) throws IOException {
        return generateWithPrompt(prompt, maxNewTokens, temperature, topP, callback);
    }

    void resetKVCache();

    GGUFTokenizer tokenizer();

    void printInfo();

    String architecture();

    String name();

    int vocabSize();

    int contextLength();

    int kvCacheContext();

    int embeddingLength();

    int layerCount();

    /** Attention heads of trunk layer 0. */
    int headCount();

    /** KV heads of trunk layer 0. */
    int headCountKv();

    /** Feed-forward width of trunk layer 0. */
    int feedForwardLength();

    /** Norm epsilon of trunk layer 0. */
    float rmsNormEps();

    /** RoPE base frequency of trunk layer 0. */
    float ropeFreqBase();

    /** Attention head dimension of trunk layer 0. */
    int headDim();

    /** K/V row width of trunk layer 0. */
    int kvDim();

    /**
     * Pooling strategy for encoder-only models (BERT family). Mirrors
     * llama.cpp's {@code llama_pooling_type}.
     */
    enum PoolingType {
        NONE,  // return all per-token hidden states
        MEAN,  // average across sequence
        CLS,   // use first token only (BERT-style)
        LAST,  // use last token (some retrieval models)
        RANK   // ranking head (not supported)
    }

    /**
     * True when the model is an encoder (BERT-family) producing embeddings
     * rather than next-token logits. Default: false (decoder-only).
     */
    default boolean isEncoder() {
        return false;
    }

    /**
     * True when the model is an encoder-decoder (T5/mT5 family) — generate()
     * consumes source tokens via the encoder and emits NEW tokens only
     * (result does NOT include the source/prompt). Default: false.
     */
    default boolean isEncoderDecoder() {
        return false;
    }

    /**
     * Pooling type applied to the final hidden states when {@link #encode(int[])}
     * is called. Only meaningful when {@link #isEncoder()} is true.
     */
    default PoolingType poolingType() {
        return PoolingType.NONE;
    }

    /**
     * Run a single bidirectional forward pass over the supplied tokens and
     * return the pooled embedding (shape depends on {@link #poolingType()}):
     * <ul>
     *   <li>CLS / MEAN / LAST → {@code embeddingLength}</li>
     *   <li>NONE → {@code tokens.length * embeddingLength} (concatenated)</li>
     * </ul>
     * Only valid for encoder-only models. Decoder models throw.
     */
    default float[] encode(int[] tokens) throws IOException {
        throw new UnsupportedOperationException(
                "encode() is only supported for encoder-only (BERT-family) models; "
                        + "this is a decoder-only model (" + architecture() + ")");
    }

    // ---------------------------------------------------------------------------------------------
    // Engine API
    // ---------------------------------------------------------------------------------------------

    /**
     * The resolved model specification (layers, memory kind, embedding and head stages).
     *
     * @throws UnsupportedOperationException for implementations not built on a {@link ModelSpec}
     */
    default ModelSpec spec() {
        throw new UnsupportedOperationException(
                "no resolved ModelSpec is available from " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /** What this model can be used for; derived from {@link #isEncoder()} and {@link #isEncoderDecoder()}. */
    default SupportClass supportClass() {
        if (isEncoderDecoder()) {
            return SupportClass.ENC_DEC;
        }
        return isEncoder() ? SupportClass.EMBED : SupportClass.GEN;
    }

    /**
     * Generates new tokens after {@code prompt} as described by {@code request}, streaming decoded text to
     * {@code callback} (may be null).
     *
     * <p>This default adapts the legacy {@link #generate(int[], int, float, float, float, int, float, int,
     * TokenCallback)} API: it maps top-k, top-p, min-p, temperature (0 when greedy) and the repetition penalty with
     * its window; it rejects request features that API cannot honour; it streams through a stop-string filter
     * and computes the result tokens, text and stop string by replaying the generated tokens through the
     * tokenizer, so the result is cut exactly after the token that completed the stop string. The legacy loop
     * reports no stop reason, so it is inferred: the request's token budget reached gives
     * {@link StopReason#MAX_TOKENS}, a filled KV cache gives {@link StopReason#CONTEXT_FULL}, and any other end is
     * {@link StopReason#EOS}. Timings are reported as a single total.
     *
     * @return the generated tokens (prompt excluded), their text, the stop reason and timings
     * @throws ContextFullException          when a decoder prompt alone exceeds {@link #kvCacheContext()}, or an
     *                                       encoder-decoder source exceeds {@link #contextLength()}
     * @throws UnsupportedOperationException when the request needs typical-p, presence or frequency penalties, a
     *                                       repetition penalty below 1, a fixed seed for non-greedy sampling,
     *                                       ignore-eos or cancellation, which the legacy API cannot express
     */
    default GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback)
            throws IOException {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        if (prompt.length == 0) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
        List<String> unsupported = legacyUnsupported(request);
        if (!unsupported.isEmpty()) {
            throw new UnsupportedOperationException(getClass().getSimpleName() + " (" + architecture()
                    + ") offers only the legacy generate API, which cannot honour: " + String.join(", ", unsupported));
        }
        boolean encDec = isEncoderDecoder();
        int capacity = kvCacheContext();
        if (encDec && prompt.length > contextLength()) {
            throw new ContextFullException(prompt.length, contextLength());
        }
        if (!encDec && prompt.length > capacity) {
            throw new ContextFullException(prompt.length, capacity);
        }
        // Legacy decoder loops stop once kvPos = prompt.length - 1 + generated reaches the capacity;
        // the encoder-decoder loop stops after `capacity` decoder positions.
        int room = encDec ? capacity : capacity - prompt.length + 1;
        int budget = request.unlimited() ? room : Math.min(request.maxNewTokens(), room);

        SamplerConfig s = request.sampler();
        float temperature = s.isGreedy() ? 0f : s.temperature();
        boolean penalize = s.penaltiesEnabled();
        float penalty = penalize ? s.repeatPenalty() : 1f;
        int window = penalize && s.penaltyLastN() > 0 ? s.penaltyLastN() : 0;
        GenerationRequest.StopStringFilter live = callback != null && !request.stopStrings().isEmpty()
                ? request.newStopFilter(callback) : null;

        long start = System.nanoTime();
        int[] out = generate(prompt, budget, temperature, s.topP(), penalty, window, s.minP(), s.topK(),
                live != null ? live : callback);
        long elapsed = System.nanoTime() - start;
        if (live != null) {
            live.finish();
        }

        int[] generated = encDec ? out : Arrays.copyOfRange(out, prompt.length, out.length);
        GGUFTokenizer.StreamingDecoder decoder = tokenizer().createStreamingDecoder();
        GenerationRequest.StopStringFilter replay = request.newStopFilter(null);
        int kept = generated.length;
        for (int i = 0; i < generated.length && !replay.stopped(); i++) {
            replay.onToken(generated[i], decoder.decodeToken(generated[i]));
            if (replay.stopped()) {
                kept = i + 1;
            }
        }
        if (!replay.stopped()) {
            replay.onToken(-1, decoder.flush());
            replay.finish();
        }

        StopReason reason;
        int used = encDec ? generated.length : prompt.length - 1 + generated.length;
        if (replay.stopped()) {
            reason = StopReason.STOP_STRING;
        } else if (!request.unlimited() && generated.length >= request.maxNewTokens()) {
            reason = StopReason.MAX_TOKENS;
        } else if (used >= capacity) {
            reason = StopReason.CONTEXT_FULL;
        } else {
            reason = StopReason.EOS;
        }
        return new GenerationResult(Arrays.copyOf(generated, kept), replay.releasedText(), reason, replay.matched(),
                GenerationResult.Timings.total(prompt.length, generated.length, elapsed));
    }

    /**
     * {@link #encode(int[])} with BERT segment ids.
     *
     * <p>The default accepts only all-zero (or null) token types, which is what {@link #encode(int[])} applies.
     *
     * @throws IllegalArgumentException      when {@code tokenTypes} and {@code tokens} differ in length
     * @throws UnsupportedOperationException when a token type other than 0 is given
     */
    default float[] encode(int[] tokens, int[] tokenTypes) throws IOException {
        Objects.requireNonNull(tokens, "tokens");
        if (tokenTypes != null) {
            if (tokenTypes.length != tokens.length) {
                throw new IllegalArgumentException(tokenTypes.length + " token types for " + tokens.length + " tokens");
            }
            for (int t : tokenTypes) {
                if (t != 0) {
                    throw new UnsupportedOperationException("token type " + t + " is not supported by "
                            + getClass().getSimpleName() + " (" + architecture() + "); only type 0 is");
                }
            }
        }
        return encode(tokens);
    }

    /**
     * Per-token embeddings: {@code [tokens.length][embeddingLength]}.
     *
     * <p>The default splits {@link #encode(int[])} for encoders whose pooling is {@link PoolingType#NONE}.
     *
     * @throws UnsupportedOperationException when the model has no per-token embedding output
     */
    default float[][] encodeTokens(int[] tokens) throws IOException {
        Objects.requireNonNull(tokens, "tokens");
        if (!isEncoder() || poolingType() != PoolingType.NONE) {
            throw new UnsupportedOperationException("per-token embeddings not supported by "
                    + getClass().getSimpleName() + " (" + architecture()
                    + (isEncoder() ? ", pooling " + poolingType() : ", not an encoder") + ")");
        }
        float[] flat = encode(tokens);
        int width = embeddingLength();
        if (flat.length != (long) tokens.length * width) {
            throw new IllegalStateException("encode returned " + flat.length + " floats for " + tokens.length
                    + " tokens of width " + width);
        }
        float[][] rows = new float[tokens.length][];
        for (int t = 0; t < rows.length; t++) {
            rows[t] = Arrays.copyOfRange(flat, t * width, (t + 1) * width);
        }
        return rows;
    }

    /**
     * Classifier head outputs for one input ({@code cls} / {@code cls.output}, decision heads), one score per
     * {@link #classLabels()} entry.
     *
     * @throws UnsupportedOperationException when the model has no classifier head
     */
    default float[] classify(int[] tokens) throws IOException {
        throw new UnsupportedOperationException(
                "classification not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /** Classifier output labels ({@code classifier.output_labels}); empty when the model has none. */
    default String[] classLabels() {
        return new String[0];
    }

    /**
     * Relevance score of {@code document} for {@code query} with a ranking head (RANK pooling).
     *
     * @throws UnsupportedOperationException when the model has no ranking head
     */
    default float rerank(int[] query, int[] document) throws IOException {
        throw new UnsupportedOperationException(
                "reranking not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * Relevance score of {@code document} for {@code query} with a ranking head (RANK pooling), over the rerank
     * input the model builds from the text: its {@code tokenizer.chat_template.rerank} prompt when it has one,
     * else the query and document packed with the vocabulary's special tokens.
     *
     * @throws UnsupportedOperationException when the model has no ranking head
     */
    default float rerank(String query, String document) throws IOException {
        throw new UnsupportedOperationException(
                "reranking not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * Masked-diffusion generation: returns the generated tokens after {@code prompt}, streaming the final text to
     * {@code callback} (may be null).
     *
     * @throws UnsupportedOperationException when the model is not a diffusion model
     */
    default int[] generateDiffusion(int[] prompt, DiffusionConfig config, TokenCallback callback) throws IOException {
        throw new UnsupportedOperationException(
                "diffusion generation not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * Decodes audio codec codes into waveform frames (wavtokenizer-dec).
     *
     * @throws UnsupportedOperationException when the model is not an audio decoder
     */
    default float[][] decodeAudioCodes(int[] codes) throws IOException {
        throw new UnsupportedOperationException(
                "audio code decoding not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /** Request features the legacy generate API has no parameter for. */
    private static List<String> legacyUnsupported(GenerationRequest request) {
        SamplerConfig s = request.sampler();
        List<String> out = new ArrayList<>();
        if (s.typicalP() != 1f) {
            out.add("typical-p " + s.typicalP());
        }
        if (s.penaltyLastN() != 0 && (s.presencePenalty() != 0f || s.frequencyPenalty() != 0f)) {
            out.add("presence/frequency penalties");
        }
        if (s.penaltyLastN() != 0 && s.repeatPenalty() < 1f) {
            out.add("repetition penalty " + s.repeatPenalty() + " below 1");
        }
        if (s.seeded() && !s.isGreedy()) {
            out.add("fixed seed " + s.seed());
        }
        if (request.ignoreEos()) {
            out.add("ignore-eos");
        }
        if (request.cancellable()) {
            out.add("cancellation");
        }
        return out;
    }
}
