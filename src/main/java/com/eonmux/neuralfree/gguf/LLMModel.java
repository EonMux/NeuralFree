package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

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
        RANK   // ranking classifier over the pooled row
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

    /**
     * The pooled output row of every token sequence of {@code sequences} (MEAN / CLS / LAST pooling, the RANK
     * classifier scores), through the dense projections: llama.cpp's parallel-sequence embedding batch, one output
     * row per sequence in order, each sequence embedded from position 0 on its own. The default embeds the sequences
     * one after another through {@link #encode(int[])}; an implementation over a paged memory runs them as
     * multi-sequence forwards packed to the memory's sequences ({@code neuralfree.parallel}) and context.
     *
     * @throws UnsupportedOperationException for NONE pooling, whose output is one embedding per token with no row
     *                                       per sequence ({@link #encodeTokens(int[])})
     * @throws IllegalArgumentException      when the batch or one of its sequences is empty
     * @throws IOException                   when the model fails to encode
     */
    default float[][] encodeBatch(int[][] sequences) throws IOException {
        PagedEvaluations.checkSequences(sequences);
        if (poolingType() == PoolingType.NONE) {
            throw new UnsupportedOperationException("batched encode pools one row per sequence; "
                    + architecture() + " pools by NONE and embeds per token");
        }
        float[][] out = new float[sequences.length][];
        for (int i = 0; i < sequences.length; i++) {
            out[i] = encode(sequences[i]);
        }
        return out;
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
     *                                       ignore-eos, cancellation, {@code stop_token_ids},
     *                                       {@code allowed_token_ids}, {@code skip_special_tokens} false or
     *                                       {@code repetition_detection}, which the legacy API cannot express
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
     * Generates new tokens after the precomputed embedding rows of {@code promptEmbeds} as described by
     * {@code request}, streaming decoded text to {@code callback} (may be null): one row per context position
     * (vLLM completions {@code prompt_embeds}), each of the model's {@code n_embd} values. The rows replace the
     * token-table lookup; no prompt token enters the penalty history, since none carries an id.
     *
     * @return the generated tokens, their text, the stop reason and timings
     * @throws ContextFullException          when the rows exceed the context
     * @throws UnsupportedOperationException when this model cannot take embedding rows (the legacy generate API
     *                                       without an engine backend, or a backend without an embedding-row
     *                                       prefill)
     */
    default GenerationResult generateEmbeds(float[][] promptEmbeds, GenerationRequest request, TokenCallback callback)
            throws IOException {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " (" + architecture()
                + ") does not take precomputed prompt embeddings; generate from token ids instead");
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

    /**
     * Classifier outputs of one input for a decision head that reads an order (vllm-decision2, clef): output
     * row {@code i} holds the score of the {@code i}-th option span of {@code order} (llama's decision
     * forward, {@code llama_batch_ext_set_decision_order}); a head given an order it cannot use answers NaN.
     * The implementations are {@code GGUFModel} and {@code GGUFModelGPU}.
     *
     * @param order one decision code per token ({@code ClassifierHead.DecisionOrder}), or null
     * @throws UnsupportedOperationException when the model has no decision head
     * @throws IllegalArgumentException      when {@code order}'s length differs from the token count
     */
    default float[] classify(int[] tokens, int[] order) throws IOException {
        throw new UnsupportedOperationException(
                "decision classification not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * Per-token classifier scores of one input ({@code cls} / {@code cls.output} applied to every token): one row
     * per token, one score per classifier output (vLLM's {@code token_classify}).
     *
     * @throws UnsupportedOperationException when the model has no token classifier
     */
    default float[][] tokenClassify(int[] tokens) throws IOException {
        throw new UnsupportedOperationException(
                "token classification not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * The classifier output of one input: the sequence classifier's scores with their labels and the head's
     * normalized flag ({@code ClassifierOutput}). Decision models refuse; their per-token scores come from
     * {@link #classify(int[])}.
     *
     * @throws UnsupportedOperationException when the model has no sequence classifier, or is a decision model
     */
    default ClassifierOutput classifyOutput(int[] tokens) throws IOException {
        throw new UnsupportedOperationException(
                "classification not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * {@link #classifyOutput(int[])} with the classifier's final probability activation applied or skipped, as
     * vLLM's pooler applies or skips it per request ({@code use_activation}); {@code false} answers the raw head
     * output, which is not normalized.
     *
     * @throws UnsupportedOperationException when the model has no sequence classifier, or is a decision model
     */
    default ClassifierOutput classifyOutput(int[] tokens, boolean activation) throws IOException {
        if (!activation) {
            throw new UnsupportedOperationException(
                    "classification not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
        }
        return classifyOutput(tokens);
    }

    /**
     * The pooled output of one media prompt (MEAN / CLS / LAST pooling, the sequence classifier scores): the prompt's
     * text and media rows pool as one sequence, the media rows where the projector placed them (vLLM's pooling
     * runners over a multimodal prompt). A paged memory runs the chunks as consecutive batch forwards; any other
     * memory refuses by name.
     *
     * @throws UnsupportedOperationException when the model does not pool media prompts (the implementations:
     *                                       {@code GGUFModel}, {@code GGUFModelGPU})
     */
    default float[] encodeMedia(MediaPrompt prompt) throws IOException {
        throw new UnsupportedOperationException(
                "media prompts not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * The classifier output of one media prompt: the sequence classifier's scores over the pooled row of the prompt's
     * text and media rows (vLLM's {@code JinaVLForSequenceClassification} reranker over a media prompt). The same
     * memory rule as {@link #encodeMedia(MediaPrompt)}.
     *
     * @throws UnsupportedOperationException when the model has no sequence classifier over media prompts
     */
    default ClassifierOutput classifyOutputMedia(MediaPrompt prompt) throws IOException {
        throw new UnsupportedOperationException(
                "media prompts not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * {@link #classifyOutputMedia(MediaPrompt)} with the classifier's final probability activation applied or
     * skipped, as vLLM's pooler applies or skips it per request ({@code use_activation}); {@code false} answers
     * the raw head output, which is not normalized.
     *
     * @throws UnsupportedOperationException when the model has no RANK classifier over media prompts
     */
    default ClassifierOutput classifyOutputMedia(MediaPrompt prompt, boolean activation) throws IOException {
        if (!activation) {
            throw new UnsupportedOperationException(
                    "media prompts not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
        }
        return classifyOutputMedia(prompt);
    }

    /** Classifier output labels ({@code classifier.output_labels}); empty when the model has none. */
    default String[] classLabels() {
        return new String[0];
    }

    /**
     * The next-token logits over the whole vocabulary after {@code tokens} (one prompt of a generative decoder),
     * as vLLM's generative scoring reads them.
     *
     * @throws UnsupportedOperationException when the model does not generate token logits
     */
    default float[] nextLogits(int[] tokens) throws IOException {
        throw new UnsupportedOperationException(
                "next-token logits not supported by " + getClass().getSimpleName() + " (" + architecture() + ")");
    }

    /**
     * The next-token logits over the whole vocabulary after the last row of a media {@code prompt}: the prompt's
     * text and media rows run as consecutive forwards and the last row's logits are read (a decision read over
     * images or video). The memory must be paged.
     *
     * @throws UnsupportedOperationException when the model does not generate token logits, or the memory is not
     *                                       paged
     */
    default float[] nextLogits(MediaPrompt prompt) throws IOException {
        throw new UnsupportedOperationException(
                "media next-token logits not supported by " + getClass().getSimpleName() + " (" + architecture()
                        + ")");
    }

    /**
     * Classifier outputs of one media prompt for a decision head that reads an order (the vllm-decision2 and clef
     * heads, and the modern-bert / lfm2 decision blocks): the prompt's text and media rows run as consecutive
     * forwards and the head reads the resulting hidden rows. The memory must be paged.
     *
     * @param order one decision code per prompt row, or null
     * @throws UnsupportedOperationException when the model has no decision head, or the memory is not paged
     */
    default float[] classifyMedia(MediaPrompt prompt, int[] order) throws IOException {
        throw new UnsupportedOperationException(
                "media decision classification not supported by " + getClass().getSimpleName() + " ("
                        + architecture() + ")");
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

    /**
     * The speech decoding geometry the model declares: {@code n_fft}, {@code hop_length} and the sample rate of
     * its spectral frames ({@link SpeechFormat}); null when the model cannot turn its features into a waveform.
     */
    default SpeechFormat speechFormat() {
        return null;
    }

    /**
     * Synthesizes the waveform of audio codec codes: the spectral features of {@link #decodeAudioCodes} through
     * the model's declared speech geometry ({@link #speechFormat}).
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     * @throws IllegalArgumentException      when a code is outside the codebook
     */
    default AudioClip synthesizeSpeech(int[] codes) {
        throw new UnsupportedOperationException("speech synthesis not supported by " + getClass().getSimpleName()
                + " (" + architecture() + ")");
    }

    /**
     * The speech decoding geometry, refusing like {@link #synthesizeSpeech(int[])} when the model has none: a
     * caller that needs the sample rate before it decodes (a streaming route writes its WAV header first) reads
     * it here and receives the same refusal a synthesis would.
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    default SpeechFormat requireSpeechFormat() {
        SpeechFormat format = speechFormat();
        if (format == null) {
            throw new UnsupportedOperationException("speech synthesis not supported by "
                    + getClass().getSimpleName() + " (" + architecture() + ")");
        }
        return format;
    }

    /**
     * {@link #synthesizeSpeech(int[])} delivered in chunks: every clip holds consecutive samples of the one-shot
     * result at the same rate, and concatenating them gives that result. The default delivers the one-shot clip.
     *
     * @throws UnsupportedOperationException as {@link #synthesizeSpeech(int[])}
     * @throws IllegalArgumentException      as {@link #synthesizeSpeech(int[])}
     */
    default void synthesizeSpeech(int[] codes, Consumer<AudioClip> sink) {
        Objects.requireNonNull(sink, "sink");
        sink.accept(synthesizeSpeech(codes));
    }

    /** The per-request token controls the legacy generate API has no parameter for, by request field. */
    private static List<String> legacyUnsupportedTokenControls(GenerationRequest request) {
        List<String> out = new ArrayList<>();
        if (request.stopTokenIds().length > 0) {
            out.add("stop_token_ids");
        }
        if (request.allowedTokenIds().length > 0) {
            out.add("allowed_token_ids");
        }
        if (!request.skipSpecialTokens()) {
            out.add("skip_special_tokens false");
        }
        if (request.repetitionDetection().enabled()) {
            out.add("repetition_detection");
        }
        return out;
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
        out.addAll(legacyUnsupportedTokenControls(request));
        return out;
    }
}
