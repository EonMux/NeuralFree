package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.batch.PromptLogprobs;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSession;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.engine.sampling.RequestStages;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.grammar.ConstraintCompiler;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import com.eonmux.neuralfree.utils.Closeables;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Autoregressive generation over a {@link ForwardBackend}: prompt prefill, then sample, emit and decode one token
 * at a time (the token loop of llama.cpp tools/completion/completion.cpp). Immutable; one generation runs at a time, on
 * the backend's thread.
 *
 * <p>Per {@link #generate} call:
 * <ol>
 *   <li><b>Model defaults.</b> The request takes the loop's {@link SamplingDefaults} for every sampler setting and
 *       the token budget it leaves unset ({@link GenerationRequest#withModelDefaults}).</li>
 *   <li><b>Memory.</b> With prefix reuse (default on, {@code neuralfree.prefix.reuse}), the longest common prefix
 *       of the stored tokens and the prompt stays (at most {@code prompt.length - 1} tokens, so the last prompt
 *       token is evaluated for logits) when the memory can drop the rest ({@link ForwardBackend#canRollback});
 *       otherwise, and for encoder-decoders, every call starts with a reset (the stored sequence is cleared with
 *       it).</li>
 *   <li><b>Prefill.</b> A decoder prompt longer than the context throws {@link ContextFullException} before any
 *       work. The unreused prompt tail is prefilled; every prompt token enters the penalty history (llama.cpp
 *       completion accepts prompt tokens into the sampler). A request scoring its prompt rows ({@code
 *       prompt_logprobs}, {@code prompt_logprob_token_ids}) reuses no prefix and reads every prompt row's logits
 *       back ({@code forwardRows}), capturing them like the batch engine does. For encoder-decoder models the history starts with the
 *       decoder tokens the prefill stored (the decoder start token), not the encoder source, as llama.cpp
 *       completion replaces the prompt by the decoder start token after encoding. A penalty window of
 *       {@link SamplerConfig#PENALTY_WHOLE_CONTEXT} covers the last {@code nCtx} tokens. An embedding-row prompt
 *       ({@link EmbedsPrompt}) prefills its token runs and embedding rows in order; its rows carry no ids, so the
 *       penalty history holds its tokens alone.</li>
 *   <li><b>Token loop.</b> Poll cancellation; sample ({@link Sampler}, llama.cpp chain); offer the token to the
 *       {@link SequenceRun} (its {@link StopCriteria} session: end of generation, stop strings, budget); if the
 *       memory is full, shift the context or stop with {@link StopReason#CONTEXT_FULL}; decode the token at the
 *       next position. A chain the device can sample selects there (llama.cpp backend sampling {@code -bs}, default
 *       on, {@code neuralfree.sampling.device}): plain argmax ({@link Sampler#argmaxOnly()}) decodes with
 *       {@link ForwardBackend#decodeGreedyOnDevice}, reading back only the id (on a backend that chains greedy
 *       steps on the device, {@link ForwardBackend#chainsGreedyOnDevice()}, each step is enqueued before the
 *       previous id reaches the host, so the device runs ahead of the host's stop checks); a chain whose selection
 *       stays inside the top-k cone ({@code Sampler#coneWidth} — a truncated chain, or a greedy chain whose bans and
 *       non-positive biases only lower a logit) on a backend with {@code LLM_Row_TopK} decodes with
 *       {@link ForwardBackend#decodeTopKOnDevice} and selects over the device's top candidates, the same token the
 *       host chain selects over the whole row. Every other chain reads the logits row back.</li>
 * </ol>
 * Positions come from the memory's {@link SequenceState}: the next position is {@code nPast}. For encoder-decoder
 * backends, {@code prefill(source, 0)} encodes the source and evaluates the decoder start, and the decoder
 * sequence supplies the positions.
 *
 * <p><b>Context shift</b> (llama.cpp completion.cpp:562-588, enabled by {@code neuralfree.context.shift}): when
 * the next token has no position left, {@code n_keep} positions are kept (1 when the prompt starts with BOS,
 * llama.cpp's {@code n_keep = add_bos}), half of the rest is discarded and later positions move down. A backend
 * whose memory cannot shift (recurrent state, sparse caches, multi-section RoPE, step35) ends generation with
 * {@link StopReason#CONTEXT_FULL}.
 *
 * <p><b>Per-request adapters</b> ({@link GenerationRequest#lora()}): with the model's {@link LoraRegistry}
 * ({@link Builder#adapters}), the request's {@link LoraPlan} acquires its adapter slots for the run
 * ({@link LoraSession}) and every pass runs in them: the prompt rows before an activated LoRA's invocation start in
 * the slot before it, every later row in the adapter's slot (llama.cpp prefills the pre-invocation rows in a batch
 * of their own the same way). Prefix reuse keeps only rows computed under the same adapters
 * ({@link SequenceState#commonPrefix(int[], String)}).</p>
 *
 * <p>When a forward call or the sampler fails, the memory is reset before the exception propagates and the memory's
 * {@link SequenceState} is cleared with it, so the next generation never reuses a half-written sequence; the state
 * is cleared even when the reset itself fails.
 */
public final class GenerationLoop {

    private static final Logger LOGGER = LogManager.getLogger(GenerationLoop.class);
    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    /** Most candidates one device top-k readback carries ({@link RowTopK#MAX_DEVICE_CANDIDATES}). */
    private static final int MAX_DEVICE_TOPK = RowTopK.MAX_DEVICE_CANDIDATES;
    /** Whether a generation's per-step selection runs on the device ({@code neuralfree.sampling.device}). */
    public static final String PROP_SAMPLING_DEVICE = "neuralfree.sampling.device";

    private final ForwardBackend backend;
    private final GGUFTokenizer tokenizer;
    /** The sampler's view of {@link #tokenizer}, shared by every request so its piece caches build once. */
    private final SamplerVocab samplerVocab;
    /** The request constraints compiled over {@link #tokenizer}, remembered across requests. */
    private final ConstraintCompiler constraints;
    private final ReasoningFilter reasoning;
    private final boolean prefixReuse;
    private final boolean contextShift;
    private final boolean encoderDecoder;
    private final SamplingDefaults samplingDefaults;
    private final boolean samplingOnDevice;
    /** Whether the host-sampling fallback of a loop was logged (once per loop). */
    private boolean hostSamplingLogged;
    /** The model's per-request adapters, or null when the loop runs none. */
    private final LoraRegistry adapters;
    /** The custom logits processors of the engine configuration, empty when it configures none. */
    private final LogitsProcessors logitsProcessors;

    private GenerationLoop(Builder b) {
        this.backend = b.backend;
        this.tokenizer = b.tokenizer;
        this.samplerVocab = Sampler.vocab(b.tokenizer);
        this.constraints = new ConstraintCompiler(text -> tokenizer.encode(text, false, true));
        this.reasoning = b.reasoning;
        this.prefixReuse = b.prefixReuse;
        this.contextShift = b.contextShift;
        this.encoderDecoder = b.encoderDecoder;
        this.samplingDefaults = b.samplingDefaults;
        this.samplingOnDevice = b.samplingOnDevice;
        this.adapters = b.adapters;
        this.logitsProcessors = b.logitsProcessors;
    }

    public static Builder builder(ForwardBackend backend, GGUFTokenizer tokenizer) {
        return new Builder(backend, tokenizer);
    }

    /**
     * The loop of an engine model: its backend and tokenizer, the configured reasoning parser
     * ({@link EngineConfig#reasoningParser()}, the chat template's by default, with its tag tokens), prefix reuse and context shift from {@code config}'s memory settings, and the model's sampling
     * defaults unless {@code config} turns them off ({@link SamplingDefaults#forModel}).
     *
     * @throws UnsupportedOperationException for models that do not generate autoregressively (embedding,
     *                                       classifier, diffusion and unsupported architectures)
     */
    public static GenerationLoop forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        SupportClass support = model.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") does not generate text autoregressively");
            case DIFF:
                throw new UnsupportedOperationException(model.architecture()
                        + " is a masked-diffusion model; generate with DiffusionGenerator");
            default:
                break;
        }
        GGUFTokenizer tok = model.tokenizer();
        ReasoningFilter rf = ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), tok);
        return builder(model.forwardBackend(), tok)
                .adapters(model.loraRegistry())
                .reasoning(rf)
                .memoryConfig(config.memory())
                .encoderDecoder(support == SupportClass.ENC_DEC)
                .samplingDefaults(SamplingDefaults.forModel(model, config))
                .samplingOnDevice(deviceSamplingEnabled())
                .logitsProcessors(LogitsProcessors.forConfig(config))
                .build();
    }

    /**
     * {@code neuralfree.sampling.device}: whether a step whose chain keeps inside a device top-k samples over that
     * readback instead of reading whole rows. True when absent; a present value must parse as a boolean. The batch
     * engine reads it when it opens, the generation loop when it is built.
     *
     * @throws IllegalArgumentException when the property holds anything but a boolean
     */
    public static boolean deviceSamplingEnabled() {
        String v = System.getProperty(PROP_SAMPLING_DEVICE);
        if (v == null || v.isBlank()) {
            return true;
        }
        String s = v.toLowerCase(java.util.Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(PROP_SAMPLING_DEVICE + "=" + v + " is not a boolean; use true or false");
    }

    public ReasoningFilter reasoning() { return reasoning; }
    public boolean prefixReuse() { return prefixReuse; }
    public boolean contextShift() { return contextShift; }
    public boolean encoderDecoder() { return encoderDecoder; }
    public SamplingDefaults samplingDefaults() { return samplingDefaults; }

    /**
     * Generates after {@code prompt}, streaming the content channel of the output to {@code callback} (reasoning
     * and tool-call text are parsed out by the reasoning filter; with {@link ReasoningFilter#none()} everything is
     * content). Text released at the end carries token id -1.
     *
     * @param callback receives content text; may be null
     * @return the generated tokens, the raw output text (cut before a matched stop string), stop reason, timings
     * @throws ContextFullException          when a decoder prompt is longer than the context
     * @throws UnsupportedOperationException when the backend has no memory
     * @throws IllegalArgumentException      for an empty prompt, a negative token id, or a model sampling default
     *                                       the sampler rejects
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        return generate(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null);
    }

    /**
     * Generates after the token runs and embedding rows of {@code prompt}, as
     * {@link #generate(int[], GenerationRequest, ReasoningFilter.Sink, String)} does after a token prompt: the
     * rows prefill the positions they occupy (vLLM {@code prompt_embeds}) and no prompt token enters the sampler
     * history, since only the token runs carry ids.
     *
     * @param renderedPrompt the prompt text used to detect a generation prefix that opens reasoning; null renders
     *                       it from the prompt's token ids
     * @throws ContextFullException          when the prompt rows exceed the context
     * @throws UnsupportedOperationException when the backend has no memory or is an encoder-decoder
     * @throws IllegalArgumentException      for an invalid prompt row or a model sampling default the sampler rejects
     */
    public GenerationResult generate(EmbedsPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        requireNoBeamSearch(request);
        ModelMemory memory = backend.memory();
        if (memory == null) {
            throw new UnsupportedOperationException("autoregressive generation needs model memory; this backend"
                    + " runs without a cache (encoder, classifier or diffusion model)");
        }
        if (encoderDecoder) {
            throw new UnsupportedOperationException("encoder-decoder models take token prompts; embedding-row"
                    + " prompts need a decoder");
        }
        SequenceState seq = memory.sequence();
        int nCtx = seq.nCtx();
        if (prompt.rowCount() > nCtx) {
            throw new ContextFullException(prompt.rowCount(), nCtx);
        }
        GenerationRequest effective = request.withModelDefaults(samplingDefaults);
        requireNoPromptLogprobsOverEmbeds(effective);
        effective.requireScoredIds(prompt.rowCount(), tokenizer.vocabSize());
        LoraPlan plan = plan(request.lora(), null);
        SequenceRun seqRun = sequenceRun(effective, sink, renderedPrompt, prompt.allTokens(), nCtx);
        try (LoraSession session = session(plan)) {
            return runEmbeds(prompt, seq, nCtx, seqRun, plan, session);
        } catch (RuntimeException e) {
            discardHalfWritten(seq, e);
            throw e;
        }
    }

    /** {@link #generate(EmbedsPrompt, GenerationRequest, ReasoningFilter.Sink, String)} with a content callback. */
    public GenerationResult generate(EmbedsPrompt prompt, GenerationRequest request, TokenCallback callback) {
        return generate(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null);
    }

    /**
     * Generates after {@code prompt}, routing every piece of output to {@code sink} with its channel.
     *
     * @param renderedPrompt the prompt text (for example the rendered chat template) used to detect a
     *                       generation prefix that opens reasoning; null renders it from the prompt tokens
     * @see #generate(int[], GenerationRequest, TokenCallback)
     */
    public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                     String renderedPrompt) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        requirePrompt(prompt);
        requireNoBeamSearch(request);
        ModelMemory memory = backend.memory();
        if (memory == null) {
            throw new UnsupportedOperationException("autoregressive generation needs model memory; this backend"
                    + " runs without a cache (encoder, classifier or diffusion model)");
        }
        SequenceState seq = memory.sequence();
        int nCtx = seq.nCtx();
        if (!encoderDecoder && prompt.length > nCtx) {
            throw new ContextFullException(prompt.length, nCtx);
        }
        GenerationRequest effective = request.withModelDefaults(samplingDefaults);
        requireNoPromptLogprobsOverEncoder(effective);
        effective.requireScoredIds(prompt.length, tokenizer.vocabSize());
        LoraPlan plan = plan(request.lora(), encoderDecoder ? null : prompt);
        SequenceRun seqRun = sequenceRun(effective, sink, renderedPrompt, prompt, nCtx);
        try (LoraSession session = session(plan)) {
            return run(prompt, seq, nCtx, seqRun, effective, plan, session);
        } catch (RuntimeException e) {
            discardHalfWritten(seq, e);
            throw e;
        }
    }

    /**
     * Beam search runs over the batch engine ({@link
     * com.eonmux.neuralfree.gguf.engine.beam.BeamSearchEngine}), whose steps resubmit every beam's full sequence;
     * this single-sequence loop has no second slot to grow a beam in and refuses a beam request by name.
     */
    private static void requireNoBeamSearch(GenerationRequest request) {
        if (request.beamSearch()) {
            throw new UnsupportedOperationException("beam_width=" + request.beamWidth()
                    + ": the single-sequence generation loop runs no beam search; submit the request to the batch"
                    + " engine's BeamSearchEngine");
        }
    }

    /**
     * An embedding-row prompt holds no token id per row, so a {@code prompt_logprobs} or
     * {@code prompt_logprob_token_ids} request over one has nothing to report; it refuses by name rather than
     * answer an empty capture.
     */
    private static void requireNoPromptLogprobsOverEmbeds(GenerationRequest effective) {
        if (effective.readsPromptRows()) {
            throw new UnsupportedOperationException(effective.promptRowsField() + " needs a token prompt; an"
                    + " embedding-row prompt (prompt_embeds) carries no token id per row to report log"
                    + " probabilities for");
        }
    }

    /**
     * An encoder-decoder prompt is the encoder's source, which no decoder row predicts, so a request that scores
     * its prompt rows over one refuses by name.
     */
    private void requireNoPromptLogprobsOverEncoder(GenerationRequest effective) {
        if (encoderDecoder && effective.readsPromptRows()) {
            throw new UnsupportedOperationException(effective.promptRowsField() + " needs a decoder-only model; an"
                    + " encoder-decoder prompt is the encoder's source, which no decoder row predicts");
        }
    }

    /**
     * The adapter plan of a request's selection over the loop's registry ({@link LoraPlan#resolve}); for a request
     * that selects none, the default plan, or a startup activated LoRA's plan from its invocation start.
     *
     * @param promptTokens the prompt's token ids by position, or null when its rows do not map to them (embedding
     *                     rows, an encoder-decoder source), where an activated LoRA is refused
     * @throws UnsupportedOperationException when the request selects adapters and the loop runs none
     */
    private LoraPlan plan(LoraSelection selection, int[] promptTokens) {
        if (selection == null) {
            return adapters == null ? LoraPlan.DEFAULT : LoraPlan.resolve(adapters, null, promptTokens);
        }
        if (adapters == null) {
            throw new UnsupportedOperationException("lora: this generation loop runs no per-request LoRA adapters ("
                    + selection + "); build it with the model's registry (GenerationLoop.forModel or"
                    + " Builder.adapters)");
        }
        return LoraPlan.resolve(adapters, selection, promptTokens);
    }

    /** The slots of {@code plan} held for one run, or null for the default plan. */
    private LoraSession session(LoraPlan plan) {
        return plan.isDefault() ? null : LoraSession.open(adapters, plan);
    }

    /**
     * The request's output side over the model defaults: its stop criteria, its sampler chain (the constraint
     * head masks first — the grammar stage before the request stages, so a completed constraint's
     * end-of-generation tokens survive the min-tokens ban, the ban yielding when nothing else remains — and the
     * budget force keeps the final say over a masked set) and the rendered prompt the reasoning prefix is
     * detected in, rendered from {@code promptTokens} when null.
     */
    private SequenceRun sequenceRun(GenerationRequest effective, ReasoningFilter.Sink sink, String renderedPrompt,
                                    int[] promptTokens, int nCtx) {
        ReasoningFilter filter = forRequest(reasoning, effective);
        StopCriteria stop = StopCriteria.of(effective, tokenizer, filter);
        String rendered = renderedPrompt != null || filter.mode() == ReasoningFilter.Mode.NONE
                ? renderedPrompt : stop.renderPrompt(promptTokens);
        SamplerVocab vocab = samplerVocab;
        List<SamplerStage> extra = new ArrayList<>();
        SamplerStage grammar = RequestStages.constraintStage(effective.constraint(), constraints, vocab);
        if (grammar != null) {
            extra.add(grammar);
        }
        extra.addAll(RequestStages.of(effective, vocab, filter, rendered));
        Sampler sampler = Sampler.create(samplerConfig(effective.sampler(), nCtx), stop.bannedTokens(),
                effective.logitBias(), extra, logitsProcessors.stages(effective), vocab);
        return SequenceRun.over(effective, tokenizer)
                .stopCriteria(stop)
                .sampler(sampler)
                .sink(sink)
                .renderedPrompt(rendered)
                .build();
    }

    /**
     * The filter {@code request} parses with: {@code configured}'s per-request selection for the request's chat
     * template variables ({@link ReasoningFilter#forRequest}), shared by the loop and the speculative decoder.
     *
     * @throws IllegalArgumentException when a variable the parser reads has the wrong type
     */
    static ReasoningFilter forRequest(ReasoningFilter configured, GenerationRequest request) {
        return configured.forRequest(request.templateVariables());
    }

    private GenerationResult run(int[] prompt, SequenceState seq, int nCtx, SequenceRun seqRun,
                                 GenerationRequest effective, LoraPlan plan, LoraSession session) {
        long t0 = System.nanoTime();
        boolean readPromptRows = effective.readsPromptRows();
        int reused = prepareMemory(prompt, seq, readPromptRows, plan.identity());
        seq.adapters(plan.identity());
        if (readPromptRows) {
            // the batch engine's prompt_logprobs path: read every prompt row's logits back and capture them;
            // no prefix is reused, so the rows cover the whole prompt (row r predicts prompt[r + 1])
            float[][] rows = prefillPromptRows(prompt, reused, plan, session);
            acceptPrompt(prompt, seq, seqRun, reused, t0);
            PromptLogprobs.Collector capture = PromptLogprobs.capture(rows, prompt, effective);
            return capture.attach(decodeLoop(rows[rows.length - 1], seq, nCtx, seqRun, prompt[0]));
        }
        float[] logits = prefillPrompt(prompt, reused, plan, session);
        acceptPrompt(prompt, seq, seqRun, reused, t0);
        return decodeLoop(logits, seq, nCtx, seqRun, prompt[0]);
    }

    /**
     * Enters the prefilled prompt into the penalty history (an encoder-decoder's decoder rows, which the prefill
     * stored) and records the prompt phase that started at {@code t0}.
     */
    private void acceptPrompt(int[] prompt, SequenceState seq, SequenceRun seqRun, int reused, long t0) {
        int[] history = encoderDecoder ? seq.tokens() : prompt;
        seqRun.sampler().acceptPrompt(history, 0, history.length);
        seqRun.prefilled(prompt.length - reused, System.nanoTime() - t0);
    }

    /**
     * Prefills the prompt rows from {@code reused} on and leaves the passes in the plan's adapter slot: the rows
     * before an activated LoRA's invocation start run first in the slot before it.
     */
    private float[] prefillPrompt(int[] prompt, int reused, LoraPlan plan, LoraSession session) {
        if (session == null) {
            return backend.prefill(reused == 0 ? prompt : Arrays.copyOfRange(prompt, reused, prompt.length), reused);
        }
        int split = Math.max(reused, plan.from());
        if (split > reused) {
            session.runPre();
            backend.prefill(Arrays.copyOfRange(prompt, reused, split), reused);
        }
        session.runKey();
        return backend.prefill(Arrays.copyOfRange(prompt, split, prompt.length), split);
    }

    /**
     * {@link #prefillPrompt} reading back the logits of every prompt row from {@code reused} on
     * ({@link ForwardBackend#forwardRows}), in prompt order across the activated LoRA's two passes.
     */
    private float[][] prefillPromptRows(int[] prompt, int reused, LoraPlan plan, LoraSession session) {
        if (session == null) {
            return backend.forwardRows(Arrays.copyOfRange(prompt, reused, prompt.length), reused, null);
        }
        int split = Math.max(reused, plan.from());
        float[][] pre = new float[0][];
        if (split > reused) {
            session.runPre();
            pre = backend.forwardRows(Arrays.copyOfRange(prompt, reused, split), reused, null);
        }
        session.runKey();
        float[][] post = backend.forwardRows(Arrays.copyOfRange(prompt, split, prompt.length), split, null);
        float[][] rows = Arrays.copyOf(pre, pre.length + post.length);
        System.arraycopy(post, 0, rows, pre.length, post.length);
        return rows;
    }

    /** The embedding-row prompt: every run prefills in order, then the token loop samples from the last row. */
    private GenerationResult runEmbeds(EmbedsPrompt prompt, SequenceState seq, int nCtx, SequenceRun seqRun,
                                       LoraPlan plan, LoraSession session) {
        long t0 = System.nanoTime();
        // rows carry no token id, so nothing stored can be matched against them: the prompt prefills whole
        backend.reset();
        seq.adapters(plan.identity());
        if (session != null) {
            session.runKey();
        }
        float[] logits = prefillRuns(prompt);
        int[] history = prompt.allTokens();
        seqRun.sampler().acceptPrompt(history, 0, history.length);
        seqRun.prefilled(prompt.rowCount(), System.nanoTime() - t0);
        int[] firstRun = prompt.tokens(0);
        return decodeLoop(logits, seq, nCtx, seqRun, firstRun.length > 0 ? firstRun[0] : -1);
    }

    /** Prefills every run of {@code prompt} in order; the logits of the prompt's last row come back. */
    private float[] prefillRuns(EmbedsPrompt prompt) {
        float[] logits = null;
        int pos = 0;
        for (int i = 0; i < prompt.segments(); i++) {
            int[] toks = prompt.tokens(i);
            if (toks.length > 0) {
                logits = backend.prefill(toks, pos);
                pos += toks.length;
            }
            if (i >= prompt.segments() - 1) {
                continue; // the last token run has no embedding rows after it
            }
            float[][] rows = prompt.embeds(i);
            if (rows.length > 0) {
                logits = backend.prefillEmbeddings(rows, pos);
                pos += rows.length;
            }
        }
        if (logits == null) {
            throw new IllegalArgumentException("the prompt holds no row to prefill");
        }
        return logits;
    }

    /**
     * The token loop after prefill: sample, offer, stop, and decode one token per step (llama.cpp completion
     * loop). {@code firstPromptToken} is the token a context shift keeps (a BOS prompt token; -1 when the prompt
     * does not begin with tokens).
     */
    private GenerationResult decodeLoop(float[] logits, SequenceState seq, int nCtx, SequenceRun seqRun,
                                        int firstPromptToken) {
        // the per-step selection: -1 the device's greedy id, > 0 a draw over the device's top candidates
        // ({@code RowTopK}, llama.cpp backend sampling -bs), 0 a host sample over the read-back row
        final int cone;
        final boolean prefiltered;
        if (seqRun.wantsHostSampling()) {
            cone = 0;
            prefiltered = false;
        } else if (seqRun.sampler().argmaxOnly()) {
            cone = -1;
            prefiltered = false;
        } else {
            Sampler sampler = seqRun.sampler();
            int c = samplingOnDevice ? sampler.coneWidth() : 0;
            boolean pre = false;
            if (c < 1 && samplingOnDevice && sampler.prefilterRequired() && backend.topKPrefilterOnDevice()) {
                c = sampler.deviceConeWidth();
                pre = c > 0;
            }
            boolean device = c > 0 && c <= MAX_DEVICE_TOPK && backend.topKOnDevice();
            cone = device ? c : 0;
            prefiltered = device && pre;
        }
        if (cone == 0 && !seqRun.wantsHostSampling() && !seqRun.sampler().argmaxOnly() && samplingOnDevice
                && !hostSamplingLogged) {
            LOGGER.info("{} samples on the host: {}", backend.getClass().getSimpleName(),
                    seqRun.sampler().coneWidth() <= 0
                            ? (seqRun.sampler().hasConstraints()
                            ? "the request's constraints (a grammar, allowed_token_ids, bad words, the min-tokens"
                            + " ban) mask over the whole row"
                            : "the sampler chain keeps candidates outside a device top-k")
                            : "the backend reads the sampled rows back");
            hostSamplingLogged = true;
        }
        if (cone == -1 && backend.chainsGreedyOnDevice()) {
            return greedyChainLoop(logits, seq, nCtx, seqRun, firstPromptToken);
        }
        int deviceToken = -1;
        while (!seqRun.done()) {
            if (seqRun.cancelled()) {
                seqRun.stop(StopReason.CANCELLED);
                break;
            }
            int token = deviceToken >= 0 ? deviceToken : seqRun.next(logits);
            if (deviceToken >= 0) {
                seqRun.accept(token);
            }
            SequenceRun.Step step = seqRun.offer(token);
            if (step.stopped()) {
                break;
            }
            if (seq.nPast() >= nCtx && !shiftContext(firstPromptToken, seq)) {
                seqRun.stop(StopReason.CONTEXT_FULL);
                break;
            }
            int pos = seq.nPast();
            if (cone == -1) {
                deviceToken = backend.decodeGreedyOnDevice(token, pos);
            } else if (cone > 0) {
                LogitPrefilter prefilter = prefiltered
                        ? LogitPrefilter.single(seqRun.sampler().prefilterRow()) : null;
                RowTopK top = prefiltered
                        ? backend.decodeTopKOnDevice(token, pos, cone, prefilter)
                        : backend.decodeTopKOnDevice(token, pos, cone);
                deviceToken = seqRun.sampleTopK(top, 0, prefiltered);
            } else {
                logits = backend.decode(token, pos);
            }
        }
        return seqRun.result();
    }

    /**
     * The token loop of a device-greedy chain on a backend that chains on the device
     * ({@link ForwardBackend#chainsGreedyOnDevice()}): the same steps as {@link #decodeLoop} in the same order, with
     * the decode of each token's greedy successor enqueued before that successor reaches the host, so the device
     * computes step {@code n + 1} while the host offers token {@code n}. The chained step is the decode the serial
     * loop runs next whenever the offered token does not stop the run; it is enqueued only at a position the memory
     * holds without a context shift and while the backend still chains (it can drop the step again), and a run that
     * ends before using it (a stop, a cancellation, a failure) drops its position again ({@link #dropChained}), so
     * the tokens, the stop and the stored sequence are the serial loop's.
     */
    private GenerationResult greedyChainLoop(float[] logits, SequenceState seq, int nCtx, SequenceRun seqRun,
                                             int firstPromptToken) {
        GreedyStep current = null;
        GreedyStep chained = null;
        try {
            while (!seqRun.done()) {
                if (seqRun.cancelled()) {
                    seqRun.stop(StopReason.CANCELLED);
                    break;
                }
                int token;
                if (current == null) {
                    token = seqRun.next(logits);
                } else {
                    token = current.id();
                    seqRun.accept(token);
                }
                SequenceRun.Step step = seqRun.offer(token);
                if (step.stopped()) {
                    break;
                }
                if (chained == null) {
                    if (seq.nPast() >= nCtx && !shiftContext(firstPromptToken, seq)) {
                        seqRun.stop(StopReason.CONTEXT_FULL);
                        break;
                    }
                    chained = backend.startGreedy(token, seq.nPast());
                }
                current = chained;
                chained = null;
                if (seq.nPast() < nCtx && backend.chainsGreedyOnDevice()) {
                    chained = backend.chainGreedy(current, seq.nPast());
                }
            }
        } catch (RuntimeException e) {
            try {
                dropChained(chained);
            } catch (RuntimeException dropFailure) {
                throw Closeables.chain(e, dropFailure);
            }
            throw e;
        }
        dropChained(chained);
        return seqRun.result();
    }

    /**
     * Drops the position of a chained greedy step the run did not use: waits for the step, then rolls its position
     * back when the memory allows ({@link ForwardBackend#canRollback}); a memory that cannot roll back keeps the
     * row, which holds the step's token computed exactly. A step that failed (or read its input from a step that
     * failed) throws instead, and the generation's failure path discards the whole sequence.
     */
    private void dropChained(GreedyStep chained) {
        if (chained == null) {
            return;
        }
        chained.id();
        if (backend.canRollback(1)) {
            backend.rollback(1);
        }
    }

    /** {@code config} with a whole-context penalty window bounded to the context length. */
    private static SamplerConfig samplerConfig(SamplerConfig config, int nCtx) {
        return config.penaltyLastN() == SamplerConfig.PENALTY_WHOLE_CONTEXT
                ? config.toBuilder().penaltyLastN(nCtx).build() : config;
    }

    /**
     * Brings the memory to the prompt's start and reports the prompt tokens it already holds: with prefix reuse,
     * the longest common prefix of the stored tokens and the prompt (at most {@code prompt.length - 1}, so the
     * last prompt token is evaluated for logits) stays, whose rows were computed with exactly those tokens
     * ({@link SequenceState#commonPrefix}, bounded by the {@link SequenceState#reusable} frontier), and the
     * positions after it are dropped behind a {@link ForwardBackend#canRollback} check — the same condition
     * llama.cpp checks before reusing a cached prefix (tools/server/server-context.cpp:4061-4070: every ring must
     * hold the keys the first evaluated query attends). A memory that cannot drop the suffix, and every generation
     * with reuse off ({@code neuralfree.prefix.reuse=false}) or an encoder-decoder (its memory holds the decoder
     * sequence, reset per call), resets instead and prefills the whole prompt.
     *
     * @return the number of prompt tokens already in memory, positions {@code [0, reused)}
     */
    private int prepareMemory(int[] prompt, SequenceState seq, boolean readPromptRows, String adapterIdentity) {
        int reused = prefixReuse && !encoderDecoder && !readPromptRows
                ? Math.min(seq.commonPrefix(prompt, adapterIdentity), prompt.length - 1) : 0;
        int drop = seq.nPast() - reused;
        if (reused > 0 && (drop == 0 || backend.canRollback(drop))) {
            if (drop > 0) {
                backend.rollback(drop);
            }
            return reused;
        }
        backend.reset();
        return 0;
    }

    /**
     * Frees positions by a context shift.
     *
     * @param firstPromptToken the prompt's first token, which a shift starting at BOS keeps (-1 when the prompt
     *                         begins with embedding rows)
     * @return false when shifting is disabled, nothing can be discarded or the memory cannot shift
     */
    private boolean shiftContext(int firstPromptToken, SequenceState seq) {
        return contextShift && shiftContext(backend, tokenizer, firstPromptToken, seq) != null;
    }

    /** The positions a context shift kept. Immutable. */
    static final class Shift {
        private final int kept;

        Shift(int kept) {
            this.kept = kept;
        }

        /** Positions kept, llama.cpp's {@code n_keep}. */
        int kept() { return kept; }
    }

    /**
     * llama.cpp completion's context shift over {@code backend}'s full memory, shared with
     * {@link SpeculativeDecoder}: {@code n_keep} positions stay (1 when the prompt starts with BOS), half of the
     * rest is discarded and later positions move down.
     *
     * @param firstPromptToken the prompt's first token (-1 when the prompt begins with embedding rows)
     * @return the positions kept, or null when nothing can be discarded or the memory cannot shift
     */
    static Shift shiftContext(ForwardBackend backend, GGUFTokenizer tokenizer, int firstPromptToken,
                              SequenceState seq) {
        int nPast = seq.nPast();
        int bos = tokenizer.specialTokens().bos();
        boolean startsWithBos = firstPromptToken >= 0 && bos != SpecialTokens.NULL && firstPromptToken == bos;
        int nKeep = Math.min(startsWithBos ? 1 : 0, nPast);
        int nDiscard = (nPast - nKeep) / 2;
        if (nDiscard <= 0) {
            return null;
        }
        try {
            backend.shiftContext(nKeep, nDiscard);
        } catch (UnsupportedOperationException e) {
            LOGGER.warn("context full and the context shift is unsupported: {}", e.getMessage());
            return null;
        }
        return new Shift(nKeep);
    }

    /**
     * A failed step leaves no recoverable sequence: the memory resets and its state clears, even when the reset
     * itself throws, so the next generation never reuses a half-written sequence.
     */
    private void discardHalfWritten(SequenceState seq, RuntimeException e) {
        try {
            backend.reset();
        } catch (RuntimeException resetFailure) {
            e.addSuppressed(resetFailure);
        } finally {
            try {
                seq.clear();
            } catch (RuntimeException clearFailure) {
                e.addSuppressed(clearFailure);
            }
        }
    }

    private static void requirePrompt(int[] prompt) {
        if (prompt.length == 0) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
        for (int i = 0; i < prompt.length; i++) {
            if (prompt[i] < 0) {
                throw new IllegalArgumentException("prompt token " + i + " has negative id " + prompt[i]);
            }
        }
    }

    @Override
    public String toString() {
        return "GenerationLoop{" + reasoning + ", prefixReuse=" + prefixReuse + ", contextShift=" + contextShift
                + (encoderDecoder ? ", encoder-decoder" : "")
                + (samplingDefaults.isEmpty() ? "" : ", " + samplingDefaults) + '}';
    }

    public static final class Builder {
        private final ForwardBackend backend;
        private final GGUFTokenizer tokenizer;
        private ReasoningFilter reasoning = ReasoningFilter.none();
        private boolean prefixReuse = true;
        private boolean contextShift;
        private boolean encoderDecoder;
        private SamplingDefaults samplingDefaults = SamplingDefaults.NONE;
        private LoraRegistry adapters;
        private boolean samplingOnDevice = true;
        private LogitsProcessors logitsProcessors = LogitsProcessors.empty();

        private Builder(ForwardBackend backend, GGUFTokenizer tokenizer) {
            this.backend = Objects.requireNonNull(backend, "backend");
            this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        }

        /** Reasoning parser of the output (default {@link ReasoningFilter#none()}). */
        public Builder reasoning(ReasoningFilter v) { this.reasoning = Objects.requireNonNull(v, "reasoning"); return this; }

        /** Keep the common prompt prefix between generations (default true; {@code neuralfree.prefix.reuse}). */
        public Builder prefixReuse(boolean v) { this.prefixReuse = v; return this; }

        /** Shift the context instead of stopping when it is full (default false; {@code neuralfree.context.shift}). */
        public Builder contextShift(boolean v) { this.contextShift = v; return this; }

        /** Prefix reuse and context shift from memory settings. */
        public Builder memoryConfig(MemoryConfig v) {
            Objects.requireNonNull(v, "memoryConfig");
            this.prefixReuse = v.prefixReuse();
            this.contextShift = v.contextShift();
            return this;
        }

        /**
         * Whether each step's selection runs on the device when the chain keeps its candidates inside a top-k cone
         * ({@code Sampler#coneWidth}; default true; {@code neuralfree.sampling.device}).
         */
        public Builder samplingOnDevice(boolean v) { this.samplingOnDevice = v; return this; }

        /** The custom logits processors every generation consults (default none). */
        public Builder logitsProcessors(LogitsProcessors v) {
            this.logitsProcessors = Objects.requireNonNull(v, "logitsProcessors");
            return this;
        }

        /** The backend is an encoder-decoder (the prompt is the encoder source; memory is reset per call). */
        public Builder encoderDecoder(boolean v) { this.encoderDecoder = v; return this; }

        /** The model's sampling defaults applied to every request (default {@link SamplingDefaults#NONE}). */
        public Builder samplingDefaults(SamplingDefaults v) {
            this.samplingDefaults = Objects.requireNonNull(v, "samplingDefaults");
            return this;
        }

        /**
         * The model's per-request adapters ({@link EngineModel#loraRegistry()}), which requests that select adapters
         * run through; null (the default) refuses such requests by name.
         */
        public Builder adapters(LoraRegistry v) {
            this.adapters = v;
            return this;
        }

        public GenerationLoop build() {
            return new GenerationLoop(this);
        }
    }
}
