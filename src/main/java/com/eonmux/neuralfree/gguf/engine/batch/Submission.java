package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * One submitted request: its prompt and output side, queued until a slot admits it. {@code prefixKey} holds the
 * extra keys its prefix-cache chains start with: the content identity of its per-request adapters
 * ({@code adapters}, at an activated LoRA's invocation start) and its cache salt ({@code cache_salt}), so rows
 * computed under other adapters or salts never match (vLLM keys its block hashes by the LoRA too);
 * {@code promptLogprobs} collects the prompt-row scores of a {@code prompt_logprobs} or
 * {@code prompt_logprob_token_ids} request while its prefill reads the rows back (null while it waits or when the
 * request asks for none); {@code pooling} carries a pooling request's rows and handle (null for a generation
 * request). {@code mediaIds} are the encoder cache entries the submission references while it lives
 * ({@link EncoderCache#reference}), released when it retires. On an encoder-decoder model {@code source} holds the
 * request's prompt, which the encoder reads, and {@code prompt} the decoder rows the sequence prefills: the decoder
 * start token (llama.cpp completion replaces the prompt by it after encoding, completion.cpp:530-543).
 */
final class Submission {
    /**
     * The {@code vllm_xargs} argument that picks a request's prefix-cache report mode (vLLM's
     * {@code kv_cache_report_mode}): {@code full} reports every stored and every reused block, the default
     * {@code incremental} only the blocks a step stores.
     */
    static final String REPORT_MODE = "kv_cache_report_mode";
    /** The report mode that also reports the prefix-cache blocks a request reuses. */
    static final String REPORT_MODE_FULL = "full";

    final BatchSequence handle;
    final MediaPrompt prompt;
    final GenerationRequest request;
    final ReasoningFilter.Sink sink;
    final String renderedPrompt;
    final com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar;
    /** The request's reasoning filter, or null for the engine's. */
    final ReasoningFilter filter;
    /** The request's prefix-cache extra keys: cache lookups, cached blocks and donor forks match them exactly. */
    final PrefixKey prefixKey;
    /** The adapter slots of the request's rows, resolved at submission ({@link LoraPlan#DEFAULT} for slot 0). */
    final LoraPlan adapters;
    /**
     * The adapter slot of the rows from {@code adapters.from()} on, held while admitted or preempted; 0 while it
     * holds none.
     */
    int adapterSlot;
    /** The adapter slot of the rows before {@code adapters.from()}, held like {@link #adapterSlot}. */
    int preAdapterSlot;
    /** The encoder source of an encoder-decoder request, or null for a decoder-only model. */
    final int[] source;
    /** The media identities whose encoder cache entries the submission references; empty without any. */
    final List<String> mediaIds;
    /** Whether the submission dropped its {@link #mediaIds} references, which it does once, at its outcome. */
    boolean mediaReleased;
    /** Prompt rows the sequence reuses from the prefix cache or its donor's fork. */
    int reused;
    /** Whether every cached block the request reuses is reported as a {@code BlockStored} (report mode full). */
    final boolean reportsReusedBlocks;
    /** The prompt-row score capture the prefill fills, or null. */
    PromptLogprobs.Collector promptLogprobs;
    /** The pooling request's output side, or null for a generation submission. */
    final PoolingRun pooling;
    /** The prompt's keyed rows for the prefix cache, built on first use. */
    private CacheRows promptRows;

    Submission(BatchSequence handle, MediaPrompt prompt, GenerationRequest request,
               ReasoningFilter.Sink sink, String renderedPrompt,
               com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar, ReasoningFilter filter,
               Collection<String> mediaIds) {
        this(handle, prompt, null, request, sink, renderedPrompt, grammar, filter, LoraPlan.DEFAULT, mediaIds,
                null);
    }

    /**
     * A pooling request whose rows run in the adapter slots of {@code adapters} ({@link LoraPlan#DEFAULT} for
     * slot 0): {@code pooling} carries the request's output side, and it references no encoder cache entries.
     */
    Submission(BatchSequence handle, MediaPrompt prompt, GenerationRequest request,
               ReasoningFilter.Sink sink, String renderedPrompt,
               com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar, ReasoningFilter filter,
               LoraPlan adapters, Collection<String> mediaIds, PoolingRun pooling) {
        this(handle, prompt, null, request, sink, renderedPrompt, grammar, filter, adapters, mediaIds, pooling);
    }

    /**
     * A request whose rows run in the adapter slots of {@code adapters} ({@link LoraPlan#DEFAULT} for slot 0) that
     * references the encoder cache entries {@code mediaIds}. On an encoder-decoder model {@code source} is the
     * encoder's input and {@code prompt} the decoder's start rows; null otherwise. {@code pooling} carries a
     * pooling request's output side, or null for a generation request.
     */
    Submission(BatchSequence handle, MediaPrompt prompt, int[] source, GenerationRequest request,
               ReasoningFilter.Sink sink, String renderedPrompt,
               com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar, ReasoningFilter filter,
               LoraPlan adapters, Collection<String> mediaIds, PoolingRun pooling) {
        this.handle = handle;
        this.source = source;
        this.prompt = prompt;
        this.request = request;
        this.sink = sink;
        this.renderedPrompt = renderedPrompt;
        this.grammar = grammar;
        this.filter = filter;
        this.adapters = adapters;
        this.prefixKey = PrefixKey.of(request.cacheSalt(), adapters);
        this.mediaIds = List.copyOf(mediaIds);
        this.pooling = pooling;
        this.reportsReusedBlocks = REPORT_MODE_FULL.equals(request.logitsProcessorArgs().get(REPORT_MODE));
    }

    /** Whether the prefill must compute every prompt row's hidden state itself: a prompt-logprobs request reads
     * every row's logits back, and a pooling request whose output pools every row cannot reuse a cached prefix. */
    boolean prefillsEveryRow() {
        return wantsPromptLogprobs() || (pooling != null && pooling.readsAllRows());
    }

    /**
     * The prompt's rows as the prefix cache keys them ({@link CacheRows#of}): built once, hashing the prompt's
     * media and prompt-embedding rows.
     */
    CacheRows promptRows() {
        if (promptRows == null) {
            promptRows = CacheRows.of(prompt);
        }
        return promptRows;
    }

    /**
     * The prompt tokens the request's prefill evaluates, as {@code GenerationLoop} counts them: the encoder source of
     * an encoder-decoder request, the prompt rows after the reused prefix otherwise.
     */
    int prefilledTokens() {
        return source != null ? source.length : prompt.nTokens() - reused;
    }

    /** The tokens a rendered prompt is derived from when the request carries none: the source or the prompt. */
    int[] renderTokens() {
        return source != null ? source : prompt.samplerTokens();
    }

    /** The adapter slot of the row at {@code row}, the sequence's KV-row space ({@link #adapters}'s numbering). */
    int adapterSlot(int row) {
        return row >= adapters.from() ? adapterSlot : preAdapterSlot;
    }

    /** Whether some row of the request runs in a slot other than 0. */
    boolean runsAdapterSlots() {
        return adapterSlot > 0 || preAdapterSlot > 0;
    }

    /**
     * Whether the prefill reads back every prompt row's logits ({@code prompt_logprobs},
     * {@code prompt_logprob_token_ids}).
     */
    boolean wantsPromptLogprobs() {
        return request.readsPromptRows();
    }

    /** The scheduling priority ({@code priority}): lower is served earlier under {@link SchedulingPolicy#PRIORITY}. */
    int priority() {
        return request.priority();
    }

    /** Scores the logits row that processes prompt row {@code r} into the prompt capture. */
    void captureRow(int r, float[] logits) {
        if (promptLogprobs == null) {
            promptLogprobs = PromptLogprobs.Collector.of(prompt.samplerTokens(), request);
        }
        promptLogprobs.captureRow(r, logits);
        if (promptLogprobs.hasEntries() && promptLogprobs.complete() && handle.promptLogprobs() == null) {
            // published before the result exists: a streamed response reads it at the first output
            handle.promptLogprobs(Collections.unmodifiableList(new ArrayList<>(promptLogprobs.result())));
        }
    }
}
