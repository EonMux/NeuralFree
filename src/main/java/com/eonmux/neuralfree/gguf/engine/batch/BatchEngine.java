package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.HostMemory;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessors;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EmbeddingRowSupport;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.engine.LookupDrafterFactory;
import com.eonmux.neuralfree.gguf.engine.PoolingOperator;
import com.eonmux.neuralfree.gguf.engine.PoolingOutput;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.SequenceEncoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SequenceRun;
import com.eonmux.neuralfree.gguf.engine.StopCriteria;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.engine.VerifyRound;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvCacheEvent;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventBatch;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublisher;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventPublishers;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.utils.LoggingUtil;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Continuous batching over the paged memory: one open model, submitted prompts and pumped engine
 * steps. Every {@link #pump} call runs exactly one step — admitting queued requests, then one forward batch over
 * the rows the step's token budget schedules: one row per decoding sequence and, beside them, a chunk of the rows
 * a prefilling sequence still lacks (vLLM V1's scheduler with {@code enable_chunked_prefill}; llama.cpp serves the
 * slots' decode tokens and a prompt chunk from one {@code n_batch} batch) — through {@code
 * ForwardBackend.forwardBatch}, which splits every batch into cross-sequence ubatches over the page tables. A
 * decoding sequence contributes one row plain, or the {@code 1 + drafted} rows of a draft-and-verify round when
 * the engine speculates. Each sequence runs its own {@link SequenceRun} pipeline (sample, stop, route), so a
 * batch computes the same texts as the same requests run one at a time through
 * {@link com.eonmux.neuralfree.gguf.engine.GenerationLoop}.
 *
 * <p>Lifecycle of a submission ({@link #submit}): the prompt is checked against the context length (an oversize
 * prompt throws {@link ContextFullException} to the caller, as {@code GenerationLoop} does), then queued.
 * {@code pump} admits the queued request the {@link SchedulingPolicy} picks next
 * ({@code neuralfree.scheduling.policy}, vLLM's {@code --scheduling-policy}): under {@code fcfs} (the default)
 * the earliest submission, and a request's {@code priority} other than 0 refuses at submit; under
 * {@code priority} the lowest {@code priority} value first, among equals the earliest submission. It admits when a
 * sequence slot is free and every KV group's pool holds the idle
 * pages its prompt takes next to the admitted prompts still waiting for prefill (vLLM's {@code allocate_slots}
 * count: the whole pages of a reused prefix cost nothing, a shared partial last page costs its copy-on-write copy),
 * evicting least recently used free cached blocks, and in a sliding-window group the window pages context
 * checkpoints pin, to make room. A prefix is reused when it is cached in the block pool ({@link PrefixCache}: whole
 * pages keyed by their hash chain, which every running and retired sequence's committed rows feed, adopted onto the
 * claimed slot) or is the committed prompt of a running sequence (its pages fork over by copy-on-write at the
 * fork's prefill), so concurrent requests with a common prefix prefill it once.
 * The page count is a hint: the paged reservation of each ubatch is all-or-nothing and stays the authority, and a
 * step that still runs out of pages (decodes outgrowing the pool) evicts cached blocks or preempts the running
 * sequence the policy serves last.
 * Over a memory with recurrent state or sliding-window groups a cached prefix restores only where a context
 * checkpoint holds the recurrent state and the window pages at exactly its rows ({@link ContextCheckpoints},
 * llama.cpp's {@code --ctx-checkpoints}, vLLM's Mamba "align" cache): a leaving sequence leaves one at its
 * committed rows and a text or media prefill takes them at {@code neuralfree.checkpoint.min.step} spacing, at the
 * prompt's message boundaries and before its last page; {@code neuralfree.ctx.checkpoints=0} turns the prefix
 * cache off for such a memory, and forks off a running donor do not run over it. A checkpoint hangs on a cached
 * block, so it restores only for a prompt whose rows match that block's identity: the same tokens under the same
 * extra keys (adapters, media items, cache salt, prompt embedding digests). A memory with DSV4 compressed caches
 * restores at checkpoints that carry its compressor streams and rollback planes beside the rest; where the memory
 * cannot copy the streams, the prefix cache stays off. With {@code neuralfree.cache.ram} set, the blocks eviction
 * drops for room move to a host-RAM tier ({@link HostBlockCache}) and come back into fresh pages when a later prompt
 * walks them; it refuses by name over a memory that restores only at checkpoints or whose blocks cannot leave it
 * ({@link BlockLayout#refusal}). With {@code neuralfree.cache.disk.mib} and {@code neuralfree.cache.disk.dir} set
 * beside it, the host tier's own drops move on to a disk tier ({@link DiskBlockCache}, vLLM's tiering filesystem
 * tier), one file per block under the directory, so the cache outlives the process. The engine's slots also save,
 * restore and erase their cached history as
 * {@link SlotFile}s ({@link #saveSlot}, llama.cpp's slot actions over its prefix cache), refusing by name wherever
 * the cache is off. After every step the engine publishes the cache's KV cache events
 * ({@link KvEventPublisher}). The engine prefills an admitted request in budget-bounded chunks — the chunk's last
 * row carries the output logits only when it completes the prompt, and a request reading back prompt logprobs
 * ({@code prompt_logprobs}) asks every prompt row's logits — and decodes one token per pump per running sequence
 * (a verify round's tokens when the engine speculates). A finished sequence reports its {@link GenerationResult}
 * on its {@link BatchSequence} handle and its slot and pages free at once, so the next pump can admit a queued
 * request into the same slot.
 *
 * <p>The step's token budget ({@code neuralfree.max.batched.tokens}, the ubatch by default) is spent first-come
 * first-served over the running sequences in admission order: earlier sequences take their rows first, so a
 * sequence admitted before a long prompt keeps decoding every step, while a later one waits (as vLLM's FCFS loop
 * does; {@code neuralfree.long.prefill.tokens} caps one chunk's share so a long prompt cannot starve its
 * successors). With chunked prefill off ({@code neuralfree.chunked.prefill=false}) the step keeps the older
 * shape: the oldest waiting prefill runs alone — its whole prompt tail in one forward batch — or, when none
 * waits, one decode row per running sequence.
 *
 * <p>Running out of pages is recoverable, as vLLM's scheduler makes it. A prefill that finds the pools full rolls
 * back what it wrote and queues again at the head; admission then waits until a retirement or eviction frees
 * memory, so a prompt that cannot fit never starves the decodes. A decode step that finds the pools full rolls back
 * the rows it committed, then frees memory by evicting free cached blocks or, with none left, by preempting one
 * running sequence (recompute preemption) — under {@code fcfs} the youngest, under {@code priority} the one the
 * policy serves last (the highest {@code priority} value, among equals the latest submission), as vLLM's scheduler
 * picks it: its pages free while its run keeps its sampler, grammar, stop filter and streamed text, and its handle
 * counts the preemption ({@link BatchSequence#preemptions()}). Once a running sequence retires it is re-admitted —
 * under {@code fcfs} ahead of the queue, under {@code priority} in the policy's order among the queued
 * submissions — its prompt and fed tokens prefill again, and it continues with the token it had already sampled,
 * so preemption never changes a result. A sequence that cannot fit even with every other sequence's memory freed
 * fails with the {@link MemoryFullException}.
 *
 * <p>Per request the engine mirrors {@code GenerationLoop}: the request takes the model's sampling defaults for
 * every unset setting, {@link StopCriteria} derives the end-of-generation set, the penalty history starts with the
 * prompt tokens, and the reasoning parser routes output pieces to the submission's sink. When a decoding
 * sequence's next position would leave the context, the engine shifts it like llama.cpp's server shifts its slots
 * ({@code ContextShift}, server-context.cpp:3783-3824) when {@code neuralfree.context.shift} is on: the request's
 * {@code n_keep}/{@code n_discard} (a negative {@code n_keep} keeps the whole prompt, a leading BOS is kept on
 * top, {@code n_keep} never passes the {@code n_ctx - 4} margin, and {@code n_discard} 0 discards half of what
 * lies beyond the kept tokens) drop positions {@code [n_keep, n_keep + n_discard)} and move every later position
 * down, the fed tokens compact with the moved rows, the handle reports {@code truncated}, and the sequence keeps
 * decoding in its slot ({@code BatchStats.contextShifts} counts the shifts); the sequence's cached blocks above
 * the kept prefix leave the prefix cache, whose kept blocks stay shareable. The shift serves the text prompts of
 * a decoder-only model, whose rows are one token per position; a media prompt's shared M-RoPE rows and an
 * encoder-decoder's fixed decoder start stop with {@link StopReason#CONTEXT_FULL} instead, and a request setting
 * {@code n_keep}/{@code n_discard} on one refuses at submit by name. With the shift off, a memory that refuses
 * the move, or nothing to discard, a full context stops with {@link StopReason#CONTEXT_FULL}. A request whose
 * {@code grp_attn_n} is above 1 instead compresses its positions as it grows ({@code SelfExtend}, llama.cpp's
 * group attention): every {@code grp_attn_w} positions are grouped and divided by the factor, so the sequence
 * attends at an effective context of {@code nCtx / grp_attn_n} and never fills the position space the shift
 * would free. A cancellation poll per step stops with {@link StopReason#CANCELLED}.
 *
 * <p>With a {@link BatchDrafter} the decode steps speculate (vLLM's {@code scheduled_spec_decode_tokens},
 * llama.cpp server's per-slot speculative loop): every step drafts the proposals of its decoding sequences in one
 * drafter call, cuts each proposal before its first token the sequence's grammar refuses (vLLM's
 * {@code validate_tokens}, as {@code SpeculativeDecoder} does), and the step's one {@code forwardBatch} call scores
 * every sequence's {@code [pendingToken, drafts..]} rows beside the plain decodes and prefill chunks. Each sequence
 * verifies its rows through its own sampler chain: by {@code MATCH}, keeping the longest prefix its chain would
 * have sampled (llama.cpp's {@code common_sampler_sample_and_accept_n}), or by the configured
 * {@code neuralfree.draft.verify} mode ({@code rejection}: vLLM's {@code RejectionSampler} over the drafter's
 * proposal probabilities; {@code block}, {@code fly}, {@code synthetic}) with its own uniform stream seeded by the
 * request's seed. The unverified tail leaves the paged sequence ({@code remove}, the same rollback the memory-full
 * path takes), the confirmed tokens join the fed history, and a stop inside a round (end of generation, stop
 * string, budget, cancel) leaves the sequence in the ordinary rejection state. Verification is lossless: the
 * lossless modes keep every emitted token a draw from the target's processed distribution, so a batch over a
 * drafter computes the same texts as one without. The configured drafters are the lookup types with per-sequence
 * state ({@link BatchLookupDrafter}) and a separate draft model over paged draft memory ({@link BatchModelDrafter});
 * the MTP head and the hidden-fed draft files refuse by name, and a caller may supply any {@link BatchDrafter}
 * through {@link #forModel(EngineModel, EngineConfig, BatchDrafter, BatchSpec)}.
 *
 * <p>A watermarked sequence verifies by rejection only, as vLLM's speculative watermarking requires, and its
 * drafter samples each proposal under the configuration's draft role: a {@link BatchModelDrafter} over a draft
 * model in the target vocabulary. The sequence's target-role sampler draws the replacements and bonuses, so the
 * emitted text is detectable under the target key exactly as in {@code SpeculativeDecoder}. A single-key
 * configuration without {@code allow_target_only_watermarking}, another mode, or a drafter that only replays
 * stored tokens refuses at submit by name.
 *
 * <p>Media prompts ({@link MediaPrompt}: text and media chunks, llama.cpp server's {@code server_tokens}) prefill
 * chunk by chunk: each chunk is one piece of the step — a text chunk runs its tokens causally beside the other
 * pieces, a media chunk runs its embedding rows as its own forward batch (its M-RoPE sections and mask rule are
 * batch-wide) with the chunk's positions and, when the chunk says so, without the causal mask. A chunk can take
 * fewer positions than rows (an M-RoPE image grid), so every slot carries a row cursor (the KV cells it stored)
 * beside the position cursor the decodes continue at. The step's encoder compute budget ({@code
 * neuralfree.max.encoder.tokens}, vLLM's {@code encoder_compute_budget} applied by its V1 scheduler) caps the
 * media item rows one step schedules across all its media prefills: whole items schedule while the budget lasts,
 * an unschedulable item ends its prompt's walk for the step (nothing prefills past an unwritten item), and an item
 * that alone exceeds the whole budget schedules on a step whose budget is untouched and spends all of it (vLLM
 * raises the budget to the largest multimodal item), so a long media prefill cannot starve the decodes and the
 * other media prompts; text and prompt-embedding chunks take no encoder budget. Media prompts share the prefix
 * cache with text prompts: a block's media rows are keyed by their item's identity ({@link MediaChunk#item()},
 * vLLM's {@code mm_hash}: the client's uuid, else a content digest, the key of the encoded-media cache too) and
 * their offset in the item, so identical media at identical places reuse cached rows and different media never do;
 * a reused prefix stops short of a partly matched item, which prefills whole — one rule
 * ({@link MediaChunk#isMediaItem()}) bounds both the reuse and the prefill's chunks. A preempted media sequence
 * recomputes its chunks and fed tokens in the same order, so results never change. A prompt of precomputed
 * embedding rows ({@link EmbedsPrompt}, vLLM's {@code prompt_embeds}) submits as the media prompt of its runs
 * ({@link EmbedsPrompt#toMediaPrompt()}): its rows run as causal prompt-embedding chunks at one position per row,
 * each row keyed by the digest of its bytes.</p>
 *
 * <p>Pooling requests ({@link #submitPooling}: embeddings, per-token embeddings, classifier and decision scores,
 * rerank scores) queue into the same machinery and batch with the generations: vLLM's scheduler handles its
 * {@code pooling_params} requests in the same token-budget scheduler, llama.cpp runs its embedding and rerank
 * tasks in slots of the shared batch, and neither wipes the prefix cache — the pooling request prefills beside
 * the running sequences under the same budget, accumulates its prompt's hidden rows (whole for a bidirectional
 * model, chunk by chunk for a causal one) and applies the model's pooling operator at the completion, its rows
 * feeding the cache like a generation's. A request whose output pools every row reuses no cached prefix, because
 * the reused rows' hidden states are not its own forward's.
 *
 * <p>Identified media chunks ({@link MediaChunk#id()}, the item's key and the chunk's piece of the item) share
 * their encoder output through the engine's encoder cache ({@code neuralfree.encoder.cache.rows}
 * rows, the step's token budget by default; vLLM's {@code EncoderCacheManager}): the rows of one identity store
 * once and every submission carrying it references the same arrays, each holding its reference until it retires,
 * and the least recently used unreferenced entries evict when a new identity needs the space (never a referenced
 * one; a chunk larger than the whole budget, and every chunk while the cache is off, keeps its prompt's own
 * arrays). {@link BatchStats} reports the cache's rows, hits and evictions.</p>
 *
 * <p>Per-request LoRA adapters ({@link GenerationRequest#lora()}, vLLM's {@code lora_request}, llama.cpp server's
 * per-request {@code lora} list) run through the model's {@link EngineModel#loraRegistry()}: a submission resolves
 * its selection to a {@link LoraPlan} at once (an unknown adapter refuses before anything queues), admission
 * acquires the plan's adapter slots — a request whose key finds every slot serving running requests of other keys
 * waits while other candidates admit, as vLLM's scheduler skips a LoRA past {@code max_loras} — and every row the
 * sequence runs carries its slot ({@code ForwardBatch.adapterSlots}): slot 0 (the startup adapters) for a request
 * that selects none, the key's slot for the rows from an activated LoRA's invocation start on and the
 * pre-invocation slot before it, so sequences of different adapters share one forward batch. A preempted sequence
 * keeps its slots; retirement releases them. The prefix cache and donor forks match only rows computed under the
 * same adapters at the same scales (and the same invocation start).</p>
 *
 * <p>An encoder-decoder model (T5) batches over a backend that keeps one encoding per sequence
 * ({@code ForwardBackend.sequenceEncoder()}): a submitted token prompt is the request's encoder source, checked
 * against the backend's source limit at submit; admission (and the re-admission of a preempted sequence) encodes
 * the source into the sequence's own cross caches, and the sequence then prefills the decoder start token and
 * decodes like any other, so each request computes what {@code GenerationLoop} computes for it alone: the penalty
 * history starts with the decoder start token and the prompt phase counts the source. Its decoder rows depend on
 * the source, so the prefix cache stays off; media prompts and {@code prompt_logprobs} refuse at submit.</p>
 *
 * <p>Failures are per step where the engine cannot attribute them: a single-sequence prefill failure lands on
 * that submission alone ({@code PagedSequences.prepare} is all-or-nothing for the failing ubatch and the step's
 * rows roll back), a drafter failure lands on that sequence alone (the proposal runs before anything is
 * written), while a step over several sequences — decode rounds beside prefill chunks — fails every sequence
 * it scheduled when its forward or sampling throws mid-batch: the split runs ubatches in order and a
 * throw abandons the rest, so no returned row can be attributed. The failing sequences retire with
 * {@link BatchSequence.State#FAILED}; the memory slot and pages free either way.
 *
 * <p>Not thread-safe: one caller thread submits and pumps (the adapters drive it from a single worker). The
 * {@link BatchSequence} handles are safe to read from other threads. The engine owns no model: the caller opens
 * and closes it. The memory must be the paged form ({@code neuralfree.parallel > 1} or
 * {@code neuralfree.kv.paged=true}) on either backend; every other memory refuses at construction by name, and a
 * backend whose forward path cannot run the batch refuses through its own {@code forwardBatch}.
 *
 * <p>The engine is a facade over package-private collaborators sharing one {@code BatchState}: the
 * {@code Admitter} admits (with {@code PrefixClaims} for the reused prefixes and {@code SlotOpener} for each
 * sequence's pipeline), the {@code StepScheduler} schedules a step's pieces ({@code PrefillChunks} for the
 * prefill chunks, {@code DecodeRounds} for the decode and verify rounds), the {@code StepExecutor} runs them, the
 * {@code OutputRouter} routes their logits, and the {@code SlotLifecycle} retires, preempts and recovers.
 */
public final class BatchEngine implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(BatchEngine.class.getName());

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    private static final double GIB = 1024.0 * 1024.0 * 1024.0;

    private final GGUFTokenizer tokenizer;
    private final BatchDrafter drafter;
    /** The speculative settings, or null when the engine decodes plainly. */
    private final BatchSpec spec;
    private final SamplingDefaults samplingDefaults;
    /** The custom logits processors of the engine configuration, empty when it configures none. */
    private final LogitsProcessors logitsProcessors;
    private final boolean ownsDrafter;
    /** The forward the engine steps; read at admission to refuse what its batched forward cannot take. */
    private final ForwardBackend backend;
    private final BatchState state;
    private final SlotLifecycle lifecycle;
    private final Admitter admitter;
    private final StepScheduler scheduler;
    private final SelfExtend selfExtend;
    /** The KV chunk reuse of admitted prompts ({@code n_cache_reuse}, llama.cpp's cached-prompt loop). */
    private final ChunkReuse chunkReuse;
    private final KvEventPublisher events;
    private final SlotStore slotStore;
    /** Why an engine without a prefix cache retains no blocks. */
    private static final String NO_PREFIX_CACHE = "the batch engine runs no prefix cache ("
            + MemoryConfig.PROP_PREFIX_REUSE + " off, a sliding-window model, a bidirectional model or an"
            + " encoder-decoder), so a slot retains no rows";
    /** The model's pooling operator, or null when the model has neither a pooling nor a classifier head. */
    private final PoolingOperator poolingOperator;
    /** How long a drain lets the disk tier's admitted writes finish before publishing the cache's events. */
    private static final long CACHE_DRAIN_MILLIS = 5000;
    /** The model's support class: a generation submit refuses on a pooling-only one, a pooling submit the rest. */
    private final SupportClass support;
    /** Whether the model's attention is causal (a bidirectional model's pooling prompts prefill whole). */
    private final boolean causal;
    /** The rows of one ubatch, the bound a bidirectional pooling prompt must fit. */
    private final int ubatch;
    private int nextSubmitId;
    private boolean closed;

    private BatchEngine(ForwardBackend backend, GGUFTokenizer tokenizer, ReasoningFilter reasoning,
                        SamplingDefaults samplingDefaults, PagedSequences sequences, int maxSequences,
                        boolean prefixReuse, MemoryConfig memoryConfig, boolean chunkedPrefill,
                        int tokenBudget, BatchDrafter drafter, BatchSpec spec, boolean ownsDrafter,
                        KvEventPublisher events, LoraRegistry adapters, int embeddingRowToken,
                        PoolingOperator poolingOperator, SupportClass support, boolean causal,
                        DiskBlockCache.Io diskIo, LogitsProcessors logitsProcessors, LookaheadOptions lookahead) {
        this.tokenizer = tokenizer;
        this.drafter = drafter;
        this.spec = spec;
        this.samplingDefaults = samplingDefaults;
        this.logitsProcessors = Objects.requireNonNull(logitsProcessors, "logitsProcessors");
        this.ownsDrafter = ownsDrafter;
        this.events = Objects.requireNonNull(events, "events");
        this.support = support;
        this.causal = causal;
        ModelMemory memory = backend.memory();
        boolean lookaheadOn = lookahead != null && lookahead.enabled();
        LookaheadOptions lookaheadRun = lookaheadOn ? lookahead : null;
        checkSequenceState(memory, drafter);
        if (lookaheadOn) {
            if (drafter != null) {
                throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " and speculative decoding"
                        + " exclude each other: lookahead decoding drafts and verifies with the target's own"
                        + " Jacobi guesses");
            }
            if (memory.hasRecurrentState()) {
                throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " rewinds the rows of a"
                        + " step, which state a model with recurrent state cannot restore");
            }
            if (memory.plan().hasCompressed()) {
                throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " rewinds the rows of a"
                        + " step, which a memory with compressed caches cannot restore");
            }
            if (memoryConfig.contextShift()) {
                throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " manages the stored positions"
                        + " of its step and cannot shift the context; clear " + MemoryConfig
                        .PROP_CONTEXT_SHIFT);
            }
            if (sequences.maxSequences() < lookahead.sequenceCount()) {
                throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " runs a group of "
                        + lookahead.sequenceCount() + " sequences per request (window " + lookahead.window()
                        + ", n-gram " + lookahead.ngramSize() + ", " + lookahead.maxNgrams() + " n-grams); the memory"
                        + " holds " + sequences.maxSequences() + ". Set " + MemoryConfig.PROP_PARALLEL + "="
                        + lookahead.sequenceCount());
            }
            for (int g = 0; g < sequences.groups(); g++) {
                if (sequences.trims(g)) {
                    throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " rewinds the rows of a"
                            + " step, which a sliding-window memory trims away");
                }
            }
        }
        int ubatch = memory.plan().ubatch();
        SequenceEncoder encoder = backend.sequenceEncoder();
        if (lookaheadOn && encoder != null) {
            throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " runs the target's decoder"
                    + " over a group of sequences; an encoder-decoder's per-sequence source encoding has no group"
                    + " form");
        }
        // the cached blocks' rows must stay valid: a windowed (SWA) memory trims old positions per commit, a
        // bidirectional model's rows attend each other, so a cached prefix's rows would not be the rows this
        // request's forward computes (vLLM runs no prefix caching for its bidirectional pooling models)
        boolean reuse = prefixReuse && cacheable(memory, encoder) && causal && !lookaheadOn;
        String lookaheadRefusal = lookaheadOn ? LookaheadOptions.PROP_ENABLED + " decoding keeps no reusable"
                + " prefix: every request prefills its prompt into its own sequence group" : null;
        String needs = ContextCheckpoints.needs(memory, sequences);
        ContextCheckpoints checkpoints = reuse && needs != null
                ? contextCheckpoints(sequences, memory, memoryConfig, needs) : null;
        String medium = memory instanceof GpuMemory ? KvCacheEvent.MEDIUM_GPU : KvCacheEvent.MEDIUM_CPU;
        boolean cacheOn = reuse && (needs == null || checkpoints != null);
        String cacheRefusal = null;
        if (!cacheOn) {
            cacheRefusal = lookaheadRefusal != null ? lookaheadRefusal
                    : reuse ? "the memory restores a cached prefix only at a context checkpoint (" + needs + "), and "
                            + MemoryConfig.PROP_CTX_CHECKPOINTS + "=0 keeps none"
                    : NO_PREFIX_CACHE;
        } else if (checkpoints != null) {
            cacheRefusal = "the memory restores a cached prefix only at a context checkpoint (" + needs + "; "
                    + MemoryConfig.PROP_CTX_CHECKPOINTS + "), and a block holding checkpoints cannot move to the host";
        }
        String layoutRefusal = BlockLayout.refusal(memory);
        BlockLayout layout = layoutRefusal == null ? BlockLayout.of(memory) : null;
        HostBlockCache host = hostTier(memoryConfig.cacheRamMib(), cacheRefusal, layoutRefusal);
        BlockHasher hasher = cacheOn ? BlockHasher.forEnvironment(memoryConfig.prefixHashAlgo()) : null;
        DiskBlockCache disk = diskTier(memoryConfig, layout, hasher, cacheRefusal, layoutRefusal, diskIo);
        PrefixCache cache = cacheOn ? new PrefixCache(sequences, memoryConfig.prefixCacheBudget(), medium,
                events.enabled(), embeddingRowToken, hasher, checkpoints, host, disk, layout) : null;
        this.backend = backend;
        this.state = new BatchState(sequences, memory, maxSequences, cache, checkpoints, drafter, adapters, encoder,
                new EncoderCache(encoderCacheBudget(memoryConfig.encoderCacheRows(), tokenBudget)),
                memoryConfig.schedulingPolicy(), poolingOperator, lookaheadRun);
        this.slotStore = new SlotStore(state, cacheRefusal != null ? cacheRefusal : layoutRefusal, layout);
        this.lifecycle = new SlotLifecycle(state);
        this.chunkReuse = new ChunkReuse(state, backend, embeddingRowToken);
        PrefixClaims claims = new PrefixClaims(state, chunkReuse);
        this.admitter = new Admitter(state, claims, new SlotOpener(tokenizer, reasoning, samplingDefaults, state, spec,
                logitsProcessors), lifecycle);
        ContextShift shifts = new ContextShift(state, backend, tokenizer, lifecycle, memoryConfig.contextShift());
        DecodeRounds rounds = new DecodeRounds(state, lifecycle, shifts, drafter, spec, ubatch, backend);
        StepExecutor executor = new StepExecutor(backend, state, lifecycle,
                new OutputRouter(state, lifecycle, rounds, shifts), GenerationLoop.deviceSamplingEnabled());
        this.scheduler = new StepScheduler(state, claims, lifecycle, rounds, executor, chunkedPrefill, tokenBudget,
                memoryConfig.longPrefillTokens(), encoderComputeBudget(memoryConfig.maxEncoderTokens(), tokenBudget),
                ubatch, causal, spec != null && spec.treeVerify());
        this.selfExtend = new SelfExtend(state, backend, lifecycle);
        this.poolingOperator = poolingOperator;
        this.ubatch = ubatch;
    }

    /**
     * The host-RAM tier of {@code cacheRamMib} MiB ({@code neuralfree.cache.ram}, -1 unbounded), or null for 0.
     *
     * @param cacheRefusal why the engine runs no prefix cache, or null when it runs one
     * @throws UnsupportedOperationException naming the property when the engine runs no prefix cache, its memory
     *                                       restores a cached prefix only at a context checkpoint (the tier moves
     *                                       plain blocks only), or its cached blocks cannot leave it
     *                                       ({@code layoutRefusal})
     */
    private static HostBlockCache hostTier(int cacheRamMib, String cacheRefusal, String layoutRefusal) {
        if (cacheRamMib == 0) {
            return null;
        }
        if (cacheRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_RAM + " refused: " + cacheRefusal);
        }
        if (layoutRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_RAM + " refused: " + layoutRefusal);
        }
        return new HostBlockCache(cacheRamMib < 0 ? -1L : cacheRamMib * (1L << 20));
    }

    /**
     * The disk tier of {@code cacheDiskMib} MiB under {@code cacheDiskDir} ({@code neuralfree.cache.disk.mib} /
     * {@code neuralfree.cache.disk.dir}, -1 unbounded), or null for 0. Its files are named by the chain hashes
     * {@code hasher} computes and kept in the namespace of its algorithm and seed, so a directory shared with
     * another seed or another engine's blocks only misses. It runs under the host-RAM tier, which
     * {@link MemoryConfig} requires to be on, and refuses for every reason the host tier refuses, naming its own
     * property.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.cache.disk.mib} when the engine runs no prefix
     *                                       cache, its memory restores a cached prefix only at a context
     *                                       checkpoint, or its cached blocks cannot leave it
     * @throws UncheckedIOException           naming {@code neuralfree.cache.disk.dir} when the directory cannot be
     *                                        created or read
     */
    private static DiskBlockCache diskTier(MemoryConfig memoryConfig, BlockLayout layout, BlockHasher hasher,
                                           String cacheRefusal, String layoutRefusal, DiskBlockCache.Io diskIo) {
        if (memoryConfig.cacheDiskMib() == 0) {
            return null;
        }
        if (cacheRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_DISK_MIB + " refused: " + cacheRefusal);
        }
        if (layoutRefusal != null) {
            throw new UnsupportedOperationException(MemoryConfig.PROP_CACHE_DISK_MIB + " refused: " + layoutRefusal);
        }
        try {
            return new DiskBlockCache(Path.of(memoryConfig.cacheDiskDir()),
                    memoryConfig.cacheDiskMib() < 0 ? -1L : memoryConfig.cacheDiskMib() * (1L << 20), layout,
                    hasher, diskIo);
        } catch (IOException e) {
            throw new UncheckedIOException(MemoryConfig.PROP_CACHE_DISK_DIR + "=" + memoryConfig.cacheDiskDir()
                    + " is not usable: " + e.getMessage(), e);
        }
    }

    /**
     * The step's encoder compute budget (vLLM's {@code encoder_compute_budget}, raised to the largest multimodal
     * item): {@code neuralfree.max.encoder.tokens} when set, the step's token budget otherwise.
     */
    private static int encoderComputeBudget(int configured, int tokenBudget) {
        return configured > 0 ? configured : tokenBudget;
    }

    /**
     * The encoder cache's row budget (vLLM's {@code encoder_cache_size}): {@code neuralfree.encoder.cache.rows}
     * when positive, the step's token budget when 0, and 0 (the cache off) when negative.
     */
    private static int encoderCacheBudget(int configured, int tokenBudget) {
        if (configured < 0) {
            return 0;
        }
        return configured > 0 ? configured : tokenBudget;
    }

    /**
     * A batch engine over {@code model}: refuses models that do not generate text (diffusion, no support),
     * memories that hold no paged sequences, naming the properties that select them, and accepts a pooling-only
     * model (an embedding or classifier with pooling heads) beside the generating ones — its pooling requests run
     * here, its generation submits refuse. Encoder-decoders need a backend that keeps one encoding for every
     * sequence. When {@code config} selects drafters the engine runs (the lookup types of {@code
     * neuralfree.draft.type}, a separate draft model {@code neuralfree.draft}), sized by {@code
     * neuralfree.draft.max} and {@code neuralfree.draft.min} and verified by {@code neuralfree.draft.verify}, the
     * decode steps speculate; the MTP head and the hidden-fed draft files refuse by name. The engine closes the
     * drafter it built from {@code config} (writing the dynamic lookup cache file when one is set). KV cache
     * events go to the publisher {@code neuralfree.kv.events} names
     * ({@link KvEventPublishers#forConfig}).
     *
     * @throws UnsupportedOperationException when the model or its memory cannot run batched generation, or the
     *                                       draft configuration is one the batch engine does not run
     * @throws UncheckedIOException          when a lookup cache file or the draft model cannot be read
     * @throws IllegalStateException         for synthetic verification without its opt-in
     */
    public static BatchEngine forModel(EngineModel model, EngineConfig config) {
        Objects.requireNonNull(config, "config");
        return forModel(model, config, HostMemory.totalBytes(), KvEventPublishers.forConfig(config.memory()),
                DiskBlockCache.FILES);
    }

    /**
     * {@link #forModel(EngineModel, EngineConfig)} publishing the KV cache events to {@code events} (an
     * in-process listener, {@link KvEventPublishers#listener}, or any publisher) instead of the one the config
     * names. The engine closes the publisher when it closes.
     *
     * @throws UnsupportedOperationException when the model or its memory cannot run batched generation
     */
    public static BatchEngine forModel(EngineModel model, EngineConfig config, KvEventPublisher events) {
        return forModel(model, config, HostMemory.totalBytes(), events, DiskBlockCache.FILES);
    }

    /**
     * {@link #forModel(EngineModel, EngineConfig, KvEventPublisher)} over {@code diskIo}, so a test can run the
     * engine beside a slow filesystem and observe that no step waits on it.
     */
    static BatchEngine forModel(EngineModel model, EngineConfig config, KvEventPublisher events,
                                DiskBlockCache.Io diskIo) {
        return forModel(model, config, HostMemory.totalBytes(), events, diskIo);
    }

    /** {@link #forModel(EngineModel, EngineConfig)} against a host whose physical RAM is
     *  {@code totalRamBytes} (&lt;= 0 = unknown, the default then stands). */
    static BatchEngine forModel(EngineModel model, EngineConfig config, long totalRamBytes) {
        Objects.requireNonNull(config, "config");
        return forModel(model, config, totalRamBytes, KvEventPublishers.forConfig(config.memory()),
                DiskBlockCache.FILES);
    }

    private static BatchEngine forModel(EngineModel model, EngineConfig config, long totalRamBytes,
                                        KvEventPublisher events, DiskBlockCache.Io diskIo) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(events, "events");
        PagedSequences paged = pagedMemory(model);
        BatchSpec spec = configSpec(model, config, paged);
        BatchDrafter drafter = spec == null ? null : configDrafter(model, config, paged);
        if (drafter != null) {
            if (spec.treeVerify() && !drafter.proposesTrees()) {
                throw new UnsupportedOperationException(SpeculativeOptions.PROP_TREE + "=true verifies tree"
                        + " proposals, and the batch drafter " + drafter + " proposes chains only; the tree verify"
                        + " runs over " + EngineConfig.PROP_DRAFT + " (a draft model), which splits each request's"
                        + " tree over " + SpeculativeOptions.PROP_TREE_BRANCHES + "=" + spec.treeBranches()
                        + " draft sequences");
            }
            LOGGER.info("speculative decoding on " + model.name() + ": " + drafter + ", draft max "
                    + spec.draftMax() + " (requested " + config.draftMax() + "), min " + spec.draftMin()
                    + ", verify " + spec.verifyMode()
                    + (spec.treeVerify() ? ", tree verify (width " + spec.treeWidth() + ", budget "
                    + spec.treeBudgetPolicyName() + ")" : ""));
        } else {
            spec = null;
        }
        try {
            return new BatchEngine(model.forwardBackend(), model.tokenizer(),
                    ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), model.tokenizer()),
                    SamplingDefaults.forModel(model, config), paged, paged.maxSequences(),
                    prefixReuseForHostRam(config.memory(), model.spec().tensors().totalBytes(), totalRamBytes),
                    config.memory(), chunkedPrefill(config), tokenBudget(config), drafter, spec, drafter != null,
                    events, model.loraRegistry(), EmbeddingRowSupport.recordToken(model.spec()),
                    model.poolingOperator(), model.supportClass(), model.spec().hparams().causal(), diskIo,
                    LogitsProcessors.forConfig(config), config.lookaheadOptions());
        } catch (RuntimeException | Error e) {
            if (drafter != null) {
                closeAll(Collections.singletonList(drafter), e);
            }
            throw e;
        }
    }

    /**
     * A batch engine over {@code model} with the caller's drafter: the decode steps verify every sequence's
     * proposal ({@code draftMax} tokens at most, {@code draftMin} shortest verified) through the sequence's own
     * sampling chain, by the spec's verify mode. The drafter stays owned by the caller; {@link #close()} drops only
     * the engine's state. The drafter must read its history from the same paged memory the model's backend holds.
     *
     * @throws UnsupportedOperationException when the model or its memory cannot run batched generation
     */
    public static BatchEngine forModel(EngineModel model, EngineConfig config, BatchDrafter drafter,
                                       BatchSpec spec) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(drafter, "drafter");
        Objects.requireNonNull(spec, "spec");
        PagedSequences paged = pagedMemory(model);
        if (spec.draftMax() > rollbackBound(paged)) {
            throw new UnsupportedOperationException(model.architecture() + ": a round of " + spec.draftMax()
                    + " drafted tokens cannot roll back: the DSV4 compressor keeps "
                    + paged.plan().rollbackSnapshots() + " rollback snapshots per sequence");
        }
        if (spec.treeVerify()) {
            checkBatchTreeRope(model, spec.treeVerify());
            if (!drafter.proposesTrees()) {
                throw new UnsupportedOperationException(SpeculativeOptions.PROP_TREE + "=true verifies tree"
                        + " proposals, and the caller's batch drafter " + drafter + " proposes chains only");
            }
        }
        return new BatchEngine(model.forwardBackend(), model.tokenizer(),
                ReasoningParsers.forModel(config.reasoningParser(), config.toolParser(), model.tokenizer()),
                SamplingDefaults.forModel(model, config), paged, paged.maxSequences(),
                prefixReuseForHostRam(config.memory(), model.spec().tensors().totalBytes(), HostMemory.totalBytes()),
                config.memory(), chunkedPrefill(config), tokenBudget(config), drafter, spec, false,
                KvEventPublishers.forConfig(config.memory()), model.loraRegistry(),
                EmbeddingRowSupport.recordToken(model.spec()), model.poolingOperator(), model.supportClass(),
                model.spec().hparams().causal(), DiskBlockCache.FILES, LogitsProcessors.forConfig(config),
                config.lookaheadOptions());
    }

    /** Whether a prompt prefill splits into budget-bounded chunks beside the decodes ({@code config.memory()}). */
    private static boolean chunkedPrefill(EngineConfig config) {
        return config.memory().chunkedPrefill();
    }

    /** The step's token budget over prefill chunk rows and decode rows: {@code neuralfree.max.batched.tokens}, the ubatch by default. */
    private static int tokenBudget(EngineConfig config) {
        MemoryConfig mem = config.memory();
        return mem.maxBatchedTokens() > 0 ? mem.maxBatchedTokens() : mem.ubatch();
    }

    /**
     * The paged memory the batch engine runs over, or the refusal by name: the model must generate text
     * autoregressively or pool (an embedding or classifier with pooling heads) on a backend whose memory is paged
     * ({@code neuralfree.parallel > 1} or {@code neuralfree.kv.paged=true}), on either compute backend; a
     * masked-diffusion or unsupported model refuses, and a pooling-only model without pooling heads refuses too —
     * with them it serves its pooling requests here beside any generation. An encoder-decoder also needs a
     * backend that keeps one encoding per sequence ({@code ForwardBackend.sequenceEncoder()}, the CPU and GPU
     * encoder-decoders).
     */
    private static PagedSequences pagedMemory(EngineModel model) {
        SupportClass support = model.supportClass();
        switch (support) {
            case DIFF:
                throw new UnsupportedOperationException(model.architecture()
                        + " is a masked-diffusion model; generate with DiffusionGenerator");
            case NA:
                throw new UnsupportedOperationException(model.architecture() + " (" + support
                        + ") does not generate text autoregressively");
            case EMBED:
            case CLS:
                if (model.poolingOperator() == null) {
                    throw new UnsupportedOperationException(model.architecture() + " (" + support
                            + ") does not generate text autoregressively");
                }
                break;
            default:
                break;
        }
        ForwardBackend backend = model.forwardBackend();
        ModelMemory memory = backend.memory();
        PagedSequences paged = memory == null ? null : memory.pagedSequences();
        if (paged == null) {
            throw new UnsupportedOperationException(model.architecture() + ": batched generation needs paged memory"
                    + " with a batched forward (neuralfree.parallel > 1 or neuralfree.kv.paged=true), got "
                    + (memory == null ? "no memory" : memory.getClass().getSimpleName()));
        }
        if (support == SupportClass.ENC_DEC && backend.sequenceEncoder() == null) {
            throw new UnsupportedOperationException(model.architecture() + " is an encoder-decoder and "
                    + backend.getClass().getSimpleName() + " keeps one encoding for all its sequences; batched"
                    + " encoder-decoder generation needs a backend whose sequences keep their own (a SequenceEncoder:"
                    + " the CPU and GPU encoder-decoder backends over paged memory)");
        }
        return paged;
    }

    /**
     * The most drafted tokens a verify round may drop: unbounded, except that DSV4 compressed caches drop at most
     * their compressor rollback snapshots per sequence ({@code MemoryPlan.rollbackSnapshots()}, as
     * {@code SpeculativeDecoder} bounds its rounds).
     */
    private static int rollbackBound(PagedSequences paged) {
        return paged.plan().hasCompressed() ? paged.plan().rollbackSnapshots() : Integer.MAX_VALUE;
    }

    /**
     * Whether cached prefix blocks can restore a sequence of {@code memory}: not the decoder rows of an
     * encoder-decoder ({@code encoder} set), which depend on the sequence's source that no cached block records;
     * and not the rows of a memory with DSV4 compressed caches whose streams the memory cannot copy for a context
     * checkpoint (the device memory — the heap memory copies them).
     */
    private static boolean cacheable(ModelMemory memory, SequenceEncoder encoder) {
        return encoder == null && memory.compressedStateRefusal() == null;
    }

    /**
     * Refuses a memory whose recurrent state the batch engine cannot keep per sequence: the state of every
     * sequence must live in its own row the engine can zero when the slot frees, and copy for a context
     * checkpoint. Refuses a drafter over recurrent state too: a rejected draft leaves the paged rows by
     * {@code remove}, which cannot rewind a recurrent state to the last accepted token.
     *
     * @throws UnsupportedOperationException naming what the memory or the drafter lacks
     */
    private static void checkSequenceState(ModelMemory memory, BatchDrafter drafter) {
        if (!memory.hasRecurrentState()) {
            return;
        }
        String refusal = memory.sequenceStateRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("batched generation over recurrent state needs a recurrent"
                    + " state row per sequence: " + refusal);
        }
        if (drafter != null) {
            throw new UnsupportedOperationException("speculative decoding (" + EngineConfig.PROP_DRAFT_TYPE
                    + ") over recurrent state is not supported by the batch engine: a rejected draft cannot rewind"
                    + " the recurrent state");
        }
    }

    /**
     * The context checkpoints of a memory whose cached prefixes restore only at a checkpoint ({@code needs} names
     * why), or null when {@code neuralfree.ctx.checkpoints} is 0: the prefix cache is then off for this memory.
     */
    private static ContextCheckpoints contextCheckpoints(PagedSequences sequences, ModelMemory memory,
                                                         MemoryConfig config, String needs) {
        if (config.ctxCheckpoints() == 0) {
            LoggingUtil.getLogger(BatchEngine.class).info(
                    "the prefix cache is off: a memory with {} restores a cached prefix only at a context"
                            + " checkpoint, and {}=0", needs, MemoryConfig.PROP_CTX_CHECKPOINTS);
            return null;
        }
        return new ContextCheckpoints(sequences, memory, config.ctxCheckpoints(), config.checkpointMinStep());
    }

    /**
     * The speculative settings {@code config} selects ({@code neuralfree.draft.max}, {@code .min}, the verify mode),
     * or null when it configures no drafter, or when the memory cannot drop a round's rejected tokens (DSV4
     * compressed caches planned without rollback snapshots), which turns speculation off, logged.
     *
     * @throws IllegalStateException    for synthetic verification without its opt-in, naming the property
     * @throws IllegalArgumentException for invalid synthetic settings
     */
    private static BatchSpec configSpec(EngineModel model, EngineConfig config, PagedSequences paged) {
        if (config.draftModelPath() == null && !config.mtp() && config.lookupTypes().isEmpty()) {
            if (config.speculativeOptions().probabilisticDraft()) {
                throw new IllegalArgumentException(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=probabilistic"
                        + " applies to a draft model, but " + EngineConfig.PROP_DRAFT + " names none");
            }
            return null;
        }
        checkBatchTreeRope(model, config.speculativeOptions().treeVerify());
        int draftMax = Math.min(config.draftMax(), rollbackBound(paged));
        if (draftMax < 1) {
            LOGGER.info("speculative decoding is off on " + model.name() + ": the DSV4 compressor keeps no rollback"
                    + " snapshots to drop a round's rejected tokens");
            return null;
        }
        return BatchSpec.of(draftMax, Math.min(config.draftMin(), draftMax), config.speculativeOptions());
    }

    /**
     * Refuses the batch engine's tree verify over a multi-section rope (M-RoPE): the tree rows run at their depth
     * positions, which the shared batch carries as the one text section the batch engine ropes every text row from;
     * a multi-section model's per-section pattern per row has no batch tree form. The single-sequence engine
     * ({@code SpeculativeDecoder}) verifies trees over a multi-section rope.
     *
     * @throws UnsupportedOperationException naming the rope type when {@code neuralfree.draft.tree} is set
     */
    private static void checkBatchTreeRope(EngineModel model, boolean treeVerify) {
        if (!treeVerify) {
            return;
        }
        com.eonmux.neuralfree.gguf.arch.RopeType ropeType = model.spec().hparams().ropeType();
        if (com.eonmux.neuralfree.gguf.ops.rope.RopeMath.positionsPerToken(ropeType) > 1) {
            throw new UnsupportedOperationException("the batch engine verifies tree proposals over a one-section"
                    + " rope; " + model.name() + " ropes a " + ropeType + " from several positions per token"
                    + " (M-RoPE), whose per-row section pattern a batch tree step does not carry. Turn "
                    + SpeculativeOptions.PROP_TREE + " off for this model");
        }
        checkBatchTreeAttention("model " + model.name(), model.spec());
    }

    /**
     * Refuses a tree-masked batch over attention whose mask rule the shared masked batch has no form for: the
     * standard paged attention consumes the per-row ancestor mask (the CPU core and the GPU paged flash kernel),
     * and the MLA layers ride the same masked paged launches (the mixer uploads each segment's relativized mask
     * rows and packs the segment's fresh-row count into the launch flags, as the single-sequence tree verify on
     * MLA does), while DSA, MSA, DSV4, recurrent and Step5 layers run their own attention whose kernels take no
     * tree mask — a step that silently ignored the mask would verify tokens no target sampled. The
     * single-sequence engine ({@code SpeculativeDecoder}) carries the same limits.
     *
     * @throws UnsupportedOperationException naming the layer's attention when the model runs a masked batch
     */
    static void checkBatchTreeAttention(String role, com.eonmux.neuralfree.gguf.spec.ModelSpec spec) {
        List<com.eonmux.neuralfree.gguf.spec.LayerSpec> layers = spec.layers();
        for (int i = 0; i < layers.size(); i++) {
            com.eonmux.neuralfree.gguf.spec.LayerSpec layer = layers.get(i);
            String takes = layer.dsa() != null ? "DSA" : layer.msa() != null ? "MSA"
                    : layer.step5() != null ? "Step5" : layer.dsv4() != null ? "DSV4" : layer.recurrent() != null
                    ? "recurrent" : null;
            if (takes != null) {
                throw new UnsupportedOperationException("a tree-masked batch runs the standard paged attention,"
                        + " whose per-row ancestor mask the " + takes + " layers of the " + role + " (layer " + i
                        + ") take no part in; turn " + SpeculativeOptions.PROP_TREE + " off for this model");
            }
        }
    }

    /**
     * The engine drafter {@code config} selects (vLLM's batched proposers, llama.cpp server's per-slot speculative
     * state): the lookup types ({@code neuralfree.draft.type}: {@code ngram-simple}, {@code ngram-map-k},
     * {@code ngram-map-k4v}, {@code ngram-mod}, {@code ngram-cache}, {@code suffix}) as a {@link BatchLookupDrafter}
     * with one drafter set per sequence slot, and a separate draft model ({@code neuralfree.draft}) as a
     * {@link BatchModelDrafter} over paged draft memory, chained in llama.cpp's ask order (the lookups first) when
     * both are set. The MTP head ({@code neuralfree.mtp}) and the hidden-fed draft files (eagle3,
     * gemma4-assistant, dflash, MTP draft files) refuse by name: they read the target's hidden rows of one
     * sequence ({@code LLMModelFactory.openSpeculative} runs them).
     *
     * @throws UnsupportedOperationException for the MTP head or a hidden-fed draft file
     * @throws UncheckedIOException          when a lookup cache file or the draft model cannot be read, naming it
     * @throws IllegalArgumentException      when the draft model cannot draft for the target, or probabilistic
     *                                       drafting is set without a draft model
     */
    private static BatchDrafter configDrafter(EngineModel model, EngineConfig config, PagedSequences paged) {
        if (config.mtp()) {
            throw new UnsupportedOperationException("the batch engine drafts with a separate draft model ("
                    + EngineConfig.PROP_DRAFT + "), the n-gram and the suffix drafters; " + EngineConfig.PROP_MTP
                    + " drafts with the target's NextN heads over one sequence's hidden rows, which runs"
                    + " single-sequence (LLMModelFactory.openSpeculative)");
        }
        boolean probabilistic = config.speculativeOptions().probabilisticDraft();
        if (probabilistic && config.draftModelPath() == null) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=probabilistic"
                    + " applies to a draft model, but " + EngineConfig.PROP_DRAFT + " names none");
        }
        List<BatchDrafter> members = new ArrayList<>(2);
        try {
            LookupDrafterFactory lookups = LookupDrafterFactory.open(config);
            if (!lookups.isEmpty()) {
                members.add(new BatchLookupDrafter(lookups, paged));
            }
            if (config.draftModelPath() != null) {
                members.add(modelDrafter(model, config, probabilistic));
            }
        } catch (IOException e) {
            closeAll(members, e);
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (RuntimeException | Error e) {
            closeAll(members, e);
            throw e;
        }
        return members.size() == 1 ? members.get(0) : new BatchDrafterChain(members, paged.maxSequences());
    }

    /** The batch draft model of {@code config.draftModelPath()} over the target {@code model}. */
    private static BatchDrafter modelDrafter(EngineModel model, EngineConfig config, boolean probabilistic)
            throws IOException {
        EngineModel draft = LLMModelFactory.openBatchDraftModel(model, config);
        try {
            return new BatchModelDrafter(draft, model, config.draftPMin(), probabilistic,
                    config.speculativeOptions().heterogeneousVocab(), config.speculativeOptions().treeBranches());
        } catch (RuntimeException | Error e) {
            try {
                draft.close();
            } catch (IOException | RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    private static void closeAll(List<BatchDrafter> members, Throwable failure) {
        for (BatchDrafter m : members) {
            try {
                m.close();
            } catch (RuntimeException e) {
                failure.addSuppressed(e);
            }
        }
    }

    /**
     * Whether the prompt cache ({@link PrefixCache}) runs for a model of {@code modelBytes} on a host of
     * {@code totalRamBytes}: an explicit setting stands, and an unset cache is turned off beside a model larger
     * than the physical RAM — the model lives in the page cache of its mapping, and the retired sequences the
     * cache keeps hold host pages the model needs (Avalon tools/server/server-context.cpp:1550-1566).
     */
    static boolean prefixReuseForHostRam(MemoryConfig memory, long modelBytes, long totalRamBytes) {
        Objects.requireNonNull(memory, "memory");
        if (!memory.prefixReuse()) {
            return false;
        }
        if (memory.prefixReuseSet() || totalRamBytes <= 0 || modelBytes <= totalRamBytes) {
            return true;
        }
        LoggingUtil.getLogger(BatchEngine.class).info(
                "model ({} GiB) is larger than RAM ({} GiB): the prefix cache is disabled, set {}=true to force it",
                modelBytes / GIB, totalRamBytes / GIB, MemoryConfig.PROP_PREFIX_REUSE);
        return false;
    }

    /** The model's tokenizer. */
    public GGUFTokenizer tokenizer() {
        return tokenizer;
    }

    /** Sequences the paged memory holds ({@code neuralfree.parallel}). */
    public int maxSequences() {
        return state.slots.length;
    }

    /** The order the engine admits and preempts its requests in ({@code neuralfree.scheduling.policy}). */
    public SchedulingPolicy schedulingPolicy() {
        return state.policy;
    }

    /** Submissions waiting for a slot. */
    public int queued() {
        return state.queue.size();
    }

    /** Positions the paged memory holds per sequence (the plan's {@code nCtx}). */
    public int contextLength() {
        return state.nCtx;
    }

    /** Submissions currently prefilling or decoding. */
    public int running() {
        return state.active.size();
    }

    /** The engine's counters and current state; call it on the engine's thread, then share the snapshot freely. */
    public BatchStats stats() {
        return state.snapshot();
    }

    /**
     * Drops every cached block and frees its pages, publishing {@code AllBlocksCleared} (vLLM's
     * {@code reset_prefix_cache}): another user of the model reset the memory under them (a pooling forward on the
     * same backend), or an operator asked for it. Call it only while the engine is {@link #idle()}.
     *
     * @throws IllegalStateException when a submission is queued, running or preempted
     */
    public void clearPrefixCache() {
        if (!idle()) {
            throw new IllegalStateException("the prefix cache clears only on an idle engine");
        }
        if (state.cache != null) {
            state.cache.clear();
        }
        publishEvents();
    }

    /**
     * Writes slot {@code seq}'s state to {@code out} (llama.cpp's slot save, {@code llama_state_seq_save_file}): the
     * prefix-cache chain its last sequence left, as far as it is still cached, with the K/V rows of every block read
     * from the memory on either backend. Call it on the engine's thread.
     *
     * @return the tokens saved and the file's bytes
     * @throws UnsupportedOperationException when the engine runs no prefix cache or its memory's blocks do not hold a
     *                                       sequence's whole state (recurrent, compressed, side or cross caches), or
     *                                       the history holds media rows
     * @throws IllegalArgumentException      for a slot id outside {@link #maxSequences()}
     * @throws IllegalStateException         while the slot runs a sequence
     * @throws IOException                   when the write fails
     */
    public SlotResult saveSlot(int seq, Path out) throws IOException {
        checkOpen();
        Objects.requireNonNull(out, "out");
        return slotStore.save(seq, out);
    }

    /**
     * Reads the slot file at {@code in} into the prefix cache as slot {@code seq}'s state (llama.cpp's slot restore):
     * its blocks take fresh pages, free cached blocks making room, so the next request whose prompt starts with the
     * restored tokens adopts their rows instead of prefilling them. A file saved on the other backend restores
     * unchanged. Call it on the engine's thread.
     *
     * @return the tokens restored and the file's bytes
     * @throws UnsupportedOperationException as {@link #saveSlot}
     * @throws IllegalArgumentException      for a slot id outside {@link #maxSequences()}
     * @throws IllegalStateException         while the slot runs a sequence, or when no page can be freed for a block
     * @throws IOException                   for an unreadable file, another model's or cache type's file, or a
     *                                       malformed chain
     */
    public SlotResult restoreSlot(int seq, Path in) throws IOException {
        checkOpen();
        Objects.requireNonNull(in, "in");
        try {
            return slotStore.restore(seq, in);
        } finally {
            publishEvents();
        }
    }

    /**
     * Exports slot {@code seq}'s cached history as the bytes of a slot file, for a KV transfer connector sending a
     * prefill instance's rows to a decode instance (disaggregated serving): the same chain {@link #saveSlot} writes,
     * read from the memory on either backend. Call it on the engine's thread, after the sequence retired and before
     * its slot serves another request.
     *
     * @throws UnsupportedOperationException when the engine runs no prefix cache or its memory's blocks do not hold a
     *                                       sequence's whole state (recurrent, compressed, side or cross caches), or
     *                                       the history holds media rows
     * @throws IllegalArgumentException      for a slot id outside {@link #maxSequences()}
     * @throws IllegalStateException         while the slot runs a sequence
     * @throws IOException                   when the rows cannot be read
     */
    public byte[] exportKv(int seq) throws IOException {
        checkOpen();
        return slotStore.saveBytes(seq);
    }

    /**
     * Imports a transferred chain (the bytes of a slot file, a prefill instance's {@link #exportKv}) into the prefix
     * cache under the first free sequence id, so a request whose prompt starts with the chain's tokens adopts its
     * rows instead of prefilling them (disaggregated serving's decode side). Call it on the engine's thread.
     *
     * @return the tokens the chain restored
     * @throws UnsupportedOperationException as {@link #exportKv}
     * @throws IllegalStateException         when every sequence id runs a sequence, or no cached block can be evicted
     *                                       for the chain's pages
     * @throws IOException                   for a malformed or foreign chain
     */
    public int importKv(byte[] chain) throws IOException {
        checkOpen();
        Objects.requireNonNull(chain, "chain");
        int seq = state.freeSlot();
        if (seq < 0) {
            throw new IllegalStateException("no free sequence id to import a transferred KV chain (" + state.slots.length
                    + " run a sequence)");
        }
        try {
            return slotStore.restoreBytes(seq, chain);
        } finally {
            publishEvents();
        }
    }

    /**
     * Imports a transferred chain as slot {@code seq}'s history; see {@link #importKv(byte[])}.
     *
     * @throws IllegalArgumentException for a slot id outside {@link #maxSequences()}
     * @throws IllegalStateException    while the slot runs a sequence
     */
    public int importKv(int seq, byte[] chain) throws IOException {
        checkOpen();
        Objects.requireNonNull(chain, "chain");
        try {
            return slotStore.restoreBytes(seq, chain);
        } finally {
            publishEvents();
        }
    }

    /**
     * Erases slot {@code seq}'s state (llama.cpp's slot erase): forgets its history and evicts the cached blocks no
     * other slot's history walks. Call it on the engine's thread.
     *
     * @return the tokens the slot held
     * @throws UnsupportedOperationException as {@link #saveSlot}
     * @throws IllegalArgumentException      for a slot id outside {@link #maxSequences()}
     * @throws IllegalStateException         while the slot runs a sequence
     */
    public int eraseSlot(int seq) {
        checkOpen();
        try {
            return slotStore.erase(seq);
        } finally {
            publishEvents();
        }
    }

    /**
     * The status of every slot (llama.cpp's {@code GET /slots} data): a running sequence's request counters and
     * tokens, or the task the id last served ({@code BatchState.recordTask}), or the bare idle status. Call it on
     * the engine's thread.
     */
    public List<SlotStatus> slotStatuses() {
        checkOpen();
        List<SlotStatus> out = new ArrayList<>(state.slots.length);
        for (int seq = 0; seq < state.slots.length; seq++) {
            out.add(state.status(seq));
        }
        return out;
    }

    /**
     * The status of slot {@code seq}, as {@link #slotStatuses()} reports it.
     *
     * @throws IllegalArgumentException for a slot id outside {@link #maxSequences()}
     */
    public SlotStatus slotStatus(int seq) {
        checkOpen();
        if (seq < 0 || seq >= state.slots.length) {
            throw new IllegalArgumentException("slot " + seq + " outside the " + state.slots.length + " slots");
        }
        return state.status(seq);
    }

    /**
     * Drops the encoder cache's shared media rows (vLLM's {@code reset_encoder_cache}): a later submission of the
     * same medium stores its own rows again. Running submissions keep the rows they hold, and the prefix cache's
     * blocks stay. Call it on the engine's thread; unlike {@link #clearPrefixCache()} it needs no idle engine.
     */
    public void clearEncoderCache() {
        state.encoderCache.clear();
    }

    /** Publishes the KV cache events the cache queued since the last publication, as one batch. */
    private void publishEvents() {
        if (state.cache == null || !events.enabled()) {
            return;
        }
        List<KvCacheEvent> pending = state.cache.takeEvents();
        if (!pending.isEmpty()) {
            KvEventPublishers.publishSafely(events,
                    new KvEventBatch(System.currentTimeMillis() / 1000.0, pending, null));
        }
    }

    /** Whether no submission is queued, running or preempted. */
    public boolean idle() {
        return state.idle();
    }

    /**
     * Cancels every queued, running and preempted submission: running sequences stop at their next step's
     * cancellation poll, queued ones when admitted, each with {@link StopReason#CANCELLED} (the same end a
     * per-request cancellation hook produces). A no-op on an idle engine; once the engine drained the cancelled
     * submissions and runs idle again, new ones submit normally.
     */
    public void cancelAll() {
        checkOpen();
        if (!idle()) {
            state.cancelPending = true;
        }
    }

    /**
     * Preempts every running sequence (recompute preemption): each frees its pages and keeps its run — sampler,
     * stop filter and fed tokens — and re-admits from the next pump on (under {@code fcfs} ahead of the queue,
     * under {@code priority} in the policy's order among the queued submissions), prefills its history again
     * and continues with the token it had already sampled, so nothing about the results changes. Queued submissions
     * keep waiting; the preempted sequences' committed rows stay cached, so their recomputes adopt them unless the
     * cache is cleared. An engine sleep uses this to free the KV cache under requests it must resume later.
     */
    public void preemptAll() {
        checkOpen();
        lifecycle.preemptAll();
        publishEvents();
    }

    /**
     * Submits {@code prompt} for generation: validated against the context length now (an oversize or empty
     * prompt throws to the caller), then queued; the {@link SchedulingPolicy} orders it among the admissions
     * already waiting. The sink receives the
     * routed output pieces with their {@link ReasoningFilter.Channel} as the engine pumps; the finished text and
     * timings land on the returned handle's {@link BatchSequence#result()}.
     *
     * @param renderedPrompt the rendered prompt text for reasoning-prefix detection (for example the chat
     *                       template's rendering); null renders it from the prompt tokens
     * @throws IllegalStateException    when the engine is closed
     * @throws ContextFullException     when the prompt does not fit the context
     * @throws IllegalArgumentException when the prompt is empty or holds a negative token id, or the request's
     *                                  {@code priority} is not 0 under the {@code fcfs} policy
     */
    public BatchSequence submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt) {
        return submit(MediaPrompt.ofText(prompt), request, sink, renderedPrompt, null);
    }

    /**
     * Submits with a grammar constraint: {@code constraint} — or, when null, the request's own
     * {@code GenerationRequest#constraint}) — compiles now ({@link
     * com.eonmux.neuralfree.gguf.grammar.GbnfBuilder#compile}), so an invalid or untranslatable spec throws to
     * the caller before anything is queued; a compiled grammar runs as the sequence's sampling stage
     * ({@link com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage}), masking every token the grammar cannot
     * accept next (guided_json, guided_regex, guided_choice map onto the spec kinds). A trigger-gated constraint
     * runs as a {@link com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage}: free text until a
     * trigger fires, the grammar from the trigger on.
     *
     * @throws IllegalArgumentException when the constraint compiles to an unusable grammar
     */
    public BatchSequence submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt, com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint) {
        return submit(MediaPrompt.ofText(prompt), request, sink, renderedPrompt, constraint, null);
    }

    /**
     * {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)} with a per-request
     * reasoning filter, which replaces the engine's for this sequence (a server request's
     * {@code reasoning_format}); null runs the engine's.
     */
    public BatchSequence submit(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt, com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint,
                                ReasoningFilter filterOverride) {
        return submit(MediaPrompt.ofText(prompt), request, sink, renderedPrompt, constraint, filterOverride);
    }

    /** {@link #submit(int[], GenerationRequest, ReasoningFilter.Sink, String)} with a content-only callback. */
    public BatchSequence submit(int[] prompt, GenerationRequest request, TokenCallback callback) {
        return submit(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null, null);
    }

    /** {@link #submit(MediaPrompt, GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)} with a
     * content-only callback. */
    public BatchSequence submit(MediaPrompt prompt, GenerationRequest request, TokenCallback callback) {
        return submit(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null, null);
    }

    /**
     * Submits a media-aware prompt ({@link MediaPrompt}: text and media chunks) for generation, validated like a
     * token prompt against the context's rows. Media chunks prefill as embedding batches with their own positions
     * and mask rule, identified media chunks share their encoder output through the engine's encoder cache, and
     * the prompt's rows are keyed for the prefix cache at once.
     *
     * @throws IllegalStateException         when the engine is closed
     * @throws ContextFullException          when the prompt does not fit the context
     * @throws IllegalArgumentException      when a media chunk starts off the prompt's position cursor, the
     *                                       constraint compiles to an unusable grammar, the request selects an
     *                                       adapter that is not loaded ({@code lora}), scores a token id outside
     *                                       the vocabulary ({@code logprob_token_ids}), or its
     *                                       {@code prompt_logprob_token_ids} rows do not match the prompt, or its
     *                                       {@code priority} is not 0 under the {@code fcfs} policy, or the
     *                                       engine verifies by another mode than {@code match} and the request's
     *                                       chain ends in {@code adaptive_p}
     * @throws UnsupportedOperationException when the prompt carries media chunks and the backend's batched forward
     *                                       does not take embedding rows ({@code ForwardBackend.embeddingRowsRefusal}),
     *                                       or the request selects adapters on a model that runs none per request
     */
    public BatchSequence submit(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt,
                                com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint) {
        return submit(prompt, request, sink, renderedPrompt, constraint, null);
    }

    /**
     * Refuses a watermarked request this engine cannot carry, naming what would make it possible: vLLM runs a
     * watermark under speculative decoding only through standard rejection sampling
     * ({@code rejection_sample_method='standard'}, {@link VerifyMode#REJECTION} here), and only when the drafter can
     * sample its proposals under the draft role ({@link BatchDrafter#supportsWatermarking()}).
     *
     * @throws IllegalArgumentException when the watermark cannot be carried
     */
    private void requireWatermarkSupport(WatermarkConfig watermark) {
        if (watermark == null) {
            return;
        }
        watermark.draftWatermarker(); // refuses a single-key configuration without allow_target_only_watermarking
        if (spec.verifyMode() != VerifyMode.REJECTION) {
            throw new IllegalArgumentException("watermarking: speculative decoding carries the watermark only"
                    + " through rejection verification (vLLM's standard rejection sampler), not " + spec.verifyMode()
                    + "; set neuralfree.draft.verify=rejection or disable the watermark");
        }
        if (drafter == null || !drafter.supportsWatermarking()) {
            throw new IllegalArgumentException("watermarking: the batch drafter " + drafter + " proposes stored"
                    + " tokens and cannot sample them under the draft watermark key; a watermarked speculative request"
                    + " needs a draft model");
        }
    }

    /**
     * {@link #submit(MediaPrompt, GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)} with a
     * per-request reasoning filter, which replaces the engine's for this sequence (a server request's
     * {@code reasoning_format}); null runs the engine's.
     */
    public BatchSequence submit(MediaPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt,
                                com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint,
                                ReasoningFilter filterOverride) {
        if (support == SupportClass.EMBED || support == SupportClass.CLS) {
            throw new UnsupportedOperationException(support + " models pool; the batch engine over them takes"
                    + " pooling requests only (submitPooling), not generation");
        }
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(sink, "sink");
        checkOpen();
        String priorityRefusal = state.policy.priorityRefusal(request.priority());
        if (priorityRefusal != null) {
            throw new IllegalArgumentException(priorityRefusal);
        }
        int[] source = state.encoder == null ? null : encoderSource(prompt, request);
        MediaPrompt rows = source == null ? prompt
                : MediaPrompt.ofText(new int[]{state.encoder.decoderStartToken()});
        checkChunkPositions(rows);
        checkEmbeddingRows(rows);
        checkContextShiftRequest(rows, request);
        checkSelfExtendRequest(rows, request);
        checkLookaheadRequest(rows, request);
        checkChunkReuseRequest(rows, request);
        if (rows.nTokens() > state.nCtx) {
            throw new ContextFullException(rows.nTokens(), state.nCtx);
        }
        request.requireScoredIds(rows.nTokens(), tokenizer.vocabSize());
        com.eonmux.neuralfree.gguf.grammar.ConstraintSpec spec =
                constraint != null ? constraint : request.constraint();
        com.eonmux.neuralfree.gguf.grammar.CompiledConstraint grammar = spec == null ? null
                : com.eonmux.neuralfree.gguf.grammar.GbnfBuilder.compile(spec,
                        text -> tokenizer.encode(text, false, true));
        if (this.spec != null) {
            requireWatermarkSupport(request.withModelDefaults(samplingDefaults).sampler().watermark());
            VerifyRound.refuseAdaptiveP(this.spec.verifyMode(),
                    request.withModelDefaults(samplingDefaults).sampler());
        }
        LoraPlan adapters = resolveAdapters(request.lora(), rows);
        Set<String> mediaIds = new LinkedHashSet<>();
        MediaPrompt run = rows.withSharedMedia((id, r) -> state.encoderCache.reference(id, r, mediaIds));
        BatchSequence handle = new BatchSequence(nextSubmitId++);
        Submission sub;
        try {
            sub = new Submission(handle, run, source, request, sink, renderedPrompt, grammar, filterOverride,
                    adapters, mediaIds, null);
            if (state.cache != null) {
                sub.promptRows(); // hashes the media rows at submit, not inside an engine step
            }
        } catch (RuntimeException e) {
            state.encoderCache.release(mediaIds);
            throw e;
        }
        state.queue.add(sub);
        return handle;
    }

    /**
     * The adapter slots of a request's rows ({@code GenerationRequest#lora}), resolved against the model's registry
     * now so an unknown adapter refuses before anything queues: the defaults of the selection's scales freeze
     * here, and an activated LoRA finds its invocation start in the prompt ({@link LoraPlan#resolve}), a media
     * prompt's among its text tokens with the match mapped to its KV row. The default plan for a request that
     * selects none, or whose selection equals the startup adapters — a startup activated LoRA's plan from its
     * invocation start then.
     *
     * @throws UnsupportedOperationException when the request selects adapters and the model runs none per request
     * @throws IllegalArgumentException      naming an adapter that is not loaded, or a selection the registry refuses
     */
    private LoraPlan resolveAdapters(LoraSelection selection, MediaPrompt prompt) {
        if (selection == null) {
            return state.adapters == null ? LoraPlan.DEFAULT : LoraPlan.resolve(state.adapters, null, prompt);
        }
        if (state.adapters == null) {
            throw new UnsupportedOperationException("lora: this model runs no per-request LoRA adapters ("
                    + selection + "); the audio decoder serves the base weights alone");
        }
        return LoraPlan.resolve(state.adapters, selection, prompt);
    }

    /**
     * The encoder source of an encoder-decoder request: its token prompt, within the encoder's source limit (the
     * encoder runs a source in one ubatch). Media prompts refuse, as {@code GenerationLoop} takes token prompts only,
     * and so do {@code prompt_logprobs} and {@code prompt_logprob_token_ids}: the decoder does not predict the
     * source.
     *
     * @throws ContextFullException          when the source is longer than the encoder's limit
     * @throws UnsupportedOperationException for a media prompt or a request that scores its prompt rows
     */
    private int[] encoderSource(MediaPrompt prompt, GenerationRequest request) {
        if (!prompt.isTextOnly()) {
            throw new UnsupportedOperationException("encoder-decoder models take token prompts; a prompt with media"
                    + " chunks or embedding rows has no encoder source");
        }
        if (request.readsPromptRows()) {
            throw new UnsupportedOperationException(request.promptRowsField() + " score prompt rows the model"
                    + " predicts; an encoder-decoder's prompt is its encoder source, which the decoder does not"
                    + " predict");
        }
        int[] source = prompt.samplerTokens();
        int limit = state.encoder.maxSourceTokens();
        if (source.length > limit) {
            throw new ContextFullException(source.length, limit);
        }
        return source;
    }

    /**
     * Refuses a media prompt — embedding rows ({@code prompt_embeds}, media parts) — when the backend's
     * batched forward cannot take them, before the prompt queues, so the refusal reaches its request alone.
     */
    private void checkEmbeddingRows(MediaPrompt prompt) {
        if (prompt.isTextOnly()) {
            return;
        }
        String refusal = backend.embeddingRowsRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("the prompt's embedding rows (prompt_embeds, media parts) do"
                    + " not run on " + backend.getClass().getSimpleName() + ": " + refusal);
        }
    }

    /**
     * Refuses a request whose {@code n_keep}/{@code n_discard} names a sequence the context shift cannot serve,
     * before the prompt queues: an encoder-decoder's first decoder row is its fixed start token, and a prompt
     * with media chunks or embedding rows stores rows that share M-RoPE positions — neither survives a position
     * move.
     *
     * @throws UnsupportedOperationException naming {@code n_keep}/{@code n_discard} when either is set
     */
    private void checkContextShiftRequest(MediaPrompt rows, GenerationRequest request) {
        if (request.nKeep() == 0 && request.nDiscard() == 0) {
            return;
        }
        if (state.encoder != null) {
            throw new UnsupportedOperationException("n_keep/n_discard shape the context shift of a decoder-only"
                    + " token prompt; an encoder-decoder's first decoder row is its fixed start token");
        }
        if (!rows.isTextOnly()) {
            throw new UnsupportedOperationException("n_keep/n_discard shape the context shift of a token prompt;"
                    + " a prompt with media chunks or embedding rows stores rows that share M-RoPE positions");
        }
    }

    /**
     * Refuses a generation request the lookahead (Jacobi) scheme cannot serve, before it queues: its step writes one
     * row per position over a group of sequences, so a prompt with media chunks or embedding rows (rows that share
     * M-RoPE positions), a self-extend compression (which moves the stored positions), the context shift's
     * {@code n_keep}/{@code n_discard}, and per-request adapters (the step's batch carries no adapter slots) refuse
     * by name. A plain generation with no lookahead passes.
     */
    private void checkLookaheadRequest(MediaPrompt rows, GenerationRequest request) {
        if (state.lookahead == null) {
            return;
        }
        if (!rows.isTextOnly()) {
            throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " runs its step with one row"
                    + " per position; a prompt with media chunks or embedding rows stores rows that share M-RoPE"
                    + " positions");
        }
        if (request.selfExtend()) {
            throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " computes the Jacobi window"
                    + " over the stored positions; grp_attn_n/grp_attn_w compress them");
        }
        if (request.nKeep() != 0 || request.nDiscard() != 0) {
            throw new UnsupportedOperationException(LookaheadOptions.PROP_ENABLED + " cannot shift its context, so"
                    + " n_keep/n_discard cannot shape it");
        }
        if (request.lora() != null) {
            throw new UnsupportedOperationException("lora: " + LookaheadOptions.PROP_ENABLED + " runs the target's"
                    + " startup adapters only; it does not select per-request adapters (" + request.lora() + ")");
        }
    }

    /**
     * Refuses a request whose {@code grp_attn_n}/{@code grp_attn_w} asks for a compression this engine cannot
     * serve, before the prompt queues: a token prompt of a decoder-only model over a memory whose positions the
     * backend can edit and whose rows no context checkpoint holds, and no context shift beside it (one sequence
     * cannot both compress its positions and drop windows of them).
     *
     * @throws UnsupportedOperationException naming {@code grp_attn_n}/{@code grp_attn_w} when the request cannot
     *                                       run them
     */
    private void checkSelfExtendRequest(MediaPrompt rows, GenerationRequest request) {
        if (!request.selfExtend()) {
            return;
        }
        if (request.nKeep() != 0 || request.nDiscard() != 0) {
            throw new UnsupportedOperationException("grp_attn_n/grp_attn_w compress the stored positions while"
                    + " n_keep/n_discard drop and shift them; a sequence runs one of the two");
        }
        if (state.encoder != null) {
            throw new UnsupportedOperationException("grp_attn_n/grp_attn_w group the positions of a decoder-only"
                    + " token prompt; an encoder-decoder's first decoder row is its fixed start token");
        }
        if (!rows.isTextOnly()) {
            throw new UnsupportedOperationException("grp_attn_n/grp_attn_w group a token prompt's positions;"
                    + " a prompt with media chunks or embedding rows stores rows that share M-RoPE positions");
        }
        if (state.checkpoints != null) {
            throw new UnsupportedOperationException("grp_attn_n/grp_attn_w need a memory whose rows keep their"
                    + " positions; this memory restores context checkpoints (recurrent state or a sliding window),"
                    + " whose state a compression would invalidate");
        }
        if (!selfExtend.supported()) {
            throw new UnsupportedOperationException("grp_attn_n/grp_attn_w need per-sequence position edits: "
                    + backend.sequenceEditRefusal());
        }
    }

    /**
     * Refuses a request whose {@code n_cache_reuse} (llama.cpp's {@code --cache-reuse}) asks for a KV chunk reuse
     * this engine cannot serve, before the prompt queues: a token prompt of a decoder-only model, over a prefix
     * cache of whole-row pages (no context checkpoints — chunk reuse re-appends rows their state cannot account),
     * whose sequences the backend can edit with a re-rotation of the re-appended keys.
     *
     * @throws UnsupportedOperationException naming {@code n_cache_reuse} when the request cannot run it
     */
    private void checkChunkReuseRequest(MediaPrompt rows, GenerationRequest request) {
        if (request.nCacheReuse() <= 0) {
            return;
        }
        if (state.encoder != null) {
            throw new UnsupportedOperationException("n_cache_reuse re-appends the cached rows of a decoder-only"
                    + " token prompt; an encoder-decoder's first decoder row is its fixed start token");
        }
        if (!rows.isTextOnly()) {
            throw new UnsupportedOperationException("n_cache_reuse re-appends a token prompt's cached rows;"
                    + " a prompt with media chunks or embedding rows stores rows that share M-RoPE positions");
        }
        if (state.cache == null) {
            throw new UnsupportedOperationException("n_cache_reuse reuses the prefix cache's cached rows; this"
                    + " engine runs no prefix cache, so no rows stand to re-append");
        }
        if (state.checkpoints != null) {
            throw new UnsupportedOperationException("n_cache_reuse needs a prefix cache of whole-row pages; this"
                    + " memory restores a cached prefix only at a context checkpoint (recurrent state or a"
                    + " sliding window), whose rows a re-append would move");
        }
        if (!chunkReuse.supported()) {
            throw new UnsupportedOperationException("n_cache_reuse needs per-sequence edits that re-rotate the"
                    + " re-appended keys: " + backend.sequenceEditRefusal());
        }
    }

    /**
     * Refuses a prompt of precomputed embedding rows ({@link EmbedsPrompt}, vLLM's {@code prompt_embeds}) as the
     * media prompt of its runs ({@link EmbedsPrompt#toMediaPrompt()}): token runs prefill as text chunks, row runs
     * as prompt-embedding chunks at one position per row, and the prefix cache keys every row by its token or by
     * the digest of its embedding row. The token runs enter the sampler history; the rows do not.
     *
     * @throws IllegalStateException    when the engine is closed
     * @throws ContextFullException     when the prompt rows do not fit the context
     * @throws IllegalArgumentException when the constraint compiles to an unusable grammar
     */
    public BatchSequence submit(EmbedsPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt,
                                com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint,
                                ReasoningFilter filterOverride) {
        Objects.requireNonNull(prompt, "prompt");
        return submit(prompt.toMediaPrompt(), request, sink, renderedPrompt, constraint, filterOverride);
    }

    /** {@link #submit(EmbedsPrompt, GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec,
     * ReasoningFilter)} with the engine's reasoning filter. */
    public BatchSequence submit(EmbedsPrompt prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                String renderedPrompt,
                                com.eonmux.neuralfree.gguf.grammar.ConstraintSpec constraint) {
        return submit(prompt, request, sink, renderedPrompt, constraint, null);
    }

    /** {@link #submit(EmbedsPrompt, GenerationRequest, ReasoningFilter.Sink, String, ConstraintSpec)} with a
     * content-only callback and no constraint. */
    public BatchSequence submit(EmbedsPrompt prompt, GenerationRequest request, TokenCallback callback) {
        return submit(prompt, request, callback == null ? DISCARD : ReasoningFilter.contentTo(callback), null, null);
    }

    /**
     * Submits a pooling request (vLLM's pooling tasks beside generation, llama.cpp's embedding tasks in slots of
     * the shared batch): {@code tokens} queue like a generation prompt, prefill under the same token budget beside
     * the running sequences — sharing the prefix cache where the output allows it — and at the prompt's completion
     * the model's pooling operator applies over the prompt's final hidden rows and the output lands on the returned
     * handle. A causal model's pooling prompt prefills in budget-bounded chunks whose hidden rows the request
     * accumulates; a bidirectional model's prefills whole (its rows attend each other), within one ubatch. A
     * request whose output pools every row (MEAN, CLS, per-token, decision scores) reuses no cached prefix, since
     * the reused rows' hidden states are not its own forward's; a last-row output (LAST pooling, the RANK
     * classifier over causal attention) reuses like a generation prompt. The engine's prefix cache never wipes for
     * a pooling request, and the request's committed rows feed it.
     *
     * @return the handle whose {@link PoolingSequence#pooled()} carries the output once
     *         {@link BatchSequence#finished()} reports done
     * @throws IllegalStateException         when the engine is closed
     * @throws UnsupportedOperationException when the model has no pooling heads, is an encoder-decoder, cannot
     *                                       answer {@code output} (naming the pooling kind), or — bidirectional —
     *                                       the prompt exceeds one ubatch
     * @throws ContextFullException          when the prompt does not fit the context
     * @throws IllegalArgumentException      when the prompt is empty or holds a negative token id
     */
    public PoolingSequence submitPooling(int[] tokens, PoolingOutput output) {
        return submitPooling(tokens, output, null);
    }

    /**
     * {@link #submitPooling(int[], PoolingOutput)} with the request's per-request adapters ({@code selection},
     * null for the startup adapters): the pooling rows run in the selection's adapter slot, and the request's
     * prefix key mixes the adapters' identity like a generation request's.
     *
     * @throws UnsupportedOperationException when the request selects adapters the model runs none per request, or
     *                                       an adapter that is not loaded
     */
    public PoolingSequence submitPooling(int[] tokens, PoolingOutput output, LoraSelection selection) {
        return submitPooling(tokens, output, selection, true);
    }

    /**
     * {@link #submitPooling(int[], PoolingOutput, LoraSelection)} with the classifier's final probability
     * activation applied or skipped (vLLM's {@code use_activation}); {@code false} answers the raw head output,
     * an embedding output pools the same either way.
     *
     * @throws UnsupportedOperationException when the request selects adapters the model runs none per request, or
     *                                       an adapter that is not loaded
     */
    public PoolingSequence submitPooling(int[] tokens, PoolingOutput output, LoraSelection selection,
                                         boolean activation) {
        checkOpen();
        if (poolingOperator == null) {
            throw new UnsupportedOperationException("submitPooling: the model carries no pooling or classifier"
                    + " head; a pooling request needs an embedding model, a classifier or a pooling type");
        }
        if (state.encoder != null) {
            throw new UnsupportedOperationException("submitPooling: an encoder-decoder model pools no inputs;"
                    + " its pooling output would read its decoder's rows");
        }
        poolingOperator.validate(output);
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("tokens must not be empty");
        }
        for (int token : tokens) {
            if (token < 0) {
                throw new IllegalArgumentException("token id " + token + " is negative");
            }
        }
        if (tokens.length > state.nCtx) {
            throw new ContextFullException(tokens.length, state.nCtx);
        }
        if (!causal && tokens.length > ubatch) {
            throw new UnsupportedOperationException("a bidirectional model pools a prompt its attention reads in"
                    + " one piece: the prompt of " + tokens.length + " tokens exceeds the " + ubatch
                    + "-token ubatch (neuralfree.ubatch)");
        }
        MediaPrompt prompt = MediaPrompt.ofText(tokens);
        LoraPlan adapters = resolveAdapters(selection, prompt);
        BatchSequence handle = new BatchSequence(nextSubmitId++);
        PoolingSequence pooling = new PoolingSequence(handle, output);
        Submission sub = new Submission(handle, prompt, GenerationRequest.builder().build(), DISCARD, null, null,
                null, adapters, List.of(), new PoolingRun(output, pooling, poolingOperator.readsAllRows(output),
                activation));
        if (state.cache != null) {
            sub.promptRows(); // hashes the prompt at submit, not inside an engine step
        }
        state.queue.add(sub);
        return pooling;
    }

    /**
     * Runs one engine step: admits queued requests while a slot and the pages of its prompt are free, then
     * schedules the running sequences first-come first-served within the step's token budget — one row per
     * decoding sequence (a verify round's rows when the engine speculates) and a bounded chunk of the rows a
     * prefilling sequence still lacks — and evaluates them in one forward batch. With chunked prefill off the
     * step keeps the older shape: the oldest waiting prefill runs alone, or one decode row per running sequence.
     * A pump over no submissions does nothing. Every routed piece reaches the submission's sink before the call
     * returns; finished sequences carry their result on their handle before it returns too. The rows the step
     * committed then feed the prefix cache's block pool, and its events publish as one batch.
     */
    public void pump() {
        checkOpen();
        if (state.cancelPending && idle()) {
            state.cancelPending = false; // the cancelled ones retired: later submissions run again
        }
        admitter.admit();
        selfExtend.apply();
        scheduler.step();
        state.cacheCommitted();
        publishEvents();
    }

    /**
     * Pumps until every submission finished, failed or was cancelled, then lets the disk tier's admitted spills
     * become durable and publishes their events: a drained engine has left its prefix cache on the disk.
     *
     * @throws RuntimeException when a pump step throws after retiring the sequences it could not attribute (see
     *                          the class doc); queued submissions then still drain through later calls
     */
    public void drain() {
        while (!idle()) {
            pump();
        }
        if (state.cache != null) {
            state.cache.awaitDisk(CACHE_DRAIN_MILLIS);
        }
        publishEvents();
    }

    /**
     * Frees every sequence slot and queued submission, then closes the drafter the engine built from the config
     * (a caller-supplied drafter stays the caller's) and the KV event publisher, after publishing its last events.
     * The backing model stays open (the caller owns it); further submits refuse.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        lifecycle.failAll("batch engine closed");
        if (ownsDrafter && drafter != null) {
            try {
                drafter.close();
            } catch (RuntimeException e) {
                LOGGER.warning("closing the batch engine's drafter failed: " + e);
            }
        }
        if (state.cache != null) {
            state.cache.close();
        }
        publishEvents();
        try {
            events.close();
        } catch (RuntimeException e) {
            LOGGER.warning("closing the batch engine's KV event publisher failed: " + e);
        }
    }

    /**
     * Checks a prompt's media chunks sit on the prompt's position cursor: each chunk continues the previous one's
     * positions (as llama.cpp's {@code server_tokens} builds them), so the prompt's chunks prefill as one
     * contiguous run.
     *
     * @throws IllegalArgumentException when a media chunk's base position differs from the cursor
     */
    private static void checkChunkPositions(MediaPrompt prompt) {
        int pos = 0;
        List<MediaChunk> chunks = prompt.chunks();
        for (int i = 0; i < chunks.size(); i++) {
            MediaChunk c = chunks.get(i);
            if (!c.isText() && c.positions().pos0() != pos) {
                throw new IllegalArgumentException("chunk " + i + " starts at position " + c.positions().pos0()
                        + ", the prompt's cursor holds " + pos + " (each chunk continues the previous one's"
                        + " positions)");
            }
            pos += c.nPositions();
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("batch engine is closed");
        }
    }
}
