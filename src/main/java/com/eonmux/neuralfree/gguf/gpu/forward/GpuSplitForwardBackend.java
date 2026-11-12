package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The {@link ForwardBackend} of a layer split: the devices run their layer spans in pipeline order, the embedding
 * and the output head run on the main device, and the residual stream moves between two devices through the host
 * at every boundary their spans do not share (the host-staged counterpart of the copies
 * {@code ggml_backend_sched} inserts). Every device holds the KV caches of its own layers, its own per-step inputs
 * and its own sequence record; the records move in lockstep behind one {@link SplitMemory}, so the caller sees one
 * memory. When the first trunk layers stay on the host ({@link SplitHostStage}, llama.cpp {@code -ngl} over
 * {@code --split-mode layer}), the embedding and those layers run on the CPU over their heap memory, and their
 * stream is uploaded to the first device.
 *
 * <p>Per step of {@code n} tokens: the per-step inputs of every device, the positions of every device, the
 * embedding on the main device (or the host part), then the spans in order with the stream hand-off between them,
 * then the head on the main device on the requested rows. The call enqueues on every device and synchronizes once,
 * when the requested logits, hidden rows or greedy ids are read back. A shard whose layers stream their weights
 * ({@link GpuForwardContext#weightSource()}) has its first layer prefetched before the embedding runs.</p>
 *
 * <p>A one-token decode step runs each span through the shard's own {@link GpuDecodeReplay} on the shard's queue, as
 * the single-device driver runs its trunk: after the warmup the span's dispatches are captured and later steps
 * replay them as one graph per device, while the per-step inputs, the stream hand-offs, the embedding and the head
 * stay outside the captures. A prefill step or a memory reset leaves the captures to be verified again; a context
 * shift, a new hidden tap and a rollback that is not host bookkeeping only drop them.</p>
 *
 * <p>Every shard's context carries its own {@link ExecutionObserver}: the stages of a span report to the observer of
 * the device that runs them, the embedding and the head to the main device's, and every observer sees the start
 * and the end of each step. Not thread-safe.</p>
 */
public final class GpuSplitForwardBackend implements ForwardBackend {

    /** One device of the split: its context, its span executor and its own memory. */
    public static final class Shard {
        private final GpuForwardContext context;
        private final GpuLayerExecutor executor;
        private final DeviceModelMemory memory;
        private final GpuStepPositions positions;
        private final ScratchSlot x;
        private final int firstIl;
        private final int firstWidth;
        private final int lastWidth;
        private final int ubatch;

        /**
         * A shard over the span the executor runs, whose first layer is stack slot {@code firstIl}.
         *
         * @param memory the device memory of the span's layers, or null for a model without memory
         */
        public Shard(GpuForwardContext context, GpuLayerExecutor executor, DeviceModelMemory memory, int firstIl,
                     int firstWidth, int lastWidth) {
            this.context = Objects.requireNonNull(context, "context");
            this.executor = Objects.requireNonNull(executor, "executor");
            this.memory = memory;
            this.positions = new GpuStepPositions(context.scratch(), context.queue(),
                    context.spec().hparams().mropeTextPositions());
            this.x = context.scratch().slot(ScratchSlots.X).require((long) context.scratch().ubatch()
                    * Math.max(firstWidth, lastWidth));
            this.firstIl = firstIl;
            this.firstWidth = firstWidth;
            this.lastWidth = lastWidth;
            this.ubatch = context.scratch().ubatch();
        }

        /** The context the embedding stage and the output head of the main shard bind to. */
        public GpuForwardContext context() { return context; }
        GpuLayerExecutor executor() { return executor; }
        DeviceModelMemory memory() { return memory; }
        ComputeQueue queue() { return context.queue(); }
        GpuStepPositions positions() { return positions; }
        ScratchSlot x() { return x; }
        int firstIl() { return firstIl; }
        int firstWidth() { return firstWidth; }
        int lastWidth() { return lastWidth; }
        int ubatch() { return ubatch; }

        @Override
        public String toString() {
            return "Shard{" + context.backend().backendType() + " " + context.backend().capabilities().getDeviceName()
                    + ", layers " + firstIl + ".." + (firstIl + executor.layers().size() - 1) + '}';
        }
    }

    private final ModelSpec spec;
    private final List<Shard> shards;
    private final int main;
    /** The embedding on the main device; null when the host part embeds. */
    private final GpuEmbeddingStage embedding;
    private final GpuLogitsHead head;
    private final SplitMemory memory;
    private final SplitHostStage host;
    private HiddenTap tap = HiddenTap.NONE;
    private final int ubatch;
    private float[] transferStage;
    /** The decode replay of every shard's span, in shard order, each on its shard's queue. */
    private final List<GpuDecodeReplay> replays;
    /** Whether a rollback keeps the captures ({@link GpuForwardBackend#CHAIN_REPLAY_PROPERTY}), as on one device. */
    private final boolean chainReplay;
    private boolean closed;

    /**
     * Binds the split over already-built shards; the embedding and the head were bound to shard {@code main}, and
     * every layer runs on a device.
     *
     * @param memory the main shard's memory, whose plan the split memory reports, or null for a model without memory
     */
    public GpuSplitForwardBackend(ModelSpec spec, List<Shard> shards, int main, GpuEmbeddingStage embedding,
                                  GpuLogitsHead head, DeviceModelMemory memory) {
        this(spec, shards, main, embedding, head, memory == null ? null : memory.plan(), null);
    }

    /**
     * Binds the split over already-built shards and, when the first trunk layers stay on the host, the host part;
     * the head and the device embedding were bound to shard {@code main}. With a host part the host embeds, so no
     * device embedding is bound: a stage that never runs serves no {@code token_embd} adapter deltas. The backend
     * closes the shards' executors, the head and the host part.
     *
     * @param embedding the embedding on shard {@code main}; null exactly when {@code host} is not null
     * @param plan      the memory plan of the whole model, or null for a model without memory
     * @param host      the host part running layers {@code [0, host.layers())}, or null when every layer runs on a
     *                  device
     * @throws IllegalArgumentException when the host part leaves a stream of another width than the first device
     *                                  reads, a model without memory has a host part, or a device embedding is
     *                                  given beside a host part
     */
    public GpuSplitForwardBackend(ModelSpec spec, List<Shard> shards, int main, GpuEmbeddingStage embedding,
                                  GpuLogitsHead head, MemoryPlan plan, SplitHostStage host) {
        this.spec = Objects.requireNonNull(spec, "spec");
        this.shards = Collections.unmodifiableList(new ArrayList<>(Objects.requireNonNull(shards, "shards")));
        if (shards.isEmpty()) {
            throw new IllegalArgumentException("a split forward backend needs at least one shard");
        }
        if (main < 0 || main >= shards.size()) {
            throw new IllegalArgumentException("main device " + main + " is outside the " + shards.size()
                    + " shard(s)");
        }
        this.main = main;
        if (host == null) {
            Objects.requireNonNull(embedding, "embedding");
        } else if (embedding != null) {
            throw new IllegalArgumentException("the host part embeds the tokens of a split with host layers; bind no"
                    + " device embedding beside it");
        }
        this.embedding = embedding;
        this.head = Objects.requireNonNull(head, "head");
        this.host = host;
        this.memory = plan == null ? null : splitMemory(plan, this.shards, main, host);
        if (host != null) {
            if (plan == null) {
                throw new IllegalArgumentException("a model without memory runs no host layers");
            }
            if (host.outputWidth() != this.shards.get(0).firstWidth()) {
                throw new IllegalArgumentException("the host layers leave a stream of " + host.outputWidth()
                        + " floats, the first device reads " + this.shards.get(0).firstWidth());
            }
        }
        this.ubatch = host == null ? this.shards.get(0).ubatch()
                : Math.min(this.shards.get(0).ubatch(), host.ubatch());
        List<GpuDecodeReplay> shardReplays = new ArrayList<>(this.shards.size());
        for (Shard shard : this.shards) {
            shardReplays.add(new GpuDecodeReplay(shard.queue()));
        }
        this.replays = Collections.unmodifiableList(shardReplays);
        this.chainReplay = GpuForwardBackend.chainReplayConfigured(
                this.shards.get(main).context().backend().backendType());
    }

    /** The memory over every shard's memory and the host part's, reporting the main shard's sequence. */
    private static SplitMemory splitMemory(MemoryPlan plan, List<Shard> shards, int main, SplitHostStage host) {
        List<ModelMemory> sides = new ArrayList<>(shards.size() + 1);
        if (host != null) {
            sides.add(host.memory());
        }
        for (Shard shard : shards) {
            sides.add(Objects.requireNonNull(shard.memory(), "a model with memory needs every shard's memory"));
        }
        return new SplitMemory(plan, sides, shards.get(main).memory());
    }

    // ================================================================== causal calls

    @Override
    public float[] prefill(int[] tokens, int pos0) {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("prefill without tokens");
        }
        float[][] rows = run(tokens, null, pos0, new int[]{tokens.length - 1}, GpuPrefillDriver.Output.LOGITS,
                0, true).rows();
        return rows[0];
    }

    @Override
    public float[] decode(int token, int pos) {
        return run(new int[]{token}, null, pos, new int[]{0}, GpuPrefillDriver.Output.LOGITS, 0, true).rows()[0];
    }

    /** Greedy selection on the main device ({@code LLM_Argmax}); only the id is read back. */
    @Override
    public int decodeGreedyOnDevice(int token, int pos) {
        return run(new int[]{token}, null, pos, new int[]{0}, GpuPrefillDriver.Output.ARGMAX, 0, true).ids()[0];
    }

    @Override
    public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return run(tokens, null, pos0, outputRows, GpuPrefillDriver.Output.LOGITS, 0, true).rows();
    }

    /** Greedy selection of every requested row on the head's device; only the ids are read back. */
    @Override
    public int[] forwardRowsGreedy(int[] tokens, int pos0, int[] outputRows) {
        Objects.requireNonNull(tokens, "tokens");
        return run(tokens, null, pos0, outputRows, GpuPrefillDriver.Output.ARGMAX, 0, true).ids();
    }

    /** Whether the head's device row top-k readout runs (its {@code LLM_Row_TopK} kernel). */
    @Override
    public boolean topKOnDevice() {
        ensureOpen();
        return head.rowTopKAvailable();
    }

    /** Whether the head's prefilter kernel runs on this backend ({@code LLM_LogitsPrefilter}). */
    @Override
    public boolean topKPrefilterOnDevice() {
        ensureOpen();
        return head.prefilterAvailable();
    }

    /** Top candidates of every requested row on the head's device; only the candidates are read back. */
    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows) {
        return forwardRowsTopK(tokens, pos0, k, outputRows, null);
    }

    /** Top candidates of every requested row after the per-row prefilter (the logit bias, penalties and DRY). */
    @Override
    public RowTopK forwardRowsTopK(int[] tokens, int pos0, int k, int[] outputRows, LogitPrefilter prefilter) {
        Objects.requireNonNull(tokens, "tokens");
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        int[] rows = outputRows != null ? outputRows : identity(tokens.length);
        int width = Math.min(k, head.vocab());
        GpuPrefillDriver.Result r = run(tokens, null, pos0, outputRows, GpuPrefillDriver.Output.TOPK, width, true,
                prefilter);
        return RowTopK.packed(width, rows.length, r.topIds(), r.topLogits());
    }

    /** Top candidates of the decoded row on the head's device; only the candidates are read back. */
    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k) {
        return decodeTopKOnDevice(token, pos, k, null);
    }

    /** Top candidates of the decoded row after its prefilter (the logit bias, penalties and DRY). */
    @Override
    public RowTopK decodeTopKOnDevice(int token, int pos, int k, LogitPrefilter prefilter) {
        if (k < 1) {
            throw new IllegalArgumentException("top-k readback of " + k + " candidates");
        }
        int width = Math.min(k, head.vocab());
        GpuPrefillDriver.Result r = run(new int[]{token}, null, pos, new int[]{0}, GpuPrefillDriver.Output.TOPK,
                width, true, prefilter);
        return RowTopK.packed(width, 1, r.topIds(), r.topLogits());
    }

    /**
     * Rolls every memory back to {@code pos0} when the call re-evaluates stored positions; the memories hold
     * identical sequence records, so they roll back together or not at all.
     */
    private void align(int pos0, int nTokens) {
        if (pos0 < 0) {
            throw new IllegalArgumentException("position " + pos0 + " is negative");
        }
        int nCtx = memory.plan().nCtx();
        long end = (long) pos0 + nTokens;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(Integer.MAX_VALUE, end), nCtx);
        }
        int past = memory.sequence().nPast();
        if (pos0 > past) {
            throw new IllegalArgumentException("position " + pos0 + " leaves a gap after the " + past
                    + " stored positions");
        }
        if (pos0 < past) {
            int drop = past - pos0;
            if (!memory.canRollback(drop)) {
                throw new IllegalStateException("the call starts at " + pos0 + " but the memory holds " + past
                        + " positions and cannot drop the last " + drop);
            }
            rollbackReplays(drop);
        }
    }

    /**
     * Rolls the memory back by {@code drop} positions. The decode captures survive a rollback that is host
     * bookkeeping only (an uncompressed plan with the speculative round's replay on), their continuity guards
     * re-armed at the rewound position, as the single-device backend keeps its own; any other rollback drops them.
     */
    private void rollbackReplays(int drop) {
        int nextPos = memory.sequence().nPast() - drop;
        memory.rollback(drop);
        boolean keep = chainReplay && !memory.plan().hasCompressed();
        for (GpuDecodeReplay replay : replays) {
            if (keep) {
                replay.rearm(nextPos);
            } else {
                replay.invalidate();
            }
        }
    }

    // ================================================================== memoryless calls

    /**
     * @throws UnsupportedOperationException for a model with memory (its attention is causal)
     * @throws IllegalArgumentException      for more tokens than one ubatch
     */
    @Override
    public float[][] forwardAll(int[] tokens, int[] outputRows) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (memory != null) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " is planned with memory; "
                    + "non-causal evaluation needs a model without memory (masked diffusion, encoders)");
        }
        return run(tokens, null, 0, outputRows, GpuPrefillDriver.Output.LOGITS, 0, false).rows();
    }

    /**
     * @throws IllegalStateException    when a model with memory holds positions (they would be overwritten)
     * @throws IllegalArgumentException for a model without memory and more tokens than one ubatch
     */
    @Override
    public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (memory == null) {
            return run(tokens, tokenTypes, 0, null, GpuPrefillDriver.Output.HIDDEN, 0, false).rows();
        }
        int past = memory.sequence().nPast();
        if (past != 0) {
            throw new IllegalStateException("hidden rows of a model with memory need an empty memory, it holds "
                    + past + " positions; reset() first");
        }
        try {
            return run(tokens, tokenTypes, 0, null, GpuPrefillDriver.Output.HIDDEN, 0, true).rows();
        } finally {
            reset();
        }
    }

    // ================================================================== the step loop

    /**
     * Evaluates {@code tokens} at positions {@code pos0 ..} as a sequence of ubatch steps (llama.cpp
     * {@code llama_decode}), reading back {@code output} for {@code outputRows}; a call without memory evaluates
     * all tokens in one step.
     */
    private GpuPrefillDriver.Result run(int[] tokens, int[] types, int pos0, int[] outputRows,
                                        GpuPrefillDriver.Output output, int topK, boolean useMemory) {
        return run(tokens, types, pos0, outputRows, output, topK, useMemory, null);
    }

    private GpuPrefillDriver.Result run(int[] tokens, int[] types, int pos0, int[] outputRows,
                                        GpuPrefillDriver.Output output, int topK, boolean useMemory,
                                        LogitPrefilter prefilter) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("forward call without tokens");
        }
        if (types != null && types.length != tokens.length) {
            throw new IllegalArgumentException(types.length + " token types for " + tokens.length + " tokens");
        }
        int[] rows = outputRows != null ? outputRows.clone() : identity(tokens.length);
        if (rows.length == 0) {
            throw new IllegalArgumentException("forward call without output rows");
        }
        for (int r : rows) {
            if (r < 0 || r >= tokens.length) {
                throw new IllegalArgumentException("output row " + r + " is outside the " + tokens.length + " tokens");
            }
        }
        if (output != GpuPrefillDriver.Output.HIDDEN && !head.hasLmHead()) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no LM head; only hidden"
                    + " rows can be read");
        }
        if (!useMemory && tokens.length > ubatch) {
            throw new IllegalArgumentException(tokens.length + " tokens exceed the ubatch of " + ubatch
                    + "; a call without memory runs in one step, raise neuralfree.ubatch");
        }
        if (useMemory) {
            if (memory == null) {
                throw new IllegalStateException(spec.arch().ggufName() + " is planned without memory; causal"
                        + " evaluation needs one");
            }
            if (host != null) {
                host.checkInputs(tokens, types, pos0);
            }
            align(pos0, tokens.length);
        }
        GpuCallReads calls = new GpuCallReads(head, shards.get(main).queue(), rows.length, output, topK, prefilter);
        GpuTapReads[] taps = tapReads(tap, tokens.length);
        GpuTapReads hostTaps = new GpuTapReads(shards.get(main).queue(), tap, tokens.length, 0);
        boolean finished = false;
        try {
            for (int c0 = 0; c0 < tokens.length; ) {
                int n = useMemory ? memory.plan().stepTokens(tokens.length - c0, ubatch)
                        : Math.min(ubatch, tokens.length - c0);
                for (GpuTapReads t : taps) {
                    t.chunk(c0);
                }
                hostTaps.chunk(c0);
                step(tokens, types, pos0 + c0, c0, n, rows, output, useMemory, calls, taps, hostTaps);
                if (useMemory) {
                    memory.append(tokens, c0, n);
                }
                c0 += n;
            }
            finished = true;
            finishAll();
            hostTaps.deliver();
            for (GpuTapReads t : taps) {
                t.deliver();
            }
            return calls.result();
        } catch (RuntimeException e) {
            if (!finished) {
                for (Shard shard : shards) {
                    try {
                        shard.queue().finish();
                    } catch (RuntimeException drain) {
                        e.addSuppressed(drain);
                    }
                }
            }
            throw e;
        } finally {
            calls.closeEvents();
            for (GpuTapReads t : taps) {
                t.closeEvents();
            }
        }
    }

    private GpuTapReads[] tapReads(HiddenTap hiddenTap, int total) {
        GpuTapReads[] out = new GpuTapReads[shards.size()];
        for (int s = 0; s < shards.size(); s++) {
            Shard shard = shards.get(s);
            out[s] = new GpuTapReads(shard.queue(), hiddenTap, total, shard.firstIl());
        }
        return out;
    }

    /**
     * The tap sink of shard {@code s}: a boundary tap point is the stack output of the device before it and the
     * stack input of the device after it, so every shard but the first skips its own input tap and the point is
     * captured once. The first shard captures its input: the host part captures only the inputs of its own layers.
     */
    private GpuLayerExecutor.TapSink shardSink(GpuTapReads reads, int s) {
        return s == 0 ? reads : new GpuLayerExecutor.TapSink() {
            @Override
            public boolean wants(int tapPoint) {
                return tapPoint > 0 && reads.wants(tapPoint);
            }

            @Override
            public void capture(int tapPoint, ScratchSlot x, int nTokens, int width) {
                reads.capture(tapPoint, x, nTokens, width);
            }
        };
    }

    /**
     * One ubatch step over every span, with the stream hand-offs between the devices; with a host part, its
     * embedding and layers first, their stream uploaded to the first device.
     */
    private void step(int[] tokens, int[] types, int p, int c0, int n, int[] rows, GpuPrefillDriver.Output output,
                      boolean useMemory, GpuCallReads calls, GpuTapReads[] taps, GpuTapReads hostTaps) {
        GpuLayerExecutor.StepMode mode = useMemory && n == 1 ? GpuLayerExecutor.StepMode.DECODE
                : GpuLayerExecutor.StepMode.PREFILL;
        observeStep(p, n, true);
        for (Shard shard : shards) {
            if (shard.memory() != null) {
                shard.memory().stepInputs().upload(p, n);
            }
            shard.executor().prefetchFirst();
            shard.positions().write(p, n);
        }
        int current;
        if (host != null) {
            host.beginStep(p, n);
            float[] x = host.forward(tokens, types, c0, n, p, hostTaps(hostTaps));
            uploadStream(shards.get(0), x, n);
            current = 0;
        } else {
            embedding.embed(tokens, c0, n, types, p);
            current = main;
        }
        int k = 0;
        int[] local = new int[rows.length];
        int[] targets = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            int r = rows[i];
            if (r >= c0 && r < c0 + n) {
                local[k] = r - c0;
                targets[k] = i;
                k++;
            }
        }
        // the last shard's last layer runs its FFN over the read rows alone (none for a chunk before the prompt's
        // last), which its stream then holds at the front: only those rows move to the head's device
        int last = shards.size() - 1;
        int[] read = Arrays.stream(local, 0, k).distinct().sorted().toArray();
        boolean pruned = shards.get(last).executor().prunesOutputRows(n, mode, shardSink(taps[last], last), read);
        for (int s = 0; s < shards.size(); s++) {
            Shard shard = shards.get(s);
            if (s != current) {
                transferStream(shards.get(current), shard, n);
                current = s;
            }
            if (s == last && pruned) {
                replays.get(s).distrust();
                shard.executor().run(p, n, mode, shardSink(taps[s], s), read);
            } else {
                runSpan(s, p, n, mode, taps[s]);
            }
        }
        int stackRows = pruned ? read.length : n;
        if (current != main && stackRows > 0) {
            transferStream(shards.get(current), shards.get(main), stackRows);
        }
        if (pruned) {
            for (int i = 0; i < k; i++) {
                local[i] = Arrays.binarySearch(read, local[i]);
            }
        }
        for (int done = 0; done < k; ) {
            int count = Math.min(n, k - done);
            ScratchSlot hidden = head.hidden(stackRows, Arrays.copyOfRange(local, done, done + count), count);
            calls.read(output, hidden, count, Arrays.copyOfRange(targets, done, done + count));
            done += count;
        }
        observeStep(p, n, false);
    }

    /**
     * Runs shard {@code s}'s span: a decode step through the shard's {@link GpuDecodeReplay}, capturable when no tap
     * point of the span is read and every layer only enqueues kernel dispatches; any other step directly, after
     * which the shard's capture is verified again before it is trusted.
     */
    private void runSpan(int s, int p, int n, GpuLayerExecutor.StepMode mode, GpuTapReads reads) {
        GpuLayerExecutor executor = shards.get(s).executor();
        GpuLayerExecutor.TapSink sink = shardSink(reads, s);
        GpuDecodeReplay replay = replays.get(s);
        if (mode == GpuLayerExecutor.StepMode.DECODE) {
            replay.trunk(p, executor.decodeReplayPatchers(),
                    !reads.wantsAny(executor.layers().size()) && executor.capturableDecode(),
                    () -> executor.run(p, n, mode, sink));
        } else {
            replay.distrust();
            executor.run(p, n, mode, sink);
        }
    }

    /** What every shard's decode replay has done, in shard order, for diagnostics and tests. */
    List<GpuDecodeReplay.Counters> decodeReplayCounters() {
        List<GpuDecodeReplay.Counters> out = new ArrayList<>(replays.size());
        for (GpuDecodeReplay replay : replays) {
            out.add(replay.counters());
        }
        return out;
    }

    private void invalidateReplays() {
        for (GpuDecodeReplay replay : replays) {
            replay.invalidate();
        }
    }

    /** The start or the end of a step, on the observer of every shard that observes. */
    private void observeStep(int p, int n, boolean start) {
        for (Shard shard : shards) {
            ExecutionObserver obs = shard.context().observer();
            if (!obs.enabled()) {
                continue;
            }
            if (start) {
                obs.onStepStart(p, n);
            } else {
                obs.onStepEnd(p, n);
            }
        }
    }

    /** The host part's captures, into the host reads of the call. */
    private static SplitHostStage.Taps hostTaps(GpuTapReads reads) {
        return new SplitHostStage.Taps() {
            @Override
            public boolean wants(int layer) {
                return reads.wants(layer);
            }

            @Override
            public void capture(int layer, float[] x, int n, int width) {
                reads.captureHost(layer, x, n, width);
            }
        };
    }

    /** Writes the host part's stream of {@code n} rows into the first device's stream slot. */
    private static void uploadStream(Shard dst, float[] x, int n) {
        long floats = (long) n * dst.firstWidth();
        GpuEvent write = dst.queue().enqueueWriteBuffer(dst.x().buffer(), (long) dst.x().offset() * Float.BYTES,
                floats * Float.BYTES, Arrays.copyOf(x, (int) floats));
        if (write != null) {
            write.close();
        }
    }

    /**
     * Moves the residual stream of {@code n} rows from one device to the next through the host: the rows are read
     * to a staging array once the source queue reaches them, then written into the target's stream slot. Different
     * stream widths on the two sides are repacked row by row.
     */
    private void transferStream(Shard src, Shard dst, int n) {
        int srcWidth = src.lastWidth();
        int dstWidth = dst.firstWidth();
        long bytes = (long) n * srcWidth * Float.BYTES;
        float[] stage = stage(n, Math.max(srcWidth, dstWidth));
        GpuEvent ev = src.queue().enqueueReadBuffer(src.x().buffer(), (long) src.x().offset() * Float.BYTES,
                bytes, stage);
        if (ev != null) {
            try {
                ev.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted during the device-to-device stream transfer", e);
            } finally {
                ev.close();
            }
        } else {
            src.queue().finish();
        }
        if (srcWidth != dstWidth) {
            widen(stage, n, srcWidth, dstWidth);
        }
        GpuEvent write = dst.queue().enqueueWriteBuffer(dst.x().buffer(), (long) dst.x().offset() * Float.BYTES,
                (long) n * dstWidth * Float.BYTES, stage);
        if (write != null) {
            write.close();
        }
    }

    /**
     * Spreads {@code n} rows of {@code srcWidth} floats at the start of {@code rows} to a row pitch of
     * {@code dstWidth} in place, zeroing each row's added columns; the last row moves first, so no row is
     * overwritten before it moved.
     *
     * @throws IllegalStateException when the destination rows are narrower than the source rows
     */
    static void widen(float[] rows, int n, int srcWidth, int dstWidth) {
        if (dstWidth < srcWidth) {
            throw new IllegalStateException("a stream of width " + srcWidth + " cannot enter a span of width "
                    + dstWidth);
        }
        for (int t = n - 1; t >= 0; t--) {
            System.arraycopy(rows, t * srcWidth, rows, t * dstWidth, srcWidth);
            Arrays.fill(rows, t * dstWidth + srcWidth, (t + 1) * dstWidth, 0f);
        }
    }

    /** The reused host staging array, at least {@code n * width} floats. */
    private float[] stage(int n, int width) {
        long floats = (long) n * width;
        if (transferStage == null || transferStage.length < floats) {
            transferStage = new float[(int) floats];
        }
        return transferStage;
    }

    private void finishAll() {
        for (Shard shard : shards) {
            shard.queue().finish();
        }
    }

    // ================================================================== memory

    @Override
    public void reset() {
        ensureOpen();
        if (memory != null) {
            memory.reset();
        }
        for (GpuDecodeReplay replay : replays) {
            replay.distrust();
        }
    }

    @Override
    public boolean canRollback(int nTokens) {
        ensureOpen();
        if (memory == null) {
            if (nTokens < 0) {
                throw new IllegalArgumentException("rollback count " + nTokens + " is negative");
            }
            return nTokens == 0;
        }
        return memory.canRollback(nTokens);
    }

    @Override
    public void rollback(int nTokens) {
        ensureOpen();
        if (memory == null) {
            if (nTokens != 0) {
                throw new IllegalStateException("a model without memory holds no positions to roll back");
            }
            return;
        }
        rollbackReplays(nTokens);
    }

    /**
     * Context shift over every device: the rows of each device's caches move on their own device, and every
     * KV-cached mixer re-rotates its keys there. A mixer refusal on any device is reported before anything moves.
     */
    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        ensureOpen();
        String arch = spec.arch().ggufName();
        String refusal = spec.contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException(arch + ": context shift is not supported (" + refusal + ")");
        }
        if (memory == null) {
            throw new UnsupportedOperationException(arch + ": context shift needs a device memory, the model is"
                    + " planned without one");
        }
        String hostRefusal = host == null ? null : host.shiftRefusal();
        if (hostRefusal != null) {
            throw new UnsupportedOperationException(arch + ": context shift is not supported (" + hostRefusal + ")");
        }
        for (Shard shard : shards) {
            String shardRefusal = shard.executor().shiftRefusal();
            if (shardRefusal != null) {
                throw new UnsupportedOperationException(arch + ": context shift is not supported (" + shardRefusal
                        + ")");
            }
        }
        invalidateReplays();
        for (Shard shard : shards) {
            shard.memory().shiftPositions(nKeep, nDiscard);
        }
        try {
            if (host != null) {
                host.shiftContext(nKeep, nDiscard);
            }
        } catch (RuntimeException e) {
            throw shiftFailed(arch, e);
        }
        for (Shard shard : shards) {
            try {
                shard.executor().shiftContext(nKeep, nDiscard);
            } catch (RuntimeException e) {
                throw shiftFailed(arch, e);
            }
        }
    }

    /** A context-shift failure after the rows moved: the memory is reset first. */
    private RuntimeException shiftFailed(String arch, RuntimeException e) {
        try {
            reset();
        } catch (RuntimeException resetFailure) {
            e.addSuppressed(resetFailure);
        }
        String msg = arch + ": context shift failed after the cache rows moved; the memory was reset";
        if (e instanceof UnsupportedOperationException) {
            return new UnsupportedOperationException(msg, e);
        }
        return new IllegalStateException(msg, e);
    }

    /** The memory over every device's memory (and the host part's), or null for a model without memory. */
    @Override
    public ModelMemory memory() {
        return memory;
    }

    @Override
    public void setHiddenTap(HiddenTap hiddenTap) {
        Objects.requireNonNull(hiddenTap, "tap");
        ensureOpen();
        this.tap = hiddenTap;
        invalidateReplays();
    }

    /**
     * The queue that captures model tap point {@code layer}: the shard whose span holds it, so a boundary point is
     * captured as the output of the shard before it. A layer the host part runs has no device queue.
     *
     * @throws IllegalArgumentException when {@code layer} is outside the model's layers
     */
    public ComputeQueue tapQueue(int layer) {
        ensureOpen();
        if (host != null && layer < host.layers()) {
            throw new IllegalArgumentException("tap point " + layer + " runs on the host part, not on a device");
        }
        for (Shard shard : shards) {
            int first = shard.firstIl();
            int end = first + shard.executor().layers().size();
            if (layer >= first && layer <= end) {
                return shard.queue();
            }
        }
        throw new IllegalArgumentException("tap point " + layer + " is outside the model's layer spans");
    }

    /** Whether {@code candidate} is one of the shards' queues. */
    public boolean isShardQueue(ComputeQueue candidate) {
        ensureOpen();
        for (Shard shard : shards) {
            if (shard.queue() == candidate) {
                return true;
            }
        }
        return false;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GPU split forward backend is closed");
        }
    }

    private static int[] identity(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    // ================================================================== lifecycle

    /**
     * Waits for every queue, then releases the decode captures and closes the span executors and the head.
     * Idempotent.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException first = null;
        for (Shard shard : shards) {
            try {
                shard.queue().finish();
            } catch (RuntimeException e) {
                first = e;
            }
        }
        for (GpuDecodeReplay replay : replays) {
            first = Closeables.close(replay, first);
        }
        for (Shard shard : shards) {
            first = Closeables.close(shard.executor(), first);
        }
        first = Closeables.close(head, first);
        first = Closeables.close(host, first);
        if (first != null) {
            throw first;
        }
    }

    @Override
    public String toString() {
        return "GpuSplitForwardBackend{" + spec.arch().ggufName() + ", " + shards.size() + " devices, main "
                + main + (host != null ? ", host layers [0, " + host.layers() + ")" : "")
                + (memory != null ? ", memory" : "") + '}';
    }
}
