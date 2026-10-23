package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.gpu.head.GpuEmbeddingStage;
import com.eonmux.neuralfree.gguf.gpu.head.GpuLogitsHead;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuMemory;
import com.eonmux.neuralfree.gguf.ops.rope.RopePositions;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Runs one forward call over any number of tokens as a sequence of ubatch steps (llama.cpp {@code llama_decode}
 * splitting a batch into ubatches) and reads back only what the caller asked for.
 *
 * <p>Per step of {@code n} tokens at position {@code p} ({@code n} at most the scratch ubatch and, when the call
 * writes memory, the memory plan's ubatch, split by {@code MemoryPlan.stepTokens} so the trailing tokens a
 * compressor rollback may drop stay in the last step): {@code GpuStepInputs.upload(p, n)} when the call
 * writes memory, the four-section positions into {@link ScratchSlots#POS}, the embedding stage, the layer executor
 * (DECODE for a single token against memory, PREFILL otherwise), then the head on the requested rows of that step:
 * their logits, their hidden rows ({@code t_embd}) or their greedy ids, each enqueued as a read into a host array.
 * Tap points the {@link HiddenTap} wants are enqueued as reads too. After the last step the queue is finished once;
 * that is the only host synchronization of the call (a call that fails while enqueuing drains the queue before its
 * pending reads are released). Memory bookkeeping ({@code SequenceState.append}) follows each
 * step.</p>
 *
 * <p>A call without memory evaluates all tokens in one step (non-causal attention cannot be split), so it needs
 * {@code tokens.length <= ubatch}. Not thread-safe.</p>
 */
public final class GpuPrefillDriver {

    /** What a call reads back per requested row. */
    public enum Output {
        /** {@code vocab} logits. */
        LOGITS,
        /** {@code n_embd} hidden values after the output norm. */
        HIDDEN,
        /** The greedy token id. */
        ARGMAX
    }

    /** Rows read back by a call, in request order. */
    public static final class Result {
        private final float[][] rows;
        private final int[] ids;

        Result(float[][] rows, int[] ids) {
            this.rows = rows;
            this.ids = ids;
        }

        /** Logits or hidden rows ({@link Output#LOGITS}, {@link Output#HIDDEN}); null for ARGMAX. */
        public float[][] rows() {
            return rows;
        }

        /** Greedy ids ({@link Output#ARGMAX}); null otherwise. */
        public int[] ids() {
            return ids;
        }
    }

    private final GpuForwardContext ctx;
    private final GpuEmbeddingStage embedding;
    private final GpuLayerExecutor executor;
    private final GpuLogitsHead head;
    private final ComputeQueue queue;
    private final MropeTextPositions textPositions;
    private final ScratchSlot posSlot;
    private final int ubatch;
    private final int[] posInts;
    private final ByteBuffer posBytes;

    public GpuPrefillDriver(GpuForwardContext ctx, GpuEmbeddingStage embedding, GpuLayerExecutor executor,
                            GpuLogitsHead head) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.embedding = Objects.requireNonNull(embedding, "embedding");
        this.executor = Objects.requireNonNull(executor, "executor");
        this.head = Objects.requireNonNull(head, "head");
        this.queue = ctx.queue();
        this.textPositions = ctx.spec().hparams().mropeTextPositions();
        this.ubatch = ctx.scratch().ubatch();
        this.posSlot = ctx.scratch().slot(ScratchSlots.POS).require((long) MropeTextPositions.SECTIONS * ubatch);
        this.posInts = new int[MropeTextPositions.SECTIONS * ubatch];
        this.posBytes = ByteBuffer.allocate(posInts.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Evaluates {@code tokens} at positions {@code pos0 ..} and reads back {@code output} for {@code outputRows}.
     *
     * @param types      token types (BERT segments) parallel to {@code tokens}, or null
     * @param outputRows indices into {@code tokens} in result order (repeats allowed), or null for every row
     * @param useMemory  true to write the model memory (causal calls); the memory must hold exactly {@code pos0}
     *                   positions
     * @param tap        hidden-state observer of this call
     * @throws ContextFullException     when the positions exceed the memory's context length; nothing is enqueued
     * @throws IllegalArgumentException for empty tokens, a row outside the call, or a memoryless call longer than
     *                                  one ubatch
     * @throws IllegalStateException    when the memory does not end at {@code pos0}
     * @throws UnsupportedOperationException when logits or ids are asked of a model without an LM head
     */
    public Result run(int[] tokens, int[] types, int pos0, int[] outputRows, Output output, boolean useMemory,
                      HiddenTap tap) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(tap, "tap");
        int total = tokens.length;
        if (total == 0) {
            throw new IllegalArgumentException("forward call without tokens");
        }
        if (types != null && types.length != total) {
            throw new IllegalArgumentException(types.length + " token types for " + total + " tokens");
        }
        int[] rows = outputRows != null ? outputRows.clone() : identity(total);
        if (rows.length == 0) {
            throw new IllegalArgumentException("forward call without output rows");
        }
        for (int r : rows) {
            if (r < 0 || r >= total) {
                throw new IllegalArgumentException("output row " + r + " is outside the " + total + " tokens");
            }
        }
        if (output != Output.HIDDEN && !head.hasLmHead()) {
            throw new UnsupportedOperationException(ctx.spec().arch().ggufName() + " has no LM head; only hidden"
                    + " rows can be read");
        }
        GpuMemory memory = useMemory ? ctx.memory() : null;
        checkSpan(memory, pos0, total);
        Calls calls = new Calls(rows.length, output);
        TapReads taps = new TapReads(tap, total);
        boolean finished = false;
        try {
            int chunk = memory != null ? Math.min(ubatch, memory.plan().ubatch()) : ubatch;
            for (int c0 = 0; c0 < total; ) {
                int n = memory != null ? memory.plan().stepTokens(total - c0, chunk) : Math.min(chunk, total - c0);
                step(tokens, types, pos0 + c0, c0, n, rows, output, memory, calls, taps);
                if (memory != null) {
                    memory.sequence().append(tokens, c0, n);
                }
                c0 += n;
            }
            finished = true;
            queue.finish();
            taps.deliver();
            return calls.result();
        } catch (RuntimeException e) {
            if (!finished) {
                try {
                    queue.finish();
                } catch (RuntimeException drain) {
                    e.addSuppressed(drain);
                }
            }
            throw e;
        } finally {
            calls.closeEvents();
            taps.closeEvents();
        }
    }

    private void checkSpan(GpuMemory memory, int pos0, int total) {
        if (memory == null) {
            if (total > ubatch) {
                throw new IllegalArgumentException(total + " tokens exceed the ubatch of " + ubatch + "; a call without"
                        + " memory runs in one step, raise neuralfree.ubatch");
            }
            if (pos0 != 0) {
                throw new IllegalArgumentException("a call without memory starts at position 0, got " + pos0);
            }
            return;
        }
        int nCtx = memory.plan().nCtx();
        long end = (long) pos0 + total;
        if (end > nCtx) {
            throw new ContextFullException((int) Math.min(Integer.MAX_VALUE, end), nCtx);
        }
        int past = memory.sequence().nPast();
        if (pos0 != past) {
            throw new IllegalStateException("the memory holds " + past + " positions, the call starts at " + pos0);
        }
    }

    private void step(int[] tokens, int[] types, int p, int c0, int n, int[] rows, Output output, GpuMemory memory,
                      Calls calls, TapReads taps) {
        ExecutionObserver obs = ctx.observer();
        boolean observed = obs.enabled();
        if (observed) {
            obs.onStepStart(p, n);
        }
        if (memory != null) {
            memory.stepInputs().upload(p, n);
        }
        executor.prefetchFirst();
        writePositions(p, n);
        embedding.embed(tokens, c0, n, types, p);
        GpuLayerExecutor.StepMode mode = memory != null && n == 1
                ? GpuLayerExecutor.StepMode.DECODE : GpuLayerExecutor.StepMode.PREFILL;
        taps.chunk(c0);
        executor.run(p, n, mode, taps);
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
        for (int done = 0; done < k; ) {
            int count = Math.min(n, k - done);
            ScratchSlot hidden = head.hidden(n, Arrays.copyOfRange(local, done, done + count), count);
            calls.read(output, hidden, count, Arrays.copyOfRange(targets, done, done + count));
            done += count;
        }
        if (observed) {
            obs.onStepEnd(p, n);
        }
    }

    /** {@code pos[t + k*n]} for the step into {@link ScratchSlots#POS} (one write). */
    private void writePositions(int p, int n) {
        RopePositions.fillStep(textPositions, p, n, posInts);
        int count = MropeTextPositions.SECTIONS * n;
        posBytes.clear();
        for (int i = 0; i < count; i++) {
            posBytes.putInt(posInts[i]);
        }
        GpuEvent ev = queue.enqueueWriteBuffer(posSlot.buffer(), (long) posSlot.offset() * Float.BYTES,
                (long) count * Integer.BYTES, posBytes.array());
        if (ev != null) {
            ev.close();
        }
    }

    private static int[] identity(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    private static void closeAll(List<GpuEvent> events) {
        for (GpuEvent e : events) {
            if (e != null) {
                e.close();
            }
        }
        events.clear();
    }

    /** Head reads of one call and their placement in the result. */
    private final class Calls {
        private final float[][] rows;
        private final int[] ids;
        private final List<float[]> floatParts = new ArrayList<>();
        private final List<byte[]> idParts = new ArrayList<>();
        private final List<int[]> targets = new ArrayList<>();
        private final List<GpuEvent> events = new ArrayList<>();
        private final Output output;

        Calls(int count, Output output) {
            this.output = output;
            this.rows = output == Output.ARGMAX ? null : new float[count][];
            this.ids = output == Output.ARGMAX ? new int[count] : null;
        }

        void read(Output out, ScratchSlot hidden, int count, int[] target) {
            ScratchSlot src;
            if (out == Output.HIDDEN) {
                src = hidden;
            } else {
                ScratchSlot logits = head.logits(count, hidden);
                src = out == Output.ARGMAX ? head.argmax(count) : logits;
            }
            int width = out == Output.HIDDEN ? head.hiddenWidth() : out == Output.LOGITS ? head.vocab() : 1;
            long bytes = (long) count * width * Float.BYTES;
            if (bytes > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("read of " + count + " rows of " + width + " floats exceeds 2 GiB");
            }
            long srcOff = (long) src.offset() * Float.BYTES;
            if (out == Output.ARGMAX) {
                byte[] dst = new byte[(int) bytes];
                events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
                idParts.add(dst);
                floatParts.add(null);
            } else {
                float[] dst = new float[count * width];
                events.add(queue.enqueueReadBuffer(src.buffer(), srcOff, bytes, dst));
                floatParts.add(dst);
                idParts.add(null);
            }
            targets.add(target);
        }

        Result result() {
            int width = output == Output.HIDDEN ? head.hiddenWidth() : output == Output.LOGITS ? head.vocab() : 1;
            for (int part = 0; part < targets.size(); part++) {
                int[] target = targets.get(part);
                if (output == Output.ARGMAX) {
                    ByteBuffer bb = ByteBuffer.wrap(idParts.get(part)).order(ByteOrder.LITTLE_ENDIAN);
                    for (int j = 0; j < target.length; j++) {
                        ids[target[j]] = bb.getInt(j * Integer.BYTES);
                    }
                    continue;
                }
                float[] data = floatParts.get(part);
                if (target.length == 1) {
                    rows[target[0]] = data;
                    continue;
                }
                for (int j = 0; j < target.length; j++) {
                    rows[target[j]] = Arrays.copyOfRange(data, j * width, (j + 1) * width);
                }
            }
            return new Result(rows, ids);
        }

        void closeEvents() {
            closeAll(events);
        }
    }

    /** Tap-point reads of one call, delivered to the {@link HiddenTap} once the queue finished. */
    private final class TapReads implements GpuLayerExecutor.TapSink {
        private final HiddenTap tap;
        private final int total;
        private final List<Integer> layers = new ArrayList<>();
        private final List<Integer> offsets = new ArrayList<>();
        private final List<Integer> counts = new ArrayList<>();
        private final List<Integer> widths = new ArrayList<>();
        private final List<float[]> parts = new ArrayList<>();
        private final List<GpuEvent> events = new ArrayList<>();
        private int chunkStart;

        TapReads(HiddenTap tap, int total) {
            this.tap = tap;
            this.total = total;
        }

        void chunk(int c0) {
            this.chunkStart = c0;
        }

        @Override
        public boolean wants(int tapPoint) {
            return tap.wants(tapPoint);
        }

        @Override
        public void capture(int tapPoint, ScratchSlot x, int nTokens, int width) {
            float[] dst = new float[Math.multiplyExact(nTokens, width)];
            events.add(queue.enqueueReadBuffer(x.buffer(), (long) x.offset() * Float.BYTES,
                    (long) dst.length * Float.BYTES, dst));
            layers.add(tapPoint);
            offsets.add(chunkStart);
            counts.add(nTokens);
            widths.add(width);
            parts.add(dst);
        }

        /** One {@link HiddenTap#onHidden} per tap point with every row of the call, in tap order. */
        void deliver() {
            List<Integer> seen = new ArrayList<>();
            for (int i = 0; i < layers.size(); i++) {
                int layer = layers.get(i);
                if (seen.contains(layer)) {
                    continue;
                }
                seen.add(layer);
                int width = widths.get(i);
                float[] all = new float[Math.multiplyExact(total, width)];
                for (int j = i; j < layers.size(); j++) {
                    if (layers.get(j) != layer) {
                        continue;
                    }
                    if (widths.get(j) != width) {
                        throw new IllegalStateException("tap point " + layer + " changed width from " + width
                                + " to " + widths.get(j) + " within a call");
                    }
                    System.arraycopy(parts.get(j), 0, all, offsets.get(j) * width, counts.get(j) * width);
                }
                tap.onHidden(layer, all, total, width);
            }
        }

        void closeEvents() {
            closeAll(events);
        }
    }
}
