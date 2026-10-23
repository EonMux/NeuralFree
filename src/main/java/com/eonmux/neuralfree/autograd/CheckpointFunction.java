package com.eonmux.neuralfree.autograd;

import com.eonmux.neuralfree.tensor.GpuCoreOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorStorage;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Gradient checkpointing: a segment runs without recording autograd nodes in the forward pass and is recomputed,
 * with recording, when its gradient is needed.
 *
 * <p>The forward pass applies the segment to a private copy of the input under {@link ComputationGraph#noGrad()}.
 * When the input requires grad and grad mode is on, the output is recorded as one node whose
 * {@link CheckpointGradFunction} keeps a reference to the input (or, with {@code preserveInput}, a copy of it). Its
 * backward recomputes the segment from a fresh copy of that input with grad mode on and propagates the output
 * gradient through the recomputed nodes with a nested first-order
 * {@link ComputationGraph#backward(Tensor, Tensor, boolean)} that releases only those nodes; gradients of parameters
 * used inside the segment accumulate in their slots as usual. Copies are made on the device
 * ({@link GpuCoreOps#contiguous}) when the train kernels are available, otherwise through the host.</p>
 *
 * <p><b>Determinism precondition.</b> The recompute must reproduce the forward pass, so a segment must be a
 * deterministic function of its input and the leaf tensors it reads. Every random draw a segment makes (dropout masks,
 * reparameterization noise) goes through {@link #replayable}: the forward pass tapes the drawn values and the
 * recompute replays them in order. A recompute that draws more or fewer values, or draws them from a different call
 * site, throws {@link IllegalStateException}.</p>
 *
 * <p>A segment reads its argument and leaf tensors (parameters, inputs); it must not read tensors produced by other
 * recorded operations, since the nested backward would propagate into and release their nodes. Because the nested
 * backward writes leaf gradient slots, a checkpointed node supports neither {@code createGraph} nor
 * {@link Autograd#grad}; both throw {@link UnsupportedOperationException}.</p>
 */
public final class CheckpointFunction {

    /** The tapes of the checkpointed forward passes and recomputes running on this thread, innermost first. */
    private static final ThreadLocal<Deque<TapeFrame>> FRAMES = ThreadLocal.withInitial(ArrayDeque::new);

    private CheckpointFunction() {
    }

    /**
     * A checkpointed segment. It returns a new tensor or its argument, never a tensor that something else owns, since
     * the recomputed output is closed after its backward.
     */
    @FunctionalInterface
    public interface CheckpointFn {
        Tensor apply(Tensor input);
    }

    @FunctionalInterface
    public interface CheckpointFnWithArgs {
        Tensor apply(Tensor input, Object... args);
    }

    public static class CheckpointResult {
        private final Tensor output;
        private final List<Tensor> savedInputs;

        private CheckpointResult(Tensor output, List<Tensor> savedInputs) {
            this.output = output;
            this.savedInputs = savedInputs;
        }

        public Tensor output() {
            return output;
        }

        public List<Tensor> savedInputs() {
            return savedInputs;
        }
    }

    /**
     * A value drawn by {@code draw}, reproducible across the recompute of a checkpointed segment. Outside a
     * checkpointed forward pass or recompute it returns {@code draw.get()}. During a checkpointed forward pass it draws
     * and appends the value to the segment's tape (and to the tapes of enclosing checkpointed forward passes); during
     * the recompute it returns the taped values in order instead of drawing. The tape is released with the
     * checkpoint's saved state.
     *
     * <p>A drawn {@link Tensor} belongs to the caller as usual: the tape keeps its own handle of it, and each replay
     * returns a new handle of the taped storage. Drawn values must not be modified after the draw (a tensor must not be
     * written in place, an array not changed), since the recompute reads the same values.</p>
     *
     * @throws IllegalStateException during a recompute, when the segment draws more values than its forward pass did
     *                               or draws one from a different call site
     */
    public static <T> T replayable(Supplier<T> draw) {
        Objects.requireNonNull(draw, "draw");
        Deque<TapeFrame> frames = FRAMES.get();
        if (frames.isEmpty()) {
            return draw.get();
        }
        List<TapeFrame> recorders = new ArrayList<>();
        TapeFrame source = null;
        for (TapeFrame frame : frames) {
            if (frame.replaying) {
                source = frame;
                break;
            }
            recorders.add(frame);
        }
        @SuppressWarnings("unchecked")
        T value = source != null ? (T) source.next(draw.getClass()) : draw.get();
        try {
            for (TapeFrame frame : recorders) {
                frame.tape.append(draw.getClass(), value);
            }
        } catch (RuntimeException e) {
            if (value instanceof Tensor) {
                try {
                    ((Tensor) value).close();
                } catch (RuntimeException closeFailure) {
                    e.addSuppressed(closeFailure);
                }
            }
            throw e;
        }
        return value;
    }

    /**
     * Backward of a checkpointed segment: recomputes the segment from the saved input, replaying the values the
     * forward pass drew through {@link #replayable}, and returns the gradient of that input (zeros when the recomputed
     * segment records no path to it). Every node the recompute records leaves the graph before the backward returns:
     * the nested pass releases those the recomputed output was computed from, and the rest are discarded. Does not
     * support create-graph.
     */
    public static class CheckpointGradFunction implements GradientFunction {
        private final CheckpointFn fn;
        private Tensor savedInput;
        private List<Tensor> savedTensors;
        private Tape tape;

        /** Keeps its own reference to {@code originalInput}'s storage, released by {@link #clearSavedTensors()}. */
        public CheckpointGradFunction(CheckpointFn fn, Tensor originalInput) {
            this(fn, originalInput, new Tape());
        }

        /** As the public constructor, taking over {@code tape} (the forward pass's draws), released with the input. */
        private CheckpointGradFunction(CheckpointFn fn, Tensor originalInput, Tape tape) {
            if (fn == null) {
                throw new IllegalArgumentException("Function cannot be null");
            }
            if (originalInput == null) {
                throw new IllegalArgumentException("Input tensor cannot be null");
            }
            this.fn = fn;
            this.savedInput = originalInput.detach();
            this.savedTensors = Collections.singletonList(savedInput);
            this.tape = tape;
        }

        /**
         * The gradient of the saved input; a recomputed output that is not a recorded node does not depend on it, so
         * its gradient is zero.
         *
         * @throws IllegalStateException when the saved input was released, the segment returns null, or the
         *                               recompute does not replay the forward pass's draws exactly
         */
        @Override
        public List<Tensor> backward(Tensor gradOutput) {
            Tensor source = savedInput;
            if (source == null) {
                throw new IllegalStateException(
                    "Checkpoint input was released; the graph cannot be backpropagated again");
            }
            ComputationGraph graph = ComputationGraph.current();
            long recomputeMark = graph.mark();
            Tensor leaf = withGrad(copyOf(source));
            Tensor recomputed = null;
            try (ComputationGraph.EnableGradScope grad = ComputationGraph.enableGrad();
                 TapeScope replay = TapeScope.replay(tape)) {
                recomputed = fn.apply(leaf);
                replay.requireReplayed();
            } catch (RuntimeException e) {
                if (recomputed != null && recomputed != leaf) {
                    discardRecompute(recomputed, e);
                }
                endRecompute(graph, recomputeMark, null, leaf, e);
                throw e;
            }
            if (recomputed == null) {
                IllegalStateException e = new IllegalStateException(
                    "Checkpoint function returned null during recompute");
                endRecompute(graph, recomputeMark, null, leaf, e);
                throw e;
            }
            if (recomputed == leaf) {
                throwIfFailed(endRecompute(graph, recomputeMark, null, leaf, null));
                return Collections.singletonList(gradOutput);
            }
            Tensor inputGrad;
            try {
                if (recomputed.gradFn() != null) {
                    propagate(recomputed, gradOutput);
                }
                inputGrad = leaf.grad();
                if (inputGrad != null) {
                    leaf.setGrad(null);
                }
            } catch (RuntimeException e) {
                endRecompute(graph, recomputeMark, recomputed, leaf, e);
                throw e;
            }
            RuntimeException cleanup = endRecompute(graph, recomputeMark, recomputed, leaf, null);
            if (cleanup != null) {
                if (inputGrad != null) {
                    closeAfterFailure(inputGrad, cleanup);
                }
                throw cleanup;
            }
            return Collections.singletonList(
                inputGrad != null ? inputGrad : Tensor.zeros(source.backend(), source.shape()));
        }

        /**
         * Ends a recompute: discards the nodes it recorded that are still in the graph (those no path from the
         * recomputed output reaches, or all of them after a failure), closes a gradient left in the recompute leaf's
         * slot, then closes {@code recomputed} (when not null) and the leaf. Failures are added to {@code failure} when
         * it is given; otherwise the first one is returned with the others suppressed.
         */
        private static RuntimeException endRecompute(ComputationGraph graph, long mark, Tensor recomputed, Tensor leaf,
                                                     RuntimeException failure) {
            RuntimeException collected = failure;
            try {
                graph.discardRecordedSince(mark);
            } catch (RuntimeException e) {
                collected = BackwardPass.collect(collected, e);
            }
            List<Tensor> held = new ArrayList<>(3);
            Tensor leftover = leaf.grad();
            if (leftover != null) {
                leaf.setGrad(null);
                held.add(leftover);
            }
            if (recomputed != null) {
                held.add(recomputed);
            }
            held.add(leaf);
            try {
                BackwardPass.closeAll(held, collected);
            } catch (RuntimeException e) {
                collected = BackwardPass.collect(collected, e);
            }
            return failure != null ? null : collected;
        }

        private static void throwIfFailed(RuntimeException failure) {
            if (failure != null) {
                throw failure;
            }
        }

        /**
         * Backpropagates {@code gradOutput} through the recomputed nodes with a nested first-order pass that does not
         * retain them, so they are released afterwards (their intermediate outputs closed; the recomputed output and
         * the leaves stay open). When the pass fails, the recomputed nodes are released all the same, so they do not
         * stay in the graph.
         */
        private static void propagate(Tensor recomputed, Tensor gradOutput) {
            ComputationGraph graph = ComputationGraph.current();
            try {
                graph.backward(recomputed, gradOutput, false);
            } catch (RuntimeException e) {
                try {
                    graph.release(recomputed);
                } catch (RuntimeException releaseFailure) {
                    e.addSuppressed(releaseFailure);
                }
                throw e;
            }
        }

        /** Releases the nodes a failed recompute recorded and closes its output. */
        private static void discardRecompute(Tensor recomputed, RuntimeException failure) {
            try {
                ComputationGraph.current().release(recomputed);
            } catch (RuntimeException e) {
                failure.addSuppressed(e);
            }
            closeAfterFailure(recomputed, failure);
        }

        @Override
        public List<Tensor> savedTensors() {
            return savedTensors;
        }

        /** Releases the reference to the saved input and the taped draws; a later {@link #backward} throws. */
        @Override
        public void clearSavedTensors() {
            RuntimeException failure = null;
            if (savedInput != null) {
                try {
                    savedInput.close();
                } catch (RuntimeException e) {
                    failure = e;
                }
                savedInput = null;
                savedTensors = Collections.emptyList();
            }
            if (tape != null) {
                Tape released = tape;
                tape = null;
                try {
                    released.release();
                } catch (RuntimeException e) {
                    failure = BackwardPass.collect(failure, e);
                }
            }
            if (failure != null) {
                throw failure;
            }
        }

        /** Always false: the nested backward of the recompute writes leaf gradient slots and records nothing. */
        @Override
        public boolean supportsCreateGraph() {
            return false;
        }
    }

    /** {@link #checkpoint(CheckpointFn, Tensor, boolean)} without preserving the input. */
    public static Tensor checkpoint(CheckpointFn fn, Tensor input) {
        return checkpoint(fn, input, false);
    }

    /**
     * Applies {@code fn} to a copy of {@code input} without recording its nodes and returns the result (a new tensor
     * the caller owns). The values {@code fn} draws through {@link #replayable} are taped. When {@code input} requires
     * grad and grad mode is on, the result requires grad and is one graph node whose backward recomputes {@code fn},
     * replaying the tape; otherwise the tape is released. With {@code preserveInput} the recompute reads a copy of
     * {@code input} taken now, so in-place changes to {@code input} after this call do not reach it; without it the
     * recompute reads {@code input}'s storage, which saves the copy.
     *
     * @throws IllegalArgumentException when {@code fn} or {@code input} is null
     * @throws IllegalStateException    when {@code fn} returns null
     */
    public static Tensor checkpoint(CheckpointFn fn, Tensor input, boolean preserveInput) {
        if (fn == null) {
            throw new IllegalArgumentException("Function cannot be null");
        }
        if (input == null) {
            throw new IllegalArgumentException("Input tensor cannot be null");
        }

        Tape tape = new Tape();
        Tensor output;
        try {
            output = forward(fn, input, tape);
        } catch (RuntimeException e) {
            releaseAfterFailure(tape, e);
            throw e;
        }

        if (!Autograd.needsGrad(input)) {
            try {
                tape.release();
            } catch (RuntimeException e) {
                closeAfterFailure(output, e);
                throw e;
            }
            return output;
        }
        boolean[] tapeHandedOver = {false};
        try {
            return Autograd.record(output, Collections.singletonList(input), out -> {
                CheckpointGradFunction gradFn;
                if (preserveInput) {
                    try (Tensor preserved = copyOf(input)) {
                        gradFn = new CheckpointGradFunction(fn, preserved, tape);
                    }
                } else {
                    gradFn = new CheckpointGradFunction(fn, input, tape);
                }
                tapeHandedOver[0] = true;
                return gradFn;
            });
        } catch (RuntimeException e) {
            if (!tapeHandedOver[0]) {
                releaseAfterFailure(tape, e);
            }
            throw e;
        }
    }

    /** The checkpointed forward pass: {@code fn} on a copy of {@code input}, without recording, its draws taped. */
    private static Tensor forward(CheckpointFn fn, Tensor input, Tape tape) {
        try (ComputationGraph.NoGradScope scope = ComputationGraph.noGrad();
             TapeScope recording = TapeScope.record(tape)) {
            Tensor inputCopy = copyOf(input);
            Tensor output;
            try {
                output = fn.apply(inputCopy);
            } catch (RuntimeException e) {
                closeAfterFailure(inputCopy, e);
                throw e;
            }
            if (output == null) {
                inputCopy.close();
                throw new IllegalStateException("Checkpoint function returned null");
            }
            if (output != inputCopy) {
                inputCopy.close();
            }
            return output;
        }
    }

    /**
     * Checkpoints each function in turn; each intermediate result is closed once the next checkpoint holds its own
     * reference to it.
     */
    public static Tensor checkpointSequential(List<CheckpointFn> functions, Tensor input) {
        if (functions == null || functions.isEmpty()) {
            throw new IllegalArgumentException("Functions list cannot be null or empty");
        }
        if (input == null) {
            throw new IllegalArgumentException("Input tensor cannot be null");
        }

        Tensor current = input;
        for (CheckpointFn fn : functions) {
            Tensor next = checkpoint(fn, current);
            if (current != input) {
                current.close();
            }
            current = next;
        }
        return current;
    }

    /**
     * Applies the segments in turn, checkpointing every {@code checkpointEvery}-th one (starting with the first;
     * {@code checkpointEvery <= 0} checkpoints only the first) and applying the others directly with recording. While
     * the graph records, an intermediate result is closed only when a checkpoint produced it and a checkpoint consumed
     * it, since a directly applied segment's nodes may keep their inputs and outputs for backward; without recording
     * every intermediate result is closed once the next segment has consumed it.
     */
    public static Tensor checkpointSegments(List<CheckpointFn> segments, int checkpointEvery, Tensor input) {
        if (segments == null || segments.isEmpty()) {
            throw new IllegalArgumentException("Segments list cannot be null or empty");
        }
        if (input == null) {
            throw new IllegalArgumentException("Input tensor cannot be null");
        }
        int every = checkpointEvery <= 0 ? segments.size() : checkpointEvery;

        Tensor current = input;
        boolean currentFromCheckpoint = false;
        int counter = 0;
        for (CheckpointFn segment : segments) {
            boolean checkpointed = counter % every == 0;
            boolean recording = ComputationGraph.isGradEnabled();
            Tensor next = checkpointed ? checkpoint(segment, current) : segment.apply(current);
            boolean intermediate = current != input && next != current;
            if (intermediate && (!recording || (checkpointed && currentFromCheckpoint))) {
                current.close();
            }
            current = next;
            currentFromCheckpoint = checkpointed;
            counter++;
        }
        return current;
    }

    public static class MemoryStats {
        private final long savedActivationsBytes;
        private final long peakMemoryBytes;
        private final int numCheckpoints;

        private MemoryStats(long savedActivationsBytes, long peakMemoryBytes, int numCheckpoints) {
            this.savedActivationsBytes = savedActivationsBytes;
            this.peakMemoryBytes = peakMemoryBytes;
            this.numCheckpoints = numCheckpoints;
        }

        public long savedActivationsBytes() {
            return savedActivationsBytes;
        }

        public long peakMemoryBytes() {
            return peakMemoryBytes;
        }

        public int numCheckpoints() {
            return numCheckpoints;
        }

        @Override
        public String toString() {
            return String.format("MemoryStats{saved=%.2f MB, peak=%.2f MB, checkpoints=%d}",
                    savedActivationsBytes / (1024.0 * 1024.0),
                    peakMemoryBytes / (1024.0 * 1024.0),
                    numCheckpoints);
        }
    }

    public static MemoryStats estimateMemorySavings(int numLayers, long activationSizeBytes, int checkpointEvery) {
        if (checkpointEvery <= 0) {
            checkpointEvery = numLayers;
        }

        int numCheckpoints = (int) Math.ceil((double) numLayers / checkpointEvery);
        long savedMemory = numCheckpoints * activationSizeBytes;
        long peakMemory = savedMemory + checkpointEvery * activationSizeBytes;

        return new MemoryStats(savedMemory, peakMemory, numCheckpoints);
    }

    public static Function<Tensor, Tensor> wrapCheckpoint(CheckpointFn fn) {
        return input -> checkpoint(fn, input);
    }

    /**
     * A new contiguous copy of {@code t} with the same shape and dtype that does not require grad: on the device
     * when the train kernels are available for the dtype, otherwise through the host.
     */
    private static Tensor copyOf(Tensor t) {
        Tensor device = GpuCoreOps.contiguous(t);
        if (device != null) {
            return device;
        }
        TensorStorage storage = TensorStorage.fromHost(t.backend(), t.toByteBuffer(), t.dtype());
        return Tensor.fromStorage(t.backend(), t.shape(), t.dtype(), storage);
    }

    /** {@code t} requiring grad; when that needs a new view, {@code t}'s reference is released. */
    private static Tensor withGrad(Tensor t) {
        if (t.requiresGrad()) {
            return t;
        }
        Tensor result = t.withRequiresGrad(true);
        t.close();
        return result;
    }

    private static void closeAfterFailure(Tensor t, RuntimeException failure) {
        try {
            t.close();
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
    }

    private static void releaseAfterFailure(Tape tape, RuntimeException failure) {
        try {
            tape.release();
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
    }

    // ------------------------------------------------------------------ replay tape

    /**
     * The values drawn through {@link #replayable} during one checkpointed forward pass, each with the class of the
     * supplier that drew it (its call site). A drawn tensor is kept as the tape's own handle.
     */
    private static final class Tape {
        private final List<Class<?>> sites = new ArrayList<>();
        private final List<Object> values = new ArrayList<>();

        void append(Class<?> site, Object value) {
            Object kept = value instanceof Tensor ? ((Tensor) value).detach() : value;
            sites.add(site);
            values.add(kept);
        }

        int size() {
            return values.size();
        }

        /** Taped value {@code index}, drawn at {@code site}; a tensor as a new handle the caller owns. */
        Object replay(int index, Class<?> site) {
            if (index >= values.size()) {
                throw new IllegalStateException("A checkpointed segment drew more values during its recompute ("
                    + (index + 1) + ") than during its forward pass (" + values.size()
                    + "); a checkpointed segment must be deterministic");
            }
            if (sites.get(index) != site) {
                throw new IllegalStateException("A checkpointed segment drew value " + index + " from "
                    + site.getName() + " during its recompute but from " + sites.get(index).getName()
                    + " during its forward pass; a checkpointed segment must be deterministic");
            }
            Object value = values.get(index);
            return value instanceof Tensor ? ((Tensor) value).detach() : value;
        }

        /** Closes the taped tensors and forgets every value; the first close failure is thrown after all were tried. */
        void release() {
            List<Tensor> tensors = new ArrayList<>();
            for (Object value : values) {
                if (value instanceof Tensor) {
                    tensors.add((Tensor) value);
                }
            }
            values.clear();
            sites.clear();
            BackwardPass.closeAll(tensors, null);
        }
    }

    /** One tape in use on this thread: recorded into by a forward pass, or replayed by a recompute. */
    private static final class TapeFrame {
        private final Tape tape;
        private final boolean replaying;
        private int cursor;

        TapeFrame(Tape tape, boolean replaying) {
            this.tape = tape;
            this.replaying = replaying;
        }

        Object next(Class<?> site) {
            return tape.replay(cursor++, site);
        }
    }

    /** Pushes a {@link TapeFrame} on this thread for the scope's lifetime. */
    private static final class TapeScope implements AutoCloseable {
        private final TapeFrame frame;

        private TapeScope(TapeFrame frame) {
            this.frame = frame;
            FRAMES.get().push(frame);
        }

        static TapeScope record(Tape tape) {
            return new TapeScope(new TapeFrame(tape, false));
        }

        static TapeScope replay(Tape tape) {
            return new TapeScope(new TapeFrame(tape, true));
        }

        /** Throws when the recompute drew fewer values than the forward pass taped. */
        void requireReplayed() {
            if (frame.cursor != frame.tape.size()) {
                throw new IllegalStateException("A checkpointed segment drew " + frame.cursor
                    + " values during its recompute but " + frame.tape.size()
                    + " during its forward pass; a checkpointed segment must be deterministic");
            }
        }

        @Override
        public void close() {
            FRAMES.get().removeFirstOccurrence(frame);
        }
    }
}
