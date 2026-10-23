package com.eonmux.neuralfree.tensor.gpu;

import com.eonmux.neuralfree.acceleration.GpuBackendException;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.BackendResources;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.KernelLauncher;
import com.eonmux.neuralfree.acceleration.kernels.KernelSet;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.TensorStorage;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * The train kernel library on one backend: kernel lookup, launches, and the per-backend state the tensor dispatchers
 * share (read-only placeholder, deferred index-error slots, reusable scratch, pending-work tracking).
 *
 * <p><b>Lifetime.</b> One context per backend, held as a {@link BackendResources} attachment together with the
 * availability result, so the library is opened, prewarmed and checked once per backend and an unavailable result
 * is remembered with its reason (logged once at WARN). The backend's {@code close()} closes the context through
 * {@link BackendResources#closeAll}; {@link #release} closes it earlier.</p>
 *
 * <p><b>Availability.</b> The library is available when {@link KernelSet#availability} says so, the device runs
 * work-groups of at least {@link #LINEAR_LOCAL} items, the library opens, and every kernel it lists can be loaded
 * and prepared at its {@link TrainOps#localSize}.</p>
 *
 * <p><b>Ordering and locking.</b> Kernels are enqueued only on {@code backend.computeQueue()}, without a
 * {@code finish()} per op. Every launch, fill and readback runs under {@link BackendResources#lock}; a multi-launch
 * sequence that must not interleave with other threads (or that uses {@link #scratch}) runs inside
 * {@link #exclusive}. Host transfers of tensor storage call {@link #beforeHostRead} and {@link #afterHostWrite}, which
 * order them against pending train work. The LLM engine does not take this lock, so a backend instance must not be
 * used by the LLM engine and by tensor ops from different threads at the same time.</p>
 *
 * <p><b>Deferred index errors.</b> A kernel that meets an out-of-range index records it in its slot of
 * {@link #errors()}: four ints {@code {flag, index, dim, size}}, claimed by the first failing work-item since the slots
 * were last read (an atomic compare-and-swap of the flag, {@code tr_report_index_error} in
 * {@code kernels/train/common/train.h}). The slots are read at the next host synchronization
 * ({@link #beforeHostRead}, {@link #readScalar}, {@link #readFlag}) or by {@link #checkErrors()}, which throws
 * {@link IndexOutOfBoundsException} with the CPU path's message for that index ({@link #gatherIndexMessage},
 * {@link #scatterIndexMessage}, {@link #embeddingIndexMessage}).</p>
 */
public final class TrainGpu implements AutoCloseable {

    private static final Logger LOGGER = LogManager.getLogger(TrainGpu.class);

    /** Local size of every 1-D train kernel. */
    public static final int LINEAR_LOCAL = 256;
    /** Work-items per row of a "rows" kernel: one group of this size per row. */
    public static final int ROW_LOCAL = 256;

    /** Error slot of TR_Gather_F32. */
    public static final int ERROR_SLOT_GATHER = 0;
    /** Error slot of TR_Scatter_F32. */
    public static final int ERROR_SLOT_SCATTER = 1;
    /** Error slot of TR_IndexSelectRows_F32 in error mode. */
    public static final int ERROR_SLOT_INDEX_SELECT = 2;

    private static final int ERROR_SLOT_COUNT = 3;
    /** Ints per error slot: {@code {flag, index, dim, size}}. */
    private static final int ERROR_SLOT_WORDS = 4;
    private static final long ERROR_BYTES = (long) ERROR_SLOT_COUNT * ERROR_SLOT_WORDS * Integer.BYTES;
    private static final long PLACEHOLDER_BYTES = 16;

    private static final Object KEY = new Object() {
        @Override
        public String toString() {
            return "train-gpu";
        }
    };

    private final ComputeBackend backend;
    private final ComputeQueue compute;
    private final KernelSet kernels;
    private final KernelLauncher launcher;
    private final ReentrantLock lock;
    private final GpuBuffer placeholder;
    private final GpuBuffer errors;
    /** Guarded by {@link #lock}. */
    private final Map<Integer, GpuBuffer> scratch = new HashMap<>();
    /** Train work was enqueued on the compute queue since it last drained; guarded by {@link #lock}. */
    private boolean pendingWork;
    /** A kernel bound {@link #errors} since the slots were last read; guarded by {@link #lock}. */
    private boolean errorsArmed;
    private volatile boolean closed;

    /**
     * Message of the {@link IndexOutOfBoundsException} for gather index {@code index} outside {@code [0, size)} of
     * dimension {@code dim}: the CPU path's whole message and the device path's text for {@link #ERROR_SLOT_GATHER}.
     */
    public static String gatherIndexMessage(long index, int dim, long size) {
        return "Gather index " + index + " out of bounds for dimension " + dim + " with size " + size;
    }

    /** As {@link #gatherIndexMessage} for a scatter index ({@link #ERROR_SLOT_SCATTER}). */
    public static String scatterIndexMessage(long index, int dim, long size) {
        return "Scatter index " + index + " out of bounds for dimension " + dim + " with size " + size;
    }

    /** As {@link #gatherIndexMessage} for an embedding row index ({@link #ERROR_SLOT_INDEX_SELECT} in error mode). */
    public static String embeddingIndexMessage(long index) {
        return "Embedding index out of range: " + index;
    }

    private TrainGpu(ComputeBackend backend, KernelSet kernels, ReentrantLock lock, GpuBuffer placeholder,
                     GpuBuffer errors) {
        this.backend = backend;
        this.compute = backend.computeQueue();
        this.kernels = kernels;
        this.launcher = new KernelLauncher(compute);
        this.lock = lock;
        this.placeholder = placeholder;
        this.errors = errors;
        this.pendingWork = true;
    }

    // ------------------------------------------------------------------ per-backend context

    /**
     * The context of a backend, opening the train library on the first call, or null when the library is
     * unavailable there (the reason is cached, logged once at WARN and returned by {@link #availability}). Dtype
     * gating is the caller's.
     *
     * @throws RuntimeException from the backend when allocating the context's buffers fails; nothing is cached then
     */
    public static TrainGpu forBackend(ComputeBackend b) {
        return slot(b).gpu;
    }

    /** Whether the train library is usable on a backend, with the reason when it is not; see {@link #forBackend}. */
    public static Availability availability(ComputeBackend b) {
        return slot(b).availability;
    }

    /** Closes and forgets a backend's context (or cached unavailability); the next {@link #forBackend} reopens it. */
    public static void release(ComputeBackend b) {
        BackendResources.detach(Objects.requireNonNull(b, "backend must not be null"), KEY);
    }

    /** {@link #release} for every backend that has a context. */
    public static void releaseAll() {
        BackendResources.detachAll(KEY);
    }

    /**
     * Orders a host transfer of tensor data on {@code q} after the train work already enqueued: checks the
     * deferred error slots when a kernel armed them (which drains the compute queue and may throw), then, when train
     * work is pending and {@code q} is not the compute queue, appends a compute-queue marker to {@code waits}. The
     * caller passes {@code waits} to its transfer, finishes {@code q}, and closes the appended events. Never opens the
     * library; does nothing for a backend without a context. The caller holds {@link BackendResources#lock}.
     *
     * @throws IndexOutOfBoundsException when an index-validating kernel reported an out-of-range index
     */
    public static void beforeHostRead(ComputeBackend b, ComputeQueue q, List<GpuEvent> waits) {
        Objects.requireNonNull(b, "backend must not be null");
        Objects.requireNonNull(q, "queue must not be null");
        Objects.requireNonNull(waits, "waits must not be null");
        TrainGpu g = existing(b);
        if (g != null) {
            g.orderHostAccess(q, waits);
        }
    }

    /**
     * Gives the compute queue a device-side dependency on a finished host upload: enqueues a compute-queue marker
     * that waits on {@code uploadEvent}, so every later train launch is ordered after the upload (on Vulkan the
     * cross-queue wait is a timeline-semaphore wait). The event stays owned by the caller. Does nothing for a null
     * event or a backend without a context. The caller holds {@link BackendResources#lock}.
     */
    public static void afterHostWrite(ComputeBackend b, GpuEvent uploadEvent) {
        Objects.requireNonNull(b, "backend must not be null");
        if (uploadEvent == null) {
            return;
        }
        TrainGpu g = existing(b);
        if (g != null) {
            g.orderAfterUpload(uploadEvent);
        }
    }

    private static Slot slot(ComputeBackend b) {
        Objects.requireNonNull(b, "backend must not be null");
        return BackendResources.attachment(b, KEY, () -> Slot.open(b));
    }

    private static TrainGpu existing(ComputeBackend b) {
        Slot s = BackendResources.existing(b, KEY);
        return s == null ? null : s.gpu;
    }

    // ------------------------------------------------------------------ kernels and launches

    public ComputeBackend backend() {
        return backend;
    }

    /**
     * The handle of a train kernel.
     *
     * @throws UnsupportedKernelException naming the kernel and the reason when it cannot be provided
     */
    public KernelHandle kernel(String name) {
        ensureOpen();
        return kernels.get(name);
    }

    /** A 1-D launch of {@code n} work-items in groups of {@link #LINEAR_LOCAL}; returns when {@code n == 0}. */
    public void launch1d(String kernel, long n, Object... args) {
        requireElements(n, kernel + " work-items");
        lock.lock();
        try {
            ensureOpen();
            launcher.launch1d(kernels.get(kernel), n, LINEAR_LOCAL, args);
            noteLaunch(n > 0, args);
        } finally {
            lock.unlock();
        }
    }

    /** A 1-D launch of {@code rows} groups of {@link #ROW_LOCAL} work-items; returns when {@code rows == 0}. */
    public void launchRows(String kernel, long rows, Object... args) {
        requireElements(rows, kernel + " rows");
        long global = Math.multiplyExact(rows, (long) ROW_LOCAL);
        lock.lock();
        try {
            ensureOpen();
            launcher.launch(kernels.get(kernel), new long[]{global}, new long[]{ROW_LOCAL}, args);
            noteLaunch(rows > 0, args);
        } finally {
            lock.unlock();
        }
    }

    /** A launch with explicit sizes ({@link KernelLauncher#launch}); returns when any global extent is 0. */
    public void launch(String kernel, long[] global, long[] local, Object... args) {
        Objects.requireNonNull(global, "global must not be null");
        lock.lock();
        try {
            ensureOpen();
            launcher.launch(kernels.get(kernel), global, local, args);
            boolean launched = true;
            for (long g : global) {
                launched &= g > 0;
            }
            noteLaunch(launched, args);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Runs a multi-launch sequence under the backend lock, so no other thread enqueues on this backend's shared
     * queues in between (required while the sequence uses {@link #scratch}).
     */
    public <T> T exclusive(Supplier<T> sequence) {
        Objects.requireNonNull(sequence, "sequence must not be null");
        lock.lock();
        try {
            ensureOpen();
            return sequence.get();
        } finally {
            lock.unlock();
        }
    }

    // ------------------------------------------------------------------ shared buffers

    /** A 16-byte zero buffer for unused buffer parameters; bind it only to read-only parameters. */
    public GpuBuffer placeholder() {
        ensureOpen();
        return placeholder;
    }

    /**
     * The deferred index-error slots: {@code int[4 * 3]}, slot {@code s} ({@link #ERROR_SLOT_GATHER},
     * {@link #ERROR_SLOT_SCATTER}, {@link #ERROR_SLOT_INDEX_SELECT}) at ints {@code 4s .. 4s + 3}. Bind it only to the
     * {@code err} parameter of an index-validating kernel.
     */
    public GpuBuffer errors() {
        ensureOpen();
        return errors;
    }

    /**
     * Drains the compute queue, reads the error slots when a kernel armed them, clears them, and throws for any slot
     * that is set.
     *
     * @throws IndexOutOfBoundsException with the CPU path's message for the first out-of-range index of every op that
     *                                   reported one
     */
    public void checkErrors() {
        lock.lock();
        try {
            ensureOpen();
            if (!errorsArmed) {
                if (pendingWork) {
                    compute.finish();
                    pendingWork = false;
                }
                return;
            }
            byte[] slots = new byte[(int) ERROR_BYTES];
            GpuEvent ev = compute.enqueueReadBuffer(errors, 0, ERROR_BYTES, slots);
            finishAndClose(ev);
            throwIfErrors(slots);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Storage for a new tensor: a buffer rounded up to a multiple of 4 bytes with the exact element length.
     *
     * @throws IllegalArgumentException when {@code elements} is negative or above {@code Integer.MAX_VALUE}
     */
    public TensorStorage allocate(long elements, DType dtype) {
        requireElements(elements, "allocation elements");
        ensureOpen();
        return TensorStorage.allocate(backend, elements, dtype);
    }

    /** Fills 32-bit words {@code [elementOffset, elementOffset + elements)} of {@code buf} with {@code value}. */
    public void fill(GpuBuffer buf, long elementOffset, long elements, float value) {
        Objects.requireNonNull(buf, "buffer must not be null");
        requireWordRange(buf, elementOffset, elements);
        launch1d(TrainOps.Kernels.FILL_F32, elements, (int) elements, value, (int) elementOffset, buf);
    }

    /** Sets every byte of {@code buf} (whose size is a multiple of 4, as every train allocation is) to zero. */
    public void zero(GpuBuffer buf) {
        Objects.requireNonNull(buf, "buffer must not be null");
        lock.lock();
        try {
            ensureOpen();
            compute.enqueueFillBuffer(buf, 0, buf.sizeInBytes(), 0);
            pendingWork = true;
        } finally {
            lock.unlock();
        }
    }

    /** Reads one float after all enqueued train work completes; a documented host synchronization point. */
    public float readScalar(GpuBuffer buf, long elementOffset) {
        return Float.intBitsToFloat(readWord(buf, elementOffset));
    }

    /** Reads one int after all enqueued train work completes; a documented host synchronization point. */
    public int readFlag(GpuBuffer buf, long elementOffset) {
        return readWord(buf, elementOffset);
    }

    /**
     * A reusable device buffer of at least {@code bytes} bytes for one scratch slot, grown on demand. The buffer may
     * be replaced by the next call for the same slot, so it is valid only inside one {@link #exclusive} sequence.
     * A replaced buffer is closed; the backends defer or synchronize the release of buffers pending kernels use.
     */
    public GpuBuffer scratch(int slot, long bytes) {
        if (slot < 0) {
            throw new IllegalArgumentException("scratch slot must be non-negative, got " + slot);
        }
        if (bytes <= 0) {
            throw new IllegalArgumentException("scratch size must be positive, got " + bytes);
        }
        lock.lock();
        try {
            ensureOpen();
            GpuBuffer current = scratch.get(slot);
            if (current != null && current.sizeInBytes() >= bytes) {
                return current;
            }
            long grown = current == null ? bytes : Math.max(bytes, current.sizeInBytes() + current.sizeInBytes() / 2);
            GpuBuffer made = backend.allocateBuffer(KernelLauncher.roundUp(grown, Integer.BYTES));
            scratch.put(slot, made);
            if (current != null) {
                current.close();
            }
            return made;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Releases this context: through {@link #release} while it is the backend's current context (the next
     * {@link #forBackend} opens a new one), otherwise it only closes this instance. Idempotent.
     */
    @Override
    public void close() {
        lock.lock();
        try {
            if (existing(backend) == this) {
                BackendResources.detach(backend, KEY);
            } else {
                closeInternal();
            }
        } finally {
            lock.unlock();
        }
    }

    @Override
    public String toString() {
        return "TrainGpu{" + describe(backend) + (closed ? ", closed" : "") + "}";
    }

    // ------------------------------------------------------------------ internals

    private void orderHostAccess(ComputeQueue q, List<GpuEvent> waits) {
        lock.lock();
        try {
            if (closed) {
                return;
            }
            if (errorsArmed) {
                checkErrors();
            }
            if (pendingWork && q != compute) {
                GpuEvent marker = compute.enqueueMarker();
                if (marker != null) {
                    waits.add(marker);
                }
            }
        } finally {
            lock.unlock();
        }
    }

    private void orderAfterUpload(GpuEvent uploadEvent) {
        lock.lock();
        try {
            if (closed) {
                return;
            }
            GpuEvent marker = compute.enqueueMarker(uploadEvent);
            if (marker != null) {
                marker.close();
            }
        } finally {
            lock.unlock();
        }
    }

    private int readWord(GpuBuffer buf, long elementOffset) {
        Objects.requireNonNull(buf, "buffer must not be null");
        requireWordRange(buf, elementOffset, 1);
        lock.lock();
        try {
            ensureOpen();
            byte[] word = new byte[Integer.BYTES];
            byte[] slots = errorsArmed ? new byte[(int) ERROR_BYTES] : null;
            GpuEvent read = compute.enqueueReadBuffer(buf, elementOffset * Integer.BYTES, Integer.BYTES, word);
            GpuEvent slotRead = null;
            try {
                if (slots != null) {
                    slotRead = compute.enqueueReadBuffer(errors, 0, ERROR_BYTES, slots);
                }
            } finally {
                finishAndClose(read);
                closeEvent(slotRead);
            }
            if (slots != null) {
                throwIfErrors(slots);
            }
            return ByteBuffer.wrap(word).order(ByteOrder.nativeOrder()).getInt(0);
        } finally {
            lock.unlock();
        }
    }

    /** Finishes the compute queue (all train work and the reads complete), then closes the event; holds the lock. */
    private void finishAndClose(GpuEvent ev) {
        try {
            compute.finish();
            pendingWork = false;
        } finally {
            closeEvent(ev);
        }
    }

    /** Decodes drained error slots, clears them on the compute queue when set, and throws; holds the lock. */
    private void throwIfErrors(byte[] slots) {
        errorsArmed = false;
        ByteBuffer words = ByteBuffer.wrap(slots).order(ByteOrder.nativeOrder());
        StringBuilder failed = new StringBuilder();
        for (int s = 0; s < ERROR_SLOT_COUNT; s++) {
            int base = s * ERROR_SLOT_WORDS * Integer.BYTES;
            if (words.getInt(base) != 0) {
                String message = errorMessage(s, words.getInt(base + Integer.BYTES),
                        words.getInt(base + 2 * Integer.BYTES), words.getInt(base + 3 * Integer.BYTES));
                failed.append(failed.length() == 0 ? "" : "; ").append(message);
            }
        }
        if (failed.length() == 0) {
            return;
        }
        compute.enqueueFillBuffer(errors, 0, ERROR_BYTES, 0);
        pendingWork = true;
        throw new IndexOutOfBoundsException(failed + " (reported by a device kernel on " + describe(backend)
                + "; a gather or embedding wrote zeros for out-of-range indices, a scatter skipped them)");
    }

    /** The CPU path's message for the index, dimension and size recorded in error slot {@code slot}. */
    private static String errorMessage(int slot, int index, int dim, int size) {
        switch (slot) {
            case ERROR_SLOT_GATHER:
                return gatherIndexMessage(index, dim, size);
            case ERROR_SLOT_SCATTER:
                return scatterIndexMessage(index, dim, size);
            case ERROR_SLOT_INDEX_SELECT:
                return embeddingIndexMessage(index);
            default:
                throw new IllegalStateException("No error slot " + slot);
        }
    }

    private void noteLaunch(boolean launched, Object[] args) {
        if (!launched) {
            return;
        }
        pendingWork = true;
        for (Object a : args) {
            if (a == errors) {
                errorsArmed = true;
                break;
            }
        }
    }

    /** Closes the buffers; called once, through the {@link Slot}, under the backend lock. */
    private void closeInternal() {
        lock.lock();
        try {
            if (closed) {
                return;
            }
            closed = true;
            RuntimeException failure = null;
            try {
                if (pendingWork) {
                    compute.finish();
                    pendingWork = false;
                }
            } catch (RuntimeException e) {
                failure = e;
            }
            for (GpuBuffer b : scratch.values()) {
                failure = closeBuffer(b, failure);
            }
            scratch.clear();
            failure = closeBuffer(placeholder, failure);
            failure = closeBuffer(errors, failure);
            if (failure != null) {
                throw failure;
            }
        } finally {
            lock.unlock();
        }
    }

    private static RuntimeException closeBuffer(GpuBuffer b, RuntimeException failure) {
        try {
            b.close();
            return failure;
        } catch (RuntimeException e) {
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
            return failure;
        }
    }

    private static void closeEvent(GpuEvent ev) {
        if (ev != null) {
            ev.close();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the train context of " + describe(backend) + " is closed");
        }
    }

    private static void requireElements(long n, String what) {
        if (n < 0 || n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(what + " must be in [0, " + Integer.MAX_VALUE + "], got " + n
                    + " (the train kernel ABI is 32-bit)");
        }
    }

    private static void requireWordRange(GpuBuffer buf, long elementOffset, long elements) {
        long words = buf.sizeInBytes() / Integer.BYTES;
        if (elementOffset < 0 || elements < 0 || elementOffset > words - elements || words > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("word range [" + elementOffset + ", " + (elementOffset + elements)
                    + ") is outside the buffer of " + words + " words");
        }
    }

    private static String describe(ComputeBackend b) {
        String device;
        try {
            device = b.capabilities().getDeviceName();
        } catch (RuntimeException e) {
            device = null;
        }
        return b.backendType() + (device == null || device.isEmpty() ? "" : " (" + device + ")");
    }

    /** The {@link BackendResources} attachment: a context, or the reason there is none. */
    private static final class Slot implements AutoCloseable {
        final TrainGpu gpu;
        final Availability availability;

        private Slot(TrainGpu gpu, Availability availability) {
            this.gpu = gpu;
            this.availability = availability;
        }

        /** Opens the library on {@code b}; runs under the backend lock inside {@link BackendResources#attachment}. */
        static Slot open(ComputeBackend b) {
            String reason;
            KernelSet set = null;
            Availability a = KernelSet.availability(b, TrainKernels.LIBRARY);
            if (!a.available()) {
                reason = a.reason();
            } else if (b.capabilities().getMaxWorkGroupSize() < LINEAR_LOCAL) {
                reason = "the device runs work-groups of at most " + b.capabilities().getMaxWorkGroupSize()
                        + " items, train kernels need " + LINEAR_LOCAL;
            } else {
                try {
                    set = KernelSet.open(b, TrainKernels.LIBRARY);
                    reason = prewarm(set);
                } catch (UnsupportedKernelException e) {
                    reason = e.getMessage();
                }
            }
            if (reason != null) {
                LOGGER.warn("Train kernels unavailable on {}: {}. Tensor ops on this backend take their CPU path.",
                        describe(b), reason);
                return new Slot(null, Availability.no(reason));
            }
            ComputeQueue q = b.computeQueue();
            GpuBuffer placeholder = null;
            GpuBuffer errors = null;
            try {
                placeholder = b.allocateBuffer(PLACEHOLDER_BYTES, true);
                errors = b.allocateBuffer(ERROR_BYTES, false);
                q.enqueueFillBuffer(placeholder, 0, PLACEHOLDER_BYTES, 0);
                q.enqueueFillBuffer(errors, 0, ERROR_BYTES, 0);
            } catch (RuntimeException e) {
                for (GpuBuffer made : new GpuBuffer[]{placeholder, errors}) {
                    if (made != null) {
                        e = closeBuffer(made, e);
                    }
                }
                throw e;
            }
            LOGGER.debug("Train kernels ready on {}: {} kernels prewarmed", describe(b), set.kernelNames().size());
            return new Slot(new TrainGpu(b, set, BackendResources.lock(b), placeholder, errors), Availability.ok());
        }

        /** Loads every listed kernel and prepares it at its local size; the first failure's reason, or null. */
        private static String prewarm(KernelSet set) {
            for (String name : new TreeSet<>(set.kernelNames())) {
                long[] local = TrainOps.localSize(name);
                try {
                    set.prewarm(name, local);
                } catch (UnsupportedKernelException | IllegalArgumentException | GpuBackendException e) {
                    return "kernel " + name + " cannot run at local size " + Arrays.toString(local) + ": "
                            + e.getMessage();
                }
            }
            return null;
        }

        @Override
        public void close() {
            if (gpu != null) {
                gpu.closeInternal();
            }
        }
    }
}
