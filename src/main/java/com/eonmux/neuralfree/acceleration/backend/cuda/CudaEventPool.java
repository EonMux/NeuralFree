package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.utils.LoggingUtil;
import jcuda.driver.CUevent;
import jcuda.driver.CUevent_flags;
import jcuda.driver.CUstream;
import org.apache.logging.log4j.Logger;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventDestroy;
import static jcuda.driver.JCudaDriver.cuEventRecord;

/**
 * Free list of timing-disabled CUDA events, so recording an event on the
 * dispatch path allocates no driver object once the pool is warm.
 *
 * <p>Every event the pool creates stays registered until {@link #close()}, which
 * destroys all of them, including events still held by open {@link CudaGpuEvent}s;
 * those handles then report a closed backend. Safe for use from several threads.</p>
 */
public final class CudaEventPool implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(CudaEventPool.class);

    private final CudaContext context;
    private final ArrayDeque<CUevent> free = new ArrayDeque<>();
    private final Set<CUevent> created = Collections.newSetFromMap(new IdentityHashMap<>());
    private boolean closed;

    /**
     * @throws IllegalArgumentException when {@code context} is null
     */
    public CudaEventPool(CudaContext context) {
        if (context == null) {
            throw new IllegalArgumentException("CUDA context must not be null");
        }
        this.context = context;
    }

    /** Records a pooled event on {@code stream}; the event has no owning queue. */
    public CudaGpuEvent record(CUstream stream) {
        return record(stream, null);
    }

    /**
     * Records a pooled event on {@code stream} owned by {@code owner}, which retires its
     * reads when the event is awaited. The event takes the owner's next sequence number.
     *
     * @throws IllegalArgumentException when {@code stream} is null
     * @throws IllegalStateException    when the pool is closed
     * @throws RuntimeException         the translated driver failure; the event returns to the pool
     */
    public CudaGpuEvent record(CUstream stream, CudaComputeQueue owner) {
        if (stream == null) {
            throw new IllegalArgumentException("CUDA stream must not be null");
        }
        CUevent event = obtain();
        try {
            CudaErrors.check("cuEventRecord", () -> cuEventRecord(event, stream));
        } catch (RuntimeException e) {
            recycle(event);
            throw e;
        }
        long sequence = owner == null ? 0L : owner.nextSequence();
        return new CudaGpuEvent(this, event, owner, sequence);
    }

    /**
     * Takes an event from the free list, creating one ({@code CU_EVENT_DISABLE_TIMING}) when
     * it is empty.
     *
     * @throws IllegalStateException when the pool is closed
     * @throws RuntimeException      the translated driver failure of {@code cuEventCreate}
     */
    public CUevent obtain() {
        synchronized (this) {
            ensureOpen();
            CUevent pooled = free.pollFirst();
            if (pooled != null) {
                return pooled;
            }
        }
        context.makeCurrent();
        CUevent event = new CUevent();
        CudaErrors.check("cuEventCreate", () -> cuEventCreate(event, CUevent_flags.CU_EVENT_DISABLE_TIMING));
        synchronized (this) {
            if (closed) {
                destroyAfterClose(event);
                throw new IllegalStateException("CUDA event pool was closed while an event was being created");
            }
            created.add(event);
        }
        return event;
    }

    /**
     * Returns an event to the free list. After {@link #close()} this does nothing, because
     * close has destroyed every event of the pool.
     *
     * @throws IllegalArgumentException when the event was not created by this pool
     */
    public synchronized void recycle(CUevent event) {
        if (closed) {
            return;
        }
        if (event == null || !created.contains(event)) {
            throw new IllegalArgumentException("CUDA event was not created by this pool");
        }
        free.addFirst(event);
    }

    /** True once {@link #close()} has run. */
    public synchronized boolean isClosed() {
        return closed;
    }

    /** Number of events created and not yet destroyed, for diagnostics. */
    public synchronized int createdCount() {
        return created.size();
    }

    /**
     * Destroys every event the pool created; idempotent. The context must still be alive.
     *
     * @throws RuntimeException the first translated driver failure, after every event was attempted
     */
    @Override
    public void close() {
        CUevent[] all;
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            all = created.toArray(new CUevent[0]);
            created.clear();
            free.clear();
        }
        if (all.length == 0) {
            return;
        }
        context.makeCurrent();
        RuntimeException failure = null;
        for (CUevent event : all) {
            try {
                CudaErrors.check("cuEventDestroy", () -> cuEventDestroy(event));
            } catch (RuntimeException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** The context the pooled events belong to. */
    CudaContext context() {
        return context;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("CUDA event pool is closed (its backend was closed)");
        }
    }

    private static void destroyAfterClose(CUevent event) {
        try {
            CudaErrors.check("cuEventDestroy", () -> cuEventDestroy(event));
        } catch (RuntimeException e) {
            LOGGER.warn("CUDA event pool: destroying an event created during close failed: {}", e.getMessage());
        }
    }
}
