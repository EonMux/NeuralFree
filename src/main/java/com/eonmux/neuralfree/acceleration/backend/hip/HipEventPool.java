package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.hip.HipDriver;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Set;

/**
 * Free list of timing-disabled HIP events, so recording an event on the dispatch path allocates
 * no runtime object once the pool is warm.
 *
 * <p>Every event the pool creates stays registered until {@link #close()}, which destroys all of
 * them, including events still held by open {@link HipGpuEvent}s; those handles then report a
 * closed backend. Safe for use from several threads.</p>
 */
public final class HipEventPool implements AutoCloseable {

    private static final Logger LOGGER = LoggingUtil.getLogger(HipEventPool.class);

    private final HipContext context;
    private final HipDriver driver;
    private final ArrayDeque<Long> free = new ArrayDeque<>();
    /** Events created and not yet destroyed, compared by value (handles are boxed longs). */
    private final Set<Long> created = new HashSet<>();
    private boolean closed;

    /**
     * @param context the device context the pooled events belong to
     * @param driver  the runtime operations the pool creates and destroys events with
     * @throws IllegalArgumentException when an argument is null
     */
    public HipEventPool(HipContext context, HipDriver driver) {
        if (context == null) {
            throw new IllegalArgumentException("HIP context must not be null");
        }
        if (driver == null) {
            throw new IllegalArgumentException("HIP driver must not be null");
        }
        this.context = context;
        this.driver = driver;
    }

    /** Records a pooled event on {@code stream}; the event has no owning queue. */
    public HipGpuEvent record(long stream) {
        return record(stream, null);
    }

    /**
     * Records a pooled event on {@code stream} owned by {@code owner}, which retires its reads
     * when the event is awaited. The event takes the owner's next sequence number.
     *
     * @throws IllegalArgumentException when {@code stream} is null
     * @throws IllegalStateException    when the pool is closed
     * @throws RuntimeException         the translated driver failure; the event returns to the pool
     */
    public HipGpuEvent record(long stream, HipComputeQueue owner) {
        long event = obtain();
        try {
            driver.eventRecord(event, stream);
        } catch (RuntimeException e) {
            recycle(event);
            throw e;
        }
        long sequence = owner == null ? 0L : owner.nextSequence();
        return new HipGpuEvent(this, event, owner, sequence);
    }

    /**
     * Takes an event from the free list, creating one when it is empty.
     *
     * @throws IllegalStateException when the pool is closed
     * @throws RuntimeException      the translated driver failure of {@code hipEventCreateWithFlags}
     */
    public long obtain() {
        synchronized (this) {
            ensureOpen();
            Long pooled = free.pollFirst();
            if (pooled != null) {
                return pooled;
            }
        }
        context.makeCurrent(driver);
        long event = driver.eventCreate();
        synchronized (this) {
            if (closed) {
                destroyAfterClose(event);
                throw new IllegalStateException("HIP event pool was closed while an event was being created");
            }
            created.add(event);
        }
        return event;
    }

    /**
     * Returns an event to the free list. After {@link #close()} this does nothing, because close
     * has destroyed every event of the pool.
     *
     * @throws IllegalArgumentException when the event was not created by this pool
     */
    public synchronized void recycle(long event) {
        if (closed) {
            return;
        }
        if (event == 0 || !created.contains(event)) {
            throw new IllegalArgumentException("HIP event was not created by this pool");
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
     * Destroys every event the pool created; idempotent. The device context must still be
     * usable.
     *
     * @throws RuntimeException the first translated driver failure, after every event was attempted
     */
    @Override
    public void close() {
        Long[] all;
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            all = created.toArray(new Long[0]);
            created.clear();
            free.clear();
        }
        if (all.length == 0) {
            return;
        }
        context.makeCurrent(driver);
        RuntimeException failure = null;
        for (Long event : all) {
            try {
                driver.eventDestroy(event);
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
    HipContext context() {
        return context;
    }

    /** The driver the pool creates and destroys events with. */
    HipDriver driver() {
        return driver;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("HIP event pool is closed (its backend was closed)");
        }
    }

    private void destroyAfterClose(long event) {
        try {
            driver.eventDestroy(event);
        } catch (RuntimeException e) {
            LOGGER.warn("HIP event pool: destroying an event created during close failed: {}", e.getMessage());
        }
    }
}
