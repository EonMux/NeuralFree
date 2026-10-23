package com.eonmux.neuralfree.acceleration.backend;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * Per-backend state shared by the tensor runtime: one lock per backend instance and a set of keyed
 * attachments that live as long as the backend.
 *
 * <p><b>Lock.</b> {@link #lock} returns the same {@link ReentrantLock} for a backend instance (identity, no
 * {@code equals}) until {@link #closeAll} runs, whether or not any kernel library is available. Every enqueue
 * that tensor code issues on the backend's shared {@code computeQueue()} or {@code transferQueue()} (launches,
 * fills and transfers) runs under it, which upholds the one-thread-per-queue contract of {@link ComputeQueue}.
 * Code holding one backend's lock never transfers on another backend.</p>
 *
 * <p><b>Attachments.</b> {@link #attachment} creates an {@link AutoCloseable} once per (backend, key) under the
 * backend's lock. {@link #closeAll} closes every attachment of a backend in reverse creation order and drops the
 * backend's entry; each backend's {@code close()} calls it first, while the device is still alive. Keys are
 * compared with {@code equals}; owners use private key objects.</p>
 *
 * <p>Backends are held weakly: an entry without attachments never keeps its backend reachable. An attachment
 * that references its backend keeps it reachable until it is detached or the backend closes.</p>
 */
public final class BackendResources {

    private static final Logger LOGGER = LogManager.getLogger(BackendResources.class);

    private static final ReferenceQueue<ComputeBackend> STALE = new ReferenceQueue<>();
    /** Guarded by itself. */
    private static final Map<Key, Entry> ENTRIES = new HashMap<>();

    private BackendResources() {
    }

    /** The lock of a backend instance; the same object for every call until {@link #closeAll}. */
    public static ReentrantLock lock(ComputeBackend b) {
        return entry(b).lock;
    }

    /**
     * The attachment of a backend under {@code key}, created by {@code factory} under the backend's lock when
     * absent. A factory failure propagates and stores nothing.
     *
     * @throws IllegalStateException when the factory returns null, or the backend's resources were closed
     *                               while this call waited for the lock
     */
    public static <T extends AutoCloseable> T attachment(ComputeBackend b, Object key, Supplier<T> factory) {
        Objects.requireNonNull(key, "key must not be null");
        Objects.requireNonNull(factory, "factory must not be null");
        Entry e = entry(b);
        e.lock.lock();
        try {
            if (e.closed) {
                throw new IllegalStateException("resources of " + describe(b) + " were closed with the backend");
            }
            AutoCloseable existing = e.get(key);
            if (existing != null) {
                return cast(existing);
            }
            T made = factory.get();
            if (made == null) {
                throw new IllegalStateException("attachment factory for key " + key + " returned null");
            }
            e.put(key, made);
            return made;
        } finally {
            e.lock.unlock();
        }
    }

    /** The attachment of a backend under {@code key}, or null when absent; never creates one. */
    public static <T extends AutoCloseable> T existing(ComputeBackend b, Object key) {
        Objects.requireNonNull(b, "backend must not be null");
        Objects.requireNonNull(key, "key must not be null");
        Entry e = lookup(b);
        if (e == null) {
            return null;
        }
        AutoCloseable a = e.get(key);
        return a == null ? null : cast(a);
    }

    /**
     * Closes and removes one attachment of a backend under the backend's lock; does nothing when absent.
     *
     * @throws RuntimeException from the attachment's {@code close()} (a checked exception is wrapped in an
     *                          {@link IllegalStateException}); the attachment is removed either way
     */
    public static void detach(ComputeBackend b, Object key) {
        Objects.requireNonNull(b, "backend must not be null");
        Objects.requireNonNull(key, "key must not be null");
        Entry e = lookup(b);
        if (e == null) {
            return;
        }
        e.lock.lock();
        try {
            AutoCloseable a = e.remove(key);
            if (a != null) {
                closeOne(a, key);
            }
        } finally {
            e.lock.unlock();
        }
    }

    /**
     * Detaches {@code key} from every backend that has it. Every backend is visited even when a close fails;
     * the first failure is rethrown at the end with the others suppressed.
     */
    public static void detachAll(Object key) {
        Objects.requireNonNull(key, "key must not be null");
        List<ComputeBackend> backends = new ArrayList<>();
        synchronized (ENTRIES) {
            expungeStale();
            for (Key k : ENTRIES.keySet()) {
                ComputeBackend b = k.get();
                if (b != null) {
                    backends.add(b);
                }
            }
        }
        RuntimeException failure = null;
        for (ComputeBackend b : backends) {
            try {
                detach(b, key);
            } catch (RuntimeException ex) {
                failure = merge(failure, ex);
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    /**
     * Closes every attachment of a backend in reverse creation order under the backend's lock, then drops the
     * backend's entry, so a later {@link #lock} returns a new lock. Every attachment is closed even when one
     * fails; each failure is logged and the first is rethrown with the others suppressed. Does nothing for a
     * backend without an entry, so it is idempotent.
     */
    public static void closeAll(ComputeBackend b) {
        Objects.requireNonNull(b, "backend must not be null");
        Entry e = lookup(b);
        if (e == null) {
            return;
        }
        RuntimeException failure = null;
        e.lock.lock();
        try {
            e.closed = true;
            List<Map.Entry<Object, AutoCloseable>> all = e.drain();
            for (int i = all.size() - 1; i >= 0; i--) {
                Map.Entry<Object, AutoCloseable> a = all.get(i);
                try {
                    closeOne(a.getValue(), a.getKey());
                } catch (RuntimeException ex) {
                    LOGGER.error("Closing the {} attachment of {} failed: {}", a.getKey(), describe(b), ex.toString());
                    failure = merge(failure, ex);
                }
            }
        } finally {
            e.lock.unlock();
            synchronized (ENTRIES) {
                Key k = new Key(b, null);
                if (ENTRIES.get(k) == e) {
                    ENTRIES.remove(k);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }

    private static Entry entry(ComputeBackend b) {
        Objects.requireNonNull(b, "backend must not be null");
        synchronized (ENTRIES) {
            expungeStale();
            Key lookupKey = new Key(b, null);
            Entry e = ENTRIES.get(lookupKey);
            if (e == null) {
                e = new Entry();
                ENTRIES.put(new Key(b, STALE), e);
            }
            return e;
        }
    }

    private static Entry lookup(ComputeBackend b) {
        synchronized (ENTRIES) {
            expungeStale();
            return ENTRIES.get(new Key(b, null));
        }
    }

    /** Removes entries whose backend was garbage collected; caller holds the ENTRIES monitor. */
    private static void expungeStale() {
        java.lang.ref.Reference<? extends ComputeBackend> r;
        while ((r = STALE.poll()) != null) {
            ENTRIES.remove(r);
        }
    }

    private static void closeOne(AutoCloseable a, Object key) {
        try {
            a.close();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("closing the attachment " + key + " failed: " + e.getMessage(), e);
        }
    }

    private static RuntimeException merge(RuntimeException first, RuntimeException next) {
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    @SuppressWarnings("unchecked")
    private static <T extends AutoCloseable> T cast(AutoCloseable a) {
        return (T) a;
    }

    private static String describe(ComputeBackend b) {
        return b.backendType() + " backend@" + Integer.toHexString(System.identityHashCode(b));
    }

    /** Weak identity key: equal to another key only for the same live backend, or to itself once cleared. */
    private static final class Key extends WeakReference<ComputeBackend> {
        private final int hash;

        Key(ComputeBackend b, ReferenceQueue<ComputeBackend> queue) {
            super(b, queue);
            this.hash = System.identityHashCode(b);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof Key)) {
                return false;
            }
            ComputeBackend mine = get();
            return mine != null && mine == ((Key) o).get();
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    /** One backend's lock and attachments; the attachment map is guarded by the entry's monitor. */
    private static final class Entry {
        final ReentrantLock lock = new ReentrantLock();
        private final Map<Object, AutoCloseable> attachments = new LinkedHashMap<>();
        /** Set under {@link #lock} by {@link #closeAll}. */
        boolean closed;

        synchronized AutoCloseable get(Object key) {
            return attachments.get(key);
        }

        synchronized void put(Object key, AutoCloseable value) {
            attachments.put(key, value);
        }

        synchronized AutoCloseable remove(Object key) {
            return attachments.remove(key);
        }

        /** Every attachment in creation order; the map is empty afterwards. */
        synchronized List<Map.Entry<Object, AutoCloseable>> drain() {
            List<Map.Entry<Object, AutoCloseable>> all = new ArrayList<>(attachments.entrySet().size());
            for (Map.Entry<Object, AutoCloseable> a : attachments.entrySet()) {
                all.add(new java.util.AbstractMap.SimpleImmutableEntry<>(a.getKey(), a.getValue()));
            }
            attachments.clear();
            return all;
        }
    }
}
