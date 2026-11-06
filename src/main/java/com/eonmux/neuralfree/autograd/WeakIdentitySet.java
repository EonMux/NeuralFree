package com.eonmux.neuralfree.autograd;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * A set that compares its elements by identity and holds them weakly: an element nobody else references leaves the
 * set once the garbage collector clears it, so membership never keeps an object reachable. Null elements are refused.
 *
 * <p>Not thread-safe; the owner confines it to one thread. {@link #size()} counts the live elements and
 * {@link #iterator()} walks a snapshot of them; the iterator's {@code remove} removes from the set.</p>
 */
final class WeakIdentitySet<T> extends AbstractSet<T> {

    private final ReferenceQueue<T> cleared = new ReferenceQueue<>();
    private final Map<Ref<T>, Boolean> refs = new HashMap<>();

    @Override
    public boolean add(T element) {
        Objects.requireNonNull(element, "element must not be null");
        expunge();
        Ref<T> probe = new Ref<>(element, null);
        if (refs.containsKey(probe)) {
            return false;
        }
        refs.put(new Ref<>(element, cleared), Boolean.TRUE);
        return true;
    }

    @Override
    public boolean contains(Object element) {
        if (element == null) {
            return false;
        }
        expunge();
        return refs.containsKey(new Ref<>(element, null));
    }

    @Override
    public boolean remove(Object element) {
        if (element == null) {
            return false;
        }
        expunge();
        return refs.remove(new Ref<>(element, null)) != null;
    }

    @Override
    public int size() {
        expunge();
        refs.keySet().removeIf(ref -> ref.get() == null);
        return refs.size();
    }

    @Override
    public void clear() {
        refs.clear();
        while (cleared.poll() != null) {
            // the references were dropped with the map
        }
    }

    @Override
    public Iterator<T> iterator() {
        expunge();
        List<T> live = new ArrayList<>(refs.size());
        for (Ref<T> ref : refs.keySet()) {
            T element = ref.get();
            if (element != null) {
                live.add(element);
            }
        }
        return new Iterator<T>() {
            private int next;
            private T last;

            @Override
            public boolean hasNext() {
                return next < live.size();
            }

            @Override
            public T next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                last = live.get(next++);
                return last;
            }

            @Override
            public void remove() {
                if (last == null) {
                    throw new IllegalStateException("next() was not called, or the element was already removed");
                }
                WeakIdentitySet.this.remove(last);
                last = null;
            }
        };
    }

    /** Drops the references the collector cleared. */
    private void expunge() {
        for (Reference<? extends T> ref = cleared.poll(); ref != null; ref = cleared.poll()) {
            refs.remove(ref);
        }
    }

    /** A weak reference equal to another while both reach the same object; a cleared one equals only itself. */
    private static final class Ref<T> extends WeakReference<T> {
        private final int hash;

        @SuppressWarnings("unchecked")
        Ref(Object referent, ReferenceQueue<T> queue) {
            super((T) referent, queue);
            this.hash = System.identityHashCode(referent);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof Ref)) {
                return false;
            }
            Object mine = get();
            return mine != null && mine == ((Ref<?>) other).get();
        }
    }
}
