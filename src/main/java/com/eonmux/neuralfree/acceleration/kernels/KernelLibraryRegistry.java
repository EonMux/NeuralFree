package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.ThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of kernel libraries, keyed by library id and backend type. A library id names the same
 * set of kernels on every backend; each backend type has its own {@link KernelLibrary}.
 */
@ThreadSafe
public final class KernelLibraryRegistry {

    private static final KernelLibraryRegistry GLOBAL = new KernelLibraryRegistry();

    private final Map<Key, KernelLibrary> libraries = new ConcurrentHashMap<>();

    /** Creates an empty registry; engine code uses {@link #global()}. */
    public KernelLibraryRegistry() {
    }

    /** The process-wide registry that {@link KernelSet} resolves against. */
    public static KernelLibraryRegistry global() {
        return GLOBAL;
    }

    /**
     * Registers a library, replacing any library with the same id and backend type.
     *
     * <p>OpenCL backends cache built programs by {@link OpenClKernelLibrary#programKey}, which names the
     * library id but not its sources, so an OpenCL library may only be replaced by one that assembles the
     * same source root; a different source tree needs its own library id.</p>
     *
     * @return the replaced library, or null
     * @throws IllegalArgumentException when an OpenCL library would be replaced by one with another source root
     */
    public KernelLibrary register(KernelLibrary library) {
        Key key = keyOf(library);
        KernelLibrary[] replaced = new KernelLibrary[1];
        libraries.compute(key, (k, previous) -> {
            requireSameOpenClSources(previous, library);
            replaced[0] = previous;
            return library;
        });
        return replaced[0];
    }

    /**
     * Registers a library unless one with the same id and backend type exists.
     *
     * @return true when this library was registered
     */
    public boolean registerIfAbsent(KernelLibrary library) {
        Key key = keyOf(library);
        return libraries.putIfAbsent(key, library) == null;
    }

    /** The library for an id and backend type, if registered. */
    public Optional<KernelLibrary> get(String libraryId, ComputeBackend.BackendType type) {
        if (libraryId == null || type == null) {
            throw new IllegalArgumentException("library id and backend type must not be null");
        }
        return Optional.ofNullable(libraries.get(new Key(libraryId, type)));
    }

    /** Unmodifiable, sorted set of the registered library ids. */
    public Set<String> libraryIds() {
        Set<String> ids = new TreeSet<>();
        for (Key k : libraries.keySet()) {
            ids.add(k.libraryId);
        }
        return Collections.unmodifiableSet(ids);
    }

    private static void requireSameOpenClSources(KernelLibrary previous, KernelLibrary next) {
        if (!(previous instanceof OpenClKernelLibrary) || !(next instanceof OpenClKernelLibrary)) {
            return;
        }
        String before = ((OpenClKernelLibrary) previous).config().sourceRoot();
        String after = ((OpenClKernelLibrary) next).config().sourceRoot();
        if (!before.equals(after)) {
            throw new IllegalArgumentException("OpenCL kernel library '" + next.libraryId() + "' serves classpath:"
                    + before + "; it cannot be replaced by sources from classpath:" + after
                    + " because OpenCL programs are cached by library id; register those sources under another id");
        }
    }

    private static Key keyOf(KernelLibrary library) {
        Objects.requireNonNull(library, "library must not be null");
        String id = library.libraryId();
        ComputeBackend.BackendType type = library.backendType();
        if (id == null || id.isEmpty() || type == null) {
            throw new IllegalArgumentException("library " + library + " has no id or backend type");
        }
        return new Key(id, type);
    }

    private static final class Key {
        final String libraryId;
        final ComputeBackend.BackendType type;

        Key(String libraryId, ComputeBackend.BackendType type) {
            this.libraryId = libraryId;
            this.type = type;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof Key)) {
                return false;
            }
            Key that = (Key) o;
            return libraryId.equals(that.libraryId) && type == that.type;
        }

        @Override
        public int hashCode() {
            return libraryId.hashCode() * 31 + type.hashCode();
        }
    }
}
