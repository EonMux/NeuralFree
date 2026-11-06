package com.eonmux.neuralfree.gguf.remote;

import java.util.Objects;

/**
 * One object of an object-store listing: its full key, its byte size and its entity tag. Immutable.
 */
public final class RemoteObject {

    private final String key;
    private final long size;
    private final String etag;

    /**
     * @param key  the object's full key
     * @param size the listed byte size, or -1 when the listing carries none
     * @param etag the entity tag, or null when the listing carries none
     */
    public RemoteObject(String key, long size, String etag) {
        this.key = Objects.requireNonNull(key, "key");
        this.size = size;
        this.etag = etag;
    }

    /** The object's full key. */
    public String key() {
        return key;
    }

    /** The listed byte size, or -1 when unknown. */
    public long size() {
        return size;
    }

    /** The entity tag, or null when the listing carries none. */
    public String etag() {
        return etag;
    }

    @Override
    public String toString() {
        return key + (size >= 0 ? " (" + size + " bytes)" : "") + (etag == null ? "" : " " + etag);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof RemoteObject)) {
            return false;
        }
        RemoteObject other = (RemoteObject) o;
        return size == other.size && key.equals(other.key) && Objects.equals(etag, other.etag);
    }

    @Override
    public int hashCode() {
        return Objects.hash(key, size, etag);
    }
}
