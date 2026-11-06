package com.eonmux.neuralfree.gguf.remote;

import java.util.Locale;
import java.util.Objects;

/**
 * An object-storage checkpoint location, the URIs vLLM's Run:ai streamer loader accepts
 * ({@code runai_utils.SUPPORTED_SCHEMES}): {@code s3://<bucket>/<prefix>} for S3 and S3-compatible stores,
 * {@code gs://<bucket>/<prefix>} for Google Cloud Storage and {@code az://<container>/<prefix>} (with
 * {@code azure://} as an alias) for Azure Blob Storage. The prefix names a checkpoint directory, or one object when
 * it is the object's full key. Immutable.
 */
public final class ObjectStoreUri {

    /** The object stores a URI can name. */
    public enum Scheme {
        /** S3 or an S3-compatible store (MinIO, Ceph, R2 with the S3 API). */
        S3("s3"),
        /** Google Cloud Storage. */
        GCS("gs"),
        /** Azure Blob Storage. */
        AZURE("az");

        private final String prefix;

        Scheme(String prefix) {
            this.prefix = prefix;
        }

        /** The scheme before {@code ://}. */
        public String prefix() {
            return prefix;
        }
    }

    private final Scheme scheme;
    private final String bucket;
    private final String prefix;

    private ObjectStoreUri(Scheme scheme, String bucket, String prefix) {
        this.scheme = scheme;
        this.bucket = bucket;
        this.prefix = prefix;
    }

    /**
     * Whether {@code path} starts with one of the supported object-store schemes (case-insensitive), the test
     * vLLM's {@code is_runai_obj_uri} makes.
     */
    public static boolean isObjectStorePath(String path) {
        if (path == null) {
            return false;
        }
        String lower = path.toLowerCase(Locale.ROOT);
        return lower.startsWith("s3://") || lower.startsWith("gs://") || lower.startsWith("az://")
                || lower.startsWith("azure://");
    }

    /**
     * Parses an object-store URI.
     *
     * @throws IllegalArgumentException with the expected form when the scheme is unsupported, the bucket is empty,
     *                                  or the URI carries a query or fragment
     */
    public static ObjectStoreUri parse(String uri) {
        Objects.requireNonNull(uri, "uri");
        String trimmed = uri.trim();
        int mark = trimmed.indexOf("://");
        if (mark <= 0) {
            throw invalid(uri, "it has no '<scheme>://' prefix");
        }
        String schemeName = trimmed.substring(0, mark).toLowerCase(Locale.ROOT);
        Scheme scheme;
        if (schemeName.equals("s3")) {
            scheme = Scheme.S3;
        } else if (schemeName.equals("gs")) {
            scheme = Scheme.GCS;
        } else if (schemeName.equals("az") || schemeName.equals("azure")) {
            scheme = Scheme.AZURE;
        } else {
            throw invalid(uri, "the scheme '" + schemeName + "' is not one of s3, gs, az");
        }
        String rest = trimmed.substring(mark + 3);
        if (rest.indexOf('?') >= 0 || rest.indexOf('#') >= 0) {
            throw invalid(uri, "it carries a query or fragment");
        }
        int slash = rest.indexOf('/');
        String bucket = slash < 0 ? rest : rest.substring(0, slash);
        String prefix = slash < 0 ? "" : rest.substring(slash + 1);
        while (prefix.endsWith("/")) {
            prefix = prefix.substring(0, prefix.length() - 1);
        }
        while (prefix.startsWith("/")) {
            prefix = prefix.substring(1);
        }
        if (bucket.isEmpty()) {
            throw invalid(uri, "the bucket is empty");
        }
        if (bucket.indexOf(' ') >= 0) {
            throw invalid(uri, "the bucket holds a space");
        }
        return new ObjectStoreUri(scheme, bucket, prefix);
    }

    private static IllegalArgumentException invalid(String uri, String reason) {
        return new IllegalArgumentException("invalid object-store URI '" + uri + "': " + reason
                + "; expected s3://<bucket>/<prefix>, gs://<bucket>/<prefix> or az://<container>/<prefix>");
    }

    /** The store. */
    public Scheme scheme() {
        return scheme;
    }

    /** The bucket (S3, GCS) or container (Azure). */
    public String bucket() {
        return bucket;
    }

    /** The key prefix, without leading or trailing slashes; empty for the bucket root. */
    public String prefix() {
        return prefix;
    }

    /** The canonical URI, {@code <scheme>://<bucket>[/<prefix>]}. */
    @Override
    public String toString() {
        return scheme.prefix() + "://" + bucket + (prefix.isEmpty() ? "" : "/" + prefix);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ObjectStoreUri)) {
            return false;
        }
        ObjectStoreUri other = (ObjectStoreUri) o;
        return scheme == other.scheme && bucket.equals(other.bucket) && prefix.equals(other.prefix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(scheme, bucket, prefix);
    }
}
