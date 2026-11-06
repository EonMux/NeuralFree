package com.eonmux.neuralfree.gguf.remote;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * A read-only object store: lists the objects under a prefix and copies one object's bytes to a file. The three
 * implementations are {@link S3ObjectStore}, {@link GcsObjectStore} and {@link AzureBlobObjectStore}; {@link
 * ObjectStores} builds the one an {@link ObjectStoreUri} names. Implementations are immutable and thread-safe.
 */
public interface ObjectStore extends AutoCloseable {

    /** What was opened (store, bucket and endpoint), for messages. */
    String description();

    /**
     * The objects whose keys start with {@code prefix}, across every listing page, in key order.
     *
     * @param prefix the key prefix; empty for the bucket root
     * @throws IOException when the store cannot be listed or answers with an invalid listing
     */
    List<RemoteObject> list(String prefix) throws IOException;

    /**
     * Copies the object {@code key} to {@code destination}, which does not exist yet.
     *
     * @throws java.io.FileNotFoundException when the store has no such object
     * @throws IOException                   when the object cannot be read or written
     */
    void download(String key, Path destination) throws IOException;

    @Override
    default void close() throws IOException {
    }
}
