package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.media.MediaKey;

import java.io.IOException;
import java.util.Locale;
import java.util.Objects;

/**
 * A store of encoder outputs shared between servers, keyed by the item's {@link MediaKey} (vLLM's encoder cache
 * connector, {@code vllm/distributed/ec_transfer/ec_connector/base.py}): an encoder-prefill-decode deployment runs
 * one server whose encoders publish the encoded items and others that load them instead of running the encoders
 * themselves (vLLM's EPD, {@code --ec-transfer-config}).
 *
 * <p>The protocol is vLLM's: a {@link Role#PRODUCER} saves the encoder output of every item it encodes
 * ({@link #put}, vLLM's {@code save_caches}), a {@link Role#CONSUMER} asks whether an item's output is there
 * ({@link #has}, vLLM's {@code has_cache_item}) and loads it ({@link #get}, vLLM's {@code start_load_caches}), and a
 * {@link Role#BOTH} does each. A {@link MediaContext} with a consumer connector loads an item before it decodes or
 * encodes it, so a consumer needs neither the media data nor the encoder for an item the producer published.
 * Implementations are safe for concurrent use by the media contexts of one server. Immutable role; close to release
 * the connector's resources.
 */
public interface EncoderCacheConnector extends AutoCloseable {

    /** Which side of a transfer the server is on (vLLM's {@code ec_role}). */
    enum Role {
        /** Saves the encoder output of every item it encodes ({@code ec_producer}). */
        PRODUCER("ec_producer"),
        /** Loads the encoder output producers saved ({@code ec_consumer}). */
        CONSUMER("ec_consumer"),
        /** Saves and loads ({@code ec_both}). */
        BOTH("ec_both");

        private final String label;

        Role(String label) {
            this.label = label;
        }

        /** vLLM's name of the role. */
        public String label() {
            return label;
        }

        /** Whether this side saves what it encodes. */
        public boolean produces() {
            return this != CONSUMER;
        }

        /** Whether this side loads what others saved. */
        public boolean consumes() {
            return this != PRODUCER;
        }

        /**
         * The role of a vLLM name ({@code ec_producer}, {@code ec_consumer} or {@code ec_both}), or of the short
         * name ({@code producer}, {@code consumer}, {@code both}), ignoring case.
         *
         * @throws IllegalArgumentException for an unknown name, naming the accepted ones
         */
        public static Role parse(String value) {
            String s = Objects.requireNonNull(value, "role").trim().toLowerCase(Locale.ROOT);
            for (Role role : values()) {
                if (role.label.equals(s) || role.name().toLowerCase(Locale.ROOT).equals(s)) {
                    return role;
                }
            }
            throw new IllegalArgumentException("'" + value + "' is not an encoder cache role; use " + PRODUCER.label
                    + ", " + CONSUMER.label + " or " + BOTH.label);
        }
    }

    /** Which side of a transfer this connector is on. */
    Role role();

    /** Whether this side saves what it encodes ({@link Role#produces}). */
    default boolean isProducer() {
        return role().produces();
    }

    /** Whether this side loads what others saved ({@link Role#consumes}). */
    default boolean isConsumer() {
        return role().consumes();
    }

    /** Whether the encoder output of the item {@code key} is stored. */
    boolean has(MediaKey key);

    /**
     * The encoder output stored for {@code key}, or null when it is not there.
     *
     * @throws IOException when the store cannot be read, or the stored entry is malformed
     */
    EncoderCacheEntry get(MediaKey key) throws IOException;

    /**
     * Stores {@code entry} as the encoder output of {@code key}, replacing an entry already there.
     *
     * @throws IllegalArgumentException when this connector's role does not produce ({@link Role#CONSUMER})
     * @throws IOException              when the store cannot be written
     */
    void put(MediaKey key, EncoderCacheEntry entry) throws IOException;

    /** Releases what the connector holds; idempotent. */
    @Override
    default void close() throws IOException {
    }
}
