package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaKey;
import com.eonmux.neuralfree.gguf.media.MediaLimits;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * One appended item: its key, its request's adapters, and its bytes, read at most once.
 */
final class MediaItem {
    private final MediaLimits.Modality modality;
    private final String uuid;
    private final Supplier<byte[]> data;
    private final MediaKey key;
    private final MediaKey baseKey;
    private final LoraSlotKey lora;
    private byte[] bytes;

    /**
     * The item a client named {@code uuid} (its key the uuid's), or of {@code data} without a uuid (its key the
     * digest of the bytes, read now), under the loader options' hash factors and the request's adapters: the
     * encoded-media cache and the prefix cache key the item by the adapter identity ({@code key}), the
     * processed layer by the plain item ({@code baseKey}).
     *
     * @throws MediaInputException when there is neither a uuid nor data
     */
    MediaItem(MediaContext context, MediaLimits.Modality modality, String uuid, Supplier<byte[]> data,
              LoraSlotKey lora) {
        this.modality = modality;
        this.uuid = uuid;
        this.data = data;
        this.lora = lora;
        if (uuid == null) {
            if (data == null) {
                throw new MediaInputException("the " + modality.key() + " has neither data nor a uuid");
            }
            bytes = Objects.requireNonNull(data.get(), "media data");
        }
        this.baseKey = context.keyOf(modality, uuid, bytes);
        this.key = context.keyOf(modality, uuid, bytes, lora);
    }

    /** The item under the key {@code key} as given, its bytes read at most once. */
    MediaItem(MediaContext context, MediaLimits.Modality modality, MediaKey key, Supplier<byte[]> data) {
        this.modality = modality;
        this.uuid = null;
        this.data = data;
        this.lora = null;
        this.key = Objects.requireNonNull(key, "key");
        this.baseKey = this.key;
    }

    /** The item's key in the encoded-media cache and the prefix cache, the adapters' identity folded in. */
    MediaKey key() {
        return key;
    }

    /** The item's key in the processed layer, which the adapters do not run. */
    MediaKey baseKey() {
        return baseKey;
    }

    /** The request's adapters, or null when it names none. */
    LoraSlotKey lora() {
        return lora;
    }

    /**
     * The item's bytes, read now when not yet; a key the cache does not hold without data is refused (vLLM's
     * cache miss with the data not provided).
     */
    byte[] bytes() {
        if (bytes == null) {
            if (data == null) {
                String kind = modality.key();
                String name = uuid != null ? uuid : key.id();
                throw new MediaInputException("the " + kind + " with uuid '" + name + "' is not in the media"
                        + " cache and its data is not provided; send the " + kind + " with its uuid again");
            }
            bytes = Objects.requireNonNull(data.get(), "media data");
        }
        return bytes;
    }
}
