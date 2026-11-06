package com.eonmux.neuralfree.gguf.engine.kvevents;

import java.util.Objects;

/**
 * One extra key a cached block's hash mixes in beside its tokens, in the untagged shape vLLM publishes in
 * {@code BlockStored.extra_keys} ({@code to_event_extra_keys}): the content identity of the request's per-request
 * adapters on a chain's first block ({@link #lora}), a media item the block covers ({@link #media}: the item's
 * {@code mm_hash} — the client's uuid or a content digest — and its first row relative to the block's first row,
 * negative when the item began in an earlier block), the request's cache salt on a chain's first block
 * ({@link #salt}) and the digest of the block's prompt-embedding rows ({@link #promptEmbeds}). Immutable.
 */
public final class BlockExtraKey {

    /** What the key stands for. */
    public enum Kind {
        /** The content identity of the request's per-request adapters (vLLM's LoRA key). */
        LORA,
        /** A media item's identity and offset (vLLM's {@code (mm_hash, offset)} pair). */
        MEDIA,
        /** The request's cache salt (vLLM's {@code cache_salt}). */
        SALT,
        /** The SHA-256 digest of the block's prompt-embedding rows (vLLM's {@code prompt_embeds} key). */
        PROMPT_EMBEDS
    }

    private final Kind kind;
    private final String value;
    private final int offset;

    private BlockExtraKey(Kind kind, String value, int offset) {
        this.kind = kind;
        this.value = Objects.requireNonNull(value, "value");
        this.offset = offset;
    }

    /** The key of a chain's first block computed under the per-request adapters of content {@code identity}. */
    public static BlockExtraKey lora(String identity) {
        return new BlockExtraKey(Kind.LORA, identity, 0);
    }

    /** The key of a media item of identity {@code hash} whose first row is block row {@code offset}. */
    public static BlockExtraKey media(String hash, int offset) {
        return new BlockExtraKey(Kind.MEDIA, hash, offset);
    }

    /** The key of a chain's first block under the cache salt {@code salt}. */
    public static BlockExtraKey salt(String salt) {
        return new BlockExtraKey(Kind.SALT, salt, 0);
    }

    /** The key of a block whose prompt-embedding rows digest to {@code digest}. */
    public static BlockExtraKey promptEmbeds(String digest) {
        return new BlockExtraKey(Kind.PROMPT_EMBEDS, digest, 0);
    }

    /** What the key stands for. */
    public Kind kind() {
        return kind;
    }

    /** The adapters' content identity, the media item's identity, the cache salt, or the rows' digest. */
    public String value() {
        return value;
    }

    /** A media item's first row relative to the block's first row; 0 for the other kinds. */
    public int offset() {
        return offset;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof BlockExtraKey)) {
            return false;
        }
        BlockExtraKey k = (BlockExtraKey) o;
        return kind == k.kind && value.equals(k.value) && offset == k.offset;
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, value, offset);
    }

    /** vLLM's untagged event form: {@code (hash, offset)} for a media item, the bare value otherwise. */
    @Override
    public String toString() {
        return kind == Kind.MEDIA ? "(" + value + ", " + offset + ")" : value;
    }
}
