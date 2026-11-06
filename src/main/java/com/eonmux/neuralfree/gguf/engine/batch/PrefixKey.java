package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.lora.LoraPlan;

import java.util.Objects;

/**
 * The extra keys the first block of a prefix-cache chain mixes into its hash (vLLM's block-hash extra keys, the
 * LoRA first, then the {@code cache_salt}): the content identity of a request's per-request adapters
 * ({@link LoraPlan#identity()}) and its cache salt. Requests with different keys never share a block; requests
 * with equal keys do. The block hasher tags the two keys with their source
 * ({@code BlockHasher}, {@code ("lora", identity)} and {@code ("cache_salt", salt)}), so no salt equals an adapter
 * identity whatever its text. A chain's first block reports both in its KV event extra keys beside its row keys
 * ({@link CacheRows#extraKeys}). {@link #loraName()} names the adapters and {@link #loraId()} ids them in KV cache
 * events (vLLM's {@code lora_name} and {@code lora_id}); neither is part of the key, as rows computed under equal
 * content are interchangeable whatever the adapters are called. Immutable.
 */
final class PrefixKey {

    /** The key of a request with neither per-request adapters nor a cache salt. */
    static final PrefixKey NONE = new PrefixKey(null, null, null);

    private final String adapters;
    private final String loraName;
    private final String salt;

    private PrefixKey(String adapters, String loraName, String salt) {
        this.adapters = adapters;
        this.loraName = loraName;
        this.salt = salt;
    }

    /** The key of a request with cache salt {@code salt} (null for none) and the adapter slots {@code adapters}. */
    static PrefixKey of(String salt, LoraPlan adapters) {
        String identity = Objects.requireNonNull(adapters, "adapters").identity();
        if (identity == null) {
            return salt == null ? NONE : new PrefixKey(null, null, salt);
        }
        return new PrefixKey(identity, adapters.label(), salt);
    }

    /**
     * The key a slot file recorded: the adapters' content identity and names and the cache salt, each null for
     * none, as {@link #adapters()}, {@link #loraName()} and {@link #salt()} answered them when it was saved.
     */
    static PrefixKey restored(String adapters, String loraName, String salt) {
        if (adapters == null && salt == null) {
            return NONE;
        }
        return new PrefixKey(adapters, adapters == null ? null : loraName, salt);
    }

    /** The cache salt ({@code cache_salt}), or null. */
    String salt() {
        return salt;
    }

    /** The content identity of the per-request adapters, or null for the base model. */
    String adapters() {
        return adapters;
    }

    /** The adapters' names as KV cache events report them (vLLM's {@code lora_name}), or null. */
    String loraName() {
        return loraName;
    }

    /**
     * The adapter id the events publish beside the name (vLLM's {@code lora_id}, its {@code lora_int_id}), or null
     * for the base model: a stable positive id folded from the adapters' content identity, so two loads of one
     * content keep one id and a changed content gets a new one.
     */
    Integer loraId() {
        if (adapters == null) {
            return null;
        }
        long h = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < adapters.length(); i++) {
            h = mix(h ^ (adapters.charAt(i) * 0xC2B2AE3D27D4EB4FL));
        }
        int id = (int) (h ^ (h >>> 32)) & 0x7fffffff;
        return id == 0 ? 1 : id;
    }

    /** The 64-bit finalizer the id folds its input with. */
    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Equal keys share blocks: the same adapter identity and the same salt (the adapter names do not count). */
    @Override
    public boolean equals(Object o) {
        if (!(o instanceof PrefixKey)) {
            return false;
        }
        PrefixKey k = (PrefixKey) o;
        return Objects.equals(adapters, k.adapters) && Objects.equals(salt, k.salt);
    }

    @Override
    public int hashCode() {
        return Objects.hash(adapters, salt);
    }

    @Override
    public String toString() {
        return "PrefixKey{lora=" + loraName + ", salt=" + salt + "}";
    }
}
