package com.eonmux.neuralfree.gguf.engine.kvtransfer;

/**
 * The role an instance plays in disaggregated serving (vLLM's {@code kv_role}): a prefill producer computes prompts
 * and exports their KV, a decode consumer imports it and decodes, and {@link #BOTH} may do either.
 */
public enum KvRole {

    /** Prefills prompts and exports their KV cache (vLLM's {@code kv_producer}). */
    PRODUCER("kv_producer"),

    /** Decodes from an imported KV cache (vLLM's {@code kv_consumer}). */
    CONSUMER("kv_consumer"),

    /** Both directions in one instance (vLLM's {@code kv_both}). */
    BOTH("kv_both");

    private final String wireName;

    KvRole(String wireName) {
        this.wireName = wireName;
    }

    /** The vLLM spelling of the role. */
    public String wireName() {
        return wireName;
    }

    /** Whether an instance of this role exports a prefill's KV. */
    public boolean produces() {
        return this != CONSUMER;
    }

    /** Whether an instance of this role imports a prefill's KV. */
    public boolean consumes() {
        return this != PRODUCER;
    }

    /**
     * The role {@code value} names ({@code kv_producer}, {@code kv_consumer} or {@code kv_both}).
     *
     * @throws IllegalArgumentException naming the value and the accepted spellings
     */
    public static KvRole parse(String value) {
        if (value != null) {
            for (KvRole role : values()) {
                if (role.wireName.equals(value)) {
                    return role;
                }
            }
        }
        throw new IllegalArgumentException("unsupported kv_role: " + value + " (supported roles are "
                + PRODUCER.wireName + ", " + CONSUMER.wireName + " and " + BOTH.wireName + ")");
    }
}
