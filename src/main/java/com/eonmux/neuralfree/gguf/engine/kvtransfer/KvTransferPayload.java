package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.util.Arrays;
import java.util.Objects;

/**
 * One transfer: the prefill instance's engine id, the prompt tokens whose KV the chain holds, and the chain itself in
 * the slot file format ({@code NFSL}: the block layout identity, the chain's extra keys and, per block, its rows'
 * tokens, positions and stored K/V bytes). A consumer validates the tokens against its own prompt and imports the
 * chain through the memory's block layout, which checks the layout identity before any row reaches the cache.
 *
 * <p>Immutable; the byte and token arrays are not copied (a chain is as large as the prompt's KV cache), so callers
 * hand over arrays they no longer mutate and read the returned ones only.</p>
 */
public final class KvTransferPayload {

    private final String transferId;
    private final String engineId;
    private final int[] promptTokens;
    private final byte[] chain;

    /**
     * A payload; the arrays are taken over, not copied.
     *
     * @throws IllegalArgumentException when an id is blank, the chain is empty or the tokens are empty
     */
    public KvTransferPayload(String transferId, String engineId, int[] promptTokens, byte[] chain) {
        this.transferId = requireText(transferId, "transfer id");
        this.engineId = requireText(engineId, "engine id");
        this.promptTokens = Objects.requireNonNull(promptTokens, "promptTokens");
        this.chain = Objects.requireNonNull(chain, "chain");
        if (promptTokens.length == 0) {
            throw new IllegalArgumentException("a KV transfer moves at least one prompt token");
        }
        if (chain.length == 0) {
            throw new IllegalArgumentException("a KV transfer moves a non-empty chain");
        }
        for (int i = 0; i < promptTokens.length; i++) {
            if (promptTokens[i] < 0) {
                throw new IllegalArgumentException("prompt token " + i + " is " + promptTokens[i]);
            }
        }
    }

    private static String requireText(String value, String what) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("a KV transfer needs a " + what);
        }
        return value;
    }

    /** The id both instances name the transfer by. */
    public String transferId() {
        return transferId;
    }

    /** The exporting instance's engine id. */
    public String engineId() {
        return engineId;
    }

    /** The prompt tokens the chain holds (the live array; callers only read it). */
    public int[] promptTokens() {
        return promptTokens;
    }

    /** The chain bytes in the slot file format (the live array; callers only read it). */
    public byte[] chain() {
        return chain;
    }

    /** Bytes the payload occupies beside its ids. */
    public long bytes() {
        return (long) chain.length + 4L * promptTokens.length;
    }

    /**
     * Whether {@code prompt} starts with this payload's tokens: the decode instance's prompt must hold the prefill's,
     * else its KV rows would describe other positions.
     */
    public boolean covers(int[] prompt) {
        if (prompt == null || prompt.length < promptTokens.length) {
            return false;
        }
        for (int i = 0; i < promptTokens.length; i++) {
            if (prompt[i] != promptTokens[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return "KvTransferPayload{" + transferId + " from " + engineId + ", " + promptTokens.length + " tokens, "
                + chain.length + " bytes}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KvTransferPayload)) {
            return false;
        }
        KvTransferPayload other = (KvTransferPayload) o;
        return transferId.equals(other.transferId) && engineId.equals(other.engineId)
                && Arrays.equals(promptTokens, other.promptTokens) && Arrays.equals(chain, other.chain);
    }

    @Override
    public int hashCode() {
        return 31 * Objects.hash(transferId, engineId) + 31 * Arrays.hashCode(promptTokens)
                + Arrays.hashCode(chain);
    }
}
