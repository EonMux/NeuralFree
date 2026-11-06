package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.Locale;
import java.util.Objects;

/**
 * The hash algorithm of the prefix cache's block hashes, vLLM's {@code PrefixCachingHashAlgo} and its
 * {@code --prefix-caching-hash-algo} default: {@code sha256} over Python's pickle serialization (vLLM's default),
 * {@code sha256_cbor} over canonical CBOR (the cross-language form), {@code xxhash} and {@code xxhash_cbor} the
 * same two serializations under XXH3's 128-bit digest. The pickle forms exist only to make the values equal vLLM's;
 * the CBOR forms a router can recompute outside Python. Immutable.
 */
public enum PrefixHashAlgo {
    /** SHA-256 over Python's pickle serialization (vLLM's default). */
    SHA256("sha256", 32),
    /** SHA-256 over canonical CBOR. */
    SHA256_CBOR("sha256_cbor", 32),
    /** XXH3's 128-bit digest over Python's pickle serialization. */
    XXHASH("xxhash", 16),
    /** XXH3's 128-bit digest over canonical CBOR. */
    XXHASH_CBOR("xxhash_cbor", 16);

    /** The name vLLM's {@code --prefix-caching-hash-algo} takes. */
    private final String name;
    /** The digest's byte length (the bytes form of a published block hash). */
    private final int digestSize;

    PrefixHashAlgo(String name, int digestSize) {
        this.name = name;
        this.digestSize = digestSize;
    }

    /**
     * The algorithm {@code value} names, case-insensitively ({@code setting} names it in the refusal).
     *
     * @throws IllegalArgumentException for any other value
     */
    public static PrefixHashAlgo parse(String value, String setting) {
        Objects.requireNonNull(value, "value");
        String name = value.trim().toLowerCase(Locale.ROOT);
        for (PrefixHashAlgo algo : values()) {
            if (algo.name.equals(name)) {
                return algo;
            }
        }
        throw new IllegalArgumentException(setting + "=" + value + " is invalid; use one of"
                + " sha256, sha256_cbor, xxhash, xxhash_cbor");
    }

    /** The name vLLM's flag takes. */
    public String algoName() {
        return name;
    }

    /** The digest's byte length (32 for the SHA-256 forms, 16 for the XXH3 forms). */
    public int digestSize() {
        return digestSize;
    }

    /**
     * Whether collisions are computationally infeasible, vLLM's crypto set: the SHA-256 forms are, the XXH3 forms
     * are not, which is why the latter derive their NONE_HASH seed from {@code PYTHONHASHSEED} or fresh randomness.
     */
    public boolean cryptographic() {
        return this == SHA256 || this == SHA256_CBOR;
    }

    /** Whether the algorithm serializes through Python's pickle ({@code sha256}, {@code xxhash}) or canonical CBOR. */
    boolean pickleSerialization() {
        return this == SHA256 || this == XXHASH;
    }

    /** Hashes {@code input} per the algorithm: SHA-256 (JCA) or {@link Xxh3_128} over the serialization. */
    byte[] digest(byte[] input) {
        if (!cryptographic()) {
            return Xxh3_128.digest(input);
        }
        try {
            return java.security.MessageDigest.getInstance("SHA-256").digest(input);
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("no SHA-256 MessageDigest", e);
        }
    }

    @Override
    public String toString() {
        return name;
    }
}
