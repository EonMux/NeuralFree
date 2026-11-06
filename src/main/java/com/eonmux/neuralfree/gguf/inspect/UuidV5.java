package com.eonmux.neuralfree.gguf.inspect;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;

/**
 * UUID version 5 (name-based, SHA-1) values, as llama.cpp's {@code gguf-hash --uuid} derives them: the SHA-1
 * digest of a namespace UUID's big-endian bytes followed by the name, whose first 16 bytes become the UUID with
 * version 5 and the RFC 4122 variant set (RFC 9562 section 5.5). {@link #NAMESPACE_LLAMA_CPP} is llama.cpp's
 * namespace, the UUIDv5 of {@code uuid.NAMESPACE_URL} and {@code en.wikipedia.org/wiki/Llama.cpp}.
 */
public final class UuidV5 {

    /** llama.cpp's UUID namespace ({@code ef001206-dadc-5f6d-a15f-3359e577d4e5}). */
    public static final UUID NAMESPACE_LLAMA_CPP = UUID.fromString("ef001206-dadc-5f6d-a15f-3359e577d4e5");

    private UuidV5() {
    }

    /** The 16 big-endian bytes of {@code namespace}, the form the hash input starts with. */
    public static byte[] namespaceBytes(UUID namespace) {
        byte[] bytes = new byte[16];
        long msb = namespace.getMostSignificantBits();
        long lsb = namespace.getLeastSignificantBits();
        for (int i = 0; i < 8; i++) {
            bytes[i] = (byte) (msb >>> (8 * (7 - i)));
            bytes[8 + i] = (byte) (lsb >>> (8 * (7 - i)));
        }
        return bytes;
    }

    /** The UUIDv5 of {@code name} in {@code namespace}. */
    public static UUID of(UUID namespace, byte[] name) {
        MessageDigest sha1 = sha1();
        sha1.update(namespaceBytes(namespace));
        return fromDigest(sha1.digest(name));
    }

    /** The UUIDv5 of the UTF-8 bytes of {@code name} in {@code namespace}. */
    public static UUID of(UUID namespace, String name) {
        return of(namespace, name.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * The UUID of a 20-byte SHA-1 digest of {@code namespace || name}: its first 16 bytes with the version 5
     * nibble and the RFC 4122 variant set.
     *
     * @throws IllegalArgumentException when the digest is shorter than 16 bytes
     */
    public static UUID fromDigest(byte[] digest) {
        if (digest == null || digest.length < 16) {
            throw new IllegalArgumentException("a UUIDv5 needs a SHA-1 digest of at least 16 bytes");
        }
        long msb = 0;
        long lsb = 0;
        for (int i = 0; i < 8; i++) {
            msb = (msb << 8) | (digest[i] & 0xFFL);
            lsb = (lsb << 8) | (digest[8 + i] & 0xFFL);
        }
        msb &= ~0xF000L;              // clear the four version bits
        msb |= 0x5000L;               // version 5
        lsb &= ~(0xC000L << 48);      // clear the two variant bits
        lsb |= 0x8000L << 48;         // RFC 4122 variant
        return new UUID(msb, lsb);
    }

    /** A fresh SHA-1 digest; every JVM provides it. */
    public static MessageDigest sha1() {
        try {
            return MessageDigest.getInstance("SHA-1");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("this JVM has no SHA-1 MessageDigest", e);
        }
    }

    /** A fresh SHA-256 digest; every JVM provides it. */
    public static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("this JVM has no SHA-256 MessageDigest", e);
        }
    }

    /** Lower-case hex of a digest, the form the tools print. */
    public static String hex(byte[] digest) {
        StringBuilder s = new StringBuilder(digest.length * 2);
        for (byte b : digest) {
            s.append(Character.forDigit((b >>> 4) & 0xF, 16));
            s.append(Character.forDigit(b & 0xF, 16));
        }
        return s.toString();
    }
}
