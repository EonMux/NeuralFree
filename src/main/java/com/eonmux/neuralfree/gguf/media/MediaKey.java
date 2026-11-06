package com.eonmux.neuralfree.gguf.media;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.Objects;

/**
 * The identity of one media item, vLLM's {@code mm_hash} ({@code ProcessorInputs.get_mm_hashes}): the one key the
 * encoded-media cache stores the item's encoding under and the prefix cache keys the item's KV rows by, so an item
 * has one identity across both caches.
 *
 * <ul>
 *   <li>{@link #ofUuid}: an item a client names with a {@code uuid} (vLLM's {@code multi_modal_uuids}); its
 *       {@link #id()} is the uuid itself, or, when the modality's loader options are hash factors
 *       ({@code --media-io-kwargs}), a SHA-256 digest of the uuid and those factors, as vLLM hashes a uuid with
 *       the modality's {@code media_io_kwargs}.</li>
 *   <li>{@link #ofContent}: an item named by its bytes; its id is the SHA-256 digest of the bytes, or of that digest
 *       and the hash factors when there are some.</li>
 *   <li>{@link #ofRows}: precomputed embedding rows ({@code image_embeds}, {@code audio_embeds}) as one item; its id
 *       is the SHA-256 digest of the row count, the width and every row's little-endian float32 bytes.</li>
 * </ul>
 *
 * <p>Two keys are equal when modality, source and id agree: a uuid never names the same item as a content digest,
 * and one uuid names separate items per modality. Immutable.</p>
 */
public final class MediaKey {

    /** What names the item. */
    public enum Source {
        /** The client's {@code uuid}. */
        UUID,
        /** The SHA-256 digest of the item's bytes. */
        CONTENT,
        /** The SHA-256 digest of precomputed embedding rows. */
        ROWS
    }

    /** The modality text of a rows key, whose rows carry no modality of their own. */
    private static final String EMBEDDINGS = "embeddings";

    private final String modality;
    private final Source source;
    private final String id;
    private final long bits;

    private MediaKey(String modality, Source source, String id) {
        this.modality = modality;
        this.source = source;
        this.id = id;
        MessageDigest d = sha256();
        d.update((source.name() + '\0' + modality + '\0' + id).getBytes(StandardCharsets.UTF_8));
        this.bits = ByteBuffer.wrap(d.digest()).getLong();
    }

    /**
     * The key of an item the client names {@code uuid}.
     *
     * @param modality    the item's modality
     * @param uuid        the client's uuid, non-blank
     * @param hashFactors the modality's loader options in canonical form, or null or empty for none
     * @throws IllegalArgumentException when the uuid is blank
     */
    public static MediaKey ofUuid(MediaLimits.Modality modality, String uuid, String hashFactors) {
        Objects.requireNonNull(modality, "modality");
        Objects.requireNonNull(uuid, "uuid");
        if (uuid.trim().isEmpty()) {
            throw new IllegalArgumentException("a media uuid must not be blank");
        }
        String id = isEmpty(hashFactors) ? uuid : factored(modality.key(), "uuid", uuid, hashFactors);
        return new MediaKey(modality.key(), Source.UUID, id);
    }

    /**
     * The key of an item named by its {@code bytes}.
     *
     * @param modality    the item's modality
     * @param bytes       the item's encoded bytes (an image or audio file)
     * @param hashFactors the modality's loader options in canonical form, or null or empty for none
     */
    public static MediaKey ofContent(MediaLimits.Modality modality, byte[] bytes, String hashFactors) {
        Objects.requireNonNull(modality, "modality");
        Objects.requireNonNull(bytes, "bytes");
        String digest = hex(sha256().digest(bytes));
        String id = isEmpty(hashFactors) ? digest : factored(modality.key(), "sha256", digest, hashFactors);
        return new MediaKey(modality.key(), Source.CONTENT, id);
    }

    /**
     * The key of precomputed embedding rows taken as one item.
     *
     * @throws IllegalArgumentException when there are no rows, or the rows are empty or differ in width
     */
    public static MediaKey ofRows(float[][] rows) {
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0 || rows[0] == null || rows[0].length == 0) {
            throw new IllegalArgumentException("an embedding item holds at least one non-empty row");
        }
        int width = rows[0].length;
        MessageDigest d = sha256();
        d.update(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(rows.length).putInt(width).array());
        for (int r = 0; r < rows.length; r++) {
            if (rows[r] == null || rows[r].length != width) {
                throw new IllegalArgumentException("embedding rows share one width: row " + r + " differs from "
                        + width);
            }
        }
        feedRows(d, rows, 0, rows.length);
        return new MediaKey(EMBEDDINGS, Source.ROWS, hex(d.digest()));
    }

    /** The digest of a source's name or digest under the modality's hash factors. */
    private static String factored(String modality, String kind, String value, String hashFactors) {
        MessageDigest d = sha256();
        d.update((modality + '\0' + kind + '\0' + value + '\0' + hashFactors).getBytes(StandardCharsets.UTF_8));
        return hex(d.digest());
    }

    private static boolean isEmpty(String s) {
        return s == null || s.isEmpty();
    }

    /**
     * The key a {@link #toString()} handle names: {@code modality:source:id}, such as
     * {@code image:content:<hex>} or {@code image:uuid:cat-1}. The id may itself contain ':'; everything after
     * the second separator is the id.
     *
     * @throws IllegalArgumentException when the handle is malformed or names an unknown source
     */
    public static MediaKey parse(String handle) {
        Objects.requireNonNull(handle, "handle");
        String[] parts = handle.split(":", 3);
        if (parts.length != 3 || parts[0].isEmpty() || parts[2].isEmpty()) {
            throw new IllegalArgumentException("a media key handle is 'modality:source:id', got '" + handle + "'");
        }
        Source source;
        switch (parts[1]) {
            case "uuid":
                source = Source.UUID;
                break;
            case "content":
                source = Source.CONTENT;
                break;
            case "rows":
                source = Source.ROWS;
                break;
            default:
                throw new IllegalArgumentException("a media key handle names the source uuid, content or rows, got '"
                        + parts[1] + "' in '" + handle + "'");
        }
        return new MediaKey(parts[0], source, parts[2]);
    }

    /** The item's modality ({@code image}, {@code audio}), or {@code embeddings} for a rows key. */
    public String modality() {
        return modality;
    }

    /** What names the item. */
    public Source source() {
        return source;
    }

    /** vLLM's {@code mm_hash} of the item: the client's uuid, or a lowercase hex SHA-256 digest. */
    public String id() {
        return id;
    }

    /** 64 bits of the key for hash chains: the leading bits of a SHA-256 digest over modality, source and id. */
    public long bits() {
        return bits;
    }

    /**
     * Feeds rows {@code [from, to)} of {@code rows} to {@code digest} as little-endian IEEE-754 binary32 bytes, row
     * after row (the byte image vLLM hashes embedding tensors and {@code prompt_embeds} rows over).
     *
     * @throws IndexOutOfBoundsException when the range is outside the rows
     */
    public static void feedRows(MessageDigest digest, float[][] rows, int from, int to) {
        Objects.requireNonNull(digest, "digest");
        Objects.checkFromToIndex(from, to, rows.length);
        if (from == to) {
            return;
        }
        ByteBuffer buf = ByteBuffer.allocate(rows[from].length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = from; r < to; r++) {
            if (buf.capacity() != rows[r].length * Float.BYTES) {
                buf = ByteBuffer.allocate(rows[r].length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            }
            buf.clear();
            for (float v : rows[r]) {
                buf.putFloat(v);
            }
            digest.update(buf.array(), 0, buf.position());
        }
    }

    /** A fresh SHA-256 digest (every Java platform provides the algorithm). */
    public static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("the Java platform lacks SHA-256", e);
        }
    }

    /** Lowercase hex digits of {@code bytes}. */
    public static String hex(byte[] bytes) {
        char[] out = new char[bytes.length * 2];
        for (int i = 0; i < bytes.length; i++) {
            out[2 * i] = Character.forDigit((bytes[i] >> 4) & 0xF, 16);
            out[2 * i + 1] = Character.forDigit(bytes[i] & 0xF, 16);
        }
        return new String(out);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof MediaKey)) {
            return false;
        }
        MediaKey k = (MediaKey) o;
        return source == k.source && modality.equals(k.modality) && id.equals(k.id);
    }

    @Override
    public int hashCode() {
        return Long.hashCode(bits);
    }

    /** {@code modality:source:id}, such as {@code image:uuid:cat-1}. */
    @Override
    public String toString() {
        return modality + ':' + source.name().toLowerCase(Locale.ROOT) + ':' + id;
    }
}
