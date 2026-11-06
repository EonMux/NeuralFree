package com.eonmux.neuralfree.gguf.edit;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * In-place changes of one scalar metadata value of a GGUF file, the operation of llama.cpp's
 * {@code gguf_set_metadata.py}: the stored key keeps its GGUF type and its byte width, so only the value bytes
 * change and every other byte of the file — the metadata around it, the tensor infos, the data section — stays
 * at its offset. The value is written little-endian, the order {@link GGUFFileParser} reads.
 *
 * <p>Only the scalar types a fixed-width write can represent are changed: the eight integer types, the two
 * float types and BOOL. STRING values change length and ARRAY values change count, so both are refused by name;
 * they are rewritten with {@code nftool new-metadata}. {@code general.alignment} is refused too, because the
 * tensor data offsets of the file are laid out to its value. A missing key is refused: this tool changes a value,
 * it never adds one.</p>
 *
 * <p>{@link #plan} validates the key, the type and the text without touching the file; {@link #apply} writes a
 * planned change and forces the bytes to disk.</p>
 */
public final class GgufMetadataEditor {

    private GgufMetadataEditor() {
    }

    /** One validated change of {@code key} from {@code oldValue} to {@code newValue}. Immutable. */
    public static final class Change {

        private final String key;
        private final GGUFValueType type;
        private final Object oldValue;
        private final Object newValue;
        private final long valueOffset;
        private final int width;

        private Change(String key, GGUFValueType type, Object oldValue, Object newValue, long valueOffset,
                       int width) {
            this.key = key;
            this.type = type;
            this.oldValue = oldValue;
            this.newValue = newValue;
            this.valueOffset = valueOffset;
            this.width = width;
        }

        /** The metadata key. */
        public String key() {
            return key;
        }

        /** The stored GGUF type, which the change keeps. */
        public GGUFValueType type() {
            return type;
        }

        /** The value the file holds now. */
        public Object oldValue() {
            return oldValue;
        }

        /** The value that will be stored. */
        public Object newValue() {
            return newValue;
        }

        /** File offset of the value's first byte. */
        public long valueOffset() {
            return valueOffset;
        }

        /** Bytes the stored value occupies, unchanged by the edit. */
        public int width() {
            return width;
        }

        /** True when the file already holds the requested value. */
        public boolean isNoOp() {
            return oldValue.equals(newValue);
        }

        /** The current value as the tool prints it (unsigned for the UINT types). */
        public String oldText() {
            return text(type, oldValue);
        }

        /** The requested value as the tool prints it (unsigned for the UINT types). */
        public String newText() {
            return text(type, newValue);
        }
    }

    /**
     * Plans the change of {@code key} to the textual value {@code text}, parsed as the key's stored type.
     *
     * @throws IOException naming the key when it is missing, is not a scalar the tool can change or holds a value
     *                     outside its type
     */
    public static Change plan(GGUFFileParser src, String key, String text) throws IOException {
        Objects.requireNonNull(src, "src");
        if (key == null || key.isEmpty()) {
            throw new IOException("set-metadata needs a metadata key");
        }
        GGUFValueType type = src.getValueType(key);
        if (type == null) {
            throw new IOException("metadata key '" + key + "' is not in '" + src.getFilePath() + "'");
        }
        if (GgufKeys.GENERAL_ALIGNMENT.equals(key)) {
            throw new IOException("metadata key '" + key + "' lays out the tensor data; rewriting it in place "
                    + "would invalidate every tensor offset, so use nftool new-metadata to re-align the file");
        }
        int width = widthOf(type);
        if (width == 0) {
            throw new IOException("set-metadata changes simple values only; metadata key '" + key
                    + "' has GGUF type " + type + ", which an in-place write cannot represent; use "
                    + "nftool new-metadata");
        }
        Long keyOffset = src.getKeyOffset(key);
        if (keyOffset == null) {
            throw new IOException("metadata key '" + key + "' has no recorded offset in '" + src.getFilePath()
                    + "'");
        }
        long valueOffset = keyOffset + 8L + key.getBytes(StandardCharsets.UTF_8).length + 4L;
        Object oldValue = src.getMetadata().get(key);
        Object newValue = parse(type, text, key);
        return new Change(key, type, oldValue, newValue, valueOffset, width);
    }

    /**
     * Writes {@code change}'s value at its recorded offset and forces the file's bytes to disk.
     *
     * @throws IOException when the file cannot be reopened or written
     */
    public static void apply(GGUFFileParser src, Change change) throws IOException {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(change, "change");
        byte[] bytes = encode(change.type, change.newValue);
        try (RandomAccessFile file = new RandomAccessFile(src.getFilePath(), "rw")) {
            file.seek(change.valueOffset);
            file.write(bytes);
            file.getFD().sync();
        }
    }

    /** Bytes per value of a scalar type, or 0 for a type an in-place write cannot change. */
    private static int widthOf(GGUFValueType type) {
        switch (type) {
            case UINT8:
            case INT8:
            case BOOL:
                return 1;
            case UINT16:
            case INT16:
                return 2;
            case UINT32:
            case INT32:
            case FLOAT32:
                return 4;
            case UINT64:
            case INT64:
            case FLOAT64:
                return 8;
            default:
                return 0;
        }
    }

    private static Object parse(GGUFValueType type, String text, String key) throws IOException {
        if (text == null) {
            throw new IOException("metadata key '" + key + "' needs a value");
        }
        switch (type) {
            case BOOL: {
                String v = text.trim().toLowerCase(java.util.Locale.ROOT);
                if (!v.equals("true") && !v.equals("false")) {
                    throw bad(key, type, text, "true or false");
                }
                return Boolean.valueOf(v);
            }
            case FLOAT32: {
                float v;
                try {
                    v = Float.parseFloat(text.trim());
                } catch (NumberFormatException e) {
                    throw bad(key, type, text, "a decimal number");
                }
                if (!Float.isFinite(v)) {
                    throw bad(key, type, text, "a finite number");
                }
                return Float.valueOf(v);
            }
            case FLOAT64: {
                double v;
                try {
                    v = Double.parseDouble(text.trim());
                } catch (NumberFormatException e) {
                    throw bad(key, type, text, "a decimal number");
                }
                if (!Double.isFinite(v)) {
                    throw bad(key, type, text, "a finite number");
                }
                return Double.valueOf(v);
            }
            case UINT8:
                return Integer.valueOf((int) unsigned(text, key, type, 0xFFL));
            case INT8:
                return Integer.valueOf((int) signed(text, key, type, Byte.MIN_VALUE, Byte.MAX_VALUE));
            case UINT16:
                return Integer.valueOf((int) unsigned(text, key, type, 0xFFFFL));
            case INT16:
                return Integer.valueOf((int) signed(text, key, type, Short.MIN_VALUE, Short.MAX_VALUE));
            case UINT32: {
                long v = unsigned(text, key, type, 0xFFFF_FFFFL);
                return Integer.valueOf((int) v);
            }
            case INT32:
                return Integer.valueOf((int) signed(text, key, type, Integer.MIN_VALUE, Integer.MAX_VALUE));
            case UINT64: {
                try {
                    return Long.valueOf(Long.parseUnsignedLong(text.trim()));
                } catch (NumberFormatException e) {
                    throw bad(key, type, text, "an unsigned 64-bit integer");
                }
            }
            case INT64: {
                try {
                    return Long.valueOf(Long.parseLong(text.trim()));
                } catch (NumberFormatException e) {
                    throw bad(key, type, text, "a signed 64-bit integer");
                }
            }
            default:
                throw new IOException("metadata key '" + key + "' has GGUF type " + type
                        + ", which set-metadata cannot change");
        }
    }

    private static long unsigned(String text, String key, GGUFValueType type, long max) throws IOException {
        long v;
        try {
            v = Long.parseLong(text.trim());
        } catch (NumberFormatException e) {
            throw bad(key, type, text, "an integer in [0, " + max + "]");
        }
        if (v < 0 || v > max) {
            throw new IOException("metadata key '" + key + "' value " + Long.toUnsignedString(v)
                    + " of type " + type + " is outside [0, " + max + "]");
        }
        return v;
    }

    private static long signed(String text, String key, GGUFValueType type, long min, long max) throws IOException {
        long v;
        try {
            v = Long.parseLong(text.trim());
        } catch (NumberFormatException e) {
            throw bad(key, type, text, "an integer in [" + min + ", " + max + "]");
        }
        if (v < min || v > max) {
            throw new IOException("metadata key '" + key + "' value " + v + " of type " + type
                    + " is outside [" + min + ", " + max + "]");
        }
        return v;
    }

    private static IOException bad(String key, GGUFValueType type, String text, String expected) {
        return new IOException("metadata key '" + key + "' of type " + type + " needs " + expected + ", not '"
                + text + "'");
    }

    private static byte[] encode(GGUFValueType type, Object value) {
        ByteBuffer b = ByteBuffer.allocate(widthOf(type)).order(ByteOrder.LITTLE_ENDIAN);
        switch (type) {
            case UINT8:
            case INT8:
                b.put((byte) ((Integer) value).intValue());
                break;
            case BOOL:
                b.put((byte) (((Boolean) value).booleanValue() ? 1 : 0));
                break;
            case UINT16:
            case INT16:
                b.putShort((short) ((Integer) value).intValue());
                break;
            case UINT32:
            case INT32:
                b.putInt(((Integer) value).intValue());
                break;
            case FLOAT32:
                b.putFloat(((Float) value).floatValue());
                break;
            case UINT64:
            case INT64:
                b.putLong(((Long) value).longValue());
                break;
            case FLOAT64:
                b.putDouble(((Double) value).doubleValue());
                break;
            default:
                throw new IllegalStateException(type + " has no fixed-width encoding");
        }
        return b.array();
    }

    /** The value as the tool prints it, unsigned for the UINT types. */
    private static String text(GGUFValueType type, Object value) {
        switch (type) {
            case UINT8:
            case UINT16:
                return Integer.toString(((Integer) value).intValue());
            case UINT32:
                return Integer.toUnsignedString(((Integer) value).intValue());
            case UINT64:
                return Long.toUnsignedString(((Long) value).longValue());
            default:
                return String.valueOf(value);
        }
    }
}
