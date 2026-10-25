package com.eonmux.neuralfree.io.safetensors;

import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Supplier;

/**
 * Writes safetensors files the way the reference {@code safetensors.serialize} does: tensors ordered by descending
 * dtype alignment ({@link StDType} declaration order) and then by name ({@link SafetensorsHeader#NAME_ORDER}), laid
 * out back to back from data offset 0, after a compact JSON header ({@code __metadata__} first, keys sorted) padded
 * with spaces to a multiple of 8 bytes. {@link #write(Path)} writes {@code <file>.tmp} beside the target and renames
 * it into place, so a reader never sees a partial file. Immutable; built by {@link Builder}.
 */
public final class SafetensorsWriter {

    /** Header alignment of the reference writer. */
    public static final int HEADER_ALIGNMENT = 8;
    /** Suffix of the temporary file that {@link #write(Path)} renames into place. */
    public static final String TEMP_SUFFIX = ".tmp";

    private static final Json.Style COMPACT = new Json.Style(false, null, ",", ":", false);
    private static final String BAD_DTYPE_NAME = "F12";
    private static final int TRAILING_BYTES = 3;

    /** Header and layout defects a test can ask {@link Builder} to write. */
    enum Malformation {
        /** The length field claims more header bytes than the file holds. */
        BAD_LENGTH,
        /** The second tensor's offsets start one byte inside the first. */
        OVERLAP,
        /** One unused byte precedes the last tensor. */
        GAP,
        /** Bytes follow the data section. */
        TRAILING_BYTES,
        /** The first tensor's entry appears twice in the header. */
        DUPLICATE_KEY,
        /** The first tensor's dtype is an unknown name. */
        BAD_DTYPE
    }

    /** One tensor to write: its layout entry and the supplier of its bytes. */
    private static final class Entry {
        final StTensor tensor;
        final Supplier<ByteBuffer> data;

        Entry(StTensor tensor, Supplier<ByteBuffer> data) {
            this.tensor = tensor;
            this.data = data;
        }
    }

    private final List<Entry> entries;
    private final List<StTensor> layout;
    private final Map<String, String> metadata;
    private final Malformation malformation;

    private SafetensorsWriter(List<Entry> entries, Map<String, String> metadata, Malformation malformation) {
        this.entries = Collections.unmodifiableList(entries);
        List<StTensor> l = new ArrayList<>(entries.size());
        for (Entry e : entries) {
            l.add(e.tensor);
        }
        this.layout = Collections.unmodifiableList(l);
        this.metadata = Collections.unmodifiableMap(metadata);
        this.malformation = malformation;
    }

    /** A new, empty builder. */
    public static Builder builder() {
        return new Builder();
    }

    /** The tensors in file order with their data offsets, as {@link #write(Path)} lays them out. */
    public List<StTensor> layout() {
        return layout;
    }

    /** The {@code __metadata__} entries, sorted by key (empty when none). */
    public Map<String, String> metadata() {
        return metadata;
    }

    /** Byte length of the data section. */
    public long dataLength() {
        return layout.isEmpty() ? 0 : layout.get(layout.size() - 1).end();
    }

    /**
     * Writes the file: {@code <path>.tmp} first (replacing a stale one), then an atomic rename onto {@code path}
     * (a plain replace where the file system has no atomic rename). On failure the temporary file is deleted and an
     * existing file at {@code path} is unchanged.
     *
     * @throws IllegalStateException when a byte supplier yields a buffer whose length differs from its tensor's
     * @throws IOException           when the file cannot be written or renamed
     */
    public void write(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        Path target = path.toAbsolutePath();
        Path tmp = target.resolveSibling(target.getFileName().toString() + TEMP_SUFFIX);
        try {
            try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING)) {
                writeContents(ch);
                ch.force(true);
            }
            try {
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException | RuntimeException e) {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException d) {
                e.addSuppressed(d);
            }
            throw e;
        }
    }

    /** The header text before padding. */
    String headerJson() {
        List<String> members = new ArrayList<>(entries.size() + 2);
        if (!metadata.isEmpty()) {
            members.add(member(SafetensorsHeader.METADATA_KEY, metadata));
        }
        for (int i = 0; i < layout.size(); i++) {
            StTensor t = layout.get(i);
            long begin = t.begin();
            if (malformation == Malformation.OVERLAP && i == 1) {
                begin -= 1;
            } else if (malformation == Malformation.GAP && i == layout.size() - 1) {
                begin += 1;
            }
            String dtype = malformation == Malformation.BAD_DTYPE && i == 0 ? BAD_DTYPE_NAME
                    : t.dtype().headerName();
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("dtype", dtype);
            info.put("shape", boxed(t.shape()));
            info.put("data_offsets", Arrays.asList(begin, begin + t.byteLength()));
            String m = member(t.name(), info);
            members.add(m);
            if (malformation == Malformation.DUPLICATE_KEY && i == 0) {
                members.add(m);
            }
        }
        return "{" + String.join(",", members) + "}";
    }

    private void writeContents(FileChannel ch) throws IOException {
        byte[] json = headerJson().getBytes(StandardCharsets.UTF_8);
        int n = (json.length + HEADER_ALIGNMENT - 1) / HEADER_ALIGNMENT * HEADER_ALIGNMENT;
        byte[] header = Arrays.copyOf(json, n);
        Arrays.fill(header, json.length, n, (byte) ' ');
        long total = dataLength() + (malformation == Malformation.GAP ? 1 : 0);
        long lengthField = malformation == Malformation.BAD_LENGTH ? n + total + 1 : n;
        ByteBuffer len = ByteBuffer.allocate(SafetensorsHeader.LENGTH_FIELD_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        len.putLong(0, lengthField);
        writeFully(ch, len);
        writeFully(ch, ByteBuffer.wrap(header));
        for (int i = 0; i < entries.size(); i++) {
            Entry e = entries.get(i);
            if (malformation == Malformation.GAP && i == entries.size() - 1) {
                writeFully(ch, ByteBuffer.allocate(1));
            }
            ByteBuffer data = Objects.requireNonNull(e.data.get(), "data of tensor '" + e.tensor.name() + "'");
            if (data.remaining() != e.tensor.byteLength()) {
                throw new IllegalStateException("tensor '" + e.tensor.name() + "': the data supplier yielded "
                        + data.remaining() + " bytes; " + Arrays.toString(e.tensor.shape()) + " of "
                        + e.tensor.dtype().headerName() + " needs " + e.tensor.byteLength());
            }
            writeFully(ch, data.duplicate());
        }
        if (malformation == Malformation.TRAILING_BYTES) {
            writeFully(ch, ByteBuffer.allocate(TRAILING_BYTES));
        }
    }

    private static String member(String key, Object value) {
        return Json.toString(key, COMPACT) + ":" + Json.toString(value, COMPACT);
    }

    private static List<Long> boxed(long[] values) {
        List<Long> out = new ArrayList<>(values.length);
        for (long v : values) {
            out.add(v);
        }
        return out;
    }

    private static void writeFully(FileChannel ch, ByteBuffer src) throws IOException {
        while (src.hasRemaining()) {
            ch.write(src);
        }
    }

    /** Collects tensors and metadata for a {@link SafetensorsWriter}. Not thread-safe. */
    public static final class Builder {

        private final List<StTensor> tensors = new ArrayList<>();
        private final List<Supplier<ByteBuffer>> suppliers = new ArrayList<>();
        private final Set<String> names = new HashSet<>();
        private final Map<String, String> metadata = new TreeMap<>(SafetensorsHeader.NAME_ORDER);
        private Malformation malformation;

        private Builder() {
        }

        /**
         * Adds a tensor whose bytes are {@code data} (copied).
         *
         * @throws IllegalArgumentException when the name is taken or reserved, the shape is invalid, or
         *                                  {@code data} does not hold exactly the shape's bytes
         */
        public Builder add(String name, StDType dtype, long[] shape, byte[] data) {
            Objects.requireNonNull(data, "data");
            StTensor t = check(name, dtype, shape);
            if (data.length != t.byteLength()) {
                throw new IllegalArgumentException("tensor '" + name + "': " + data.length + " bytes given; "
                        + Arrays.toString(shape) + " of " + dtype.headerName() + " needs " + t.byteLength());
            }
            byte[] copy = data.clone();
            return put(t, () -> ByteBuffer.wrap(copy).asReadOnlyBuffer());
        }

        /**
         * Adds a tensor whose bytes come from {@code data} when the file is written: each call must return a buffer
         * whose remaining bytes are exactly the tensor's (its position is not changed).
         *
         * @throws IllegalArgumentException when the name is taken or reserved or the shape is invalid
         */
        public Builder add(String name, StDType dtype, long[] shape, Supplier<ByteBuffer> data) {
            Objects.requireNonNull(data, "data");
            return put(check(name, dtype, shape), data);
        }

        /** Sets a {@code __metadata__} entry. */
        public Builder metadata(String key, String value) {
            metadata.put(Objects.requireNonNull(key, "key"), Objects.requireNonNull(value, "value"));
            return this;
        }

        /** Sets every entry of {@code entries} as {@code __metadata__}. */
        public Builder metadata(Map<String, String> entries) {
            for (Map.Entry<String, String> e : entries.entrySet()) {
                metadata(e.getKey(), e.getValue());
            }
            return this;
        }

        /** Makes the written file carry {@code m}, for tests of the reader's validation. */
        Builder malformed(Malformation m) {
            this.malformation = Objects.requireNonNull(m, "malformation");
            return this;
        }

        /**
         * The writer, with the tensors in reference order and their offsets assigned.
         *
         * @throws IllegalStateException when a requested malformation needs two tensors with data and fewer exist
         */
        public SafetensorsWriter build() {
            Integer[] order = new Integer[tensors.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> {
                StTensor ta = tensors.get(a);
                StTensor tb = tensors.get(b);
                int c = Integer.compare(tb.dtype().ordinal(), ta.dtype().ordinal());
                return c != 0 ? c : SafetensorsHeader.NAME_ORDER.compare(ta.name(), tb.name());
            });
            List<Entry> entries = new ArrayList<>(order.length);
            long offset = 0;
            for (int i : order) {
                StTensor t = tensors.get(i);
                entries.add(new Entry(new StTensor(t.name(), t.dtype(), t.shape(), offset, t.byteLength()),
                        suppliers.get(i)));
                offset = Math.addExact(offset, t.byteLength());
            }
            if ((malformation == Malformation.OVERLAP || malformation == Malformation.GAP)
                    && (entries.size() < 2 || entries.get(0).tensor.byteLength() == 0
                    || entries.get(entries.size() - 1).tensor.byteLength() == 0)) {
                throw new IllegalStateException(malformation + " needs at least two tensors, the first and last"
                        + " with data");
            }
            if ((malformation == Malformation.DUPLICATE_KEY || malformation == Malformation.BAD_DTYPE)
                    && entries.isEmpty()) {
                throw new IllegalStateException(malformation + " needs a tensor");
            }
            return new SafetensorsWriter(entries, new TreeMap<>(metadata), malformation);
        }

        private Builder put(StTensor t, Supplier<ByteBuffer> data) {
            if (!names.add(t.name())) {
                throw new IllegalArgumentException("tensor '" + t.name() + "' is added twice");
            }
            tensors.add(t);
            suppliers.add(data);
            return this;
        }

        private static StTensor check(String name, StDType dtype, long[] shape) {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(dtype, "dtype");
            Objects.requireNonNull(shape, "shape");
            if (name.equals(SafetensorsHeader.METADATA_KEY)) {
                throw new IllegalArgumentException("'" + name + "' is reserved for the header metadata");
            }
            if (!isEncodable(name)) {
                throw new IllegalArgumentException("tensor name '" + name + "' holds an unpaired surrogate and"
                        + " has no UTF-8 encoding");
            }
            if (shape.length > SafetensorsHeader.MAX_RANK) {
                throw new IllegalArgumentException("tensor '" + name + "': rank " + shape.length
                        + " exceeds the limit of " + SafetensorsHeader.MAX_RANK);
            }
            for (long d : shape) {
                if (d < 0 || d >= SafetensorsHeader.DIMENSION_LIMIT) {
                    throw new IllegalArgumentException("tensor '" + name + "': dimension " + d
                            + " of shape " + Arrays.toString(shape) + " is outside [0, 2^53)");
                }
            }
            long numel = StTensor.elementCount(name, shape);
            return new StTensor(name, dtype, shape, 0, StTensor.byteSize(name, dtype, numel));
        }

        private static boolean isEncodable(String s) {
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (Character.isHighSurrogate(c)) {
                    if (i + 1 >= s.length() || !Character.isLowSurrogate(s.charAt(i + 1))) {
                        return false;
                    }
                    i++;
                } else if (Character.isLowSurrogate(c)) {
                    return false;
                }
            }
            return true;
        }
    }
}
