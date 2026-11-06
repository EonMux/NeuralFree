package com.eonmux.neuralfree.gguf.edit;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.io.Closeable;
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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Converts a GGUF file between little- and big-endian byte order, the operation of llama.cpp's
 * {@code gguf_convert_endian.py}: every multi-byte field of the header, the key/value store and the tensor infos
 * is written in the target order, and every multi-byte field inside a tensor block is byte-swapped. Strings,
 * single-byte fields and the zero padding keep their bytes, so a converted file differs only in the byte order of
 * its numbers.
 *
 * <p>GGUF files are little-endian as ggml writes them ({@link com.eonmux.neuralfree.gguf.GGUFFileParser} reads
 * only that order); a big-endian file is the format other consumers exchange. The file's order is detected from
 * its version field, whose two readings cannot both be a supported version; the layout of versions 2 and 3 is the
 * same and both convert. The tensor types with a defined block layout in {@link BlockLayout} convert; a type
 * NeuralFree only defines little-endian (the extension range) and an unknown type are refused by tensor name.</p>
 *
 * <p>{@link #inspect} reads and validates a file without writing anything. {@link #convert} writes the converted
 * file to a temporary twin and renames it over its target only after every byte went out, so a failed conversion
 * leaves the previous target intact. For an in-place conversion the input is closed before the rename.</p>
 */
public final class GgufEndianConverter {

    /** Byte order of a GGUF file. */
    public enum Endian {
        LITTLE(ByteOrder.LITTLE_ENDIAN),
        BIG(ByteOrder.BIG_ENDIAN);

        private final ByteOrder order;

        Endian(ByteOrder order) {
            this.order = order;
        }

        /** The {@link ByteOrder} of this endianness. */
        public ByteOrder order() {
            return order;
        }

        /** The order of the JVM's host. */
        public static Endian host() {
            return ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN ? LITTLE : BIG;
        }

        /** {@code little}, {@code big} or {@code native}, ignoring case, or null for any other text. */
        public static Endian named(String text) {
            if (text == null) {
                return null;
            }
            String t = text.trim().toLowerCase(Locale.ROOT);
            if (t.equals("little") || t.equals("le")) {
                return LITTLE;
            }
            if (t.equals("big") || t.equals("be")) {
                return BIG;
            }
            return t.equals("native") ? host() : null;
        }
    }

    /** The facts of a GGUF file the converter reads. Immutable. */
    public static final class Info {

        private final Path file;
        private final Endian fileEndian;
        private final long version;
        private final long tensorCount;
        private final long kvCount;
        private final long alignment;
        private final long dataOffset;
        private final long dataLength;

        Info(Path file, Endian fileEndian, long version, long tensorCount, long kvCount, long alignment,
             long dataOffset, long dataLength) {
            this.file = file;
            this.fileEndian = fileEndian;
            this.version = version;
            this.tensorCount = tensorCount;
            this.kvCount = kvCount;
            this.alignment = alignment;
            this.dataOffset = dataOffset;
            this.dataLength = dataLength;
        }

        public Path file() {
            return file;
        }

        /** The order the file is stored in. */
        public Endian fileEndian() {
            return fileEndian;
        }

        public long version() {
            return version;
        }

        public long tensorCount() {
            return tensorCount;
        }

        public long kvCount() {
            return kvCount;
        }

        /** The data alignment ({@code general.alignment}, 32 when the key is absent). */
        public long alignment() {
            return alignment;
        }

        /** File offset of the data section. */
        public long dataOffset() {
            return dataOffset;
        }

        /** Bytes from the data section to the end of the file. */
        public long dataLength() {
            return dataLength;
        }
    }

    /** Newest GGUF version the converter understands. */
    static final int GGUF_VERSION = 3;

    private static final long MAX_STRING_LENGTH = 1L << 30;
    private static final long MAX_ARRAY_ELEMENTS = 1L << 30;
    private static final long MAX_ENTRIES = Integer.MAX_VALUE - 8L;
    private static final long MIN_KV_BYTES = 8 + 4 + 1;
    private static final long MIN_TENSOR_INFO_BYTES = 8 + 4 + 4 + 8;
    private static final long DEFAULT_ALIGNMENT = 32;
    private static final int CHUNK_BYTES = 1 << 20;
    private static final int BUFFER_BYTES = 1 << 16;

    private GgufEndianConverter() {
    }

    /**
     * Reads and validates the file's header without writing anything.
     *
     * @throws IOException naming the file and the malformed field
     */
    public static Info inspect(Path file) throws IOException {
        try (Reader r = new Reader(file)) {
            Layout layout = parse(r, Writer.counting(r.order));
            return layout.info(file);
        }
    }

    /**
     * Checks that every tensor of the file has a byte-swap layout, without writing anything (the check a dry run
     * makes before a conversion).
     *
     * @throws IOException naming the tensor whose type cannot be converted
     */
    public static Info checkConvertible(Path file) throws IOException {
        try (Reader r = new Reader(file)) {
            Layout layout = parse(r, Writer.counting(r.order));
            swapPlans(layout.tensors);
            return layout.info(file);
        }
    }

    /**
     * Writes {@code input} to {@code output} in {@code target} order, byte-swapping every multi-byte field.
     * {@code output} may be {@code input} (an in-place conversion, which always replaces it); a different
     * existing output is replaced only when {@code overwrite} is set. When the file already has the target order
     * nothing is written and its {@link Info} is returned.
     *
     * @throws IOException naming the file, the tensor or the type that cannot be converted
     */
    public static Info convert(Path input, Endian target, Path output, boolean overwrite) throws IOException {
        if (target == null) {
            throw new IllegalArgumentException("target endianness is null");
        }
        if (output == null) {
            throw new IllegalArgumentException("output path is null");
        }
        boolean inPlace = output.toAbsolutePath().normalize().equals(input.toAbsolutePath().normalize());
        if (!inPlace && Files.exists(output) && !overwrite) {
            throw new IOException("file '" + output + "' already exists; pass --overwrite to replace it");
        }
        Endian current = readEndian(input);
        if (current == target) {
            return inspect(input);
        }
        Layout layout;
        Path tmp = output.resolveSibling(output.getFileName().toString() + ".tmp");
        try {
            try (Reader r = new Reader(input); Writer w = new Writer(tmp, target)) {
                layout = parse(r, w);
                if (layout.fileEndian != current) {
                    throw new IOException("file '" + input + "' changed endianness while it was read");
                }
                swapPlans(layout.tensors);
                w.zeros(layout.dataOffset - w.position());
                copyData(r, w, layout);
            }
            move(tmp, output);
        } catch (IOException | RuntimeException e) {
            try {
                Files.deleteIfExists(tmp);
            } catch (IOException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        layout.fileEndian = target;
        return layout.info(output);
    }

    private static void move(Path tmp, Path output) throws IOException {
        try {
            Files.move(tmp, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tmp, output, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    // ------------------------------------------------------------------ layout

    /** One tensor of the file: its table entry and the size of its stored blocks. */
    private static final class TensorMeta {
        final String name;
        final long[] dims;
        final long offset;
        GGMLType type;
        long size;

        TensorMeta(String name, long[] dims, GGMLType type, long offset) {
            this.name = name;
            this.dims = dims;
            this.type = type;
            this.offset = offset;
        }

        GGUFTensorInfo info() {
            return new GGUFTensorInfo(name, dims, type, offset);
        }
    }

    /** The parsed header and tensor table of one file. */
    private static final class Layout {
        Endian fileEndian;
        long version;
        long tensorCount;
        long kvCount;
        long alignment = DEFAULT_ALIGNMENT;
        long dataOffset;
        long dataLength;
        final List<TensorMeta> tensors = new ArrayList<>();

        Info info(Path file) {
            return new Info(file, fileEndian, version, tensorCount, kvCount, alignment, dataOffset, dataLength);
        }
    }

    /** The endianness of {@code file}, from its version field. */
    private static Endian readEndian(Path file) throws IOException {
        try (Reader r = new Reader(file)) {
            return r.order;
        }
    }

    private static Endian detectEndian(byte[] version, Path file) throws IOException {
        long little = decodeU32(version, 0, Endian.LITTLE);
        long big = decodeU32(version, 0, Endian.BIG);
        boolean littleOk = little == 2 || little == GGUF_VERSION;
        boolean bigOk = big == 2 || big == GGUF_VERSION;
        if (littleOk) {
            return Endian.LITTLE;
        }
        if (bigOk) {
            return Endian.BIG;
        }
        throw new IOException("Invalid GGUF file '" + file + "': version " + Long.toUnsignedString(little)
                + " read little-endian and " + Long.toUnsignedString(big) + " read big-endian; neither is a"
                + " supported version (2 or " + GGUF_VERSION + ")");
    }

    // ------------------------------------------------------------------ header

    /** Reads the whole header of {@code r}, validates it and writes it to {@code w} in the target order. */
    private static Layout parse(Reader r, Writer w) throws IOException {
        Layout layout = new Layout();
        layout.fileEndian = r.order;
        byte[] magic = new byte[4];
        r.readFully(magic, 0, 4);
        checkMagic(magic, r.path);
        w.bytes(magic, 0, 4);

        byte[] versionBytes = new byte[4];
        r.readFully(versionBytes, 0, 4);
        Endian detected = detectEndian(versionBytes, r.path);
        if (detected != r.order) {
            throw new IOException("Invalid GGUF file '" + r.path + "': the reader assumes "
                    + r.order + " endian but the version field says " + detected);
        }
        layout.version = decodeU32(versionBytes, 0, r.order);
        w.u32((int) layout.version);

        layout.tensorCount = r.u64();
        layout.kvCount = r.u64();
        long tensorLimit = Math.min(MAX_ENTRIES, Math.max(0L, r.remaining() / MIN_TENSOR_INFO_BYTES));
        if (layout.tensorCount < 0 || layout.tensorCount > tensorLimit) {
            throw invalid(r.path, "number of tensors is " + Long.toUnsignedString(layout.tensorCount)
                    + " but must be in [0, " + tensorLimit + "] for a file of " + r.size + " bytes");
        }
        long kvLimit = Math.min(MAX_ENTRIES, Math.max(0L,
                r.remaining() - layout.tensorCount * MIN_TENSOR_INFO_BYTES) / MIN_KV_BYTES);
        if (layout.kvCount < 0 || layout.kvCount > kvLimit) {
            throw invalid(r.path, "number of key/value pairs is " + Long.toUnsignedString(layout.kvCount)
                    + " but must be in [0, " + kvLimit + "] for a file of " + r.size + " bytes and "
                    + layout.tensorCount + " tensors");
        }
        w.u64(layout.tensorCount);
        w.u64(layout.kvCount);

        readKvs(r, w, layout);
        readTensorInfos(r, w, layout);
        finishLayout(r, layout);
        return layout;
    }

    private static void readKvs(Reader r, Writer w, Layout layout) throws IOException {
        Set<String> keys = new HashSet<>();
        for (long i = 0; i < layout.kvCount; i++) {
            String key = r.string("key " + i);
            if (key.isEmpty()) {
                throw invalid(r.path, "key " + i + " is empty");
            }
            if (!keys.add(key)) {
                throw invalid(r.path, "duplicate key '" + key + "' for key/value pair " + i);
            }
            w.string(key);
            GGUFValueType type = valueType(r, key);
            w.u32(type.id());
            if (GgufKeys.GENERAL_ALIGNMENT.equals(key) && type != GGUFValueType.UINT32) {
                throw invalid(r.path, GgufKeys.GENERAL_ALIGNMENT + " must be of type UINT32 but is " + type);
            }
            if (type == GGUFValueType.ARRAY) {
                GGUFValueType elementType = valueType(r, "array key '" + key + "'");
                if (elementType == GGUFValueType.ARRAY) {
                    throw invalid(r.path, "key '" + key + "' has invalid GGUF type: an array of arrays");
                }
                w.u32(elementType.id());
                long count = r.u64();
                if (count < 0 || count > MAX_ARRAY_ELEMENTS) {
                    throw invalid(r.path, "array key '" + key + "' has " + Long.toUnsignedString(count)
                            + " elements, more than the maximum of " + MAX_ARRAY_ELEMENTS);
                }
                long minBytes = count * minElementBytes(elementType);
                if (minBytes > r.remaining()) {
                    throw invalid(r.path, "array key '" + key + "' has " + count + " " + elementType
                            + " elements needing at least " + minBytes + " bytes, but only " + r.remaining()
                            + " bytes are left in the file");
                }
                w.u64(count);
                for (long k = 0; k < count; k++) {
                    copyValue(r, w, elementType, "element " + k + " of array key '" + key + "'");
                }
            } else if (GgufKeys.GENERAL_ALIGNMENT.equals(key)) {
                long value = r.u32() & 0xFFFF_FFFFL;
                w.u32((int) value);
                layout.alignment = value;
            } else {
                copyValue(r, w, type, "key '" + key + "'");
            }
        }
        if (layout.alignment <= 0 || (layout.alignment & (layout.alignment - 1)) != 0) {
            throw invalid(r.path, GgufKeys.GENERAL_ALIGNMENT + " " + layout.alignment
                    + " is not a positive power of two");
        }
    }

    private static void readTensorInfos(Reader r, Writer w, Layout layout) throws IOException {
        Set<String> names = new HashSet<>();
        for (long i = 0; i < layout.tensorCount; i++) {
            byte[] nameBytes = r.stringBytes("name of tensor " + i);
            if (nameBytes.length >= GGUFTensorInfo.MAX_NAME_BYTES) {
                throw invalid(r.path, "name of tensor " + i + " is " + nameBytes.length + " bytes long, more than"
                        + " the " + (GGUFTensorInfo.MAX_NAME_BYTES - 1) + " allowed");
            }
            String name = new String(nameBytes, StandardCharsets.UTF_8);
            if (!names.add(name)) {
                throw invalid(r.path, "duplicate tensor name '" + name + "'");
            }
            w.u64(nameBytes.length);
            w.bytes(nameBytes, 0, nameBytes.length);

            long nDims = r.u32() & 0xFFFF_FFFFL;
            if (nDims > GGUFTensorInfo.MAX_DIMS) {
                throw invalid(r.path, "tensor '" + name + "' has " + nDims + " dimensions, more than the maximum"
                        + " of " + GGUFTensorInfo.MAX_DIMS);
            }
            w.u32((int) nDims);
            long[] dims = new long[(int) nDims];
            for (int d = 0; d < dims.length; d++) {
                dims[d] = r.u64();
                w.u64(dims[d]);
            }
            int typeId = r.u32();
            GGMLType type = GGMLType.fromId(typeId);
            w.u32(typeId);
            long offset = r.u64();
            w.u64(offset);
            TensorMeta meta = new TensorMeta(name, dims, type, offset);
            GGUFTensorInfo.validate("Invalid GGUF file '" + r.path + "'", meta.info());
            layout.tensors.add(meta);
        }
    }

    /** Resolves the shared type ids, validates the data layout and locates the data section. */
    private static void finishLayout(Reader r, Layout layout) throws IOException {
        layout.dataOffset = alignUp(r.position(), layout.alignment);
        if (layout.dataOffset > r.size) {
            throw invalid(r.path, "the data section starts at byte " + layout.dataOffset + ", beyond the end of"
                    + " the file (" + r.size + " bytes)");
        }
        long dataLength = r.size - layout.dataOffset;
        resolveAvalonTypes(r.path, layout.tensors, layout.alignment, dataLength);
        long expected = 0;
        for (TensorMeta t : layout.tensors) {
            t.size = t.type.getBytesForElements(t.info().numElements());
            if (t.offset != expected) {
                throw invalid(r.path, "tensor '" + t.name + "' has data offset "
                        + Long.toUnsignedString(t.offset) + ", expected " + expected
                        + " (tensor data must be contiguous and padded to " + layout.alignment + " bytes)");
            }
            if (t.size > Long.MAX_VALUE - layout.dataOffset - expected) {
                throw invalid(r.path, "tensor '" + t.name + "' of " + t.size
                        + " bytes overflows the data section");
            }
            long end = layout.dataOffset + expected + t.size;
            if (end > r.size) {
                throw invalid(r.path, "data of tensor '" + t.name + "' ends at byte " + end
                        + ", beyond the end of the file (" + r.size + " bytes)");
            }
            expected = alignUp(expected + t.size, layout.alignment);
        }
        layout.dataLength = dataLength;
    }

    /**
     * Re-reads the ggml type ids two type families share from the slot sizes, as
     * {@link com.eonmux.neuralfree.gguf.GGUFFileParser} does: type id 43 is F8_E4M3FN per element in mainline ggml
     * and IQ3_K in Avalon's block format; id 44 that is not the IQ3_KS block format is rejected.
     */
    private static void resolveAvalonTypes(Path path, List<TensorMeta> tensors, long alignment, long dataLength)
            throws IOException {
        for (int i = 0; i < tensors.size(); i++) {
            TensorMeta t = tensors.get(i);
            if (t.type != GGMLType.F8_E4M3FN && t.type != GGMLType.IQ3_KS) {
                continue;
            }
            long next = i + 1 < tensors.size() ? tensors.get(i + 1).offset : dataLength;
            long room = next - t.offset;
            if (room < 0) {
                continue;
            }
            long elements = t.info().numElements();
            boolean perElement = elements <= room && room < elements + alignment;
            GGMLType avalon = t.type == GGMLType.F8_E4M3FN ? GGMLType.IQ3_K : t.type;
            long blockBytes = avalon.getBytesForElements(elements);
            boolean blockFormat = blockBytes <= room && room < blockBytes + alignment;
            if (t.type == GGMLType.F8_E4M3FN) {
                if (!perElement && blockFormat && t.dims.length > 0
                        && t.dims[0] % GGMLType.IQ3_K.getBlockSize() == 0) {
                    t.type = GGMLType.IQ3_K;
                }
            } else if (!blockFormat && perElement) {
                throw invalid(path, "tensor '" + t.name + "' declares ggml type id 44 (IQ3_KS) but its data is"
                        + " laid out at 1 byte per element, the layout of mainline llama.cpp's type id 43"
                        + " (F8_E4M3); its endianness is ambiguous, convert it with the tool of the library that"
                        + " produced it");
            }
        }
    }

    // ------------------------------------------------------------------ values

    private static GGUFValueType valueType(Reader r, String owner) throws IOException {
        int id = r.u32();
        try {
            return GGUFValueType.fromId(id);
        } catch (IllegalArgumentException e) {
            throw invalid(r.path, owner + " has invalid GGUF type " + Integer.toUnsignedString(id));
        }
    }

    /** Copies one scalar or string value, swapping its multi-byte representation into the target order. */
    private static void copyValue(Reader r, Writer w, GGUFValueType type, String owner) throws IOException {
        switch (type) {
            case UINT8:
            case INT8:
            case BOOL:
                w.u8(r.u8());
                break;
            case UINT16:
            case INT16:
                w.u16(r.u16());
                break;
            case UINT32:
            case INT32:
            case FLOAT32:
                w.u32(r.u32());
                break;
            case UINT64:
            case INT64:
            case FLOAT64:
                w.u64(r.u64());
                break;
            case STRING: {
                byte[] bytes = r.stringBytes(owner);
                w.u64(bytes.length);
                w.bytes(bytes, 0, bytes.length);
                break;
            }
            default:
                throw invalid(r.path, owner + " has invalid GGUF type " + type);
        }
    }

    private static int minElementBytes(GGUFValueType type) {
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
            default:
                return 8;
        }
    }

    // ------------------------------------------------------------------ tensor data

    /** One byte-swapped field of a block: {@code count} units of {@code width} bytes at {@code offset}. */
    private static final class SwapField {
        final int offset;
        final int width;
        final int count;

        SwapField(int offset, int width, int count) {
            this.offset = offset;
            this.width = width;
            this.count = count;
        }
    }

    /** Rejects the types the converter cannot represent and gives every tensor its swap plan. */
    private static void swapPlans(List<TensorMeta> tensors) throws IOException {
        for (TensorMeta t : tensors) {
            if (t.type.isExtension()) {
                throw new IOException("tensor '" + t.name + "' stores NeuralFree extension type " + t.type
                        + ", whose block layout is defined little-endian only; it cannot be converted");
            }
            if (!BlockLayout.isDefined(t.type)) {
                throw new IOException("tensor '" + t.name + "' has ggml type " + t.type
                        + ", which has no block layout to byte-swap");
            }
            fieldsOf(t.type);
        }
    }

    /**
     * The multi-byte fields of one block of {@code type}, at the offsets {@link BlockLayout} documents. Throws for
     * an undefined type.
     */
    private static List<SwapField> fieldsOf(GGMLType type) {
        switch (type) {
            case F32:
            case I32:
                return one(4);
            case F16:
            case BF16:
            case I16:
                return one(2);
            case F64:
            case I64:
                return one(8);
            case I8:
            case F8_E4M3FN:
            case IQ1_M:
            case MXFP4:
            case NVFP4:
                return Collections.emptyList();
            default:
                return blockFields(type);
        }
    }

    /** The multi-byte fields of one block of a quantized type. */
    private static List<SwapField> blockFields(GGMLType type) {
        switch (type) {
            case Q1_0:
                return swap(BlockLayout.Q1_0_D, 2, 1);
            case Q2_0:
                return swap(BlockLayout.Q2_0_D, 2, 1);
            case Q4_0:
                return swap(BlockLayout.Q4_0_D, 2, 1);
            case Q4_1:
                return swap(BlockLayout.Q4_1_D, 2, 1, BlockLayout.Q4_1_M, 2, 1);
            case Q5_0:
                return swap(BlockLayout.Q5_0_D, 2, 1, BlockLayout.Q5_0_QH, 4, 1);
            case Q5_1:
                return swap(BlockLayout.Q5_1_D, 2, 1, BlockLayout.Q5_1_M, 2, 1, BlockLayout.Q5_1_QH, 4, 1);
            case Q8_0:
                return swap(BlockLayout.Q8_0_D, 2, 1);
            case Q8_1:
                return swap(BlockLayout.Q8_1_D, 2, 1, BlockLayout.Q8_1_S, 2, 1);
            case Q2_K:
                return swap(BlockLayout.Q2_K_D, 2, 1, BlockLayout.Q2_K_DMIN, 2, 1);
            case Q3_K:
                return swap(BlockLayout.Q3_K_D, 2, 1);
            case Q4_K:
                return swap(BlockLayout.Q4_K_D, 2, 1, BlockLayout.Q4_K_DMIN, 2, 1);
            case Q5_K:
                return swap(BlockLayout.Q5_K_D, 2, 1, BlockLayout.Q5_K_DMIN, 2, 1);
            case Q6_K:
                return swap(BlockLayout.Q6_K_D, 2, 1);
            case Q8_K:
                return swap(BlockLayout.Q8_K_D, 4, 1, BlockLayout.Q8_K_BSUMS, 2, 16);
            case IQ2_XXS:
                return swap(BlockLayout.IQ2_XXS_D, 2, 1, BlockLayout.IQ2_XXS_QS, 2, 32);
            case IQ2_XS:
                return swap(BlockLayout.IQ2_XS_D, 2, 1, BlockLayout.IQ2_XS_QS, 2, 32);
            case IQ2_S:
                return swap(BlockLayout.IQ2_S_D, 2, 1);
            case IQ3_XXS:
                return swap(BlockLayout.IQ3_XXS_D, 2, 1);
            case IQ3_S:
                return swap(BlockLayout.IQ3_S_D, 2, 1);
            case IQ1_S:
                return swap(BlockLayout.IQ1_S_D, 2, 1, BlockLayout.IQ1_S_QH, 2, 8);
            case IQ4_NL:
                return swap(BlockLayout.IQ4_NL_D, 2, 1);
            case IQ4_XS:
                return swap(BlockLayout.IQ4_XS_D, 2, 1, BlockLayout.IQ4_XS_SCALES_H, 2, 1);
            case TQ1_0:
                return swap(BlockLayout.TQ1_0_D, 2, 1);
            case TQ2_0:
                return swap(BlockLayout.TQ2_0_D, 2, 1);
            case IQ3_K:
                return swap(BlockLayout.IQ3_K_D, 2, 1, BlockLayout.IQ3_K_EXTRA, 2, 1,
                        BlockLayout.IQ3_K_SCALES_H, 2, 1);
            case IQ3_KS:
                return swap(BlockLayout.IQ3_KS_D, 2, 1, BlockLayout.IQ3_KS_EXTRA, 2, 1);
            default:
                throw new IllegalArgumentException("ggml type " + type + " has no byte-swap layout");
        }
    }

    /** A single field of element size {@code width} per block, for the element types. */
    private static List<SwapField> one(int width) {
        return Collections.singletonList(new SwapField(0, width, 1));
    }

    /** Fields given as {@code offset, width, count} triples. */
    private static List<SwapField> swap(int... offsetWidthCount) {
        List<SwapField> out = new ArrayList<>(offsetWidthCount.length / 3);
        for (int i = 0; i < offsetWidthCount.length; i += 3) {
            out.add(new SwapField(offsetWidthCount[i], offsetWidthCount[i + 1], offsetWidthCount[i + 2]));
        }
        return out;
    }

    /** Copies the data section, byte-swapping every tensor's blocks and leaving padding and trailing bytes. */
    private static void copyData(Reader r, Writer w, Layout layout) throws IOException {
        long cursor = layout.dataOffset;
        for (TensorMeta t : layout.tensors) {
            long start = layout.dataOffset + t.offset;
            if (start > cursor) {
                copyRange(r, w, cursor, start - cursor);
                cursor = start;
            }
            swapTensor(r, w, start, t);
            cursor = start + t.size;
        }
        if (r.size > cursor) {
            copyRange(r, w, cursor, r.size - cursor);
        }
    }

    private static void copyRange(Reader r, Writer w, long position, long length) throws IOException {
        r.position(position);
        byte[] buf = new byte[(int) Math.min(CHUNK_BYTES, length)];
        long left = length;
        while (left > 0) {
            int n = (int) Math.min(buf.length, left);
            r.readFully(buf, 0, n);
            w.bytes(buf, 0, n);
            left -= n;
        }
    }

    private static void swapTensor(Reader r, Writer w, long position, TensorMeta t) throws IOException {
        List<SwapField> fields = fieldsOf(t.type);
        int blockBytes = t.type.getTypeSize();
        if (fields.isEmpty() || blockBytes <= 1) {
            copyRange(r, w, position, t.size);
            return;
        }
        r.position(position);
        int chunkBlocks = Math.max(1, CHUNK_BYTES / blockBytes);
        byte[] buf = new byte[chunkBlocks * blockBytes];
        long left = t.size;
        while (left > 0) {
            int n = (int) Math.min(buf.length, left);
            r.readFully(buf, 0, n);
            swapBlocks(buf, n, blockBytes, fields);
            w.bytes(buf, 0, n);
            left -= n;
        }
    }

    /** Swaps every planned field of every whole block in {@code buf[0, len)}. */
    private static void swapBlocks(byte[] buf, int len, int blockBytes, List<SwapField> fields) {
        for (int block = 0; block + blockBytes <= len; block += blockBytes) {
            for (SwapField f : fields) {
                for (int unit = 0; unit < f.count; unit++) {
                    int at = block + f.offset + unit * f.width;
                    for (int i = 0; i < f.width / 2; i++) {
                        byte tmp = buf[at + i];
                        buf[at + i] = buf[at + f.width - 1 - i];
                        buf[at + f.width - 1 - i] = tmp;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ primitives

    private static void checkMagic(byte[] magic, Path path) throws IOException {
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
            throw new IOException("Invalid GGUF file '" + path + "': bad magic '"
                    + new String(magic, StandardCharsets.ISO_8859_1) + "', expected 'GGUF'");
        }
    }

    private static long decodeU32(byte[] b, int off, Endian order) {
        if (order == Endian.LITTLE) {
            return (b[off] & 0xFFL) | (b[off + 1] & 0xFFL) << 8 | (b[off + 2] & 0xFFL) << 16
                    | (b[off + 3] & 0xFFL) << 24;
        }
        return (b[off] & 0xFFL) << 24 | (b[off + 1] & 0xFFL) << 16 | (b[off + 2] & 0xFFL) << 8
                | (b[off + 3] & 0xFFL);
    }

    private static long decodeU64(byte[] b, int off, Endian order) {
        long v = 0;
        if (order == Endian.LITTLE) {
            for (int i = 7; i >= 0; i--) {
                v = (v << 8) | (b[off + i] & 0xFFL);
            }
        } else {
            for (int i = 0; i < 8; i++) {
                v = (v << 8) | (b[off + i] & 0xFFL);
            }
        }
        return v;
    }

    private static long alignUp(long value, long alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }

    private static IOException invalid(Path path, String detail) {
        return new IOException("Invalid GGUF file '" + path + "': " + detail);
    }

    /** A positional, buffered reader of one file in its own byte order. */
    private static final class Reader implements Closeable {

        final Path path;
        final FileChannel channel;
        final long size;
        final Endian order;
        private long position;
        private final byte[] scratch = new byte[8];

        Reader(Path path) throws IOException {
            this(path, null);
        }

        Reader(Path path, Endian order) throws IOException {
            this.path = path;
            this.channel = FileChannel.open(path, StandardOpenOption.READ);
            this.size = channel.size();
            this.position = 0;
            this.order = order == null ? readEndianAt(path, channel) : order;
        }

        private static Endian readEndianAt(Path path, FileChannel channel) throws IOException {
            ByteBuffer b = ByteBuffer.allocate(8);
            long at = 0;
            while (b.hasRemaining()) {
                int n = channel.read(b, at);
                if (n < 0) {
                    throw new IOException("Invalid GGUF file '" + path + "': the file ends before its version");
                }
                at += n;
            }
            byte[] bytes = b.array();
            if (bytes[0] != 'G' || bytes[1] != 'G' || bytes[2] != 'U' || bytes[3] != 'F') {
                throw new IOException("Invalid GGUF file '" + path + "': bad magic '"
                        + new String(bytes, 0, 4, StandardCharsets.ISO_8859_1) + "', expected 'GGUF'");
            }
            byte[] version = new byte[]{bytes[4], bytes[5], bytes[6], bytes[7]};
            return detectEndian(version, path);
        }

        long position() {
            return position;
        }

        void position(long p) {
            position = p;
        }

        long remaining() {
            return size - position;
        }

        void readFully(byte[] dst, int off, int len) throws IOException {
            if (len < 0 || len > remaining()) {
                throw new IOException("Invalid GGUF file '" + path + "': " + len + " bytes needed at byte "
                        + position + ", but only " + Math.max(0, remaining()) + " are left");
            }
            ByteBuffer b = ByteBuffer.wrap(dst, off, len);
            long at = position;
            while (b.hasRemaining()) {
                int n = channel.read(b, at);
                if (n < 0) {
                    throw new IOException("Invalid GGUF file '" + path + "': the file ended at byte " + at);
                }
                at += n;
            }
            position += len;
        }

        int u8() throws IOException {
            readFully(scratch, 0, 1);
            return scratch[0] & 0xFF;
        }

        int u16() throws IOException {
            readFully(scratch, 0, 2);
            return order == Endian.LITTLE
                    ? (scratch[0] & 0xFF) | (scratch[1] & 0xFF) << 8
                    : (scratch[0] & 0xFF) << 8 | (scratch[1] & 0xFF);
        }

        int u32() throws IOException {
            readFully(scratch, 0, 4);
            return (int) decodeU32(scratch, 0, order);
        }

        long u64() throws IOException {
            readFully(scratch, 0, 8);
            return decodeU64(scratch, 0, order);
        }

        byte[] stringBytes(String owner) throws IOException {
            long len = u64();
            if (len < 0 || len > MAX_STRING_LENGTH) {
                throw new IOException("Invalid GGUF file '" + path + "': string length "
                        + Long.toUnsignedString(len) + " of " + owner + " exceeds the maximum of "
                        + MAX_STRING_LENGTH);
            }
            if (len > remaining()) {
                throw new IOException("Invalid GGUF file '" + path + "': string length " + len + " of " + owner
                        + " exceeds the " + remaining() + " bytes left in the file");
            }
            byte[] bytes = new byte[(int) len];
            readFully(bytes, 0, bytes.length);
            return bytes;
        }

        String string(String owner) throws IOException {
            return new String(stringBytes(owner), StandardCharsets.UTF_8);
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    /** A buffered writer in the target order; a null channel only counts the bytes {@link #position()}. */
    private static final class Writer implements Closeable {

        private final FileChannel channel;
        private final Endian order;
        private final ByteBuffer buf;
        private long position;

        Writer(Path path, Endian order) throws IOException {
            this(open(path), order);
        }

        /** A writer that keeps no bytes and only counts them, for a validating read. */
        static Writer counting(Endian order) {
            return new Writer((FileChannel) null, order);
        }

        private Writer(FileChannel channel, Endian order) {
            this.channel = channel;
            this.order = order;
            this.buf = channel == null ? null : ByteBuffer.allocate(BUFFER_BYTES).order(order.order());
        }

        private static FileChannel open(Path path) throws IOException {
            return FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING);
        }

        long position() {
            return position;
        }

        void u8(int v) throws IOException {
            if (buf != null) {
                ensure(1);
                buf.put((byte) v);
            }
            position += 1;
        }

        void u16(int v) throws IOException {
            if (buf != null) {
                ensure(2);
                buf.putShort((short) v);
            }
            position += 2;
        }

        void u32(int v) throws IOException {
            if (buf != null) {
                ensure(4);
                buf.putInt(v);
            }
            position += 4;
        }

        void u64(long v) throws IOException {
            if (buf != null) {
                ensure(8);
                buf.putLong(v);
            }
            position += 8;
        }

        void string(String s) throws IOException {
            byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
            u64(bytes.length);
            bytes(bytes, 0, bytes.length);
        }

        void bytes(byte[] src, int off, int len) throws IOException {
            position += len;
            if (buf == null) {
                return;
            }
            int i = off;
            int left = len;
            while (left > 0) {
                if (!buf.hasRemaining()) {
                    flush();
                }
                int n = Math.min(left, buf.remaining());
                buf.put(src, i, n);
                i += n;
                left -= n;
            }
        }

        void zeros(long count) throws IOException {
            byte[] zeros = new byte[(int) Math.min(BUFFER_BYTES, count)];
            long left = count;
            while (left > 0) {
                int n = (int) Math.min(zeros.length, left);
                bytes(zeros, 0, n);
                left -= n;
            }
        }

        void force() throws IOException {
            flush();
            if (channel != null) {
                channel.force(true);
            }
        }

        private void ensure(int n) throws IOException {
            if (buf.remaining() < n) {
                flush();
            }
        }

        private void flush() throws IOException {
            if (buf == null || channel == null) {
                return;
            }
            buf.flip();
            while (buf.hasRemaining()) {
                int n = channel.write(buf);
                if (n <= 0) {
                    throw new IOException("failed to write a converted GGUF file at byte " + position);
                }
            }
            buf.clear();
        }

        @Override
        public void close() throws IOException {
            try {
                force();
            } finally {
                if (channel != null) {
                    channel.close();
                }
            }
        }
    }
}
