package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Builds GGUF files in memory for tests: every metadata value type, typed arrays, and tensors of
 * any storage {@link GGMLType}.
 *
 * <p>The layout follows llama.cpp ggml/src/gguf.cpp: little-endian header (magic, version, tensor count,
 * KV count), KV pairs, tensor infos ({@code n_dims} as uint32, {@code ne[]} as uint64, type id,
 * offset relative to the data section), zero padding to the alignment, then each tensor's data
 * padded to the alignment. The alignment comes from a UINT32 {@code general.alignment} entry with a
 * positive value, else {@link #DEFAULT_ALIGNMENT}; any other form of that key is written as given so
 * that tests can build files a reader must reject.
 *
 * <p>Rules enforced as in gguf.cpp: non-empty keys, tensor names shorter than 64 bytes, 1 to 4
 * dimensions, {@code ne[0]} a multiple of the block size, data length equal to
 * {@code type.getBytesForElements(n)}, and no nested arrays. Tensor sizes always come from
 * {@link GGMLType}, so the writer and the parser agree on every block layout. Keys and tensor
 * names stay unique: adding an existing one replaces it in place, keeping its write position.
 *
 * <p>Float encoders exist for F32, F16, BF16 and Q8_0 ({@link #canEncodeFloats(GGMLType)}); other
 * types take raw block bytes or zero blocks, which decode to zero for every ggml format.
 */
public final class GgufTestWriter {

    public static final int GGUF_VERSION = 3;
    public static final int DEFAULT_ALIGNMENT = 32;
    public static final String ALIGNMENT_KEY = "general.alignment";
    public static final String ARCHITECTURE_KEY = "general.architecture";
    public static final int MAX_DIMS = 4;
    public static final int MAX_NAME_BYTES = 63;

    private static final byte[] MAGIC = {'G', 'G', 'U', 'F'};
    private static final int QK8_0 = 32;

    /** GGUF metadata value types with their on-disk ids. */
    public enum ValueType {
        UINT8(0), INT8(1), UINT16(2), INT16(3), UINT32(4), INT32(5), FLOAT32(6), BOOL(7), STRING(8),
        ARRAY(9), UINT64(10), INT64(11), FLOAT64(12);

        private final int id;

        ValueType(int id) {
            this.id = id;
        }

        public int id() {
            return id;
        }
    }

    /** Scalar value or array payload (defensively copied primitive or String array). */
    private static final class Kv {
        final ValueType type;
        final ValueType elementType;
        final Object value;

        Kv(ValueType type, ValueType elementType, Object value) {
            this.type = type;
            this.elementType = elementType;
            this.value = value;
        }
    }

    private static final class Tensor {
        final String name;
        final GGMLType type;
        final long[] ne;
        final byte[] data;

        Tensor(String name, GGMLType type, long[] ne, byte[] data) {
            this.name = name;
            this.type = type;
            this.ne = ne;
            this.data = data;
        }
    }

    private final Map<String, Kv> kvs = new LinkedHashMap<>();
    private final Map<String, Tensor> tensors = new LinkedHashMap<>();
    private int version = GGUF_VERSION;

    // ================================================================== header

    /** Sets the header version; 2 and 3 share the little-endian layout written here. */
    public GgufTestWriter version(int v) {
        if (v != 2 && v != 3) {
            throw new IllegalArgumentException("only GGUF versions 2 and 3 share this layout, got " + v);
        }
        this.version = v;
        return this;
    }

    /** Writes {@code general.alignment} as UINT32; the value must be a power of two. */
    public GgufTestWriter alignment(int alignment) {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IllegalArgumentException("alignment must be a positive power of two, got " + alignment);
        }
        return u32(ALIGNMENT_KEY, alignment);
    }

    /** Writes {@code general.architecture}, which also sets the prefix used by {@link #archKey}. */
    public GgufTestWriter architecture(String arch) {
        return str(ARCHITECTURE_KEY, arch);
    }

    /** {@code "<arch>.<suffix>"} for the architecture set by {@link #architecture(String)}. */
    public String archKey(String suffix) {
        Kv kv = kvs.get(ARCHITECTURE_KEY);
        if (kv == null || kv.type != ValueType.STRING) {
            throw new IllegalStateException("general.architecture must be set as a string before archKey(\""
                    + suffix + "\")");
        }
        return kv.value + "." + suffix;
    }

    // ================================================================== scalar KV

    public GgufTestWriter u8(String key, int v) {
        return put(key, ValueType.UINT8, null, checkRange(key, v, 0, 0xFF));
    }

    public GgufTestWriter i8(String key, int v) {
        return put(key, ValueType.INT8, null, checkRange(key, v, Byte.MIN_VALUE, Byte.MAX_VALUE));
    }

    public GgufTestWriter u16(String key, int v) {
        return put(key, ValueType.UINT16, null, checkRange(key, v, 0, 0xFFFF));
    }

    public GgufTestWriter i16(String key, int v) {
        return put(key, ValueType.INT16, null, checkRange(key, v, Short.MIN_VALUE, Short.MAX_VALUE));
    }

    /** UINT32; accepts the full unsigned range 0..4294967295. */
    public GgufTestWriter u32(String key, long v) {
        return put(key, ValueType.UINT32, null, checkRange(key, v, 0, 0xFFFFFFFFL));
    }

    public GgufTestWriter i32(String key, int v) {
        return put(key, ValueType.INT32, null, v);
    }

    public GgufTestWriter f32(String key, float v) {
        return put(key, ValueType.FLOAT32, null, v);
    }

    public GgufTestWriter bool(String key, boolean v) {
        return put(key, ValueType.BOOL, null, v);
    }

    public GgufTestWriter str(String key, String v) {
        return put(key, ValueType.STRING, null, Objects.requireNonNull(v, "value of " + key));
    }

    /** UINT64; the long's bits are written as is, so negative values encode values above 2^63. */
    public GgufTestWriter u64(String key, long v) {
        return put(key, ValueType.UINT64, null, v);
    }

    public GgufTestWriter i64(String key, long v) {
        return put(key, ValueType.INT64, null, v);
    }

    public GgufTestWriter f64(String key, double v) {
        return put(key, ValueType.FLOAT64, null, v);
    }

    // ================================================================== array KV

    public GgufTestWriter u8Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, 0, 0xFF);
        }
        return put(key, ValueType.ARRAY, ValueType.UINT8, v.clone());
    }

    public GgufTestWriter i8Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, Byte.MIN_VALUE, Byte.MAX_VALUE);
        }
        return put(key, ValueType.ARRAY, ValueType.INT8, v.clone());
    }

    public GgufTestWriter u16Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, 0, 0xFFFF);
        }
        return put(key, ValueType.ARRAY, ValueType.UINT16, v.clone());
    }

    public GgufTestWriter i16Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, Short.MIN_VALUE, Short.MAX_VALUE);
        }
        return put(key, ValueType.ARRAY, ValueType.INT16, v.clone());
    }

    public GgufTestWriter u32Array(String key, long... v) {
        for (long x : v) {
            checkRange(key, x, 0, 0xFFFFFFFFL);
        }
        return put(key, ValueType.ARRAY, ValueType.UINT32, v.clone());
    }

    public GgufTestWriter i32Array(String key, int... v) {
        return put(key, ValueType.ARRAY, ValueType.INT32, v.clone());
    }

    public GgufTestWriter f32Array(String key, float... v) {
        return put(key, ValueType.ARRAY, ValueType.FLOAT32, v.clone());
    }

    public GgufTestWriter boolArray(String key, boolean... v) {
        return put(key, ValueType.ARRAY, ValueType.BOOL, v.clone());
    }

    public GgufTestWriter strArray(String key, String... v) {
        for (String s : v) {
            Objects.requireNonNull(s, "element of " + key);
        }
        return put(key, ValueType.ARRAY, ValueType.STRING, v.clone());
    }

    public GgufTestWriter strArray(String key, List<String> v) {
        return strArray(key, v.toArray(new String[0]));
    }

    public GgufTestWriter u64Array(String key, long... v) {
        return put(key, ValueType.ARRAY, ValueType.UINT64, v.clone());
    }

    public GgufTestWriter i64Array(String key, long... v) {
        return put(key, ValueType.ARRAY, ValueType.INT64, v.clone());
    }

    public GgufTestWriter f64Array(String key, double... v) {
        return put(key, ValueType.ARRAY, ValueType.FLOAT64, v.clone());
    }

    public GgufTestWriter remove(String key) {
        if (kvs.remove(key) == null) {
            throw new IllegalArgumentException("no metadata key '" + key + "' to remove");
        }
        return this;
    }

    public boolean has(String key) {
        return kvs.containsKey(key);
    }

    /** The value type of {@code key}, or null when absent. */
    public ValueType typeOf(String key) {
        Kv kv = kvs.get(key);
        return kv == null ? null : kv.type;
    }

    /** Metadata keys in write order. */
    public List<String> keys() {
        return Collections.unmodifiableList(new ArrayList<>(kvs.keySet()));
    }

    // ================================================================== tensors

    /** Adds a tensor from raw storage bytes, which must match the type's block layout exactly. */
    public GgufTestWriter tensor(String name, GGMLType type, long[] ne, byte[] data) {
        Objects.requireNonNull(data, "data of " + name);
        long bytes = validateTensor(name, type, ne);
        if (data.length != bytes) {
            throw new IllegalArgumentException("tensor " + name + " (" + type + " " + shape(ne) + ") needs "
                    + bytes + " bytes, got " + data.length);
        }
        tensors.put(name, new Tensor(name, type, ne.clone(), data.clone()));
        return this;
    }

    /** Adds a tensor whose blocks are all zero bytes. */
    public GgufTestWriter tensorZeros(String name, GGMLType type, long... ne) {
        long bytes = validateTensor(name, type, ne);
        tensors.put(name, new Tensor(name, type, ne.clone(), new byte[toIntSize(name, bytes)]));
        return this;
    }

    public GgufTestWriter tensorF32(String name, long[] ne, float... values) {
        return tensorFromFloats(name, GGMLType.F32, ne, values);
    }

    /** Encodes {@code values} (ne[0] fastest) into {@code type}; see {@link #canEncodeFloats}. */
    public GgufTestWriter tensorFromFloats(String name, GGMLType type, long[] ne, float[] values) {
        Objects.requireNonNull(values, "values of " + name);
        long bytes = validateTensor(name, type, ne);
        long n = elementCount(ne);
        if (values.length != n) {
            throw new IllegalArgumentException("tensor " + name + " " + shape(ne) + " has " + n
                    + " elements, got " + values.length + " values");
        }
        byte[] data = encodeFloats(type, values);
        if (data.length != bytes) {
            throw new IllegalStateException("encoded " + data.length + " bytes for " + name + " but " + type
                    + " reports " + bytes);
        }
        tensors.put(name, new Tensor(name, type, ne.clone(), data));
        return this;
    }

    /** Adds a tensor filled with {@code value}. */
    public GgufTestWriter tensorFilled(String name, GGMLType type, float value, long... ne) {
        validateTensor(name, type, ne);
        float[] values = new float[toIntSize(name, elementCount(ne))];
        Arrays.fill(values, value);
        return tensorFromFloats(name, type, ne, values);
    }

    /** Adds a tensor of uniform values in {@code [lo, hi)} from a generator seeded with {@code seed}. */
    public GgufTestWriter tensorRandom(String name, GGMLType type, long seed, float lo, float hi, long... ne) {
        if (!(hi > lo)) {
            throw new IllegalArgumentException("tensor " + name + ": empty value range [" + lo + ", " + hi + ")");
        }
        validateTensor(name, type, ne);
        SplittableRandom rng = new SplittableRandom(seed);
        float[] values = new float[toIntSize(name, elementCount(ne))];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (lo + (hi - lo) * rng.nextDouble());
        }
        return tensorFromFloats(name, type, ne, values);
    }

    public GgufTestWriter removeTensor(String name) {
        if (tensors.remove(name) == null) {
            throw new IllegalArgumentException("no tensor '" + name + "' to remove");
        }
        return this;
    }

    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    /** Tensor names in write order. */
    public List<String> tensorNames() {
        return Collections.unmodifiableList(new ArrayList<>(tensors.keySet()));
    }

    /** {@code ne[]} of a tensor (copy). */
    public long[] tensorShape(String name) {
        return requireTensor(name).ne.clone();
    }

    public GGMLType tensorType(String name) {
        return requireTensor(name).type;
    }

    // ================================================================== layout and output

    /** The alignment the layout uses (see the class contract). */
    public int effectiveAlignment() {
        Kv kv = kvs.get(ALIGNMENT_KEY);
        if (kv != null && kv.type == ValueType.UINT32) {
            long a = (Long) kv.value;
            if (a > 0 && a <= Integer.MAX_VALUE) {
                return (int) a;
            }
        }
        return DEFAULT_ALIGNMENT;
    }

    /** Absolute file offset of the data section. */
    public long dataSectionOffset() {
        return alignUp(headerBytes().length, effectiveAlignment());
    }

    /** Offset of a tensor's data relative to the data section, as stored in its tensor info. */
    public long tensorDataOffset(String name) {
        requireTensor(name);
        int alignment = effectiveAlignment();
        long offset = 0;
        for (Tensor t : tensors.values()) {
            if (t.name.equals(name)) {
                return offset;
            }
            offset += alignUp(t.data.length, alignment);
        }
        throw new IllegalStateException("unreachable: tensor " + name + " vanished");
    }

    public byte[] toByteArray() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            writeTo(out);
        } catch (IOException e) {
            throw new IllegalStateException("in-memory GGUF serialisation failed", e);
        }
        return out.toByteArray();
    }

    public void writeTo(Path path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path))) {
            writeTo(out);
        }
    }

    /** Writes to a new temporary {@code .gguf} file that is deleted when the JVM exits. */
    public Path writeTempFile() throws IOException {
        Path path = Files.createTempFile("nf-synthetic-", ".gguf");
        path.toFile().deleteOnExit();
        writeTo(path);
        return path;
    }

    public void writeTo(OutputStream out) throws IOException {
        int alignment = effectiveAlignment();
        byte[] header = headerBytes();
        out.write(header);
        writeZeros(out, alignUp(header.length, alignment) - header.length);
        for (Tensor t : tensors.values()) {
            out.write(t.data);
            writeZeros(out, alignUp(t.data.length, alignment) - t.data.length);
        }
        out.flush();
    }

    // ================================================================== float encoders

    /** True when {@link #encodeFloats} supports {@code type}. */
    public static boolean canEncodeFloats(GGMLType type) {
        return type == GGMLType.F32 || type == GGMLType.F16 || type == GGMLType.BF16 || type == GGMLType.Q8_0;
    }

    /** Encodes floats into the storage bytes of {@code type}. Q8_0 needs a multiple of 32 values. */
    public static byte[] encodeFloats(GGMLType type, float[] values) {
        switch (type) {
            case F32: {
                byte[] out = new byte[values.length * 4];
                for (int i = 0; i < values.length; i++) {
                    FakeGpuBuffer.putIntLE(out, i * 4, Float.floatToRawIntBits(values[i]));
                }
                return out;
            }
            case F16:
            case BF16: {
                byte[] out = new byte[values.length * 2];
                for (int i = 0; i < values.length; i++) {
                    short h = type == GGMLType.F16 ? floatToHalfBits(values[i]) : floatToBf16Bits(values[i]);
                    out[i * 2] = (byte) h;
                    out[i * 2 + 1] = (byte) (h >>> 8);
                }
                return out;
            }
            case Q8_0:
                return quantizeQ8_0(values);
            default:
                throw new IllegalArgumentException("no float encoder for " + type
                        + "; supported: F32, F16, BF16, Q8_0. Use tensor(raw bytes) or tensorZeros");
        }
    }

    /**
     * IEEE binary16 bits of {@code f} with round-to-nearest-even, NaN kept quiet and overflow to
     * infinity, matching llama.cpp ggml/src/ggml-impl.h {@code ggml_compute_fp32_to_fp16}.
     */
    public static short floatToHalfBits(float f) {
        int x = Float.floatToRawIntBits(f);
        int sign = (x >>> 16) & 0x8000;
        int abs = x & 0x7FFFFFFF;
        if (abs >= 0x7F800000) {
            return (short) (sign | 0x7C00 | (abs > 0x7F800000 ? 0x0200 : 0));
        }
        if (abs >= 0x477FF000) {
            return (short) (sign | 0x7C00);
        }
        if (abs < 0x38800000) {
            int e = abs >>> 23;
            int shift = 126 - e;
            if (e == 0 || shift > 24) {
                return (short) sign;
            }
            int m = (abs & 0x7FFFFF) | 0x800000;
            int q = m >>> shift;
            int rem = m & ((1 << shift) - 1);
            int half = 1 << (shift - 1);
            if (rem > half || (rem == half && (q & 1) != 0)) {
                q++;
            }
            return (short) (sign | q);
        }
        int h = (((abs >>> 23) - 112) << 10) | ((abs >>> 13) & 0x3FF);
        int rem = abs & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0)) {
            h++;
        }
        return (short) (sign | h);
    }

    /** bfloat16 bits of {@code f}, matching llama.cpp ggml-impl.h {@code ggml_compute_fp32_to_bf16}. */
    public static short floatToBf16Bits(float f) {
        int u = Float.floatToRawIntBits(f);
        if ((u & 0x7FFFFFFF) > 0x7F800000) {
            return (short) ((u >>> 16) | 64);
        }
        return (short) ((u + (0x7FFF + ((u >>> 16) & 1))) >>> 16);
    }

    /** llama.cpp ggml/src/ggml-quants.c {@code quantize_row_q8_0_ref}: d = amax/127, q = roundf(x/d). */
    private static byte[] quantizeQ8_0(float[] x) {
        if (x.length % QK8_0 != 0) {
            throw new IllegalArgumentException("Q8_0 needs a multiple of " + QK8_0 + " values, got " + x.length);
        }
        int nb = x.length / QK8_0;
        int blockBytes = GGMLType.Q8_0.getTypeSize();
        byte[] out = new byte[nb * blockBytes];
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                amax = Math.max(amax, Math.abs(x[i * QK8_0 + j]));
            }
            float d = amax / ((1 << 7) - 1);
            float id = d != 0.0f ? 1.0f / d : 0.0f;
            int p = i * blockBytes;
            short dh = floatToHalfBits(d);
            out[p] = (byte) dh;
            out[p + 1] = (byte) (dh >>> 8);
            for (int j = 0; j < QK8_0; j++) {
                float x0 = x[i * QK8_0 + j] * id;
                out[p + 2 + j] = (byte) roundHalfAwayFromZero(x0);
            }
        }
        return out;
    }

    private static int roundHalfAwayFromZero(float v) {
        return (int) (v < 0 ? -Math.floor(-(double) v + 0.5) : Math.floor((double) v + 0.5));
    }

    // ================================================================== internals

    private GgufTestWriter put(String key, ValueType type, ValueType elementType, Object value) {
        if (key == null || key.isEmpty()) {
            throw new IllegalArgumentException("metadata key must be non-empty");
        }
        kvs.put(key, new Kv(type, elementType, value));
        return this;
    }

    private static long checkRange(String key, long v, long lo, long hi) {
        if (v < lo || v > hi) {
            throw new IllegalArgumentException("value " + v + " for '" + key + "' is outside [" + lo + ", " + hi + "]");
        }
        return v;
    }

    private long validateTensor(String name, GGMLType type, long[] ne) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("tensor name must be non-empty");
        }
        if (name.getBytes(StandardCharsets.UTF_8).length > MAX_NAME_BYTES) {
            throw new IllegalArgumentException("tensor name '" + name + "' exceeds " + MAX_NAME_BYTES + " bytes");
        }
        Objects.requireNonNull(type, "type of " + name);
        if (type == GGMLType.UNKNOWN || type.getBlockSize() <= 0 || type.getTypeSize() <= 0) {
            throw new IllegalArgumentException("tensor " + name + ": " + type + " is not a storage type");
        }
        if (ne == null || ne.length < 1 || ne.length > MAX_DIMS) {
            throw new IllegalArgumentException("tensor " + name + " needs 1 to " + MAX_DIMS + " dimensions");
        }
        for (long d : ne) {
            if (d <= 0) {
                throw new IllegalArgumentException("tensor " + name + " has a non-positive dimension " + shape(ne));
            }
        }
        if (ne[0] % type.getBlockSize() != 0) {
            throw new IllegalArgumentException("tensor " + name + ": ne[0]=" + ne[0] + " is not a multiple of the "
                    + type + " block size " + type.getBlockSize());
        }
        long bytes = type.getBytesForElements(elementCount(ne));
        toIntSize(name, bytes);
        return bytes;
    }

    private Tensor requireTensor(String name) {
        Tensor t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException("no tensor '" + name + "'");
        }
        return t;
    }

    private static long elementCount(long[] ne) {
        long n = 1;
        for (long d : ne) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    private static int toIntSize(String name, long n) {
        if (n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("tensor " + name + " is too large for an in-memory fixture: " + n);
        }
        return (int) n;
    }

    private static String shape(long[] ne) {
        return Arrays.toString(ne);
    }

    private static long alignUp(long v, int alignment) {
        return (v + alignment - 1) / alignment * alignment;
    }

    private static void writeZeros(OutputStream out, long n) throws IOException {
        for (long i = 0; i < n; i++) {
            out.write(0);
        }
    }

    private byte[] headerBytes() {
        LittleEndianSink s = new LittleEndianSink();
        s.bytes(MAGIC);
        s.u32(version);
        s.u64(tensors.size());
        s.u64(kvs.size());
        for (Map.Entry<String, Kv> e : kvs.entrySet()) {
            s.string(e.getKey());
            Kv kv = e.getValue();
            s.u32(kv.type.id());
            if (kv.type == ValueType.ARRAY) {
                writeArray(s, kv.elementType, kv.value);
            } else {
                writeScalar(s, kv.type, kv.value);
            }
        }
        int alignment = effectiveAlignment();
        long offset = 0;
        for (Tensor t : tensors.values()) {
            s.string(t.name);
            s.u32(t.ne.length);
            for (long d : t.ne) {
                s.u64(d);
            }
            s.u32(t.type.getId());
            s.u64(offset);
            offset += alignUp(t.data.length, alignment);
        }
        return s.toByteArray();
    }

    private static void writeScalar(LittleEndianSink s, ValueType type, Object v) {
        switch (type) {
            case UINT8:
            case INT8:
                s.u8(((Long) v).intValue());
                break;
            case UINT16:
            case INT16:
                s.u16(((Long) v).intValue());
                break;
            case UINT32:
                s.u32(((Long) v).intValue());
                break;
            case INT32:
                s.u32((Integer) v);
                break;
            case FLOAT32:
                s.u32(Float.floatToRawIntBits((Float) v));
                break;
            case BOOL:
                s.u8((Boolean) v ? 1 : 0);
                break;
            case STRING:
                s.string((String) v);
                break;
            case UINT64:
            case INT64:
                s.u64((Long) v);
                break;
            case FLOAT64:
                s.u64(Double.doubleToRawLongBits((Double) v));
                break;
            default:
                throw new IllegalStateException("not a scalar type: " + type);
        }
    }

    private static void writeArray(LittleEndianSink s, ValueType elementType, Object array) {
        s.u32(elementType.id());
        switch (elementType) {
            case UINT8:
            case INT8: {
                int[] a = (int[]) array;
                s.u64(a.length);
                for (int x : a) {
                    s.u8(x);
                }
                break;
            }
            case UINT16:
            case INT16: {
                int[] a = (int[]) array;
                s.u64(a.length);
                for (int x : a) {
                    s.u16(x);
                }
                break;
            }
            case UINT32:
            case UINT64:
            case INT64: {
                long[] a = (long[]) array;
                s.u64(a.length);
                for (long x : a) {
                    if (elementType == ValueType.UINT32) {
                        s.u32((int) x);
                    } else {
                        s.u64(x);
                    }
                }
                break;
            }
            case INT32: {
                int[] a = (int[]) array;
                s.u64(a.length);
                for (int x : a) {
                    s.u32(x);
                }
                break;
            }
            case FLOAT32: {
                float[] a = (float[]) array;
                s.u64(a.length);
                for (float x : a) {
                    s.u32(Float.floatToRawIntBits(x));
                }
                break;
            }
            case BOOL: {
                boolean[] a = (boolean[]) array;
                s.u64(a.length);
                for (boolean x : a) {
                    s.u8(x ? 1 : 0);
                }
                break;
            }
            case STRING: {
                String[] a = (String[]) array;
                s.u64(a.length);
                for (String x : a) {
                    s.string(x);
                }
                break;
            }
            case FLOAT64: {
                double[] a = (double[]) array;
                s.u64(a.length);
                for (double x : a) {
                    s.u64(Double.doubleToRawLongBits(x));
                }
                break;
            }
            default:
                throw new IllegalStateException("unsupported array element type " + elementType);
        }
    }

    /** Growable little-endian byte sink for the header. */
    private static final class LittleEndianSink {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        void bytes(byte[] b) {
            out.write(b, 0, b.length);
        }

        void u8(int v) {
            out.write(v & 0xFF);
        }

        void u16(int v) {
            out.write(v & 0xFF);
            out.write((v >>> 8) & 0xFF);
        }

        void u32(int v) {
            for (int i = 0; i < 4; i++) {
                out.write((v >>> (8 * i)) & 0xFF);
            }
        }

        void u64(long v) {
            for (int i = 0; i < 8; i++) {
                out.write((int) (v >>> (8 * i)) & 0xFF);
            }
        }

        void string(String str) {
            byte[] b = str.getBytes(StandardCharsets.UTF_8);
            u64(b.length);
            bytes(b);
        }

        byte[] toByteArray() {
            return out.toByteArray();
        }
    }
}
