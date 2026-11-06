package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufEncoder;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
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
 * <p>The bytes all come from {@link GgufEncoder}, the repository's single GGUF encoder: it writes the
 * header, the KV pairs and the tensor infos, and the writer routes its tensor data through it too, so a
 * test fixture and a {@code GgufWriter} export of the same model have the same bytes. Everything the
 * reader must reject that the encoder refuses — a GGUF version other than 3, and a malformed
 * {@code general.alignment} entry — is produced by editing those bytes around the encoder.</p>
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

    private static final int QK8_0 = 32;

    /** Scalar value or array payload (defensively copied primitive or String array). */
    private static final class Kv {
        final GGUFValueType type;
        final GGUFValueType elementType;
        final Object value;

        Kv(GGUFValueType type, GGUFValueType elementType, Object value) {
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
        if (kv == null || kv.type != GGUFValueType.STRING) {
            throw new IllegalStateException("general.architecture must be set as a string before archKey(\""
                    + suffix + "\")");
        }
        return kv.value + "." + suffix;
    }

    // ================================================================== scalar KV

    public GgufTestWriter u8(String key, int v) {
        return put(key, GGUFValueType.UINT8, null, checkRange(key, v, 0, 0xFF));
    }

    public GgufTestWriter i8(String key, int v) {
        return put(key, GGUFValueType.INT8, null, checkRange(key, v, Byte.MIN_VALUE, Byte.MAX_VALUE));
    }

    public GgufTestWriter u16(String key, int v) {
        return put(key, GGUFValueType.UINT16, null, checkRange(key, v, 0, 0xFFFF));
    }

    public GgufTestWriter i16(String key, int v) {
        return put(key, GGUFValueType.INT16, null, checkRange(key, v, Short.MIN_VALUE, Short.MAX_VALUE));
    }

    /** UINT32; accepts the full unsigned range 0..4294967295. */
    public GgufTestWriter u32(String key, long v) {
        return put(key, GGUFValueType.UINT32, null, checkRange(key, v, 0, 0xFFFFFFFFL));
    }

    public GgufTestWriter i32(String key, int v) {
        return put(key, GGUFValueType.INT32, null, v);
    }

    public GgufTestWriter f32(String key, float v) {
        return put(key, GGUFValueType.FLOAT32, null, v);
    }

    public GgufTestWriter bool(String key, boolean v) {
        return put(key, GGUFValueType.BOOL, null, v);
    }

    public GgufTestWriter str(String key, String v) {
        return put(key, GGUFValueType.STRING, null, Objects.requireNonNull(v, "value of " + key));
    }

    /** UINT64; the long's bits are written as is, so negative values encode values above 2^63. */
    public GgufTestWriter u64(String key, long v) {
        return put(key, GGUFValueType.UINT64, null, v);
    }

    public GgufTestWriter i64(String key, long v) {
        return put(key, GGUFValueType.INT64, null, v);
    }

    public GgufTestWriter f64(String key, double v) {
        return put(key, GGUFValueType.FLOAT64, null, v);
    }

    // ================================================================== array KV

    public GgufTestWriter u8Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, 0, 0xFF);
        }
        return put(key, GGUFValueType.ARRAY, GGUFValueType.UINT8, v.clone());
    }

    public GgufTestWriter i8Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, Byte.MIN_VALUE, Byte.MAX_VALUE);
        }
        return put(key, GGUFValueType.ARRAY, GGUFValueType.INT8, v.clone());
    }

    public GgufTestWriter u16Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, 0, 0xFFFF);
        }
        return put(key, GGUFValueType.ARRAY, GGUFValueType.UINT16, v.clone());
    }

    public GgufTestWriter i16Array(String key, int... v) {
        for (int x : v) {
            checkRange(key, x, Short.MIN_VALUE, Short.MAX_VALUE);
        }
        return put(key, GGUFValueType.ARRAY, GGUFValueType.INT16, v.clone());
    }

    public GgufTestWriter u32Array(String key, long... v) {
        for (long x : v) {
            checkRange(key, x, 0, 0xFFFFFFFFL);
        }
        return put(key, GGUFValueType.ARRAY, GGUFValueType.UINT32, v.clone());
    }

    public GgufTestWriter i32Array(String key, int... v) {
        return put(key, GGUFValueType.ARRAY, GGUFValueType.INT32, v.clone());
    }

    public GgufTestWriter f32Array(String key, float... v) {
        return put(key, GGUFValueType.ARRAY, GGUFValueType.FLOAT32, v.clone());
    }

    public GgufTestWriter boolArray(String key, boolean... v) {
        return put(key, GGUFValueType.ARRAY, GGUFValueType.BOOL, v.clone());
    }

    public GgufTestWriter strArray(String key, String... v) {
        for (String s : v) {
            Objects.requireNonNull(s, "element of " + key);
        }
        return put(key, GGUFValueType.ARRAY, GGUFValueType.STRING, v.clone());
    }

    public GgufTestWriter strArray(String key, List<String> v) {
        return strArray(key, v.toArray(new String[0]));
    }

    public GgufTestWriter u64Array(String key, long... v) {
        return put(key, GGUFValueType.ARRAY, GGUFValueType.UINT64, v.clone());
    }

    public GgufTestWriter i64Array(String key, long... v) {
        return put(key, GGUFValueType.ARRAY, GGUFValueType.INT64, v.clone());
    }

    public GgufTestWriter f64Array(String key, double... v) {
        return put(key, GGUFValueType.ARRAY, GGUFValueType.FLOAT64, v.clone());
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
    public GGUFValueType typeOf(String key) {
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

    /** The raw storage bytes of a tensor (copy). */
    public byte[] tensorData(String name) {
        return requireTensor(name).data.clone();
    }

    public GGMLType tensorType(String name) {
        return requireTensor(name).type;
    }

    // ================================================================== layout and output

    /** The alignment the layout uses (see the class contract). */
    public int effectiveAlignment() {
        Kv kv = kvs.get(ALIGNMENT_KEY);
        if (kv != null && kv.type == GGUFValueType.UINT32) {
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
        byte[] file = fileBytes();
        out.write(file);
        out.flush();
    }

    // ================================================================== float encoders

    /** True when {@link #encodeFloats} supports {@code type}. */
    public static boolean canEncodeFloats(GGMLType type) {
        return type == GGMLType.F32 || type == GGMLType.F16 || type == GGMLType.BF16 || type == GGMLType.Q8_0
                || type == GGMLType.F8_E4M3FN;
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
            case F8_E4M3FN: {
                byte[] out = new byte[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = (byte) FpBlockDecoders.fp32ToE4m3(values[i]);
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

    private GgufTestWriter put(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
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

    // ================================================================== encoder output

    /**
     * The whole file, produced by {@link GgufEncoder} and then patched: a version below {@link #GGUF_VERSION}
     * overwrites the four version bytes, so a malformed fixture is a raw byte edit around the encoder.
     */
    private byte[] fileBytes() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (WritableByteChannel channel = Channels.newChannel(out)) {
            GgufEncoder enc = new GgufEncoder(channel, "GGUF test fixture");
            enc.writeHeader(channel, tensors.size(), kvs.size());
            for (Map.Entry<String, Kv> e : kvs.entrySet()) {
                Kv kv = e.getValue();
                enc.writeKv(e.getKey(), kv.type, kv.elementType, kvValue(kv));
            }
            int alignment = effectiveAlignment();
            long offset = 0;
            for (Tensor t : tensors.values()) {
                GGUFTensorInfo info = new GGUFTensorInfo(t.name, t.ne, t.type, offset);
                enc.writeTensorInfo(info, offset);
                offset += alignUp(t.data.length, alignment);
            }
            enc.align(alignment);
            for (Tensor t : tensors.values()) {
                enc.writeData(t.data);
                enc.align(alignment);
            }
            enc.flush();
        }
        byte[] file = out.toByteArray();
        if (version != GgufEncoder.GGUF_VERSION) {
            file[4] = (byte) version;
            file[5] = (byte) (version >>> 8);
            file[6] = (byte) (version >>> 16);
            file[7] = (byte) (version >>> 24);
        }
        return file;
    }

    /** The value the encoder takes for a KV entry: a decoded scalar or a list of decoded scalars. */
    private Object kvValue(Kv kv) {
        if (kv.type != GGUFValueType.ARRAY) {
            return scalarValue(kv.type, kv.value);
        }
        List<Object> elements = new ArrayList<>();
        if (kv.elementType == GGUFValueType.UINT8 || kv.elementType == GGUFValueType.INT8
                || kv.elementType == GGUFValueType.UINT16 || kv.elementType == GGUFValueType.INT16
                || kv.elementType == GGUFValueType.INT32) {
            for (int x : (int[]) kv.value) {
                elements.add(x);
            }
        } else if (kv.elementType == GGUFValueType.UINT32) {
            // UINT32 keeps its bit pattern, so values of 2^31 and above arrive negative
            for (long x : (long[]) kv.value) {
                elements.add((int) x);
            }
        } else if (kv.elementType == GGUFValueType.UINT64 || kv.elementType == GGUFValueType.INT64) {
            for (long x : (long[]) kv.value) {
                elements.add(x);
            }
        } else if (kv.elementType == GGUFValueType.FLOAT32) {
            for (float x : (float[]) kv.value) {
                elements.add(x);
            }
        } else if (kv.elementType == GGUFValueType.FLOAT64) {
            for (double x : (double[]) kv.value) {
                elements.add(x);
            }
        } else if (kv.elementType == GGUFValueType.BOOL) {
            for (boolean x : (boolean[]) kv.value) {
                elements.add(x);
            }
        } else {
            Collections.addAll(elements, (String[]) kv.value);
        }
        return elements;
    }

    /**
     * A decoded scalar the encoder takes: 8/16/32-bit integers keep their value in an Integer, 64-bit
     * integers a Long. The writer stores ranges as JVM longs, which the 32-bit types narrow here.
     */
    private static Object scalarValue(GGUFValueType type, Object v) {
        if (v instanceof Long && type.isInteger()
                && type != GGUFValueType.UINT64 && type != GGUFValueType.INT64) {
            return ((Long) v).intValue();
        }
        return v;
    }

    /** Unpadded byte length of the header (magic, version, counts, KV pairs and tensor infos). */
    private byte[] headerBytes() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (WritableByteChannel channel = Channels.newChannel(out)) {
            GgufEncoder enc = new GgufEncoder(channel, "GGUF test fixture header");
            enc.writeHeader(channel, tensors.size(), kvs.size());
            for (Map.Entry<String, Kv> e : kvs.entrySet()) {
                Kv kv = e.getValue();
                enc.writeKv(e.getKey(), kv.type, kv.elementType, kvValue(kv));
            }
            long offset = 0;
            int alignment = effectiveAlignment();
            for (Tensor t : tensors.values()) {
                enc.writeTensorInfo(new GGUFTensorInfo(t.name, t.ne, t.type, offset), offset);
                offset += alignUp(t.data.length, alignment);
            }
            enc.flush();
        } catch (IOException e) {
            throw new IllegalStateException("in-memory GGUF header serialisation failed", e);
        }
        byte[] header = out.toByteArray();
        if (version != GgufEncoder.GGUF_VERSION) {
            header[4] = (byte) version;
            header[5] = (byte) (version >>> 8);
            header[6] = (byte) (version >>> 16);
            header[7] = (byte) (version >>> 24);
        }
        return header;
    }
}
