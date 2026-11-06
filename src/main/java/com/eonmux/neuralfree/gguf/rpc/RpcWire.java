package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelArtifact;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The wire format of the remote-compute protocol: length-prefixed frames over one TCP connection, the request
 * opcodes, typed payload writers and readers, the capability, kernel-signature and kernel-argument codecs, and the
 * HMAC-SHA256 handshake both ends run.
 *
 * <p>Frames carry a big-endian 32-bit length followed by that many payload bytes; the first payload byte of a request
 * is its {@code OP_*} opcode. A response frame starts with {@link #STATUS_OK} or {@link #STATUS_ERROR}; an error frame
 * follows its status with an {@link #ERR_*} kind and a message.</p>
 *
 * <p>Version 2 adds pipelining and the weight cache: an {@link #OP_BATCH} frame carries several commands with no
 * per-command response and is answered once, after the server executed them; {@link #OP_WRITE_HASH} asks whether the
 * server's weight cache already holds a payload by its {@link #fnv1a} content hash; the write carries a cache flag;
 * and the allocation, kernel-load and free-VRAM requests carry a device index into the server's device list.</p>
 */
final class RpcWire {

    static final int MAGIC = 0x4E465250;
    static final int VERSION = 2;
    /** Bytes of the server's handshake nonce. */
    static final int NONCE_BYTES = 32;
    /** Largest frame either end accepts; writes and reads are chunked below it. */
    static final int MAX_FRAME = 64 << 20;
    /** Payload bytes one write or read request carries. */
    static final int CHUNK = 32 << 20;
    /** Payload bytes one cached write probe hashes; smaller writes skip the probe (llama.cpp HASH_THRESHOLD). */
    static final int HASH_THRESHOLD = 1 << 20;

    static final byte STATUS_OK = 0;
    static final byte STATUS_ERROR = 1;
    static final byte ERR_ILLEGAL_STATE = 0;
    static final byte ERR_UNSUPPORTED = 1;
    static final byte ERR_ILLEGAL_ARGUMENT = 2;
    /** The server's backend cannot provide a kernel ({@code UnsupportedKernelException}); optional features skip it. */
    static final byte ERR_UNSUPPORTED_KERNEL = 3;
    /** The server's device ran out of memory ({@code GpuOutOfMemoryException}). */
    static final byte ERR_OUT_OF_MEMORY = 4;

    static final byte OP_ALLOC = 1;
    static final byte OP_FREE = 2;
    static final byte OP_WRITE = 3;
    static final byte OP_READ = 4;
    static final byte OP_FILL = 5;
    static final byte OP_COPY = 6;
    static final byte OP_COMPILE = 7;
    static final byte OP_LOAD = 8;
    static final byte OP_CLONE = 9;
    static final byte OP_PREPARE = 10;
    static final byte OP_ENQUEUE = 11;
    static final byte OP_FINISH = 12;
    static final byte OP_FLUSH = 13;
    static final byte OP_ALLOCATED = 14;
    static final byte OP_FREE_VRAM = 15;
    static final byte OP_RELEASE_KERNEL = 16;
    /** One frame carrying several commands the server executes in order, answered once. */
    static final byte OP_BATCH = 17;
    /** A weight-cache probe: whether the server holds a payload by its content hash, and may write it. */
    static final byte OP_WRITE_HASH = 18;

    /** Whether {@code op} is answered per request and so cannot be a {@link #OP_BATCH} member. */
    static boolean roundTrip(byte op) {
        switch (op) {
            case OP_BATCH:
            case OP_ALLOC:
            case OP_READ:
            case OP_COMPILE:
            case OP_LOAD:
            case OP_CLONE:
            case OP_ALLOCATED:
            case OP_FREE_VRAM:
            case OP_WRITE_HASH:
                return true;
            default:
                return false;
        }
    }

    /** The wire name of an opcode, for messages. */
    static String opName(byte op) {
        switch (op) {
            case OP_ALLOC: return "ALLOC";
            case OP_FREE: return "FREE";
            case OP_WRITE: return "WRITE";
            case OP_READ: return "READ";
            case OP_FILL: return "FILL";
            case OP_COPY: return "COPY";
            case OP_COMPILE: return "COMPILE";
            case OP_LOAD: return "LOAD";
            case OP_CLONE: return "CLONE";
            case OP_PREPARE: return "PREPARE";
            case OP_ENQUEUE: return "ENQUEUE";
            case OP_FINISH: return "FINISH";
            case OP_FLUSH: return "FLUSH";
            case OP_ALLOCATED: return "ALLOCATED";
            case OP_FREE_VRAM: return "FREE_VRAM";
            case OP_RELEASE_KERNEL: return "RELEASE_KERNEL";
            case OP_BATCH: return "BATCH";
            case OP_WRITE_HASH: return "WRITE_HASH";
            default: return "OP_" + op;
        }
    }

    private static final byte ARG_BUFFER = 0;
    private static final byte ARG_INT = 1;
    private static final byte ARG_BOOL = 2;
    private static final byte ARG_FLOAT = 3;
    private static final byte ARG_LONG = 4;
    private static final byte ARG_DOUBLE = 5;
    private static final byte ARG_INT_ARRAY = 6;
    private static final byte ARG_FLOAT_ARRAY = 7;
    private static final byte ARG_LONG_ARRAY = 8;
    private static final byte ARG_DOUBLE_ARRAY = 9;
    private static final byte ARG_BYTE_ARRAY = 10;
    private static final byte ARG_SHORT_ARRAY = 11;

    private final DataInputStream in;
    private final DataOutputStream out;

    RpcWire(DataInputStream in, DataOutputStream out) {
        this.in = in;
        this.out = out;
    }

    DataInputStream input() {
        return in;
    }

    DataOutputStream output() {
        return out;
    }

    /** Writes one frame. */
    void writeFrame(byte[] payload) throws IOException {
        if (payload.length > MAX_FRAME) {
            throw new IOException("frame of " + payload.length + " bytes exceeds the " + MAX_FRAME + "-byte limit");
        }
        out.writeInt(payload.length);
        out.write(payload);
        out.flush();
    }

    /**
     * Reads one frame.
     *
     * @throws EOFException when the peer closed the connection cleanly
     * @throws IOException  for a malformed or oversized frame
     */
    byte[] readFrame() throws IOException {
        int length = in.readInt();
        if (length < 0 || length > MAX_FRAME) {
            throw new IOException("frame length " + length + " is outside 0.." + MAX_FRAME);
        }
        byte[] payload = new byte[length];
        in.readFully(payload);
        return payload;
    }

    /** The HMAC-SHA256 of {@code message} under {@code key}. */
    static byte[] hmac(byte[] key, byte[] message) {        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(key, "HmacSHA256"));
            return mac.doFinal(message);
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("HmacSHA256 is unavailable", e);
        }
    }

    /** Constant-time comparison. */
    static boolean sameMac(byte[] expected, byte[] got) {
        return got != null && MessageDigest.isEqual(expected, got);
    }

    /** The handshake message both ends MAC: the magic, the version and the server's nonce, big-endian. */
    static byte[] authMessage(byte[] nonce) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(8 + nonce.length);
        b.putInt(MAGIC).putInt(VERSION).put(nonce);
        return b.array();
    }

    /** The 64-bit FNV-1a content hash of {@code data}, the key of a cached write (llama.cpp's fnv_hash). */
    static long fnv1a(byte[] data) {
        long hash = 0xcbf29ce484222325L;
        for (byte b : data) {
            hash ^= (b & 0xff);
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    /** The 8 big-endian bytes of {@code hash}, as a probe carries it. */
    static byte[] hashBytes(long hash) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(8);
        b.putLong(hash);
        return b.array();
    }

    /** The 16 lowercase hex digits of {@code hash}, as a cache file is named. */
    static String hashName(long hash) {
        return String.format("%016x", hash);
    }

    /**
     * Appends the handshake body of {@code infos} with whether the server serves a weight cache. The frame carries
     * the device count, each device's backend type, capabilities and host-mapped support, and the cache flag.
     */
    static void writeHandshake(Writer w, List<RemoteDeviceInfo> infos, boolean cache) {
        w.i32(infos.size());
        for (RemoteDeviceInfo info : infos) {
            w.utf(info.backendType().name());
            writeCapabilities(w, info.capabilities());
            w.bool(info.hostMappedBuffers());
        }
        w.bool(cache);
    }

    /** Reads the handshake body {@link #writeHandshake} wrote; the endpoint text goes into every device info. */
    static List<RemoteDeviceInfo> readHandshake(Reader r, String endpoint) {
        int n = r.i32();
        if (n < 1 || n > 1024) {
            throw new IllegalStateException("remote device " + endpoint + " reported " + n
                    + " devices; the protocol serves 1..1024");
        }
        List<RemoteDeviceInfo> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            ComputeBackend.BackendType type = backendType(r.utf());
            DeviceCapabilities capabilities = readCapabilities(r);
            boolean hostMapped = r.bool();
            out.add(new RemoteDeviceInfo(type, capabilities, hostMapped, endpoint));
        }
        return out;
    }

    /** An append-only payload being built. */
    static final class Writer {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream(64);
        private final DataOutputStream out = new DataOutputStream(bytes);
        private final byte opcode;

        Writer(byte opcode) {
            this.opcode = opcode;
            u8(opcode);
        }

        Writer u8(int v) {
            try {
                out.writeByte(v);
            } catch (IOException e) {
                throw new IllegalStateException("in-memory write failed", e);
            }
            return this;
        }

        Writer bool(boolean v) {
            return u8(v ? 1 : 0);
        }

        Writer i32(int v) {
            try {
                out.writeInt(v);
            } catch (IOException e) {
                throw new IllegalStateException("in-memory write failed", e);
            }
            return this;
        }

        Writer i64(long v) {
            try {
                out.writeLong(v);
            } catch (IOException e) {
                throw new IllegalStateException("in-memory write failed", e);
            }
            return this;
        }

        Writer f32(float v) {
            return i32(Float.floatToRawIntBits(v));
        }

        Writer f64(double v) {
            return i64(Double.doubleToRawLongBits(v));
        }

        Writer utf(String v) {
            try {
                out.writeUTF(v);
            } catch (IOException e) {
                throw new IllegalStateException("in-memory write failed", e);
            }
            return this;
        }

        Writer nullableUtf(String v) {
            bool(v != null);
            return v == null ? this : utf(v);
        }

        Writer rawBytes(byte[] v) {
            try {
                out.writeInt(v.length);
                out.write(v);
            } catch (IOException e) {
                throw new IllegalStateException("in-memory write failed", e);
            }
            return this;
        }

        /** Appends {@code v} verbatim, without a length prefix; a batch frame's member bodies use it. */
        Writer body(byte[] v) {
            try {
                out.write(v);
            } catch (IOException e) {
                throw new IllegalStateException("in-memory write failed", e);
            }
            return this;
        }

        /** A string of any length, UTF-8 encoded (a kernel program is larger than {@code writeUTF} allows). */
        Writer rawString(String v) {
            return rawBytes(v.getBytes(StandardCharsets.UTF_8));
        }

        Writer longArray(long[] v) {
            if (v == null) {
                return i32(-1);
            }
            i32(v.length);
            for (long l : v) {
                i64(l);
            }
            return this;
        }

        Writer strs(Set<String> v) {
            List<String> sorted = new ArrayList<>(v);
            Collections.sort(sorted);
            i32(sorted.size());
            for (String s : sorted) {
                utf(s);
            }
            return this;
        }

        byte[] toBytes() {
            try {
                out.flush();
            } catch (IOException e) {
                throw new IllegalStateException("in-memory flush failed", e);
            }
            return bytes.toByteArray();
        }

        @Override
        public String toString() {
            return "Writer{op=" + opcode + "}";
        }
    }

    /** A payload being consumed. */
    static final class Reader {
        private final DataInputStream in;

        Reader(byte[] payload) {
            this.in = new DataInputStream(new ByteArrayInputStream(payload));
        }

        byte opcode() {
            return u8();
        }

        byte u8() {
            try {
                return in.readByte();
            } catch (IOException e) {
                throw new IllegalArgumentException("truncated RPC payload", e);
            }
        }

        boolean bool() {
            byte b = u8();
            if (b != 0 && b != 1) {
                throw new IllegalArgumentException("malformed boolean " + b + " in the RPC payload");
            }
            return b == 1;
        }

        int i32() {
            try {
                return in.readInt();
            } catch (IOException e) {
                throw new IllegalArgumentException("truncated RPC payload", e);
            }
        }

        long i64() {
            try {
                return in.readLong();
            } catch (IOException e) {
                throw new IllegalArgumentException("truncated RPC payload", e);
            }
        }

        float f32() {
            return Float.intBitsToFloat(i32());
        }

        double f64() {
            return Double.longBitsToDouble(i64());
        }

        String utf() {
            try {
                return in.readUTF();
            } catch (IOException e) {
                throw new IllegalArgumentException("truncated RPC payload", e);
            }
        }

        String nullableUtf() {
            return bool() ? utf() : null;
        }

        byte[] rawBytes() {
            int n = i32();
            if (n < 0 || n > MAX_FRAME) {
                throw new IllegalArgumentException("byte array of " + n + " bytes is outside 0.." + MAX_FRAME);
            }
            byte[] v = new byte[n];
            try {
                in.readFully(v);
            } catch (IOException e) {
                throw new IllegalArgumentException("truncated RPC payload", e);
            }
            return v;
        }

        /** A string of any length, UTF-8 encoded. */
        String rawString() {
            return new String(rawBytes(), StandardCharsets.UTF_8);
        }

        long[] longArray() {
            int n = i32();
            if (n < 0) {
                return null;
            }
            if (n > MAX_FRAME / 8) {
                throw new IllegalArgumentException("long array of " + n + " entries is too large");
            }
            long[] v = new long[n];
            for (int i = 0; i < n; i++) {
                v[i] = i64();
            }
            return v;
        }

        Set<String> strs() {
            int n = i32();
            if (n < 0 || n > 100000) {
                throw new IllegalArgumentException("string set of " + n + " entries is malformed");
            }
            Set<String> v = new java.util.LinkedHashSet<>();
            for (int i = 0; i < n; i++) {
                v.add(utf());
            }
            return v;
        }
    }

    // ------------------------------------------------------------------ capabilities

    static void writeCapabilities(Writer w, DeviceCapabilities c) {
        w.nullableUtf(c.getDeviceName()).i64(c.getTotalVramBytes()).i64(c.getMaxAllocSizeBytes())
                .i32(c.getComputeUnits()).i32(c.getMaxClockFrequencyMhz()).i64(c.getMaxWorkGroupSize())
                .bool(c.supportsAsyncTransfer()).nullableUtf(c.getDriverVersion())
                .bool(c.supportsFloat64()).bool(c.supportsFloat16()).bool(c.supportsSubgroups())
                .i64(c.getMaxLocalMemoryBytes())
                .strs(c.getSupportedActivationFunctions()).strs(c.getSupportedCostFunctions())
                .strs(c.getSupportedBLASLevels())
                .utf(c.getDeviceClass().name());
        DeviceIdentity id = c.getIdentity();
        w.utf(id.name()).i32(id.vendorId()).nullableUtf(id.pciBusId()).nullableUtf(id.uuid());
        w.i32(c.getSubgroupSize()).i32(c.getMinSubgroupSize()).i32(c.getMaxSubgroupSize());
        long[] groups = c.getMaxWorkGroupCount();
        for (long g : groups) {
            w.i64(g);
        }
        w.i32(c.getMaxPushConstantBytes()).i64(c.getMaxSharedMemoryBytes()).i64(c.getMaxStorageBufferRange())
                .i32(c.getComputeCapability());
        int n = c.getFeatures().size();
        w.i32(n);
        for (DeviceFeature f : c.getFeatures()) {
            w.utf(f.name());
        }
    }

    static DeviceCapabilities readCapabilities(Reader r) {
        String name = r.nullableUtf();
        long vram = r.i64();
        long maxAlloc = r.i64();
        int computeUnits = r.i32();
        int clock = r.i32();
        long maxWorkGroup = r.i64();
        boolean async = r.bool();
        String driver = r.nullableUtf();
        boolean f64 = r.bool();
        boolean f16 = r.bool();
        boolean subgroups = r.bool();
        long localMemory = r.i64();
        Set<String> activations = r.strs();
        Set<String> costs = r.strs();
        Set<String> blas = r.strs();
        DeviceClass deviceClass = DeviceClass.valueOf(r.utf());
        String identityName = r.utf();
        int vendor = r.i32();
        String pci = r.nullableUtf();
        String uuid = r.nullableUtf();
        int subgroup = r.i32();
        int minSubgroup = r.i32();
        int maxSubgroup = r.i32();
        long[] groups = new long[]{r.i64(), r.i64(), r.i64()};
        int pushConstant = r.i32();
        long sharedMemory = r.i64();
        long storageRange = r.i64();
        int capability = r.i32();
        int featureCount = r.i32();
        EnumSet<DeviceFeature> features = EnumSet.noneOf(DeviceFeature.class);
        for (int i = 0; i < featureCount; i++) {
            features.add(DeviceFeature.valueOf(r.utf()));
        }
        return new DeviceCapabilities.Builder()
                .deviceName(name)
                .totalVramBytes(vram)
                .maxAllocSizeBytes(maxAlloc)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(clock)
                .maxWorkGroupSize(maxWorkGroup)
                .supportsAsyncTransfer(async)
                .driverVersion(driver)
                .supportsFloat64(f64)
                .supportsFloat16(f16)
                .supportsSubgroups(subgroups)
                .maxLocalMemoryBytes(localMemory)
                .supportedActivationFunctions(activations)
                .supportedCostFunctions(costs)
                .supportedBLASLevels(blas)
                .deviceClass(deviceClass)
                .identity(new DeviceIdentity(identityName, vendor, pci, uuid))
                .subgroupSize(subgroup)
                .minSubgroupSize(minSubgroup)
                .maxSubgroupSize(maxSubgroup)
                .maxWorkGroupCount(groups)
                .maxPushConstantBytes(pushConstant)
                .maxSharedMemoryBytes(sharedMemory)
                .maxStorageBufferRange(storageRange)
                .computeCapability(capability)
                .features(features)
                .build();
    }

    // ------------------------------------------------------------------ kernel signatures

    static void writeSignature(Writer w, KernelSignature s) {
        boolean present = s != null;
        w.bool(present);
        if (!present) {
            return;
        }
        boolean[] ro = s.bufferReadOnly();
        w.i32(ro.length);
        for (boolean b : ro) {
            w.bool(b);
        }
        char[] kinds = s.pushKinds();
        w.i32(kinds.length);
        for (char k : kinds) {
            w.u8(k);
        }
        w.bool(s.hasLayout());
        if (s.hasLayout()) {
            boolean[] flags = s.layout();
            StringBuilder layout = new StringBuilder();
            int scalar = 0;
            for (boolean buffer : flags) {
                layout.append(buffer ? KernelSignature.LAYOUT_BUFFER : kinds[scalar++]);
            }
            w.utf(layout.toString());
        }
    }

    static KernelSignature readSignature(Reader r) {
        if (!r.bool()) {
            return null;
        }
        int buffers = r.i32();
        boolean[] ro = new boolean[buffers];
        for (int i = 0; i < buffers; i++) {
            ro[i] = r.bool();
        }
        int scalars = r.i32();
        char[] kinds = new char[scalars];
        for (int i = 0; i < scalars; i++) {
            kinds[i] = (char) r.u8();
        }
        if (r.bool()) {
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < kinds.length; i++) {
                if (i > 0) {
                    text.append(',');
                }
                text.append(kinds[i]);
            }
            String layout = r.utf();
            return KernelSignature.parseInventory(String.valueOf(buffers), String.valueOf(scalars),
                    text.toString(), layout);
        }
        return new KernelSignature(ro, kinds);
    }

    // ------------------------------------------------------------------ kernel artifacts

    static void writeArtifact(Writer w, KernelArtifact a) {
        w.utf(a.kind().name()).utf(a.kernelName()).utf(a.programKey());
        if (a.kind() == KernelArtifact.Kind.OPENCL_SOURCE) {
            w.rawString(a.sourceCode()).utf(a.buildOptions());
        }
        w.nullableUtf(a.resourcePath());
        Map<Integer, Integer> specs = a.specConstants();
        w.i32(specs.size());
        for (Map.Entry<Integer, Integer> e : specs.entrySet()) {
            w.i32(e.getKey()).i32(e.getValue());
        }
        writeSignature(w, a.signature());
    }

    static KernelArtifact readArtifact(Reader r) {
        KernelArtifact.Kind kind = KernelArtifact.Kind.valueOf(r.utf());
        String name = r.utf();
        String programKey = r.utf();
        String source = null;
        String options = null;
        if (kind == KernelArtifact.Kind.OPENCL_SOURCE) {
            source = r.rawString();
            options = r.utf();
        }
        String resource = r.nullableUtf();
        int specs = r.i32();
        Map<Integer, Integer> constants = new LinkedHashMap<>();
        for (int i = 0; i < specs; i++) {
            constants.put(r.i32(), r.i32());
        }
        KernelSignature signature = readSignature(r);
        switch (kind) {
            case OPENCL_SOURCE:
                final String text = source;
                return KernelArtifact.openClSource(name, programKey, () -> text, options);
            case CUDA_IMAGE:
                return KernelArtifact.cudaImage(name, resource);
            case HIP_IMAGE:
                return KernelArtifact.hipImage(name, resource);
            case SPIRV:
                return KernelArtifact.spirv(name, resource, new java.util.TreeMap<>(constants), signature);
            default:
                throw new IllegalArgumentException("unknown kernel artifact kind " + kind);
        }
    }

    // ------------------------------------------------------------------ kernel arguments

    static void writeArgs(Writer w, Object[] args) {
        w.i32(args.length);
        for (Object arg : args) {
            writeArg(w, arg);
        }
    }

    private static void writeArg(Writer w, Object arg) {
        if (arg == null) {
            throw new IllegalArgumentException("kernel argument is null");
        }
        if (arg instanceof GpuBuffer) {
            GpuBuffer buffer = (GpuBuffer) arg;
            if (!(buffer instanceof RemoteBuffer)) {
                throw new IllegalArgumentException("kernel argument buffer is not a remote buffer");
            }
            w.u8(ARG_BUFFER).i64(((RemoteBuffer) arg).id());
        } else if (arg instanceof Integer) {
            w.u8(ARG_INT).i32((Integer) arg);
        } else if (arg instanceof Boolean) {
            w.u8(ARG_BOOL).bool((Boolean) arg);
        } else if (arg instanceof Float) {
            w.u8(ARG_FLOAT).f32((Float) arg);
        } else if (arg instanceof Long) {
            w.u8(ARG_LONG).i64((Long) arg);
        } else if (arg instanceof Double) {
            w.u8(ARG_DOUBLE).f64((Double) arg);
        } else if (arg instanceof int[]) {
            int[] a = (int[]) arg;
            w.u8(ARG_INT_ARRAY).i32(a.length);
            for (int v : a) {
                w.i32(v);
            }
        } else if (arg instanceof float[]) {
            float[] a = (float[]) arg;
            w.u8(ARG_FLOAT_ARRAY).i32(a.length);
            for (float v : a) {
                w.f32(v);
            }
        } else if (arg instanceof long[]) {
            long[] a = (long[]) arg;
            w.u8(ARG_LONG_ARRAY).i32(a.length);
            for (long v : a) {
                w.i64(v);
            }
        } else if (arg instanceof double[]) {
            double[] a = (double[]) arg;
            w.u8(ARG_DOUBLE_ARRAY).i32(a.length);
            for (double v : a) {
                w.f64(v);
            }
        } else if (arg instanceof byte[]) {
            w.u8(ARG_BYTE_ARRAY).rawBytes((byte[]) arg);
        } else if (arg instanceof short[]) {
            short[] a = (short[]) arg;
            w.u8(ARG_SHORT_ARRAY).i32(a.length);
            for (short v : a) {
                w.i32(v);
            }
        } else {
            throw new IllegalArgumentException("unsupported kernel argument type " + arg.getClass().getName());
        }
    }

    static Object[] readArgs(Reader r, Map<Long, GpuBuffer> buffers) {
        int n = r.i32();
        if (n < 0 || n > 65536) {
            throw new IllegalArgumentException("kernel argument count " + n + " is malformed");
        }
        Object[] args = new Object[n];
        for (int i = 0; i < n; i++) {
            args[i] = readArg(r, buffers);
        }
        return args;
    }

    private static Object readArg(Reader r, Map<Long, GpuBuffer> buffers) {
        byte tag = r.u8();
        switch (tag) {
            case ARG_BUFFER:
                long id = r.i64();
                GpuBuffer buffer = buffers.get(id);
                if (buffer == null) {
                    throw new IllegalArgumentException("buffer id " + id + " was not allocated by this connection");
                }
                return buffer;
            case ARG_INT:
                return r.i32();
            case ARG_BOOL:
                return r.bool();
            case ARG_FLOAT:
                return r.f32();
            case ARG_LONG:
                return r.i64();
            case ARG_DOUBLE:
                return r.f64();
            case ARG_INT_ARRAY:
                return readIntArray(r);
            case ARG_FLOAT_ARRAY:
                return readFloatArray(r);
            case ARG_LONG_ARRAY:
                return readLongArray(r);
            case ARG_DOUBLE_ARRAY:
                return readDoubleArray(r);
            case ARG_BYTE_ARRAY:
                return r.rawBytes();
            case ARG_SHORT_ARRAY:
                return readShortArray(r);
            default:
                throw new IllegalArgumentException("unknown kernel argument tag " + tag);
        }
    }

    private static int[] readIntArray(Reader r) {
        int n = arrayLength(r, 4);
        int[] v = new int[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.i32();
        }
        return v;
    }

    private static float[] readFloatArray(Reader r) {
        int n = arrayLength(r, 4);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.f32();
        }
        return v;
    }

    private static long[] readLongArray(Reader r) {
        int n = arrayLength(r, 8);
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.i64();
        }
        return v;
    }

    private static double[] readDoubleArray(Reader r) {
        int n = arrayLength(r, 8);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.f64();
        }
        return v;
    }

    private static short[] readShortArray(Reader r) {
        int n = arrayLength(r, 2);
        short[] v = new short[n];
        for (int i = 0; i < n; i++) {
            v[i] = (short) r.i32();
        }
        return v;
    }

    private static int arrayLength(Reader r, int width) {
        int n = r.i32();
        if (n < 0 || (long) n * width > MAX_FRAME) {
            throw new IllegalArgumentException("kernel argument array of " + n + " entries is too large");
        }
        return n;
    }

    /** The backend type of a server's device, by its wire name. */
    static ComputeBackend.BackendType backendType(String name) {
        return ComputeBackend.BackendType.valueOf(name);
    }

    /** Little-endian bytes of {@code values}, as a device write of floats carries them. */
    static byte[] floatsToBytes(float[] values) {
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(values.length * Float.BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(values);
        return b.array();
    }

    /** The floats of little-endian {@code bytes}; its length must be a multiple of four. */
    static float[] bytesToFloats(byte[] bytes) {
        if ((bytes.length & 3) != 0) {
            throw new IllegalArgumentException("read of " + bytes.length + " bytes is not a whole number of floats");
        }
        float[] out = new float[bytes.length / Float.BYTES];
        java.nio.ByteBuffer b = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().get(out);
        return out;
    }

    private RpcWire() {
        throw new AssertionError("no instances");
    }
}
