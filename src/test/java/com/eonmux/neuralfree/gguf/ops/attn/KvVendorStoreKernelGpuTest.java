package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvVendorCodec;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceKvFormats;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertTrue;

/**
 * The vendor store kernels ({@code LLM_KV_Store_<format>} of kernels/llm/kv_cache.cl, the GLSL twin
 * shaders/llm/kv_store.comp) against the host stores of {@link KvVendorCodec} on every GPU API with a device: the same
 * rows go through both, and the device's packed codes and side planes are decoded with the host decoder and compared
 * with the host store's own decode. Every format of the vendor families is covered, including the DeepSeek MLA latent
 * rows whose scale planes are float tiles.
 */
@RunWith(Parameterized.class)
public class KvVendorStoreKernelGpuTest {

    /** Cache rows of the fixture; the stored tokens land on cells 0, 2 and 5. */
    private static final int ROWS = 8;
    private static final int[] CELLS = {0, 2, 5};
    private static final int HEADS = 2;
    private static final int HEAD_ELEMS = 32;

    private static final List<KvDType> FORMATS = Arrays.asList(
            KvDType.NVFP4, KvDType.NVFP4_4OVER6, KvDType.TQ_K8_KEYS, KvDType.TQ_4BIT_KEYS, KvDType.TQ_3BIT_KEYS,
            KvDType.TQ_4BIT_VALUES, KvDType.TQ_3BIT_VALUES, KvDType.FP8_DS_MLA, KvDType.NVFP4_DS_MLA);

    @Parameterized.Parameters(name = "{0}")
    public static Collection<Object[]> apis() {
        List<Object[]> rows = new ArrayList<>();
        for (ComputeBackend.BackendType t : EnumSet.of(ComputeBackend.BackendType.OPENCL,
                ComputeBackend.BackendType.CUDA, ComputeBackend.BackendType.VULKAN)) {
            rows.add(new Object[]{t});
        }
        return rows;
    }

    private final ComputeBackend.BackendType api;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;

    public KvVendorStoreKernelGpuTest(ComputeBackend.BackendType api) {
        this.api = api;
    }

    @Before
    public void open() {
        switch (api) {
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                break;
            case CUDA:
                GpuTestUtil.assumeCudaAvailable();
                break;
            default:
                GpuTestUtil.assumeVulkanAvailable();
                break;
        }
        hw = new HWAcceleration(EnumSet.of(api));
        HardwareDevice device = firstDevice();
        Assume.assumeTrue("no " + api + " device enumerated", device != null);
        backend = hw.createBackend(device);
        Availability a = LlmKernels.availability(backend);
        Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + a.reason(), a.available());
        kernels = LlmKernels.open(backend);
    }

    @After
    public void close() {
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void everyVendorFormatStoresWhatTheHostStoreStores() {
        for (KvDType type : FORMATS) {
            storeMatches(type);
        }
    }

    private void storeMatches(KvDType type) {
        boolean latent = type.latentFormat();
        int heads = latent ? 1 : HEADS;
        int headElems = latent ? KvDType.DS_MLA_ROW_VALUES : HEAD_ELEMS;
        int rowElems = heads * headElems;
        KvLayout layout = latent
                ? KvLayout.latent(ROWS, false, rowElems, KvDType.DS_MLA_NOPE_VALUES, type, KvWindow.FULL_CAUSAL)
                        .withHeads(1)
                : KvLayout.of(ROWS, false, rowElems, rowElems, type, type, KvWindow.FULL_CAUSAL).withHeads(heads);
        int tokens = CELLS.length;
        Random rnd = new Random(11 + type.ordinal());
        float[] src = new float[tokens * rowElems];
        for (int i = 0; i < src.length; i++) {
            src[i] = rnd.nextFloat() * 2f - 1f;
        }
        int rowBytes = (int) type.rowBytes(rowElems);
        int planes = DeviceKvFormats.kPlaneRowBytes(layout);
        int sideStride = DeviceKvFormats.sideRowBytes(layout);
        long sideBytes = DeviceKvFormats.sideBytes(layout);
        float[] scales = new float[2 * heads];
        Arrays.fill(scales, 1f);
        byte[] side = new byte[(int) sideBytes];
        header(side, scales, layout);

        List<GpuBuffer> bufs = new ArrayList<>();
        try {
            GpuBuffer dsrc = upload(bufs, src);
            GpuBuffer dcell = upload(bufs, CELLS);
            GpuBuffer dscale = upload(bufs, ones(2 * heads));
            GpuBuffer ddst = zeros(bufs, (long) ROWS * rowBytes);
            GpuBuffer dside = bytes(bufs, side);
            Object[] args = RopeKernels.kvStoreArgs(type, tokens, rowElems, rowElems, 0, rowElems, headElems,
                    dsrc, dcell, dscale, ddst, null, dside, sideStride, 0, 0);
            GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(RopeKernels.kvStore(type)),
                    new long[]{tokens * heads}, new long[]{1}, args);
            if (e != null) {
                e.close();
            }
            backend.computeQueue().finish();
            byte[] devRows = new byte[ROWS * rowBytes];
            close(backend.computeQueue().enqueueReadBuffer(ddst, 0, devRows.length, devRows));
            byte[] devSide = new byte[(int) sideBytes];
            close(backend.computeQueue().enqueueReadBuffer(dside, 0, devSide.length, devSide));
            backend.computeQueue().finish();

            byte[] cpuRow = new byte[rowBytes];
            byte[] cpuSide = new byte[planes];
            for (int t = 0; t < tokens; t++) {
                int cell = CELLS[t];
                float[] want = reference(type, src, t * rowElems, cpuRow, cpuSide, headElems, heads);
                float[] got = decode(type, devRows, cell * rowBytes, devSide,
                        (int) DeviceKvFormats.sidePlaneBase(heads) + cell * sideStride, rowElems, headElems, heads);
                for (int i = 0; i < want.length; i++) {
                    assertTrue(api + " " + type + " row " + cell + " value " + i + " is not finite",
                            Float.isFinite(got[i]));
                    assertTrue(api + " " + type + " row " + cell + " value " + i + ": host " + want[i] + ", device "
                                    + got[i] + " (the device store wrote a different value)",
                            Math.abs(want[i] - got[i]) <= 1e-6f + 1e-5f * Math.abs(want[i]));
                }
            }
        } finally {
            for (GpuBuffer b : bufs) {
                b.close();
            }
        }
    }

    /** The host store of one row, decoded back to its values. */
    private static float[] reference(KvDType type, float[] src, int off, byte[] row, byte[] side, int headElems,
                                     int heads) {
        int rowElems = heads * headElems;
        float[] want = new float[rowElems];
        if (type == KvDType.FP8_DS_MLA) {
            float[] tile = new float[4];
            KvVendorCodec.fp8DsMlaStore(src, off, row, 0, tile, 0);
            putFloats(side, 0, tile);
            KvVendorCodec.fp8DsMlaLoad(row, 0, tile, 0, want, 0);
            return want;
        }
        if (type == KvDType.NVFP4_DS_MLA) {
            float[] sf = new float[32];
            KvVendorCodec.nvfp4DsMlaStore(src, off, row, 0, sf, 0);
            for (int g = 0; g < 32; g++) {
                side[g] = (byte) KvVendorCodec.e4m3Code(sf[g]);
            }
            float[] back = new float[32];
            for (int g = 0; g < 32; g++) {
                back[g] = KvVendorCodec.e4m3Value(side[g]);
            }
            KvVendorCodec.nvfp4DsMlaLoad(row, 0, back, 0, want, 0);
            return want;
        }
        if (type.nvfp4()) {
            boolean four = type == KvDType.NVFP4_4OVER6;
            int groups = headElems / KvVendorCodec.NVFP4_GROUP;
            for (int h = 0; h < heads; h++) {
                for (int g = 0; g < groups; g++) {
                    int at = off + h * headElems + g * KvVendorCodec.NVFP4_GROUP;
                    float sf = KvVendorCodec.nvfp4GroupScale(src, at, KvVendorCodec.NVFP4_GROUP, four);
                    side[h * groups + g] = (byte) KvVendorCodec.e4m3Code(sf);
                    KvVendorCodec.nvfp4Pack(src, at, KvVendorCodec.NVFP4_GROUP, sf, row,
                            h * (headElems / 2) + g * (KvVendorCodec.NVFP4_GROUP / 2));
                }
            }
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < headElems; i++) {
                    float sf = KvVendorCodec.e4m3Value(side[h * groups + i / KvVendorCodec.NVFP4_GROUP] & 0xFF);
                    want[h * headElems + i] = KvVendorCodec.e2m1Value(
                            KvVendorCodec.nvfp4Code(row, h * (headElems / 2), i)) * sf;
                }
            }
            return want;
        }
        if (type.turboQuantValue()) {
            boolean three = type == KvDType.TQ_3BIT_VALUES;
            int bits = three ? 3 : 4;
            int[] codes = new int[headElems];
            float[] scale = new float[1];
            float[] zero = new float[1];
            byte[] packed = new byte[codeBytes(headElems, bits)];
            for (int h = 0; h < heads; h++) {
                KvVendorCodec.turboQuantValueCodes(src, off + h * headElems, headElems, three ? 7 : 15, codes, scale,
                        zero);
                putHalf(side, 2 * h, scale[0]);
                putHalf(side, 2 * heads + 2 * h, zero[0]);
                Arrays.fill(packed, (byte) 0);
                KvVendorCodec.turboQuantPack(codes, headElems, bits, packed, 0);
                System.arraycopy(packed, 0, row, h * codeBytes(headElems, bits), packed.length);
            }
            for (int h = 0; h < heads; h++) {
                float s = half(side, 2 * h);
                float z = half(side, 2 * heads + 2 * h);
                for (int i = 0; i < headElems; i++) {
                    want[h * headElems + i] = KvVendorCodec.turboQuantCode(row, h * codeBytes(headElems, bits), i, bits)
                            * s + z;
                }
            }
            return want;
        }
        int bits = type == KvDType.TQ_3BIT_KEYS ? 3 : 4;
        float[] centroids = KvVendorCodec.centroids(headElems, bits);
        if (type == KvDType.TQ_K8_KEYS) {
            for (int i = 0; i < rowElems; i++) {
                row[i] = (byte) KvVendorCodec.e4m3Code(src[off + i]);
                want[i] = KvVendorCodec.e4m3Value(row[i] & 0xFF);
            }
            return want;
        }
        for (int h = 0; h < heads; h++) {
            float[] x = Arrays.copyOfRange(src, off + h * headElems, off + (h + 1) * headElems);
            float norm = 0f;
            for (float v : x) {
                norm += v * v;
            }
            norm = (float) Math.sqrt(norm);
            float[] rot = new float[headElems];
            float[] unit = new float[headElems];
            for (int i = 0; i < headElems; i++) {
                unit[i] = x[i] / (norm + 1e-8f);
            }
            KvVendorCodec.hadamard(unit, 0, headElems, rot, 0);
            putHalf(side, 2 * h, norm);
            int[] codes = new int[headElems];
            for (int i = 0; i < headElems; i++) {
                codes[i] = KvVendorCodec.centroidIndex(rot[i], KvVendorCodec.midpoints(headElems, bits));
            }
            byte[] packed = new byte[codeBytes(headElems, bits)];
            KvVendorCodec.turboQuantPack(codes, headElems, bits, packed, 0);
            System.arraycopy(packed, 0, row, h * codeBytes(headElems, bits), packed.length);
        }
        float[] indices = new float[rowElems];
        for (int h = 0; h < heads; h++) {
            float vecNorm = half(side, 2 * h);
            int[] idx = new int[headElems];
            for (int i = 0; i < headElems; i++) {
                idx[i] = KvVendorCodec.turboQuantCode(row, h * codeBytes(headElems, bits), i, bits);
            }
            float[] c = new float[headElems];
            float[] n = new float[1];
            KvVendorCodec.centroidsWithNorm(idx, headElems, centroids, true, c, 0, n);
            float[] back = new float[headElems];
            KvVendorCodec.hadamard(c, 0, headElems, back, 0);
            for (int i = 0; i < headElems; i++) {
                want[h * headElems + i] = back[i] * vecNorm;
            }
        }
        assertTrue("unused", indices.length > 0);
        return want;
    }

    /** Decodes one stored device row with the host decoder of the layout's fields. */
    private static float[] decode(KvDType type, byte[] rows, int rowOff, byte[] side, int sideOff, int rowElems,
                                  int headElems, int heads) {
        byte[] row = Arrays.copyOfRange(rows, rowOff, rowOff + (int) type.rowBytes(rowElems));
        float[] want = new float[rowElems];
        if (type == KvDType.FP8_DS_MLA) {
            float[] tile = new float[4];
            ByteBuffer bb = ByteBuffer.wrap(side).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < 4; i++) {
                tile[i] = bb.getFloat(sideOff + 4 * i);
            }
            KvVendorCodec.fp8DsMlaLoad(row, 0, tile, 0, want, 0);
            return want;
        }
        if (type == KvDType.NVFP4_DS_MLA) {
            float[] sf = new float[32];
            for (int g = 0; g < 32; g++) {
                sf[g] = KvVendorCodec.e4m3Value(side[sideOff + g] & 0xFF);
            }
            KvVendorCodec.nvfp4DsMlaLoad(row, 0, sf, 0, want, 0);
            return want;
        }
        if (type.nvfp4()) {
            int groups = headElems / KvVendorCodec.NVFP4_GROUP;
            for (int h = 0; h < heads; h++) {
                for (int i = 0; i < headElems; i++) {
                    float sf = KvVendorCodec.e4m3Value(side[sideOff + h * groups + i / KvVendorCodec.NVFP4_GROUP] & 0xFF);
                    want[h * headElems + i] = KvVendorCodec.e2m1Value(
                            KvVendorCodec.nvfp4Code(row, h * (headElems / 2), i)) * sf;
                }
            }
            return want;
        }
        if (type.turboQuantValue()) {
            int bits = type == KvDType.TQ_3BIT_VALUES ? 3 : 4;
            for (int h = 0; h < heads; h++) {
                float s = half(side, sideOff + 2 * h);
                float z = half(side, sideOff + 2 * heads + 2 * h);
                for (int i = 0; i < headElems; i++) {
                    want[h * headElems + i] =
                            KvVendorCodec.turboQuantCode(row, h * codeBytes(headElems, bits), i, bits) * s + z;
                }
            }
            return want;
        }
        if (type == KvDType.TQ_K8_KEYS) {
            for (int i = 0; i < rowElems; i++) {
                want[i] = KvVendorCodec.e4m3Value(row[i] & 0xFF);
            }
            return want;
        }
        int bits = type == KvDType.TQ_3BIT_KEYS ? 3 : 4;
        float[] centroids = KvVendorCodec.centroids(headElems, bits);
        for (int h = 0; h < heads; h++) {
            float norm = half(side, sideOff + 2 * h);
            int[] idx = new int[headElems];
            for (int i = 0; i < headElems; i++) {
                idx[i] = KvVendorCodec.turboQuantCode(row, h * codeBytes(headElems, bits), i, bits);
            }
            float[] c = new float[headElems];
            float[] n = new float[1];
            KvVendorCodec.centroidsWithNorm(idx, headElems, centroids, true, c, 0, n);
            float[] back = new float[headElems];
            KvVendorCodec.hadamard(c, 0, headElems, back, 0);
            for (int i = 0; i < headElems; i++) {
                want[h * headElems + i] = back[i] * norm;
            }
        }
        return want;
    }

    private static int codeBytes(int n, int bits) {
        return bits == 4 ? n / 2 : n / 8 * 3;
    }

    private static void header(byte[] side, float[] scales, KvLayout layout) {
        int floats = 2 * layout.nHeadKv() + DeviceKvFormats.SIDE_CTR_FLOATS;
        ByteBuffer bb = ByteBuffer.wrap(side).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < scales.length; i++) {
            bb.putFloat(4 * i, scales[i]);
        }
        int bits = layout.dtypeK() == KvDType.TQ_3BIT_KEYS ? 3 : 4;
        if (layout.nHeadKv() > 0 && (layout.dtypeK() == KvDType.TQ_4BIT_KEYS
                || layout.dtypeK() == KvDType.TQ_3BIT_KEYS)) {
            float[] table = KvVendorCodec.centroids(layout.widthK() / layout.nHeadKv(), bits);
            for (int i = 0; i < table.length; i++) {
                bb.putFloat(4 * (2 * layout.nHeadKv() + i), table[i]);
            }
        }
        assertTrue("header fits", floats * 4 <= side.length);
    }

    private static void putFloats(byte[] dst, int off, float[] values) {
        ByteBuffer bb = ByteBuffer.wrap(dst).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) {
            bb.putFloat(off + 4 * i, values[i]);
        }
    }

    private static void putHalf(byte[] dst, int off, float v) {
        int h = f16(v);
        dst[off] = (byte) h;
        dst[off + 1] = (byte) (h >> 8);
    }

    private static float half(byte[] src, int off) {
        return f32((short) ((src[off] & 0xFF) | ((src[off + 1] & 0xFF) << 8)));
    }

    private static float[] ones(int n) {
        float[] a = new float[n];
        Arrays.fill(a, 1f);
        return a;
    }

    private HardwareDevice firstDevice() {
        switch (api) {
            case OPENCL:
                return hw.getCLDevices().isEmpty() ? null : hw.getCLDevices().get(0);
            case CUDA:
                return hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0);
            default:
                return hw.getVkDevices().isEmpty() ? null : hw.getVkDevices().get(0);
        }
    }

    private GpuBuffer alloc(List<GpuBuffer> bufs, long bytes) {
        GpuBuffer b = backend.allocateBuffer(bytes, true);
        bufs.add(b);
        return b;
    }

    private GpuBuffer upload(List<GpuBuffer> bufs, float[] data) {
        GpuBuffer b = alloc(bufs, (long) data.length * 4);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
        return b;
    }

    private GpuBuffer upload(List<GpuBuffer> bufs, int[] data) {
        byte[] bytes = new byte[data.length * 4];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < data.length; i++) {
            bb.putInt(4 * i, data[i]);
        }
        return bytes(bufs, bytes);
    }

    private GpuBuffer bytes(List<GpuBuffer> bufs, byte[] data) {
        GpuBuffer b = alloc(bufs, Math.max(4, data.length));
        if (data.length > 0) {
            close(backend.computeQueue().enqueueWriteBuffer(b, 0, data.length, data));
        }
        return b;
    }

    private GpuBuffer zeros(List<GpuBuffer> bufs, long bytes) {
        return bytes(bufs, new byte[(int) bytes]);
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    private static short f16(float f) {
        int bits = Float.floatToIntBits(f);
        int sign = (bits >>> 16) & 0x8000;
        int mag = bits & 0x7fffffff;
        int exp = (mag >>> 23) - 127 + 15;
        if (exp <= 0) {
            return (short) sign;
        }
        int mant = (mag & 0x7fffff) >>> 13;
        int rest = mag & 0x1fff;
        if (rest > 0x1000 || (rest == 0x1000 && (mant & 1) == 1)) {
            mant++;
            if (mant == 0x400) {
                mant = 0;
                exp++;
            }
        }
        return (short) (sign | (exp << 10) | mant);
    }

    private static float f32(short h) {
        int sign = (h & 0x8000) << 16;
        int exp = (h >>> 10) & 0x1f;
        int mant = h & 0x3ff;
        if (exp == 0) {
            return Float.intBitsToFloat(sign);
        }
        return Float.intBitsToFloat(sign | ((exp - 15 + 127) << 23) | (mant << 13));
    }
}
