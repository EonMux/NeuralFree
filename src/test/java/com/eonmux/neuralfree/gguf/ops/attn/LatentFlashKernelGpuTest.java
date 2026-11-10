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
import com.eonmux.neuralfree.gguf.memory.KvVendorCodec;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek MLA latent row formats on a real device: {@code LLM_KV_Store_<latent>} writes the row bytes and side
 * planes of {@link KvVendorCodec}, and {@code LLM_FlashDecode_<latent>} over the stored latent rows (whose V side is
 * the K row) reproduces {@link CpuAttentionCore} on the decoded rows.
 *
 * <p>Every test repeats over the machine's OpenCL, CUDA and Vulkan devices.</p>
 */
@RunWith(Parameterized.class)
public class LatentFlashKernelGpuTest {

    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 1;
    private static final int D_Q = KvDType.DS_MLA_ROW_VALUES;
    private static final int D_V = KvDType.DS_MLA_NOPE_VALUES;
    private static final int N_KV = 5;
    private static final int SIDE_PLANE_BASE = 4 * (2 * N_HEAD_KV + 16);

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

    public LatentFlashKernelGpuTest(ComputeBackend.BackendType api) {
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

    private HardwareDevice firstDevice() {
        switch (api) {
            case OPENCL:
                return hw.getCLDevices().isEmpty() ? null : hw.getCLDevices().get(0);
            case CUDA:
                return hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0);
            default:
                return VkDeviceRanking.fastestDevice(hw.getVkDevices(), null);
        }
    }

    @Test
    public void fp8LatentStoreAndDecodeMatchTheCpuCodec() {
        runCase(KvDType.FP8_DS_MLA, Math.toIntExact(KvDType.FP8_DS_MLA.rowBytes(D_Q)));
    }

    @Test
    public void nvfp4LatentStoreAndDecodeMatchTheCpuCodec() {
        runCase(KvDType.NVFP4_DS_MLA, Math.toIntExact(KvDType.NVFP4_DS_MLA.rowBytes(D_Q)));
    }

    private void runCase(KvDType type, int rowBytes) {
        Random rnd = new Random(11);
        float[] rows = new float[N_KV * D_Q];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = rnd.nextFloat() - 0.5f;
        }
        int[] cells = new int[N_KV];
        for (int i = 0; i < N_KV; i++) {
            cells[i] = i;
        }
        int[] kvPos = new int[N_KV];
        for (int i = 0; i < N_KV; i++) {
            kvPos[i] = i;
        }
        float[] q = new float[N_HEAD * D_Q];
        for (int i = 0; i < q.length; i++) {
            q[i] = rnd.nextFloat() - 0.5f;
        }
        float scale = (float) (1.0 / Math.sqrt(D_Q));
        int sideBytes = SIDE_PLANE_BASE + N_KV * sideRowBytes(type);
        byte[] sideHeader = new byte[sideBytes];
        byte[] expectedRows = new byte[N_KV * rowBytes];
        byte[] expectedSide = new byte[sideBytes];
        for (int i = 0; i < 2 * N_HEAD_KV; i++) {
            putFloat(sideHeader, 4 * i, 1.0f);
        }
        System.arraycopy(sideHeader, 0, expectedSide, 0, SIDE_PLANE_BASE);
        for (int row = 0; row < N_KV; row++) {
            storeCpu(type, rows, row * D_Q, expectedRows, row * rowBytes, expectedSide,
                    SIDE_PLANE_BASE + row * sideRowBytes(type));
        }

        List<GpuBuffer> bufs = new ArrayList<>();
        try {
            GpuBuffer src = upload(bufs, rows);
            GpuBuffer cellIdx = upload(bufs, cells);
            GpuBuffer kvScale = upload(bufs, new float[]{1f});
            GpuBuffer dst = alloc(bufs, (long) N_KV * rowBytes);
            GpuBuffer side = alloc(bufs, sideBytes);
            write(side, sideHeader);
            dispatch(RopeKernels.kvStore(type), new long[]{N_KV}, new long[]{1},
                    RopeKernels.kvStoreArgs(type, N_KV, D_Q, D_Q, 0, D_Q, D_Q, src, cellIdx, kvScale,
                            dst, null, side, sideRowBytes(type), 0, 0));
            byte[] gotRows = readBytes(dst, expectedRows.length);
            byte[] gotSide = readBytes(side, sideBytes);
            assertEquals("row bytes of " + type, hex(expectedRows), hex(gotRows));
            assertEquals("side planes of " + type, hex(expectedSide), hex(gotSide));

            // The reference reads the device's own bytes, so a storing difference shows up as a decode difference.
            float[] kDec = new float[N_KV * D_Q];
            float[] vDec = new float[N_KV * D_V];
            float[] scales = new float[4];
            for (int t = 0; t < 4; t++) {
                scales[t] = getFloat(gotSide, SIDE_PLANE_BASE + 4 * t);
            }
            for (int row = 0; row < N_KV; row++) {
                loadCpu(type, gotRows, row * rowBytes, scales, gotSide,
                        SIDE_PLANE_BASE + row * sideRowBytes(type), kDec, row * D_Q, vDec, row * D_V);
            }
            CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(N_HEAD).scale(scale)
                    .window(KvWindow.FULL_CAUSAL).build();
            float[] want = new float[N_HEAD * D_V];
            CpuAttentionCore.attendRows(params, 1, q, 0, N_HEAD * D_Q, new int[]{N_KV - 1}, 0,
                    CpuAttentionCore.DenseKv.of(kDec, 0, D_Q, D_Q, vDec, 0, D_V, D_V, N_HEAD_KV, D_Q, D_V, kvPos, 0),
                    0, N_KV, want, 0, N_HEAD * D_V);

            FlashAttnTiling tiling = FlashAttnTiling.decode(N_HEAD, N_HEAD_KV, D_Q, D_V, N_KV,
                    kernels.faSmemFloats(), Math.max(1, backend.capabilities().getComputeUnits()),
                    FlashAttnTiling.decodeFastPath(kernels.decodeWarpPath(),
                            type, type, D_Q, D_V));
            Assume.assumeTrue("no decode plan for this shape: " + tiling,
                    tiling.plan() == FlashAttnTiling.Plan.DECODE);
            GpuBuffer dq = upload(bufs, q);
            GpuBuffer dkvPos = upload(bufs, kvPos);
            GpuBuffer dsinks = upload(bufs, new float[1]);
            GpuBuffer drel = upload(bufs, new float[1]);
            GpuBuffer dout = alloc(bufs, (long) want.length * 4);
            GpuBuffer dpartial = alloc(bufs, 4096);
            GpuBuffer dmask = upload(bufs, new long[N_KV]);
            Object[] args = decodeArgs(tiling, dq, dst, dkvPos, dsinks, drel, dout, dpartial, side, dmask);
            dispatch(AttnKernels.flashDecode(type), tiling.decodeGlobal(1), FlashAttnTiling.decodeLocal(), args);
            float[] got = readFloats(dout, want.length);
            for (int i = 0; i < want.length; i++) {
                assertTrue(type + " output " + i + " is not finite: " + got[i], Float.isFinite(got[i]));
                assertEquals(type + " decode output " + i, want[i], got[i], 1e-4f);
            }

            FlashAttnTiling pt = FlashAttnTiling.prefill(N_HEAD, N_HEAD_KV, D_Q, D_V, N_KV,
                    kernels.faSmemFloats(), Math.max(1, backend.capabilities().getComputeUnits()));
            Assume.assumeTrue("no tiled prefill plan for this shape: " + pt,
                    pt.plan() == FlashAttnTiling.Plan.TILED);
            float[] qRows = new float[N_KV * N_HEAD * D_Q];
            for (int i = 0; i < qRows.length; i++) {
                qRows[i] = rnd.nextFloat() - 0.5f;
            }
            float[] wantRows = new float[N_KV * N_HEAD * D_V];
            CpuAttentionCore.attendRows(params, N_KV, qRows, 0, N_HEAD * D_Q, kvPos, 0,
                    CpuAttentionCore.DenseKv.of(kDec, 0, D_Q, D_Q, vDec, 0, D_V, D_V, N_HEAD_KV, D_Q, D_V, kvPos, 0),
                    0, N_KV, wantRows, 0, N_HEAD * D_V);
            GpuBuffer dqRows = upload(bufs, qRows);
            GpuBuffer dqPos = upload(bufs, kvPos);
            GpuBuffer doutRows = alloc(bufs, (long) wantRows.length * 4);
            Object[] prefill = new Object[AttnKernels.FA_ARG_COUNT];
            prefill[AttnKernels.ARG_FA_N_TOKENS] = N_KV;
            prefill[AttnKernels.ARG_FA_N_HEAD] = N_HEAD;
            prefill[AttnKernels.ARG_FA_N_HEAD_KV] = N_HEAD_KV;
            prefill[AttnKernels.ARG_FA_D_K] = D_Q;
            prefill[AttnKernels.ARG_FA_D_V] = D_V;
            prefill[AttnKernels.ARG_FA_BQ] = pt.bq();
            prefill[AttnKernels.ARG_FA_BK] = pt.bk();
            prefill[AttnKernels.ARG_FA_G_TILE] = pt.gTile();
            prefill[AttnKernels.ARG_FA_N_SPLIT] = 1;
            prefill[AttnKernels.ARG_FA_SPLIT_LEN] = pt.bk();
            prefill[AttnKernels.ARG_FA_KV_START] = 0;
            prefill[AttnKernels.ARG_FA_KV_END] = N_KV;
            prefill[AttnKernels.ARG_FA_Q_ROW_STRIDE] = N_HEAD * D_Q;
            prefill[AttnKernels.ARG_FA_OUT_ROW_STRIDE] = N_HEAD * D_V;
            prefill[AttnKernels.ARG_FA_K_ROW_STRIDE] = D_Q;
            prefill[AttnKernels.ARG_FA_K_HEAD_STRIDE] = D_Q;
            prefill[AttnKernels.ARG_FA_V_ROW_STRIDE] = D_Q;
            prefill[AttnKernels.ARG_FA_V_HEAD_STRIDE] = D_V;
            prefill[AttnKernels.ARG_FA_SCALE] = scale;
            prefill[AttnKernels.ARG_FA_SOFTCAP] = 0f;
            prefill[AttnKernels.ARG_FA_MAX_BIAS] = 0f;
            prefill[AttnKernels.ARG_FA_FLAGS] = AttnKernels.attnFlags(true, false, true) | AttnKernels.FLAG_V_ALIAS;
            prefill[AttnKernels.ARG_FA_SWA_TYPE] = KvWindow.FULL_CAUSAL.type().code();
            prefill[AttnKernels.ARG_FA_N_SWA] = 0;
            prefill[AttnKernels.ARG_FA_BIAS_MODE] = 0;
            prefill[AttnKernels.ARG_FA_N_BUCKETS] = 0;
            prefill[AttnKernels.ARG_FA_MAX_DISTANCE] = 0;
            prefill[AttnKernels.ARG_FA_Q_OFF] = 0;
            prefill[AttnKernels.ARG_FA_OUT_OFF] = 0;
            prefill[AttnKernels.ARG_FA_BUF_Q] = dqRows;
            prefill[AttnKernels.ARG_FA_BUF_K] = dst;
            prefill[AttnKernels.ARG_FA_BUF_V] = dst;
            prefill[AttnKernels.ARG_FA_BUF_Q_POS] = dqPos;
            prefill[AttnKernels.ARG_FA_BUF_KV_POS] = dkvPos;
            prefill[AttnKernels.ARG_FA_BUF_SINKS] = dsinks;
            prefill[AttnKernels.ARG_FA_BUF_REL_BIAS] = drel;
            prefill[AttnKernels.ARG_FA_BUF_OUT] = doutRows;
            prefill[AttnKernels.ARG_FA_BUF_PARTIAL] = dpartial;
            prefill[AttnKernels.ARG_FA_BUF_KV_SCALES] = side;
            prefill[AttnKernels.ARG_FA_BUF_TREE_MASK] = dmask;
            dispatch(AttnKernels.flashAttn(type), pt.prefillGlobal(N_KV, 1), FlashAttnTiling.prefillLocal(),
                    prefill);
            float[] gotRowsOut = readFloats(doutRows, wantRows.length);
            for (int i = 0; i < wantRows.length; i++) {
                assertTrue(type + " prefill output " + i + " is not finite: " + gotRowsOut[i],
                        Float.isFinite(gotRowsOut[i]));
                assertEquals(type + " prefill output " + i, wantRows[i], gotRowsOut[i], 1e-4f);
            }
        } finally {
            for (GpuBuffer b : bufs) {
                b.close();
            }
        }
    }

    /** {@code LLM_FlashDecode_*} arguments of one direct split over the whole cache. */
    private Object[] decodeArgs(FlashAttnTiling t, GpuBuffer q, GpuBuffer kv, GpuBuffer kvPos, GpuBuffer sinks,
                                GpuBuffer relBias, GpuBuffer out, GpuBuffer partial, GpuBuffer side,
                                GpuBuffer treeMask) {
        Object[] a = new Object[AttnKernels.FD_ARG_COUNT];
        a[AttnKernels.ARG_FD_N_HEAD] = N_HEAD;
        a[AttnKernels.ARG_FD_N_HEAD_KV] = N_HEAD_KV;
        a[AttnKernels.ARG_FD_D_K] = D_Q;
        a[AttnKernels.ARG_FD_D_V] = D_V;
        a[AttnKernels.ARG_FD_G_TILE] = t.gTile();
        a[AttnKernels.ARG_FD_N_SPLIT] = 1;
        a[AttnKernels.ARG_FD_SPLIT_LEN] = N_KV;
        a[AttnKernels.ARG_FD_KV_START] = 0;
        a[AttnKernels.ARG_FD_KV_END] = N_KV;
        a[AttnKernels.ARG_FD_Q_POS] = N_KV - 1;
        a[AttnKernels.ARG_FD_K_ROW_STRIDE] = D_Q;
        a[AttnKernels.ARG_FD_K_HEAD_STRIDE] = D_Q;
        a[AttnKernels.ARG_FD_V_ROW_STRIDE] = D_Q;
        a[AttnKernels.ARG_FD_V_HEAD_STRIDE] = D_V;
        a[AttnKernels.ARG_FD_SCALE] = (float) (1.0 / Math.sqrt(D_Q));
        a[AttnKernels.ARG_FD_SOFTCAP] = 0f;
        a[AttnKernels.ARG_FD_MAX_BIAS] = 0f;
        a[AttnKernels.ARG_FD_FLAGS] = AttnKernels.attnFlags(true, false, true) | AttnKernels.FLAG_V_ALIAS;
        a[AttnKernels.ARG_FD_SWA_TYPE] = KvWindow.FULL_CAUSAL.type().code();
        a[AttnKernels.ARG_FD_N_SWA] = 0;
        a[AttnKernels.ARG_FD_BIAS_MODE] = 0;
        a[AttnKernels.ARG_FD_N_BUCKETS] = 0;
        a[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
        a[AttnKernels.ARG_FD_Q_OFF] = 0;
        a[AttnKernels.ARG_FD_OUT_OFF] = 0;
        a[AttnKernels.ARG_FD_BUF_Q] = q;
        a[AttnKernels.ARG_FD_BUF_K] = kv;
        a[AttnKernels.ARG_FD_BUF_V] = kv;
        a[AttnKernels.ARG_FD_BUF_KV_POS] = kvPos;
        a[AttnKernels.ARG_FD_BUF_SINKS] = sinks;
        a[AttnKernels.ARG_FD_BUF_REL_BIAS] = relBias;
        a[AttnKernels.ARG_FD_BUF_OUT] = out;
        a[AttnKernels.ARG_FD_BUF_PARTIAL] = partial;
        a[AttnKernels.ARG_FD_BUF_KV_SCALES] = side;
        // the mask slot stays bound as in the engine's launches; the tree flag alone admits it
        a[AttnKernels.ARG_FD_BUF_TREE_MASK] = treeMask;
        return a;
    }

    /** Bytes per row of the packed side planes of one latent row format. */
    private static int sideRowBytes(KvDType type) {
        return type == KvDType.FP8_DS_MLA ? 4 * 4 : KvDType.DS_MLA_NOPE_VALUES / KvVendorCodec.NVFP4_GROUP;
    }

    private static void storeCpu(KvDType type, float[] rows, int rowOff, byte[] dst, int dstOff, byte[] side,
                                 int sideOff) {
        if (type == KvDType.FP8_DS_MLA) {
            float[] scales = new float[4];
            KvVendorCodec.fp8DsMlaStore(rows, rowOff, dst, dstOff, scales, 0);
            for (int t = 0; t < 4; t++) {
                putFloat(side, sideOff + 4 * t, scales[t]);
            }
        } else {
            float[] sf = new float[KvDType.DS_MLA_NOPE_VALUES / KvVendorCodec.NVFP4_GROUP];
            KvVendorCodec.nvfp4DsMlaStore(rows, rowOff, dst, dstOff, sf, 0);
            for (int g = 0; g < sf.length; g++) {
                side[sideOff + g] = (byte) KvVendorCodec.e4m3Code(sf[g]);
            }
        }
    }

    private static void loadCpu(KvDType type, byte[] src, int srcOff, float[] fp8Scales, byte[] side, int sideOff,
                                float[] k, int kOff, float[] v, int vOff) {
        float[] row = new float[D_Q];
        if (type == KvDType.FP8_DS_MLA) {
            for (int t = 0; t < 4; t++) {
                fp8Scales[t] = getFloat(side, sideOff + 4 * t);
            }
            KvVendorCodec.fp8DsMlaLoad(src, srcOff, fp8Scales, 0, row, 0);
        } else {
            float[] sf = new float[KvDType.DS_MLA_NOPE_VALUES / KvVendorCodec.NVFP4_GROUP];
            for (int g = 0; g < sf.length; g++) {
                sf[g] = KvVendorCodec.e4m3Value(side[sideOff + g] & 0xFF);
            }
            KvVendorCodec.nvfp4DsMlaLoad(src, srcOff, sf, 0, row, 0);
        }
        System.arraycopy(row, 0, k, kOff, D_Q);
        System.arraycopy(row, 0, v, vOff, D_V);
    }

    private void dispatch(String kernel, long[] global, long[] local, Object[] args) {
        GpuEvent e = backend.computeQueue().enqueueKernel(kernels.get(kernel), global, local, args);
        if (e != null) {
            e.close();
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
        for (int i = 0; i < data.length; i++) {
            putInt(bytes, 4 * i, data[i]);
        }
        GpuBuffer b = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
    }

    /** The kernels' {@code ulong} words are little-endian, as the Java uploads are. */
    private GpuBuffer upload(List<GpuBuffer> bufs, long[] data) {
        byte[] bytes = new byte[data.length * 8];
        for (int i = 0; i < data.length; i++) {
            for (int b = 0; b < 8; b++) {
                bytes[8 * i + b] = (byte) (data[i] >>> (8 * b));
            }
        }
        GpuBuffer buf = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(buf, 0, bytes.length, bytes));
        return buf;
    }

    private void write(GpuBuffer b, byte[] data) {
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, data.length, data));
    }

    private byte[] readBytes(GpuBuffer b, int n) {
        byte[] dst = new byte[n];
        close(backend.computeQueue().enqueueReadBuffer(b, 0, n, dst));
        backend.computeQueue().finish();
        return dst;
    }

    private float[] readFloats(GpuBuffer b, int n) {
        float[] dst = new float[n];
        close(backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * 4, dst));
        backend.computeQueue().finish();
        return dst;
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    private static void putInt(byte[] b, int off, int v) {
        b[off] = (byte) v;
        b[off + 1] = (byte) (v >> 8);
        b[off + 2] = (byte) (v >> 16);
        b[off + 3] = (byte) (v >> 24);
    }

    private static void putFloat(byte[] b, int off, float v) {
        putInt(b, off, Float.floatToRawIntBits(v));
    }

    private static float getFloat(byte[] b, int off) {
        return Float.intBitsToFloat((b[off] & 0xFF) | ((b[off + 1] & 0xFF) << 8) | ((b[off + 2] & 0xFF) << 16)
                | ((b[off + 3] & 0xFF) << 24));
    }

    private static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < b.length; i++) {
            sb.append(Character.forDigit((b[i] >> 4) & 0xF, 16)).append(Character.forDigit(b[i] & 0xF, 16));
        }
        return sb.toString();
    }
}
