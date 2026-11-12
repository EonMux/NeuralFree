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
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.tensor.HalfFloats;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The dense flash-decode kernel ({@code LLM_FlashDecode_F16} and {@code LLM_FlashDecode_Q8_0}) against the CPU
 * reference at the head dims that surround the warp fast path's limit ({@code NF_FA_FAST_DIM}): one coalesced row
 * float4 per lane covers at most 32 * 4 dims, so d = 128 and the non-multiple-of-128 dims (96, 80) run the fast path on a warp-path backend,
 * while d = 256 must take the scalar body (one lane cannot hold the row, and the mma plan refuses the shape
 * anyway). A Q8_0 cache takes the same route at head dims of whole 32-value blocks (64 replaces 80). Every case
 * checks both step shapes of a decode: a direct launch over one workgroup of keys and a split-K launch with its
 * combine.
 *
 * <p>Every test repeats over the machine's OpenCL, CUDA and Vulkan devices, so the fast path (CUDA, and Vulkan on
 * 32-lane subgroups) and the scalar body (OpenCL, other Vulkan devices) answer the same reference.</p>
 */
@RunWith(Parameterized.class)
public class FlashDecodeKernelGpuTest {

    private static final int N_HEAD = 8;
    private static final int N_HEAD_KV = 2;
    private static final int N_KV = 256;
    private static final float TOL = 1e-4f;

    /** Head dims around the fast path's 128-dim limit: at the limit, below it, mixed, and over it. */
    private static final int[][] DIMS = {{128, 128}, {96, 96}, {80, 128}, {256, 256}};
    /** One direct step (a single workgroup of keys) and one split-K step. */
    private static final int[] KEY_COUNTS = {5, 200};

    @Parameterized.Parameters(name = "{0} {3} d={1}x{2}")
    public static List<Object[]> cases() {
        List<Object[]> rows = new ArrayList<>();
        for (ComputeBackend.BackendType t : EnumSet.of(ComputeBackend.BackendType.OPENCL,
                ComputeBackend.BackendType.CUDA, ComputeBackend.BackendType.VULKAN)) {
            for (KvDType type : new KvDType[]{KvDType.F16, KvDType.Q8_0}) {
                for (int[] dims : DIMS) {
                    int dK = type == KvDType.Q8_0 && dims[0] % Q8_BLOCK != 0 ? 64 : dims[0];
                    rows.add(new Object[]{t, dK, dims[1], type});
                }
            }
        }
        return rows;
    }

    /** Values per Q8_0 block. */
    private static final int Q8_BLOCK = 32;

    private final ComputeBackend.BackendType api;
    private final int dK;
    private final int dV;
    private final KvDType type;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;

    public FlashDecodeKernelGpuTest(ComputeBackend.BackendType api, int dK, int dV, KvDType type) {
        this.api = api;
        this.dK = dK;
        this.dV = dV;
        this.type = type;
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
    public void decodeMatchesTheCpuReferenceAroundTheFastPathDimLimit() {
        // The plan must mirror the kernel gate: over the 128-dim limit no warp-path backend runs the fast path.
        boolean expectFast = FlashAttnTiling.decodeFastPath(kernels.decodeWarpPath(), type, type, dK, dV);
        FlashAttnTiling tiling = FlashAttnTiling.decode(N_HEAD, N_HEAD_KV, dK, dV, N_KV, kernels.faSmemFloats(),
                Math.max(1, backend.capabilities().getComputeUnits()), expectFast);
        Assume.assumeTrue("no decode plan for d=" + dK + "x" + dV,
                tiling.plan() == FlashAttnTiling.Plan.DECODE);

        Random rnd = new Random(7);
        float scale = (float) (1.0 / Math.sqrt(dK));
        // K and V rows encoded host-side; the reference decodes the same values the GPU sees.
        float[] kF = cacheValues(rnd, N_KV * N_HEAD_KV * dK);
        float[] vF = cacheValues(rnd, N_KV * N_HEAD_KV * dV);
        int[] kvPos = new int[N_KV];
        for (int i = 0; i < N_KV; i++) {
            kvPos[i] = i;
        }

        List<GpuBuffer> bufs = new ArrayList<>();
        try {
            float[] qHost = randomFloats(rnd, N_HEAD * dK);
            GpuBuffer q = upload(bufs, qHost);
            GpuBuffer dk = uploadCache(bufs, kF);
            GpuBuffer dv = uploadCache(bufs, vF);
            GpuBuffer dpos = upload(bufs, kvPos);
            GpuBuffer dsinks = upload(bufs, new float[1]);
            GpuBuffer drel = upload(bufs, new float[1]);
            // the kernels read one K-side and one V-side scale per KV head (kv_types.h nf_kv_head_scale)
            float[] scaleOnes = new float[2 * N_HEAD_KV];
            java.util.Arrays.fill(scaleOnes, 1f);
            GpuBuffer dscales = upload(bufs, scaleOnes);
            GpuBuffer dmask = upload(bufs, new long[1]);
            GpuBuffer dout = alloc(bufs, (long) N_HEAD * dV * 4);
            GpuBuffer dpartial = alloc(bufs, tiling.decodePartialFloats() * 4);

            CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(N_HEAD).scale(scale)
                    .window(KvWindow.FULL_CAUSAL).build();
            for (int n : KEY_COUNTS) {
                float[] want = new float[N_HEAD * dV];
                CpuAttentionCore.attendRows(params, 1, qHost, 0, N_HEAD * dK, new int[]{n - 1}, 0,
                        CpuAttentionCore.DenseKv.of(kF, 0, N_HEAD_KV * dK, dK, vF, 0, N_HEAD_KV * dV, dV,
                                N_HEAD_KV, dK, dV, kvPos, 0),
                        0, n, want, 0, N_HEAD * dV);

                FlashAttnTiling.Split split = tiling.decodeSplit(n);
                boolean direct = split.count() == 1;
                Object[] args = new Object[AttnKernels.FD_ARG_COUNT];
                args[AttnKernels.ARG_FD_N_HEAD] = N_HEAD;
                args[AttnKernels.ARG_FD_N_HEAD_KV] = N_HEAD_KV;
                args[AttnKernels.ARG_FD_D_K] = dK;
                args[AttnKernels.ARG_FD_D_V] = dV;
                args[AttnKernels.ARG_FD_G_TILE] = tiling.gTile();
                args[AttnKernels.ARG_FD_N_SPLIT] = split.count();
                args[AttnKernels.ARG_FD_SPLIT_LEN] = split.length();
                args[AttnKernels.ARG_FD_KV_START] = 0;
                args[AttnKernels.ARG_FD_KV_END] = n;
                args[AttnKernels.ARG_FD_Q_POS] = n - 1;
                args[AttnKernels.ARG_FD_K_ROW_STRIDE] = N_HEAD_KV * dK;
                args[AttnKernels.ARG_FD_K_HEAD_STRIDE] = dK;
                args[AttnKernels.ARG_FD_V_ROW_STRIDE] = N_HEAD_KV * dV;
                args[AttnKernels.ARG_FD_V_HEAD_STRIDE] = dV;
                args[AttnKernels.ARG_FD_SCALE] = scale;
                args[AttnKernels.ARG_FD_SOFTCAP] = 0f;
                args[AttnKernels.ARG_FD_MAX_BIAS] = 0f;
                args[AttnKernels.ARG_FD_FLAGS] = AttnKernels.attnFlags(true, false, direct);
                args[AttnKernels.ARG_FD_SWA_TYPE] = KvWindow.FULL_CAUSAL.type().code();
                args[AttnKernels.ARG_FD_N_SWA] = 0;
                args[AttnKernels.ARG_FD_BIAS_MODE] = 0;
                args[AttnKernels.ARG_FD_N_BUCKETS] = 0;
                args[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
                args[AttnKernels.ARG_FD_Q_OFF] = 0;
                args[AttnKernels.ARG_FD_OUT_OFF] = 0;
                args[AttnKernels.ARG_FD_BUF_Q] = q;
                args[AttnKernels.ARG_FD_BUF_K] = dk;
                args[AttnKernels.ARG_FD_BUF_V] = dv;
                args[AttnKernels.ARG_FD_BUF_KV_POS] = dpos;
                args[AttnKernels.ARG_FD_BUF_SINKS] = dsinks;
                args[AttnKernels.ARG_FD_BUF_REL_BIAS] = drel;
                args[AttnKernels.ARG_FD_BUF_OUT] = dout;
                args[AttnKernels.ARG_FD_BUF_PARTIAL] = dpartial;
                args[AttnKernels.ARG_FD_BUF_KV_SCALES] = dscales;
                args[AttnKernels.ARG_FD_BUF_TREE_MASK] = dmask;
                dispatch(AttnKernels.flashDecode(type), tiling.decodeGlobal(split.count()),
                        FlashAttnTiling.decodeLocal(), args);
                if (!direct) {
                    Object[] combine = {N_HEAD, dV, split.count(), N_HEAD, 0, 0, dpartial, dsinks, dout};
                    dispatch(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(N_HEAD),
                            FlashAttnTiling.combineLocal(), combine);
                }
                float[] got = readFloats(dout, want.length);
                for (int i = 0; i < want.length; i++) {
                    String what = api + " " + type + " d=" + dK + "x" + dV + " n=" + n + " output " + i;
                    assertTrue(what + " is not finite: " + got[i], Float.isFinite(got[i]));
                    assertEquals(what, want[i], got[i], TOL);
                }
            }
        } finally {
            for (GpuBuffer b : bufs) {
                b.close();
            }
        }
    }

    /**
     * {@code n} cache values exactly as the kernels decode them: F16-rounded randoms, or Q8_0 codes in [-127, 127]
     * times their block's {@link #q8Scale}.
     */
    private float[] cacheValues(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = type == KvDType.Q8_0 ? q8Scale(i / Q8_BLOCK) * (rnd.nextInt(255) - 127)
                    : HalfFloats.halfToFloat(HalfFloats.floatToHalf(rnd.nextFloat() - 0.5f));
        }
        return v;
    }

    /** The scale of Q8_0 block {@code b}: an exact half, so every value is an exact float product. */
    private static float q8Scale(int b) {
        return HalfFloats.halfToFloat(HalfFloats.floatToHalf(0.004f * (1 + (b % 7) / 8f)));
    }

    /** The cache of {@link #cacheValues}: little-endian F16 words, or Q8_0 blocks (half scale, then 32 codes). */
    private GpuBuffer uploadCache(List<GpuBuffer> bufs, float[] values) {
        if (type != KvDType.Q8_0) {
            short[] h = new short[values.length];
            for (int i = 0; i < values.length; i++) {
                h[i] = (short) HalfFloats.floatToHalf(values[i]);
            }
            return uploadHalfs(bufs, h);
        }
        int blockBytes = 2 + Q8_BLOCK;
        byte[] bytes = new byte[values.length / Q8_BLOCK * blockBytes];
        for (int b = 0; b < values.length / Q8_BLOCK; b++) {
            float d = q8Scale(b);
            int h = HalfFloats.floatToHalf(d);
            bytes[b * blockBytes] = (byte) h;
            bytes[b * blockBytes + 1] = (byte) (h >> 8);
            for (int i = 0; i < Q8_BLOCK; i++) {
                bytes[b * blockBytes + 2 + i] = (byte) Math.round(values[b * Q8_BLOCK + i] / d);
            }
        }
        GpuBuffer buf = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(buf, 0, bytes.length, bytes));
        return buf;
    }

    private static float[] randomFloats(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() - 0.5f;
        }
        return v;
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
            int v = data[i];
            bytes[4 * i] = (byte) v;
            bytes[4 * i + 1] = (byte) (v >> 8);
            bytes[4 * i + 2] = (byte) (v >> 16);
            bytes[4 * i + 3] = (byte) (v >> 24);
        }
        GpuBuffer b = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
    }

    /** The kernels' {@code half} rows are little-endian F16 words. */
    private GpuBuffer uploadHalfs(List<GpuBuffer> bufs, short[] data) {
        byte[] bytes = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            bytes[2 * i] = (byte) data[i];
            bytes[2 * i + 1] = (byte) (data[i] >> 8);
        }
        GpuBuffer b = alloc(bufs, bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
    }

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
}
