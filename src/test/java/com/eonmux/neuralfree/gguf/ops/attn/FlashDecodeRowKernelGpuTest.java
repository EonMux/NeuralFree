package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryConfig;
import com.eonmux.neuralfree.acceleration.kernels.VulkanKernelLibrary;
import com.eonmux.neuralfree.acceleration.vulkan.VkDeviceRanking;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.tensor.HalfFloats;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The one-row decode kernel ({@code LLM_FlashDecodeRow_F16}: {@code flash_decode_row.cl} on CUDA, its twin
 * {@code flash_decode_row.comp} on Vulkan) against the CPU reference and against the generic decode kernel it
 * replaces ({@code LLM_FlashDecode_F16}, same arguments): at every head dim it serves (64, 128, 256), GQA, key
 * counts from one key through direct steps to split-K steps with the combine, and the step forms the plain one-row
 * route meets — a key range starting past row 0, a sliding window, a softcap and empty cache cells (kv_pos -1). The
 * CPU reference covers the causal steps; the generic kernel, itself held to the same reference by
 * {@link FlashDecodeKernelGpuTest}, covers the window, softcap and empty-cell steps. The Vulkan cases run the module
 * the device binds (the subgroup module on subgroup-arithmetic devices) and the plain module, whose lane groups
 * reduce through shared memory, and both read caches whose row stride is a multiple of 4 elements (8-byte lane
 * loads) or only of 2 (two 4-byte loads).
 */
@RunWith(Parameterized.class)
public class FlashDecodeRowKernelGpuTest {

    private static final int N_HEAD = 8;
    private static final int N_HEAD_KV = 2;
    private static final int N_KV = 800;
    private static final float TOL = 1e-4f;
    private static final int[] KEY_COUNTS = {1, 5, 37, 128, 129, 200, 700};

    /** The row kernel's module: the one the backend binds, or (Vulkan) the plain module without subgroups. */
    enum Module { BOUND, PLAIN }

    @Parameterized.Parameters(name = "{0} {1} d={2}")
    public static List<Object[]> cases() {
        List<Object[]> rows = new ArrayList<>();
        for (int d : new int[]{64, 128, 256}) {
            rows.add(new Object[]{ComputeBackend.BackendType.CUDA, Module.BOUND, d});
            rows.add(new Object[]{ComputeBackend.BackendType.VULKAN, Module.BOUND, d});
            rows.add(new Object[]{ComputeBackend.BackendType.VULKAN, Module.PLAIN, d});
        }
        return rows;
    }

    private final ComputeBackend.BackendType api;
    private final Module module;
    private final int d;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private KernelHandle rowHandle;
    private FlashAttnTiling tiling;
    private final List<GpuBuffer> bufs = new ArrayList<>();

    public FlashDecodeRowKernelGpuTest(ComputeBackend.BackendType api, Module module, int d) {
        this.api = api;
        this.module = module;
        this.d = d;
    }

    @Before
    public void open() {
        if (api == ComputeBackend.BackendType.CUDA) {
            GpuTestUtil.assumeCudaAvailable();
        } else {
            GpuTestUtil.assumeVulkanAvailable();
        }
        hw = new HWAcceleration(EnumSet.of(api));
        HardwareDevice device = api == ComputeBackend.BackendType.CUDA
                ? (hw.getCUDevices().isEmpty() ? null : hw.getCUDevices().get(0))
                : VkDeviceRanking.fastestDevice(hw.getVkDevices(), null);
        Assume.assumeTrue("no " + api + " device enumerated", device != null);
        backend = hw.createBackend(device);
        Availability a = LlmKernels.availability(backend);
        Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + a.reason(), a.available());
        kernels = LlmKernels.open(backend);
        Assume.assumeTrue("no " + AttnKernels.FLASH_DECODE_ROW_F16 + " kernel",
                kernels.has(AttnKernels.FLASH_DECODE_ROW_F16));
        rowHandle = module == Module.PLAIN ? plainVulkanKernel(AttnKernels.FLASH_DECODE_ROW_F16)
                : kernels.get(AttnKernels.FLASH_DECODE_ROW_F16);
        boolean fast = FlashAttnTiling.decodeFastPath(backend.capabilities().hasFeature(DeviceFeature.SUBGROUP_SHUFFLE),
                KvDType.F16, KvDType.F16, d, d);
        tiling = FlashAttnTiling.decode(N_HEAD, N_HEAD_KV, d, d, N_KV, kernels.faSmemFloats(),
                Math.max(1, backend.capabilities().getComputeUnits()), fast);
    }

    /** The Vulkan LLM library's module of {@code name} without subgroup variants: the plain SPIR-V. */
    private KernelHandle plainVulkanKernel(String name) {
        KernelLibraryConfig c = KernelLibraries.LLM_CONFIG;
        KernelLibraryConfig plain = new KernelLibraryConfig(c.libraryId(), c.sourceRoot(), c.cudaRoot(), c.spvRoot(),
                c.requireCompleteInventory(), c.openClOptions(), false, c.vulkanFeatures());
        return backend.loadKernel(new VulkanKernelLibrary(plain).resolve(name, backend));
    }

    @After
    public void close() {
        for (GpuBuffer b : bufs) {
            b.close();
        }
        bufs.clear();
        if (module == Module.PLAIN && rowHandle != null) {
            rowHandle.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    @Test
    public void causalStepsMatchTheCpuReference() {
        causalSteps(0);
    }

    /** A row stride of 2 mod 4 elements: the Vulkan twin loads each lane's four halves as two words. */
    @Test
    public void causalStepsOverAHalfAlignedRowStrideMatchTheCpuReference() {
        causalSteps(2);
    }

    private void causalSteps(int pad) {
        Fixture f = new Fixture(new Random(11), identityPositions(), pad);
        CpuAttentionCore.Params params = CpuAttentionCore.Params.builder(N_HEAD).scale(f.scale)
                .window(KvWindow.FULL_CAUSAL).build();
        for (int n : KEY_COUNTS) {
            float[] want = new float[N_HEAD * d];
            CpuAttentionCore.attendRows(params, 1, f.qHost, 0, N_HEAD * d, new int[]{n - 1}, 0,
                    CpuAttentionCore.DenseKv.of(f.kF, 0, f.rowStride, d, f.vF, 0, f.rowStride, d,
                            N_HEAD_KV, d, d, f.kvPos, 0),
                    0, n, want, 0, N_HEAD * d);
            float[] got = f.run(AttnKernels.FLASH_DECODE_ROW_F16, 0, n, n - 1, 0f, SwaType.NONE, 0);
            assertClose("pad=" + pad + " n=" + n, want, got);
        }
    }

    @Test
    public void windowSoftcapOffsetAndEmptyCellsMatchTheGenericKernel() {
        int[] positions = identityPositions();
        for (int i = 3; i < N_KV; i += 17) {
            positions[i] = -1;   // empty cells are never visible
        }
        Fixture f = new Fixture(new Random(13), positions, 0);
        int[][] steps = {
            // kv_start, kv_end, q_pos, swa type, n_swa, softcap * 100
            {0, 300, 299, SwaType.STANDARD.code(), 64, 0},
            {0, 700, 699, SwaType.STANDARD.code(), 250, 3000},
            {40, 200, 199, SwaType.NONE.code(), 0, 0},
            {0, 129, 128, SwaType.NONE.code(), 0, 5000},
            {0, 5, 4, SwaType.NONE.code(), 0, 0},
        };
        for (int[] s : steps) {
            String where = "range [" + s[0] + ", " + s[1] + ") swa " + s[3] + "/" + s[4] + " softcap " + s[5];
            float softcap = s[5] / 100f;
            SwaType swa = s[3] == SwaType.STANDARD.code() ? SwaType.STANDARD : SwaType.NONE;
            float[] want = f.run(AttnKernels.flashDecode(KvDType.F16), s[0], s[1], s[2], softcap, swa, s[4]);
            float[] got = f.run(AttnKernels.FLASH_DECODE_ROW_F16, s[0], s[1], s[2], softcap, swa, s[4]);
            assertClose(where, want, got);
        }
    }

    private int[] identityPositions() {
        int[] p = new int[N_KV];
        for (int i = 0; i < N_KV; i++) {
            p[i] = i;
        }
        return p;
    }

    private void assertClose(String where, float[] want, float[] got) {
        for (int i = 0; i < want.length; i++) {
            String at = api + " " + module + " d=" + d + " " + where + " output " + i;
            assertTrue(at + " is not finite: " + got[i], Float.isFinite(got[i]));
            assertEquals(at, want[i], got[i], TOL);
        }
    }

    /** One cache and query on the device, run through either kernel at a step's range and window. */
    private final class Fixture {
        final float scale = (float) (1.0 / Math.sqrt(d));
        final int rowStride;
        final float[] qHost;
        final float[] kF;
        final float[] vF;
        final int[] kvPos;
        final GpuBuffer q;
        final GpuBuffer k;
        final GpuBuffer v;
        final GpuBuffer pos;
        final GpuBuffer one;
        final GpuBuffer scales;
        final GpuBuffer mask;
        final GpuBuffer out;
        final GpuBuffer partial;

        Fixture(Random rnd, int[] kvPos, int pad) {
            this.kvPos = kvPos;
            this.rowStride = N_HEAD_KV * d + pad;
            short[] k16 = new short[N_KV * rowStride];
            short[] v16 = new short[k16.length];
            kF = new float[k16.length];
            vF = new float[k16.length];
            for (int i = 0; i < k16.length; i++) {
                k16[i] = (short) HalfFloats.floatToHalf(rnd.nextFloat() - 0.5f);
                kF[i] = HalfFloats.halfToFloat(k16[i] & 0xFFFF);
                v16[i] = (short) HalfFloats.floatToHalf(rnd.nextFloat() - 0.5f);
                vF[i] = HalfFloats.halfToFloat(v16[i] & 0xFFFF);
            }
            qHost = new float[N_HEAD * d];
            for (int i = 0; i < qHost.length; i++) {
                qHost[i] = 2f * (rnd.nextFloat() - 0.5f);
            }
            q = upload(qHost);
            k = uploadHalfs(k16);
            v = uploadHalfs(v16);
            pos = upload(kvPos);
            one = upload(new float[1]);
            float[] ones = new float[2 * N_HEAD_KV];
            Arrays.fill(ones, 1f);
            scales = upload(ones);
            mask = alloc(8);
            out = alloc((long) N_HEAD * d * 4);
            partial = alloc(tiling.decodePartialFloats() * 4);
        }

        float[] run(String kernel, int kvStart, int kvEnd, int qPos, float softcap, SwaType swa, int nSwa) {
            FlashAttnTiling.Split split = tiling.decodeSplit(kvEnd - kvStart);
            boolean direct = split.count() == 1;
            Object[] args = new Object[AttnKernels.FD_ARG_COUNT];
            args[AttnKernels.ARG_FD_N_HEAD] = N_HEAD;
            args[AttnKernels.ARG_FD_N_HEAD_KV] = N_HEAD_KV;
            args[AttnKernels.ARG_FD_D_K] = d;
            args[AttnKernels.ARG_FD_D_V] = d;
            args[AttnKernels.ARG_FD_G_TILE] = tiling.gTile();
            args[AttnKernels.ARG_FD_N_SPLIT] = split.count();
            args[AttnKernels.ARG_FD_SPLIT_LEN] = split.length();
            args[AttnKernels.ARG_FD_KV_START] = kvStart;
            args[AttnKernels.ARG_FD_KV_END] = kvEnd;
            args[AttnKernels.ARG_FD_Q_POS] = qPos;
            args[AttnKernels.ARG_FD_K_ROW_STRIDE] = rowStride;
            args[AttnKernels.ARG_FD_K_HEAD_STRIDE] = d;
            args[AttnKernels.ARG_FD_V_ROW_STRIDE] = rowStride;
            args[AttnKernels.ARG_FD_V_HEAD_STRIDE] = d;
            args[AttnKernels.ARG_FD_SCALE] = scale;
            args[AttnKernels.ARG_FD_SOFTCAP] = softcap;
            args[AttnKernels.ARG_FD_MAX_BIAS] = 0f;
            args[AttnKernels.ARG_FD_FLAGS] = AttnKernels.attnFlags(true, false, direct);
            args[AttnKernels.ARG_FD_SWA_TYPE] = swa.code();
            args[AttnKernels.ARG_FD_N_SWA] = nSwa;
            args[AttnKernels.ARG_FD_BIAS_MODE] = 0;
            args[AttnKernels.ARG_FD_N_BUCKETS] = 0;
            args[AttnKernels.ARG_FD_MAX_DISTANCE] = 0;
            args[AttnKernels.ARG_FD_Q_OFF] = 0;
            args[AttnKernels.ARG_FD_OUT_OFF] = 0;
            args[AttnKernels.ARG_FD_BUF_Q] = q;
            args[AttnKernels.ARG_FD_BUF_K] = k;
            args[AttnKernels.ARG_FD_BUF_V] = v;
            args[AttnKernels.ARG_FD_BUF_KV_POS] = pos;
            args[AttnKernels.ARG_FD_BUF_SINKS] = one;
            args[AttnKernels.ARG_FD_BUF_REL_BIAS] = one;
            args[AttnKernels.ARG_FD_BUF_OUT] = out;
            args[AttnKernels.ARG_FD_BUF_PARTIAL] = partial;
            args[AttnKernels.ARG_FD_BUF_KV_SCALES] = scales;
            args[AttnKernels.ARG_FD_BUF_TREE_MASK] = mask;
            dispatch(kernel, tiling.decodeGlobal(split.count()), FlashAttnTiling.decodeLocal(), args);
            if (!direct) {
                Object[] combine = {N_HEAD, d, split.count(), N_HEAD, 0, 0, partial, one, out};
                dispatch(AttnKernels.FLASH_COMBINE, FlashAttnTiling.combineGlobal(N_HEAD),
                        FlashAttnTiling.combineLocal(), combine);
            }
            return readFloats(out, N_HEAD * d);
        }
    }

    private void dispatch(String kernel, long[] global, long[] local, Object[] args) {
        KernelHandle h = kernel.equals(AttnKernels.FLASH_DECODE_ROW_F16) ? rowHandle : kernels.get(kernel);
        close(backend.computeQueue().enqueueKernel(h, global, local, args));
    }

    private GpuBuffer alloc(long bytes) {
        GpuBuffer b = backend.allocateBuffer(bytes, true);
        bufs.add(b);
        return b;
    }

    private GpuBuffer upload(float[] data) {
        GpuBuffer b = alloc((long) data.length * 4);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * 4, data));
        return b;
    }

    private GpuBuffer upload(int[] data) {
        byte[] bytes = new byte[data.length * 4];
        for (int i = 0; i < data.length; i++) {
            int x = data[i];
            bytes[4 * i] = (byte) x;
            bytes[4 * i + 1] = (byte) (x >> 8);
            bytes[4 * i + 2] = (byte) (x >> 16);
            bytes[4 * i + 3] = (byte) (x >> 24);
        }
        GpuBuffer b = alloc(bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
    }

    /** The kernels' {@code half} rows are little-endian F16 words. */
    private GpuBuffer uploadHalfs(short[] data) {
        byte[] bytes = new byte[data.length * 2];
        for (int i = 0; i < data.length; i++) {
            bytes[2 * i] = (byte) data[i];
            bytes[2 * i + 1] = (byte) (data[i] >> 8);
        }
        GpuBuffer b = alloc(bytes.length);
        close(backend.computeQueue().enqueueWriteBuffer(b, 0, bytes.length, bytes));
        return b;
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
