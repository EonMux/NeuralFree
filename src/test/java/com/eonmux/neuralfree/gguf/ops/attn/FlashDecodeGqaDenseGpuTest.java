package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.HardwareDevice;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryConfig;
import com.eonmux.neuralfree.acceleration.kernels.VulkanKernelLibrary;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
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
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The grouped-query dense decode ({@code LLM_FlashDecodeGqa_<KV>}: kernels/llm/flash_decode_gqa.cl on CUDA, its twin
 * shaders/llm/flash_decode_gqa.comp on Vulkan) against the CPU reference ({@link CpuAttentionCore}) over a dense
 * F16, BF16 or Q8_0 cache (Q8_0 also with per-head K and V scales, which the kernels read as the generic decode
 * does): query groups of 1 to 4 heads at head dims 64 and 128; key counts from one key through ranges of several
 * staged blocks; the splits of the online-softmax plan ({@link FlashAttnTiling#decodeOnline}) merged by
 * {@code LLM_FlashAttn_Combine}, one direct split over the whole range (several stages in one work-group) and fixed
 * split layouts; a key range starting past row 0, a sliding window, a softcap, ALiBi, T5 and Inkling relative
 * biases, sink logits (in the direct output and through the merge), query positions below the cached ones and empty
 * cells (kv_pos -1). It runs on CUDA and on Vulkan on each device (the NVIDIA GPU and the Intel iGPU), there with
 * the module the device binds (the subgroup module on subgroup-arithmetic devices) and with the plain module, whose
 * transpose blocks synchronize with work-group barriers; what the machine lacks is skipped.
 */
@RunWith(Parameterized.class)
public class FlashDecodeGqaDenseGpuTest {

    private static final int N_KV = 2100;
    private static final int[] KEY_COUNTS = {1, 5, 37, 300, 777, 2100};
    /** Splits the partial buffer holds: one per 32 keys of the cache, the shortest split any plan makes. */
    private static final int MAX_SPLITS = (N_KV + 31) / 32;
    /** T5 buckets and distance limit, and the Inkling table extent, of the relative-bias steps. */
    private static final int T5_BUCKETS = 32;
    private static final int T5_MAX_DISTANCE = 128;
    private static final int INKLING_EXTENT = 300;

    /** The kernel's module: the one the backend binds, or (Vulkan) the plain module without subgroups. */
    enum Module { BOUND, PLAIN }

    @Parameterized.Parameters(name = "{0} {1} {2}")
    public static List<Object[]> cases() {
        List<Object[]> rows = new ArrayList<>();
        rows.add(new Object[]{ComputeBackend.BackendType.CUDA, "", Module.BOUND});
        for (String device : new String[]{"NVIDIA", "Intel"}) {
            rows.add(new Object[]{ComputeBackend.BackendType.VULKAN, device, Module.BOUND});
            rows.add(new Object[]{ComputeBackend.BackendType.VULKAN, device, Module.PLAIN});
        }
        return rows;
    }

    private final ComputeBackend.BackendType api;
    private final String deviceName;
    private final Module module;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private final Map<KvDType, KernelHandle> plainHandles = new HashMap<>();
    private final List<GpuBuffer> buffers = new ArrayList<>();

    public FlashDecodeGqaDenseGpuTest(ComputeBackend.BackendType api, String deviceName, Module module) {
        this.api = api;
        this.deviceName = deviceName;
        this.module = module;
    }

    @Before
    public void openDevice() {
        if (api == ComputeBackend.BackendType.CUDA) {
            GpuTestUtil.assumeCudaAvailable();
        } else {
            GpuTestUtil.assumeVulkanAvailable();
        }
        hwAccel = new HWAcceleration(EnumSet.of(api));
        backend = openBackend();
        Assume.assumeTrue("no " + api + " " + deviceName + " device enumerated", backend != null);
        Availability availability = LlmKernels.availability(backend);
        Assume.assumeTrue("LLM kernels unavailable: " + availability.reason(), availability.available());
        kernels = LlmKernels.open(backend);
        Assume.assumeTrue("no grouped-query dense decode on this device",
                kernels.has(AttnKernels.flashDecodeGqa(KvDType.F16, KvDType.F16)));
    }

    /** A backend over the first device of the API whose name holds {@link #deviceName}, or null. */
    private ComputeBackend openBackend() {
        List<HardwareDevice> all = new ArrayList<>(api == ComputeBackend.BackendType.CUDA
                ? hwAccel.getCUDevices() : hwAccel.getVkDevices());
        for (HardwareDevice d : all) {
            ComputeBackend b = hwAccel.createBackend(d);
            String name = b.capabilities().getDeviceName();
            if (deviceName.isEmpty() || (name != null && name.contains(deviceName))) {
                return b;
            }
            b.close();
        }
        return null;
    }

    /** The kernel of cache type {@code type} in the case's module. */
    private KernelHandle kernel(KvDType type) {
        String name = AttnKernels.flashDecodeGqa(type, type);
        if (module == Module.BOUND) {
            return kernels.get(name);
        }
        return plainHandles.computeIfAbsent(type, t -> {
            KernelLibraryConfig c = KernelLibraries.LLM_CONFIG;
            KernelLibraryConfig plain = new KernelLibraryConfig(c.libraryId(), c.sourceRoot(), c.cudaRoot(),
                    c.spvRoot(), c.requireCompleteInventory(), c.openClOptions(), false, c.vulkanFeatures());
            return backend.loadKernel(new VulkanKernelLibrary(plain).resolve(name, backend));
        });
    }

    @After
    public void closeDevice() {
        for (GpuBuffer b : buffers) {
            b.close();
        }
        buffers.clear();
        for (KernelHandle h : plainHandles.values()) {
            h.close();
        }
        plainHandles.clear();
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
    }

    @Test
    public void planSplitsMatchTheCpuReference() throws InterruptedException {
        int seed = 1;
        for (KvDType type : new KvDType[]{KvDType.F16, KvDType.BF16, KvDType.Q8_0}) {
            for (int d : new int[]{64, 128}) {
                for (int g : new int[]{1, 2, 3, 4}) {
                    Case c = new Case(type, d, 2, g, seed++, false);
                    FlashAttnTiling plan = FlashAttnTiling.decodeOnline(c.nHead, c.nHeadKv, d, d, N_KV,
                            kernels.faSmemFloats(), Math.max(1, backend.capabilities().getComputeUnits()));
                    assertEquals(g, plan.gTile());
                    for (int n : KEY_COUNTS) {
                        FlashAttnTiling.Split s = plan.decodeSplit(n);
                        c.check(Step.causal(n), s.count(), s.length());
                    }
                }
            }
        }
    }

    /** One split over up to 2100 keys stages them in blocks of 512 inside one work-group. */
    @Test
    public void oneDirectSplitOverSeveralStagesMatchesTheCpuReference() throws InterruptedException {
        int seed = 40;
        for (int d : new int[]{64, 128}) {
            for (int g : new int[]{2, 4}) {
                Case c = new Case(KvDType.F16, d, 2, g, seed++, false);
                for (int n : new int[]{511, 512, 513, 1500, 2100}) {
                    c.check(Step.causal(n), 1, n);
                    c.check(Step.causal(n), 3, (n + 2) / 3);
                }
            }
        }
    }

    @Test
    public void windowsBiasesSinksAndEmptyCellsMatchTheCpuReference() throws InterruptedException {
        int seed = 80;
        for (KvDType type : new KvDType[]{KvDType.F16, KvDType.BF16, KvDType.Q8_0}) {
            for (int d : new int[]{64, 128}) {
                Case c = new Case(type, d, 2, 4, seed++, true);
                Step[] steps = {
                    Step.causal(900).window(SwaType.STANDARD, 128),
                    Step.causal(2100).window(SwaType.STANDARD, 1000).softcap(30f),
                    Step.causal(700).range(40),
                    Step.causal(1200).queryAt(1100),
                    Step.causal(600).alibi(8f),
                    Step.causal(1300).relative(AttentionSpec.BiasMode.T5_BIDIR),
                    Step.causal(900).queryAt(700).relative(AttentionSpec.BiasMode.T5_CAUSAL),
                    Step.causal(1000).relative(AttentionSpec.BiasMode.INKLING),
                    Step.causal(800).sinks(),
                    Step.causal(3).sinks(),
                };
                for (Step st : steps) {
                    c.check(st, 1, st.kvEnd - st.kvStart);
                    c.check(st, 5, (st.kvEnd - st.kvStart + 4) / 5);
                }
            }
        }
    }

    /**
     * Per-head K and V scales of a Q8_0 cache apply to the scores and to the output. The engine binds all-ones scales
     * for a Q8_0 cache (only FP8 and NVFP4 caches are checkpoint-scaled); the kernels read them as the generic decode
     * reads {@code nf_kv_head_scale}, and this holds that contract.
     */
    @Test
    public void q8ScalesMatchTheCpuReference() throws InterruptedException {
        int seed = 120;
        for (int d : new int[]{64, 128}) {
            Case c = new Case(KvDType.Q8_0, d, 2, 4, seed++, false, new float[]{0.5f, 1.25f, 2f, 0.75f});
            for (int n : new int[]{9, 700}) {
                c.check(Step.causal(n), 1, n);
                c.check(Step.causal(n), 4, (n + 3) / 4);
            }
        }
    }

    /** One decode step's key range, query position, masking options and position bias. */
    private static final class Step {
        final int kvStart;
        final int kvEnd;
        final int qPos;
        final SwaType swa;
        final int nSwa;
        final float softcap;
        final AttentionSpec.BiasMode bias;
        final float maxBias;
        final boolean sinks;

        private Step(int kvStart, int kvEnd, int qPos, SwaType swa, int nSwa, float softcap,
                     AttentionSpec.BiasMode bias, float maxBias, boolean sinks) {
            this.kvStart = kvStart;
            this.kvEnd = kvEnd;
            this.qPos = qPos;
            this.swa = swa;
            this.nSwa = nSwa;
            this.softcap = softcap;
            this.bias = bias;
            this.maxBias = maxBias;
            this.sinks = sinks;
        }

        static Step causal(int n) {
            return new Step(0, n, n - 1, SwaType.NONE, 0, 0f, AttentionSpec.BiasMode.NONE, 0f, false);
        }

        Step window(SwaType type, int n) {
            return new Step(kvStart, kvEnd, qPos, type, n, softcap, bias, maxBias, sinks);
        }

        Step softcap(float cap) {
            return new Step(kvStart, kvEnd, qPos, swa, nSwa, cap, bias, maxBias, sinks);
        }

        Step range(int start) {
            return new Step(start, kvEnd, qPos, swa, nSwa, softcap, bias, maxBias, sinks);
        }

        Step queryAt(int pos) {
            return new Step(kvStart, kvEnd, pos, swa, nSwa, softcap, bias, maxBias, sinks);
        }

        Step alibi(float maxBias) {
            return new Step(kvStart, kvEnd, qPos, swa, nSwa, softcap, AttentionSpec.BiasMode.ALIBI, maxBias, sinks);
        }

        /** A T5 or Inkling relative bias, read from the case's tables. */
        Step relative(AttentionSpec.BiasMode mode) {
            return new Step(kvStart, kvEnd, qPos, swa, nSwa, softcap, mode, 0f, sinks);
        }

        Step sinks() {
            return new Step(kvStart, kvEnd, qPos, swa, nSwa, softcap, bias, maxBias, true);
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "[%d, %d) q_pos %d swa %s/%d softcap %.0f bias %s/%.0f sinks %b",
                    kvStart, kvEnd, qPos, swa, nSwa, softcap, bias, maxBias, sinks);
        }
    }

    /** One random dense cache and query row on the device. */
    private final class Case {
        final KvDType type;
        final int d;
        final int nHeadKv;
        final int nHead;
        final int rowStride;
        final float[] hq;
        final float[] hk;
        final float[] hv;
        final float[] hSinks;
        final float[] hT5;
        final float[] hInkling;
        final float[] hScales;
        final int[] hPos;
        final GpuBuffer q;
        final GpuBuffer k;
        final GpuBuffer v;
        final GpuBuffer kvPos;
        final GpuBuffer sinks;
        final GpuBuffer t5;
        final GpuBuffer inkling;
        final GpuBuffer dummy;
        final GpuBuffer ones;
        final GpuBuffer out;
        final GpuBuffer partial;

        Case(KvDType type, int d, int nHeadKv, int g, long seed, boolean emptyCells) throws InterruptedException {
            this(type, d, nHeadKv, g, seed, emptyCells, null);
        }

        /** {@code scales}: the K scales of the KV heads then their V scales, or null for none (all ones). */
        Case(KvDType type, int d, int nHeadKv, int g, long seed, boolean emptyCells, float[] scales)
                throws InterruptedException {
            this.type = type;
            this.d = d;
            this.nHeadKv = nHeadKv;
            this.nHead = nHeadKv * g;
            this.rowStride = nHeadKv * d + (type == KvDType.Q8_0 ? 32 : 8);
            Random rnd = new Random(seed);
            hPos = new int[N_KV];
            for (int i = 0; i < N_KV; i++) {
                hPos[i] = emptyCells && i % 19 == 7 ? -1 : i;
            }
            hq = randoms(rnd, nHead * d, 1f);
            hk = stored(randoms(rnd, N_KV * rowStride, 1f));
            hv = stored(randoms(rnd, N_KV * rowStride, 1f));
            hSinks = randoms(rnd, nHead, 2f);
            hT5 = randoms(rnd, T5_BUCKETS * nHead, 2f);
            hInkling = randoms(rnd, nHead * INKLING_EXTENT, 2f);
            q = floats(hq);
            k = cache(hk);
            v = cache(hv);
            kvPos = ints(hPos);
            sinks = floats(hSinks);
            t5 = floats(hT5);
            inkling = floats(hInkling);
            dummy = floats(new float[16]);
            hScales = new float[2 * nHeadKv];
            Arrays.fill(hScales, 1f);
            if (scales != null) {
                System.arraycopy(scales, 0, hScales, 0, hScales.length);
            }
            ones = floats(hScales);
            out = floats(new float[nHead * d]);
            partial = floats(new float[nHead * MAX_SPLITS * (d + 2)]);
        }

        /** Runs the step in {@code nSplit} splits of {@code splitLen} keys and checks the rows against the CPU. */
        void check(Step st, int nSplit, int splitLen) throws InterruptedException {
            int n = st.kvEnd - st.kvStart;
            int splits = Math.max(nSplit, (n + splitLen - 1) / splitLen);
            assertTrue(splits + " splits exceed the partial buffer", splits <= MAX_SPLITS);
            float[] got = run(st, splits, splitLen);
            float[] want = new float[nHead * d];
            CpuAttentionCore.Builder b = CpuAttentionCore.Params.builder(nHead)
                    .scale((float) (1.0 / Math.sqrt(d))).softcap(st.softcap)
                    .window(KvWindow.of(st.swa, st.nSwa, true));
            if (st.bias == AttentionSpec.BiasMode.ALIBI) {
                b.alibi(st.maxBias);
            } else if (st.bias == AttentionSpec.BiasMode.T5_BIDIR || st.bias == AttentionSpec.BiasMode.T5_CAUSAL) {
                b.t5(st.bias == AttentionSpec.BiasMode.T5_BIDIR, T5_BUCKETS, T5_MAX_DISTANCE, hT5);
            } else if (st.bias == AttentionSpec.BiasMode.INKLING) {
                b.inkling(INKLING_EXTENT, hInkling);
            }
            if (st.sinks) {
                b.sinks(hSinks);
            }
            CpuAttentionCore.attendRows(b.build(), 1, hq, 0, nHead * d, new int[]{st.qPos}, 0,
                    CpuAttentionCore.DenseKv.of(scaled(hk, 0), 0, rowStride, d, scaled(hv, nHeadKv), 0, rowStride, d,
                            nHeadKv, d, d, hPos, 0),
                    st.kvStart, st.kvEnd, want, 0, nHead * d);
            String where = String.format(Locale.ROOT, "%s d=%d heads=%d/%d %s splits=%d x %d", type, d, nHead,
                    nHeadKv, st, splits, splitLen);
            for (int i = 0; i < want.length; i++) {
                double tol = 1e-4 * Math.max(1.0, Math.abs(want[i]));
                assertTrue(where + " output " + i + ": cpu " + want[i] + ", gpu " + got[i],
                        Math.abs(want[i] - got[i]) <= tol);
            }
        }

        private float[] run(Step st, int splits, int splitLen) throws InterruptedException {
            boolean direct = splits == 1;
            int flags = AttnKernels.FLAG_CAUSAL | (direct ? AttnKernels.FLAG_DIRECT : 0)
                    | (st.sinks ? AttnKernels.FLAG_SINKS : 0);
            float[] poison = new float[nHead * d];
            Arrays.fill(poison, Float.NaN);
            backend.computeQueue().enqueueWriteBuffer(out, 0, (long) poison.length * 4, poison).awaitCompletion();
            Object[] args = new Object[AttnKernels.FD_ARG_COUNT];
            args[AttnKernels.ARG_FD_N_HEAD] = nHead;
            args[AttnKernels.ARG_FD_N_HEAD_KV] = nHeadKv;
            args[AttnKernels.ARG_FD_D_K] = d;
            args[AttnKernels.ARG_FD_D_V] = d;
            args[AttnKernels.ARG_FD_G_TILE] = nHead / nHeadKv;
            args[AttnKernels.ARG_FD_N_SPLIT] = splits;
            args[AttnKernels.ARG_FD_SPLIT_LEN] = splitLen;
            args[AttnKernels.ARG_FD_KV_START] = st.kvStart;
            args[AttnKernels.ARG_FD_KV_END] = st.kvEnd;
            args[AttnKernels.ARG_FD_Q_POS] = st.qPos;
            args[AttnKernels.ARG_FD_K_ROW_STRIDE] = rowStride;
            args[AttnKernels.ARG_FD_K_HEAD_STRIDE] = d;
            args[AttnKernels.ARG_FD_V_ROW_STRIDE] = rowStride;
            args[AttnKernels.ARG_FD_V_HEAD_STRIDE] = d;
            args[AttnKernels.ARG_FD_SCALE] = (float) (1.0 / Math.sqrt(d));
            args[AttnKernels.ARG_FD_SOFTCAP] = st.softcap;
            args[AttnKernels.ARG_FD_MAX_BIAS] = st.maxBias;
            args[AttnKernels.ARG_FD_FLAGS] = flags;
            args[AttnKernels.ARG_FD_SWA_TYPE] = st.swa.code();
            args[AttnKernels.ARG_FD_N_SWA] = st.nSwa;
            boolean t5Bias = st.bias == AttentionSpec.BiasMode.T5_BIDIR || st.bias == AttentionSpec.BiasMode.T5_CAUSAL;
            boolean inklingBias = st.bias == AttentionSpec.BiasMode.INKLING;
            args[AttnKernels.ARG_FD_BIAS_MODE] = st.bias.code();
            args[AttnKernels.ARG_FD_N_BUCKETS] = t5Bias ? T5_BUCKETS : inklingBias ? INKLING_EXTENT : 0;
            args[AttnKernels.ARG_FD_MAX_DISTANCE] = t5Bias ? T5_MAX_DISTANCE : 0;
            args[AttnKernels.ARG_FD_Q_OFF] = 0;
            args[AttnKernels.ARG_FD_OUT_OFF] = 0;
            args[AttnKernels.ARG_FD_BUF_Q] = q;
            args[AttnKernels.ARG_FD_BUF_K] = k;
            args[AttnKernels.ARG_FD_BUF_V] = v;
            args[AttnKernels.ARG_FD_BUF_KV_POS] = kvPos;
            args[AttnKernels.ARG_FD_BUF_SINKS] = sinks;
            args[AttnKernels.ARG_FD_BUF_REL_BIAS] = t5Bias ? t5 : inklingBias ? inkling : dummy;
            args[AttnKernels.ARG_FD_BUF_OUT] = out;
            args[AttnKernels.ARG_FD_BUF_PARTIAL] = partial;
            args[AttnKernels.ARG_FD_BUF_KV_SCALES] = ones;
            args[AttnKernels.ARG_FD_BUF_TREE_MASK] = dummy;
            backend.computeQueue().enqueueKernel(kernel(type),
                    FlashAttnTiling.decodeGqaGlobal(1, nHeadKv, splits), FlashAttnTiling.decodeLocal(), args);
            if (!direct) {
                Object[] combine = {nHead, d, splits, nHead, st.sinks ? 1 : 0, 0, partial, sinks, out};
                backend.computeQueue().enqueueKernel(kernels.get(AttnKernels.FLASH_COMBINE),
                        FlashAttnTiling.combineGlobal(nHead), FlashAttnTiling.combineLocal(), combine);
            }
            float[] y = new float[nHead * d];
            backend.computeQueue().enqueueReadBuffer(out, 0, (long) y.length * 4, y).awaitCompletion();
            return y;
        }

        /** The cache values {@code x} times their head's scale ({@code first}: the scales' K or V half). */
        private float[] scaled(float[] x, int first) {
            float[] r = x.clone();
            for (int row = 0; row < N_KV; row++) {
                for (int h = 0; h < nHeadKv; h++) {
                    for (int e = 0; e < d; e++) {
                        r[row * rowStride + h * d + e] *= hScales[first + h];
                    }
                }
            }
            return r;
        }

        /** The values the cache stores for {@code x}: rounded or quantized to the cache type. */
        private float[] stored(float[] x) {
            return KvCacheTestValues.stored(type, x);
        }

        private GpuBuffer cache(float[] values) throws InterruptedException {
            return bytes(KvCacheTestValues.bytes(type, values));
        }
    }

    private static float[] randoms(Random rnd, int n, float scale) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (rnd.nextFloat() * 2f - 1f) * scale;
        }
        return x;
    }

    private GpuBuffer floats(float[] x) throws InterruptedException {
        GpuBuffer b = backend.allocateBuffer((long) x.length * 4);
        buffers.add(b);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) x.length * 4, x).awaitCompletion();
        return b;
    }

    private GpuBuffer ints(int[] x) throws InterruptedException {
        ByteBuffer bb = ByteBuffer.allocate(x.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i : x) {
            bb.putInt(i);
        }
        return bytes(bb.array());
    }

    private GpuBuffer bytes(byte[] x) throws InterruptedException {
        GpuBuffer b = backend.allocateBuffer(x.length);
        buffers.add(b);
        backend.computeQueue().enqueueWriteBuffer(b, 0, x.length, x).awaitCompletion();
        return b;
    }
}
