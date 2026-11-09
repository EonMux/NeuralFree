package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Opt-in timing of the flash-decode split quantum (environment {@code NEURALFREE_BENCH_FLASH=1}; the test
 * task forwards no system properties): the same key count and device run through the split plan of
 * {@link FlashAttnTiling#decodeSplit} (quantum {@value #QUANTUM_NARROW} keys) and through the same plan with
 * the target number of splits rounded to 16-key splits, in interleaved rounds with per-round means and the
 * median reported, so a shared device's load variance cancels between the two. Asserts nothing; the numbers
 * go to the decode throughput documentation.
 */
@RunWith(Parameterized.class)
public class FlashDecodeSplitBenchmarkTest {

    private static final String SWITCH = "NEURALFREE_BENCH_FLASH";
    private static final int QUANTUM_NARROW = 16;
    private static final int N_HEAD = 32;
    private static final int N_HEAD_KV = 8;
    private static final int D_K = 128;
    private static final int D_V = 128;
    private static final int G_TILE = 1;
    private static final int N_KV_MAX = 256;
    private static final int WARMUP = 10;
    private static final int ROUNDS = 9;
    private static final int RUNS_PER_ROUND = 20;
    /** Live key counts of a short decode: the range a 128-token generation walks through. */
    private static final int[] KEYS = {8, 16, 20, 24, 32, 40, 48, 64, 96, 128};

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private boolean previousCuda;

    public FlashDecodeSplitBenchmarkTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        Assume.assumeTrue("set " + SWITCH + "=1 to run", "1".equals(System.getenv(SWITCH)));
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    private ComputeBackend openBackend() {
        if (api.equals("vulkan")) {
            return new VkComputeBackend();
        }
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            HWAcceleration accel = new HWAcceleration();
            hwAccel = accel;
            Assume.assumeFalse("no CUDA device", accel.getCUDevices().isEmpty());
            CUDevice device = accel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        HWAcceleration accel = new HWAcceleration();
        hwAccel = accel;
        Assume.assumeFalse("no OpenCL device", accel.getCLDevices().isEmpty());
        return new OclComputeBackend(accel.getCLDevices().get(0));
    }

    @After
    public void closeDevice() {
        if (backend != null) {
            try {
                backend.close();
            } catch (RuntimeException ignored) {
            }
            backend = null;
        }
        if (hwAccel != null) {
            hwAccel.shutdown();
            hwAccel = null;
        }
        HWAcceleration.enableCuda = previousCuda;
    }

    @Test
    public void narrowSplitQuantumAgainstTheNarrowestPlannedSplit() throws Exception {
        int smem = kernels.faSmemFloats();
        int computeUnits = Math.max(1, backend.capabilities().getComputeUnits());
        FlashAttnTiling plan = FlashAttnTiling.decode(N_HEAD, N_HEAD_KV, D_K, D_V, N_KV_MAX, smem, computeUnits,
                api.equals("cuda"));
        int groupTiles = plan.groupTiles();
        int wanted = (int) Math.max(1L, ceilDiv(2L * computeUnits, (long) N_HEAD_KV * groupTiles));
        int maxLen = plan.maxSplitLen();
        int maxSplits = Math.max(ceilDiv(N_KV_MAX, maxLen), Math.min(wanted, ceilDiv(N_KV_MAX, QUANTUM_NARROW)));
        System.out.println(String.format(Locale.ROOT,
                "device %s (%s), wanted %d splits, split_len 32..%d, partial splits %d",
                backend.capabilities().getDeviceName(), api, wanted, maxLen, maxSplits));

        GpuBuffer q = floats(N_HEAD * D_K);
        GpuBuffer kv = floats(N_KV_MAX * N_HEAD_KV * Math.max(D_K, D_V));
        GpuBuffer kvPos = ints(N_KV_MAX);
        GpuBuffer sinks = floats(1);
        GpuBuffer relBias = floats(1);
        GpuBuffer out = floats(N_HEAD * D_V);
        GpuBuffer partial = floats((long) N_HEAD * maxSplits * (D_V + 2L));
        GpuBuffer kvScales = floats(1);
        try {
            fill(q, 0.1f);
            fill(kv, 0.1f);
            fill(sinks, 0f);
            fill(relBias, 0f);
            fill(kvScales, 1f);
            ByteBuffer positions = ByteBuffer.allocate(N_KV_MAX * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            for (int t = 0; t < N_KV_MAX; t++) {
                positions.putInt(t);
            }
            backend.computeQueue().enqueueWriteBuffer(kvPos, 0, (long) N_KV_MAX * Integer.BYTES,
                    positions.array()).awaitCompletion();
            String kernel = requireKernel(AttnKernels.flashDecode(KvDType.F16));
            Object[] decodeArgs = decodeArgs(q, kv, kvPos, sinks, relBias, out, partial, kvScales);
            Object[] combineArgs = {N_HEAD, D_V, 1, N_HEAD, 0, 0, partial, sinks, out};
            for (int nKeys : KEYS) {
                FlashAttnTiling.Split wide = split(nKeys, wanted, FlashAttnTiling.SPLIT_QUANTUM, maxLen);
                FlashAttnTiling.Split narrow = split(nKeys, wanted, QUANTUM_NARROW, maxLen);
                long[] wideRounds = new long[ROUNDS];
                long[] narrowRounds = new long[ROUNDS];
                for (int r = 0; r < WARMUP; r++) {
                    launch(kernel, decodeArgs, combineArgs, wide, nKeys);
                    launch(kernel, decodeArgs, combineArgs, narrow, nKeys);
                }
                backend.computeQueue().finish();
                for (int r = 0; r < ROUNDS; r++) {
                    wideRounds[r] = timeRound(() -> launch(kernel, decodeArgs, combineArgs, wide, nKeys));
                    narrowRounds[r] = timeRound(() -> launch(kernel, decodeArgs, combineArgs, narrow, nKeys));
                }
                double wideUs = median(wideRounds) / 1e3;
                double narrowUs = median(narrowRounds) / 1e3;
                System.out.println(String.format(Locale.ROOT,
                        "%-5s keys %3d  wide %2dx%-3d %8.3f us  narrow %2dx%-3d %8.3f us  ratio %5.2fx",
                        api, nKeys, wide.count(), wide.length(), wideUs,
                        narrow.count(), narrow.length(), narrowUs, wideUs / narrowUs));
            }
        } finally {
            q.close();
            kv.close();
            kvPos.close();
            sinks.close();
            relBias.close();
            out.close();
            partial.close();
            kvScales.close();
        }
    }

    private String requireKernel(String name) {
        Assume.assumeTrue(api + " build has no " + name + " kernel", kernels.has(name));
        return name;
    }

    /** The decode plan of {@code nKeys} with the split length rounded to {@code quantum} keys. */
    private static FlashAttnTiling.Split split(int nKeys, int wanted, int quantum, int maxLen) {
        int length = roundUp(ceilDiv(nKeys, wanted), quantum);
        length = Math.min(Math.max(length, quantum), maxLen);
        return new FlashAttnTiling.Split(Math.max(1, ceilDiv(nKeys, length)), length);
    }

    private void launch(String kernel, Object[] args, Object[] combineArgs, FlashAttnTiling.Split split,
                        int nKeys) {
        boolean direct = split.count() == 1;
        args[AttnKernels.ARG_FD_N_HEAD] = N_HEAD;
        args[AttnKernels.ARG_FD_N_HEAD_KV] = N_HEAD_KV;
        args[AttnKernels.ARG_FD_D_K] = D_K;
        args[AttnKernels.ARG_FD_D_V] = D_V;
        args[AttnKernels.ARG_FD_G_TILE] = G_TILE;
        args[AttnKernels.ARG_FD_N_SPLIT] = split.count();
        args[AttnKernels.ARG_FD_SPLIT_LEN] = split.length();
        args[AttnKernels.ARG_FD_KV_START] = 0;
        args[AttnKernels.ARG_FD_KV_END] = nKeys;
        args[AttnKernels.ARG_FD_Q_POS] = nKeys - 1;
        args[AttnKernels.ARG_FD_FLAGS] = AttnKernels.attnFlags(true, false, direct) | AttnKernels.FLAG_V_ALIAS;
        backend.computeQueue().enqueueKernel(kernels.get(kernel),
                new long[]{(long) split.count() * FlashAttnTiling.WG,
                        (long) N_HEAD_KV * ceilDiv(N_HEAD / N_HEAD_KV, G_TILE)},
                FlashAttnTiling.decodeLocal(), args);
        if (!direct) {
            combineArgs[AttnKernels.ARG_FC_N_SPLIT] = split.count();
            backend.computeQueue().enqueueKernel(kernels.get(AttnKernels.FLASH_COMBINE),
                    FlashAttnTiling.combineGlobal(N_HEAD), FlashAttnTiling.combineLocal(), combineArgs);
        }
    }

    private Object[] decodeArgs(GpuBuffer q, GpuBuffer kv, GpuBuffer kvPos, GpuBuffer sinks, GpuBuffer relBias,
                                GpuBuffer out, GpuBuffer partial, GpuBuffer kvScales) {
        Object[] a = new Object[AttnKernels.FD_ARG_COUNT];
        a[AttnKernels.ARG_FD_K_ROW_STRIDE] = N_HEAD_KV * D_K;
        a[AttnKernels.ARG_FD_K_HEAD_STRIDE] = D_K;
        a[AttnKernels.ARG_FD_V_ROW_STRIDE] = N_HEAD_KV * D_V;
        a[AttnKernels.ARG_FD_V_HEAD_STRIDE] = D_V;
        a[AttnKernels.ARG_FD_SCALE] = (float) (1.0 / Math.sqrt(D_K));
        a[AttnKernels.ARG_FD_SOFTCAP] = 0f;
        a[AttnKernels.ARG_FD_MAX_BIAS] = 0f;
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
        a[AttnKernels.ARG_FD_BUF_KV_SCALES] = kvScales;
        /* The one-float buffer stands in for the tree mask, as the production placeholder does: the kernels
         * never read the slot without the tree flag these launches never set. */
        a[AttnKernels.ARG_FD_BUF_TREE_MASK] = relBias;
        return a;
    }

    private GpuBuffer floats(long n) {
        return backend.allocateBuffer(n * Float.BYTES);
    }

    private GpuBuffer ints(long n) {
        return backend.allocateBuffer(n * Integer.BYTES);
    }

    private void fill(GpuBuffer b, float value) throws InterruptedException {
        float[] values = new float[(int) (b.sizeInBytes() / Float.BYTES)];
        Arrays.fill(values, value);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) values.length * Float.BYTES, values)
                .awaitCompletion();
    }

    private long timeRound(Runnable launches) {
        for (int i = 0; i < RUNS_PER_ROUND; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        long start = System.nanoTime();
        for (int i = 0; i < RUNS_PER_ROUND; i++) {
            launches.run();
        }
        backend.computeQueue().finish();
        return (System.nanoTime() - start) / RUNS_PER_ROUND;
    }

    private static long median(long[] values) {
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static int ceilDiv(int a, int b) {
        return a / b + (a % b != 0 ? 1 : 0);
    }

    private static long ceilDiv(long a, long b) {
        return (a + b - 1) / b;
    }

    private static int roundUp(int a, int m) {
        return ceilDiv(a, m) * m;
    }
}
