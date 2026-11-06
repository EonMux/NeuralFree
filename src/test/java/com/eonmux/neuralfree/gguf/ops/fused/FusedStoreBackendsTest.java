package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
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
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The fused K-row store kernels ({@code LLM_NormRopeStore} and {@code LLM_RopeStore}) against the unfused
 * normalization, rotation and cache store they replace, on every GPU backend. Both sides run on the same device, so
 * the expected values are the unfused path's; the tolerance covers only the float summation order of the
 * reductions. A backend is skipped when its GPU or its kernel library is unavailable.
 */
@RunWith(Parameterized.class)
public class FusedStoreBackendsTest {

    private static final float EPS = 1e-6f;
    private static final float TOL = 2e-5f;
    private static final int N_TOKENS = 2;
    private static final int N_HEADS = 2;
    private static final int HEAD = 8;
    private static final int ROW_STRIDE = N_HEADS * HEAD;
    private static final int CELLS = 3;
    private static final int DST_ROW = 20;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;
    private HWAcceleration hwAccel;
    private ComputeBackend backend;
    private LlmKernels kernels;
    private GpuOps ops;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private int local;
    private boolean previousCuda;

    public FusedStoreBackendsTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        previousCuda = HWAcceleration.enableCuda;
        try {
            backend = openBackend();
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
            kernels = LlmKernels.open(backend);
            local = kernels.matvecLocal();
            scratch = GpuScratch.builder(backend, 8).slot(ScratchSlots.X, 8L * 64).build();
            placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            ops = new GpuOps(backend.computeQueue(), kernels, scratch, placeholders);
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        if (ops != null) {
            ops.close();
            ops = null;
        }
        if (placeholders != null) {
            placeholders.close();
            placeholders = null;
        }
        if (scratch != null) {
            scratch.close();
            scratch = null;
        }
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

    private ComputeBackend openBackend() {
        if (api.equals("vulkan")) {
            return new VkComputeBackend();
        }
        if (api.equals("cuda")) {
            HWAcceleration.enableCuda = true;
            hwAccel = new HWAcceleration();
            Assume.assumeFalse("no CUDA device", hwAccel.getCUDevices().isEmpty());
            CUDevice device = hwAccel.getCUDevices().get(0);
            return new CudaComputeBackend(device);
        }
        hwAccel = new HWAcceleration();
        Assume.assumeFalse("no OpenCL device", hwAccel.getCLDevices().isEmpty());
        return new OclComputeBackend(hwAccel.getCLDevices().get(0));
    }

    // ------------------------------------------------------------------ fixtures

    private GpuBuffer floats(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Float.BYTES);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data);
        return b;
    }

    private GpuBuffer ints(int[] data) {
        ByteBuffer le = ByteBuffer.allocate(data.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        le.asIntBuffer().put(data);
        GpuBuffer b = backend.allocateBuffer(le.capacity());
        backend.computeQueue().enqueueWriteBuffer(b, 0, le.capacity(), le);
        return b;
    }

    private GpuBuffer sentinel(long bytes, byte value) {
        GpuBuffer b = backend.allocateBuffer(bytes);
        byte[] fill = new byte[(int) bytes];
        Arrays.fill(fill, value);
        backend.computeQueue().enqueueWriteBuffer(b, 0, bytes, fill);
        return b;
    }

    private float[] read(GpuBuffer b, int n) {
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out);
        backend.computeQueue().finish();
        return out;
    }

    private byte[] readBytes(GpuBuffer b, int n) {
        byte[] out = new byte[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, n, out);
        backend.computeQueue().finish();
        return out;
    }

    private static float[] random(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }

    private static void near(float[] expected, float[] actual, String what) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(what + "[" + i + "] on " + expected.length + " values", expected[i], actual[i],
                    TOL * (1f + Math.abs(expected[i])));
        }
    }

    private void run(String kernel, long global, Object[] args) {
        backend.computeQueue().enqueueKernel(kernels.get(kernel), new long[]{global}, new long[]{local}, args);
    }

    private void runKvStore(KvDType type, GpuBuffer src, GpuBuffer cellIdx, GpuBuffer dst) {
        long items = type.perTokenHead()
                ? RopeKernels.kvStoreWorkItemsPerTokenHead(type, N_TOKENS, ROW_STRIDE, HEAD)
                : RopeKernels.kvStoreWorkItems(type, N_TOKENS, ROW_STRIDE);
        run(RopeKernels.kvStore(type), ActKernels.global1d(items, local)[0],
                RopeKernels.kvStoreArgs(type, N_TOKENS, ROW_STRIDE, ROW_STRIDE, 0, DST_ROW, HEAD, src, cellIdx,
                        placeholders.kvScalesOnes(), dst, placeholders.kvScalesOnes(), 0));
    }

    private static RopeSpec neox() {
        return RopeSpec.builder(RopeType.NEOX).nDims(HEAD).nOffs(0).freqBase(10000f).build();
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void normRopeStoreMatchesNormRopeThenStoreF32() {
        assertNormRopeStoreF32(neox(), 201, "norm-rope store F32 on " + api);
    }

    /** The interleaved pair mapping of the non-NeoX rotations, which shares the kernel with the NeoX branch. */
    @Test
    public void normRopeStoreMatchesNormRopeThenStoreF32InNormMode() {
        RopeSpec rope = RopeSpec.builder(RopeType.NORM).nDims(HEAD).nOffs(0).freqBase(10000f).build();
        assertNormRopeStoreF32(rope, 241, "norm-mode norm-rope store F32 on " + api);
    }

    private void assertNormRopeStoreF32(RopeSpec rope, long seed, String what) {
        Assume.assumeTrue("backend has no norm-rope store on " + api, ops.fused(FusedKernels.NORM_ROPE_STORE));
        float[] x = random(N_TOKENS * ROW_STRIDE, seed);
        float[] w = random(ROW_STRIDE, seed + 1);
        float[] b = random(ROW_STRIDE, seed + 2);
        int cacheElems = CELLS * DST_ROW;
        GpuBuffer xb = floats(x);
        GpuBuffer wb = floats(w);
        GpuBuffer bb = floats(b);
        GpuBuffer pos = ints(new int[]{3, 5});
        GpuBuffer cellIdx = ints(new int[]{2, 0});
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer tmp = floats(x);
        GpuBuffer ref = sentinel((long) cacheElems * Float.BYTES, (byte) 0xAB);
        GpuBuffer out = sentinel((long) cacheElems * Float.BYTES, (byte) 0xAB);

        run(NormKernels.ROW_NORM, (long) N_TOKENS * N_HEADS * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, N_TOKENS * N_HEADS, HEAD, HEAD, tmp, 0, tmp, 0, wb, bb,
                        N_HEADS, ff));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), N_TOKENS, N_HEADS, rope.nDims()),
                        local)[0],
                RopeKernels.ropeExtArgs(rope, N_TOKENS, N_HEADS, HEAD, ROW_STRIDE, 0, N_TOKENS, 0, false, pos, ff,
                        tmp));
        runKvStore(KvDType.F32, tmp, cellIdx, ref);

        run(FusedKernels.NORM_ROPE_STORE, FusedKernels.normRopeStoreGlobal(N_TOKENS, N_HEADS, local),
                FusedKernels.normRopeStoreArgs(rope, N_TOKENS, N_HEADS, HEAD, ROW_STRIDE, HEAD, 0, N_TOKENS, 0,
                        NormKernels.FLAG_HAS_W | NormKernels.FLAG_HAS_B, N_HEADS, EPS, false, HEAD, DST_ROW,
                        FusedKernels.KV_STORE_F32, pos, ff, wb, bb, xb, cellIdx, out));

        near(read(ref, cacheElems), read(out, cacheElems), what);
    }

    @Test
    public void normRopeStoreMatchesNormRopeThenStoreF16() {
        Assume.assumeTrue("backend has no norm-rope store on " + api, ops.fused(FusedKernels.NORM_ROPE_STORE));
        RopeSpec rope = neox();
        float[] x = random(N_TOKENS * ROW_STRIDE, 211);
        float[] w = random(ROW_STRIDE, 212);
        int cacheBytes = CELLS * DST_ROW * 2;
        GpuBuffer xb = floats(x);
        GpuBuffer wb = floats(w);
        GpuBuffer pos = ints(new int[]{3, 5});
        GpuBuffer cellIdx = ints(new int[]{2, 0});
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer tmp = floats(x);
        GpuBuffer ref = sentinel(cacheBytes, (byte) 0xAB);
        GpuBuffer out = sentinel(cacheBytes, (byte) 0xAB);

        run(NormKernels.ROW_NORM, (long) N_TOKENS * N_HEADS * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, N_TOKENS * N_HEADS, HEAD, HEAD, tmp, 0, tmp, 0, wb, null,
                        N_HEADS, ff));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), N_TOKENS, N_HEADS, rope.nDims()),
                        local)[0],
                RopeKernels.ropeExtArgs(rope, N_TOKENS, N_HEADS, HEAD, ROW_STRIDE, 0, N_TOKENS, 0, false, pos, ff,
                        tmp));
        runKvStore(KvDType.F16, tmp, cellIdx, ref);

        run(FusedKernels.NORM_ROPE_STORE, FusedKernels.normRopeStoreGlobal(N_TOKENS, N_HEADS, local),
                FusedKernels.normRopeStoreArgs(rope, N_TOKENS, N_HEADS, HEAD, ROW_STRIDE, HEAD, 0, N_TOKENS, 0,
                        NormKernels.FLAG_HAS_W, N_HEADS, EPS, false, HEAD, DST_ROW, FusedKernels.KV_STORE_F16, pos,
                        ff, wb, placeholders.readOnlyFloats(), xb, cellIdx, out));

        assertArrayEquals("norm-rope store F16 bytes on " + api, readBytes(ref, cacheBytes),
                readBytes(out, cacheBytes));
    }

    @Test
    public void ropeStoreMatchesRopeThenStoreF32() {
        assertRopeStoreMatches(neox(), 221, false, "rope store F32 on " + api);
    }

    /** The interleaved pair mapping of the non-NeoX rotations, which shares the kernel with the NeoX branch. */
    @Test
    public void ropeStoreMatchesRopeThenStoreF32InNormMode() {
        RopeSpec rope = RopeSpec.builder(RopeType.NORM).nDims(HEAD).nOffs(0).freqBase(10000f).build();
        assertRopeStoreMatches(rope, 231, false, "norm-mode rope store F32 on " + api);
    }

    /** An F16 destination writes the same halves the separate store writes, byte for byte. */
    @Test
    public void ropeStoreMatchesRopeThenStoreF16() {
        assertRopeStoreMatches(neox(), 251, true, "rope store F16 on " + api);
    }

    private void assertRopeStoreMatches(RopeSpec rope, long seed, boolean f16, String what) {
        Assume.assumeTrue("backend has no rope store on " + api, ops.fused(FusedKernels.ROPE_STORE));
        float[] x = random(N_TOKENS * ROW_STRIDE, seed);
        int cacheElems = CELLS * DST_ROW;
        int bytesPerElem = f16 ? Short.BYTES : Float.BYTES;
        GpuBuffer xb = floats(x);
        GpuBuffer pos = ints(new int[]{7, 9});
        GpuBuffer cellIdx = ints(new int[]{1, 2});
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer tmp = floats(x);
        GpuBuffer ref = sentinel((long) cacheElems * bytesPerElem, (byte) 0xAB);
        GpuBuffer out = sentinel((long) cacheElems * bytesPerElem, (byte) 0xAB);

        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), N_TOKENS, N_HEADS, rope.nDims()),
                        local)[0],
                RopeKernels.ropeExtArgs(rope, N_TOKENS, N_HEADS, HEAD, ROW_STRIDE, 0, N_TOKENS, 0, false, pos, ff,
                        tmp));
        runKvStore(f16 ? KvDType.F16 : KvDType.F32, tmp, cellIdx, ref);

        int dstType = f16 ? FusedKernels.KV_STORE_F16 : FusedKernels.KV_STORE_F32;
        run(FusedKernels.ROPE_STORE, FusedKernels.normRopeStoreGlobal(N_TOKENS, N_HEADS, local),
                FusedKernels.ropeStoreArgs(rope, N_TOKENS, N_HEADS, HEAD, ROW_STRIDE, 0, N_TOKENS, 0, false, HEAD,
                        DST_ROW, dstType, pos, ff, xb, cellIdx, out));

        if (f16) {
            assertArrayEquals(what, readBytes(ref, cacheElems * bytesPerElem),
                    readBytes(out, cacheElems * bytesPerElem));
        } else {
            near(read(ref, cacheElems), read(out, cacheElems), what);
        }
    }
}
