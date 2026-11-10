package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import org.junit.After;
import org.junit.Assume;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * {@code LLM_RmsNormRope} of shaders/llm/fused.comp against the unfused {@code LLM_RowNorm} then
 * {@code LLM_RoPE_Ext} pair on every Vulkan device, at the shape of a Qwen3 prefill (128-dim heads, several tokens
 * and heads per launch) and in place as the attention mixer launches it: the rotation reads normalized values that
 * other invocations of the work-group wrote, so this checks that the work-group barrier between the passes makes
 * them visible. Devices are labelled by index and name so a run can select one (e.g. {@code --tests '*Intel*'}).
 */
@RunWith(Parameterized.class)
public class RmsNormRopeVulkanGpuTest {

    private static final float TOL = 2e-5f;
    private static final float EPS = 1e-6f;
    private static final int TOKENS = 37;
    private static final int HEADS = 6;
    private static final int HEAD_DIM = 128;

    private final int deviceIndex;
    private HWAcceleration hw;
    private ComputeBackend backend;
    private GpuPlaceholders placeholders;
    private LlmKernels kernels;
    private int local;
    private final List<GpuBuffer> buffers = new ArrayList<>();

    public RmsNormRopeVulkanGpuTest(String label, int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> devices() {
        List<Object[]> devices = new ArrayList<>();
        try {
            HWAcceleration probe = new HWAcceleration(EnumSet.of(BackendType.VULKAN));
            try {
                List<VkDevice> found = probe.getVkDevices();
                for (int i = 0; i < found.size(); i++) {
                    devices.add(new Object[]{"vulkan:" + i + " " + found.get(i).getInfo().name(), i});
                }
            } finally {
                probe.shutdown();
            }
        } catch (RuntimeException | LinkageError e) {
            devices.clear();
        }
        if (devices.isEmpty()) {
            devices.add(new Object[]{"vulkan:none", -1});
        }
        return devices;
    }

    private void open() {
        Assume.assumeTrue("no Vulkan device", deviceIndex >= 0);
        hw = new HWAcceleration(EnumSet.of(BackendType.VULKAN));
        backend = hw.createBackend(hw.getVkDevices().get(deviceIndex));
        kernels = LlmKernels.open(backend);
        Assume.assumeTrue("the kernels hold no " + FusedKernels.RMS_NORM_ROPE,
                kernels.has(FusedKernels.RMS_NORM_ROPE));
        local = kernels.matvecLocal();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
    }

    @After
    public void close() {
        for (GpuBuffer b : buffers) {
            b.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (backend != null) {
            backend.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    private GpuBuffer floats(float[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Float.BYTES);
        buffers.add(b);
        backend.computeQueue().enqueueWriteBuffer(b, 0, (long) data.length * Float.BYTES, data);
        return b;
    }

    private GpuBuffer ints(int[] data) {
        GpuBuffer b = backend.allocateBuffer((long) data.length * Integer.BYTES);
        buffers.add(b);
        ByteBuffer le = ByteBuffer.allocate(data.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        le.asIntBuffer().put(data);
        backend.computeQueue().enqueueWriteBuffer(b, 0, le.capacity(), le);
        return b;
    }

    private float[] read(GpuBuffer b, int n) {
        backend.computeQueue().finish();
        float[] out = new float[n];
        backend.computeQueue().enqueueReadBuffer(b, 0, (long) n * Float.BYTES, out);
        backend.computeQueue().finish();
        return out;
    }

    private void run(String kernel, long global, Object[] args) {
        backend.computeQueue().enqueueKernelNoEvent(kernels.get(kernel), new long[]{global}, new long[]{local},
                args);
    }

    private static float[] random(int n, long seed) {
        Random rnd = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = rnd.nextFloat() * 2f - 1f;
        }
        return v;
    }

    @Test
    public void inPlaceAndOutOfPlaceMatchRowNormThenRope() {
        open();
        int rowStride = HEADS * HEAD_DIM;
        int total = TOKENS * rowStride;
        RopeSpec rope = RopeSpec.builder(RopeType.NEOX).nDims(HEAD_DIM).nOffs(0).freqBase(1000000f).build();
        float[] x = random(total, 21);
        int[] pos = new int[TOKENS];
        for (int t = 0; t < TOKENS; t++) {
            pos[t] = 3 + 5 * t;
        }
        GpuBuffer w = floats(random(HEAD_DIM, 22));
        GpuBuffer ff = placeholders.readOnlyFloats();
        GpuBuffer posb = ints(pos);
        GpuBuffer ref = floats(x);
        run(NormKernels.ROW_NORM, (long) TOKENS * HEADS * local,
                NormKernels.rowNormArgs(NormKind.RMS, EPS, TOKENS * HEADS, HEAD_DIM, HEAD_DIM, ref, 0, ref, 0, w, null,
                        1, placeholders.readOnlyFloats()));
        run(RopeKernels.ROPE_EXT,
                ActKernels.global1d(RopeKernels.ropeExtWorkItems(rope.type(), TOKENS, HEADS, rope.nDims()), local)[0],
                RopeKernels.ropeExtArgs(rope, TOKENS, HEADS, HEAD_DIM, rowStride, 0, TOKENS, 0, false, posb, ff,
                        ref));
        float[] want = read(ref, total);

        GpuBuffer src = floats(x);
        GpuBuffer out = floats(new float[total]);
        GpuBuffer inPlace = floats(x);
        for (GpuBuffer dst : new GpuBuffer[]{out, inPlace}) {
            GpuBuffer in = dst == inPlace ? inPlace : src;
            run(FusedKernels.RMS_NORM_ROPE, FusedKernels.rmsNormRopeGlobal(TOKENS, HEADS, local),
                    FusedKernels.rmsNormRopeArgs(rope, TOKENS, HEADS, HEAD_DIM, rowStride, HEAD_DIM, 0, TOKENS, 0, 0,
                            NormKernels.FLAG_HAS_W, 1, EPS, false, posb, ff, w, placeholders.readOnlyFloats(), in,
                            dst));
            float[] got = read(dst, total);
            String what = dst == inPlace ? "in place" : "out of place";
            for (int i = 0; i < total; i++) {
                assertEquals(what + " [" + i + "]", want[i], got[i], TOL * (1f + Math.abs(want[i])));
            }
        }
    }
}
