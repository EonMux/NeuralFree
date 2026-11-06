package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The GPU vision encoder against the CPU encoder and the double-precision reference of the clip graphs, one small
 * synthetic mmproj file per runnable family (llava MLP with separate and fused QKV, MLP-norm, SigLIP + Gemma3 pool
 * with and without layer scales and Q/K norms, Qwen2-VL over a rectangle, Qwen2.5-VL with windows that regroup the
 * tokens and with its RMS pre-norm) plus F16 weights, on an OpenCL device shared through
 * {@link MmprojModel#openVisionEncoder(ComputeBackend)}; and the CPU fallback of a model the flash kernels cannot
 * tile or whose mixture-of-experts blocks route beyond the GPU router, decided without a GPU. The other runnable
 * families are checked against the CPU encoder by {@link GpuVisionFamiliesTest}.
 */
public class GpuVisionEncoderTest {

    private static final float GPU_TOL = 1e-3f;
    private static final float REF_TOL = 2e-3f;

    @ClassRule
    public static final TemporaryFolder TMP = new TemporaryFolder();

    private static HWAcceleration acceleration;
    private static ComputeBackend backend;

    @BeforeClass
    public static void openDevice() {
        if (!GpuTestUtil.isOpenCLAvailable()) {
            return;
        }
        acceleration = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        try {
            backend = acceleration.createBackend(Config.ACCELERATION_MODE.OPENCL, LlmKernels.requirement());
        } catch (RuntimeException e) {
            backend = null;
        }
    }

    @AfterClass
    public static void closeDevice() {
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static void assumeGpu() {
        Assume.assumeTrue("no OpenCL device running the LLM kernels", backend != null);
    }

    // ================================================================== fixtures

    private static Path write(MmprojFixture f, Map<String, Object> meta, Set<String> f16) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        for (Map.Entry<String, Object> e : meta.entrySet()) {
            Object v = e.getValue();
            if (e.getKey().equals("general.architecture")) {
                continue;
            }
            if (v instanceof Boolean) {
                w.bool(e.getKey(), (Boolean) v);
            } else if (v instanceof Integer) {
                w.u32(e.getKey(), (Integer) v);
            } else if (v instanceof Float) {
                w.f32(e.getKey(), (Float) v);
            } else if (v instanceof float[]) {
                w.f32Array(e.getKey(), (float[]) v);
            } else if (v instanceof int[]) {
                w.i32Array(e.getKey(), (int[]) v);
            } else {
                w.str(e.getKey(), (String) v);
            }
        }
        for (Map.Entry<String, long[]> e : f.shapes.entrySet()) {
            GGMLType type = f16.contains(e.getKey()) ? GGMLType.F16 : GGMLType.F32;
            w.tensorFromFloats(e.getKey(), type, e.getValue(), f.get(e.getKey()));
        }
        Path file = TMP.newFolder().toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    private static Map<String, Object> meta(String projector, int image, int patch, boolean gelu, boolean silu,
                                            Integer waPattern, Integer window, Integer merge, int layers) {
        return CpuVisionEncoderTest.clipMeta(projector, image, patch, 8, 12, layers, 6, 2, 2, gelu, silu,
                waPattern, window, merge);
    }

    private static MmprojFixture mlpNormFixture(long seed) {
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(seed, false, false);
        f.norm("mm.1.weight", "mm.1.bias", 6);
        f.weight("mm.3.weight", 6, 6);
        f.weight("mm.3.bias", 6);
        f.norm("mm.4.weight", "mm.4.bias", 6);
        return f;
    }

    /**
     * SigLIP + Gemma3 pool with the {@code build_vit} extras on every block: layer scales and Q/K norms, a per-head
     * Q norm and a per-head (fused) or whole-row (separate) K norm.
     */
    private static MmprojFixture siglipExtrasFixture(long seed, boolean fused) {
        int e = 8;
        int ff = 12;
        int dh = 4;
        MmprojFixture f = new MmprojFixture(seed);
        f.weight("v.patch_embd.weight", 4, 4, 3, e);
        f.weight("v.position_embd.weight", e, 16);
        f.norm("v.post_ln.weight", "v.post_ln.bias", e);
        for (int il = 0; il < 2; il++) {
            String p = "v.blk." + il + ".";
            f.norm(p + "ln1.weight", p + "ln1.bias", e);
            f.norm(p + "ln2.weight", p + "ln2.bias", e);
            if (fused) {
                f.weight(p + "attn_qkv.weight", e, 3 * e);
                f.weight(p + "attn_qkv.bias", 3 * e);
            } else {
                for (String n : new String[]{"attn_q", "attn_k", "attn_v"}) {
                    f.weight(p + n + ".weight", e, e);
                    f.weight(p + n + ".bias", e);
                }
            }
            f.weight(p + "attn_out.weight", e, e);
            f.weight(p + "ffn_up.weight", e, ff);
            f.weight(p + "ffn_gate.weight", e, ff);
            f.weight(p + "ffn_down.weight", ff, e);
            f.put(p + "ls1.weight", 0.2f, 1.5f, e);
            f.put(p + "ls2.weight", 0.2f, 1.5f, e);
            f.put(p + "attn_q_norm.weight", 0.8f, 1.2f, dh);
            f.put(p + "attn_k_norm.weight", 0.8f, 1.2f, fused ? dh : e);
        }
        f.weight("mm.input_projection.weight", e, 6);
        f.weight("mm.soft_emb_norm.weight", e);
        return f;
    }

    /**
     * Encodes on the CPU and the shared GPU device, checks the GPU rows against the CPU rows and the reference
     * (when given), and returns the GPU rows.
     */
    private static float[][] parity(Path file, ClipImage image, float[][] reference) throws IOException {
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder cpu = mm.openCpuVisionEncoder();
             VisionEncoder gpu = mm.openVisionEncoder(backend)) {
            assertTrue(gpu.toString(), gpu instanceof GpuVisionEncoder);
            assertNotEquals(VisionEncoder.CPU_DEVICE, gpu.device());
            float[][] want = cpu.encode(image);
            float[][] got = gpu.encode(image);
            assertRows(want, got, GPU_TOL);
            if (reference != null) {
                assertRows(reference, got, REF_TOL);
            }
            assertRows(got, gpu.encode(image), 0f);
            return got;
        }
    }

    private static void assertRows(float[][] want, float[][] got, float tol) {
        assertEquals(want.length, got.length);
        for (int t = 0; t < want.length; t++) {
            assertArrayEquals("row " + t, want[t], got[t], tol);
        }
    }

    private static ClipHparams hp(Map<String, Object> meta) {
        return CpuVisionEncoderTest.hparams(meta);
    }

    // ================================================================== parity per family

    @Test
    public void llavaMlpWithSeparateQkvMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(51, false, false);
        Map<String, Object> m = meta("mlp", 8, 4, true, false, null, null, null, 2);
        float[] px = MmprojFixture.pixels(8, 8, 510);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(8, 8, px),
                MmprojFixture.Reference.clip(f, hp(m), px, 8, 8));
        // the llava graph reads n_patches rows, from row 1 on past the class token appended last
        assertEquals(4, got.length);
        assertEquals(6, got[0].length);
    }

    @Test
    public void llavaStackedFeatureLayersMatchTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = VisionFamilyFixturesB.llavaStacked(54);
        Map<String, Object> m = VisionFamilyFixturesB.llavaStackedMeta();
        float[] px = MmprojFixture.pixels(16, 16, 540);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(16, 16, px),
                MmprojFixture.Reference.clip(f, hp(m), px, 16, 16));
        assertEquals(16, got.length);
        assertEquals(VisionFamilyFixturesB.OUT, got[0].length);
    }

    @Test
    public void llavaMlpWithFusedQkvAndTwoLayersMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(52, true, true);
        Map<String, Object> m = meta("mlp", 8, 4, true, false, null, null, null, 2);
        float[] px = MmprojFixture.pixels(8, 8, 520);
        parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(8, 8, px),
                MmprojFixture.Reference.clip(f, hp(m), px, 8, 8));
    }

    @Test
    public void llavaMlpNormMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = mlpNormFixture(53);
        Map<String, Object> m = meta("mlp_norm", 8, 4, true, false, null, null, null, 2);
        float[] px = MmprojFixture.pixels(8, 8, 530);
        parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(8, 8, px),
                MmprojFixture.Reference.clip(f, hp(m), px, 8, 8));
    }

    @Test
    public void siglipGemma3PoolWithGatedGeluQuickMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.siglipFixture(54);
        Map<String, Object> m = meta("gemma3", 16, 4, false, false, null, null, 2, 2);
        float[] px = MmprojFixture.pixels(16, 16, 540);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(16, 16, px),
                MmprojFixture.Reference.clip(f, hp(m), px, 16, 16));
        assertEquals(4, got.length);
    }

    @Test
    public void siglipLayerScaleAndFusedQkNormsMatchTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = siglipExtrasFixture(61, true);
        Map<String, Object> m = meta("gemma3", 16, 4, false, false, null, null, 2, 2);
        ClipVisionModel model = ClipVisionModel.from(hp(m), f.catalog());
        assertNotNull(model.layers().get(0).ls1Weight());
        assertNotNull(model.layers().get(0).qNormWeight());
        float[][] got = parity(write(f, m, Collections.<String>emptySet()),
                ClipImage.of(16, 16, MmprojFixture.pixels(16, 16, 610)), null);
        assertEquals(4, got.length);
    }

    @Test
    public void siglipPerHeadAndWholeRowQkNormsMatchTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = siglipExtrasFixture(62, false);
        Map<String, Object> m = meta("gemma3", 16, 4, true, false, null, null, 2, 2);
        parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(16, 16, MmprojFixture.pixels(16, 16, 620)),
                null);
    }

    @Test
    public void qwen2vlRectangleMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.qwenFixture(55, false);
        Map<String, Object> m = meta("qwen2vl_merger", 8, 2, false, true, null, null, null, 3);
        float[] px = MmprojFixture.pixels(8, 12, 550);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(8, 12, px),
                MmprojFixture.Reference.qwen(f, hp(m), px, 8, 12));
        assertEquals(6, got.length);
    }

    @Test
    public void qwen2vlSecondFrameKernelAndWideMergerMatchTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.qwenTemporalFixture(58, false);
        Map<String, Object> m = meta("qwen2vl_merger", 8, 2, false, true, null, null, null, 3);
        float[] px = MmprojFixture.pixels(8, 12, 580);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(8, 12, px),
                MmprojFixture.Reference.qwen(f, hp(m), px, 8, 12));
        assertEquals(6, got.length);
    }

    @Test
    public void qwen2vlTemporalFramePairMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.qwenTemporalFixture(58, false);
        Map<String, Object> m = meta("qwen2vl_merger", 8, 2, false, true, null, null, null, 3);
        // two different frames of one video: the first runs patch kernel 0, the second kernel 1
        float[] px = MmprojFixture.pixels(8, 12, 580);
        float[] px2 = MmprojFixture.pixels(8, 12, 917);
        ClipImage pair = ClipImage.of(8, 12, px).withSecondFrame(px2);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), pair, null);
        assertEquals(6, got.length);
    }

    @Test
    public void qwen25vlRegroupingWindowsMatchTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.qwenFixture(56, true);
        Map<String, Object> m = meta("qwen2.5vl_merger", 8, 2, false, true, 2, 8, null, 3);
        float[] px = MmprojFixture.pixels(12, 12, 560);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(12, 12, px),
                MmprojFixture.Reference.qwen(f, hp(m), px, 12, 12));
        assertEquals(9, got.length);
    }

    @Test
    public void qwen25vlRmsPreNormMatchesTheCpuEncoder() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.qwenFixture(63, true);
        // qwen2vl.cpp normalizes v.pre_ln with the encoder's norm type, RMS for 2.5
        f.weight("v.pre_ln.weight", 8);
        Map<String, Object> m = meta("qwen2.5vl_merger", 8, 2, false, true, 2, 4, null, 3);
        float[] px = MmprojFixture.pixels(8, 8, 630);
        float[][] got = parity(write(f, m, Collections.<String>emptySet()), ClipImage.of(8, 8, px),
                MmprojFixture.Reference.qwen(f, hp(m), px, 8, 8));
        assertEquals(4, got.length);
    }

    @Test
    public void f16WeightsMatchTheCpuEncoderOverTheSameFile() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(57, true, true);
        Set<String> f16 = new HashSet<>();
        for (String name : f.shapes.keySet()) {
            if (f.shapes.get(name).length >= 2) {
                f16.add(name);
            }
        }
        Map<String, Object> m = meta("mlp", 8, 4, true, false, null, null, null, 2);
        parity(write(f, m, f16), ClipImage.of(8, 8, MmprojFixture.pixels(8, 8, 570)), null);
    }

    // ================================================================== device choice and fallback

    @Test
    public void offloadThroughTheOptionsOwnsItsDevice() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(58, true, false);
        Path file = write(f, meta("mlp", 8, 4, true, false, null, null, null, 2), Collections.<String>emptySet());
        ClipImage img = ClipImage.of(8, 8, MmprojFixture.pixels(8, 8, 580));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            float[][] want = mm.openCpuVisionEncoder().encode(img);
            VisionEncoder gpu = mm.openVisionEncoder(
                    MmprojOptions.defaults().withBackend(EngineConfig.BackendMode.OPENCL));
            assertTrue(gpu instanceof GpuVisionEncoder);
            assertRows(want, gpu.encode(img), GPU_TOL);
            gpu.close();
            gpu.close();
            assertThrows(IllegalStateException.class, () -> gpu.encode(img));
        }
    }

    @Test
    public void wrongGridIsRefusedByName() throws IOException {
        assumeGpu();
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(59, false, false);
        Path file = write(f, meta("mlp", 8, 4, true, false, null, null, null, 2), Collections.<String>emptySet());
        try (MmprojModel mm = MmprojModel.open(file.toString());
             VisionEncoder gpu = mm.openVisionEncoder(backend)) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> gpu.encode(ClipImage.of(12, 12, new float[12 * 12 * 3])));
            assertTrue(e.getMessage().contains("patch grid"));
        }
    }

    @Test
    public void aMixtureOfExpertsBeyondTheRouterLimitFallsBackToTheCpu() throws IOException {
        int experts = MoeKernels.MAX_K + 8;
        MmprojFixture f = VisionFamiliesCFixtures.dots3note(61, experts);
        Map<String, Object> m = VisionFamiliesCFixtures.dots3noteMeta();
        m.put("clip.vision.expert_used_count", MoeKernels.MAX_K + 1);
        ClipVisionModel wide = ClipVisionModel.from(VisionFamilyFixtures.hparams(m), f.catalog());
        String reason = GpuVisionEncoder.unsupportedReason(wide, 4096, 8);
        assertNotNull(reason);
        assertTrue(reason, reason.contains("dots3note_v") && reason.contains("routes " + (MoeKernels.MAX_K + 1)
                + " of " + experts + " experts"));

        if (backend != null) {
            Path file = TMP.newFolder().toPath().resolve("mmproj.gguf");
            VisionFamilyFixtures.gguf(f, m).writeTo(file);
            try (MmprojModel mm = MmprojModel.open(file.toString());
                 VisionEncoder enc = mm.openVisionEncoder(backend)) {
                assertTrue(enc instanceof CpuVisionEncoder);
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
            }
        }
    }

    @Test
    public void aHeadWidthTheFlashKernelsCannotTileFallsBackToTheCpu() throws IOException {
        // 4 heads of n_embd 8: head width 2 is not a multiple of 4
        MmprojFixture f = CpuVisionEncoderTest.llavaFixture(60, false, false);
        Map<String, Object> m = CpuVisionEncoderTest.clipMeta("mlp", 8, 4, 8, 12, 2, 6, 4, 4, true, false,
                null, null, null);
        ClipVisionModel narrow = ClipVisionModel.from(hp(m), f.catalog());
        String reason = GpuVisionEncoder.unsupportedReason(narrow, 4096, 8);
        assertTrue(reason, reason.contains("mlp") && reason.contains("multiple of 4"));

        ClipVisionModel ok = ClipVisionModel.from(hp(meta("mlp", 8, 4, true, false, null, null, null, 2)),
                f.catalog());
        assertNull(GpuVisionEncoder.unsupportedReason(ok, 4096, 8));
        String tiny = GpuVisionEncoder.unsupportedReason(ok, 64, 8);
        assertTrue(tiny, tiny.contains("no tiled flash-attention plan"));

        if (backend != null) {
            try (MmprojModel mm = MmprojModel.open(write(f, m, Collections.<String>emptySet()).toString());
                 VisionEncoder enc = mm.openVisionEncoder(backend)) {
                assertTrue(enc instanceof CpuVisionEncoder);
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
            }
        }
    }
}
