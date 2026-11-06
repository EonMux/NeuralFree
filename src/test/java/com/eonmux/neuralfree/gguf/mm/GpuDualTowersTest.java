package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.mm.ClipDualEncoderTest.image;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * The towers of a CLIP dual encoder on an OpenCL device ({@link GpuDualTowers}) against the same towers on the CPU:
 * the causal CLIP text tower and its class-token vision tower, the bidirectional SigLIP text tower over the padded
 * input and its attention-pooled vision tower; the host keeps the token embedding, heads and projections, and the
 * device choice follows {@link MmprojOptions} (offload off keeps the CPU).
 */
public class GpuDualTowersTest {

    private static final float TOL = 1e-3f;

    private static MmprojOptions openCl() {
        Assume.assumeTrue("no OpenCL device", GpuTestUtil.isOpenCLAvailable());
        return MmprojOptions.defaults().withBackend(EngineConfig.BackendMode.OPENCL);
    }

    private static void assertClose(String what, float[] want, float[] got) {
        assertEquals(what + ": the width", want.length, got.length);
        float scale = 0;
        for (float v : want) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < want.length; i++) {
            assertEquals(what + " value " + i, want[i], got[i], TOL * Math.max(1, scale));
        }
    }

    private static void assertParity(Path file, int[] ids, RasterImage img, MmprojOptions device)
            throws IOException {
        ModelSourceOptions o = ModelSourceOptions.defaults();
        try (ClipDualEncoder cpu = ClipDualEncoder.open(file.toString(), o);
             ClipDualEncoder gpu = ClipDualEncoder.open(file.toString(), o, device)) {
            assertEquals("cpu", cpu.device());
            Assume.assumeFalse("no OpenCL device runs the LLM kernels", "cpu".equals(gpu.device()));
            assertClose("the text embedding", cpu.embedText(ids), gpu.embedText(ids));
            assertClose("the image embedding", cpu.embedImage(img), gpu.embedImage(img));
            assertTrue(gpu.toString(), gpu.toString().contains(gpu.device()));
        }
    }

    @Test
    public void clipTowersOnTheDeviceMatchTheCpuTowers() throws IOException {
        MmprojOptions device = openCl();
        assertParity(ClipDualFixtures.twoTower(), ClipDualFixtures.catDogIds(), image(8, 8, 21), device);
    }

    @Test
    public void siglipTowersOnTheDeviceMatchTheCpuTowers() throws IOException {
        MmprojOptions device = openCl();
        Path file = ClipDualFixtures.writer(ClipDualFixtures.siglipTensors(71), ClipDualFixtures.siglipMeta())
                .writeTempFile();
        assertParity(file, new int[]{6, 11, 15}, image(12, 8, 22), device);
    }

    @Test
    public void theCausalMaskReachesTheDeviceTextTower() throws IOException {
        MmprojOptions device = openCl();
        Map<String, Object> meta = ClipDualFixtures.meta(true, true);
        meta.put(ClipDualEncoder.POOLING_KEY, 2);
        Path file = ClipDualFixtures.writer(ClipDualFixtures.tensors(72, true, true), meta).writeTempFile();
        try (ClipDualEncoder gpu = ClipDualEncoder.open(file.toString(), ModelSourceOptions.defaults(), device)) {
            Assume.assumeFalse("no OpenCL device runs the LLM kernels", "cpu".equals(gpu.device()));
            float[] first = gpu.embedText(new int[]{6, 11, 13});
            assertClose("the first row attends to itself only", first, gpu.embedText(new int[]{6, 9, 2, 14}));
            assertNotEquals("the first token still reaches the first row", first[0],
                    gpu.embedText(new int[]{7, 11, 13})[0], 1e-6f);
        }
    }

    @Test
    public void offloadOffKeepsTheTowersOnTheCpu() throws IOException {
        try (ClipDualEncoder enc = ClipDualEncoder.open(ClipDualFixtures.twoTower().toString(),
                ModelSourceOptions.defaults(), MmprojOptions.cpu())) {
            assertEquals("cpu", enc.device());
        }
        try (ClipDualEncoder enc = ClipDualEncoder.open(ClipDualFixtures.twoTower().toString(),
                ModelSourceOptions.defaults(), MmprojOptions.onDevice(MmprojOptions.NO_DEVICE))) {
            assertEquals("cpu", enc.device());
        }
    }

    @Test
    public void theHostKeepsTheEmbeddingHeadsAndProjections() {
        for (String name : new String[]{ClipTextTower.TOKEN_EMBD, ClipTextTower.PROJECTION, ClipImageTower.PROJECTION,
                "t.head.weight", "t.head.bias", "v.head.probe", "v.head.attention.in_weight", "v.head.ffn_up.weight",
                ClipDualEncoder.LOGIT_SCALE, ClipDualEncoder.LOGIT_BIAS}) {
            assertTrue(name, GpuDualTowers.hostOnly(name));
        }
        for (String name : new String[]{"t.blk.0.attn_q.weight", "v.blk.1.ffn_down.weight", "v.patch_embd.weight",
                "v.position_embd.weight", "t.post_ln.weight", "v.pre_ln.bias", "v.class_embd"}) {
            assertFalse(name, GpuDualTowers.hostOnly(name));
        }
    }
}
