package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Opening an mmproj GGUF beside a text model: capability probing, memory usage, the CPU encoder and the device
 * choice over a real file written by {@link GgufTestWriter}, plus the refusals of the loader (llama.cpp
 * mtmd_init_from_file / mtmd_get_cap_from_file counterparts).
 */
public class MmprojModelTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int E = 8;
    private static final int PS = 4;
    private static final int FF = 12;
    private static final int PROJ = 6;
    private static final int LAYERS = 2;
    private static final int IMAGE = 8;

    /** A runnable llava-style mmproj file: clip metadata and the full CLIP ViT + MLP tensor set. */
    private GgufTestWriter llavaMmproj(String projector) {
        GgufTestWriter w = new GgufTestWriter()
                .architecture("clip")
                .bool("clip.has_vision_encoder", true)
                .str("clip.projector_type", projector)
                .u32("clip.vision.image_size", IMAGE)
                .u32("clip.vision.patch_size", PS)
                .u32("clip.vision.embedding_length", E)
                .u32("clip.vision.feed_forward_length", FF)
                .u32("clip.vision.block_count", LAYERS)
                .u32("clip.vision.projection_dim", PROJ)
                .u32("clip.vision.attention.head_count", 2)
                .u32("clip.vision.attention.head_count_kv", 2)
                .f32("clip.vision.attention.layer_norm_epsilon", 1e-6f)
                .f32Array("clip.vision.image_mean", 0.48f, 0.45f, 0.4f)
                .f32Array("clip.vision.image_std", 0.26f, 0.26f, 0.27f)
                .bool("clip.use_gelu", true);

        Random rnd = new Random(7);
        tensor(w, rnd, "v.patch_embd.weight", PS, PS, 3, E);
        tensor(w, rnd, "v.patch_embd.bias", E);
        tensor(w, rnd, "v.class_embd", E, 1);
        tensor(w, rnd, "v.position_embd.weight", E, 5);
        tensor(w, rnd, "v.pre_ln.weight", E);
        tensor(w, rnd, "v.pre_ln.bias", E);
        tensor(w, rnd, "v.post_ln.weight", E);
        tensor(w, rnd, "v.post_ln.bias", E);
        for (int il = 0; il < LAYERS; il++) {
            String p = "v.blk." + il + ".";
            tensor(w, rnd, p + "ln1.weight", E);
            tensor(w, rnd, p + "ln1.bias", E);
            tensor(w, rnd, p + "ln2.weight", E);
            tensor(w, rnd, p + "ln2.bias", E);
            tensor(w, rnd, p + "attn_qkv.weight", E, 3 * E);
            tensor(w, rnd, p + "attn_qkv.bias", 3 * E);
            tensor(w, rnd, p + "attn_out.weight", E, E);
            tensor(w, rnd, p + "attn_out.bias", E);
            tensor(w, rnd, p + "ffn_up.weight", E, FF);
            tensor(w, rnd, p + "ffn_up.bias", FF);
            tensor(w, rnd, p + "ffn_down.weight", FF, E);
            tensor(w, rnd, p + "ffn_down.bias", E);
        }
        tensor(w, rnd, "mm.0.weight", E, PROJ);
        tensor(w, rnd, "mm.0.bias", PROJ);
        return w;
    }

    private static void tensor(GgufTestWriter w, Random rnd, String name, long... ne) {
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = rnd.nextFloat() - 0.5f;
        }
        w.tensorFromFloats(name, GGMLType.F32, ne, v);
    }

    private Path write(GgufTestWriter w) throws IOException {
        Path file = tmp.newFolder("mmproj").toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        return file;
    }

    @Test
    public void openProbesCapsMemoryAndEncodes() throws IOException {
        Path file = write(llavaMmproj("mlp"));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            MmprojModel.Caps caps = mm.caps();
            assertTrue(caps.inpVision());
            assertFalse(caps.inpAudio());
            assertFalse(caps.genAudio());
            assertTrue(mm.supportsVision());
            assertFalse(mm.supportsAudio());
            assertEquals("clip", mm.metadata().archName());

            ClipHparams hp = mm.visionHparams();
            assertEquals(ProjectorType.MLP, hp.projectorType());
            assertEquals(E, hp.nEmbd());
            assertEquals(IMAGE, hp.imageSize());

            long bytes = mm.memoryUsageBytes();
            long weights = 3L * PS * PS * E + E + E + 5L * E + 4L * E
                    + LAYERS * (4L * E + 3L * E * E + 3L * E + 1L * E * E + E + 1L * E * FF + FF + 1L * FF * E + E)
                    + 1L * E * PROJ + PROJ;
            assertEquals(4 * weights, bytes);
            assertTrue(mm.tensorNames().contains("v.blk.1.attn_qkv.weight"));

            try (CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
                assertEquals(ProjectorType.MLP, enc.model().hparams().projectorType());
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
                float[] pixels = MmprojFixture.pixels(IMAGE, IMAGE, 42);
                float[][] rows = enc.encode(ClipImage.of(IMAGE, IMAGE, pixels));
                // the llava graph reads n_patches rows (class token last, rows 1..n_patches)
                assertEquals(4, rows.length);
                assertEquals(PROJ, rows[0].length);
                boolean finite = true;
                for (float[] row : rows) {
                    for (float v : row) {
                        finite &= !Float.isNaN(v) && !Float.isInfinite(v);
                    }
                }
                assertTrue(finite);
            }
            // the model stays usable after one encoder was closed
            try (VisionEncoder enc = mm.openVisionEncoder(MmprojOptions.cpu())) {
                assertEquals(VisionEncoder.CPU_DEVICE, enc.device());
                assertEquals(4, enc.encode(ClipImage.of(IMAGE, IMAGE,
                        MmprojFixture.pixels(IMAGE, IMAGE, 42))).length);
            }
        }
    }

    @Test
    public void offloadOffAndDeviceNoneKeepTheEncoderOnTheCpu() throws IOException {
        Path file = write(llavaMmproj("mlp"));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            for (MmprojOptions o : new MmprojOptions[]{MmprojOptions.cpu(), MmprojOptions.onDevice("none"),
                    MmprojOptions.defaults().withOffload(false),
                    MmprojOptions.defaults().withBackend(EngineConfig.BackendMode.CPU)}) {
                try (VisionEncoder enc = mm.openVisionEncoder(o)) {
                    assertTrue(o.toString(), enc instanceof CpuVisionEncoder);
                }
            }
        }
    }

    @Test
    public void anUnknownMmprojDeviceIsRefusedByName() throws IOException {
        Path file = write(llavaMmproj("mlp"));
        try (MmprojModel mm = MmprojModel.open(file.toString())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> mm.openVisionEncoder(MmprojOptions.onDevice("no-such-device-7f3a")));
            assertTrue(e.getMessage(), e.getMessage().contains(MmprojOptions.PROP_DEVICE));
            assertTrue(e.getMessage(), e.getMessage().contains("no-such-device-7f3a"));
        }
    }

    @Test
    public void tensorTablesOfAnotherFamilyFailAtOpen() throws IOException {
        // an mlp_norm registry row needs mm.1/mm.3/mm.4, which this mlp-only file does not carry
        Path file = write(llavaMmproj("mlp_norm"));
        assertThrows(com.eonmux.neuralfree.gguf.spec.MissingTensorException.class,
                () -> MmprojModel.open(file.toString()));
    }

    @Test
    public void nonClipFilesAndUnknownTypesAreRefusedByname() throws IOException {
        Path file = tmp.newFolder("wrong").toPath().resolve("model.gguf");
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.u32("llama.embedding_length", 64);
        w.writeTo(file);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojModel.open(file.toString()));
        assertTrue(e.getMessage().contains("llama"));

        GgufTestWriter unknown = llavaMmproj("warp_drive");
        Path unknownFile = write(unknown);
        assertThrows(com.eonmux.neuralfree.gguf.hparams.GgufFormatException.class,
                () -> MmprojModel.open(unknownFile.toString()));
    }

    @Test
    public void missingTensorsFailAtOpenWithTheName() throws IOException {
        GgufTestWriter w = llavaMmproj("mlp");
        w.removeTensor("v.position_embd.weight");
        Path file = write(w);
        assertThrows(com.eonmux.neuralfree.gguf.spec.MissingTensorException.class,
                () -> MmprojModel.open(file.toString()));
    }
}
