package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link Yasa2Graph}: the ConvNeXt-V2 encoder and its MLP projector over the {@link Yasa2Fixture} files give the
 * rows of the stored references ({@code yasa2-reference.txt}): an exact evaluation of llama.cpp's graph within
 * {@code 1e-4} of the output scale, and llama.cpp's own CPU rows within the drift of its F16 convolution inputs.
 * Files with transformer blocks, without stages, with a grid that does not pool to 8x8 or a position table of
 * another width are refused by name.
 */
public class Yasa2GraphTest {

    private static final String REFERENCE = "com/eonmux/neuralfree/gguf/mm/yasa2-reference.txt";
    /** Largest difference to the exact rows, as a fraction of their largest magnitude. */
    private static final double EXACT_TOLERANCE = 1e-4;
    /**
     * Largest difference to llama.cpp's rows, as a fraction of their largest magnitude: its {@code ggml_conv_2d}
     * and {@code ggml_conv_2d_dw} round every convolution input and kernel to F16 (about 1e-3 of the scale here).
     */
    private static final double LLAMA_TOLERANCE = 2e-3;

    @Test
    public void everyOptionalTensorMatchesTheReferenceRows() throws IOException {
        assertMatchesReference(Yasa2Fixture.full());
    }

    @Test
    public void absentOptionalsAndAPooledGridMatchTheReferenceRows() throws IOException {
        assertMatchesReference(Yasa2Fixture.sparse());
    }

    private static void assertMatchesReference(Yasa2Fixture fx) throws IOException {
        Path file = fx.gguf().writeTempFile();
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
            assertEquals(ProjectorType.YASA2, mm.visionHparams().projectorType());
            float[][] got = enc.encode(ClipImage.of(fx.imageSize, fx.imageSize, fx.pixels()));
            assertEquals(Yasa2Graph.POOLED_SIDE * Yasa2Graph.POOLED_SIDE, got.length);
            assertEquals(Yasa2Fixture.OUT, got[0].length);
            ReferenceRows.assertClose(fx.name + " exact", ReferenceRows.read(REFERENCE, fx.name, "exact"), got, 1,
                    EXACT_TOLERANCE);
            ReferenceRows.assertClose(fx.name + " llama.cpp", ReferenceRows.read(REFERENCE, fx.name, "llama.cpp"),
                    got, 1, LLAMA_TOLERANCE);
        }
    }

    @Test
    public void thePoolingKernelIsTheReferences() {
        assertEquals(1, Yasa2Graph.poolKernel(8));
        assertEquals(8, Yasa2Graph.pooledSide(8));
        assertEquals(2, Yasa2Graph.poolKernel(16));
        assertEquals(8, Yasa2Graph.pooledSide(16));
        assertEquals(2, Yasa2Graph.poolKernel(17));
        assertEquals(8, Yasa2Graph.pooledSide(17));
        // a 20-cell side pools with kernel 2 to 10 cells, a 6-cell side keeps its 6
        assertEquals(10, Yasa2Graph.pooledSide(20));
        assertEquals(6, Yasa2Graph.pooledSide(6));
    }

    @Test
    public void transformerBlocksAreRefusedByKey() {
        Yasa2Fixture fx = Yasa2Fixture.full();
        fx.meta.put("clip.vision.block_count", 2);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> openEncoder(fx.gguf()));
        assertTrue(e.getMessage(), e.getMessage().contains("clip.vision.block_count' must be 0"));
    }

    @Test
    public void aFileWithoutStagesIsRefusedNamingTheFirstBlock() {
        Yasa2Fixture fx = Yasa2Fixture.full();
        GgufTestWriter w = fx.gguf();
        for (String name : w.tensorNames()) {
            if (name.startsWith("v.stage.")) {
                w.removeTensor(name);
            }
        }
        MissingTensorException e = assertThrows(MissingTensorException.class, () -> openEncoder(w));
        assertTrue(e.getMessage(), e.getMessage().contains("v.stage.0.blk.0.dw.weight"));
    }

    @Test
    public void anImageSizeWhoseGridDoesNotPoolTo8x8IsRefusedAtBind() {
        Yasa2Fixture fx = Yasa2Fixture.full();
        // 48 / 4 = 12 cells, one downsample to 6: the grid keeps 6x6 cells
        fx.meta.put("clip.vision.image_size", 48);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> openEncoder(fx.gguf()));
        assertTrue(e.getMessage(), e.getMessage().contains("clip.vision.image_size 48"));
        assertTrue(e.getMessage(), e.getMessage().contains("6x6"));
    }

    @Test
    public void anImageWhoseGridDoesNotPoolTo8x8IsRefusedAtEncode() throws IOException {
        Yasa2Fixture fx = Yasa2Fixture.full();
        Path file = fx.gguf().writeTempFile();
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> enc.encode(ClipImage.of(80, 80, MmprojFixture.pixels(80, 80, 5))));
            assertTrue(e.getMessage(), e.getMessage().contains("image 80x80"));
            assertTrue(e.getMessage(), e.getMessage().contains("10x10"));
        }
    }

    @Test
    public void aPositionTableOfAnotherWidthIsRefused() {
        Yasa2Fixture fx = Yasa2Fixture.full();
        GgufTestWriter w = fx.gguf();
        w.removeTensor("v.vision_pos_embed");
        w.tensorFromFloats("v.vision_pos_embed", GGMLType.F32, new long[]{8, 64}, new float[8 * 64]);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> openEncoder(w));
        assertTrue(e.getMessage(), e.getMessage().contains("v.vision_pos_embed"));
        assertTrue(e.getMessage(), e.getMessage().contains("12 channels"));
    }

    @Test
    public void theEncoderRunsOnTheGpu() {
        Yasa2Fixture fx = Yasa2Fixture.full();
        ClipVisionModel model = ClipVisionModel.from(fx.hparams(), fx.tensors.catalog());
        assertNull(GpuVisionEncoder.unsupportedReason(model, 1 << 14, 64));
    }

    private static void openEncoder(GgufTestWriter w) throws IOException {
        Path file = w.writeTempFile();
        try (MmprojModel mm = MmprojModel.open(file.toString());
             CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
            assertNotNull(enc);
        }
    }
}
