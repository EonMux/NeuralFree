package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.engine.DraftHeadOptions;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The derived draft head's selection and encoding ({@link DraftHeadQuantizer}) and the derived source's tensor
 * table and payloads ({@link DerivedDraftSource}): the kept ids ascend with the mandatory specials in, the
 * ranking file ranks, the encodings store rows the decoders read back, and the derived source presents the
 * derived head, the synthesized {@code d2t} and the lower-precision matrices over an unchanged draft.
 */
public class DraftHeadQuantizerTest {

    private Path draft;
    private Path borrowed;

    @Before
    public void writeDrafts() throws IOException {
        draft = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(4, 1, 2)).writeTempFile();
        borrowed = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(4, 1, 2).shareTargetHead()).writeTempFile();
    }

    @After
    public void deleteDrafts() throws IOException {
        Files.deleteIfExists(draft);
        Files.deleteIfExists(borrowed);
    }

    // ------------------------------------------------------------------ subset

    @Test
    public void theKeptIdsAscendWithTheMandatoryOnesIn() {
        // mask id 319 is mandatory, 63 more ids fill ascending
        int[] kept = DraftHeadQuantizer.subset(320, 64, null, new int[] {319}, "v");
        assertEquals(64, kept.length);
        for (int i = 0; i < 63; i++) {
            assertEquals(i, kept[i]);
        }
        assertEquals(319, kept[63]);
    }

    @Test
    public void theRankingRanksBeforeTheAscendingFill() {
        int[] ranked = {10, 5, 300, 7};
        int[] kept = DraftHeadQuantizer.subset(320, 6, ranked, new int[] {0}, "v");
        assertArrayEquals(new int[] {0, 1, 5, 7, 10, 300}, kept);
    }

    @Test
    public void rankedIdsAlreadyKeptAreNotRepeated() {
        int[] kept = DraftHeadQuantizer.subset(320, 4, new int[] {9, 9, 3, 1}, new int[] {1}, "v");
        assertArrayEquals(new int[] {0, 1, 3, 9}, kept);
    }

    @Test(expected = IllegalArgumentException.class)
    public void aSizeThatReducesNothingIsRefused() {
        DraftHeadQuantizer.subset(320, 320, null, new int[] {0}, "neuralfree.draft.head.vocab");
    }

    @Test(expected = IllegalArgumentException.class)
    public void aNonPositiveSizeIsRefused() {
        DraftHeadQuantizer.subset(320, 0, null, new int[] {0}, "v");
    }

    @Test(expected = IllegalArgumentException.class)
    public void mandatoryIdsBeyondTheSizeAreRefused() {
        DraftHeadQuantizer.subset(320, 2, null, new int[] {0, 1, 2, 3}, "v");
    }

    @Test(expected = IllegalArgumentException.class)
    public void anIdOutsideTheVocabularyIsRefused() {
        DraftHeadQuantizer.subset(320, 4, null, new int[] {320}, "v");
    }

    // ------------------------------------------------------------------ ranking file

    @Test
    public void aRankedIdsObjectRanks() throws IOException {
        Path file = Files.createTempFile("rank", ".json");
        try {
            Files.writeString(file, "{\"ranked_ids\": [7, 3, 100]}");
            assertArrayEquals(new int[] {7, 3, 100},
                    DraftHeadQuantizer.readRankedIds(file, 320));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void aBareArrayRanksAndLinesCarryCounts() throws IOException {
        Path array = Files.createTempFile("rank", ".json");
        Path lines = Files.createTempFile("rank", ".txt");
        try {
            Files.writeString(array, "[4, 9]");
            assertArrayEquals(new int[] {4, 9}, DraftHeadQuantizer.readRankedIds(array, 320));
            Files.writeString(lines, "5 1200\n30 7\n\n");
            assertArrayEquals(new int[] {5, 30}, DraftHeadQuantizer.readRankedIds(lines, 320));
        } finally {
            Files.deleteIfExists(array);
            Files.deleteIfExists(lines);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void aRepeatedRankIsRefused() throws IOException {
        Path file = Files.createTempFile("rank", ".json");
        try {
            Files.writeString(file, "[7, 7]");
            DraftHeadQuantizer.readRankedIds(file, 320);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void anOutOfRangeRankIsRefused() throws IOException {
        Path file = Files.createTempFile("rank", ".txt");
        try {
            Files.writeString(file, "5\n999\n");
            DraftHeadQuantizer.readRankedIds(file, 320);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void aNonIntegerRankIsRefused() throws IOException {
        Path file = Files.createTempFile("rank", ".json");
        try {
            Files.writeString(file, "{\"ranked_ids\": [7.5]}");
            DraftHeadQuantizer.readRankedIds(file, 320);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    // ------------------------------------------------------------------ encodings

    @Test
    public void q8ZeroStoresWholeBlocksAndFp8PicksTheRowType() {
        assertEquals(GGMLType.Q8_0, DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.Q8_0, 64));
        assertEquals(GGMLType.F8_E4M3_B128, DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.FP8, 128));
        assertEquals(GGMLType.F8_E4M3_B32, DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.FP8, 64));
        assertEquals(GGMLType.NVFP4, DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.NVFP4, 64));
        assertEquals(GGMLType.NVFP4, DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.NVFP4, 128));
    }

    @Test(expected = IllegalArgumentException.class)
    public void aRowOfNoWholeBlocksIsRefused() {
        DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.FP8, 48);
    }

    /** An NVFP4 row needs whole 64-value blocks; the refusal names the type and the row. */
    @Test
    public void anNvfp4RowOfNoWholeBlocksIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> DraftHeadQuantizer.rowType(DraftHeadOptions.Prec.NVFP4, 48));
        assertTrue(e.getMessage(), e.getMessage().contains("NVFP4"));
        assertTrue(e.getMessage(), e.getMessage().contains("48"));
    }

    @Test
    public void theVerbatimCopyCarriesTheKeptRowsExactly() throws IOException {
        int nVocab = 320;
        int nEmbd = 64;
        float[][] head = new float[nVocab][nEmbd];
        for (int r = 0; r < nVocab; r++) {
            for (int c = 0; c < nEmbd; c++) {
                head[r][c] = (r % 7 - 3) * 0.05f + c * 0.001f;
            }
        }
        CpuTensorRef ref = CpuTensorRef.of(GGMLType.F32, buffer(head, nVocab, nEmbd), nVocab, nEmbd);
        int[] kept = {5, 100, 319};
        byte[] copied = DraftHeadQuantizer.copyRows(ref, kept, nEmbd);
        assertEquals(nEmbd * 4L * kept.length, copied.length);
        float[] row = new float[nEmbd];
        for (int i = 0; i < kept.length; i++) {
            CpuTensorRef restored = CpuTensorRef.of(GGMLType.F32,
                    ByteBuffer.wrap(copied).order(ByteOrder.LITTLE_ENDIAN), kept.length, nEmbd);
            restored.decodeRows(i, 1, row, 0);
            assertArrayEquals("row " + kept[i], head[kept[i]], row, 0.0f);
        }
    }

    @Test
    public void theQuantizedRowsRoundTripWithinTheirPrecision() {
        int rows = 3;
        int cols = 64;
        float[][] head = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                head[r][c] = (float) (Math.sin(r * 31 + c) * (0.4 - 0.03 * r));
            }
        }
        float[] flat = flatten(head, rows, cols);
        DraftHeadQuantizer.Encoded q8 = DraftHeadQuantizer.encode(DraftHeadOptions.Prec.Q8_0, rows, cols,
                (r, dst) -> System.arraycopy(flat, r * cols, dst, 0, cols));
        DraftHeadQuantizer.Encoded fp8 = DraftHeadQuantizer.encode(DraftHeadOptions.Prec.FP8, rows, cols,
                (r, dst) -> System.arraycopy(flat, r * cols, dst, 0, cols));
        assertEquals(rows * 2 * (2 + 32), q8.bytes().length);
        assertEquals(rows * 2 * (4 + 32), fp8.bytes().length);
        assertNull("Q8_0 and FP8 blocks carry their full scale", q8.outputScale());
        assertNull(fp8.outputScale());
        assertRowsApproximate(GGMLType.Q8_0, q8.bytes(), rows, cols, flat, 0.05f);
        assertRowsApproximate(GGMLType.F8_E4M3_B32, fp8.bytes(), rows, cols, flat, 0.05f);
    }

    /**
     * The NVFP4 rows round trip through a dequantize oracle: the blocks decode to the source values without the
     * global scale (within the E2M1 grid and E4M3 scale rounding) and the output multiplier is vLLM's
     * {@code weight_scale_2 = 1 / ((1/amax) * 6 * 448)} over the whole tensor.
     */
    @Test
    public void theNvfp4RowsRoundTripThroughTheirGlobalScale() {
        int rows = 3;
        int cols = 64;
        float[][] head = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                head[r][c] = (float) (Math.sin(r * 31 + c) * (0.4 - 0.03 * r));
            }
        }
        float[] flat = flatten(head, rows, cols);
        float amax = 0.0f;
        for (float v : flat) {
            amax = Math.max(amax, Math.abs(v));
        }
        DraftHeadQuantizer.Encoded nvfp4 = DraftHeadQuantizer.encode(DraftHeadOptions.Prec.NVFP4, rows, cols,
                (r, dst) -> System.arraycopy(flat, r * cols, dst, 0, cols));
        assertEquals("one 36-byte block per 64-value row", rows * 36, nvfp4.bytes().length);
        float[] ws2 = nvfp4.outputScale();
        assertEquals("one global scale for the tensor", 1, ws2.length);
        float gs = (1.0f / Math.max(amax, 1e-8f)) * (6.0f * 448.0f);
        assertEquals("the scale of vLLM's ref_nvfp4_quant", 1.0f / gs, ws2[0], 0.0f);
        CpuTensorRef ref = CpuTensorRef.of(GGMLType.NVFP4,
                ByteBuffer.wrap(nvfp4.bytes()).order(ByteOrder.LITTLE_ENDIAN), rows, cols);
        float[] row = new float[cols];
        for (int r = 0; r < rows; r++) {
            ref.decodeRows(r, 1, row, 0);
            for (int c = 0; c < cols; c++) {
                assertEquals("row " + r + " col " + c, flat[r * cols + c], row[c] * ws2[0], 0.3f * amax);
            }
        }
    }

    private static void assertRowsApproximate(GGMLType type, byte[] payload, int rows, int cols, float[] flat,
                                              float tol) {
        CpuTensorRef ref = CpuTensorRef.of(type, ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN),
                rows, cols);
        float[] row = new float[cols];
        for (int r = 0; r < rows; r++) {
            ref.decodeRows(r, 1, row, 0);
            for (int c = 0; c < cols; c++) {
                assertEquals("row " + r + " col " + c, flat[r * cols + c], row[c], tol);
            }
        }
    }

    private static float[] flatten(float[][] m, int rows, int cols) {
        float[] out = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(m[r], 0, out, r * cols, cols);
        }
        return out;
    }

    private static ByteBuffer buffer(float[][] m, int rows, int cols) {
        ByteBuffer b = ByteBuffer.allocate(rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                b.putFloat(m[r][c]);
            }
        }
        b.position(0);
        return b;
    }

    // ------------------------------------------------------------------ derived source

    @Test
    public void disabledOptionsGiveTheDraftItself() throws IOException {
        try (ModelSource source = ModelSources.open(draft.toString())) {
            assertSame(source, DerivedDraftSource.wrap(source, null, DraftHeadOptions.defaults()));
            assertSame(source, DerivedDraftSource.wrap(source, null, DraftHeadOptions.builder().build()));
        }
    }

    @Test
    public void theDerivedTableCarriesTheReducedHeadAndTheMapping() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options)) {
            assertEquals(source.getTensorCount() + 1, derived.getTensorCount());
            assertTrue(derived.getTensors().stream().anyMatch(t -> t.name().equals(TensorNames.D2T)));
            GGUFTensorInfo head = info(derived, "output.weight");
            assertArrayEquals("the derived head keeps nEmbd columns of the kept count",
                    new long[] {64, 64}, new long[] {head.dim(0), head.dim(1)});
            assertEquals(source.getTensors().stream().filter(t -> t.name().equals("output.weight"))
                    .findFirst().orElseThrow().type(), head.type());
        }
    }

    @Test
    public void theMappingReadsBackAscendingWithTheSpecialsIn() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64).keepIds(40).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options);
             TensorRegions regions = derived.openRegions(TensorReadMode.READ)) {
            ByteBuffer d2t = regions.region(dInfo(derived), 0L, 64L * 8);
            int[] kept = new int[64];
            for (int i = 0; i < kept.length; i++) {
                kept[i] = (int) d2t.getLong();
            }
            for (int i = 1; i < kept.length; i++) {
                assertTrue("ascending at " + i, kept[i] > kept[i - 1]);
            }
            assertEquals("the draft's mask token stays kept", 319, kept[kept.length - 1]);
            assertTrue("the kept id stays kept", Arrays.binarySearch(kept, 40) >= 0);
        }
    }

    @Test
    public void theDerivedHeadCarriesTheKeptFileRowsExactly() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options);
             TensorRegions regions = derived.openRegions(TensorReadMode.READ);
             TensorRegions file = source.openRegions(TensorReadMode.READ)) {
            GGUFTensorInfo headInfo = info(derived, "output.weight");
            CpuTensorRef derivedHead = CpuTensorRef.of(headInfo.type(),
                    regions.region(headInfo, 0L, headInfo.sizeInBytes()), headInfo.dim(1), headInfo.dim(0));
            ByteBuffer d2t = regions.region(dInfo(derived), 0L, 64L * 8);
            int[] kept = new int[64];
            for (int i = 0; i < kept.length; i++) {
                kept[i] = (int) d2t.getLong();
            }
            GGUFTensorInfo fileHead = info(source, "output.weight");
            CpuTensorRef fileTensor = CpuTensorRef.of(fileHead.type(),
                    file.region(fileHead, 0L, fileHead.sizeInBytes()), fileHead.dim(1), fileHead.dim(0));
            float[] expected = new float[64];
            float[] got = new float[64];
            for (int r = 0; r < kept.length; r++) {
                fileTensor.decodeRows(kept[r], 1, expected, 0);
                derivedHead.decodeRows(r, 1, got, 0);
                assertArrayEquals("derived row " + r + " carries row " + kept[r], expected, got, 0.0f);
            }
        }
    }

    @Test
    public void theDerivedHeadStoresTheConfiguredPrecision() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64)
                .headPrec(DraftHeadOptions.Prec.FP8).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options)) {
            GGUFTensorInfo head = info(derived, "output.weight");
            assertEquals(GGMLType.F8_E4M3_B32, head.type());
            assertArrayEquals(new long[] {64, 64}, new long[] {head.dim(0), head.dim(1)});
        }
    }

    /**
     * The NVFP4 derived head carries the per-tensor global scale in its {@code output.scale} sibling: the
     * sibling is one F32 value holding vLLM's {@code weight_scale_2} over the kept rows, and the decoded rows
     * times it match the file rows within the E2M1 grid.
     */
    @Test
    public void theDerivedNvfp4HeadCarriesItsGlobalScaleSibling() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64)
                .headPrec(DraftHeadOptions.Prec.NVFP4).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options);
             TensorRegions regions = derived.openRegions(TensorReadMode.READ);
             TensorRegions file = source.openRegions(TensorReadMode.READ)) {
            GGUFTensorInfo head = info(derived, "output.weight");
            assertEquals(GGMLType.NVFP4, head.type());
            assertArrayEquals(new long[] {64, 64}, new long[] {head.dim(0), head.dim(1)});
            GGUFTensorInfo scale = info(derived, TensorNames.scale("output"));
            assertEquals(GGMLType.F32, scale.type());
            assertArrayEquals(new long[] {1}, scale.dims());
            ByteBuffer d2t = regions.region(dInfo(derived), 0L, 64L * 8);
            int[] kept = new int[64];
            for (int i = 0; i < kept.length; i++) {
                kept[i] = (int) d2t.getLong();
            }
            GGUFTensorInfo fileHead = info(source, "output.weight");
            CpuTensorRef fileTensor = CpuTensorRef.of(fileHead.type(),
                    file.region(fileHead, 0L, fileHead.sizeInBytes()), fileHead.dim(1), fileHead.dim(0));
            float[] row = new float[64];
            float amax = 0.0f;
            for (int id : kept) {
                fileTensor.decodeRows(id, 1, row, 0);
                for (float v : row) {
                    amax = Math.max(amax, Math.abs(v));
                }
            }
            float gs = (1.0f / Math.max(amax, 1e-8f)) * (6.0f * 448.0f);
            float ws2 = 1.0f / gs;
            assertEquals("the sibling holds the global scale of the kept rows", ws2,
                    regions.region(scale, 0L, 4L).getFloat(), 0.0f);
            CpuTensorRef derivedHead = CpuTensorRef.of(GGMLType.NVFP4,
                    regions.region(head, 0L, head.sizeInBytes()), head.dim(1), head.dim(0));
            float[] expected = new float[64];
            float[] got = new float[64];
            for (int r = 0; r < kept.length; r++) {
                fileTensor.decodeRows(kept[r], 1, expected, 0);
                derivedHead.decodeRows(r, 1, got, 0);
                for (int c = 0; c < 64; c++) {
                    assertEquals("derived row " + r + " col " + c, expected[c], got[c] * ws2, 0.3f * amax);
                }
            }
        }
    }

    /**
     * The NVFP4 drafter matrices are restated one by one, each with its own F32 {@code <base>.scale} sibling,
     * and one of them round trips through its global scale against the file matrix.
     */
    @Test
    public void theNvfp4MatricesCarryTheirGlobalScaleSiblings() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder()
                .matricesPrec(DraftHeadOptions.Prec.NVFP4).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options);
             TensorRegions regions = derived.openRegions(TensorReadMode.READ);
             TensorRegions file = source.openRegions(TensorReadMode.READ)) {
            List<GGUFTensorInfo> restated = new ArrayList<>();
            for (int i = 0; i < derived.getTensorCount(); i++) {
                GGUFTensorInfo t = derived.getTensors().get(i);
                GGUFTensorInfo fileT = infoOrNull(source, t.name());
                if (fileT == null) {
                    continue; // an appended <base>.scale sibling
                }
                if (t.type() != fileT.type()) {
                    assertEquals("the restated matrices are nvfp4: " + t.name(), GGMLType.NVFP4, t.type());
                    restated.add(t);
                }
            }
            assertTrue("the drafter matrices are restated, got " + restated.size(), restated.size() >= 1);
            assertEquals("each restated matrix gains its own scale sibling",
                    source.getTensorCount() + restated.size(), derived.getTensorCount());
            for (GGUFTensorInfo t : restated) {
                String base = t.name().substring(0, t.name().length() - TensorNames.WEIGHT.length() - 1);
                GGUFTensorInfo scale = info(derived, TensorNames.scale(base));
                assertEquals("the sibling of " + t.name() + " is one F32 value", GGMLType.F32, scale.type());
                assertArrayEquals(new long[] {1}, scale.dims());
            }
            GGUFTensorInfo fc = info(derived, "fc.weight");
            GGUFTensorInfo fcFile = info(source, "fc.weight");
            int cols = (int) fc.dim(0);
            int rows = (int) fc.dim(1);
            CpuTensorRef derivedFc = CpuTensorRef.of(GGMLType.NVFP4,
                    regions.region(fc, 0L, fc.sizeInBytes()), rows, cols);
            CpuTensorRef fileFc = CpuTensorRef.of(fcFile.type(),
                    file.region(fcFile, 0L, fcFile.sizeInBytes()), rows, cols);
            GGUFTensorInfo fcScale = info(derived, TensorNames.scale("fc"));
            float ws2 = regions.region(fcScale, 0L, 4L).getFloat();
            assertTrue("a positive global scale", ws2 > 0.0f);
            float[] expected = new float[cols];
            float[] got = new float[cols];
            for (int r = 0; r < rows; r++) {
                fileFc.decodeRows(r, 1, expected, 0);
                derivedFc.decodeRows(r, 1, got, 0);
                float rowAmax = 0.0f;
                for (int c = 0; c < cols; c++) {
                    rowAmax = Math.max(rowAmax, Math.abs(expected[c]));
                }
                for (int c = 0; c < cols; c++) {
                    assertEquals("fc row " + r + " col " + c, expected[c], got[c] * ws2, 0.3f * rowAmax);
                }
            }
        }
    }

    @Test
    public void theMatricesPrecisionRestatesTheDrafterMatrices() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().matricesPrec(DraftHeadOptions.Prec.Q8_0).build();
        try (ModelSource source = ModelSources.open(draft.toString());
             ModelSource derived = DerivedDraftSource.wrap(source, null, options)) {
            assertEquals(source.getTensorCount(), derived.getTensorCount());
            int restated = 0;
            for (int i = 0; i < derived.getTensorCount(); i++) {
                GGUFTensorInfo info = derived.getTensors().get(i);
                GGUFTensorInfo file = info(source, info.name());
                if (info.type() != file.type()) {
                    restated++;
                    assertEquals("fc and the block matrices are q8_0 now: " + info.name(), GGMLType.Q8_0,
                            info.type());
                    assertArrayEquals("the shape is unchanged", file.dims(), info.dims());
                }
            }
            assertTrue("the encoder and block matrices are restated, got " + restated, restated >= 1);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void aBorrowedHeadWithoutTheTargetIsRefused() throws IOException {
        DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64).build();
        try (ModelSource source = ModelSources.open(borrowed.toString())) {
            DerivedDraftSource.wrap(source, null, options);
        }
    }

    @Test
    public void theGatheredMarkovTableReadsTheDerivedRows() throws IOException {
        Path dspark = SyntheticDflash.dflash(SyntheticModels.Config.llama(),
                SyntheticDflash.Options.of(4, 1, 2).dspark(8)).writeTempFile();
        try {
            DraftHeadOptions options = DraftHeadOptions.builder().vocabSize(64).build();
            try (ModelSource source = ModelSources.open(dspark.toString());
                 ModelSource derived = DerivedDraftSource.wrap(source, null, options);
                 TensorRegions regions = derived.openRegions(TensorReadMode.READ);
                 TensorRegions file = source.openRegions(TensorReadMode.READ)) {
                GGUFTensorInfo gatheredInfo = info(derived, "markov_w2.weight");
                assertArrayEquals("the gathered table has the kept count of rows",
                        new long[] {8, 64}, new long[] {gatheredInfo.dim(0), gatheredInfo.dim(1)});
                assertEquals("the gathered table keeps its element type", info(source, "markov_w2.weight").type(),
                        gatheredInfo.type());
                float[] gathered = readRows(regions.region(gatheredInfo, 0L, 64L * 8 * 4), 64 * 8);
                ByteBuffer d2t = regions.region(dInfo(derived), 0L, 64L * 8);
                GGUFTensorInfo fileW2 = info(source, "markov_w2.weight");
                CpuTensorRef fileTensor = CpuTensorRef.of(fileW2.type(),
                        file.region(fileW2, 0L, fileW2.sizeInBytes()), fileW2.dim(1), fileW2.dim(0));
                float[] row = new float[8];
                for (int r = 0; r < 64; r++) {
                    int id = (int) d2t.getLong();
                    fileTensor.decodeRows(id, 1, row, 0);
                    assertArrayEquals("gathered row " + r + " carries row " + id, row,
                            Arrays.copyOfRange(gathered, r * 8, (r + 1) * 8), 0.0f);
                }
            }
        } finally {
            Files.deleteIfExists(dspark);
        }
    }

    private static float[] readRows(ByteBuffer region, int floats) {
        float[] out = new float[floats];
        for (int i = 0; i < floats; i++) {
            out[i] = region.getFloat();
        }
        return out;
    }

    private static GGUFTensorInfo dInfo(ModelSource derived) {
        return info(derived, TensorNames.D2T);
    }

    private static GGUFTensorInfo info(ModelSource source, String name) {
        return source.getTensors().stream().filter(t -> t.name().equals(name)).findFirst()
                .orElseThrow(() -> new AssertionError("no tensor " + name));
    }

    private static GGUFTensorInfo infoOrNull(ModelSource source, String name) {
        return source.getTensors().stream().filter(t -> t.name().equals(name)).findFirst().orElse(null);
    }
}
