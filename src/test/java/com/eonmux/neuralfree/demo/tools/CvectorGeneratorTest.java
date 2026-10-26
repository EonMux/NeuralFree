package com.eonmux.neuralfree.demo.tools;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link CvectorGenerator} on files in a temporary directory: the prompt entries (llama.cpp's
 * {@code prepare_entries}), the {@link CvectorGenerator.Differences} collector over the tap pairs
 * ({@code calc_diff} and {@code filter_nonzero_rows}), both reductions (pca.hpp, mean.hpp) and the GGUF export
 * round-tripping through {@code ControlVector.File.load} (the loader {@code common_control_vector_load_one}
 * mirrors). No model, engine or device is touched: the collector is fed synthetic rows directly, and the tap
 * drivers over a running engine are TL-4's serving side.
 */
public class CvectorGeneratorTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void entriesReadOnePairPerNonEmptyLineWithEscapes() throws Exception {
        Path pos = write("pos.txt", "happy day\\n", "", "line with \\t tab", "");
        Path neg = write("neg.txt", "sad day", "", "other line", "skipped");
        var pairs = CvectorGenerator.entries(pos, neg);
        assertEquals(2, pairs.size());
        assertEquals("happy day\n", pairs.get(0).positive());
        assertEquals("sad day", pairs.get(0).negative());
        assertEquals("line with \t tab", pairs.get(1).positive());
        assertEquals("other line", pairs.get(1).negative());
        // The counts must agree; a lone positive entry is refused.
        Path one = write("one.txt", "just one\n");
        assertThrows(Exception.class, () -> CvectorGenerator.entries(one, neg));
        assertThrows(Exception.class, () -> CvectorGenerator.entries(pos, one));
    }

    @Test
    public void theCollectorTracksOnlyItsLayersAndValidatesTheRows() {
        CvectorGenerator.Differences d = new CvectorGenerator.Differences(new int[]{0, 2, 5}, 4);
        assertTrue(d.tracks(0));
        assertFalse(d.tracks(1));
        assertTrue(d.tracks(5));
        assertFalse(d.tracks(-1));
        assertFalse(d.tracks(6));
        assertArrayEquals(new int[]{0, 2, 5}, d.layers());
        assertEquals(4, d.width());
        assertThrows(IllegalArgumentException.class, () -> new CvectorGenerator.Differences(new int[]{}, 4));
        assertThrows(IllegalArgumentException.class, () -> new CvectorGenerator.Differences(new int[]{1, 1}, 4));
        assertThrows(IllegalArgumentException.class, () -> new CvectorGenerator.Differences(new int[]{2, 1}, 4));
        assertThrows(IllegalArgumentException.class, () -> new CvectorGenerator.Differences(new int[]{-1}, 4));
        assertThrows(IllegalArgumentException.class, () -> new CvectorGenerator.Differences(new int[]{0}, 0));
        d.beginPair();
        // Untracked layers, wrong widths, short arrays and taps outside a pair are refused.
        assertThrows(IllegalStateException.class, () -> d.addSide(1, new float[4], 1, 4));
        assertThrows(IllegalArgumentException.class, () -> d.addSide(0, new float[3], 1, 3));
        assertThrows(IllegalArgumentException.class, () -> d.addSide(0, new float[3], 1, 4));
        assertThrows(IllegalArgumentException.class, () -> d.addSide(0, new float[8], -1, 4));
        CvectorGenerator.Differences unstarted = new CvectorGenerator.Differences(new int[]{0}, 4);
        assertThrows(IllegalStateException.class, () -> unstarted.addSide(0, new float[4], 1, 4));
        assertThrows(IllegalStateException.class, unstarted::endPair);
        assertThrows(IllegalStateException.class, () -> d.setSide(false));
    }

    @Test
    public void endPairKeepsTheNonZeroDifferencesInRowOrder() {
        CvectorGenerator.Differences d = new CvectorGenerator.Differences(new int[]{0, 1}, 2);
        float[] pos = {1, 2, 3, 4, 5, 6};
        float[] neg = {1, 2, 0, 0, -1, -2};
        d.beginPair();
        d.addSide(0, pos, 3, 2);
        d.addSide(1, neg, 3, 2);
        d.setSide(false);
        d.addSide(0, neg, 3, 2);
        d.addSide(1, pos, 3, 2);
        d.endPair();
        // Row 0 cancels (|v| <= 1e-6 for every element) and drops; rows 1 and 2 keep their order.
        assertArrayEquals(new float[]{3, 4, 6, 8}, d.rowBlock(0), 0f);
        assertArrayEquals(new float[]{-3, -4, -6, -8}, d.rowBlock(1), 0f);
        assertEquals(3, d.rows(0));
        assertEquals(3, d.rows(1));
        assertEquals(1, d.pairs());
        // A second pair appends to what the first kept.
        float[] p2 = {10, 20, 30, 40};
        float[] n2 = {0, 0, 30, 40};
        d.beginPair();
        d.addSide(0, p2, 2, 2);
        d.addSide(1, p2, 2, 2);
        d.setSide(false);
        d.addSide(0, n2, 2, 2);
        d.addSide(1, n2, 2, 2);
        d.endPair();
        assertArrayEquals(new float[]{3, 4, 6, 8, 10, 20}, d.rowBlock(0), 0f);
        assertEquals(4, d.rows(0));
        assertEquals(2, d.pairs());
    }

    @Test
    public void theCollectorRequiresEqualSideRowCountsPerLayer() {
        CvectorGenerator.Differences d = new CvectorGenerator.Differences(new int[]{0}, 2);
        d.beginPair();
        d.addSide(0, new float[4], 2, 2);
        d.setSide(false);
        assertThrows(IllegalStateException.class, () -> d.addSide(0, new float[2], 1, 2));
        // Completing the pair repairs nothing: the mismatch is named, not padded.
        d.addSide(0, new float[2], 1, 2);
        assertThrows(IllegalStateException.class, d::endPair);
        // A side with no rows at all is its own error.
        CvectorGenerator.Differences e = new CvectorGenerator.Differences(new int[]{0}, 2);
        e.beginPair();
        assertThrows(IllegalStateException.class, e::endPair);
    }

    @Test
    public void theCollectorSurvivesChunkedSides() {
        // The GPU driver commits one read per ubatch: several chunks of rows per side and layer.
        CvectorGenerator.Differences d = new CvectorGenerator.Differences(new int[]{0, 1}, 2);
        d.beginPair();
        d.addSide(0, new float[]{1, 2, 3, 4}, 2, 2);
        d.addSide(0, new float[]{5, 6}, 1, 2);
        d.addSide(1, new float[]{-1, -2, -3, -4, -5, -6}, 3, 2);
        d.setSide(false);
        d.addSide(0, new float[]{1, 0, 3, 4}, 2, 2);
        d.addSide(0, new float[]{0, 6}, 1, 2);
        d.addSide(1, new float[]{1, 2, 3, 4, 5, 6}, 3, 2);
        d.endPair();
        assertArrayEquals(new float[]{0, 2, 0, 0, 5, 0}, d.rowBlock(0), 0f);
        assertArrayEquals(new float[]{-2, -4, -6, -8, -10, -12}, d.rowBlock(1), 0f);
    }

    @Test
    public void pcaFindsTheDominantAxisAndMeanTheRowMean() {
        // Rows along x with a small y component: the top component is (near) the x axis.
        float[] rows = {3, 0.2f, -4, -0.3f, 5, 0.1f, -6, -0.2f};
        float[] pc = CvectorGenerator.Pca.component(rows, 4, 2, 20, 1000, 1e-7f);
        assertEquals(1f, Math.abs(pc[0]), 1e-4f);
        assertTrue(Math.abs(pc[1]) < 0.05f);
        assertThrows(IllegalArgumentException.class,
                () -> CvectorGenerator.Pca.component(rows, 4, 2, 1, 0, 1e-7f));
        assertThrows(IllegalArgumentException.class,
                () -> CvectorGenerator.Pca.component(new float[4], 2, 2, 1, 1, 1e-7f));
        // The mean direction is the normalized row mean; a zero mean stays zero.
        float[] mean = CvectorGenerator.Mean.of(new float[]{2, 0, 4, 0}, 2, 2);
        assertEquals(1f, mean[0], 1e-6f);
        assertEquals(0f, mean[1], 1e-6f);
        assertArrayEquals(new float[]{0f, 0f}, CvectorGenerator.Mean.of(new float[]{2, 0, -2, 0}, 2, 2), 0f);
        assertThrows(IllegalArgumentException.class, () -> CvectorGenerator.Mean.of(new float[2], 0, 2));
    }

    @Test
    public void trainReducesEveryTrackedLayerAndRefusesAnEmptyOne() throws Exception {
        CvectorGenerator.Differences d = new CvectorGenerator.Differences(new int[]{0, 1}, 2);
        d.beginPair();
        d.addSide(0, new float[]{2, 0, 4, 0}, 2, 2);
        d.addSide(1, new float[]{0, 2, 0, 4}, 2, 2);
        d.setSide(false);
        d.addSide(0, new float[]{0, 0, 0, 0}, 2, 2);
        d.addSide(1, new float[]{0, 0, 0, 0}, 2, 2);
        d.endPair();
        CvectorGenerator.Options pca = CvectorGenerator.Options.defaults().toBuilder()
                .method(CvectorGenerator.Method.MEAN).build();
        var directions = CvectorGenerator.train(d, pca);
        assertEquals(2, directions.size());
        assertArrayEquals(new float[]{1, 0}, directions.get(0), 1e-6f);
        assertArrayEquals(new float[]{0, 1}, directions.get(1), 1e-6f);
        // A layer whose rows all cancelled has nothing to reduce.
        CvectorGenerator.Differences zero = new CvectorGenerator.Differences(new int[]{0, 1}, 2);
        zero.beginPair();
        zero.addSide(0, new float[]{1, 1}, 1, 2);
        zero.addSide(1, new float[]{1, 1}, 1, 2);
        zero.setSide(false);
        zero.addSide(0, new float[]{1, 1}, 1, 2);
        zero.addSide(1, new float[]{1, 1}, 1, 2);
        zero.endPair();
        assertThrows(java.io.IOException.class, () -> CvectorGenerator.train(zero, pca));
    }

    @Test
    public void exportWritesTheLlamaCppFileFormThatTheLoaderReadsBack() throws Exception {
        Path out = tmp.getRoot().toPath().resolve("cv.gguf");
        var directions = Arrays.asList(new float[]{1, 0, 0, 0}, new float[]{0, 1, 0, 0});
        Path written = CvectorGenerator.export(directions, 4, "llama", out, false);
        assertEquals(out, written);
        ControlVector.File f = ControlVector.File.load(out.toString());
        assertEquals(4, f.nEmbd());
        assertEquals(2, f.maxLayer());
        assertArrayEquals(new int[]{1, 2}, f.layers());
        assertArrayEquals(new float[]{1, 0, 0, 0}, f.row(1), 0f);
        assertArrayEquals(new float[]{0, 1, 0, 0}, f.row(2), 0f);
        // A gap layer reads zero, as llama.cpp's zero-filled span does.
        assertArrayEquals(new float[]{0, 0, 0, 0}, f.row(5), 0f);
        assertEquals("controlvector", f.metadata().get("general.architecture"));
        assertEquals("llama", f.metadata().get("controlvector.model_hint"));
        assertEquals("2", f.metadata().get("controlvector.layer_count"));
        // Refusals: no direction, a wrong width, a blank hint, an existing file without --overwrite.
        assertThrows(IllegalArgumentException.class, () -> CvectorGenerator.export(
                Arrays.asList(), 4, "llama", out, true));
        assertThrows(IllegalArgumentException.class, () -> CvectorGenerator.export(
                Arrays.asList(new float[]{1, 2}), 4, "llama", out, true));
        assertThrows(IllegalArgumentException.class, () -> CvectorGenerator.export(
                directions, 0, "llama", out, true));
        assertThrows(IllegalArgumentException.class, () -> CvectorGenerator.export(
                directions, 4, " ", out, true));
        assertThrows(java.io.IOException.class, () -> CvectorGenerator.export(
                directions, 4, "llama", out, false));
    }

    @Test
    public void theOptionParserCarriesTheLlamaCppFlagsAndProperties() throws Exception {
        CvectorGenerator.Options o = CvectorGenerator.parse(new String[]{
                "--positive-file", "p.txt", "--negative-file", "n.txt", "--out-file", "o.gguf",
                "--pca-batch", "7", "--pca-iter", "99", "--method", "mean", "--model-hint", "qwen",
                "--overwrite"});
        assertEquals("p.txt", o.positiveFile());
        assertEquals("n.txt", o.negativeFile());
        assertEquals("o.gguf", o.outFile());
        assertEquals(7, o.pcaBatch());
        assertEquals(99, o.pcaIterations());
        assertEquals(CvectorGenerator.Method.MEAN, o.method());
        assertEquals("qwen", o.modelHint());
        assertTrue(o.overwrite());
        assertNull(CvectorGenerator.parse(new String[]{"--help"}));
        assertThrows(CvectorGenerator.UsageException.class, () -> CvectorGenerator.parse(new String[]{
                "--pca-batch", "x"}));
        assertThrows(CvectorGenerator.UsageException.class, () -> CvectorGenerator.parse(new String[]{
                "--method", "wavelet"}));
        assertThrows(CvectorGenerator.UsageException.class, () -> CvectorGenerator.parse(new String[]{
                "--unknown"}));
        assertThrows(CvectorGenerator.UsageException.class, () -> CvectorGenerator.parse(new String[]{
                "--positive-file"}));
        assertThrows(CvectorGenerator.UsageException.class, () -> CvectorGenerator.parse(new String[]{}));
        java.util.Properties p = new java.util.Properties();
        p.setProperty(CvectorGenerator.Options.PROP_METHOD, "mean");
        p.setProperty(CvectorGenerator.Options.PROP_PCA_BATCH, "5");
        assertEquals(CvectorGenerator.Method.MEAN, CvectorGenerator.Options.fromProperties(p).method());
        assertEquals(5, CvectorGenerator.Options.fromProperties(p).pcaBatch());
        p.setProperty(CvectorGenerator.Options.PROP_METHOD, "nope");
        assertThrows(IllegalArgumentException.class, () -> CvectorGenerator.Options.fromProperties(p));
    }

    private Path write(String name, String... lines) throws Exception {
        Path p = tmp.getRoot().toPath().resolve(name);
        Files.write(p, String.join("\n", Arrays.asList(lines)).getBytes(StandardCharsets.UTF_8));
        return p;
    }
}
