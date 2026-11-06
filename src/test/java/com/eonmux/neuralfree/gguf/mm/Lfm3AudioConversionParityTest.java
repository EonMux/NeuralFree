package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfLfm3AudioFixtures;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The LFM3 audio projector export end to end: a synthetic Hugging Face checkpoint holding the same encoder weights as
 * {@link Lfm3aTowerTest#fixture()} converts to an mmproj GGUF, that file opens through {@link MmprojModel}, and its
 * {@link CpuAudioEncoder} reproduces the double-precision lfm3a reference. The converted file carries a fourth
 * encoder block whose weights are zero, so the reference's three-block graph and the runtime's four-block graph agree
 * exactly while the extra block exercises the {@code a.blk.3.*} names.
 */
public class Lfm3AudioConversionParityTest {

    private static final double TOL = 2e-4;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theConvertedAudioTowerMatchesTheReferenceGraph() throws IOException {
        MmprojFixture reference = Lfm3aTowerTest.fixture();
        Path dir = HfLfm3AudioFixtures.audioCheckpoint(tower(reference), false, StDType.F32)
                .write(tmp.newFolder("lfm3a-parity").toPath());
        Path file = tmp.newFolder("lfm3a-out").toPath().resolve("mmproj-lfm3a.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, file, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }

        MelChunk chunk = AudioTowerSupport.chunk(HfLfm3AudioFixtures.N_MEL, 77, 5);
        try (MmprojModel mm = MmprojModel.open(file.toString(), options);
             CpuAudioEncoder enc = mm.openAudioEncoder()) {
            assertTrue(mm.caps().toString(), mm.caps().inpAudio());
            assertTrue(mm.supportsAudio());
            assertEquals("lfm3a", mm.audioHparams().projectorType().typeName());
            assertEquals(HfLfm3AudioFixtures.LAYERS, mm.audioHparams().nLayer());
            assertEquals(HfLfm3AudioFixtures.N_MEL, mm.audioHparams().audio().nMelBins());
            assertTrue(mm.tensorNames().contains("a.blk.3.conv_pw2.weight"));
            assertTrue(mm.tensorNames().contains("mm.a.mlp.3.weight"));

            float[][] got = enc.encode(chunk);
            AudioTowerSupport.assertRows(Lfm3aTowerTest.reference(reference, chunk), got, TOL);
            assertEquals(10, got.length);
            assertEquals(HfLfm3AudioFixtures.TEXT, got[0].length);
        }
    }

    @Test
    public void theConvertedOmniAudioTowerMatchesTheReferenceGraph() throws IOException {
        MmprojFixture reference = Lfm3aTowerTest.fixture();
        Path dir = HfLfm3AudioFixtures.omniCheckpoint(tower(reference), false, StDType.F32)
                .write(tmp.newFolder("lfm3omni-parity").toPath());
        Path file = tmp.newFolder("lfm3omni-out").toPath().resolve("mmproj-lfm3omni.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-omni").toPath()).build();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, file, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }

        MelChunk chunk = AudioTowerSupport.chunk(HfLfm3AudioFixtures.N_MEL, 77, 5);
        try (MmprojModel mm = MmprojModel.open(file.toString(), options);
             CpuAudioEncoder enc = mm.openAudioEncoder()) {
            assertTrue(mm.caps().toString(), mm.caps().inpVision());
            assertTrue(mm.caps().toString(), mm.caps().inpAudio());
            assertEquals("lfm3vl", mm.visionHparams().projectorType().typeName());
            assertEquals("lfm3a", mm.audioHparams().projectorType().typeName());
            assertTrue(mm.supportsAudio());
            assertFalse("the fixture ships no vision tensors", mm.supportsVision());

            float[][] got = enc.encode(chunk);
            AudioTowerSupport.assertRows(Lfm3aTowerTest.reference(reference, chunk), got, TOL);
        }
    }

    @Test
    public void theConvertedBlocksCarryTheirSourceWeights() throws IOException {
        MmprojFixture reference = Lfm3aTowerTest.fixture();
        Path dir = HfLfm3AudioFixtures.audioCheckpoint(tower(reference), false, StDType.F32)
                .write(tmp.newFolder("lfm3a-weights").toPath());
        Path file = tmp.newFolder("lfm3a-weights-out").toPath().resolve("mmproj-lfm3a.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-w").toPath()).build();
        try (HfModelSource s = HfModelSource.openProjector(dir, options)) {
            GgufWriter.write(s, file, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (MmprojModel mm = MmprojModel.open(file.toString(), options);
             CpuWeightsHolder holder = new CpuWeightsHolder(file)) {
            assertArrayEquals(reference.get("a.blk.0.conv_pw1.weight"),
                    holder.floats("a.blk.0.conv_pw1.weight"), 1e-6f);
            assertArrayEquals(reference.get("a.blk.1.attn_q.weight"),
                    holder.floats("a.blk.1.attn_q.weight"), 1e-6f);
            assertArrayEquals(reference.get("a.pre_encode.out.weight"),
                    holder.floats("a.pre_encode.out.weight"), 1e-6f);
            assertArrayEquals(reference.get("mm.a.mlp.3.bias"), holder.floats("mm.a.mlp.3.bias"), 1e-6f);
            assertArrayEquals(new float[HfLfm3AudioFixtures.HIDDEN * HfLfm3AudioFixtures.FF],
                    holder.floats("a.blk.3.ffn_up.weight"), 1e-6f);
        }
    }

    /** The runtime fixture's encoder tensors plus an all-zero fourth block. */
    private static HfLfm3AudioFixtures.Tower tower(MmprojFixture f) {
        Map<String, long[]> shapes = new LinkedHashMap<>(f.shapes);
        Map<String, float[]> values = new LinkedHashMap<>(f.values);
        int hidden = Lfm3aTowerTest.HIDDEN;
        int ff = Lfm3aTowerTest.FF;
        String p = "a.blk." + Lfm3aTowerTest.CONV.length + ".";
        shapes.put(p + "ln1.weight", new long[]{hidden});
        shapes.put(p + "ln2.weight", new long[]{hidden});
        shapes.put(p + "ffn_up.weight", new long[]{hidden, ff});
        shapes.put(p + "ffn_down.weight", new long[]{ff, hidden});
        shapes.put(p + "conv_pw1.weight", new long[]{hidden, 3L * hidden});
        shapes.put(p + "conv_dw.weight", new long[]{Lfm3aTowerTest.TAPS, hidden});
        shapes.put(p + "conv_pw2.weight", new long[]{hidden, hidden});
        float[] ones = new float[hidden];
        java.util.Arrays.fill(ones, 1f);
        values.put(p + "ln1.weight", ones);
        values.put(p + "ln2.weight", ones.clone());
        values.put(p + "ffn_up.weight", new float[hidden * ff]);
        values.put(p + "ffn_down.weight", new float[ff * hidden]);
        values.put(p + "conv_pw1.weight", new float[hidden * 3 * hidden]);
        values.put(p + "conv_dw.weight", new float[Lfm3aTowerTest.TAPS * hidden]);
        values.put(p + "conv_pw2.weight", new float[hidden * hidden]);
        return HfLfm3AudioFixtures.tower(shapes, values);
    }

    /** The converted weights read back as plain floats. */
    private static final class CpuWeightsHolder implements AutoCloseable {
        private final com.eonmux.neuralfree.gguf.ModelSource source;
        private final com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights weights;

        CpuWeightsHolder(Path file) throws IOException {
            source = com.eonmux.neuralfree.gguf.ModelSources.parseGguf(file.toString());
            weights = com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights.open(source);
        }

        float[] floats(String name) {
            return weights.floats(name);
        }

        @Override
        public void close() throws IOException {
            weights.close();
            source.close();
        }
    }
}
