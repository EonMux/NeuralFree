package com.eonmux.neuralfree.gguf.quant.write;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.load.MetadataOverrides;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The quantize tool end to end over the synthetic llama model: recipes and mixtures land in real GGUF
 * output, the metadata rules hold, requantize and the imatrix requirement refuse by name, keep-split
 * reproduces the input shards, prune-layers renumbers, and the dry run writes nothing.
 */
public class QuantizeToolTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    /** A 4-layer llama whose every weight row is 256 values, stored F32. */
    private Path llamaModel() throws IOException {
        GgufTestWriter w = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nLayer(4).nEmbd(256).nFf(256).headDim(64).nRot(64).build());
        Path p = tmp.newFile("llama-f32.gguf").toPath();
        w.writeTo(p);
        return p;
    }

    /** The type of one tensor of a GGUF file. */
    private static GGMLType typeOf(Path file, String name) throws IOException {
        try (GGUFFileParser parser = ModelSources.parseGguf(file.toString())) {
            for (GGUFTensorInfo t : parser.getTensors()) {
                if (t.name().equals(name)) {
                    return t.type();
                }
            }
            throw new AssertionError("no tensor " + name + " in " + file);
        }
    }

    private static Long kvValue(Path file, String key) throws IOException {
        try (GGUFFileParser parser = ModelSources.parseGguf(file.toString())) {
            return parser.getUInt32(key);
        }
    }

    private static String string(Path file, String key) throws IOException {
        try (GGUFFileParser parser = ModelSources.parseGguf(file.toString())) {
            return parser.getString(key);
        }
    }

    private static List<String> names(Path file) throws IOException {
        try (GGUFFileParser parser = ModelSources.parseGguf(file.toString())) {
            List<String> out = new ArrayList<>();
            for (GGUFTensorInfo t : parser.getTensors()) {
                out.add(t.name());
            }
            return out;
        }
    }

    private static long elements(GGUFTensorInfo t) {
        long n = 1;
        for (long d : t.dims()) {
            n *= d;
        }
        return n;
    }

    @Test
    public void q8_0QuantizesTheWeightsAndKeepsTheRest() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("q8.gguf");
        QuantizeTool.Result r = QuantizeTool.run(in, out, QuantizeParams.builder()
                .ftype(QuantizeFtype.Q8_0).build());
        assertEquals(List.of(out), r.files);
        assertEquals(7L, kvValue(out, "general.file_type").longValue());
        assertEquals(QuantizeTool.QNT_VERSION, kvValue(out, "general.quantization_version").longValue());
        for (String name : new String[] {"token_embd.weight", "output.weight", "blk.0.attn_q.weight",
                "blk.3.attn_v.weight", "blk.2.ffn_down.weight"}) {
            assertEquals(name, GGMLType.Q8_0, typeOf(out, name));
        }
        for (String name : new String[] {"blk.0.attn_norm.weight", "output_norm.weight"}) {
            assertEquals(name, GGMLType.F32, typeOf(out, name));
        }
        assertTrue("quantized output is smaller", r.totalSizeNew < r.totalSizeOrg);
    }

    @Test
    public void q8_0OutputRoundTripsToItsValues() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("q8rt.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q8_0).build());
        try (GGUFFileParser q = ModelSources.parseGguf(out.toString());
             GGUFFileParser src = ModelSources.parseGguf(in.toString())) {
            GGUFTensorInfo info = q.getTensor("blk.1.attn_q.weight");
            byte[] raw = new byte[(int) info.sizeInBytes()];
            q.readTensorData(info, 0, raw, 0, raw.length);
            float[] back = CpuBlockDecoders.dequantize(raw, GGMLType.Q8_0, elements(info));
            float[] original = src.readTensorDataAsFloat(src.getTensor("blk.1.attn_q.weight"));
            assertEquals(original.length, back.length);
            float maxErr = 0;
            for (int i = 0; i < original.length; i++) {
                maxErr = Math.max(maxErr, Math.abs(original[i] - back[i]));
            }
            assertTrue("round trip error " + maxErr, maxErr < 1e-3f);
        }
    }

    @Test
    public void q4KMixtureLandsInRealTypes() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("q4km.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q4_K_M).build());
        assertEquals(15L, kvValue(out, "general.file_type").longValue());
        // the mixture over the 4-layer model: the third and fourth of each counter take Q6_K
        assertEquals(GGMLType.Q4_K, typeOf(out, "blk.0.attn_v.weight"));
        assertEquals(GGMLType.Q4_K, typeOf(out, "blk.0.ffn_down.weight"));
        assertEquals(GGMLType.Q6_K, typeOf(out, "blk.3.attn_v.weight"));
        assertEquals(GGMLType.Q6_K, typeOf(out, "blk.3.ffn_down.weight"));
        assertEquals(GGMLType.Q6_K, typeOf(out, "output.weight"));
        assertEquals(GGMLType.Q4_K, typeOf(out, "token_embd.weight"));
        assertEquals(GGMLType.F32, typeOf(out, "blk.1.attn_norm.weight"));
    }

    @Test
    public void iq4XsRecipe() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("iq4xs.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.IQ4_XS).build());
        assertEquals(30L, kvValue(out, "general.file_type").longValue());
        assertEquals(GGMLType.IQ4_XS, typeOf(out, "blk.0.attn_q.weight"));
        assertEquals(GGMLType.Q6_K, typeOf(out, "output.weight"));
    }

    @Test
    public void requantizeRefusesThenAllows() throws IOException {
        Path in = llamaModel();
        Path q8 = tmp.newFile("base-q8.gguf").toPath();
        QuantizeTool.run(in, q8, QuantizeParams.builder().ftype(QuantizeFtype.Q8_0).build());
        Path out = tmp.getRoot().toPath().resolve("requant.gguf");
        try {
            QuantizeTool.run(q8, out, QuantizeParams.builder().ftype(QuantizeFtype.Q4_K_M).build());
            fail("requantizing must be refused without --allow-requantize");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("requantizing from type q8_0 is disabled"));
        }
        assertFalse(Files.exists(out));
        QuantizeTool.Result r = QuantizeTool.run(q8, out, QuantizeParams.builder()
                .ftype(QuantizeFtype.Q4_K_M).allowRequantize(true).build());
        assertTrue(r.totalSizeNew < r.totalSizeOrg);
        assertEquals(GGMLType.Q4_K, typeOf(out, "blk.0.attn_q.weight"));
    }

    @Test
    public void q2_KSNeedsTheImatrix() throws IOException {
        Path in = llamaModel();
        Path plain = tmp.getRoot().toPath().resolve("q2k-plain.gguf");
        try {
            QuantizeTool.run(in, plain, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K_S).build());
            fail("Q2_K_S without an imatrix must be refused");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("requires an importance matrix"));
            assertFalse(Files.exists(plain));
        }
        // Q2_K, one step up, runs without one
        QuantizeTool.run(in, plain, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K).build());
        assertEquals(GGMLType.Q2_K, typeOf(plain, "blk.0.attn_q.weight"));
    }

    @Test
    public void imatrixDrivesTheQuantizationAndItsKeys() throws IOException {
        Path in = llamaModel();
        Path matrix = imatrixFor(in, "imatrix.gguf", 256);
        Path out = tmp.getRoot().toPath().resolve("q2ks.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K_S)
                .imatrixFile(matrix).build());
        assertEquals(GGMLType.Q2_K, typeOf(out, "blk.0.ffn_down.weight"));
        assertEquals("imatrix", string(matrix, "general.type"));
        assertEquals(matrix.toString(), string(out, "quantize.imatrix.file"));
        assertTrue(kvValue(out, "quantize.imatrix.entries_count") > 0);
        // an include filter that misses every tensor empties the matrix, and the low-bit quantization
        // still refuses: llama.cpp behaves the same when its filtered matrix comes back empty
        Path filtered = tmp.getRoot().toPath().resolve("q2ks-filtered.gguf");
        try {
            QuantizeTool.run(in, filtered, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K_S)
                    .imatrixFile(matrix).includeWeights(List.of("no-such-tensor")).build());
            fail("an emptied imatrix must not silently quantize Q2_K_S");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("importance matrix"));
        }
        assertFalse(Files.exists(filtered));
    }

    /** Writes an imatrix GGUF with one entry per 2-D weight tensor of {@code model}. */
    private Path imatrixFor(Path model, String name, int cols) throws IOException {
        List<ImatrixFile.Entry> entries = new ArrayList<>();
        Random r = new Random(41);
        try (GGUFFileParser parser = ModelSources.parseGguf(model.toString())) {
            for (GGUFTensorInfo t : parser.getTensors()) {
                if (t.dims().length < 2 || !t.name().endsWith("weight")) {
                    continue;
                }
                float[] sums = new float[cols];
                for (int i = 0; i < cols; i++) {
                    sums[i] = 0.5f + r.nextFloat();
                }
                entries.add(new ImatrixFile.Entry(t.name(), 1, cols, sums, new long[] {1000}));
            }
        }
        Path out = tmp.newFile(name).toPath();
        ImatrixFile.writeGguf(out, entries, List.of("calibration.txt"), 3, 512);
        return out;
    }

    @Test
    public void missingImatrixEntryForALowBitTensorRefuses() throws IOException {
        Path in = llamaModel();
        Path matrix = tmp.newFile("partial.gguf").toPath();
        ImatrixFile.writeGguf(matrix, List.of(new ImatrixFile.Entry("blk.0.attn_q.weight", 1, 256,
                new float[256], new long[] {1000})), List.of("calibration.txt"), 1, 512);
        Path out = tmp.getRoot().toPath().resolve("q2ks-partial.gguf");
        try {
            QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K_S)
                    .imatrixFile(matrix).build());
            fail("a missing low-bit imatrix entry must be refused");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Missing importance matrix"));
        }
    }

    @Test
    public void legacyDatImatrixWorks() throws IOException {
        Path in = llamaModel();
        Path legacy = tmp.newFile("legacy.dat").toPath();
        java.io.DataOutputStream o = new java.io.DataOutputStream(
                new java.io.FileOutputStream(legacy.toFile()));
        byte[] name = "blk.0.attn_q.weight".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        writeInt(o, 1); // one entry
        writeInt(o, name.length);
        o.write(name);
        writeInt(o, 1); // ncall
        writeInt(o, 256); // nval
        for (int i = 0; i < 256; i++) {
            writeInt(o, Float.floatToIntBits(1.0f));
        }
        o.flush();
        o.close();
        Path out = tmp.getRoot().toPath().resolve("legacy-quant.gguf");
        // Q2_K (not _S) runs without per-tensor entries; the legacy matrix feeds the ones it names
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K)
                .imatrixFile(legacy).build());
        assertEquals(GGMLType.Q2_K, typeOf(out, "blk.0.attn_q.weight"));
        assertEquals(legacy.toString(), string(out, "quantize.imatrix.file"));
    }

    @Test
    public void tensorTypeOverrideAppliesToItsMatches() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("override.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q8_0)
                .tensorTypes(List.of(new QuantizeParams.TensorType(
                        java.util.regex.Pattern.compile("attn_q"), "attn_q", GGMLType.Q4_0)))
                .build());
        assertEquals(GGMLType.Q4_0, typeOf(out, "blk.0.attn_q.weight"));
        assertEquals(GGMLType.Q8_0, typeOf(out, "blk.0.attn_k.weight"));
    }

    @Test
    public void leaveOutputTensorKeepsOutput() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("leave.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q4_K_M)
                .quantizeOutputTensor(false).build());
        assertEquals(GGMLType.F32, typeOf(out, "output.weight"));
    }

    @Test
    public void overrideKvLandsInTheMetadata() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("kv.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q8_0)
                .kvOverrides(List.of(MetadataOverrides.parse("general.name=str:quantized-by-nf", "test")))
                .build());
        assertEquals("quantized-by-nf", string(out, "general.name"));
    }

    @Test
    public void dryRunWritesNothing() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("dry.gguf");
        QuantizeTool.Result r = QuantizeTool.run(in, out, QuantizeParams.builder()
                .ftype(QuantizeFtype.Q4_K_M).dryRun(true).build());
        assertTrue(r.files.isEmpty());
        assertFalse(Files.exists(out));
        assertTrue(r.totalSizeNew < r.totalSizeOrg);
    }

    @Test
    public void sameInputAndOutputRefuses() throws IOException {
        Path in = llamaModel();
        try {
            QuantizeTool.run(in, in, QuantizeParams.builder().ftype(QuantizeFtype.Q8_0).build());
            fail("in == out must be refused");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("input and output files are the same"));
        }
    }

    @Test
    public void pruneLayersRenumbersAndRemapsTheImatrix() throws IOException {
        Path in = llamaModel();
        Path matrix = imatrixFor(in, "prune.gguf", 256);
        Path out = tmp.getRoot().toPath().resolve("pruned.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q2_K_S)
                .imatrixFile(matrix).pruneLayers(List.of(1)).build());
        List<String> outNames = names(out);
        assertTrue(outNames.contains("blk.0.attn_q.weight"));
        assertTrue(outNames.contains("blk.1.attn_q.weight")); // renumbered from blk.2
        assertTrue(outNames.contains("blk.2.attn_q.weight")); // renumbered from blk.3
        assertFalse(outNames.contains("blk.3.attn_q.weight")); // the pruned layer is gone
        assertTrue(outNames.contains("blk.1.ffn_down.weight")); // was blk.2
        try (GGUFFileParser parser = ModelSources.parseGguf(out.toString())) {
            assertEquals(3L, parser.getUInt32("llama.block_count").longValue());
        }
        // 39 tensors of the input minus the 9 of the pruned layer
        assertEquals(30, outNames.size());
    }

    @Test
    public void keepSplitReproducesTheInputShards() throws IOException {
        Path in = llamaModel();
        Path splitIn = tmp.newFile("split-in.gguf").toPath();
        GgufWriter.Options split = new GgufWriter.Options(GgufWriter.Mode.NATIVE, true, 32, false, 12, 0);
        List<Path> shards;
        try (ModelSource src = ModelSources.open(in.toString())) {
            shards = GgufWriter.write(src, splitIn, split);
        }
        assertEquals(4, shards.size());

        Path out = tmp.getRoot().toPath().resolve("split-out.gguf");
        QuantizeTool.Result r = QuantizeTool.run(shards.get(0), out, QuantizeParams.builder()
                .ftype(QuantizeFtype.Q8_0).keepSplit(true).build());
        assertEquals(4, r.files.size());
        // loading the split through its first shard validates every shard's split.no/count/tensors.count
        Set<String> all = new HashSet<>();
        for (int s = 0; s < r.files.size(); s++) {
            Path shard = r.files.get(s);
            assertTrue(shard.getFileName().toString(),
                    shard.getFileName().toString().endsWith(String.format("-0000%d-of-00004.gguf", s + 1)));
        }
        try (GGUFFileParser parser = ModelSources.parseGguf(r.files.get(0).toString())) {
            assertEquals(4L, parser.getUInt32("split.count").longValue());
            assertEquals(0L, parser.getUInt32("split.no").longValue());
            assertEquals(Integer.valueOf(names(in).size()), parser.getInt("split.tensors.count"));
            assertEquals(4, parser.getShards().size());
            for (GGUFTensorInfo t : parser.getTensors()) {
                all.add(t.name());
            }
        }
        assertEquals(names(in).size(), all.size());
        assertTrue(all.contains("token_embd.weight"));
        assertTrue(all.contains("blk.3.ffn_down.weight"));
        assertEquals(GGMLType.Q8_0, typeOf(r.files.get(0), "token_embd.weight"));
    }

    /** The largest relative RMS error of {@code name}'s decoded rows in {@code out} against {@code in}. */
    private static float relativeRmse(Path in, Path out, String name) throws IOException {
        try (GGUFFileParser a = ModelSources.parseGguf(in.toString());
             GGUFFileParser b = ModelSources.parseGguf(out.toString())) {
            float[] x = a.readTensorDataAsFloat(a.getTensor(name));
            GGUFTensorInfo q = b.getTensor(name);
            byte[] raw = b.readTensorData(q);
            float[] y = CpuBlockDecoders.dequantize(raw, q.type(), x.length);
            double err = 0;
            double sig = 0;
            for (int i = 0; i < x.length; i++) {
                err += (x[i] - y[i]) * (x[i] - y[i]);
                sig += x[i] * x[i];
            }
            return (float) Math.sqrt(err / sig);
        }
    }

    @Test
    public void iq2XxsRecipeNeedsTheImatrixThenQuantizes() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("iq2xxs.gguf");
        try {
            QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.IQ2_XXS).build());
            fail("IQ2_XXS without an imatrix must be refused");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("requires an importance matrix"));
            assertTrue(e.getMessage(), e.getMessage().contains("iq2_xxs"));
        }
        assertFalse(Files.exists(out));
        Path matrix = imatrixFor(in, "iq2xxs-imatrix.gguf", 256);
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.IQ2_XXS)
                .imatrixFile(matrix).build());
        assertEquals(19L, kvValue(out, "general.file_type").longValue());
        assertEquals(GGMLType.IQ2_XXS, typeOf(out, "blk.1.attn_q.weight"));
        assertEquals(GGMLType.IQ2_XXS, typeOf(out, "blk.0.ffn_down.weight"));
        assertEquals(GGMLType.IQ2_XXS, typeOf(out, "blk.2.attn_output.weight"));
        assertEquals(GGMLType.Q2_K, typeOf(out, "blk.0.attn_v.weight")); // n_gqa 2
        assertEquals(GGMLType.Q2_K, typeOf(out, "token_embd.weight"));
        assertEquals(GGMLType.Q5_K, typeOf(out, "output.weight"));
        float e = relativeRmse(in, out, "blk.1.attn_q.weight");
        assertTrue("IQ2_XXS tensor relative RMSE " + e, e < 0.6f);
    }

    @Test
    public void iq2RecipesBumpAttnVOnGroupedQueryModels() throws IOException {
        GgufTestWriter w = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nLayer(4).nEmbd(256).nFf(256).nHead(4).nHeadKv(1).headDim(64).nRot(64).build());
        Path in = tmp.newFile("llama-gqa4.gguf").toPath();
        w.writeTo(in);
        Path out = tmp.getRoot().toPath().resolve("iq2m-gqa4.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.IQ2_M)
                .imatrixFile(imatrixFor(in, "gqa4-imatrix.gguf", 256)).build());
        assertEquals(29L, kvValue(out, "general.file_type").longValue());
        // attention.head_count 4 over head_count_kv 1: n_gqa 4 takes attn_v to Q4_K
        assertEquals(GGMLType.Q4_K, typeOf(out, "blk.0.attn_v.weight"));
        assertEquals(GGMLType.IQ2_S, typeOf(out, "blk.0.attn_q.weight"));
        assertEquals(GGMLType.IQ3_S, typeOf(out, "blk.0.attn_output.weight"));
        assertEquals(GGMLType.IQ3_S, typeOf(out, "token_embd.weight"));
    }

    @Test
    public void iq3SRecipeRunsWithoutAnImatrix() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("iq3s.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.IQ3_S).build());
        assertEquals(26L, kvValue(out, "general.file_type").longValue());
        assertEquals(GGMLType.IQ3_S, typeOf(out, "blk.0.attn_q.weight"));
        assertEquals(GGMLType.IQ3_S, typeOf(out, "blk.3.ffn_down.weight"));
        assertEquals(GGMLType.Q6_K, typeOf(out, "output.weight"));
        float e = relativeRmse(in, out, "blk.3.ffn_down.weight");
        assertTrue("IQ3_S tensor relative RMSE " + e, e < 0.3f);
    }

    @Test
    public void iq3XsAndIq1MRecipesMixTheirTypes() throws IOException {
        Path in = llamaModel();
        Path matrix = imatrixFor(in, "mix-imatrix.gguf", 256);
        Path xs = tmp.getRoot().toPath().resolve("iq3xs.gguf");
        QuantizeTool.run(in, xs, QuantizeParams.builder().ftype(QuantizeFtype.IQ3_XS)
                .imatrixFile(matrix).build());
        assertEquals(22L, kvValue(xs, "general.file_type").longValue());
        assertEquals(GGMLType.IQ3_XXS, typeOf(xs, "blk.0.attn_q.weight"));
        assertEquals(GGMLType.IQ3_XXS, typeOf(xs, "blk.2.ffn_gate.weight"));  // layers [n/8, 7n/8)
        assertEquals(GGMLType.IQ3_S, typeOf(xs, "blk.3.ffn_up.weight"));      // the last layer keeps IQ3_S
        assertEquals(GGMLType.IQ3_S, typeOf(xs, "blk.0.attn_v.weight"));
        Path m = tmp.getRoot().toPath().resolve("iq1m.gguf");
        QuantizeTool.run(in, m, QuantizeParams.builder().ftype(QuantizeFtype.IQ1_M)
                .imatrixFile(matrix).build());
        assertEquals(31L, kvValue(m, "general.file_type").longValue());
        assertEquals(GGMLType.IQ1_M, typeOf(m, "blk.1.ffn_up.weight"));
        assertEquals(GGMLType.IQ2_XXS, typeOf(m, "blk.1.attn_output.weight"));
        float e = relativeRmse(in, m, "blk.1.ffn_up.weight");
        assertTrue("IQ1_M tensor relative RMSE " + e, e < 0.8f);
    }

    @Test
    public void weightlessTargetOfAWeightedQuantizerRefusesByName() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("embd-iq1s.gguf");
        // token_embd is exempt from the imatrix requirement, but the IQ1_S quantizer cannot run without
        // weights; llama.cpp aborts inside ggml_quantize_chunk here
        try {
            QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.Q4_K_M)
                    .tokenEmbeddingType(GGMLType.IQ1_S).build());
            fail("an IQ1_S token embedding without weights must be refused");
        } catch (IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("token_embd.weight"));
            assertTrue(e.getMessage(), e.getMessage().contains("iq1_s"));
            assertTrue(e.getMessage(), e.getMessage().contains("importance weights"));
        }
        assertFalse(Files.exists(out));
    }

    @Test
    public void copyLeavesEveryTypeAndByte() throws IOException {
        Path in = llamaModel();
        Path out = tmp.getRoot().toPath().resolve("copy.gguf");
        QuantizeTool.run(in, out, QuantizeParams.builder().ftype(QuantizeFtype.COPY).build());
        assertEquals(GGMLType.F32, typeOf(out, "blk.0.attn_q.weight"));
        assertEquals(0L, kvValue(out, "general.file_type").longValue()); // a copy keeps the input value
        assertEquals(QuantizeTool.QNT_VERSION, kvValue(out, "general.quantization_version").longValue());
        try (GGUFFileParser a = ModelSources.parseGguf(in.toString());
             GGUFFileParser b = ModelSources.parseGguf(out.toString())) {
            GGUFTensorInfo ta = a.getTensor("blk.2.ffn_gate.weight");
            GGUFTensorInfo tb = b.getTensor("blk.2.ffn_gate.weight");
            assertArrayEquals(a.readTensorDataAsFloat(ta), b.readTensorDataAsFloat(tb), 0.0f);
        }
    }

    private static void writeInt(java.io.DataOutputStream o, int v) throws IOException {
        o.writeByte(v & 0xFF);
        o.writeByte((v >>> 8) & 0xFF);
        o.writeByte((v >>> 16) & 0xFF);
        o.writeByte((v >>> 24) & 0xFF);
    }
}
