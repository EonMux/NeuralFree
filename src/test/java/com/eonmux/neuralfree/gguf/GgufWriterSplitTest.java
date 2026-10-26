package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.GgufWriterTest.TestHfSource;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Split output and {@code --vocab-only} of {@link GgufWriter} (PARITY.md WP TL-1), checked through the
 * readers the acceptance names: a split opens with {@link GGUFFileParser#loadShards()} from the llama.cpp
 * shard names the writer derives ({@code llama_split_path}), every shard carries the {@code split.*} keys
 * that loader reads, the tensors round trip over the virtual offsets, a tensor no shard can hold is a
 * named error that writes nothing, and a vocab-only file is metadata plus tokenizer keys with an empty
 * tensor table that {@link GGUFTokenizer} opens as its source.
 */
public class GgufWriterSplitTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ================================================================== fixtures

    /** One F32 tensor of the little-endian bytes of {@code values}, appended to the source's table. */
    private static void tensor(TestHfSource src, String name, List<Float> values) {
        int[] bytes = new int[4 * values.size()];
        for (int i = 0; i < values.size(); i++) {
            int bits = Float.floatToRawIntBits(values.get(i));
            bytes[4 * i] = bits & 0xFF;
            bytes[4 * i + 1] = (bits >>> 8) & 0xFF;
            bytes[4 * i + 2] = (bits >>> 16) & 0xFF;
            bytes[4 * i + 3] = (bits >>> 24) & 0xFF;
        }
        src.tensor(name, GGMLType.F32, new long[]{values.size()}, bytes);
    }

    /** {@code n} F32 tensor tables, each 10 floats ramping from its index, over the llama metadata. */
    private static TestHfSource sourceOf(int n) {
        TestHfSource src = new TestHfSource("synthetic://split");
        src.kv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, "llama");
        for (int i = 0; i < n; i++) {
            List<Float> v = ramp(i);
            tensor(src, "t" + i, v);
        }
        return src;
    }

    private static List<Float> ramp(int start) {
        List<Float> v = new ArrayList<>(10);
        for (int j = 0; j < 10; j++) {
            v.add(start + 0.25f * j);
        }
        return v;
    }

    /** A {@code llama} tokenizer vocabulary in metadata, as GGUF files carry it. */
    private static void tokenizerKvs(TestHfSource src) {
        src.kv(GgufKeys.TOKENIZER_MODEL, GGUFValueType.STRING, "llama");
        src.kvArray(GgufKeys.TOKENIZER_LIST, GGUFValueType.STRING, "<unk>", "<s>", "</s>");
        src.kvArray(GgufKeys.TOKENIZER_SCORES, GGUFValueType.FLOAT32, 0f, 0f, 0f);
        src.kvArray(GgufKeys.TOKENIZER_TOKEN_TYPE, GGUFValueType.INT32, 2, 3, 3);
        src.kv(GgufKeys.TOKENIZER_BOS_ID, GGUFValueType.UINT32, 1);
        src.kv(GgufKeys.TOKENIZER_EOS_ID, GGUFValueType.UINT32, 2);
        src.kv(GgufKeys.TOKENIZER_UNK_ID, GGUFValueType.UINT32, 0);
    }

    /** The llama.cpp shard name of the export base {@code prefix}. */
    private static String shard(String prefix, int oneBased, int count) {
        return String.format(Locale.ROOT, "%s-%05d-of-%05d.gguf", prefix, oneBased, count);
    }

    /** The key/value section of one shard file, read straight from its header. */
    private static Map<String, Object> shardMetadata(Path shard) throws IOException {
        Map<String, Object> meta = new LinkedHashMap<>();
        try (RandomAccessFile raf = new RandomAccessFile(shard.toFile(), "r")) {
            GGUFHeaderReader reader = new GGUFHeaderReader(shard.toString(), raf.getChannel());
            GGUFHeaderReader.Header header = reader.readHeader();
            for (long i = 0; i < header.kvCount; i++) {
                reader.readKVPair(i, meta, new LinkedHashMap<>(), null);
            }
        }
        return meta;
    }

    private static List<String> namesOf(List<Path> written) {
        List<String> names = new ArrayList<>(written.size());
        for (Path p : written) {
            names.add(p.getFileName().toString());
        }
        return names;
    }

    // ================================================================== split output

    @Test
    public void tensorLimitSplitWritesLlamaShardNamesAndRoundTrips() throws IOException {
        TestHfSource src = sourceOf(5);
        Path out = new File(tmp.getRoot(), "m.gguf").toPath();
        List<Path> written = GgufWriter.write(src, out,
                new GgufWriter.Options(GgufWriter.Mode.NATIVE).splitMaxTensors(2));
        assertEquals(3, written.size());
        assertEquals(Arrays.asList(shard("m", 1, 3), shard("m", 2, 3), shard("m", 3, 3)), namesOf(written));

        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            assertTrue(p.isMultiShard());
            assertEquals(5, p.getTensors().size());
            List<GGUFFileParser.Shard> shards = p.getShards();
            assertEquals(3, shards.size());
            assertEquals(2, shards.get(0).tensorCount());
            assertEquals(2, shards.get(1).firstTensor());
            assertEquals(2, shards.get(1).tensorCount());
            assertEquals(4, shards.get(2).firstTensor());
            assertEquals(1, shards.get(2).tensorCount());

            // the split.* keys of shard 0, read back with llama.cpp's types, over the full metadata
            assertEquals(GGUFValueType.UINT16, p.getValueType(GgufKeys.SPLIT_NO));
            assertEquals(Integer.valueOf(0), p.getInt(GgufKeys.SPLIT_NO));
            assertEquals(GGUFValueType.UINT16, p.getValueType(GgufKeys.SPLIT_COUNT));
            assertEquals(Integer.valueOf(3), p.getInt(GgufKeys.SPLIT_COUNT));
            assertEquals(GGUFValueType.INT32, p.getValueType(GgufKeys.SPLIT_TENSORS_COUNT));
            assertEquals(Integer.valueOf(5), p.getInt(GgufKeys.SPLIT_TENSORS_COUNT));
            assertEquals("llama", p.getString(GgufKeys.GENERAL_ARCHITECTURE));

            for (int i = 0; i < 5; i++) {
                float[] got = p.readTensorDataAsFloat(p.getTensor("t" + i));
                List<Float> want = ramp(i);
                for (int j = 0; j < 10; j++) {
                    assertEquals("tensor t" + i + "[" + j + "]", want.get(j), got[j], 0f);
                }
            }
        }
    }

    @Test
    public void aLaterShardCarriesOnlyTheAlignmentAndTheSplitKeys() throws IOException {
        TestHfSource src = sourceOf(4);
        Path out = new File(tmp.getRoot(), "m.gguf").toPath();
        List<Path> written = GgufWriter.write(src, out,
                new GgufWriter.Options(GgufWriter.Mode.NATIVE).splitMaxTensors(2));

        Map<String, Object> shard2 = shardMetadata(written.get(1));
        assertEquals(4, shard2.size());
        assertTrue(shard2.containsKey(GgufKeys.GENERAL_ALIGNMENT));
        assertEquals(32, shard2.get(GgufKeys.GENERAL_ALIGNMENT));
        assertEquals(1, shard2.get(GgufKeys.SPLIT_NO));
        assertEquals(2, shard2.get(GgufKeys.SPLIT_COUNT));
        assertEquals(4, shard2.get(GgufKeys.SPLIT_TENSORS_COUNT));
        assertFalse(shard2.containsKey(GgufKeys.GENERAL_ARCHITECTURE));
        // and the two tensors of shard 2 still read through the virtual offsets
        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            assertArrayEquals(rampToArray(3), p.readTensorDataAsFloat(p.getTensor("t3")), 0f);
        }
    }

    @Test
    public void byteLimitSplitSplitsBeforeTheTensorThatWouldOverflow() throws IOException {
        // three 40-byte tensors at alignment 32: t1 lands at offset 64 and ends within 128, t2 would
        // overflow the limit, so it alone becomes the second shard
        TestHfSource src = sourceOf(3);
        Path out = new File(tmp.getRoot(), "m.gguf").toPath();
        List<Path> written = GgufWriter.write(src, out,
                new GgufWriter.Options(GgufWriter.Mode.NATIVE).splitMaxBytes(128));
        assertEquals(Arrays.asList(shard("m", 1, 2), shard("m", 2, 2)), namesOf(written));
        try (GGUFFileParser p = new GGUFFileParser(written.get(0).toString())) {
            assertTrue(p.isMultiShard());
            List<GGUFFileParser.Shard> shards = p.getShards();
            assertEquals(2, shards.size());
            assertEquals(2, shards.get(0).tensorCount());
            assertEquals(1, shards.get(1).tensorCount());
            assertEquals(1, shards.get(1).firstTensor());
            assertArrayEquals(rampToArray(2), p.readTensorDataAsFloat(p.getTensor("t2")), 0f);
        }
    }

    @Test
    public void aTensorLargerThanTheByteLimitFailsAndWritesNothing() throws IOException {
        TestHfSource src = sourceOf(2);
        Path out = new File(tmp.getRoot(), "m.gguf").toPath();
        IOException e = writeFails(src, out,
                new GgufWriter.Options(GgufWriter.Mode.NATIVE).splitMaxBytes(32));
        assertTrue(e.getMessage(), e.getMessage().contains("t0"));
        assertTrue(e.getMessage(), e.getMessage().contains("no split can produce a smaller shard"));
        assertFalse(Files.exists(out));
        assertFalse(Files.exists(out.resolveSibling("m-00001-of-00001.gguf")));
        assertFalse(Files.exists(tmp.getRoot().toPath().resolve("m.gguf.tmp")));
        File[] left = tmp.getRoot().listFiles();
        assertTrue(left != null);
        assertEquals(0, left.length);
    }

    // ================================================================== vocab-only

    @Test
    public void vocabOnlyExportsMetadataWithAnEmptyTensorTable() throws IOException {
        TestHfSource src = new TestHfSource("synthetic://full");
        src.kv(GgufKeys.GENERAL_ARCHITECTURE, GGUFValueType.STRING, "llama");
        tokenizerKvs(src);
        tensor(src, "token_embd.weight", ramp(0));
        Path out = new File(tmp.getRoot(), "vocab.gguf").toPath();
        List<Path> written = GgufWriter.write(src, out,
                new GgufWriter.Options(GgufWriter.Mode.NATIVE).vocabOnly(true));
        assertEquals(1, written.size());
        assertEquals(out.getFileName().toString(), written.get(0).getFileName().toString());

        try (GGUFFileParser p = new GGUFFileParser(out.toString())) {
            assertEquals(0, p.getTensors().size());
            assertEquals(0, p.getTensorCount());
            // every metadata key of the source survives, with its type and value
            for (Map.Entry<String, Object> e : src.getMetadata().entrySet()) {
                assertEquals("type of " + e.getKey(), src.getValueType(e.getKey()), p.getValueType(e.getKey()));
                assertEquals("value of " + e.getKey(), e.getValue(), p.getMetadata().get(e.getKey()));
            }
            // it opens as a tokenizer source
            GGUFTokenizer tok = new GGUFTokenizer(p);
            assertEquals("llama", tok.tokenizerModel());
            assertEquals(3, tok.vocabSize());
            assertEquals("<s>", tok.getToken(tok.bosId()));
        }
    }

    @Test
    public void vocabOnlyWithSplitLimitsIsRefusedByTheOptions() {
        assertThrows(IllegalArgumentException.class, () -> new GgufWriter.Options(
                GgufWriter.Mode.NATIVE, false, 32, true, 2, 0L));
        GgufWriter.Options limits = new GgufWriter.Options(
                GgufWriter.Mode.NATIVE, false, 32, false, 2, 1024L);
        assertThrows(IllegalArgumentException.class, () -> limits.vocabOnly(true));
    }

    // ================================================================== helpers

    private static float[] rampToArray(int start) {
        List<Float> v = ramp(start);
        float[] f = new float[v.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = v.get(i);
        }
        return f;
    }

    private static IOException writeFails(ModelSource src, Path out, GgufWriter.Options o) {
        try {
            GgufWriter.write(src, out, o);
            throw new AssertionError("the export was expected to fail");
        } catch (IOException e) {
            return e;
        }
    }
}
