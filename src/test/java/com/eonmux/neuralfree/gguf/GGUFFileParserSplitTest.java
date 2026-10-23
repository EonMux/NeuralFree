package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Split models in {@link GGUFFileParser}, checked as llama.cpp src/llama-model-loader.cpp:596-672 checks them: the
 * opened file must be split 0, every split must carry its own {@code split.no}, the tensor total must equal
 * {@code split.tensors.count}, and tensors of later splits are read through the virtual offsets of
 * {@link GGUFFileParser#getShards()}.
 */
public class GGUFFileParserSplitTest {

    static final float[] EMBD = ramp(64 * 2, 0.5f);
    static final float[] NORM = ramp(64, -1f);
    static final float[] Q = ramp(64 * 3, 2f);

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    static float[] ramp(int n, float start) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = start + 0.25f * i;
        }
        return v;
    }

    /** Split {@code no} of a two-split llama model with three tensors in total. */
    static GgufTestWriter split(int no) {
        GgufTestWriter w = new GgufTestWriter()
                .u16(GgufKeys.SPLIT_NO, no)
                .u16(GgufKeys.SPLIT_COUNT, 2)
                .i32(GgufKeys.SPLIT_TENSORS_COUNT, 3);
        if (no == 0) {
            w.architecture("llama")
                    .tensorF32("token_embd.weight", new long[]{64, 2}, EMBD)
                    .tensorF32("output_norm.weight", new long[]{64}, NORM);
        } else {
            w.tensorF32("blk.0.attn_q.weight", new long[]{64, 3}, Q);
        }
        return w;
    }

    static String splitName(File dir, int no) {
        return new File(dir, String.format(Locale.ROOT, "model-%05d-of-00002.gguf", no + 1)).getPath();
    }

    /** Writes both splits into {@code dir} and returns the path of the first. */
    static String writeSplits(File dir, GgufTestWriter first, GgufTestWriter second) throws IOException {
        first.writeTo(new File(splitName(dir, 0)).toPath());
        second.writeTo(new File(splitName(dir, 1)).toPath());
        return splitName(dir, 0);
    }

    @Test
    public void loadsBothSplitsInOneVirtualOffsetSpace() throws IOException {
        File dir = tmp.newFolder();
        GgufTestWriter first = split(0);
        GgufTestWriter second = split(1);
        GGUFFileParser p = new GGUFFileParser(writeSplits(dir, first, second));
        try {
            assertTrue(p.isMultiShard());
            assertEquals(3, p.getTensors().size());
            List<GGUFFileParser.Shard> shards = p.getShards();
            assertEquals(2, shards.size());

            GGUFFileParser.Shard s0 = shards.get(0);
            GGUFFileParser.Shard s1 = shards.get(1);
            assertEquals(0, s0.firstTensor());
            assertEquals(2, s0.tensorCount());
            assertEquals(first.dataSectionOffset(), s0.fileDataOffset());
            assertEquals(2, s1.firstTensor());
            assertEquals(1, s1.tensorCount());
            assertEquals(splitName(dir, 1), s1.path());
            assertEquals(second.dataSectionOffset(), s1.fileDataOffset());
            assertEquals(s0.dataLength(), s1.virtualBase());

            int q = p.getTensorIndex("blk.0.attn_q.weight");
            assertSame(s1, p.getShardOf(q));
            assertSame(s0, p.getShardOf(p.getTensorIndex("output_norm.weight")));
            assertEquals(s1.fileDataOffset(), s1.fileOffset(p.getTensor(q).offset()));

            assertArrayEquals(EMBD, p.readTensorDataAsFloat(p.getTensor("token_embd.weight")), 0f);
            assertArrayEquals(NORM, p.readTensorDataAsFloat(p.getTensor("output_norm.weight")), 0f);
            assertArrayEquals(Q, p.readTensorDataAsFloat(p.getTensor("blk.0.attn_q.weight")), 0f);
            assertEquals(s0.dataLength() + s1.dataLength(), p.getDataLength());
        } finally {
            p.close();
        }
    }

    @Test
    public void rejectsOpeningALaterSplit() throws IOException {
        File dir = tmp.newFolder();
        writeSplits(dir, split(0), split(1));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(splitName(dir, 1)));
        assertTrue(e.getMessage(), e.getMessage().contains("must be loaded with the first split"));
    }

    @Test
    public void rejectsSplitWithoutSplitNo() throws IOException {
        File dir = tmp.newFolder();
        String path = writeSplits(dir, split(0), split(1).remove(GgufKeys.SPLIT_NO));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("key not found in model: " + GgufKeys.SPLIT_NO));
    }

    @Test
    public void rejectsSplitWithWrongSplitNo() throws IOException {
        File dir = tmp.newFolder();
        String path = writeSplits(dir, split(0), split(1).u16(GgufKeys.SPLIT_NO, 0));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("invalid split file idx: 0"));
    }

    @Test
    public void rejectsFirstSplitWithoutSplitNo() throws IOException {
        File dir = tmp.newFolder();
        String path = writeSplits(dir, split(0).remove(GgufKeys.SPLIT_NO), split(1));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("key not found in model: " + GgufKeys.SPLIT_NO));
    }

    @Test
    public void rejectsTensorCountMismatch() throws IOException {
        File dir = tmp.newFolder();
        String path = writeSplits(dir, split(0).i32(GgufKeys.SPLIT_TENSORS_COUNT, 4), split(1));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("corrupted model: 4 tensors expected but 3 found"));
    }

    @Test
    public void rejectsSplitKeyOfWrongType() throws IOException {
        File dir = tmp.newFolder();
        String path = writeSplits(dir, split(0).u32(GgufKeys.SPLIT_COUNT, 2), split(1));
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(path));
        assertTrue(e.getMessage(), e.getMessage().contains("has wrong type UINT32 but expected type UINT16"));
    }

    @Test
    public void rejectsFirstSplitWithWrongFileName() throws IOException {
        File dir = tmp.newFolder();
        File renamed = new File(dir, "model.gguf");
        split(0).writeTo(renamed.toPath());
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(renamed.getPath()));
        assertTrue(e.getMessage(), e.getMessage().contains("invalid split file name"));
    }

    @Test
    public void reportsMissingSplitFile() throws IOException {
        File dir = tmp.newFolder();
        split(0).writeTo(new File(splitName(dir, 0)).toPath());
        IOException e = assertThrows(IOException.class, () -> new GGUFFileParser(splitName(dir, 0)));
        assertTrue(e.getMessage(), e.getMessage().contains("failed to load GGUF split 2 of 2"));
    }

    @Test
    public void singleFileModelIsOneShard() throws IOException {
        File f = tmp.newFile("single.gguf");
        GgufTestWriter w = new GgufTestWriter().architecture("llama").tensorF32("output_norm.weight",
                new long[]{64}, NORM);
        w.writeTo(f.toPath());
        GGUFFileParser p = new GGUFFileParser(f.getPath());
        try {
            assertEquals(1, p.getShardCount());
            GGUFFileParser.Shard s = p.getShards().get(0);
            assertEquals(f.getPath(), s.path());
            assertEquals(p.getDataOffset(), s.fileDataOffset());
            assertEquals(0L, s.virtualBase());
            assertEquals(1, s.tensorCount());
            assertEquals(64L * 4, p.getDataLength());
            assertArrayEquals(NORM, p.readTensorDataAsFloat(p.getTensor("output_norm.weight")), 0f);
        } finally {
            p.close();
        }
    }
}
