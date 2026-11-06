package com.eonmux.neuralfree.gguf.hf.torch;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.st.SafetensorsCheckpoint;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

import static com.eonmux.neuralfree.gguf.hf.torch.RestrictedUnpicklerTest.Pb;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link TorchCheckpoint} and its factory: the zip reading and the pickle views of a synthetic {@code .bin}, the
 * offset, stride, contiguity and storage-bound checks, the sharded index, the registry's built-in factory
 * (safetensors win, rule 6 through {@code ModelSources.open}), the malicious-global refusal, deflated storages,
 * and the not-a-torch-zip and absent-pickle refusals. The writer is minimal on purpose: exact central directory,
 * EOCD and pickle bytes as torch writes them.
 */
public class TorchCheckpointTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path dir() {
        return tmp.getRoot().toPath();
    }

    // ------------------------------------------------------------------ the writer

    /** A minimal torch.save zip writer: stored and deflated entries, an honest central directory and EOCD. */
    static final class Bin {
        private final ByteArrayOutputStream body = new ByteArrayOutputStream();
        private final List<Rec> recs = new ArrayList<>();

        static final class Rec {
            final byte[] name;
            final long offset;
            final long crc;
            final int method;
            final int csize;
            final int size;

            Rec(byte[] name, long offset, long crc, int method, int csize, int size) {
                this.name = name;
                this.offset = offset;
                this.crc = crc;
                this.method = method;
                this.csize = csize;
                this.size = size;
            }
        }

        /** Adds {@code name}, its bytes stored as they are. */
        Bin entry(String name, byte[] content) {
            return add(name, content, content, 0);
        }

        /** Adds {@code name}, its bytes deflated. */
        Bin deflate(String name, byte[] content) {
            ByteArrayOutputStream z = new ByteArrayOutputStream();
            try {
                DeflaterOutputStream out = new DeflaterOutputStream(z, new Deflater(Deflater.BEST_SPEED, true));
                out.write(content);
                out.finish();
                out.close();
            } catch (IOException e) {
                throw new AssertionError(e);
            }
            return add(name, z.toByteArray(), content, 8);
        }

        private Bin add(String name, byte[] stream, byte[] raw, int method) {
            CRC32 crc = new CRC32();
            crc.update(raw);
            byte[] encoded = name.getBytes(StandardCharsets.UTF_8);
            long offset = body.size();
            w(l32(0x04034b50));
            w(l16(20));
            w(l16(0));
            w(l16(method));
            w(new byte[4]); // time and date
            w(l32(crc.getValue()));
            w(l32(stream.length));
            w(l32(raw.length));
            w(l16(encoded.length));
            w(l16(0)); // no extra
            w(encoded);
            w(stream.clone());
            recs.add(new Rec(encoded, offset, crc.getValue(), method, stream.length, raw.length));
            return this;
        }

        /** The zip: the local headers and data already written, then the central directory and the EOCD. */
        byte[] build() {
            ByteArrayOutputStream zip = new ByteArrayOutputStream();
            byte[] data = body.toByteArray();
            zip.write(data, 0, data.length);
            long dirStart = data.length;
            for (Rec r : recs) {
                w2(zip, l32(0x02014b50));
                w2(zip, l16(0));
                w2(zip, l16(20));
                w2(zip, l16(0));
                w2(zip, l16(r.method));
                w2(zip, new byte[4]);
                w2(zip, l32(r.crc));
                w2(zip, l32(r.csize));
                w2(zip, l32(r.size));
                w2(zip, l16(r.name.length));
                w2(zip, l16(0));
                w2(zip, l16(0));
                w2(zip, l16(0));
                w2(zip, l16(0));
                w2(zip, l32(0));
                w2(zip, l32(r.offset));
                w2(zip, r.name);
            }
            long dirLength = zip.size() - dirStart;
            w2(zip, l32(0x06054b50));
            w2(zip, l16(0));
            w2(zip, l16(0));
            w2(zip, l16(recs.size()));
            w2(zip, l16(recs.size()));
            w2(zip, l32((int) dirLength));
            w2(zip, l32((int) dirStart));
            w2(zip, l16(0));
            return zip.toByteArray();
        }

        private void w(byte[] b) {
            body.write(b, 0, b.length);
        }

        private static void w2(ByteArrayOutputStream into, byte[] b) {
            into.write(b, 0, b.length);
        }

        private static byte[] l16(int v) {
            return new byte[]{(byte) v, (byte) (v >>> 8)};
        }

        private static byte[] l32(long v) {
            return new byte[]{(byte) v, (byte) (v >>> 8), (byte) (v >>> 16), (byte) (v >>> 24)};
        }
    }

    // ------------------------------------------------------------------ pickle and value helpers

    /** Little-endian float bytes, the storage layout of a torch float checkpoint. */
    private static byte[] f32(float... values) {
        ByteBuffer b = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            b.putFloat(v);
        }
        return b.array();
    }

    /** Writes {@code name} with the bin's bytes. */
    private Path write(String name, byte[] bytes) throws IOException {
        Path p = dir().resolve(name);
        Files.createDirectories(p.getParent());
        Files.write(p, bytes);
        return p;
    }

    /** A state dict pickle of the (tensorName, storageType, key, offset, size[], stride[]) entries in order. */
    private static byte[] statePickle(Object... spec) {
        Pb p = new Pb().proto(2).emptyDict();
        for (int i = 0; i < spec.length; i += 6) {
            p.str((String) spec[i]);
            p.tensorOf((String) spec[i + 1], (String) spec[i + 2], (Long) spec[i + 3],
                    (long[]) spec[i + 4], (long[]) spec[i + 5]);
            p.raw(0x73);
        }
        return p.stop().build();
    }

    /** A state dict pickle over {@code weightMap} (name to shape), each tensor its own stored entry <i>. */
    private static byte[] statePickle(LinkedHashMap<String, long[]> weights) {
        Pb p = new Pb().proto(2).emptyDict();
        int i = 0;
        for (Map.Entry<String, long[]> e : weights.entrySet()) {
            p.str(e.getKey());
            p.tensorOf("FloatStorage", String.valueOf(1000 + i), 0L, e.getValue(), strides(e.getValue()));
            p.raw(0x73);
            i++;
        }
        return p.stop().build();
    }

    /** The row-major strides of {@code shape}. */
    private static long[] strides(long[] shape) {
        long[] s = new long[shape.length];
        long next = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = next;
            next *= shape[i];
        }
        return s;
    }

    /** A synthetic one-layer F32 Llama saved as one {@code pytorch_model.bin} in {@code dir} (rule 6). */
    private void writeLlamaBin(Path dir) throws IOException {
        int dim = 32;
        int heads = 4;
        int kvDim = (dim / heads) * 2;
        int ffn = 2 * dim;
        int vocab = 16;
        LinkedHashMap<String, long[]> w = new LinkedHashMap<>();
        w.put("model.embed_tokens.weight", new long[]{vocab, dim});
        w.put("model.layers.0.input_layernorm.weight", new long[]{dim});
        w.put("model.layers.0.self_attn.q_proj.weight", new long[]{dim, dim});
        w.put("model.layers.0.self_attn.k_proj.weight", new long[]{kvDim, dim});
        w.put("model.layers.0.self_attn.v_proj.weight", new long[]{kvDim, dim});
        w.put("model.layers.0.self_attn.o_proj.weight", new long[]{dim, dim});
        w.put("model.layers.0.post_attention_layernorm.weight", new long[]{dim});
        w.put("model.layers.0.mlp.gate_proj.weight", new long[]{ffn, dim});
        w.put("model.layers.0.mlp.up_proj.weight", new long[]{ffn, dim});
        w.put("model.layers.0.mlp.down_proj.weight", new long[]{dim, ffn});
        w.put("model.norm.weight", new long[]{dim});
        w.put("lm_head.weight", new long[]{vocab, dim});
        Bin b = new Bin().entry("archive/data.pkl", statePickle(w));
        int i = 0;
        for (Map.Entry<String, long[]> e : w.entrySet()) {
            long numel = 1;
            for (long d : e.getValue()) {
                numel *= d;
            }
            b.entry("archive/data/" + (1000 + i), SyntheticHf.f32(SyntheticHf.values(i, (int) numel)));
            i++;
        }
        Files.createDirectories(dir);
        Files.write(dir.resolve("pytorch_model.bin"), b.build());
        Files.write(dir.resolve("config.json"), SyntheticHf.json(
                "{'architectures':['LlamaForCausalLM'],'model_type':'llama','hidden_size':32,'intermediate_size':64,"
                        + "'num_hidden_layers':1,'num_attention_heads':4,'num_key_value_heads':2,"
                        + "'max_position_embeddings':128,'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':16,"
                        + "'torch_dtype':'float32','bos_token_id':0,'eos_token_id':1}").getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("tokenizer.json"),
                SyntheticHf.GPT2_TOKENIZER.getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("tokenizer_config.json"),
                SyntheticHf.GPT2_TOKENIZER_CONFIG.getBytes(StandardCharsets.UTF_8));
    }

    /** A one-tensor torch save zip: {@code name}'s F32 view of the storage {@code key} with {@code bytes}. */
    private static byte[] binFor(String name, String key, byte[] bytes, long[] shape) throws IOException {
        return new Bin()
                .entry("archive/data.pkl", statePickle(name, "FloatStorage", key, 0L, shape, strides(shape)))
                .entry("archive/data/" + key, bytes)
                .build();
    }

    /** The raw bytes of a tensor read whole through its rows. */
    private static byte[] read(com.eonmux.neuralfree.gguf.hf.tensor.RowSource source) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (source.rows() * source.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        source.readRows(0, (int) source.rows(), b);
        return b.array();
    }

    private static List<String> names(CheckpointContainer c) {
        List<String> out = new ArrayList<>();
        for (StTensor t : c.tensors()) {
            out.add(t.name());
        }
        return out;
    }

    // ------------------------------------------------------------------ single file and its checks

    @Test
    public void singleFileOpensWithItsViewsReadAndMapped() throws IOException {
        byte[] w4 = f32(0.5f, -1.0f, 2.0f, 3.5f);
        byte[] b2 = {7, 9};
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data.pkl", statePickle(
                        "w", "FloatStorage", "1001", 0L, new long[]{2, 2}, new long[]{2, 1},
                        "b", "ByteStorage", "1002", 0L, new long[]{2}, new long[]{1}))
                .entry("archive/data/1001", w4)
                .entry("archive/data/1002", b2)
                .build());
        try (TorchCheckpoint c = TorchCheckpoint.open(bin, dir(), false)) {
            assertEquals(Arrays.asList("w", "b"), names(c));
            StTensor w = c.tensor("w");
            assertEquals(StDType.F32, w.dtype());
            assertArrayEquals(new long[]{2, 2}, w.shape());
            assertEquals(16, w.byteLength());
            StTensor b = c.tensor("b");
            assertEquals(StDType.U8, b.dtype());
            assertEquals(2, b.byteLength());
            // The pickle and the two storages are three segments; the weights digest covers them all.
            assertEquals(3, c.segments().size());
            long direct = c.directVirtualOffset("w");
            assertTrue(direct >= 0);
            DataSegment seg = c.segments().get(1);
            assertEquals(seg.virtualBase(), direct - w.begin());
            com.eonmux.neuralfree.gguf.hf.tensor.RowSource source = c.source("w");
            assertEquals(2, source.rows());
            assertEquals(8, source.rowBytes());
            assertArrayEquals(w4, read(source));
            assertNull(c.tensor("missing"));
            assertThrows(IllegalArgumentException.class, () -> c.source("missing"));
            assertTrue(c.description().contains("1 file, 2 tensors"));
            assertTrue(source.describe().contains("'1001'"));
        }
    }

    @Test
    public void nonContiguousViewIsRefused() throws IOException {
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data.pkl", statePickle(
                        "t", "FloatStorage", "s", 0L, new long[]{2, 2}, new long[]{1, 1}))
                .entry("archive/data/s", f32(1, 2, 3, 4))
                .build());
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(bin, dir(), false));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid PyTorch checkpoint file '" + bin + "':"));
        assertTrue(e.getMessage(), e.getMessage().contains("non-contiguous view of storage 's'"));
    }

    @Test
    public void viewPastItsStorageIsRefused() throws IOException {
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data.pkl", statePickle(
                        "t", "FloatStorage", "s", 0L, new long[]{4, 2}, new long[]{2, 1}))
                .entry("archive/data/s", f32(1, 2, 3, 4)) // 16 bytes for a 32 byte view (4 x 2 floats)
                .build());
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(bin, dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("spans bytes [0, 32) of storage 's' of 16 bytes"));
    }

    @Test
    public void sharedStorageIsUsedOnceAndReadAtBothViews() throws IOException {
        byte[] six = f32(1, 2, 3, 4, 5, 6);
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data.pkl", statePickle(
                        "a", "FloatStorage", "700", 0L, new long[]{4}, new long[]{1},
                        "b", "FloatStorage", "700", 4L, new long[]{2}, new long[]{1}))
                .entry("archive/data/700", six)
                .build());
        try (TorchCheckpoint c = TorchCheckpoint.open(bin, dir(), false)) {
            assertEquals(2, c.segments().size()); // the pickle and the one storage
            assertArrayEquals(Arrays.copyOf(six, 16), read(c.source("a")));
            assertArrayEquals(Arrays.copyOfRange(six, 16, 24), read(c.source("b")));
            assertTrue(c.directVirtualOffset("b") > c.directVirtualOffset("a"));
        }
    }

    @Test
    public void deflatedStorageInflatesWholeAndIsNotMapped() throws IOException {
        byte[] w4 = f32(1, 2, 3, 4);
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data.pkl", statePickle(
                        "w", "FloatStorage", "2001", 0L, new long[]{4}, new long[]{1}))
                .deflate("archive/data/2001", w4)
                .build());
        try (TorchCheckpoint c = TorchCheckpoint.open(bin, dir(), false)) {
            assertEquals(1, c.segments().size()); // only the pickle; the deflated storage is not mapped
            assertEquals(-1, c.directVirtualOffset("w"));
            assertArrayEquals(w4, read(c.source("w")));
        }
    }

    // ------------------------------------------------------------------ the sharded index

    @Test
    public void shardedIndexSelectsItsShardsInFileOrderAndIgnoresOthers() throws IOException {
        byte[] a4 = f32(1, 2, 3, 4);
        byte[] b2 = f32(5, 6);
        write("pytorch_model-00002-of-00002.bin", binFor("t1", "s2", b2, new long[]{2}));
        write("pytorch_model-00001-of-00002.bin", binFor("t0", "s1", a4, new long[]{2, 2}));
        write("pytorch_model.stray.bin", binFor("stale", "k", f32(0), new long[]{1}));
        write("pytorch_model.bin.index.json",
                SyntheticHf.json("{'metadata':{'total_size':0},'weight_map':{'t0':'pytorch_model-00001-of-00002.bin',"
                        + "'t1':'pytorch_model-00002-of-00002.bin'}}").getBytes(StandardCharsets.UTF_8));
        try (TorchCheckpoint c = TorchCheckpoint.open(dir(), dir(), false)) {
            assertEquals(Arrays.asList("t0", "t1"), names(c)); // shard 1 before shard 2, the stray never read
            assertTrue(c.description().contains("2 files, 2 tensors"));
            assertArrayEquals(a4, read(c.source("t0")));
            assertArrayEquals(b2, read(c.source("t1")));
            assertTrue(c.directVirtualOffset("t0") >= 0);
        }
    }

    @Test
    public void indexWithoutWeightMapIsRefused() throws IOException {
        write("pytorch_model.bin.index.json", SyntheticHf.json(
                "{'metadata':{'total_size':0}}").getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(dir(), dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("'weight_map' is missing or not an object"));
    }

    @Test
    public void indexNamingAMissingShardIsRefused() throws IOException {
        write("pytorch_model.bin.index.json", SyntheticHf.json(
                "{'metadata':{'total_size':0},'weight_map':{'t':'pytorch_model-00001-of-00002.bin'}}")
                .getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(dir(), dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("which does not exist"));
    }

    // ------------------------------------------------------------------ the factory and rule 6

    @Test
    public void factoryAcceptsThePicklesWithSafetensorsWinning() throws IOException {
        // No files: no factory.
        Path other = tmp.newFolder("thin").toPath();
        assertNull(CheckpointContainers.forFile(other));
        // A directory with the pickle files and no safetensors: the registered factory.
        writeLlamaBin(dir());
        assertSame(TorchCheckpointFactory.FACTORY, CheckpointContainers.forFile(dir()));
        // Safetensors next to the pickle: the registry's first factory wins.
        SafetensorsWriter.builder()
                .add("x", StDType.F32, new long[]{2}, f32(1, 2))
                .build()
                .write(dir().resolve("model.safetensors"));
        assertSame(SafetensorsCheckpoint.FACTORY, CheckpointContainers.forFile(dir()));
    }

    @Test
    public void ruleSixOpensTheSyntheticBinThroughModelSources() throws IOException {
        writeLlamaBin(dir());
        assertSame(TorchCheckpointFactory.FACTORY, CheckpointContainers.forFile(dir()));
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource hf = ModelSources.open(dir().toString(), options, null)) {
            assertTrue(hf instanceof HfModelSource); // rule 6 reached the registered factory
            assertEquals("llama", hf.getString(GgufKeys.GENERAL_ARCHITECTURE));
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void notATorchZipIsRefused() throws IOException {
        Path bin = write("pytorch_model.bin", "not a zip".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(bin, dir(), false));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid PyTorch checkpoint file '" + bin + "':"));
        assertTrue(e.getMessage(), e.getMessage().contains("no end-of-central-directory record"));
    }

    @Test
    public void maliciousGlobalRefusesAtTheOpen() throws IOException {
        Pb p = new Pb().proto(2);
        p.global("os", "system");
        p.str("rm -rf /");
        p.reduce();
        p.stop();
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data.pkl", p.build())
                .entry("archive/data/3001", f32(1))
                .build());
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(bin, dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("the global 'os.system' is not on the allowed list"));
    }

    @Test
    public void absentPickleEntryIsRefused() throws IOException {
        Path bin = write("pytorch_model.bin", new Bin()
                .entry("archive/data/misc", f32(1))
                .build());
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(bin, dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("archive/data.pkl"));
    }

    @Test
    public void directoryWithoutPicklesIsRefused() {
        IOException e = assertThrows(IOException.class, () -> TorchCheckpoint.open(dir(), dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("it holds no PyTorch checkpoint files"));
    }

    @Test
    public void missingPathIsRefused() {
        Path bin = dir().resolve("pytorch_model.bin");
        java.io.FileNotFoundException e =
                assertThrows(java.io.FileNotFoundException.class, () -> TorchCheckpoint.open(bin, dir(), false));
        assertTrue(e.getMessage(), e.getMessage().contains("does not exist"));
    }
}
