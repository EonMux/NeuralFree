package com.eonmux.neuralfree.gguf.remote;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ObjectStoreCheckpoints} end to end over a fake S3 bucket: the mirror's files, its reuse across opens, the
 * re-download of changed objects, the removal of stale ones, the key allow-list and path safety, and a whole
 * synthetic Llama checkpoint opened through {@link ModelSources#open} from {@code s3://}.
 */
public class ObjectStoreCheckpointsTest {

    private static final String URI = "s3://bucket/models/tiny";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private FakeObjectStoreServer s3;
    private ModelSourceOptions options;

    @Before
    public void start() throws IOException {
        s3 = new FakeObjectStoreServer(FakeObjectStoreServer.Kind.S3);
        System.setProperty(ObjectStores.PROP_S3_ENDPOINT, s3.base());
        System.setProperty(ObjectStores.PROP_S3_REGION, "us-east-1");
        options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath())
                .remoteConcurrency(4).build();
    }

    @After
    public void stop() {
        System.clearProperty(ObjectStores.PROP_S3_ENDPOINT);
        System.clearProperty(ObjectStores.PROP_S3_REGION);
        s3.close();
    }

    private Path checkpoint() throws IOException {
        return SyntheticHf.llama(1, 32, 4, 2, 8).write(tmp.newFolder().toPath());
    }

    private void upload(Path dir, String prefix) throws IOException {
        try (Stream<Path> files = Files.walk(dir)) {
            for (Path file : (Iterable<Path>) files.filter(Files::isRegularFile)::iterator) {
                String rel = dir.relativize(file).toString().replace('\\', '/');
                s3.put(prefix + rel, Files.readAllBytes(file));
            }
        }
    }

    @Test
    public void mirrorsTheCheckpointAndReusesIt() throws IOException {
        upload(checkpoint(), "models/tiny/");
        Path local = ObjectStoreCheckpoints.localize(URI, options);
        assertTrue(Files.isRegularFile(local.resolve("config.json")));
        assertTrue(Files.isRegularFile(local.resolve("model.safetensors")));
        assertTrue(Files.isRegularFile(local.resolve("tokenizer.json")));
        assertTrue(Files.isRegularFile(local.resolve(ObjectStoreCheckpoints.MANIFEST)));

        int gets = s3.objectGets();
        assertEquals("a second open downloads nothing", local, ObjectStoreCheckpoints.localize(URI, options));
        assertEquals(gets, s3.objectGets());
    }

    @Test
    public void downloadsOnlyChangedObjectsAndRemovesStaleOnes() throws IOException {
        upload(checkpoint(), "models/tiny/");
        Path local = ObjectStoreCheckpoints.localize(URI, options);
        assertTrue(Files.isRegularFile(local.resolve("tokenizer_config.json")));

        byte[] changed = ("{\"architectures\":[\"LlamaForCausalLM\"],\"model_type\":\"llama\",\"hidden_size\":32,"
                + "\"intermediate_size\":64,\"num_hidden_layers\":1,\"num_attention_heads\":4,"
                + "\"num_key_value_heads\":2,\"max_position_embeddings\":128,\"rms_norm_eps\":1e-05,"
                + "\"rope_theta\":10000.0,\"vocab_size\":8,\"bos_token_id\":0,\"eos_token_id\":1}")
                .getBytes(StandardCharsets.UTF_8);
        s3.put("models/tiny/config.json", changed);
        s3.remove("models/tiny/tokenizer_config.json");
        int before = s3.objectGets();
        ObjectStoreCheckpoints.localize(URI, options);
        assertEquals("only the changed object is downloaded", before + 1, s3.objectGets());
        assertArrayEquals(changed, Files.readAllBytes(local.resolve("config.json")));
        assertFalse("a stale file is removed", Files.exists(local.resolve("tokenizer_config.json")));
    }

    @Test
    public void keepsModelFilesOnly() throws IOException {
        upload(checkpoint(), "models/tiny/");
        s3.put("models/tiny/README.md", "readme".getBytes(StandardCharsets.UTF_8));
        s3.put("models/tiny/original/config.json", "{}".getBytes(StandardCharsets.UTF_8));
        s3.put("models/tiny/.git/HEAD", "ref".getBytes(StandardCharsets.UTF_8));
        s3.put("models/tiny/logo.png", new byte[]{1, 2, 3});
        Path local = ObjectStoreCheckpoints.localize(URI, options);
        assertFalse(Files.exists(local.resolve("README.md")));
        assertFalse(Files.exists(local.resolve("original")));
        assertFalse(Files.exists(local.resolve(".git")));
        assertFalse(Files.exists(local.resolve("logo.png")));
    }

    @Test
    public void aSingleObjectUriResolvesToTheFile() throws IOException {
        s3.put("models/solo/model.gguf", new byte[]{1, 2, 3, 4});
        Path local = ObjectStoreCheckpoints.localize("s3://bucket/models/solo/model.gguf", options);
        assertTrue(Files.isRegularFile(local));
        assertEquals("model.gguf", local.getFileName().toString());
    }

    @Test
    public void refusesAKeyOutsideTheMirror() {
        s3.put("models/evil/../escape.json", "{}".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class,
                () -> ObjectStoreCheckpoints.localize("s3://bucket/models/evil", options));
        assertTrue(e.getMessage(), e.getMessage().contains("models/evil/../escape.json"));
    }

    @Test
    public void refusesALocationWithoutModelFiles() {
        s3.put("models/empty/README.md", "readme".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class,
                () -> ObjectStoreCheckpoints.localize("s3://bucket/models/empty", options));
        assertTrue(e.getMessage(), e.getMessage().contains("no model files"));
    }

    @Test
    public void localizesATensorizerObject() throws IOException {
        s3.put("models/tensor/model.tensors", new byte[]{1});
        Path file = ObjectStoreCheckpoints.localize("s3://bucket/models/tensor", options);
        assertTrue(file.toString(), java.nio.file.Files.isRegularFile(file));
        assertEquals("model.tensors", file.getFileName().toString());
    }

    @Test
    public void requireOpenableAcceptsRemoteUrisAndRefusesMalformedOnes() throws IOException {
        ModelSources.requireOpenable(URI);
        assertThrows(FileNotFoundException.class, () -> ModelSources.requireOpenable("s3://"));
    }

    @Test
    public void opensASyntheticLlamaFromS3() throws IOException {
        upload(checkpoint(), "models/tiny/");
        try (ModelSource opened = ModelSources.open(URI, options)) {
            assertEquals(ModelFormat.SAFETENSORS, opened.format());
            HfModelSource source = (HfModelSource) opened;
            assertEquals("llama", source.getString("general.architecture"));
            assertEquals(3 + 9, source.getTensorCount());
            try (WeightStore store = source.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
                for (GGUFTensorInfo tensor : source.getTensors()) {
                    assertArrayEquals(tensor.name(), produce(source.plan().tensor(tensor.name()).source()),
                            read(store, tensor));
                }
            }
        }
    }

    private static byte[] produce(RowSource source) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate((int) (source.rows() * source.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        source.readRows(0, (int) source.rows(), buffer);
        return buffer.array();
    }

    private static byte[] read(WeightStore store, GGUFTensorInfo tensor) throws IOException {
        byte[] out = new byte[(int) tensor.sizeInBytes()];
        store.readTensor(tensor.offset(), out.length, out);
        return out;
    }
}
