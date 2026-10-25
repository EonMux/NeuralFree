package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.FileTensorRegions;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystemException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import static com.eonmux.neuralfree.io.safetensors.SafetensorsTestFiles.ramp;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A Hugging Face cache snapshot ({@code models--org--name/snapshots/<rev>/<file> -> ../../blobs/<sha>}, huggingface_hub
 * {@code file_download.py}) opens like a plain directory: the links are followed, the index's file names stay inside
 * the snapshot directory although the bytes live in {@code blobs/}, and segments name the link paths.
 */
public class HfSnapshotLayoutTest {

    private static final String REV = "0123456789abcdef0123456789abcdef01234567";
    private static final String SHARD1 = "model-00001-of-00002.safetensors";
    private static final String SHARD2 = "model-00002-of-00002.safetensors";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path repo() {
        return tmp.getRoot().toPath().resolve("models--org--tiny");
    }

    private Path snapshot() {
        return repo().resolve("snapshots").resolve(REV);
    }

    /** Writes a blob and links {@code snapshots/<rev>/<name>} to it with the hub's relative target. */
    private void blob(String name, String sha, byte[] content) throws IOException {
        Path blobs = repo().resolve("blobs");
        Files.createDirectories(blobs);
        Files.createDirectories(snapshot());
        Files.write(blobs.resolve(sha), content);
        link(name, sha);
    }

    private void link(String name, String sha) throws IOException {
        try {
            Files.createSymbolicLink(snapshot().resolve(name), Paths.get("..", "..", "blobs", sha));
        } catch (UnsupportedOperationException | FileSystemException e) {
            Assume.assumeNoException("symbolic links are not available here", e);
        }
    }

    private static byte[] safetensors(Path scratch, String tensor, int first, int len) throws IOException {
        Path f = scratch.resolve(tensor + ".bin");
        SafetensorsWriter.builder().add(tensor, StDType.F32, new long[]{2, len / 8}, ramp(first, len)).build().write(f);
        byte[] bytes = Files.readAllBytes(f);
        Files.delete(f);
        return bytes;
    }

    private void shardedSnapshot() throws IOException {
        Path scratch = tmp.newFolder("scratch").toPath();
        blob(SHARD1, "a1b2c3", safetensors(scratch, "model.embed_tokens.weight", 0, 32));
        blob(SHARD2, "d4e5f6", safetensors(scratch, "lm_head.weight", 100, 16));
        String index = "{\"metadata\":{\"total_size\":48},\"weight_map\":{\"model.embed_tokens.weight\":\"" + SHARD1
                + "\",\"lm_head.weight\":\"" + SHARD2 + "\"}}";
        blob(SafetensorsCheckpoint.INDEX, "0f0f0f", index.getBytes(StandardCharsets.UTF_8));
        blob("config.json", "9e9e9e", "{\"architectures\":[\"LlamaForCausalLM\"]}".getBytes(StandardCharsets.UTF_8));
        Files.createDirectories(repo().resolve("refs"));
        Files.write(repo().resolve("refs").resolve("main"), REV.getBytes(StandardCharsets.US_ASCII));
    }

    @Test
    public void shardedSnapshotOpensThroughItsLinks() throws IOException {
        shardedSnapshot();
        assertTrue(Files.isSymbolicLink(snapshot().resolve(SHARD1)));
        try (CheckpointContainer c = CheckpointContainers.open(snapshot(), snapshot(), false)) {
            HfTensorIndex index = HfTensorIndex.of(c);
            assertEquals(Arrays.asList("model.embed_tokens.weight", "lm_head.weight"), index.names());
            assertEquals(2, c.segments().size());
            DataSegment s2 = c.segments().get(1);
            assertTrue(s2.path(), s2.path().endsWith(Paths.get("snapshots", REV, SHARD2).toString()));
            assertEquals(64, s2.virtualBase());

            RowSource head = index.get("lm_head.weight");
            assertEquals(2, head.rows());
            ByteBuffer dst = ByteBuffer.allocate(16);
            head.readRows(0, 2, dst);
            assertArrayEquals(ramp(100, 16), dst.array());

            try (FileTensorRegions regions = FileTensorRegions.open(c.segments(), TensorReadMode.MMAP, "snapshot")) {
                GGUFTensorInfo embd = new GGUFTensorInfo("token_embd.weight", new long[]{4, 2}, GGMLType.F32,
                        c.directVirtualOffset("model.embed_tokens.weight"));
                ByteBuffer row = regions.region(embd, 16, 16);
                byte[] got = new byte[16];
                row.get(got);
                assertArrayEquals(ramp(16, 16), got);
            }
        }
    }

    @Test
    public void singleFileSnapshotOpensFromTheLinkedFile() throws IOException {
        Path scratch = tmp.newFolder("scratch").toPath();
        blob("model.safetensors", "abcdef", safetensors(scratch, "w", 7, 24));
        Path file = snapshot().resolve("model.safetensors");
        try (CheckpointContainer c = CheckpointContainers.open(file, snapshot(), false)) {
            assertEquals(1, c.tensors().size());
            assertEquals(file.toString(), c.segments().get(0).path());
            ByteBuffer dst = ByteBuffer.allocate(12);
            c.source("w").readRows(1, 1, dst);
            assertArrayEquals(ramp(19, 12), dst.array());
        }
    }

    @Test
    public void danglingLinkInTheIndexIsReportedMissing() throws IOException {
        shardedSnapshot();
        Files.delete(repo().resolve("blobs").resolve("d4e5f6"));
        IOException e = assertThrows(IOException.class,
                () -> SafetensorsCheckpoint.open(snapshot(), snapshot(), false));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Invalid safetensors file '"));
        assertTrue(e.getMessage(), e.getMessage().contains("'" + SHARD2 + "', which does not exist"));
    }
}
