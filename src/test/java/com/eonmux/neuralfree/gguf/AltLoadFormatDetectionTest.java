package com.eonmux.neuralfree.gguf;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's alternative checkpoints are model checkpoints, not refused extensions: {@code sharded_state} and
 * {@code tensorizer} inputs are detected as safetensors checkpoints, and a file that cannot be read is reported with
 * its own format's error instead of an unsupported-format refusal.
 */
public class AltLoadFormatDetectionTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path writes(Path dir, String name, String text) throws IOException {
        Path file = dir.resolve(name);
        Files.createDirectories(file.getParent());
        Files.write(file, text.getBytes(StandardCharsets.UTF_8));
        return file;
    }

    private static Path fixture(String rel) throws URISyntaxException {
        URL url = AltLoadFormatDetectionTest.class.getResource("/vllm/fixtures/" + rel);
        assertNotNull("fixture resource " + rel, url);
        return Paths.get(url.toURI());
    }

    @Test
    public void shardedStateAndTensorizerCheckpointsAreDetected() throws Exception {
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(fixture("llama-tp2/vllm")));
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(fixture("qwen3moe-tp2/tensorizer")));
        assertEquals(ModelFormat.SAFETENSORS,
                ModelSources.detect(fixture("qwen3moe-tp2/tensorizer/model-0000.tensors")));
    }

    @Test
    public void aPreShardedFileIsOpenedAsASafetensorsCheckpoint() throws IOException {
        Path dir = tmp.newFolder().toPath();
        writes(dir, "config.json", "{\"architectures\":[\"LlamaForCausalLM\"]}");
        writes(dir, "model-rank-0-part-0.safetensors", "not a safetensors file");
        IOException e = assertThrows(IOException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.defaults()));
        assertFalse(e.getMessage(), e.getMessage().contains("sharded_state"));
        assertFalse(e.getMessage(), e.getMessage().contains("runai_streamer_sharded"));
        assertTrue(e.getMessage(), e.getMessage().toLowerCase().contains("safetensors"));
    }

    @Test
    public void aTensorizerFileIsOpenedAsATensorizerCheckpoint() throws IOException {
        Path dir = tmp.newFolder().toPath();
        writes(dir, "config.json", "{\"architectures\":[\"LlamaForCausalLM\"]}");
        Path file = writes(dir, "model.tensors", "not a tensorizer file");
        IOException e = assertThrows(IOException.class,
                () -> ModelSources.open(file.toString(), ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("tensorizer"));
        assertTrue(e.getMessage(), e.getMessage().contains("|TZR|"));
    }

    @Test
    public void aTensorizerFileInsideADirectoryIsDetectedAsTheCheckpoint() throws IOException {
        Path dir = tmp.newFolder().toPath();
        writes(dir, "config.json", "{\"architectures\":[\"LlamaForCausalLM\"]}");
        writes(dir, "model.tensors", "not a tensorizer file");
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
        IOException e = assertThrows(IOException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.defaults()));
        assertTrue(e.getMessage(), e.getMessage().contains("tensorizer"));
    }
}
