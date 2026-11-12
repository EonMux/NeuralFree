package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * When a persistent pipeline cache writes its file: only after a pipeline was created through it, and only when the
 * driver's cache data differs in size from what the file holds, so a run whose pipelines all came from the loaded data
 * leaves the file alone.
 */
public class VkPipelineCachePersistenceTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private VkComputeBackend backend;

    @Before
    public void openBackend() {
        try {
            backend = new VkComputeBackend();
        } catch (Throwable e) {
            Assume.assumeNoException("no Vulkan GPU", e);
        }
    }

    @After
    public void closeBackend() {
        if (backend != null) {
            backend.close();
        }
    }

    private VkPipelineCache cache(Path file) {
        return new VkPipelineCache(backend.device(), backend.info(), file);
    }

    @Test
    public void aCacheWithoutNewPipelinesWritesNothing() {
        Path file = tmp.getRoot().toPath().resolve("cache.bin");
        try (VkPipelineCache cache = cache(file)) {
            assertFalse(cache.isDirty());
            cache.save();
        }
        assertFalse(Files.exists(file));
    }

    @Test
    public void aCreatedPipelineIsSavedOnceAndAReloadThatAddsNothingIsNotRewritten() throws IOException {
        Path file = tmp.getRoot().toPath().resolve("cache.bin");
        byte[] written;
        try (VkPipelineCache cache = cache(file)) {
            cache.pipelineCreated();
            assertTrue(cache.isDirty());
            cache.save();
            assertFalse(cache.isDirty());
            assertTrue(Files.exists(file));
            written = Files.readAllBytes(file);
            Files.delete(file);
            cache.pipelineCreated();
            cache.save();
            assertFalse("a save that adds nothing keeps the file as it was", Files.exists(file));
        }
        Files.write(file, written);
        try (VkPipelineCache reloaded = cache(file)) {
            Files.delete(file);
            reloaded.pipelineCreated();
        }
        assertFalse("closing a reload whose data did not grow rewrites nothing", Files.exists(file));
    }
}
