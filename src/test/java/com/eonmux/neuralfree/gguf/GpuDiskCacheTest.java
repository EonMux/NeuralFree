package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The disk tier of the prefix cache over a device memory ({@code neuralfree.cache.disk.mib} beside
 * {@code neuralfree.cache.ram}): the whole blocks a one-MiB host tier cannot keep leave the device through the
 * memory's state transfer, sit as files under the directory, and come back into fresh device pages when the prompt
 * that owns them runs again. Small, so it fits the shared GPU.
 */
public class GpuDiskCacheTest {

    private static final int PAGE = 256;
    private static final int N_CTX = 3 * PAGE;
    private static final int N_NEW = 5;

    private static EngineConfig config(Path diskDir) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(PAGE).kvPaged(true)
                        .kvPageSize(PAGE).typeK(KvDType.F32).typeV(KvDType.F32)
                        .cacheRamMib(1).cacheDiskMib(-1).cacheDiskDir(diskDir.toString()).build())
                .build();
    }

    private static Path model() throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nEmbd(512).nHead(8).nHeadKv(8).headDim(64).nRot(64).build()).writeTempFile();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 31 + i * 37) % 250;
        }
        return t;
    }

    private static long countFiles(Path dir) throws IOException {
        try (Stream<Path> files = Files.walk(dir)) {
            return files.filter(f -> f.getFileName().toString().endsWith(".bin")).count();
        }
    }

    @Test
    public void evictedDeviceBlocksComeBackFromDisk() throws Exception {
        Path model = model();
        Path dir = Files.createTempDirectory("nf-disk-gpu-");
        EngineConfig config = config(dir);
        int[] a = prompt(1, PAGE);
        int[] b = prompt(2, 2 * PAGE);
        BatchSequence first;
        BatchSequence again;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, config);
             BatchEngine engine = BatchEngine.forModel(m, config)) {
            first = run(engine, a);
            run(engine, b);
            assertTrue("the blocks the one-MiB host tier cannot keep sit on the disk", countFiles(dir) > 0);
            again = run(engine, a);
        }
        assertEquals("the disk tier serves the whole prompt again (the last token always prefills)", PAGE - 1,
                again.reusedPrefix());
        assertArrayEquals("the promoted rows decode exactly like the first run", first.result().tokens(),
                again.result().tokens());
    }

    private static BatchSequence run(BatchEngine engine, int[] prompt) throws InterruptedException {
        BatchSequence seq = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
        engine.drain();
        assertEquals(BatchSequence.State.DONE, seq.state());
        return seq;
    }
}
