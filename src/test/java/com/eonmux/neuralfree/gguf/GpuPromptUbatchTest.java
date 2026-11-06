package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.forward.ExecutionObserver;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * The steps a real device runs a prompt in: a prompt that does not fill its ubatch splits into as few steps of
 * equal size as the ubatch allows, on the contiguous call ({@code ForwardBackend.prefill}) and on the paged batch
 * path ({@code ForwardBackend.forwardBatch} over one sequence), so no short step trails it. The device's execution
 * observer reports every step's position and row count. Skipped when OpenCL is unavailable.
 */
public class GpuPromptUbatchTest {

    /** 20 rows at a ubatch of 8: 3 steps of 7, 7 and 6. */
    private static final int ROWS = 20;
    private static final int UBATCH = 8;

    private Path model;

    @Before
    public void writeModel() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
    }

    @After
    public void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The step sizes one call runs, as the device's observer reports them. */
    private static List<Integer> stepSizes(GGUFModelGPU m, Runnable call) {
        Recording recording = new Recording();
        m.observerSlot(0).use(recording);
        try {
            call.run();
        } finally {
            m.observerSlot(0).restore();
        }
        return recording.steps;
    }

    @Test
    public void aContiguousPromptSplitsIntoEqualSteps() throws IOException {
        EngineConfig cfg = config().build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            ForwardBackend f = m.forwardBackend();
            List<Integer> sizes = stepSizes(m, () -> f.prefill(tokens(1, ROWS), 0));
            assertEquals(Arrays.asList(7, 7, 6), sizes);
        }
    }

    @Test
    public void aPagedPromptBatchSplitsIntoEqualUbatchSizes() throws IOException {
        EngineConfig cfg = config().memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(UBATCH)
                .parallel(2).build()).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            ForwardBackend f = m.forwardBackend();
            ForwardBatch batch = rows(tokens(2, ROWS), 0);
            List<Integer> sizes = stepSizes(m, () -> f.forwardBatch(batch));
            assertEquals(Arrays.asList(7, 7, 6), sizes);
        }
    }

    private static EngineConfig.Builder config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(UBATCH).build());
    }

    /** One sequence's rows at positions {@code pos0 ..}, the last one an output row. */
    private static ForwardBatch rows(int[] tokens, int pos0) {
        ForwardBatch.Builder b = ForwardBatch.builder(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            b.add(tokens[i], pos0 + i, 0, i == tokens.length - 1);
        }
        return b.build();
    }

    /** An enabled observer that records the row count of every step the device starts. */
    private static final class Recording implements ExecutionObserver {
        final List<Integer> steps = new ArrayList<>();

        @Override
        public boolean enabled() {
            return true;
        }

        @Override
        public void onStepStart(int pos0, int nTokens) {
            steps.add(nTokens);
        }
    }
}
