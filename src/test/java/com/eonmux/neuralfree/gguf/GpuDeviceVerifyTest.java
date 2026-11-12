package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.eonmux.neuralfree.gguf.SplitFixtures.GREEDY_TOKENS;
import static com.eonmux.neuralfree.gguf.SplitFixtures.PROMPT;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Greedy verification on the GPU (llama.cpp backend sampling {@code -bs}): {@link ForwardBackend#forwardRowsGreedy}
 * of the GPU backend selects the same ids as the host argmax of the read-back rows and leaves the memory where
 * {@code forwardRows} leaves it, and a speculative session over a GPU target emits the plain greedy tokens with the
 * device selection on and off. Skipped when OpenCL is unavailable.
 */
public class GpuDeviceVerifyTest {

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

    @Test
    public void deviceGreedyRowsEqualTheHostArgmaxOfTheRows() throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            ForwardBackend f = m.forwardBackend();
            f.prefill(PROMPT, 0);
            int[] batch = {7, 19, 3, 64, 11};
            int pos = PROMPT.length;
            float[][] rows = f.forwardRows(batch, pos, null);
            int[] hostIds = new int[rows.length];
            for (int i = 0; i < rows.length; i++) {
                hostIds[i] = ForwardBackend.argmax(rows[i]);
            }
            f.rollback(batch.length);

            int[] deviceIds = f.forwardRowsGreedy(batch, pos, null);

            assertArrayEquals(hostIds, deviceIds);
            assertEquals("the call stores the rows like forwardRows", pos + batch.length,
                    f.memory().sequence().nPast());
            f.rollback(batch.length);
            assertArrayEquals("requested rows come back in request order",
                    new int[]{hostIds[4], hostIds[1]}, f.forwardRowsGreedy(batch, pos, new int[]{4, 1}));
            f.reset();
        }
    }

    @Test
    public void speculationKeepsThePlainGreedyTokensWithDeviceSelectionOnAndOff() throws IOException {
        int[] reference;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            reference = SplitFixtures.greedy(m);
        }
        assertArrayEquals(reference, speculativeGreedy(true));
        assertArrayEquals(reference, speculativeGreedy(false));
    }

    /**
     * The device top-k readback of the verification rows and of a decode step equals the host selection over the
     * read-back rows ({@code RowTopK} over {@code CpuOps.topK}), in request order.
     */
    @Test
    public void deviceTopKCandidatesEqualTheHostSelection() throws IOException {
        int k = 8;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            assertTrue(m.forwardBackend().topKOnDevice());
            ForwardBackend f = m.forwardBackend();
            f.prefill(PROMPT, 0);
            int[] batch = {7, 19, 3, 64};
            int pos = PROMPT.length;
            RowTopK host = RowTopK.of(f.forwardRows(batch, pos, null), k);
            f.rollback(batch.length);
            RowTopK device = f.forwardRowsTopK(batch, pos, k, null);
            assertEquals(host.rows(), device.rows());
            assertEquals(host.k(), device.k());
            for (int r = 0; r < host.rows(); r++) {
                assertArrayEquals("row " + r + " ids", host.ids(r), device.ids(r));
                assertArrayEquals("row " + r + " logits", host.logits(r), device.logits(r), 1e-4f);
            }
            f.rollback(batch.length);
            float[] row = f.decode(batch[0], pos);
            RowTopK hostDecode = RowTopK.of(new float[][]{row}, k);
            f.rollback(1);
            RowTopK deviceDecode = f.decodeTopKOnDevice(batch[0], pos, k);
            assertArrayEquals(hostDecode.ids(0), deviceDecode.ids(0));
            assertArrayEquals(hostDecode.logits(0), deviceDecode.logits(0), 1e-4f);
            f.rollback(1);
            f.reset();
        }
    }

    /**
     * Readouts of changing widths over one bound readout, wider than 64 candidates: a narrower width after
     * a wider one packs its own candidates per row (the readback strides by the requested width, not the bound one),
     * and a wider one rebinds; each equals the host selection over the read-back rows.
     */
    @Test
    public void deviceTopKReadoutsOfChangingWidthsEqualTheHostSelection() throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            ForwardBackend f = m.forwardBackend();
            f.prefill(PROMPT, 0);
            int[] batch = {7, 19, 3, 64};
            int pos = PROMPT.length;
            float[][] rows = f.forwardRows(batch, pos, null);
            f.rollback(batch.length);
            for (int k : new int[]{200, 40, 8, 300, RowTopK.MAX_DEVICE_CANDIDATES}) {
                RowTopK host = RowTopK.of(rows, k);
                RowTopK device = f.forwardRowsTopK(batch, pos, k, null);
                f.rollback(batch.length);
                assertEquals("k " + k, host.k(), device.k());
                for (int r = 0; r < host.rows(); r++) {
                    assertArrayEquals("k " + k + " row " + r + " ids", host.ids(r), device.ids(r));
                    assertArrayEquals("k " + k + " row " + r + " logits", host.logits(r), device.logits(r), 1e-4f);
                }
            }
            f.reset();
        }
    }

    /**
     * A seeded temperature chain generates the same tokens with the per-step device sampling on and off: the draw
     * over the device's top candidates is the draw the host chain makes over the whole row.
     */
    @Test
    public void aSeededTemperatureChainGeneratesTheSameTokensOnTheDevice() throws IOException {
        int[] deviceTokens;
        int[] hostTokens;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            deviceTokens = temperature(m);
        }
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            hostTokens = temperature(m);
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
        assertArrayEquals(hostTokens, deviceTokens);
    }

    /**
     * A greedy chain whose only edits are a ban and a non-positive bias generates the same tokens with the device
     * prefix readback on and off: the readback selects the token the whole-row chain selects.
     */
    @Test
    public void aGreedySparseChainGeneratesTheSameTokensOnTheDevice() throws IOException {
        int[] deviceTokens;
        int[] hostTokens;
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            deviceTokens = sparseGreedy(m);
        }
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, oneDevice().build())) {
            hostTokens = sparseGreedy(m);
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
        assertArrayEquals(hostTokens, deviceTokens);
    }

    /** The tokens of a greedy generation with a ban and a negative {@code logit_bias}. */
    private static int[] sparseGreedy(GGUFModelGPU m) {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(SamplerConfig.builder().greedy(true).topK(0).penaltyLastN(0).build())
                .logitBias(5, -100f)
                .logitBias(9, -2f)
                .build();
        return m.generate(PROMPT, request, null).tokens();
    }

    /** The tokens of a seeded temperature generation. */
    private static int[] temperature(GGUFModelGPU m) {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(SamplerConfig.builder().temperature(0.9f).topK(40).topP(0.95f).minP(0.05f).typicalP(1f)
                        .penaltyLastN(0).seed(1234L).build()).build();
        return m.generate(PROMPT, request, null).tokens();
    }

    /** The greedy tokens of a session drafting with the model itself, with device selection {@code onDevice}. */
    private int[] speculativeGreedy(boolean onDevice) throws IOException {
        EngineConfig cfg = oneDevice().draftModelPath(model.toString()).draftMax(3)
                .speculativeOptions(SpeculativeOptions.builder().verifyOnDevice(onDevice).build()).build();
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg);
             SpeculativeSession s = LLMModelFactory.speculate(target, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = s.generate(PROMPT, request, null).tokens();
            SpeculativeDecoder.Stats stats = s.lastStats();
            assertTrue(String.valueOf(stats), stats.rounds() > 0);
            assertTrue("a self-draft is confirmed: " + stats, stats.accepted() > 0);
            return tokens;
        }
    }
}
