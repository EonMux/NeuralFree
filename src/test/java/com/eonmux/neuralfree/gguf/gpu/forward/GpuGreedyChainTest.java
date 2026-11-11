package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeQueue;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.GreedyStep;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Greedy steps chained on the device ({@code GpuForwardBackend.chainGreedy}): each step copies the previous step's
 * greedy id into its own input on the device and is enqueued before that id reaches the host. On every GPU backend
 * that runs here, a synthetic llama decodes the same ids and stores the same sequence chained as with every step
 * waiting for its id, across the decode replay's warmup, capture, verify and trusted steps; a call between chained
 * steps sees their tokens stored; a dropped chained step rolls back like a decode; and the generation loop produces
 * the same run with the chain as with {@code neuralfree.gpu.greedyChain=false}. A long chained run selects the serial
 * ids, and on OpenCL, whose queue recycles its staging buffers as reads complete, it holds no more staging buffers,
 * pending reads and open events than its last two steps use, however many steps ran without a finish.
 */
public class GpuGreedyChainTest {

    private static final int[] PROMPT = {5, 9, 17, 23, 31, 2};
    private static final int STEPS = 24;
    private static final int N_CTX = 64;
    private static final int LONG_STEPS = 1000;
    private static final int LONG_CTX = 1024;
    /** Staging buffers two chained steps hold: each writes its step inputs and reads its id. */
    private static final int MAX_STAGING_IN_FLIGHT = 12;

    private static Path model;
    private static Path longModel;

    @BeforeClass
    public static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        longModel = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nCtxTrain(LONG_CTX).build())
                .writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
        if (longModel != null) {
            Files.deleteIfExists(longModel);
        }
    }

    @Test
    public void chainedStepsDecodeTheSerialIdsOnCuda() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        checkChain(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void chainedStepsDecodeTheSerialIdsOnVulkan() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        checkChain(EngineConfig.BackendMode.VULKAN);
    }

    @Test
    public void chainedStepsDecodeTheSerialIdsOnOpenCl() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        checkChain(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void theGenerationLoopRunsTheSameChainedOnCuda() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        checkLoop(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void theGenerationLoopRunsTheSameChainedOnVulkan() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        checkLoop(EngineConfig.BackendMode.VULKAN);
    }

    @Test
    public void theGenerationLoopRunsTheSameChainedOnOpenCl() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        checkLoop(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void aLongChainedRunSelectsTheSerialIdsOnCuda() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        checkLongRun(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void aLongChainedRunSelectsTheSerialIdsOnVulkan() throws IOException {
        Assume.assumeTrue("Vulkan not available", GpuTestUtil.isVulkanAvailable());
        checkLongRun(EngineConfig.BackendMode.VULKAN);
    }

    @Test
    public void aLongChainedRunKeepsTheOpenClQueueBounded() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        checkLongRun(EngineConfig.BackendMode.OPENCL);
    }

    private static EngineConfig config(EngineConfig.BackendMode backend) {
        return config(backend, N_CTX);
    }

    private static EngineConfig config(EngineConfig.BackendMode backend, int nCtx) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(nCtx).typeK(KvDType.F32).typeV(KvDType.F32).build();
        return EngineConfig.builder().backend(backend).gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(memory).build();
    }

    /**
     * {@link #LONG_STEPS} greedy steps waiting for their ids, then as many chained without any other call: the same
     * ids, every later step chained on the device, and on an OpenCL queue a bounded number of staging buffers and
     * pending reads after every step.
     */
    private static void checkLongRun(EngineConfig.BackendMode backend) throws IOException {
        try (EngineModel gpu = LLMModelFactory.open(longModel.toString(), config(backend, LONG_CTX))) {
            Assume.assumeTrue("the model runs on the GPU engine",
                    gpu.forwardBackend() instanceof GpuForwardBackend);
            GpuForwardBackend f = (GpuForwardBackend) gpu.forwardBackend();
            String device = backend + " on " + f.context().backend().capabilities().getDeviceName();
            int token = argmax(f.prefill(PROMPT, 0));
            int first = token;
            int[] serial = new int[LONG_STEPS];
            for (int i = 0; i < LONG_STEPS; i++) {
                token = f.decodeGreedyOnDevice(token, PROMPT.length + i);
                serial[i] = token;
            }
            f.reset();
            assertEquals(first, argmax(f.prefill(PROMPT, 0)));
            OclComputeQueue ocl = f.context().queue() instanceof OclComputeQueue
                    ? (OclComputeQueue) f.context().queue() : null;
            OclComputeBackend oclBackend = f.context().backend() instanceof OclComputeBackend
                    ? (OclComputeBackend) f.context().backend() : null;
            int eventsBefore = oclBackend != null ? oclBackend.openEvents() : 0;
            long chainedBefore = f.greedyDeviceChained();
            int maxInFlight = 0;
            int maxPending = 0;
            int maxEvents = 0;
            GreedyStep step = f.startGreedy(first, PROMPT.length);
            for (int i = 0; i < LONG_STEPS; i++) {
                GreedyStep next = i + 1 < LONG_STEPS ? f.chainGreedy(step, PROMPT.length + i + 1) : null;
                assertEquals(device + ": step " + i, serial[i], step.id());
                if (ocl != null) {
                    maxInFlight = Math.max(maxInFlight, ocl.stagingBuffersInFlight());
                    maxPending = Math.max(maxPending, ocl.pendingReadCount());
                }
                if (oclBackend != null) {
                    maxEvents = Math.max(maxEvents, oclBackend.openEvents() - eventsBefore);
                }
                step = next;
            }
            assertEquals(device + ": every later step read its id on the device", LONG_STEPS - 1,
                    f.greedyDeviceChained() - chainedBefore);
            if (ocl != null) {
                assertTrue(device + ": staging buffers in flight stay bounded, at most " + maxInFlight,
                        maxInFlight <= MAX_STAGING_IN_FLIGHT);
                assertTrue(device + ": pending reads stay bounded, at most " + maxPending, maxPending <= 2);
            }
            if (oclBackend != null) {
                assertTrue(device + ": open events stay bounded, at most " + maxEvents + " more",
                        maxEvents <= MAX_STAGING_IN_FLIGHT);
            }
        }
    }

    private static void checkChain(EngineConfig.BackendMode backend) throws IOException {
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), config(backend))) {
            Assume.assumeTrue("the model runs on the GPU engine",
                    gpu.forwardBackend() instanceof GpuForwardBackend);
            GpuForwardBackend f = (GpuForwardBackend) gpu.forwardBackend();
            assertTrue(backend + ": the cached attention layers read the memory's step positions",
                    !f.decodeWritesPositions());

            int[] serial = serialIds(f);
            int[] serialStored = f.memory().sequence().tokens();
            assertTrue(backend + ": greedy steps chain on the device", f.chainsGreedyOnDevice());

            f.reset();
            int first = argmax(f.prefill(PROMPT, 0));
            long chainedBefore = f.greedyDeviceChained();
            int[] chained = new int[STEPS];
            GreedyStep step = f.startGreedy(first, PROMPT.length);
            for (int i = 0; i < STEPS; i++) {
                GreedyStep next = i + 1 < STEPS ? f.chainGreedy(step, PROMPT.length + i + 1) : null;
                chained[i] = step.id();
                step = next;
            }
            assertArrayEquals(backend + ": the chained steps select the serial ids", serial, chained);
            assertEquals(backend + ": every later step read its id on the device", STEPS - 1,
                    f.greedyDeviceChained() - chainedBefore);
            assertArrayEquals(backend + ": the chained steps store the serial sequence", serialStored,
                    f.memory().sequence().tokens());

            int pos = PROMPT.length + STEPS;
            GreedyStep last = f.startGreedy(chained[STEPS - 1], pos);
            GreedyStep dropped = f.chainGreedy(last, pos + 1);
            assertEquals("the chained step is stored when the call returns", pos + 2,
                    f.memory().sequence().nPast());
            int lastId = last.id();
            assertEquals("a call between chained steps sees their tokens stored", lastId,
                    f.memory().sequence().token(pos + 1));
            dropped.id();
            assertTrue(f.canRollback(1));
            f.rollback(1);
            assertEquals("the dropped step's position is gone", pos + 1, f.memory().sequence().nPast());
            assertEquals("the decode after the drop selects what the dropped step selected", dropped.id(),
                    f.decodeGreedyOnDevice(lastId, pos + 1));
        }
    }

    /** The prompt, then {@link #STEPS} greedy steps each waiting for its id. */
    private static int[] serialIds(ForwardBackend f) {
        int token = argmax(f.prefill(PROMPT, 0));
        int[] ids = new int[STEPS];
        for (int i = 0; i < STEPS; i++) {
            token = f.decodeGreedyOnDevice(token, PROMPT.length + i);
            ids[i] = token;
        }
        return ids;
    }

    private static void checkLoop(EngineConfig.BackendMode backend) throws IOException {
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(STEPS)
                .sampler(SamplerConfig.builder().greedy(true).penaltyLastN(0).build()).build();
        System.setProperty(GpuForwardBackend.GREEDY_CHAIN_PROPERTY, "false");
        GenerationResult serial;
        int[] serialStored;
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), config(backend))) {
            Assume.assumeTrue("the model runs on the GPU engine",
                    gpu.forwardBackend() instanceof GpuForwardBackend);
            assertTrue("the property keeps the steps serial", !gpu.forwardBackend().chainsGreedyOnDevice());
            serial = GenerationLoop.builder(gpu.forwardBackend(), gpu.tokenizer()).build()
                    .generate(PROMPT, request, null);
            serialStored = gpu.forwardBackend().memory().sequence().tokens();
        } finally {
            System.clearProperty(GpuForwardBackend.GREEDY_CHAIN_PROPERTY);
        }
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), config(backend))) {
            GpuForwardBackend f = (GpuForwardBackend) gpu.forwardBackend();
            GenerationResult chained = GenerationLoop.builder(f, gpu.tokenizer()).build()
                    .generate(PROMPT, request, null);
            assertArrayEquals(backend + ": tokens", serial.tokens(), chained.tokens());
            assertEquals(backend + ": text", serial.text(), chained.text());
            assertEquals(backend + ": stop reason", serial.stopReason(), chained.stopReason());
            assertArrayEquals(backend + ": stored sequence", serialStored, f.memory().sequence().tokens());
            assertTrue(backend + ": the loop chained its steps", serial.tokens().length < 3
                    || f.greedyDeviceChained() > 0);
        }
    }

    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }
}
