package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The decode replay on a Vulkan device: a synthetic llama decodes sixteen tokens through the captured graph and
 * produces the same logits as the same decode with the replay disabled. The flash decode's key count, split
 * count and query position change every step, so the captured graph replays only by adopting them: the body
 * replays re-record the changed dispatches, and the trusted steps patch them from the plan without running the
 * body, each recorded ahead while the step before it runs. The replay's counters must show exactly that: one
 * capture after the warmup, the verify steps as body replays, and every later step as a prepared patched replay, so
 * a replay that fell back to running the step directly fails here although its logits would match. Three models
 * cover the decode attention kernels: dim-16 heads over an fp32 cache (the split flash decode), and dim-64 heads
 * over an fp16 cache with two query heads per KV head (the grouped-query decode kernel) and with one (the per-head
 * one-row decode kernel), the one-row kernels being what the capture must hold on an NVIDIA device and nowhere
 * else. The
 * engine picks its Vulkan device as usual; {@code -Dneuralfree.test.replay.device=<substring>} narrows that choice
 * for the run as {@code -Dneuralfree.device} does an application's (for instance {@code Intel} on a machine whose
 * first-ranked Vulkan device is another GPU).
 */
public class GpuDecodeReplayVulkanTest {

    private static final int[] TOKENS = {5, 9, 17, 23, 31, 2, 7, 13, 19, 29, 3, 11, 37, 41, 43, 47};
    private static final int N_CTX = 64;
    private static final float TOL = 1e-5f;
    private static final int PCI_VENDOR_NVIDIA = 0x10DE;
    private static final String DEVICE_PROPERTY = "neuralfree.test.replay.device";
    private static final String ENGINE_DEVICE_PROPERTY = "neuralfree.device";

    @Test
    public void replayMatchesDirectDecodeOverAnFp32Cache() throws IOException {
        Decode replayed = compare(SyntheticModels.llama().writeTempFile(), KvDType.F32);
        assertFalse("the dim-16 heads run the split flash decode: " + replayed.kernels,
                replayed.kernels.contains(AttnKernels.FLASH_DECODE_ROW_F16));
    }

    @Test
    public void replayMatchesDirectDecodeOverAnFp16CacheOfGroupedDim64Heads() throws IOException {
        assertOneRowKernel(4, 2, AttnKernels.flashDecodeGqa(KvDType.F16, KvDType.F16));
    }

    @Test
    public void replayMatchesDirectDecodeOverAnFp16CacheOfUngroupedDim64Heads() throws IOException {
        assertOneRowKernel(4, 4, AttnKernels.FLASH_DECODE_ROW_F16);
    }

    /**
     * Decodes a dim-64 llama of {@code nHead} query and {@code nHeadKv} KV heads over an fp16 cache with and without
     * the replay, and checks that the capture holds {@code kernel} in place of the split flash decode on an NVIDIA
     * device and neither one-row kernel elsewhere.
     */
    private static void assertOneRowKernel(int nHead, int nHeadKv, String kernel) throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder()
                .nEmbd(256).nHead(nHead).nHeadKv(nHeadKv).headDim(64).nRot(64).build();
        Decode replayed = compare(SyntheticModels.llama(c).writeTempFile(), KvDType.F16);
        boolean nvidia = replayed.vendorId == PCI_VENDOR_NVIDIA;
        assertEquals(kernel + " serves the step on NVIDIA only: " + replayed.kernels, nvidia,
                replayed.kernels.contains(kernel));
        if (nvidia) {
            assertFalse("the one-row kernel replaces the split flash decode: " + replayed.kernels,
                    replayed.kernels.contains(AttnKernels.flashDecode(KvDType.F16, KvDType.F16)));
        } else {
            assertFalse("no one-row kernel off NVIDIA: " + replayed.kernels,
                    replayed.kernels.contains(AttnKernels.FLASH_DECODE_ROW_F16)
                            || replayed.kernels.contains(AttnKernels.flashDecodeGqa(KvDType.F16, KvDType.F16)));
        }
    }

    private static Decode compare(Path model, KvDType kv) throws IOException {
        try {
            GpuTestUtil.assumeVulkanAvailable();
            Decode replayed = decodeAll(model, kv, "true");
            Decode direct = decodeAll(model, kv, "false");
            for (int step = 0; step < TOKENS.length; step++) {
                assertArrayEquals("logits of decode step " + step + " with the decode replay", direct.logits[step],
                        replayed.logits[step], TOL);
            }
            assertCounters(replayed.counters);
            assertEquals("the replay off captures nothing", 0, direct.counters.captures);
            return replayed;
        } finally {
            Files.deleteIfExists(model);
        }
    }

    /** One capture after the warmup, the verify steps through the body, then prepared patched replays only. */
    private static void assertCounters(GpuDecodeReplay.Counters c) {
        int patched = TOKENS.length - GpuDecodeReplay.DEFAULT_WARMUP_STEPS - 1 - GpuDecodeReplay.VERIFY_STEPS;
        String all = c.toString();
        assertEquals("one capture: " + all, 1, c.captures);
        assertEquals("the verify steps replay the body: " + all, GpuDecodeReplay.VERIFY_STEPS, c.bodyReplays);
        assertEquals("every trusted step is a patched replay: " + all, patched, c.patchedReplays);
        assertEquals("every patched replay prepares its follow-up: " + all, patched, c.preparedSteps);
    }

    private static Decode decodeAll(Path model, KvDType kv, String replay) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        String device = System.getProperty(DEVICE_PROPERTY);
        String previousDevice = System.getProperty(ENGINE_DEVICE_PROPERTY);
        if (device != null) {
            System.setProperty(ENGINE_DEVICE_PROPERTY, device);
        }
        try {
            MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).typeK(kv).typeV(kv).build();
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.VULKAN)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(memory)
                    .build();
            try (EngineModel gpu = LLMModelFactory.open(model.toString(), cfg)) {
                float[][] out = new float[TOKENS.length][];
                for (int i = 0; i < TOKENS.length; i++) {
                    out[i] = gpu.forwardBackend().decode(TOKENS[i], i);
                }
                GpuForwardBackend f = (GpuForwardBackend) gpu.forwardBackend();
                LlmKernels kernels = f.kernels();
                DeviceIdentity id = kernels.caps().getIdentity();
                System.out.println("REPLAY device " + kernels.caps().getDeviceName() + " replay=" + replay);
                return new Decode(out, f.decodeReplayCounters(), f.decodeReplayKernels(),
                        id != null ? id.vendorId() : DeviceIdentity.UNKNOWN_VENDOR);
            }
        } finally {
            System.clearProperty(GpuDecodeReplay.PROPERTY);
            if (device != null) {
                if (previousDevice == null) {
                    System.clearProperty(ENGINE_DEVICE_PROPERTY);
                } else {
                    System.setProperty(ENGINE_DEVICE_PROPERTY, previousDevice);
                }
            }
        }
    }

    /** One decode run: its logits per step, the replay's counters, its captured kernels and the device vendor. */
    private static final class Decode {
        final float[][] logits;
        final GpuDecodeReplay.Counters counters;
        final Set<String> kernels;
        final int vendorId;

        Decode(float[][] logits, GpuDecodeReplay.Counters counters, Set<String> kernels, int vendorId) {
            this.logits = logits;
            this.counters = counters;
            this.kernels = kernels;
            this.vendorId = vendorId;
        }
    }
}
