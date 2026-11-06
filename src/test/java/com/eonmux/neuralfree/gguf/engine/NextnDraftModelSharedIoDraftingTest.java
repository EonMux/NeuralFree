package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticQwen4ExpModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * A qwen4exp MTP-only sidecar without its own {@code token_embd} or LM head loads and drafts the same tokens as
 * sidecars carrying copies of the target's tensors, on the CPU and on the GPU: the drafter binds the missing tensors
 * to the target's ({@code llama_model_share_mtp_io}, qwen4exp.cpp:278-300). The sidecars differ only in which of
 * the two tensors they ship; the target's trunk, the NextN block's weights and the prompt are the same.
 */
public class NextnDraftModelSharedIoDraftingTest {

    /** A prompt past the first positions, so a draft has trunk rows to read. */
    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};

    @Test
    public void aSharedSidecarDraftsWhatCopiesOfTheTargetsIoTensorsDraft() throws IOException {
        assertParity(cpuConfig());
    }

    @Test
    public void theGpuDraftsTheSameTokensFromASharedSidecar() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertParity(gpuConfig());
    }

    @Test
    public void theSyntheticSharedSidecarIsMtpOnlyWithoutItsOwnIo() throws IOException {
        Path target = SyntheticQwen4ExpModels.qwen4expMtp().writeTempFile();
        Path shared = SyntheticQwen4ExpModels.qwen4expMtpSidecar(false).writeTempFile();
        Path copies = SyntheticQwen4ExpModels.qwen4expMtpSidecar(true).writeTempFile();
        try (GGUFModel model = new GGUFModel(target.toString(), cpuConfig());
             ModelSource sharedSource = ModelSources.open(shared.toString());
             ModelSource copiesSource = ModelSources.open(copies.toString())) {
            assertTrue(ModelSpecBuilder.isMtpOnly(GgufKv.of(sharedSource), TensorCatalog.of(sharedSource)));
            ModelSpec sharedSpec = ModelSpecBuilder.buildMtpDraft(GgufKv.of(sharedSource),
                    TensorCatalog.of(sharedSource), MemoryConfig.defaults(), model.spec());
            assertNull(sharedSpec.embedding().tokenEmbd());
            assertNull(sharedSpec.head().output());
            ModelSpec copiesSpec = ModelSpecBuilder.buildMtpDraft(GgufKv.of(copiesSource),
                    TensorCatalog.of(copiesSource), MemoryConfig.defaults(), model.spec());
            assertNotNull(copiesSpec.embedding().tokenEmbd());
            assertNotNull(copiesSpec.head().output());
        }
    }

    /**
     * One prompt pass and the tokens drafted after it over four sidecars that differ only in the IO tensors they
     * ship. A qwen4exp trunk keeps recurrent state, so the targets verify no draft here: every sidecar sees the same
     * rows of the same prompt and proposes the same tokens.
     */
    private static void assertParity(EngineConfig cfg) throws IOException {
        Path target = SyntheticQwen4ExpModels.qwen4expMtp().writeTempFile();
        int[] shared = draft(cfg, target, SyntheticQwen4ExpModels.qwen4expMtpSidecar(false).writeTempFile());
        int[] embeddingOnly =
                draft(cfg, target, SyntheticQwen4ExpModels.qwen4expMtpSidecar(true, false).writeTempFile());
        int[] headOnly = draft(cfg, target, SyntheticQwen4ExpModels.qwen4expMtpSidecar(false, true).writeTempFile());
        int[] copies = draft(cfg, target, SyntheticQwen4ExpModels.qwen4expMtpSidecar(true, true).writeTempFile());
        assertTrue("the target premise: the drafter proposes tokens", shared.length > 0);
        assertArrayEquals(shared, embeddingOnly);
        assertArrayEquals(shared, headOnly);
        assertArrayEquals(shared, copies);
    }

    /**
     * One prompt pass of the target and the tokens its drafter then proposes, through the factory path that loads
     * the sidecar as the target's MTP drafter.
     */
    private static int[] draft(EngineConfig cfg, Path target, Path sidecar) throws IOException {
        try (SpeculativeSession session =
                     LLMModelFactory.openSpeculative(target.toString(), sidecar.toString(), cfg)) {
            assertTrue(session.drafterKind(), session.drafterKind().contains("mtp"));
            EngineModel model = session.target();
            model.forwardBackend().forwardRows(PROMPT, 0, null);
            return session.drafter().draft(PROMPT[PROMPT.length - 1], 3);
        }
    }

    private static EngineConfig cpuConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static EngineConfig gpuConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory())
                .build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }
}
