package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDevice;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashDrafter;
import com.eonmux.neuralfree.gguf.ops.flow.dflash.DflashSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticDflash;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * {@link LLMModelFactory#draftCompanion} of a {@code dflash} draft: on a device target that drafts on its device the
 * context fit reserves the matrices the device stack reads ({@link DflashDevice#reads}) and its device cache and
 * working buffers ({@link DflashDevice#deviceBytes}); with {@code neuralfree.draft.dflash.device=false} or a cache
 * type the device refuses nothing lands on the device; on a host target the host cache is counted.
 */
public class DflashDraftCompanionTest {

    private static final int N_CTX = 2048;
    private static final long WEIGHTS = 12345L;

    private Path target;
    private Path draft;
    private final List<Path> files = new ArrayList<>();

    @Before
    public void writeModels() throws IOException {
        target = SyntheticModels.llama().writeTempFile();
        files.add(target);
        draft = SyntheticDflash.dflash(SyntheticModels.Config.llama(), SyntheticDflash.Options.of(4, 1, 2))
                .writeTempFile();
        files.add(draft);
    }

    @After
    public void deleteModels() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    @Test
    public void aDeviceTargetReservesTheDeviceStackBesideItsMatrices() throws IOException {
        ModelSpec spec = targetSpec();
        List<Predicate<GGUFTensorInfo>> placed = new ArrayList<>();
        EngineConfig cfg = EngineConfig.builder().draftModelPath(draft.toString()).build();
        ContextFitter.Companion c = LLMModelFactory.draftCompanion(cfg, spec, false, (dp, p) -> {
            placed.add(p);
            return WEIGHTS;
        });
        MemoryPlan plan = MemoryPlanner.planForContext(spec, MemoryConfig.defaults(), N_CTX);
        assertEquals(WEIGHTS + deviceBytes(draft, plan.nCtx(), cfg.memory().draftTypeK()), c.bytes(plan));
        assertEquals(1, placed.size());
        assertTrue(placed.get(0).test(t("fc.weight")));
        assertTrue(placed.get(0).test(t("blk.1.ffn_up.weight")));
        assertFalse(placed.get(0).test(t("blk.1.attn_norm.weight")));
        assertFalse(placed.get(0).test(t("token_embd.weight")));
    }

    @Test
    public void aHostDraftReservesNothingOnTheDeviceAndAQuantizedCacheIsCounted() throws IOException {
        ModelSpec spec = targetSpec();
        EngineConfig off = EngineConfig.builder().draftModelPath(draft.toString())
                .speculativeOptions(SpeculativeOptions.builder().dflashOnDevice(false).build()).build();
        assertSame(ContextFitter.Companion.NONE, LLMModelFactory.draftCompanion(off, spec, false, (dp, p) -> WEIGHTS));
        // the quantized cache needs the wider head size (a multiple of 32), else the device refuses the draft
        Path wide = SyntheticDflash.dflash(SyntheticModels.Config.llama().toBuilder().headDim(32).nRot(32).build(),
                SyntheticDflash.Options.of(4, 1, 2)).writeTempFile();
        files.add(wide);
        EngineConfig quantized = EngineConfig.builder().draftModelPath(wide.toString())
                .memory(MemoryConfig.builder().draftTypeK(KvDType.Q8_0).draftTypeV(KvDType.Q8_0).build()).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, quantized.memory(), N_CTX);
        assertEquals(WEIGHTS + deviceBytes(wide, plan.nCtx(), quantized.memory().draftTypeK()),
                LLMModelFactory.draftCompanion(quantized, spec, false, (dp, p) -> WEIGHTS).bytes(plan));
    }

    @Test
    public void aHostTargetCountsTheHostCache() throws IOException {
        ModelSpec spec = targetSpec();
        EngineConfig cfg = EngineConfig.builder().draftModelPath(draft.toString()).build();
        ContextFitter.Companion c = LLMModelFactory.draftCompanion(cfg, spec, true, (dp, p) -> WEIGHTS);
        MemoryPlan plan = MemoryPlanner.planForContext(spec, MemoryConfig.defaults(), N_CTX);
        DflashSpec dspec;
        try (ModelSource dp = ModelSources.open(draft.toString())) {
            dspec = DflashDrafter.readSpec(dp);
        }
        assertEquals(DflashDrafter.cacheBytes(dspec, plan.nCtx(), cfg.memory().draftTypeK()), c.bytes(plan));
    }

    private ModelSpec targetSpec() throws IOException {
        try (ModelSource src = ModelSources.open(target.toString())) {
            return ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
        }
    }

    private long deviceBytes(Path draftPath, int nCtx, KvDType kvType) throws IOException {
        try (ModelSource dp = ModelSources.open(draftPath.toString())) {
            return DflashDevice.deviceBytes(DflashDrafter.readSpec(dp), TensorCatalog.of(dp), nCtx, kvType,
                    EngineConfig.defaults().draftPMin());
        }
    }

    private static GGUFTensorInfo t(String name) {
        return new GGUFTensorInfo(name, new long[] {1}, GGMLType.F32, 0L);
    }
}
