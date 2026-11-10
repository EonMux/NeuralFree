package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.MemoryPlannerTest;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.ARCH;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.N_EMBD;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.N_VOCAB;
import static com.eonmux.neuralfree.gguf.spec.resolve.NemotronNextnFixture.TRUNK;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link LLMModelFactory#draftCompanion}: the weight bytes the context fit reserves for a draft file are measured over
 * the tensors of the route the attach binds ({@link LLMModelFactory#draftRoute} with the configured draft type and the
 * actual target), so a full file with NextN blocks that drafts as a separate model reserves its trunk, and an MTP
 * draft file of the target's model reserves only what its NextN drafter reads.
 */
public class DraftCompanionPlacementTest {

    private static final String TRUNK_TENSOR = "blk.0.ssm_in.weight";
    private static final String NEXTN_TENSOR = "blk." + TRUNK + ".nextn.eh_proj.weight";

    private static ModelSpec target() {
        return ModelSpecBuilder.build(GgufKv.ofMap(NemotronNextnFixture.metadata()),
                TensorCatalog.of(ARCH, NemotronNextnFixture.tensors()), MemoryConfig.defaults());
    }

    /** A draft file of the fixture family with width {@code nEmbd}, written to a temporary GGUF. */
    private static String draftFile(int nEmbd) throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        for (Map.Entry<String, Object> e : NemotronNextnFixture.metadata(nEmbd, TRUNK, N_VOCAB).entrySet()) {
            put(w, e.getKey(), e.getValue());
        }
        for (GGUFTensorInfo t : NemotronNextnFixture.tensors(nEmbd, TRUNK, N_VOCAB)) {
            w.tensorZeros(t.name(), GGMLType.F32, t.dims());
        }
        Path p = w.writeTempFile();
        return p.toString();
    }

    private static void put(GgufTestWriter w, String key, Object v) {
        if (v instanceof String) {
            w.str(key, (String) v);
        } else if (v instanceof Integer) {
            w.u32(key, (Integer) v);
        } else if (v instanceof int[]) {
            w.i32Array(key, (int[]) v);
        } else if (v instanceof Float) {
            w.f32(key, (Float) v);
        } else if (v instanceof Boolean) {
            w.bool(key, (Boolean) v);
        } else {
            throw new IllegalArgumentException("unexpected fixture value for " + key + ": " + v);
        }
    }

    /** The placement predicates the fit measured, one per measured draft file. */
    private static List<Predicate<GGUFTensorInfo>> measured(String draft, DraftType type) {
        List<Predicate<GGUFTensorInfo>> seen = new ArrayList<>();
        EngineConfig cfg = EngineConfig.builder().draftModelPath(draft).draftType(type).build();
        LLMModelFactory.draftCompanion(cfg, target(), false, (dp, placed) -> {
            seen.add(placed);
            return 0L;
        });
        return seen;
    }

    private static GGUFTensorInfo t(String name) {
        return new GGUFTensorInfo(name, new long[] {1}, GGMLType.F32, 0L);
    }

    @Test
    public void aSiblingWithNextnBlocksDraftingAsASeparateModelReservesItsTrunk() throws IOException {
        List<Predicate<GGUFTensorInfo>> seen = measured(draftFile(2 * N_EMBD), DraftType.AUTO);
        assertEquals(1, seen.size());
        assertTrue(seen.get(0).test(t(TRUNK_TENSOR)));
        assertFalse(seen.get(0).test(t(NEXTN_TENSOR)));
    }

    @Test
    public void theDraftModelCompanionCountsStepTokensAtTheDraftUbatch() throws IOException {
        String draft = draftFile(2 * N_EMBD);
        long perToken = LLMModelFactory.scratchBytesPerToken(draftSpec(2 * N_EMBD));
        ContextFitter.Companion plain = companion(draft, 0);
        ContextFitter.Companion stepped = companion(draft, 64);
        MemoryPlan plan = MemoryPlanner.planForContext(target(), MemoryConfig.defaults(), 2048);
        assertEquals(512, plan.ubatch());
        // the draft's one-group memory plans its addressing at the draft ubatch too
        long addressing = MemoryPlannerTest.contiguousAddressing(1, plan.ubatch())
                - MemoryPlannerTest.contiguousAddressing(1, 64);
        assertEquals("the scratch and the addressing follow the draft ubatch, the caches stay at the plan's",
                perToken * (plan.ubatch() - 64) + addressing, plain.bytes(plan) - stepped.bytes(plan));
    }

    @Test
    public void theTargetsOwnFileReservesWhatItsRouteReads() throws IOException {
        String own = draftFile(N_EMBD);
        List<Predicate<GGUFTensorInfo>> mtp = measured(own, DraftType.AUTO);
        assertEquals(1, mtp.size());
        assertFalse(mtp.get(0).test(t(TRUNK_TENSOR)));
        assertTrue(mtp.get(0).test(t(NEXTN_TENSOR)));

        List<Predicate<GGUFTensorInfo>> simple = measured(own, DraftType.MODEL);
        assertEquals(1, simple.size());
        assertTrue(simple.get(0).test(t(TRUNK_TENSOR)));
        assertFalse(simple.get(0).test(t(NEXTN_TENSOR)));
    }

    /** The companion of a separate-model draft file at the given draft ubatch (0 = the target plan's ubatch). */
    private static ContextFitter.Companion companion(String draft, int draftUbatch) {
        EngineConfig cfg = EngineConfig.builder().draftModelPath(draft).draftType(DraftType.MODEL)
                .memory(MemoryConfig.builder().draftUbatch(draftUbatch).build()).build();
        return LLMModelFactory.draftCompanion(cfg, target(), false, (dp, placed) -> 0L);
    }

    /** The spec of a draft file of the fixture family with width {@code nEmbd}. */
    private static ModelSpec draftSpec(int nEmbd) {
        return ModelSpecBuilder.build(GgufKv.ofMap(NemotronNextnFixture.metadata(nEmbd, TRUNK, N_VOCAB)),
                TensorCatalog.of(ARCH, NemotronNextnFixture.tensors(nEmbd, TRUNK, N_VOCAB)), MemoryConfig.defaults());
    }
}
