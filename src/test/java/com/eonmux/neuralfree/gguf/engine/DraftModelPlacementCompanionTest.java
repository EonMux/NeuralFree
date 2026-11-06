package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The draft model's placement settings reach the context fit ({@link LLMModelFactory#draftCompanion}): a device
 * list that opens the draft off the target's devices ({@code -devd}) reserves nothing beside the target's context,
 * and the draft load's layer count, tensor overrides and expert placement ({@code -ngld}, {@code -otd},
 * {@code -cmoed}, {@code -ncmoed}) cut the measured weights as the draft's own GPU load cuts them. A draft file
 * that drafts by another route refuses the placement settings naming them.
 */
public class DraftModelPlacementCompanionTest {

    private static final String TRUNK_TENSOR = "blk.0.ssm_in.weight";
    private static final String SECOND_LAYER_TENSOR = "blk.1.attn_q.weight";
    private static final String LAST_LAYER_TENSOR = "blk.2.ffn_gate_inp.weight";
    private static final String EXPERT_TENSOR = "blk.2.ffn_up_exps.weight";
    private static final String NEXTN_TENSOR = "blk." + TRUNK + ".nextn.eh_proj.weight";

    private static ModelSpec target() {
        return ModelSpecBuilder.build(GgufKv.ofMap(NemotronNextnFixture.metadata()),
                TensorCatalog.of(ARCH, NemotronNextnFixture.tensors()), MemoryConfig.defaults());
    }

    /** A draft file of the fixture family of its own width, written to a temporary GGUF. */
    private static String draftFile() throws IOException {
        GgufTestWriter w = new GgufTestWriter();
        for (Map.Entry<String, Object> e : NemotronNextnFixture.metadata(2 * N_EMBD, TRUNK, N_VOCAB).entrySet()) {
            put(w, e.getKey(), e.getValue());
        }
        for (GGUFTensorInfo t : NemotronNextnFixture.tensors(2 * N_EMBD, TRUNK, N_VOCAB)) {
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

    /** The placement predicate the fit measured for the configured draft file. */
    private static Predicate<GGUFTensorInfo> measured(EngineConfig cfg) throws IOException {
        List<Predicate<GGUFTensorInfo>> seen = new ArrayList<>();
        LLMModelFactory.draftCompanion(cfg, target(), false, (dp, placed) -> {
            seen.add(placed);
            return 0L;
        });
        assertFalse("the companion measured the draft file", seen.isEmpty());
        return seen.get(0);
    }

    private static EngineConfig.Builder draft(String file) {
        return EngineConfig.builder().draftModelPath(file).draftType(DraftType.MODEL);
    }

    private static GGUFTensorInfo t(String name) {
        return new GGUFTensorInfo(name, new long[] {1}, GGMLType.F32, 0L);
    }

    @Test
    public void aDraftLayerCountCutsTheMeasuredLoadLikeItsGpuLoad() throws IOException {
        Predicate<GGUFTensorInfo> placed = measured(draft(draftFile()).draftGpuLayers(2).build());
        assertFalse("blk.0 stays on the host", placed.test(t(TRUNK_TENSOR)));
        assertFalse("blk.1 stays on the host", placed.test(t(SECOND_LAYER_TENSOR)));
        assertTrue("blk.2 stays on the device", placed.test(t(LAST_LAYER_TENSOR)));
        assertTrue(placed.test(t("token_embd.weight")));
        assertFalse("the NextN blocks stay off the draft's device", placed.test(t(NEXTN_TENSOR)));
    }

    @Test
    public void aHostOverridePullsTheMeasuredLayersPrefixToTheHost() throws IOException {
        Predicate<GGUFTensorInfo> placed = measured(draft(draftFile())
                .addDraftTensorOverride("blk\\.1\\..*", "cpu").build());
        assertFalse(placed.test(t(TRUNK_TENSOR)));
        assertFalse(placed.test(t(SECOND_LAYER_TENSOR)));
        assertTrue(placed.test(t(LAST_LAYER_TENSOR)));
    }

    @Test
    public void cpuMoeKeepsTheMeasuredExpertStacksOnTheHostWithoutPullingTheirLayers() throws IOException {
        Predicate<GGUFTensorInfo> placed = measured(draft(draftFile()).draftCpuMoe(true).build());
        assertFalse("the expert stack of blk.2 stays in RAM", placed.test(t(EXPERT_TENSOR)));
        assertTrue("its layer stays on the device", placed.test(t(LAST_LAYER_TENSOR)));
        assertTrue("an expert setting pulls no layer", placed.test(t(TRUNK_TENSOR)));
    }

    @Test
    public void aLayerCountThatCoversEverythingMeasuresTheDefaultPlacement() throws IOException {
        Predicate<GGUFTensorInfo> placed = measured(draft(draftFile()).draftGpuLayers(TRUNK + 1).build());
        assertTrue(placed.test(t(TRUNK_TENSOR)));
        assertTrue(placed.test(t(EXPERT_TENSOR)));
        assertFalse(placed.test(t(NEXTN_TENSOR)));
    }

    @Test
    public void aDraftOnItsOwnDevicesReservesNothingBesideTheTargetsContext() throws IOException {
        for (List<String> devices : List.of(List.of(EngineConfig.DRAFT_NO_DEVICE), List.of("0"), List.of("0", "1"))) {
            ContextFitter.Companion c = LLMModelFactory.draftCompanion(
                    draft(draftFile()).draftDevices(devices).build(), target(), false, (dp, placed) -> 0L);
            assertSame(devices + " open the draft on devices of its own", ContextFitter.Companion.NONE, c);
        }
    }

    @Test
    public void anUnusableDraftLayerCountIsTheLoadTimeRefusal() throws IOException {
        ContextFitter.Companion c = LLMModelFactory.draftCompanion(
                draft(draftFile()).draftGpuLayers(0).build(), target(), false, (dp, placed) -> 0L);
        assertSame("the companion measures nothing; the load refuses neuralfree.draft.n.gpu.layers=0",
                ContextFitter.Companion.NONE, c);
    }

    @Test
    public void anotherRouteRefusesThePlacementSettingsNamingThem() {
        EngineConfig cfg = EngineConfig.builder().draftModelPath("own.gguf").draftType(DraftType.AUTO)
                .draftDevices(List.of("none")).draftNCpuMoe(1).build();
        EngineConfig plain = EngineConfig.builder().draftModelPath("own.gguf").build();
        for (LLMModelFactory.DraftRoute route : List.of(LLMModelFactory.DraftRoute.MTP_FILE,
                LLMModelFactory.DraftRoute.HEAD, LLMModelFactory.DraftRoute.DFLASH)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> LLMModelFactory.refuseDraftPlacement(route, Arch.NEMOTRON_H_MOE, "own.gguf", cfg));
            assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_DRAFT_GPU_DEVICES));
            assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_DRAFT_N_CPU_MOE));
            assertTrue(e.getMessage(), e.getMessage().contains("draft-simple"));
            LLMModelFactory.refuseDraftPlacement(route, Arch.NEMOTRON_H_MOE, "own.gguf", plain);
        }
        LLMModelFactory.refuseDraftPlacement(LLMModelFactory.DraftRoute.MODEL, Arch.NEMOTRON_H_MOE, "own.gguf", cfg);
    }
}
