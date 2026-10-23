package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchEntry;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** {@link ScheduleResolver}: the trunk/NextN split of llama.cpp's {@code n_layer()} and NextN probe. */
public class ScheduleResolverTest {

    private static ModelHparams hparams(Arch arch, int nLayerAll, int nextn) {
        return new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(nLayerAll, nextn).build();
    }

    private static TensorCatalog catalog(Arch arch, String... names) {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        long offset = 0;
        for (String n : names) {
            infos.add(new GGUFTensorInfo(n, new long[]{4, 4}, GGMLType.F32, offset));
            offset += 64;
        }
        return TensorCatalog.of(arch.ggufName(), infos);
    }

    private static String ehProj(int il) {
        return TensorNames.layerName(il, TensorNames.weight(TensorNames.NEXTN_EH_PROJ));
    }

    @Test
    public void gemma4AssistantBlocksAreAllNextnWithAnEmptyTrunk() {
        ScheduleResolver r = new ScheduleResolver(hparams(Arch.GEMMA4_ASSISTANT, 2, 2),
                catalog(Arch.GEMMA4_ASSISTANT, "blk.0.attn_q.weight", "blk.1.attn_q.weight"));

        assertEquals(0, r.trunkLayers());
        assertEquals(2, r.nextnLayers());
        assertTrue(r.isNextnLayer(0));
        assertTrue(r.isNextnLayer(1));
        assertFalse(r.isNextnLayer(2));
        LayerSchedule s = r.schedule();
        assertSame(LayerSchedule.Kind.LINEAR, s.kind());
        assertEquals(0, s.slots());
    }

    @Test
    public void gemma4AssistantWithTrunkLayersIsRejected() {
        try {
            new ScheduleResolver(hparams(Arch.GEMMA4_ASSISTANT, 2, 0), catalog(Arch.GEMMA4_ASSISTANT));
            fail("a gemma4-assistant file without nextn_predict_layers = block_count must be rejected");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("assistant block"));
        }
    }

    @Test
    public void nextnBlocksArePresentOnlyWithTheirEhProjection() {
        ScheduleResolver with = new ScheduleResolver(hparams(Arch.QWEN35, 3, 1), catalog(Arch.QWEN35, ehProj(2)));
        assertEquals(2, with.trunkLayers());
        assertEquals(1, with.nextnLayers());
        assertEquals(2, with.schedule().slots());

        ScheduleResolver without = new ScheduleResolver(hparams(Arch.QWEN35, 3, 1), catalog(Arch.QWEN35));
        assertEquals(2, without.trunkLayers());
        assertEquals(0, without.nextnLayers());
        assertEquals(2, without.schedule().slots());
    }

    @Test
    public void deepseek4FoldsAbsentNextnBlocksIntoTheTrunkBeforeTheHparamsAreBuilt() {
        ArchEntry plain = ArchEntry.of(ArchRegistry.traits(Arch.DEEPSEEK4), ArchHparamsHook.NONE);

        ModelHparamsBuilder folded = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK4)).layerCounts(3, 1);
        ModelSpecBuilder.withNextnProbe(plain, catalog(Arch.DEEPSEEK4)).hook().apply(folded, null);
        assertEquals(3, folded.nLayerAll());
        assertEquals(0, folded.nextnLayers());
        ScheduleResolver r = new ScheduleResolver(folded.build(), catalog(Arch.DEEPSEEK4));
        assertEquals(3, r.trunkLayers());
        assertEquals(0, r.nextnLayers());
        assertEquals(3, r.schedule().slots());

        ModelHparamsBuilder kept = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK4)).layerCounts(3, 1);
        ModelSpecBuilder.withNextnProbe(plain, catalog(Arch.DEEPSEEK4, ehProj(2))).hook().apply(kept, null);
        assertEquals(1, kept.nextnLayers());
    }

    @Test
    public void absentNextnBlocksOfOtherArchsStayOutOfTheTrunk() {
        ScheduleResolver r = new ScheduleResolver(hparams(Arch.DEEPSEEK2, 3, 1), catalog(Arch.DEEPSEEK2));
        assertEquals(2, r.trunkLayers());
        assertEquals(0, r.nextnLayers());
        assertEquals(2, r.schedule().slots());
    }
}
