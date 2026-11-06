package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.PlacementFitter.Fit;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The placement ladder of {@link PlacementFitter} on the synthetic llama (dense) and qwen3next (MoE) fixtures: a
 * budget that holds every weight keeps them resident; a MoE model streams its expert stacks first and brings the
 * experts of the leading layers back while they fit; a dense model streams the smallest prefix of whole layers that
 * fits; the streamed fallback keeps a user context or refuses it by name; and one stage of a layer split plans only
 * its span's caches and places and streams only its span's layers. Every budget boundary is derived from
 * the same public primitives the fit plans with ({@link LayerWeightSet}, {@link StreamingLayerWeightSource#plannedBytes},
 * {@link ContextFitter#requiredBytes}).
 */
public class PlacementFitterTest {

    private static final int FLOOR = ContextFitter.CONTEXT_STEP;
    private static final LayerWeightSet.Options OPTIONS = LayerWeightSet.Options.builder(t -> t.type()).build();
    private static final Predicate<TensorRef> EXPERT = t -> WeightShards.isExpert(t.name(), t.rank());
    private static final LayerWeightSet.Options EXPERT_OPTIONS =
            LayerWeightSet.Options.builder(t -> t.type()).filter(EXPERT).build();

    /** A fixture with the byte totals the budgets are derived from. */
    private static final class Fixture {
        final ModelSpec spec;
        final long weightBytes;
        final long trunkBytes;
        final long baseBytes;
        final long expertBytes;
        final List<LayerWeightSet> allSets;
        final List<LayerWeightSet> expertSets;

        Fixture(ModelSpec spec) {
            this.spec = spec;
            TensorCatalog catalog = spec.tensors();
            this.weightBytes = tensorBytes(catalog, t -> true);
            this.allSets = LayerWeightSet.forTrunk(spec, OPTIONS);
            this.expertSets = LayerWeightSet.forTrunk(spec, EXPERT_OPTIONS);
            long trunk = 0L;
            for (LayerWeightSet s : allSets) {
                trunk += s.uploadBytes();
            }
            this.trunkBytes = trunk;
            this.baseBytes = weightBytes - trunk;
            long experts = 0L;
            for (LayerWeightSet s : expertSets) {
                experts += s.uploadBytes();
            }
            this.expertBytes = experts;
        }

        /** Bytes of the whole-layer set of weight layer {@code wl}. */
        long layerBytes(int wl) {
            return set(allSets, wl).uploadBytes();
        }

        /** Bytes of the expert-stack set of weight layer {@code wl}. */
        long expertLayerBytes(int wl) {
            return set(expertSets, wl).uploadBytes();
        }

        /** Device bytes of the streaming slots for {@code sets} at the depth the fit reserves. */
        long slots(List<LayerWeightSet> sets) {
            return sets.isEmpty() ? 0L
                    : StreamingLayerWeightSource.plannedBytes(spec, sets,
                            Math.min(PlacementFitter.MIN_SLOTS, sets.size()));
        }

        long ctxBytes(int nCtx) {
            return ContextFitter.requiredBytes(spec, MemoryConfig.defaults(), nCtx, ContextFitter.Companion.NONE);
        }

        /** Device bytes of the all-experts-on-the-host placement (the fit's {@code MOE_OVERFLOW} cost). */
        long denseOnlyBytes(int nCtx) {
            return baseBytes + (trunkBytes - expertBytes) + slots(expertSets) + ctxBytes(nCtx);
        }

        /** Device bytes of the placement whose first {@code s} trunk weight layers stream entirely. */
        long partialBytes(int s, int nCtx) {
            List<LayerWeightSet> sets = new ArrayList<>(allSets.subList(0, s));
            long denseTail = trunkBytes;
            for (int i = 0; i < s; i++) {
                denseTail -= allSets.get(i).uploadBytes();
            }
            for (LayerWeightSet e : expertSets) {
                if (e.weightLayer() >= s) {
                    sets.add(e);
                }
            }
            return baseBytes + denseTail + slots(sets) + ctxBytes(nCtx);
        }

        Fit fit(long usable) {
            return fit(MemoryConfig.defaults(), usable, FLOOR);
        }

        Fit fit(MemoryConfig mem, long usable, int floor) {
            return new PlacementFitter.Builder(spec, spec, spec.tensors(), OPTIONS, t -> true)
                    .usableBytes(usable)
                    .weightBytes(weightBytes)
                    .memory(mem)
                    .companion(ContextFitter.Companion.NONE)
                    .contextFloor(floor)
                    .backendType("OPENCL")
                    .build();
        }

        private static LayerWeightSet set(List<LayerWeightSet> sets, int wl) {
            for (LayerWeightSet s : sets) {
                if (s.weightLayer() == wl) {
                    return s;
                }
            }
            throw new IllegalArgumentException("no set for weight layer " + wl);
        }
    }

    private static ModelSpec build(GgufTestWriter w) throws IOException {
        Path path = w.writeTempFile();
        GGUFFileParser parser = null;
        try {
            parser = new GGUFFileParser(path.toString());
            return ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser), MemoryConfig.defaults());
        } finally {
            try {
                if (parser != null) {
                    parser.close();
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }

    private static long tensorBytes(TensorCatalog catalog, Predicate<TensorRef> filter) {
        long sum = 0L;
        for (String name : catalog.names()) {
            TensorRef t = catalog.get(name);
            if (filter.test(t)) {
                sum += t.rows() * t.type().getBytesForElements(t.cols());
            }
        }
        return sum;
    }

    private static Fixture dense() throws IOException {
        return new Fixture(build(SyntheticModels.llama(
                SyntheticModels.Config.llama().toBuilder().nLayer(4).nFf(256).build())));
    }

    private static Fixture moe() throws IOException {
        return new Fixture(build(SyntheticModels.qwen3next()));
    }

    private static Set<String> names(Fit fit) {
        Set<String> out = new HashSet<>();
        for (LayerWeightSet s : fit.streamedSets()) {
            out.addAll(s.names());
        }
        return out;
    }

    private static void assertAccounts(Fixture f, Fit fit) {
        assertEquals(f.weightBytes, fit.deviceWeightBytes() + fit.hostWeightBytes());
        long streamed = 0L;
        for (LayerWeightSet s : fit.streamedSets()) {
            streamed += s.uploadBytes();
        }
        assertEquals(fit.hostWeightBytes(), streamed);
    }

    // ================================================================== resident

    @Test
    public void aBudgetThatHoldsEveryWeightKeepsThemResident() throws IOException {
        Fixture f = moe();
        Fit fit = f.fit(f.weightBytes + f.ctxBytes(FLOOR));

        assertEquals(PlacementFitter.Mode.RESIDENT, fit.mode());
        assertEquals(FLOOR, fit.nCtx());
        assertTrue(fit.streamedSets().isEmpty());
        assertEquals(0, fit.streamedLayers());
        assertEquals(4, fit.gpuLayers());
        assertAccounts(f, fit);
    }

    // ================================================================== MoE overflow

    @Test
    public void moeOverflowStreamsOnlyTheExpertTensors() throws IOException {
        Fixture f = moe();
        Fit fit = f.fit(f.denseOnlyBytes(FLOOR));

        assertEquals(PlacementFitter.Mode.MOE_OVERFLOW, fit.mode());
        assertEquals(0, fit.fullLayers());
        assertEquals(0, fit.streamedLayers());
        assertEquals(4, fit.gpuLayers());
        assertEquals(FLOOR, fit.nCtx());
        assertEquals(4, fit.streamedSets().size());
        for (LayerWeightSet set : fit.streamedSets()) {
            for (LayerWeightSet.Entry e : set.entries()) {
                assertTrue(e.name() + " is not an expert stack", EXPERT.test(e.source()));
            }
        }
        assertTrue(names(fit).contains("blk.0.ffn_up_exps.weight"));
        assertTrue(names(fit).contains("blk.3.ffn_down_exps.weight"));
        assertTrue(!names(fit).contains("blk.0.ffn_up_shexp.weight"));
        assertTrue(!names(fit).contains("blk.0.ffn_gate_inp.weight"));
        assertTrue(!names(fit).contains("blk.0.attn_q.weight"));
        assertEquals(f.baseBytes + (f.trunkBytes - f.expertBytes), fit.deviceWeightBytes());
        assertAccounts(f, fit);
    }

    @Test
    public void moeUpgradeBringsTheLeadingExpertsBackWhileTheyFit() throws IOException {
        Fixture f = moe();
        long denseOnly = f.denseOnlyBytes(FLOOR);
        long e0 = f.expertLayerBytes(0);

        Fit one = f.fit(denseOnly + e0);
        assertEquals(PlacementFitter.Mode.MOE_UPGRADED, one.mode());
        assertEquals(1, one.fullLayers());
        assertEquals(3, one.streamedSets().size());
        for (LayerWeightSet set : one.streamedSets()) {
            assertTrue("weight layer " + set.weightLayer() + " streams its experts", set.weightLayer() >= 1);
        }
        assertAccounts(f, one);

        Fit none = f.fit(denseOnly + e0 - 1);
        assertEquals(PlacementFitter.Mode.MOE_OVERFLOW, none.mode());
        assertEquals(0, none.fullLayers());

        Fit almostAll = f.fit(denseOnly + 2L * e0 - 1);
        assertEquals(PlacementFitter.Mode.MOE_UPGRADED, almostAll.mode());
        assertEquals(1, almostAll.fullLayers());
    }

    @Test
    public void aBudgetThatHoldsEveryExpertKeepsThemResident() throws IOException {
        Fixture f = moe();
        Fit fit = f.fit(f.denseOnlyBytes(FLOOR) + 2L * f.expertLayerBytes(0));

        assertEquals(PlacementFitter.Mode.RESIDENT, fit.mode());
        assertTrue(fit.streamedSets().isEmpty());
    }

    // ================================================================== partial streaming

    @Test
    public void denseModelStreamsTheSmallestPrefixOfLayersThatFits() throws IOException {
        Fixture f = dense();
        long cost = f.partialBytes(3, FLOOR);

        Fit fit = f.fit(cost);
        assertEquals(PlacementFitter.Mode.PARTIAL, fit.mode());
        assertEquals(3, fit.streamedLayers());
        assertEquals(1, fit.gpuLayers());
        assertEquals(FLOOR, fit.nCtx());
        assertEquals(3, fit.streamedSets().size());
        assertTrue(names(fit).contains("blk.0.ffn_gate.weight"));
        assertTrue(names(fit).contains("blk.2.ffn_down.weight"));
        assertTrue(!names(fit).contains("blk.3.ffn_gate.weight"));
        assertEquals(f.layerBytes(0) + f.layerBytes(1) + f.layerBytes(2), fit.hostWeightBytes());
        assertAccounts(f, fit);
    }

    @Test
    public void aSmallerBudgetStreamsEveryLayer() throws IOException {
        Fixture f = dense();
        Fit fit = f.fit(f.partialBytes(3, FLOOR) - 1);

        assertEquals(PlacementFitter.Mode.PARTIAL, fit.mode());
        assertEquals(4, fit.streamedLayers());
        assertEquals(0, fit.gpuLayers());
        assertEquals(FLOOR, fit.nCtx());
        assertEquals(4, fit.streamedSets().size());
        assertEquals(f.trunkBytes, fit.hostWeightBytes());
        assertAccounts(f, fit);
    }

    @Test
    public void aResidentBudgetWinsOverEqualCostStreaming() throws IOException {
        Fixture f = dense();
        Fit fit = f.fit(f.partialBytes(1, FLOOR));

        assertEquals(PlacementFitter.Mode.RESIDENT, fit.mode());
        assertEquals(FLOOR, fit.nCtx());
        assertTrue(fit.streamedSets().isEmpty());
    }

    // ================================================================== streamed fallback

    @Test
    public void aModelWithoutMemoryStreamsAtItsFixedContext() throws IOException {
        Fixture f = new Fixture(build(SyntheticModels.bert(
                SyntheticModels.Config.bert().toBuilder().nLayer(4).build())));
        long cost = f.partialBytes(3, SyntheticModels.Config.bert().nCtxTrain());

        Fit fit = f.fit(cost);

        assertEquals(PlacementFitter.Mode.PARTIAL, fit.mode());
        assertEquals(3, fit.streamedLayers());
        assertEquals(SyntheticModels.Config.bert().nCtxTrain(), fit.nCtx());
        assertEquals(1, fit.gpuLayers());
        assertAccounts(f, fit);
    }

    @Test
    public void belowTheFloorContextEveryLayerStreams() throws IOException {
        Fixture f = moe();
        MemoryConfig mem = MemoryConfig.defaults();
        long slots = f.slots(f.allSets);
        long between = (f.ctxBytes(FLOOR) + f.ctxBytes(2 * FLOOR)) / 2L;
        assertTrue(between > f.ctxBytes(FLOOR) && between < f.ctxBytes(2 * FLOOR));

        Fit fit = f.fit(mem, f.baseBytes + slots + between, 2 * FLOOR);

        assertEquals(PlacementFitter.Mode.STREAMED, fit.mode());
        assertEquals(FLOOR, fit.nCtx());
        assertEquals(0, fit.gpuLayers());
        assertEquals(4, fit.streamedSets().size());
        assertEquals(f.allSets.size(), fit.streamedSets().size());
        assertAccounts(f, fit);
    }

    @Test
    public void weightsThatDoNotFitEvenStreamedAreRefused() throws IOException {
        Fixture f = moe();
        assertThrows(GpuOutOfMemoryException.class, () -> f.fit(f.baseBytes + f.slots(f.allSets) - 1));
    }

    // ================================================================== user context

    @Test
    public void aUserContextIsKeptWhileWeightsMoveToTheHost() throws IOException {
        Fixture f = moe();
        int ctx = 3000;
        MemoryConfig mem = MemoryConfig.builder().nCtx(ctx).build();

        Fit fit = f.fit(mem, f.denseOnlyBytes(ctx), 4096);

        assertEquals(PlacementFitter.Mode.MOE_OVERFLOW, fit.mode());
        assertEquals(ctx, fit.nCtx());
        assertAccounts(f, fit);
    }

    @Test
    public void aUserContextThatCannotFitIsRefusedByName() throws IOException {
        Fixture f = moe();
        MemoryConfig mem = MemoryConfig.builder().nCtx(3000).build();

        ContextFitter.ContextBudgetException e = assertThrows(ContextFitter.ContextBudgetException.class,
                () -> f.fit(mem, f.baseBytes + f.slots(f.allSets) + 1, 4096));
        assertTrue(e.getMessage(), e.getMessage().contains(MemoryConfig.PROP_CTX));
    }

    // ================================================================== one stage of a layer split

    /** Whether {@code name} is a trunk tensor of layer {@code from} or later: the span of the last stage. */
    private static boolean inSpan(String name, int from) {
        if (!name.startsWith("blk.")) {
            return false;
        }
        return Integer.parseInt(name.substring(4, name.indexOf('.', 4))) >= from;
    }

    /** The fit of the stage that owns trunk layers {@code from} and later, its context over those layers or all. */
    private static Fit stageFit(Fixture f, int from, long usable, boolean spanContext) {
        Predicate<TensorRef> owned = t -> inSpan(t.name(), from);
        PlacementFitter.Builder b = new PlacementFitter.Builder(f.spec, f.spec, f.spec.tensors(), OPTIONS, owned)
                .usableBytes(usable)
                .weightBytes(tensorBytes(f.spec.tensors(), owned))
                .memory(MemoryConfig.defaults())
                .contextFloor(FLOOR)
                .backendType("OPENCL");
        if (spanContext) {
            b.layers(il -> il >= from);
        }
        return b.build();
    }

    private static long spanCtxBytes(Fixture f, int from, int nCtx) {
        return ContextFitter.requiredLayerBytes(f.spec, MemoryConfig.defaults(), nCtx, il -> il >= from);
    }

    @Test
    public void aStageHoldsOnlyTheCachesOfItsSpan() throws IOException {
        Fixture f = dense();
        long spanWeights = f.layerBytes(2) + f.layerBytes(3);
        assertTrue(spanCtxBytes(f, 2, FLOOR) < f.ctxBytes(FLOOR));
        long usable = spanWeights + spanCtxBytes(f, 2, FLOOR);

        Fit fit = stageFit(f, 2, usable, true);

        assertEquals(PlacementFitter.Mode.RESIDENT, fit.mode());
        assertEquals(FLOOR, fit.nCtx());
        assertEquals("the stage places its own two layers", 2, fit.weightLayers());
        assertEquals(spanWeights, fit.deviceWeightBytes());
        try {
            Fit whole = stageFit(f, 2, usable, false);
            assertTrue("caches of every layer leave the stage short of the floor: " + whole,
                    whole.mode() != PlacementFitter.Mode.RESIDENT);
        } catch (GpuOutOfMemoryException | ContextFitter.ContextBudgetException expected) {
            // the caches of every layer do not fit beside the span's weights at all
        }
    }

    @Test
    public void aStageStreamsAPrefixOfItsOwnLayers() throws IOException {
        // a four-layer span of a six-layer model: streaming its first three through two slots costs less than
        // keeping them resident, while a shorter prefix does not
        Fixture f = new Fixture(build(SyntheticModels.llama(
                SyntheticModels.Config.llama().toBuilder().nLayer(6).nFf(256).build())));
        List<LayerWeightSet> prefix = new ArrayList<>(f.allSets.subList(2, 5));
        long usable = f.layerBytes(5) + f.slots(prefix) + spanCtxBytes(f, 2, FLOOR);

        Fit fit = stageFit(f, 2, usable, true);

        assertEquals(PlacementFitter.Mode.PARTIAL, fit.mode());
        assertEquals("the stage places its own four layers", 4, fit.weightLayers());
        assertEquals(3, fit.streamedLayers());
        assertEquals(1, fit.gpuLayers());
        assertEquals(FLOOR, fit.nCtx());
        assertTrue(names(fit).contains("blk.2.ffn_gate.weight"));
        assertTrue(names(fit).contains("blk.4.ffn_down.weight"));
        assertTrue(!names(fit).contains("blk.5.ffn_gate.weight"));
        assertTrue(!names(fit).contains("blk.0.ffn_gate.weight"));
        assertEquals(f.layerBytes(2) + f.layerBytes(3) + f.layerBytes(4), fit.hostWeightBytes());
        assertEquals(f.layerBytes(5), fit.deviceWeightBytes());
        assertTrue(fit.describe(), fit.describe().contains("1 of 4 trunk weight layers"));
    }

    @Test
    public void aStageFitHoldsItsCompanionBesideTheSpan() throws IOException {
        Fixture f = dense();
        long spanWeights = f.layerBytes(2) + f.layerBytes(3);
        long drafter = 4L * 1024 * 1024;
        long usable = spanWeights + spanCtxBytes(f, 2, FLOOR);

        Fit alone = stageFit(f, 2, usable, true);
        assertEquals(PlacementFitter.Mode.RESIDENT, alone.mode());

        Predicate<TensorRef> owned = t -> inSpan(t.name(), 2);
        Fit withDrafter = new PlacementFitter.Builder(f.spec, f.spec, f.spec.tensors(), OPTIONS, owned)
                .usableBytes(usable + drafter)
                .weightBytes(tensorBytes(f.spec.tensors(), owned))
                .memory(MemoryConfig.defaults())
                .contextFloor(FLOOR)
                .backendType("OPENCL")
                .layers(il -> il >= 2)
                .companion(ContextFitter.Companion.of("drafter", drafter, 0L))
                .build();
        assertEquals("the drafter's bytes are held beside the span's caches", PlacementFitter.Mode.RESIDENT,
                withDrafter.mode());
        assertEquals(FLOOR, withDrafter.nCtx());
        assertEquals(spanCtxBytes(f, 2, FLOOR) + drafter, ContextFitter.requiredLayerBytes(f.spec,
                MemoryConfig.defaults(), FLOOR, il -> il >= 2, ContextFitter.Companion.of("drafter", drafter, 0L)));
    }

    // ================================================================== inputs

    @Test
    public void missingOrInconsistentInputsAreRefused() throws IOException {
        Fixture f = moe();
        assertThrows(IllegalArgumentException.class, () -> new PlacementFitter.Builder(f.spec, f.spec,
                f.spec.tensors(), OPTIONS, t -> true).build());
        assertThrows(IllegalArgumentException.class, () -> builder(f, f.weightBytes)
                .usableBytes(-1L).build());
        assertThrows(IllegalArgumentException.class, () -> builder(f, f.weightBytes)
                .contextFloor(0).build());
        assertThrows(IllegalArgumentException.class, () -> new PlacementFitter.Builder(f.spec, f.spec,
                f.spec.tensors(), OPTIONS, t -> true)
                .usableBytes(f.weightBytes).weightBytes(f.weightBytes)
                .memory(MemoryConfig.defaults()).contextFloor(FLOOR).build());
        assertThrows(IllegalArgumentException.class, () -> builder(f, f.trunkBytes - 1).build());
    }

    /** A builder with every required input but the memory settings. */
    private static PlacementFitter.Builder builder(Fixture f, long weightBytes) {
        return new PlacementFitter.Builder(f.spec, f.spec, f.spec.tensors(), OPTIONS, t -> true)
                .usableBytes(f.weightBytes + f.ctxBytes(FLOOR))
                .weightBytes(weightBytes)
                .memory(MemoryConfig.defaults())
                .contextFloor(FLOOR)
                .backendType("OPENCL");
    }
}
