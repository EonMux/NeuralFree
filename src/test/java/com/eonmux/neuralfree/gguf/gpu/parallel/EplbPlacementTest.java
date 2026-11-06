package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The EPLB packing algorithm against vLLM's {@code DefaultEplbPolicy.balanced_packing} and its application to a
 * synthetic model's routed MoE layer: the heaviest experts spread over the lightest ranks, one expert per rank
 * keeping its order, every expert placed once, and the layers a profile does not count keeping their contiguous
 * blocks. No GPU is needed.
 */
public class EplbPlacementTest {

    private Path file;
    private GGUFFileParser parser;
    private ModelSpec spec;

    @Before
    public void open() throws IOException {
        file = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1).build())
                .writeTempFile();
        parser = new GGUFFileParser(file.toString());
        spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                MemoryConfig.defaults().toBuilder().ubatch(3).build());
    }

    @After
    public void close() throws IOException {
        try {
            parser.close();
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void theHeaviestExpertsSpreadOverTheLightestRanks() {
        // vLLM: e0 (10) -> pack 0, e1 (10) -> pack 1, e2 (1) -> pack 0, e3 (1) -> pack 1.
        assertArrayEquals(new int[]{0, 2, 1, 3}, EplbPlacement.phy2log(new double[]{10, 10, 1, 1}, 2));
        // The contiguous blocks would leave 20 of load on a rank and 2 on the other.
        EplbPlacement placement = EplbPlacement.of(spec, 2, loads(10, 10, 1, 1));
        LayerSpec layer = spec.layers().get(0);
        assertTrue(placement.covers(layer));
        assertArrayEquals(new int[]{0, 2}, placement.experts(layer, 0));
        assertArrayEquals(new int[]{1, 3}, placement.experts(layer, 1));
    }

    @Test
    public void evenLoadsPackRoundRobin() {
        assertArrayEquals(new int[]{0, 2, 1, 3}, EplbPlacement.phy2log(new double[]{5, 5, 5, 5}, 2));
    }

    @Test
    public void oneExpertPerRankKeepsTheOrder() {
        assertArrayEquals(new int[]{0, 1}, EplbPlacement.phy2log(new double[]{9, 1}, 2));
    }

    @Test
    public void everyExpertIsPlacedExactlyOnce() {
        double[] load = {7, 1, 9, 3, 2, 8, 4, 6};
        int ranks = 4;
        int[] phy2log = EplbPlacement.phy2log(load, ranks);
        int[] sorted = phy2log.clone();
        Arrays.sort(sorted);
        assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5, 6, 7}, sorted);
        double[] rankLoad = new double[ranks];
        for (int slot = 0; slot < phy2log.length; slot++) {
            rankLoad[slot / (phy2log.length / ranks)] += load[phy2log[slot]];
        }
        double max = 0;
        double min = Double.MAX_VALUE;
        for (double l : rankLoad) {
            max = Math.max(max, l);
            min = Math.min(min, l);
        }
        assertEquals("the packing balances the heaviest rank within one expert of the lightest", 0.0,
                max - min, 3.0);
    }

    @Test
    public void theProfileFillsThePlacementOfItsLayers() {
        EplbPlacement placement = EplbPlacement.of(spec, 2, loads(100, 100, 1, 1));
        assertEquals(1, placement.layered());
        assertArrayEquals(new int[]{0, 2}, placement.experts(spec.layers().get(0), 0));
        assertArrayEquals(new int[]{1, 3}, placement.experts(spec.layers().get(0), 1));
        assertThrows(IllegalArgumentException.class, () -> placement.experts(spec.layers().get(0), 2));
    }

    @Test
    public void anUncountedLayerKeepsItsContiguousBlock() {
        EplbPlacement none = EplbPlacement.of(spec, 2, List.of());
        assertFalse(none.covers(spec.layers().get(0)));
        assertNull(none.experts(spec.layers().get(0), 0));
    }

    @Test
    public void thePlacementRefusesAnExpertOutsideItsLayer() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EplbPlacement.of(spec, 2, List.of(new RankedExpert(0, 5, 1))));
        assertTrue(e.getMessage(), e.getMessage().contains("5"));
        assertTrue(e.getMessage(), e.getMessage().contains("4"));
    }

    @Test
    public void thePlacementRefusesBadRankCountsAndLoads() {
        assertThrows(IllegalArgumentException.class, () -> EplbPlacement.phy2log(new double[]{1, 1, 1}, 2));
        assertThrows(IllegalArgumentException.class, () -> EplbPlacement.phy2log(new double[]{-1, 1}, 2));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPlacement.phy2log(new double[]{Double.NaN, 1}, 2));
        assertThrows(IllegalArgumentException.class, () -> EplbPlacement.of(spec, 1, List.of()));
    }

    private static List<RankedExpert> loads(int... heat) {
        RankedExpert[] out = new RankedExpert[heat.length];
        for (int e = 0; e < heat.length; e++) {
            out[e] = new RankedExpert(0, e, heat[e]);
        }
        return Arrays.asList(out);
    }
}
