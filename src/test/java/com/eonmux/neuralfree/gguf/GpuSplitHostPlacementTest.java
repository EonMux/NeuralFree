package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static com.eonmux.neuralfree.gguf.SplitFixtures.assertParity;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static com.eonmux.neuralfree.gguf.SplitFixtures.run;
import static com.eonmux.neuralfree.gguf.SplitFixtures.split;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Host placement over a two-device layer split, each setting against the one-device load of the same setting (greedy
 * parity of the logits and of the generated tokens): the first layers on the host ({@code -ngl}, an {@code -ot} match
 * on a layer tensor, {@code --n-cpu-ffn}), the routed experts in RAM ({@code --cpu-moe}, {@code --n-cpu-moe}, an
 * {@code -ot} match on expert tensors, {@code --fit} per device), and the MoE expert cache of every device over its
 * own layers; the layouts the split cannot run are refused by name.
 */
public class GpuSplitHostPlacementTest {

    private static final int LAYERS = 4;

    private final List<Path> files = new ArrayList<>();

    @Before
    public void requireTwoDevices() {
        SplitFixtures.assumeTwoDevices();
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private Path llama() throws IOException {
        return written(SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(LAYERS).build()));
    }

    private Path qwen3moe() throws IOException {
        return written(SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(LAYERS)
                .build()));
    }

    /** The split and the one-device load of {@code setting} agree; the split runs {@code hostLayers} on the host. */
    private static void assertSplitMatches(Path model, Setting setting, int hostLayers) throws IOException {
        SplitFixtures.Run reference = run(model, setting.apply(oneDevice()).build());
        SplitFixtures.Run split = run(model, setting.apply(split()).build());
        assertTrue(split.backend, split.backend.startsWith("GpuSplitForwardBackend"));
        assertEquals(split.backend, hostLayers > 0, split.backend.contains("host layers [0, " + hostLayers + ")"));
        assertParity(split.backend, reference, split);
    }

    /** One placement setting over a configuration. */
    private interface Setting {
        EngineConfig.Builder apply(EngineConfig.Builder b);
    }

    // ------------------------------------------------------------------ host layers

    @Test
    public void aLayerCountKeepsTheFirstLayersOnTheHost() throws IOException {
        assertSplitMatches(llama(), b -> b.gpuLayers(3), 2);
    }

    @Test
    public void anOverrideOfALayerTensorKeepsItsLayerAndTheEarlierOnesOnTheHost() throws IOException {
        assertSplitMatches(llama(), b -> b.addTensorOverride("blk\\.1\\.ffn_up", "CPU"), 2);
    }

    @Test
    public void denseFfnWeightsOnTheHostRunTheirLayersOnTheHost() throws IOException {
        assertSplitMatches(llama(), b -> b.nCpuFfn(1), 1);
    }

    // ------------------------------------------------------------------ routed experts

    @Test
    public void everyRoutedExpertInRamMatchesTheOneDeviceLoad() throws IOException {
        assertSplitMatches(qwen3moe(), b -> b.cpuMoe(true), 0);
    }

    @Test
    public void theExpertsOfTheFirstLayersInRamMatchTheOneDeviceLoad() throws IOException {
        assertSplitMatches(qwen3moe(), b -> b.nCpuMoe(3), 0);
    }

    @Test
    public void anOverrideOfExpertTensorsKeepsThemInRamBesideTheirDeviceLayer() throws IOException {
        assertSplitMatches(qwen3moe(), b -> b.addTensorOverride("blk\\.3\\.ffn_(up|down|gate)_exps", "CPU"), 0);
    }

    @Test
    public void theFitMovesTheExpertsOfEveryDeviceThatHoldsItsDenseWeights() throws IOException {
        assertSplitMatches(qwen3moe(), b -> b.fit(true), 0);
    }

    @Test
    public void hostLayersAndHostedExpertsCombine() throws IOException {
        assertSplitMatches(qwen3moe(), b -> b.gpuLayers(4).cpuMoe(true), 1);
    }

    // ------------------------------------------------------------------ MoE expert cache

    @Test
    public void everyDeviceCachesTheExpertsOfItsOwnLayers() throws IOException {
        Path model = qwen3moe();
        Setting cache = b -> b.moeCache(MoeCacheConfig.parse("on"));
        SplitFixtures.Run reference = run(model, cache.apply(oneDevice()).build());
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cache.apply(split()).build())) {
            SplitFixtures.Run split = new SplitFixtures.Run(SplitFixtures.logits(m), SplitFixtures.greedy(m),
                    m.forwardBackend().toString());
            assertParity("MoE cache over the split", reference, split);
            assertEquals(2, m.deviceCount());
            for (int d = 0; d < m.deviceCount(); d++) {
                MoeExpertCache c = m.moeExpertCacheOf(d);
                assertNotNull("device " + d + " caches the experts of its layers", c);
                int first = m.firstLayerOf(d);
                int end = first + m.layerCountOf(d);
                for (MoeExpertCache.LayerPlan p : c.layerPlans()) {
                    assertTrue("layer " + p.weightLayer() + " is cached on device " + d + " (layers " + first
                            + ".." + (end - 1) + ")", p.weightLayer() >= first && p.weightLayer() < end);
                }
                assertEquals("device " + d + " caches the experts of its " + m.layerCountOf(d) + " layers",
                        m.layerCountOf(d), c.layerPlans().size());
            }
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void aLayerCountThatLeavesADeviceWithoutLayersIsRefusedByName() throws IOException {
        String path = llama().toString();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new GGUFModelGPU(path, null, split().gpuLayers(2).build()).close());
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_N_GPU_LAYERS));
        assertTrue(e.getMessage(), e.getMessage().contains("at least one layer per device"));
    }

    @Test
    public void theExpertCacheBesideHostLayersIsRefusedByName() throws IOException {
        String path = qwen3moe().toString();
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> new GGUFModelGPU(path, null, split().gpuLayers(4).moeCache(MoeCacheConfig.parse("on"))
                        .build()).close());
        assertTrue(e.getMessage(), e.getMessage().contains(MoeCacheConfig.PROP_MOE_CACHE));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_N_GPU_LAYERS));
    }
}
