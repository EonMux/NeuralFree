package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The hosted MoE layers of a GPU load: a placement that keeps every expert set on the device hosts none and leaves
 * the block registry undecorated; a placement that keeps the routed experts of a qwen3moe model in RAM hosts every MoE
 * layer on the hosted expert path (the cached MoE pass over the host weights), which needs no FFN decorator.
 */
public class GpuHostedLayersTest {

    @Test
    public void aDevicePlacementHostsNoLayerAndKeepsTheRegistry() throws IOException {
        withModel(SyntheticModels.qwen3moe(), (spec, catalog) -> {
            GpuHostedLayers hosted = GpuHostedLayers.resolve(spec, catalog, WeightPlacement.of(spec, false), null);
            assertFalse(hosted.hasCached());
            assertTrue(hosted.cached().isEmpty());
            GpuBlockRegistry base = GpuFactoryCatalog.defaults();
            assertSame(base, hosted.decorate(base, null));
        });
    }

    @Test
    public void hostedRoutedExpertsRunOnTheHostedPath() throws IOException {
        withModel(SyntheticModels.qwen3moe(), (spec, catalog) -> {
            WeightPlacement placement = WeightPlacement.of(spec, false)
                    .withHostTensors(name -> name.contains("_exps."));
            GpuHostedLayers hosted = GpuHostedLayers.resolve(spec, catalog, placement, null);
            assertTrue(hosted.hasCached());
            List<Integer> layers = new ArrayList<>();
            for (LayerSpec l : hosted.cached()) {
                layers.add(l.il());
            }
            List<Integer> expected = new ArrayList<>();
            for (int il = 0; il < spec.nLayer(); il++) {
                expected.add(il);
            }
            assertEquals(expected, layers);
            GpuBlockRegistry base = GpuFactoryCatalog.defaults();
            assertSame(base, hosted.decorate(base, null));
        });
    }

    @Test
    public void noneHostsNothing() {
        assertFalse(GpuHostedLayers.NONE.hasCached());
        GpuBlockRegistry base = GpuFactoryCatalog.defaults();
        assertSame(base, GpuHostedLayers.NONE.decorate(base, null));
    }

    private interface ModelCheck {
        void run(ModelSpec spec, TensorCatalog catalog);
    }

    private static void withModel(GgufTestWriter writer, ModelCheck check) throws IOException {
        Path file = writer.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            TensorCatalog catalog = TensorCatalog.of(parser);
            check.run(ModelSpecBuilder.build(GgufKv.of(parser), catalog, MemoryConfig.defaults()), catalog);
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
