package com.eonmux.neuralfree.gguf.gpu.block;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;

/**
 * Successive FFN decorators of a block registry compose: the second receives the block the first returned, so a row
 * split laid over a hybrid load's host-expert decorator replaces only the layers it plans.
 */
public class GpuBlockRegistryDecoratorTest {

    @Test
    public void decoratorsComposeInInstallationOrder() throws IOException {
        Path file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build())
                .writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                    MemoryConfig.defaults());
            LayerSpec layer = spec.layers().get(0);
            GpuFfnBlock inner = new Marker("inner");
            GpuFfnBlock outer = new Marker("outer");
            GpuFfnBlock[] seen = new GpuFfnBlock[1];
            GpuBlockRegistry composed = GpuFactoryCatalog.defaults()
                    .withFfnDecorator((l, block) -> {
                        assertNotNull("the factory block reaches the first decorator", block);
                        return inner;
                    })
                    .withFfnDecorator((l, block) -> {
                        seen[0] = block;
                        return outer;
                    });
            assertSame(outer, composed.ffn(layer));
            assertSame("the second decorator sees the first's block", inner, seen[0]);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void ffnDecorationKeepsTheInstalledMixerDecorator() throws IOException {
        Path file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build())
                .writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                    MemoryConfig.defaults());
            LayerSpec layer = spec.layers().get(0);
            GpuMixer mixer = new MixerMarker("mixer");
            GpuFfnBlock ffn = new Marker("ffn");
            GpuBlockRegistry decorated = GpuFactoryCatalog.defaults()
                    .withMixerDecorator((l, block) -> mixer)
                    .withFfnDecorator((l, block) -> ffn);
            assertSame("the FFN decoration keeps the host-KV mixer decoration", mixer, decorated.mixer(layer));
            assertSame(ffn, decorated.ffn(layer));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    @Test
    public void mixerDecorationKeepsTheInstalledFfnDecorator() throws IOException {
        Path file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build())
                .writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(file.toString())) {
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                    MemoryConfig.defaults());
            LayerSpec layer = spec.layers().get(0);
            GpuMixer mixer = new MixerMarker("mixer");
            GpuFfnBlock ffn = new Marker("ffn");
            GpuBlockRegistry decorated = GpuFactoryCatalog.defaults()
                    .withFfnDecorator((l, block) -> ffn)
                    .withMixerDecorator((l, block) -> mixer);
            assertSame(mixer, decorated.mixer(layer));
            assertSame("the mixer decoration keeps the FFN decoration", ffn, decorated.ffn(layer));
        } finally {
            Files.deleteIfExists(file);
        }
    }

    /** A mixer that is only an identity marker here. */
    private static final class MixerMarker implements GpuMixer {
        private final String name;

        MixerMarker(String name) {
            this.name = name;
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            throw new UnsupportedOperationException(name + " is not bound in this test");
        }

        @Override
        public void decode(int pos, ScratchSlot in, ScratchSlot out) {
            throw new UnsupportedOperationException(name + " is not run in this test");
        }

        @Override
        public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
            throw new UnsupportedOperationException(name + " is not run in this test");
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /** A block that is only an identity marker here. */
    private static final class Marker implements GpuFfnBlock {
        private final String name;

        Marker(String name) {
            this.name = name;
        }

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            throw new UnsupportedOperationException(name + " is not bound in this test");
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
            throw new UnsupportedOperationException(name + " is not run in this test");
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
