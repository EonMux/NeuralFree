package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfMoeClassicFixturesA;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertTrue;

/**
 * The HK-MC part A fixtures ({@code Llama4Model}, {@code DbrxModel}, {@code GrokModel}, {@code ArcticModel}) build
 * through the registered entries: {@link ModelSpecBuilder} succeeds and every catalog tensor is claimed
 * ({@code TensorClaims}), which is the done-getting-tensors check of the converter's tensor plan. {@code
 * UltravoxModel}, the named rejection of the table, refuses with the converter's reason.
 */
public class HfMoeClassicASourcesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyPartAFixtureBuildsASpecClaimingItsTensors() throws IOException {
        for (HfMoeClassicFixturesA.Named f : HfMoeClassicFixturesA.all()) {
            Path dir = f.fixture.get().write(tmp.newFolder("hf").toPath());
            buildAndClaim(dir, tmp.newFolder("cache").toPath());
        }
    }

    @Test
    public void ultravoxIsANamedRejection() throws IOException {
        Path dir = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['UltravoxModel'], 'vocab_size': 8}")).write(tmp.newFolder("hf").toPath());
        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg = HfConfigLoader.fromJson(dir,
                SyntheticHf.json("{'architectures': ['UltravoxModel'], 'vocab_size': 8}"));
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        assertTrue(entry.isRejected());
        assertTrue(entry.rejection().contains("UltravoxModel has no text model"));
    }

    /** Builds and claims the spec of one checkpoint dir. */
    private static void buildAndClaim(Path dir, Path cacheDir) throws IOException {
        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(cacheDir).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(),
                    ArchRegistry.get(spec.arch()).tensors());
        }
    }
}
