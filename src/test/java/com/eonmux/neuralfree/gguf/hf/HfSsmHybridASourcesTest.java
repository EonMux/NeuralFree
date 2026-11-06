package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfSsmFixturesHybridA;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertTrue;

/**
 * The first-half hybrid fixtures ({@code FalconH1ForCausalLM}, {@code JambaForCausalLM},
 * {@code Plamo2ForCausalLM}, {@code BambaForCausalLM}, {@code NemotronHForCausalLM} and
 * {@code NemotronHPuzzleForCausalLM}) build through the registered entries of
 * {@code HfArchTableHybridSsm}: {@link ModelSpecBuilder} succeeds and every catalog tensor is claimed
 * ({@code TensorClaims}), the done-getting-tensors check of the converter's tensor plan.
 */
public class HfSsmHybridASourcesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyHybridAFixtureBuildsASpecClaimingItsTensors() throws IOException {
        for (HfSsmFixturesHybridA.Named f : HfSsmFixturesHybridA.all()) {
            Path dir = f.fixture.get().write(tmp.newFolder(f.hfArch + "-hf").toPath());
            buildAndClaim(dir, tmp.newFolder(f.hfArch + "-cache").toPath());
        }
    }

    /** Builds and claims the spec of one checkpoint dir. */
    private static void buildAndClaim(Path dir, Path cacheDir) throws IOException {
        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(cacheDir).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            com.eonmux.neuralfree.gguf.spec.ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src),
                    TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(),
                    ArchRegistry.get(spec.arch()).tensors());
        }
    }
}
