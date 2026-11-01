package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDenseFixturesC;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertTrue;

/**
 * The HK-D2 part C fixtures ({@code HunYuanModel}, {@code HunyuanVLTextModel}, {@code Mistral3Model} on the Ministral 3
 * text branch, {@code VoxtralRealtimeModel}) build through the registered entries: {@link ModelSpecBuilder} succeeds
 * and every catalog tensor is claimed ({@code TensorClaims}), which is the done-getting-tensors check of the
 * converter's tensor plan.
 *
 * <p>Excluded from the fixtures because their vocabulary step needs a file the GPT-2 {@code tokenizer.json} of
 * {@code com.eonmux.neuralfree.gguf.testutil.SyntheticHf} cannot stand in for: {@code Ernie4_5Model} ({@code
 * Ernie4_5ForCausalLM}) and {@code PaddleOCRModel} ({@code PaddleOCRVLForConditionalGeneration}), whose {@code
 * set_vocab} reads a SentencePiece {@code tokenizer.model}.
 */
public class HfDenseCSourcesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyPartCFixtureBuildsASpecClaimingItsTensors() throws IOException {
        for (HfDenseFixturesC.Named f : HfDenseFixturesC.all()) {
            Path dir = f.fixture.get().write(tmp.newFolder(f.hfArch + "-hf").toPath());
            buildAndClaim(dir, tmp.newFolder(f.hfArch + "-cache").toPath());
        }
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
