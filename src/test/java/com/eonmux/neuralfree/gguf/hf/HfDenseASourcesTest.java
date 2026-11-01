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
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDenseFixturesA;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertTrue;

/**
 * The HK-D2 part A fixtures ({@code Gemma3Model}, {@code Gemma3NModel}, {@code SenseNovaU1Model}, {@code
 * Qwen2VLModel}, {@code Qwen3VLTextModel}, {@code Qwen3ASRTextModel}) build through the registered entries:
 * {@link ModelSpecBuilder} succeeds and every catalog tensor is claimed ({@code TensorClaims}), which is the
 * done-getting-tensors check of the converter's tensor plan.
 *
 * <p>Excluded from the fixtures because their vocabulary step needs a file the GPT-2 {@code tokenizer.json} of
 * {@code com.eonmux.neuralfree.gguf.testutil.SyntheticHf} cannot stand in for: {@code GemmaModel} ({@code
 * GemmaForCausalLM}) and {@code Gemma2Model}
 * ({@code Gemma2ForCausalLM}), whose {@code set_vocab} reads a SentencePiece {@code tokenizer.model}; and {@code
 * QwenModel} ({@code QWenLMHeadModel}), whose {@code set_vocab} reads a tiktoken vocabulary file.
 */
public class HfDenseASourcesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void everyPartAFixtureBuildsASpecClaimingItsTensors() throws IOException {
        for (HfDenseFixturesA.Named f : HfDenseFixturesA.all()) {
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
