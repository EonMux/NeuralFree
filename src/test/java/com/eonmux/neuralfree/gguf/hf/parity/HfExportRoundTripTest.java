package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * PT-1, HfExportRoundTripTest: the {@code GgufWriter} NATIVE export of a Hugging Face checkpoint is a GGUF file whose
 * {@link ModelSpec} equals the spec of the checkpoint itself (PT-1 spec, HF.md GW-1). Both sides are resolved without
 * opening weights, so equality is decided by the metadata and tensor infos only, exactly what {@code GgufWriter}
 * copies (GW-1: the header and KV entries from the source's view, tensor infos backed by the copied payload).
 *
 * <p>The spec comparison is field-by-field: {@code ModelSpec} has no {@code equals} of its own, so this ports the
 * fields {@code ModelSpecBuilderTest} asserts. The NATIVE mode is asserted through the marker key
 * {@code neuralfree.native_skip}-free read-back of every tensor the checkpoint plans (GW-1 §4: NATIVE keeps every
 * extension tensor); a plain Llama checkpoint has none, so NATIVE and COMPAT write the same infos, and the test pins
 * the round trip precisely without an extension tensor in play.</p>
 */
public class HfExportRoundTripTest {

    private static final int LAYERS = 2;
    private static final int DIM = 64;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 16;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void nativeExportOfACheckpointResolvesToTheSameModelSpec() throws IOException {
        Path dir = SyntheticHf.llama(LAYERS, DIM, HEADS, KV_HEADS, VOCAB).write(tmp.newFolder("hf").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();

        Path out = tmp.getRoot().toPath().resolve("llama-roundtrip.gguf");
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null)) {
            assertEquals("the checkpoint opens as an HF source", "llama", hf.getString("general.architecture"));
            GgufWriter.write(hf, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }

        try (ModelSource exported = ModelSources.open(out.toString(), options, null)) {
            assertTrue(exported instanceof GGUFFileParser);
            assertEquals("llama", exported.getString("general.architecture"));
            assertSpecs(roundTripSpec(dir, options), specOf(exported));
        }
    }

    @Test
    public void theExportedFileKeepsEveryTensorInfo() throws IOException {
        Path dir = SyntheticHf.llama(LAYERS, DIM, HEADS, KV_HEADS, VOCAB).write(tmp.newFolder("hf").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();

        Path out = tmp.getRoot().toPath().resolve("llama-roundtrip-tensors.gguf");
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null)) {
            GgufWriter.write(hf, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null);
             GGUFFileParser g = new GGUFFileParser(out.toString())) {
            assertEquals(hf.getTensorCount(), g.getTensorCount());
            for (var info : hf.getTensors()) {
                assertEquals(info.name(), g.getTensor(info.name()).type(), info.type());
                var dims = g.getTensor(info.name()).dims();
                assertEquals(info.name(), dims.length, info.dims().length);
                for (int i = 0; i < dims.length; i++) {
                    assertEquals(info.name(), dims[i], info.dims()[i]);
                }
            }
        }
    }

    /** The spec of the exported GGUF file. */
    private static ModelSpec specOf(ModelSource source) {
        return ModelSpecBuilder.build(GgufKv.of(source), TensorCatalog.of(source), MemoryConfig.defaults());
    }

    /** Opens the HF path again (ModelSources.open twice, WP spec) and returns its spec. */
    private static ModelSpec roundTripSpec(Path dir, ModelSourceOptions options) throws IOException {
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null)) {
            return specOf(hf);
        }
    }

    /** Field-by-field {@link ModelSpec} equality, the fields {@code ModelSpecBuilderTest} pins. */
    private static void assertSpecs(ModelSpec expected, ModelSpec actual) {
        assertEquals(expected.arch(), actual.arch());
        assertEquals(expected.memoryKind(), actual.memoryKind());
        assertEquals(expected.nLayer(), actual.nLayer());
        assertEquals(expected.schedule().slots(), actual.schedule().slots());
        assertEquals(expected.head().tied(), actual.head().tied());
        assertEquals(expected.nextnLayers().size(), actual.nextnLayers().size());
        assertEquals(expected.encoderLayers().size(), actual.encoderLayers().size());
        assertNotNull(actual.embedding().tokenEmbd());
        assertEquals(Arch.LLAMA, expected.arch());
        assertFalse(actual.nextnLayers().stream().anyMatch(l -> l.il() >= 0));
        assertEquals(expected.layers().size(), actual.layers().size());
    }
}
