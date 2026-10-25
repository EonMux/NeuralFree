package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A two-layer Llama checkpoint opened through the registered Llama entry (no test overlay) has the GGUF tensor infos
 * of the same model written as a GGUF file by {@link SyntheticModels#llama()}: the same names, dimensions and types.
 * The order of the infos follows the converter (checkpoint files in order, names sorted within a file), so the lists
 * are compared by name.
 */
public class HfLlamaParityTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void llamaCheckpointHasTheTensorInfosOfTheGgufModel() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        assertEquals("the checkpoint's FFN is twice the width", 2 * c.nEmbd(), c.nFf());
        Path dir = SyntheticHf.llama(c.nLayer(), c.nEmbd(), c.nHead(), c.nHeadKv(), c.nVocab(), StDType.F32)
                .write(tmp.newFolder("hf").toPath());
        Path gguf = tmp.getRoot().toPath().resolve("llama.gguf");
        SyntheticModels.llama(c).writeTo(gguf);

        ModelSourceOptions options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource hf = ModelSources.open(dir.toString(), options, null);
             GGUFFileParser expected = new GGUFFileParser(gguf.toString())) {
            assertTrue(hf instanceof HfModelSource);
            assertEquals(Arch.LLAMA, ((HfModelSource) hf).context().arch());
            assertEquals(expected.getString(GgufKeys.GENERAL_ARCHITECTURE),
                    hf.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(byName(expected.getTensors()), byName(hf.getTensors()));
        }
    }

    /** {@code name -> "[dims] TYPE"} for every tensor info, sorted by name. */
    private static Map<String, String> byName(List<GGUFTensorInfo> infos) {
        Map<String, String> out = new TreeMap<>();
        for (GGUFTensorInfo t : infos) {
            String prev = out.put(t.name(), Arrays.toString(t.dims()) + " " + t.type());
            assertEquals("tensor " + t.name() + " is listed once", null, prev);
        }
        return out;
    }
}
