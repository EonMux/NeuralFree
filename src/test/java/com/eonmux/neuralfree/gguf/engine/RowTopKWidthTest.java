package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The device readback width the engines cone a sampled row into is the widest readout {@code LLM_Row_TopK} serves in
 * its radix-select form on every backend that builds it: a wider cone would run one work-group argmax round per
 * candidate, a narrower cap would read whole rows for chains the readout serves.
 */
public class RowTopKWidthTest {

    private static final Pattern MAX_K = Pattern.compile("#define\\s+NF_ROW_TOPK_MAX_K\\s+(\\d+)");

    @Test
    public void theDeviceCandidateCapIsTheReadoutsRadixWidthOnEveryDialect() throws IOException {
        assertEquals(RowTopK.MAX_DEVICE_CANDIDATES, maxK("kernels/llm/sampling_topk.cl"));
        assertEquals(RowTopK.MAX_DEVICE_CANDIDATES, maxK("shaders/llm/row_topk.comp"));
    }

    @Test
    public void theCapCoversTheUsualTopKSettings() {
        assertTrue("top_k 40 (llama.cpp), 50 (HF), 100 and 1000 fit one readback",
                RowTopK.MAX_DEVICE_CANDIDATES >= 1000);
    }

    /** The cap a kernel source under {@code src/main/resources} defines (the shader sources are not packaged). */
    private static int maxK(String relative) throws IOException {
        Path path = Path.of("src", "main", "resources").resolve(relative);
        assertTrue("missing source file " + path.toAbsolutePath() + " (run from the project root)",
                Files.exists(path));
        Matcher m = MAX_K.matcher(Files.readString(path, StandardCharsets.UTF_8));
        assertTrue(relative + " defines NF_ROW_TOPK_MAX_K", m.find());
        return Integer.parseInt(m.group(1));
    }
}
