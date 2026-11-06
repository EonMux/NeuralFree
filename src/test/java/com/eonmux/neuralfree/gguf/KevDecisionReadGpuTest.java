package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The kev decision read on the device ({@link GGUFModelGPU#classify(int[], int[])}): the GPU model's decision scorer
 * is the same kev head, so its two option scores match the CPU model's on the same prompt (the dot products of the
 * last token with the {@code <|box_end|>} markers).
 */
public class KevDecisionReadGpuTest {

    private static EngineConfig cfg() {
        return EngineConfig.fromSystemProperties().toBuilder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(16)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A kev prompt of two options over the fixture's template. */
    private static String prompt() {
        Map<String, Object> inp = new LinkedHashMap<>();
        inp.put("type", "choice");
        inp.put("instructions", "Pick one");
        inp.put("state", "the sky is blue");
        List<Map<String, Object>> options = new ArrayList<>();
        for (String[] kv : new String[][] {{"red", "warm"}, {"blue", "cool"}}) {
            Map<String, Object> option = new LinkedHashMap<>();
            option.put("key", kv[0]);
            option.put("description", kv[1]);
            options.add(option);
        }
        inp.put("options", options);
        inp.put("images", List.of());
        return MiniJinja.render(SyntheticModels.KEV_TEMPLATE, inp);
    }

    @Test
    public void theDeviceScoresTheMarkersLikeTheCpuModel() throws Exception {
        Path file = SyntheticModels.decisionKev().writeTempFile();
        try (GGUFModel cpu = new GGUFModel(file.toString(), pagedCpu())) {
            int[] tokens = cpu.tokenizer().encode(prompt(), false, true);
            float[] expected = cpu.classify(tokens, null);
            GGUFModelGPU gpu = null;
            try {
                gpu = new GGUFModelGPU(file.toString(), null, cfg());
                assertArrayEquals(expected, gpu.classify(tokens, null), 1e-4f);
            } finally {
                try {
                    if (gpu != null) {
                        gpu.close();
                    }
                } finally {
                    Files.deleteIfExists(file);
                }
            }
        }
    }

    private static EngineConfig pagedCpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(128).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }
}
