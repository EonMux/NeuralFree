package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * A synthetic llama whose matmul weights are ggml's per-element F8_E4M3FN (id 43) computes the same logits as
 * the same model with the codes dequantized to F32, and a weight's {@code .scale} sibling multiplies its
 * outputs like an element-wise fold. The fixture's rows are multiples of 32; rows of any length are covered by
 * the kernel tests of {@code F8E4M3FnTest}.
 */
public class CpuForwardF8E4M3FnTest {

    private static final int N_TOKENS = 8;
    private static final float TOL = 1e-4f;

    private static EngineConfig.Builder config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).build());
    }

    private static int[] tokens() {
        int[] t = new int[N_TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + i * 13;
        }
        return t;
    }

    /** The synthetic model with every matmul weight stored as F8_E4M3FN. */
    private static GgufTestWriter f8Model() {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.F8_E4M3FN).build());
    }

    /** Overwrites the float model's copy of {@code name} with {@code values}. */
    private static void setFloatWeight(GgufTestWriter w, String name, float[] values) {
        w.tensor(name, GGMLType.F32, w.tensorShape(name), GgufTestWriter.encodeFloats(GGMLType.F32, values));
    }

    /** The F8 codes of {@code name} decoded to floats. */
    private static float[] decoded(GgufTestWriter w, String name) {
        long[] ne = w.tensorShape(name);
        long n = 1;
        for (long d : ne) {
            n *= d;
        }
        return CpuBlockDecoders.dequantize(w.tensorData(name), GGMLType.F8_E4M3FN, n);
    }

    @Test
    public void f8E4M3FnWeightsComputeTheDequantizedFloatModel() throws IOException {
        GgufTestWriter w8 = f8Model();
        // The float reference stores exactly the values the codes decode to, so both models multiply the same
        // weight values and the logits agree to float rounding.
        GgufTestWriter w32 = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.F32).build());
        for (String name : w8.tensorNames()) {
            if (w8.tensorType(name) == GGMLType.F8_E4M3FN) {
                setFloatWeight(w32, name, decoded(w8, name));
            }
        }
        Path p8 = w8.writeTempFile();
        Path p32 = w32.writeTempFile();
        try (GGUFFileParser parser = new GGUFFileParser(p8.toString())) {
            assertEquals("the plan writes ggml's upstream id", GGMLType.F8_E4M3FN,
                    parser.getTensor("blk.0.attn_q.weight").type());
        }
        int[] tokens = tokens();
        float[] logits8;
        float[] logits32;
        try (GGUFModel m8 = new GGUFModel(p8.toString(), config().build());
             GGUFModel m32 = new GGUFModel(p32.toString(), config().build())) {
            logits8 = m8.forwardBackend().forwardRows(tokens, 0, new int[]{N_TOKENS - 1})[0];
            logits32 = m32.forwardBackend().forwardRows(tokens, 0, new int[]{N_TOKENS - 1})[0];
        }
        assertArrayEquals("F8_E4M3FN logits equal the dequantized model", logits32, logits8, TOL);
    }

    @Test
    public void theScaleSiblingMultipliesTheWeightOutputs() throws IOException {
        GgufTestWriter w8 = f8Model();
        w8.tensor("blk.0.attn_q.scale", GGMLType.F32, new long[]{1}, GgufTestWriter.encodeFloats(
                GGMLType.F32, new float[]{2.0f}));
        // The reference stores every weight as the values the codes decode to and folds the scale into the
        // float attn_q weight: the projection has no bias, so code * 2.0f produces the same matmul outputs as
        // the sibling-scaled run.
        GgufTestWriter w32 = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .weightType(GGMLType.F32).build());
        for (String name : w8.tensorNames()) {
            if (w8.tensorType(name) == GGMLType.F8_E4M3FN) {
                setFloatWeight(w32, name, decoded(w8, name));
            }
        }
        float[] v = decoded(w8, "blk.0.attn_q.weight");
        for (int i = 0; i < v.length; i++) {
            v[i] *= 2.0f;
        }
        setFloatWeight(w32, "blk.0.attn_q.weight", v);

        Path p8 = w8.writeTempFile();
        Path p32 = w32.writeTempFile();
        int[] tokens = tokens();
        try (GGUFModel m8 = new GGUFModel(p8.toString(), config().build());
             GGUFModel m32 = new GGUFModel(p32.toString(), config().build())) {
            float[] logits8 = m8.forwardBackend().forwardRows(tokens, 0, new int[]{N_TOKENS - 1})[0];
            float[] logits32 = m32.forwardBackend().forwardRows(tokens, 0, new int[]{N_TOKENS - 1})[0];
            assertArrayEquals("the .scale sibling rides the matmul", logits32, logits8, TOL);
        }
    }
}
