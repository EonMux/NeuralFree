package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * A generic GGUF LoRA adapter on the host forward pass of a synthetic two-layer llama ({@code --lora} parity): the
 * adapter adapts {@code attn_q/k/v/output} and {@code ffn_gate/up/down} of every layer, {@code output.weight} and
 * {@code token_embd.weight}. Since {@code W x + s * B (A x) = (W + s * B A) x} (build_lora_mm, llama-graph.cpp:1648-1664)
 * and {@code E[id] + s * B A[id] = (E + s * A B^T)[id]} (llama-graph.cpp:2694-2709), the model with the adapter must
 * produce the logits of the same model with every adapted tensor merged, over a prefill split into several ubatches
 * and a decode step; the adapter must also change the logits of the base model.
 */
public class CpuLoraForwardTest {

    private static final int RANK = 4;
    private static final float ALPHA = 8.0f;
    private static final float USER_SCALE = 0.75f;
    /** {@code scale = user * alpha / rank} (llama-adapter.h:53-57). */
    private static final float SCALE = USER_SCALE * ALPHA / RANK;
    private static final float TOL = 1e-3f;
    private static final String EMBD = "token_embd.weight";
    private static final String[] LAYER_PROJECTIONS = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
            "ffn_up", "ffn_down"};
    private static final int[] PROMPT = {1, 17, 42, 5, 99, 260};
    private static final int NEXT = 77;

    /** One adapted tensor: {@code A} and {@code B} row-major as the pair layout of the adapter file. */
    private static final class Lora {
        final String name;
        final long[] ne;
        final float[] a;
        final float[] b;

        Lora(String name, long[] ne, float[] a, float[] b) {
            this.name = name;
            this.ne = ne;
            this.a = a;
            this.b = b;
        }

        boolean embedding() {
            return EMBD.equals(name);
        }
    }

    @Test
    public void theAdapterGivesTheLogitsOfTheMergedModel() throws IOException {
        GgufTestWriter baseWriter = SyntheticModels.llama();
        Path base = baseWriter.writeTempFile();
        List<Lora> loras = loras(baseWriter);
        Path adapter = adapter(loras).writeTempFile();
        Path merged = merged(base, loras).writeTempFile();

        float[][] withAdapter = logits(base, config().addLora(adapter.toString(), USER_SCALE).build());
        float[][] mergedLogits = logits(merged, config().build());
        float[][] baseLogits = logits(base, config().build());

        for (int i = 0; i < 2; i++) {
            assertArrayEquals("step " + i, mergedLogits[i], withAdapter[i], TOL);
            assertTrue("the adapter must change the logits of step " + i,
                    maxAbsDiff(baseLogits[i], withAdapter[i]) > 100 * TOL);
        }
    }

    private static EngineConfig.Builder config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(32).ubatch(4).typeK(KvDType.F32)
                .typeV(KvDType.F32).build());
    }

    /** Prefill logits of {@link #PROMPT} (two ubatches) and decode logits of {@link #NEXT}. */
    private static float[][] logits(Path model, EngineConfig cfg) throws IOException {
        GGUFModel m = new GGUFModel(model.toString(), cfg);
        try {
            ForwardBackend f = m.forwardBackend();
            float[] prefill = f.prefill(PROMPT, 0).clone();
            float[] decode = f.decode(NEXT, PROMPT.length).clone();
            return new float[][]{prefill, decode};
        } finally {
            m.close();
        }
    }

    private static List<Lora> loras(GgufTestWriter w) {
        List<String> names = new ArrayList<>();
        for (int il = 0; il < SyntheticModels.Config.llama().nLayer(); il++) {
            for (String p : LAYER_PROJECTIONS) {
                names.add("blk." + il + "." + p + ".weight");
            }
        }
        names.add("output.weight");
        names.add(EMBD);
        List<Lora> out = new ArrayList<>();
        long seed = 1000;
        for (String n : names) {
            long[] ne = w.tensorShape(n);
            int rows = Math.toIntExact(ne[1]);
            int cols = Math.toIntExact(ne[0]);
            if (EMBD.equals(n)) {
                out.add(new Lora(n, ne, random(seed++, rows * RANK), random(seed++, cols * RANK)));
            } else {
                out.add(new Lora(n, ne, random(seed++, RANK * cols), random(seed++, rows * RANK)));
            }
        }
        return out;
    }

    /**
     * The adapter file (llama-adapter.cpp:203-375): a projection {@code W {in, out}} takes {@code A {in, r}} and
     * {@code B {r, out}}, the token table {@code E {n_embd, n_vocab}} takes {@code A {r, n_vocab}} and
     * {@code B {r, n_embd}}.
     */
    private static GgufTestWriter adapter(List<Lora> loras) {
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", ALPHA);
        for (Lora l : loras) {
            if (l.embedding()) {
                w.tensorF32(l.name + ".lora_a", new long[]{RANK, l.ne[1]}, l.a);
                w.tensorF32(l.name + ".lora_b", new long[]{RANK, l.ne[0]}, l.b);
            } else {
                w.tensorF32(l.name + ".lora_a", new long[]{l.ne[0], RANK}, l.a);
                w.tensorF32(l.name + ".lora_b", new long[]{RANK, l.ne[1]}, l.b);
            }
        }
        return w;
    }

    /** The base model with every adapted tensor replaced by its merge with {@code SCALE * B A}. */
    private static GgufTestWriter merged(Path base, List<Lora> loras) throws IOException {
        Map<String, float[]> values = new HashMap<>();
        GGUFFileParser parser = new GGUFFileParser(base.toString());
        try (CpuWeights weights = CpuWeights.open(parser)) {
            for (Lora l : loras) {
                CpuTensorRef t = weights.find(l.name);
                float[] v = new float[Math.toIntExact(t.rows() * t.cols())];
                t.decodeRows(0L, Math.toIntExact(t.rows()), v, 0);
                values.put(l.name, v);
            }
        } finally {
            parser.close();
        }
        GgufTestWriter w = SyntheticModels.llama();
        for (Lora l : loras) {
            float[] v = values.get(l.name);
            int rows = Math.toIntExact(l.ne[1]);
            int cols = Math.toIntExact(l.ne[0]);
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    double acc = 0.0;
                    for (int r = 0; r < RANK; r++) {
                        acc += l.embedding()
                                ? (double) l.b[col * RANK + r] * l.a[row * RANK + r]
                                : (double) l.b[row * RANK + r] * l.a[r * cols + col];
                    }
                    v[row * cols + col] += (float) (SCALE * acc);
                }
            }
            w.removeTensor(l.name);
            w.tensorF32(l.name, l.ne.clone(), v);
        }
        return w;
    }

    private static float[] random(long seed, int n) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 0.2f * (r.nextFloat() - 0.5f);
        }
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
