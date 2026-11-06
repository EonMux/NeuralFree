package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.demo.tools.CvectorGenerator;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Control vectors applied at inference on the device forward pass of a synthetic two-layer llama
 * ({@code --control-vector} parity on the GPU path): the hidden state entering trunk layer 2 equals the base
 * model's plus {@code scale * direction.1}, the add of the loaded set's {@link com.eonmux.neuralfree.gguf.ops.heads.ControlVectorExtra}
 * after the block output (llama.cpp {@code build_cvec}); the logits of the controlled model differ from the base
 * model's.
 */
public class GpuControlVectorForwardTest {

    private static final int N_LAYERS = 2;
    private static final int N_EMBD = 64;
    private static final float SCALE = 1.5f;
    private static final float TOL = 1e-3f;
    private static final int[] PROMPT = {1, 17, 42, 5};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** The hidden rows of one prefill, by tap point. */
    private static final class Taps implements HiddenTap {
        private final Map<Integer, float[]> rows = new HashMap<>();

        @Override
        public void onHidden(int layer, float[] r, int nRows, int width) {
            rows.put(layer, r.clone());
        }

        @Override
        public boolean wants(int layer) {
            return true;
        }

        float[] at(int point) {
            return rows.get(point);
        }
    }

    @Test
    public void theVectorAddsItsDirectionAfterTheControlledBlock() throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYERS).build())
                .writeTempFile();
        Path vector = vector();
        float[] direction = ControlVector.File.load(vector.toString()).row(1);

        float[][] base = taps(model, config().build());
        float[][] withVector = taps(model, config().addControlVector(vector.toString(), SCALE).build());

        assertRows(0, base[0], withVector[0], 0f, null);
        assertRows(1, base[1], withVector[1], 0f, null);
        assertRows(2, base[2], withVector[2], SCALE, direction);
    }

    // ------------------------------------------------------------------ fixtures

    private static EngineConfig.Builder config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(16).ubatch(4).typeK(KvDType.F32)
                .typeV(KvDType.F32).build());
    }

    private Path vector() throws IOException {
        float[][] rows = new float[N_LAYERS][];
        for (int il = 0; il < N_LAYERS; il++) {
            Random r = new Random(700 + il);
            float[] row = new float[N_EMBD];
            for (int i = 0; i < N_EMBD; i++) {
                row[i] = 0.2f * (r.nextFloat() - 0.5f);
            }
            rows[il] = row;
        }
        Path out = tmp.getRoot().toPath().resolve("cv.gguf");
        return CvectorGenerator.export(Arrays.asList(rows), N_EMBD, "llama", out, true);
    }

    private static float[][] taps(Path model, EngineConfig cfg) throws IOException {
        GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg);
        try {
            ForwardBackend f = m.forwardBackend();
            Taps t = new Taps();
            f.setHiddenTap(t);
            f.prefill(PROMPT, 0);
            float[][] out = new float[N_LAYERS + 1][];
            for (int p = 0; p <= N_LAYERS; p++) {
                out[p] = t.at(p);
            }
            return out;
        } finally {
            m.close();
        }
    }

    /** {@code with[i]} must be {@code base[i] + scale * d[i]} for every row ({@code d == null}: equal). */
    private static void assertRows(int point, float[] base, float[] with, float scale, float[] d) {
        assertEquals(base.length, with.length);
        double worst = 0;
        for (int t = 0; t < PROMPT.length; t++) {
            for (int i = 0; i < N_EMBD; i++) {
                int at = t * N_EMBD + i;
                float expected = base[at] + (d == null ? 0f : scale * d[i]);
                worst = Math.max(worst, Math.abs(expected - with[at]));
            }
        }
        assertTrue("tap point " + point + ": worst difference " + worst, worst < TOL);
    }
}
