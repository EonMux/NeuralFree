package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.demo.tools.CvectorGenerator;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Control vectors applied at inference on the host forward pass of a synthetic four-layer llama
 * ({@code --control-vector} parity): the hidden state entering the first trunk layer after a controlled block
 * equals the base model's plus {@code scale * direction.<il>}, the add llama.cpp's {@code build_cvec} performs on
 * the layer output (llama-adapter.cpp apply_to, models/llama.cpp:223). Later taps carry the add through the
 * downstream layers' nonlinear blocks, so they assert the propagation instead of a linear sum. Layer 0 and the top
 * graph layer never receive an add (llama.cpp {@code common/common.cpp:1484-1485} defaults, llama-adapter.cpp
 * apply), the layer range restricts the adds, several vectors sum, and a vector whose direction rows are not exactly
 * the model's {@code n_embd} wide is refused, as llama.cpp's {@code llama_adapter_cvec::apply} refuses it.
 */
public class CpuControlVectorForwardTest {

    private static final int N_LAYERS = 4;
    private static final int N_EMBD = 64;
    private static final float SCALE = 1.75f;
    private static final float TOL = 1e-4f;
    private static final int[] PROMPT = {1, 17, 42, 5, 99, 260};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** The hidden rows of one prefill, by tap point (the stream entering trunk layer {@code point}). */
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
    public void theVectorAddsItsDirectionAfterTheFirstControlledBlock() throws IOException {
        Path model = model();
        Path vector = vector(700);
        float[] d1 = direction(vector, 1);

        float[][] base = taps(model, config().build());
        float[][] withVector = taps(model, config().addControlVector(vector.toString(), SCALE).build());

        // Layer 0 gets no add (llama.cpp never reads a layer-0 row), so taps 0 and 1 stay the base stream.
        assertEqualStream(0, base[0], withVector[0]);
        assertEqualStream(1, base[1], withVector[1]);
        // direction.1 is the add of graph layer 1: it reaches tap 2 unscaled by any downstream block.
        assertAddedStream(2, base[2], withVector[2], SCALE, d1);
        // From there the modified stream runs through block 2, so tap 3 must carry the change but not linearly.
        assertTrue(maxAbsDiff(base[3], withVector[3]) > 100 * TOL);
    }

    @Test
    public void theLayerRangeRestrictsTheAdds() throws IOException {
        Path model = model();
        Path vector = vector(700);
        float[] d2 = direction(vector, 2);

        float[][] base = taps(model, config().build());
        EngineConfig cfg = config().addControlVector(vector.toString(), SCALE)
                .controlVectorLayerRange(2, 3).build();
        float[][] withVector = taps(model, cfg);

        assertEqualStream(0, base[0], withVector[0]);
        assertEqualStream(1, base[1], withVector[1]);
        assertEqualStream(2, base[2], withVector[2]);
        // With the range starting at layer 2, direction.2 is the first add and reaches tap 3 directly.
        assertAddedStream(3, base[3], withVector[3], SCALE, d2);
        assertTrue(maxAbsDiff(base[4], withVector[4]) > 100 * TOL);
    }

    @Test
    public void severalVectorsSumTheirScaledRows() throws IOException {
        Path model = model();
        Path first = vector(700);
        Path second = vector(900);
        float[] a1 = direction(first, 1);
        float[] b1 = direction(second, 1);

        float[][] base = taps(model, config().build());
        EngineConfig cfg = config().addControlVector(first.toString(), 0.5f)
                .addControlVector(second.toString(), -2.0f).build();
        float[][] withVectors = taps(model, cfg);

        assertEqualStream(1, base[1], withVectors[1]);
        assertAddedStream(2, base[2], withVectors[2], 0.5f, a1, -2.0f, b1);
    }

    @Test
    public void theTopLayerDirectionIsNeverApplied() throws IOException {
        Path model = model();
        // Two files whose rows 1..3 are identical and whose row 4 differs: the top graph layer of a 4-layer model
        // is 3, so direction.4 must never reach the stream and every tap stays identical.
        float[][] keptRows = rows(700);
        Path kept = vector(keptRows, "kept");
        Path changed = vector(new float[][]{keptRows[0], keptRows[1], keptRows[2], row(953, N_EMBD)}, "changed");

        float[][] first = taps(model, config().addControlVector(kept.toString(), SCALE).build());
        float[][] second = taps(model, config().addControlVector(changed.toString(), SCALE).build());
        for (int p = 0; p <= N_LAYERS; p++) {
            assertEqualStream(p, first[p], second[p]);
        }
    }

    @Test
    public void aVectorWiderThanTheModelIsRefusedByName() throws IOException {
        Path model = model();
        Path wide = tmp.getRoot().toPath().resolve("wide.gguf");
        CvectorGenerator.export(Arrays.asList(row(1, 129)), 129, "llama", wide, true);
        EngineConfig cfg = config().addControlVector(wide.toString(), SCALE).build();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> taps(model, cfg));
        String message = e.getMessage();
        assertTrue(message, message.contains("n_embd does not match model"));
        assertTrue(message, message.contains(wide.toString()));
        assertTrue(message, message.contains("129"));
        assertTrue(message, message.contains(String.valueOf(N_EMBD)));
    }

    @Test
    public void aVectorNarrowerThanTheModelIsRefusedByName() throws IOException {
        Path model = model();
        Path narrow = tmp.getRoot().toPath().resolve("narrow.gguf");
        CvectorGenerator.export(Arrays.asList(row(1, 32)), 32, "llama", narrow, true);
        EngineConfig cfg = config().addControlVector(narrow.toString(), SCALE).build();
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> taps(model, cfg));
        String message = e.getMessage();
        assertTrue(message, message.contains("n_embd does not match model"));
        assertTrue(message, message.contains(narrow.toString()));
        assertTrue(message, message.contains("32"));
        assertTrue(message, message.contains(String.valueOf(N_EMBD)));
        // The message names the layers that would have received the add (llama.cpp's default range).
        assertTrue(message, message.contains("1.." + N_LAYERS));
    }

    // ------------------------------------------------------------------ fixtures

    private static EngineConfig.Builder config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(32).ubatch(4).typeK(KvDType.F32)
                .typeV(KvDType.F32).build());
    }

    private Path model() throws IOException {
        return SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(N_LAYERS).build())
                .writeTempFile();
    }

    /** A control vector of {@code N_LAYERS} small rows, one file per seed. */
    private Path vector(long seed) throws IOException {
        return vector(rows(seed), "cv-" + seed);
    }

    private static float[][] rows(long seed) {
        float[][] out = new float[N_LAYERS][];
        for (int il = 0; il < N_LAYERS; il++) {
            out[il] = row(seed + il, N_EMBD);
        }
        return out;
    }

    private Path vector(float[][] rows, String name) throws IOException {
        Path out = tmp.getRoot().toPath().resolve(name + ".gguf");
        return CvectorGenerator.export(Arrays.asList(rows), N_EMBD, "llama", out, true);
    }

    private static float[] row(long seed, int n) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = 0.2f * (r.nextFloat() - 0.5f);
        }
        return out;
    }

    private static float[] direction(Path vector, int il) throws IOException {
        return ControlVector.File.load(vector.toString()).row(il);
    }

    /** Prefill tap rows 0..nLayer of one fresh model run. */
    private static float[][] taps(Path model, EngineConfig cfg) throws IOException {
        GGUFModel m = new GGUFModel(model.toString(), cfg);
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

    /** Every row of the two streams must hold the same floats. */
    private static void assertEqualStream(int point, float[] base, float[] with) {
        assertArrayEquals("tap point " + point, base, with, 0f);
    }

    /** {@code with[i]} must be {@code base[i] + scale * v[i] + scale2 * v2[i] ...} for every row. */
    private static void assertAddedStream(int point, float[] base, float[] with, float scale, float[] v) {
        assertAddedStream(point, base, with, new float[][]{{scale}, v});
    }

    private static void assertAddedStream(int point, float[] base, float[] with, float scale, float[] v,
                                          float scale2, float[] v2) {
        assertAddedStream(point, base, with, new float[][]{{scale}, v, {scale2}, v2});
    }

    /** {@code groups} alternates {@code {scale}} markers and direction rows. */
    private static void assertAddedStream(int point, float[] base, float[] with, float[][] groups) {
        double worst = 0;
        for (int t = 0; t < PROMPT.length; t++) {
            for (int i = 0; i < N_EMBD; i++) {
                int at = t * N_EMBD + i;
                float expected = base[at];
                for (int g = 0; g < groups.length; g += 2) {
                    expected += groups[g][0] * groups[g + 1][i];
                }
                worst = Math.max(worst, Math.abs(expected - with[at]));
            }
        }
        assertTrue("tap point " + point + ": worst difference " + worst, worst < TOL);
    }

    private static double maxAbsDiff(float[] a, float[] b) {
        double m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
