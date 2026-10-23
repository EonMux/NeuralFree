package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * Traces of {@link DiffusionGenerator} against llama.cpp examples/diffusion/diffusion.cpp on a 4-token vocabulary
 * {@code a b c <mask>} (mask id 3) with a scripted non-causal backend. Logit gaps of 100 make every draw
 * deterministic: the losing probabilities vanish against 1 in double precision.
 */
public class DiffusionGeneratorTest {

    private static final int MASK = 3;
    private static final float LOW = -1e4f;
    private static final float[] PEAKED = {100f, 0f, 0f, LOW};

    private static GGUFTokenizer tokenizer(boolean withMask) {
        List<Object> tokens = new ArrayList<>(Arrays.<Object>asList("a", "b", "c", "<mask>"));
        List<Object> types = new ArrayList<>(Arrays.<Object>asList(1, 1, 1, 3));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, "gpt-2");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>());
        if (withMask) {
            md.put("tokenizer.ggml.mask_token_id", MASK);
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static DiffusionConfig.Builder base() {
        return DiffusionConfig.builder().topK(0).topP(1f).temperature(1f).seed(7L)
                .shiftLogits(DiffusionConfig.ShiftLogits.NO_SHIFT);
    }

    /**
     * ENTROPY_BASED, timestep schedule with 2 steps over masks at positions 1..3. Step 0 transfers
     * {@code floor(3 * (1 - 0.5005 / 1)) = 1} row. Negative entropies: row 1 [100, 0, 0] ~ 0, row 2 [2, 2, 0]
     * (p = [0.4683, 0.4683, 0.0634]) -0.8854, row 3 [6, 0, 0] (p = [0.99507, 0.00247, 0.00247]) -0.0345; the least uncertain row 1 is unmasked with its sampled token 0.
     */
    @Test
    public void entropyStepUnmasksTheLeastUncertainRow() {
        FakeBackend fb = new FakeBackend((canvas, row) -> {
            switch (row) {
                case 1: return PEAKED.clone();
                case 2: return new float[]{2f, 2f, 0f, LOW};
                default: return new float[]{6f, 0f, 0f, LOW};
            }
        });
        DiffusionGenerator gen = new DiffusionGenerator(fb, tokenizer(true), 8, true);
        DiffusionConfig cfg = base().steps(2).timestepSchedule(1e-3f).maxLength(4)
                .algorithm(DiffusionConfig.Algorithm.ENTROPY_BASED).build();
        StringBuilder text = new StringBuilder();
        int[] out = gen.generate(new int[]{0}, cfg, (id, t) -> text.append(t), (step, total, canvas) -> step < 1);

        assertArrayEquals(new int[]{0, MASK, MASK}, out);
        assertEquals(1, fb.rows.size());
        assertArrayEquals(new int[]{1, 2, 3}, fb.rows.get(0));
        assertEquals("a", text.toString());
    }

    @Test
    public void shiftedLogitsReadThePreviousRow() {
        FakeBackend fb = new FakeBackend((canvas, row) -> PEAKED.clone());
        DiffusionGenerator gen = new DiffusionGenerator(fb, tokenizer(true), 8, true);
        DiffusionConfig cfg = base().shiftLogits(DiffusionConfig.ShiftLogits.FROM_MODEL).steps(1)
                .timestepSchedule(1e-3f).maxLength(4).build();
        assertArrayEquals(new int[]{0, 0, 0}, gen.generate(new int[]{1}, cfg, null));
        assertArrayEquals(new int[]{0, 1, 2}, fb.rows.get(0));
    }

    /** CFG scale 1: {@code u + 2 (c - u)} turns cond [100, 60] and uncond [200, 0] into [0, 120]. */
    @Test
    public void classifierFreeGuidanceCombinesTwoPasses() {
        BiFunction<int[], Integer, float[]> script = (canvas, row) -> canvas[0] == MASK
                ? new float[]{200f, 0f, 0f, LOW}
                : new float[]{100f, 60f, 0f, LOW};
        FakeBackend guided = new FakeBackend(script);
        DiffusionConfig cfg = base().steps(1).timestepSchedule(1e-3f).maxLength(3).cfgScale(1f).build();
        int[] out = new DiffusionGenerator(guided, tokenizer(true), 8, false).generate(new int[]{2}, cfg, null);
        assertArrayEquals(new int[]{1, 1}, out);
        assertEquals(2, guided.canvases.size());
        assertArrayEquals(new int[]{2, MASK, MASK}, guided.canvases.get(0));
        assertArrayEquals(new int[]{MASK, MASK, MASK}, guided.canvases.get(1));

        FakeBackend plain = new FakeBackend(script);
        int[] unguided = new DiffusionGenerator(plain, tokenizer(true), 8, false)
                .generate(new int[]{2}, cfg.toBuilder().cfgScale(0f).build(), null);
        assertArrayEquals(new int[]{0, 0}, unguided);
        assertEquals(1, plain.canvases.size());
    }

    /** Block length 2 over a canvas of 4 after a 1-token prompt: blocks [1, 3) and [3, 4), one step each. */
    @Test
    public void blockScheduleWalksBlocksAfterThePrompt() {
        FakeBackend fb = new FakeBackend((canvas, row) -> PEAKED.clone());
        DiffusionConfig cfg = base().steps(2).blockSchedule(2).maxLength(4).build();
        int[] out = new DiffusionGenerator(fb, tokenizer(true), 8, false).generate(new int[]{1}, cfg, null);
        assertArrayEquals(new int[]{0, 0, 0}, out);
        assertEquals(2, fb.rows.size());
        assertArrayEquals(new int[]{1, 2}, fb.rows.get(0));
        assertArrayEquals(new int[]{3}, fb.rows.get(1));
    }

    @Test
    public void scheduleArithmeticMatchesLlamaCpp() {
        assertEquals(1, DiffusionGenerator.transferCount(0, 4, 8, 1e-3f, null));
        assertEquals(2, DiffusionGenerator.transferCount(2, 4, 6, 1e-3f, null));
        assertEquals(5, DiffusionGenerator.transferCount(3, 4, 5, 1e-3f, null));
        assertArrayEquals(new int[]{2, 2, 1}, DiffusionGenerator.blockTransferCounts(5, 3));
        assertEquals(1, DiffusionGenerator.transferCount(2, 3, 1, 0f, new int[]{2, 2, 1}));
        assertEquals(0, DiffusionGenerator.logitsRow(0, true));
        assertEquals(4, DiffusionGenerator.logitsRow(5, true));
        assertEquals(5, DiffusionGenerator.logitsRow(5, false));
    }

    @Test
    public void algTempDrawsDistinctRows() {
        int[] picks = DiffusionGenerator.drawWithoutReplacement(new float[]{0f, 0f, 0f, 0f}, 4,
                new java.util.SplittableRandom(1));
        int[] sorted = picks.clone();
        Arrays.sort(sorted);
        assertArrayEquals(new int[]{0, 1, 2, 3}, sorted);
    }

    @Test
    public void missingMaskTokenIsRejected() {
        FakeBackend fb = new FakeBackend((canvas, row) -> PEAKED.clone());
        DiffusionGenerator gen = new DiffusionGenerator(fb, tokenizer(false), 8, true);
        assertThrows(IllegalStateException.class, () -> gen.generate(new int[]{0}, base().maxLength(4).build(), null));
        assertArrayEquals(new int[]{0, 0, 0},
                gen.generate(new int[]{0}, base().steps(1).maxLength(4).maskTokenId(MASK).build(), null));
    }

    @Test
    public void canvasMustFitTheUbatchAndLeaveRoom() {
        FakeBackend fb = new FakeBackend((canvas, row) -> PEAKED.clone());
        DiffusionGenerator gen = new DiffusionGenerator(fb, tokenizer(true), 8, true);
        assertThrows(IllegalArgumentException.class, () -> gen.generate(new int[]{0}, base().maxLength(16).build(), null));
        assertThrows(IllegalArgumentException.class, () -> gen.generate(new int[]{0, 1, 2, 0}, base().maxLength(4).build(), null));
        assertThrows(IllegalArgumentException.class, () -> gen.generate(new int[0], base().build(), null));
    }

    /** Non-causal backend that answers each requested row from a script and records the calls. */
    private static final class FakeBackend implements ForwardBackend {

        private final BiFunction<int[], Integer, float[]> script;
        final List<int[]> canvases = new ArrayList<>();
        final List<int[]> rows = new ArrayList<>();

        FakeBackend(BiFunction<int[], Integer, float[]> script) {
            this.script = script;
        }

        @Override
        public float[][] forwardAll(int[] tokens, int[] outputRows) {
            canvases.add(tokens.clone());
            rows.add(outputRows.clone());
            float[][] out = new float[outputRows.length][];
            for (int k = 0; k < outputRows.length; k++) {
                out[k] = script.apply(tokens, outputRows[k]);
            }
            return out;
        }

        @Override
        public float[] prefill(int[] tokens, int pos0) {
            throw new UnsupportedOperationException("not used by diffusion");
        }

        @Override
        public float[] decode(int token, int pos) {
            throw new UnsupportedOperationException("not used by diffusion");
        }

        @Override
        public float[][] forwardRows(int[] tokens, int pos0, int[] outputRows) {
            throw new UnsupportedOperationException("not used by diffusion");
        }

        @Override
        public float[][] hiddenRows(int[] tokens, int[] tokenTypes) {
            throw new UnsupportedOperationException("not used by diffusion");
        }

        @Override
        public void reset() {
        }

        @Override
        public boolean canRollback(int nTokens) {
            return nTokens == 0;
        }

        @Override
        public void rollback(int nTokens) {
            throw new UnsupportedOperationException("not used by diffusion");
        }

        @Override
        public void shiftContext(int nKeep, int nDiscard) {
            throw new UnsupportedOperationException("not used by diffusion");
        }

        @Override
        public ModelMemory memory() {
            return null;
        }

        @Override
        public void setHiddenTap(HiddenTap tap) {
        }

        @Override
        public void close() {
        }
    }
}
