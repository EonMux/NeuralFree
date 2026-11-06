package com.eonmux.neuralfree.gguf.engine.draft;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.FakeTarget;
import com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.PlainDrafter;
import org.junit.Test;

import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.logits;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.request;
import static com.eonmux.neuralfree.gguf.engine.draft.SpeculativeDecoderFixtures.tokenizer;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@link VerifyMode} surface of the SpeculativeDecoder: mode-name parsing, the builder's validation of the verify
 * settings, and the greedy fallback that degenerates every lossless mode to MATCH's argmax rule.
 */
public class VerifyModeTest {

    /**
     * A greedy target chain degenerates REJECTION, BLOCK and FLY deferrals alike to the argmax rule: each mode's
     * trace equals MATCH's (llama.cpp common/sampling.cpp target-sample-match).
     */
    @Test
    public void greedyTargetChainsFallBackToTheArgmaxMatchRule() {
        int[][] drafts = {{2, 3}};
        float[][][] rows = {
                // the third row is never read: the mismatch at position 1 ends every mode's round
                {logits(0.0, 0.0, 1.0, 0.0), logits(1.0, 0.0, 0.0, 0.0), logits(0.0, 1.0, 0.0, 0.0)},
        };
        String[] expected = genTrace(VerifyMode.MATCH, drafts, rows);
        assertArrayEquals(expected, genTrace(VerifyMode.REJECTION, drafts, rows));
        assertArrayEquals(expected, genTrace(VerifyMode.BLOCK, drafts, rows));
        assertArrayEquals(expected, genTrace(VerifyMode.FLY, drafts, rows));
    }

    private static String[] genTrace(VerifyMode mode, int[][] drafts, float[][][] verifyRows) {
        FakeTarget target = new FakeTarget(false)
                .add(logits(0.0, 1.0, 0.0, 0.0))
                .add(verifyRows[0]);
        SpeculativeDecoder.Builder builder = SpeculativeDecoder.builder(target, new PlainDrafter(drafts),
                tokenizer()).draftMax(2).verifyMode(mode);
        if (mode == VerifyMode.FLY) {
            builder.flyEntropyThreshold(0.3);
        }
        GenerationResult r = builder.build().generate(new int[]{0}, request(3, SamplerConfig.greedyDecoding()),
                (TokenCallback) null);
        return new String[]{Arrays.toString(r.tokens()), Arrays.toString(target.rollbacks.toArray())};
    }

    @Test
    public void verifyModeParsesTheModeNames() {
        assertEquals(VerifyMode.MATCH, VerifyMode.parse(null));
        assertEquals(VerifyMode.MATCH, VerifyMode.parse("match"));
        assertEquals(VerifyMode.REJECTION, VerifyMode.parse("Rejection"));
        assertEquals(VerifyMode.BLOCK, VerifyMode.parse("block"));
        assertEquals(VerifyMode.FLY, VerifyMode.parse(" fly "));
        assertEquals(VerifyMode.SYNTHETIC, VerifyMode.parse("SYNTHETIC"));
        assertEquals("rejection", VerifyMode.REJECTION.toString());
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> VerifyMode.parse("strict"));
        assertTrue(unknown.getMessage().contains("match"));
    }

    @Test
    public void builderValidatesTheVerifySettings() {
        FakeTarget target = new FakeTarget(false);
        SpeculativeDecoder.Builder base = SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2}),
                tokenizer()).draftMax(2);
        SpeculativeDecoder dec = base.build();
        assertEquals(VerifyMode.MATCH, dec.verifyMode());
        assertTrue(dec.toString().contains("verify=match"));
        assertEquals(VerifyMode.REJECTION,
                SpeculativeDecoder.builder(target, new PlainDrafter(new int[]{2}), tokenizer())
                        .draftMax(2).verifyMode(VerifyMode.REJECTION).build().verifyMode());
        assertThrows(IllegalArgumentException.class, () -> SpeculativeDecoder
                .builder(target, new PlainDrafter(new int[]{2}), tokenizer()).draftMax(2)
                .verifyMode((VerifyMode) null).build());
        assertThrows(IllegalArgumentException.class, () -> base.verifyMode(null));
        assertThrows(IllegalArgumentException.class, () -> base.flyWindow(-1));
        assertThrows(IllegalArgumentException.class, () -> base.flyEntropyThreshold(-0.1));
        assertThrows(IllegalArgumentException.class, () -> base.flyEntropyTopK(0));
        assertThrows(IllegalArgumentException.class, () -> base.syntheticAcceptanceRates(1.0, 0.5, 0.6));
        assertThrows(IllegalArgumentException.class, () -> base.syntheticAcceptanceRates(1.2));
        assertThrows(IllegalArgumentException.class, () -> base.syntheticAcceptanceLength(0.5));
    }
}
