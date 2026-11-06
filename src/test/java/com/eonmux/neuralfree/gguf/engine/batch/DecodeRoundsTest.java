package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A broken drafter in the batch engine's decode rounds ({@link DecodeRounds}): a proposal that is null, longer
 * than asked or holds a negative id fails the sequence it was drafted for with an {@link IllegalStateException}
 * naming the defect, before anything is written — the sequence beside it keeps decoding to the same tokens as a
 * plain single-sequence run, and the failed slot's drafter state is reset.
 */
public class DecodeRoundsTest {

    private static final int N_NEW = 6;

    private static EngineConfig config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 17) % 200;
        }
        return t;
    }

    /** The reference: the prompt through the single-sequence GenerationLoop. */
    private static GenerationResult reference(Path model, int[] prompt) throws IOException {
        EngineConfig cfg = config(1);
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
        }
    }

    @Test
    public void aBrokenProposalFailsItsOwnSequenceAlone() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] healthyPrompt = tokens(2, 9);
        GenerationResult ref = reference(model, healthyPrompt);
        String[][] cases = {
                {"null", "returned no proposal array"},
                {"long", "proposed 4 tokens, at most 3 were asked"},
                {"negative", "proposed negative token id -7"},
        };
        for (String[] c : cases) {
            BrokenDrafter drafter = new BrokenDrafter(c[0]);
            EngineConfig cfg = config(2);
            try (GGUFModel m = new GGUFModel(model.toString(), cfg);
                 BatchEngine engine = BatchEngine.forModel(m, cfg, drafter, new BatchSpec(3, 1))) {
                BatchSequence broken = engine.submit(tokens(1, 7), request(),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                BatchSequence healthy = engine.submit(healthyPrompt, request(),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine.drain();

                assertEquals(c[0], BatchSequence.State.FAILED, broken.state());
                assertTrue(c[0], broken.error() instanceof IllegalStateException);
                assertTrue(c[0] + ": " + broken.error().getMessage(),
                        broken.error().getMessage().contains(c[1]));
                assertEquals(c[0] + ": " + healthy.error(), BatchSequence.State.DONE, healthy.state());
                assertArrayEquals(c[0], ref.tokens(), healthy.result().tokens());
                assertEquals(c[0], ref.text(), healthy.result().text());
                assertTrue(c[0] + ": the failed slot's drafter state resets", drafter.resets.contains(0));
                BatchStats stats = engine.stats();
                assertEquals(c[0], 1, stats.failed());
                assertEquals(c[0], 0, stats.specRounds());
            }
        }
    }

    /**
     * Breaks every proposal for sequence 0 the way {@code mode} names, and proposes nothing for the others (they
     * decode plain).
     */
    private static final class BrokenDrafter implements BatchDrafter {

        private final String mode;
        final List<Integer> resets = new ArrayList<>();

        BrokenDrafter(String mode) {
            this.mode = mode;
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            if (seq != 0) {
                return new int[0];
            }
            switch (mode) {
                case "null":
                    return null;
                case "long":
                    return new int[n + 1];
                default:
                    return new int[] {5, -7};
            }
        }

        @Override
        public void accept(int seq, int nAccepted) {
        }

        @Override
        public void reset(int seq) {
            resets.add(seq);
        }
    }
}
