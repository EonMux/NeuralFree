package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerPrefilterTest;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.SplitReadout;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A batch engine's {@code MATCH} verify rounds sample their rows from the device cone, as single-row decodes do: over a
 * backend that reports a device top-k readout (and, for a chain with penalties, a device prefilter), a speculating
 * sequence whose chain keeps its candidates inside the cone reads no row back whole — every verify row's candidates
 * come from the step's one readout, a prefiltered chain's row {@code i} edited over the history plus the round's
 * drafts before it — and every sequence emits exactly the tokens it emits with every row read whole. A verifier
 * mode other than {@code MATCH} keeps reading its verify rows whole.
 */
public class BatchConedVerifyTest {

    private static final int N_NEW = 12;

    private static Path model;

    @BeforeClass
    public static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @AfterClass
    public static void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    /** The batch calls the engine made through the proxied backend. */
    private static final class Calls {
        /** Output rows of whole-row forwards (a prompt chunk that completes no prompt reads none). */
        int wholeRows;
        int topK;
        int splitRows;
        int prefiltered;
    }

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(1024).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(3).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(mc).build();
    }

    private static int[] prompt(int seed) {
        int[] t = new int[10 + seed];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /**
     * Proposes the sequence's last token and the two after it in the vocabulary, every round: a fixed draft some of
     * whose rows the target confirms, so every decode step after a prompt is a verify round.
     */
    private static final class RepeatDrafter implements BatchDrafter {
        @Override
        public int[] draft(int seq, int lastToken, int n) {
            int[] out = new int[Math.min(n, 3)];
            for (int i = 0; i < out.length; i++) {
                out[i] = 4 + (lastToken + i) % 250;
            }
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
        }

        @Override
        public void close() {
        }
    }

    /** Proposes {@code token} at every draft position, every round. */
    private static final class FixedDrafter implements BatchDrafter {
        private final int token;

        FixedDrafter(int token) {
            this.token = token;
        }

        @Override
        public int[] draft(int seq, int lastToken, int n) {
            int[] out = new int[Math.min(n, 3)];
            java.util.Arrays.fill(out, token);
            return out;
        }

        @Override
        public void accept(int seq, int nAccepted) {
        }

        @Override
        public void close() {
        }
    }

    /**
     * {@code (4, 5, v)} triples over v = 6..105 ending on the pair (4, 5): the fixture's unpenalized greedy
     * continuation is a run of token 86 ({@code BatchEngineSpeculativeTest}).
     */
    private static int[] runOf86Prompt() {
        int[] prompt = new int[3 * 100 + 2];
        int w = 0;
        for (int v = 6; v <= 105; v++) {
            prompt[w++] = 4;
            prompt[w++] = 5;
            prompt[w++] = v;
        }
        prompt[w++] = 4;
        prompt[w] = 5;
        return prompt;
    }

    /**
     * {@code model} whose forward backend reports a device top-k readout and a device prefilter (applied on the host
     * by the reference edits) and counts the engine's batch calls.
     */
    private static EngineModel deviceTopK(EngineModel engineModel, Calls calls) {
        ForwardBackend real = engineModel.forwardBackend();
        InvocationHandler backend = (proxy, method, args) -> {
            switch (method.getName()) {
                case "batchTopKOnDevice":
                case "batchPrefilterOnDevice":
                    return true;
                case "forwardBatch":
                    calls.wholeRows += ((ForwardBatch) args[0]).outputRows().length;
                    break;
                case "forwardBatchTopK":
                    calls.topK++;
                    if (args.length == 3 && args[2] != null && !((LogitPrefilter) args[2]).isEmpty()) {
                        calls.prefiltered++;
                        float[][] rows = real.forwardBatch((ForwardBatch) args[0]);
                        return RowTopK.of(edited(rows, (LogitPrefilter) args[2]), (Integer) args[1]);
                    }
                    break;
                case "forwardBatchSplit":
                    boolean[] full = (boolean[]) args[1];
                    for (boolean f : full) {
                        calls.splitRows += f ? 1 : 0;
                    }
                    LogitPrefilter prefilter = (LogitPrefilter) args[3];
                    if (prefilter != null && !prefilter.isEmpty()) {
                        calls.prefiltered++;
                        float[][] rows = real.forwardBatch((ForwardBatch) args[0]);
                        float[][] whole = new float[rows.length][];
                        for (int i = 0; i < rows.length; i++) {
                            whole[i] = full[i] ? rows[i] : null;
                        }
                        return new SplitReadout(RowTopK.of(edited(rows, prefilter), (Integer) args[2]), whole);
                    }
                    break;
                default:
                    break;
            }
            return invoke(method, real, args);
        };
        ForwardBackend wrapped = (ForwardBackend) Proxy.newProxyInstance(ForwardBackend.class.getClassLoader(),
                new Class<?>[]{ForwardBackend.class}, backend);
        InvocationHandler engine = (proxy, method, args) -> method.getName().equals("forwardBackend")
                && method.getParameterCount() == 0 ? wrapped : invoke(method, engineModel, args);
        return (EngineModel) Proxy.newProxyInstance(EngineModel.class.getClassLoader(),
                new Class<?>[]{EngineModel.class}, engine);
    }

    private static float[][] edited(float[][] rows, LogitPrefilter prefilter) {
        float[][] out = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            out[i] = SamplerPrefilterTest.applyPrefilter(rows[i], prefilter.row(i));
        }
        return out;
    }

    private static Object invoke(Method method, Object target, Object[] args) throws Throwable {
        try {
            return method.invoke(target, args);
        } catch (InvocationTargetException e) {
            throw e.getCause();
        }
    }

    private static final class Run {
        final List<GenerationResult> results = new ArrayList<>();
        BatchStats stats;
    }

    private static Run run(List<GenerationRequest> requests, VerifyMode mode, Calls calls) throws IOException {
        return run(requests, mode, calls, new RepeatDrafter(), false);
    }

    private static Run run(List<GenerationRequest> requests, VerifyMode mode, Calls calls, BatchDrafter drafter,
                           boolean runOf86) throws IOException {
        EngineConfig cfg = config();
        BatchSpec spec = BatchSpec.of(3, 0, SpeculativeOptions.builder().verifyMode(mode).build());
        Run out = new Run();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(calls == null ? m : deviceTopK(m, calls), cfg,
                     drafter, spec)) {
            List<BatchSequence> running = new ArrayList<>();
            for (int i = 0; i < requests.size(); i++) {
                running.add(engine.submit(runOf86 ? runOf86Prompt() : prompt(i), requests.get(i),
                        (TokenCallback) null));
            }
            engine.drain();
            for (BatchSequence s : running) {
                assertEquals(String.valueOf(s.error()), BatchSequence.State.DONE, s.state());
                out.results.add(s.result());
            }
            out.stats = engine.stats();
        }
        return out;
    }

    private static List<GenerationRequest> requests(SamplerConfig.Builder sampler) {
        List<GenerationRequest> out = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            out.add(GenerationRequest.builder().maxNewTokens(N_NEW).sampler(sampler.seed(70L + i).build()).build());
        }
        return out;
    }

    private static void assertSameTokens(Run whole, Run coned) {
        for (int i = 0; i < whole.results.size(); i++) {
            assertArrayEquals("request " + i + " tokens", whole.results.get(i).tokens(),
                    coned.results.get(i).tokens());
            assertEquals("request " + i + " text", whole.results.get(i).text(), coned.results.get(i).text());
        }
    }

    @Test
    public void greedyMatchRoundsVerifyOverTheDeviceCandidates() throws IOException {
        List<GenerationRequest> requests = requests(SamplerConfig.builder().greedy(true));
        Run whole = run(requests, VerifyMode.MATCH, null);
        Calls calls = new Calls();
        Run coned = run(requests, VerifyMode.MATCH, calls);
        assertSameTokens(whole, coned);
        assertTrue("the sequences speculated: " + coned.stats, coned.stats.specRounds() > 0);
        assertEquals("no step read its rows whole", 0, calls.wholeRows);
        assertEquals("no verify row was flagged whole", 0, calls.splitRows);
        assertTrue("every emitted token came from the readout: " + coned.stats,
                coned.stats.deviceSampledRows() >= coned.stats.generatedTokens());
    }

    @Test
    public void sampledMatchRoundsVerifyOverTheDeviceCandidates() throws IOException {
        List<GenerationRequest> requests = requests(SamplerConfig.builder().topK(8).temperature(0.9f));
        Run whole = run(requests, VerifyMode.MATCH, null);
        Calls calls = new Calls();
        Run coned = run(requests, VerifyMode.MATCH, calls);
        assertSameTokens(whole, coned);
        assertTrue(coned.stats.specRounds() > 0);
        assertEquals(0, calls.wholeRows);
        assertEquals(0, calls.splitRows);
    }

    /**
     * Penalties edit each verify row over the history the walk holds there: row {@code i} carries the round's drafts
     * before it in its prefilter, so the readout's candidates are the chain's own and the tokens are unchanged.
     */
    @Test
    public void penalizedMatchRoundsPrefilterEveryVerifyRowOverItsOwnHistory() throws IOException {
        List<GenerationRequest> requests = requests(SamplerConfig.builder().topK(8).temperature(0.7f)
                .repeatPenalty(1.6f).presencePenalty(0.8f).frequencyPenalty(0.5f).penaltyLastN(64));
        Run whole = run(requests, VerifyMode.MATCH, null);
        Calls calls = new Calls();
        Run coned = run(requests, VerifyMode.MATCH, calls);
        assertSameTokens(whole, coned);
        assertTrue(coned.stats.specRounds() > 0);
        assertTrue("the readout ran each row's prefilter", calls.prefiltered > 0);
        assertEquals(0, calls.wholeRows);
        assertEquals(0, calls.splitRows);
    }

    /**
     * A greedy chain with a presence penalty over a target whose unpenalized continuation repeats one token, with a
     * drafter proposing that token: row {@code i}'s prefilter must penalize the drafts before it, or the readout
     * would confirm a repeat the chain refuses.
     */
    @Test
    public void aPenalizedRepeatIsRefusedOnEveryVerifyRow() throws IOException {
        List<GenerationRequest> requests = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            requests.add(GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.builder().greedy(true)
                    .topK(1).presencePenalty(4f).penaltyLastN(64).build()).build());
        }
        Run whole = run(requests, VerifyMode.MATCH, null, new FixedDrafter(86), true);
        Calls calls = new Calls();
        Run coned = run(requests, VerifyMode.MATCH, calls, new FixedDrafter(86), true);
        assertSameTokens(whole, coned);
        assertTrue(coned.stats.specRounds() > 0);
        assertTrue("the readout ran each row's prefilter", calls.prefiltered > 0);
        assertEquals(0, calls.splitRows);
    }

    @Test
    public void aVerifierModeReadsItsVerifyRowsWhole() throws IOException {
        List<GenerationRequest> requests = requests(SamplerConfig.builder().topK(8).temperature(0.9f));
        Calls calls = new Calls();
        Run coned = run(requests, VerifyMode.REJECTION, calls);
        Run whole = run(requests, VerifyMode.REJECTION, null);
        assertSameTokens(whole, coned);
        assertTrue(coned.stats.specRounds() > 0);
        assertTrue("the verify rows were read whole", calls.splitRows + calls.wholeRows > 0);
    }
}
