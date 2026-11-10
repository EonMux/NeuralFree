package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A step whose sampling pieces mostly fit a device cone reads only their candidates, while the pieces that need whole
 * rows (a logprobs capture, a grammar) read theirs from the same forward ({@code ForwardBackend.forwardBatchSplit}):
 * over a backend that reports a device top-k readout, no step reads every row whole because of one such piece, the
 * split flags exactly that piece's rows, and every request emits what it emits when every row is read whole.
 */
public class BatchSplitReadoutTest {

    private static final int N_NEW = 8;

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(96).ubatch(16).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(4).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(mc).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** The batch calls the engine made through the proxied backend, and the whole-row flags of its split reads. */
    private static final class Calls {
        int wholeRows;
        int topK;
        final List<boolean[]> splits = new ArrayList<>();
    }

    /** {@code model} whose forward backend reports a device top-k readout and records the engine's batch calls. */
    private static EngineModel deviceTopK(EngineModel model, Calls calls) {
        ForwardBackend real = model.forwardBackend();
        InvocationHandler backend = (proxy, method, args) -> {
            switch (method.getName()) {
                case "batchTopKOnDevice":
                    return true;
                case "forwardBatch":
                    calls.wholeRows++;
                    break;
                case "forwardBatchTopK":
                    calls.topK++;
                    break;
                case "forwardBatchSplit":
                    calls.splits.add(((boolean[]) args[1]).clone());
                    break;
                default:
                    break;
            }
            return invoke(method, real, args);
        };
        ForwardBackend wrapped = (ForwardBackend) Proxy.newProxyInstance(ForwardBackend.class.getClassLoader(),
                new Class<?>[]{ForwardBackend.class}, backend);
        InvocationHandler engine = (proxy, method, args) -> method.getName().equals("forwardBackend")
                && method.getParameterCount() == 0 ? wrapped : invoke(method, model, args);
        return (EngineModel) Proxy.newProxyInstance(EngineModel.class.getClassLoader(),
                new Class<?>[]{EngineModel.class}, engine);
    }

    private static Object invoke(Method method, Object target, Object[] args) throws Throwable {
        try {
            return method.invoke(target, args);
        } catch (InvocationTargetException e) {
            throw e.getCause();
        }
    }

    private static List<GenerationRequest> requests() {
        List<GenerationRequest> out = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            out.add(GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.builder().topK(8)
                    .temperature(0.9f).seed(30L + i).build()).build());
        }
        out.add(GenerationRequest.builder().maxNewTokens(N_NEW).logprobs(3).sampler(SamplerConfig.builder()
                .topK(8).temperature(0.9f).seed(40L).build()).build());
        return out;
    }

    private static List<GenerationResult> run(Path model, List<GenerationRequest> requests, Calls calls)
            throws IOException {
        EngineConfig cfg = config();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(calls == null ? m : deviceTopK(m, calls), cfg)) {
            List<BatchSequence> running = new ArrayList<>();
            for (int i = 0; i < requests.size(); i++) {
                running.add(engine.submit(tokens(i + 1, 6 + i), requests.get(i), (TokenCallback) null));
            }
            engine.drain();
            List<GenerationResult> out = new ArrayList<>();
            for (BatchSequence s : running) {
                assertEquals(String.valueOf(s.error()), BatchSequence.State.DONE, s.state());
                out.add(s.result());
            }
            if (calls != null) {
                assertTrue("the cone pieces sampled over the candidates", engine.stats().deviceSampledRows() > 0);
            }
            return out;
        }
    }

    @Test
    public void oneLogprobsPieceReadsItsRowWholeBesideTheConePieces() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        List<GenerationRequest> requests = requests();
        List<GenerationResult> whole = run(model, requests, null);
        Calls calls = new Calls();
        List<GenerationResult> split = run(model, requests, calls);
        assertEquals("no step read every row whole", 0, calls.wholeRows);
        assertTrue("the steps with the logprobs piece split their readout", calls.splits.size() > 0);
        for (boolean[] flags : calls.splits) {
            int flagged = 0;
            for (boolean f : flags) {
                flagged += f ? 1 : 0;
            }
            assertEquals("only the logprobs piece's row is read whole", 1, flagged);
            assertTrue("the other rows read candidates", flags.length > 1);
        }
        for (int i = 0; i < whole.size(); i++) {
            assertArrayEquals("request " + i + " tokens", whole.get(i).tokens(), split.get(i).tokens());
            assertEquals("request " + i + " text", whole.get(i).text(), split.get(i).text());
        }
        List<TokenScore[]> expected = whole.get(3).logprobs();
        List<TokenScore[]> actual = split.get(3).logprobs();
        assertEquals(expected.size(), actual.size());
        for (int t = 0; t < expected.size(); t++) {
            for (int j = 0; j < expected.get(t).length; j++) {
                assertEquals(expected.get(t)[j].tokenId(), actual.get(t)[j].tokenId());
                assertEquals(expected.get(t)[j].logprob(), actual.get(t)[j].logprob(), 0f);
            }
        }
    }

    @Test
    public void aGrammarPieceReadsItsRowWholeBesideTheConePieces() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        List<GenerationRequest> requests = requests().subList(0, 3);
        List<GenerationRequest> withGrammar = new ArrayList<>(requests);
        withGrammar.add(GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.builder().topK(8).temperature(0.9f).seed(41L).build())
                .constraint(ConstraintSpec.grammar("root ::= [a-z ]+")).build());
        List<GenerationResult> whole = run(model, withGrammar, null);
        Calls calls = new Calls();
        List<GenerationResult> split = run(model, withGrammar, calls);
        assertEquals("no step read every row whole", 0, calls.wholeRows);
        assertTrue(calls.splits.size() > 0);
        for (int i = 0; i < whole.size(); i++) {
            assertArrayEquals("request " + i + " tokens", whole.get(i).tokens(), split.get(i).tokens());
        }
    }
}
