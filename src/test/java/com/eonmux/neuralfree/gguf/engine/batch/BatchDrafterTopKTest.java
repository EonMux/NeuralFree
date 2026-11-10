package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
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
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The batch draft model reads its rows' top candidates instead of whole rows when every draft of the step picks
 * greedily over an identity vocabulary on a backend that reads candidates back on the device: the proposals equal
 * the whole-row drafter's, no whole row is read, and a step with a sampled draft keeps the whole rows.
 */
public class BatchDrafterTopKTest {

    private static EngineConfig config() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(256).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(3).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(mc).draftMax(4).build();
    }

    /** Calls of one backend method name. */
    private static final class Calls {
        final AtomicInteger rows = new AtomicInteger();
        final AtomicInteger topK = new AtomicInteger();
    }

    /** {@code model} whose forward backend reports a device top-k readout and counts the batch forwards. */
    private static EngineModel deviceTopK(EngineModel model, Calls calls) {
        ForwardBackend real = model.forwardBackend();
        InvocationHandler backend = (proxy, method, args) -> {
            switch (method.getName()) {
                case "batchTopKOnDevice":
                    return true;
                case "forwardBatch":
                    calls.rows.incrementAndGet();
                    break;
                case "forwardBatchTopK":
                    calls.topK.incrementAndGet();
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

    private static List<BatchDrafter.Request> requests(float temperature) {
        List<BatchDrafter.Request> out = new ArrayList<>();
        out.add(new BatchDrafter.Request(0, 10, 4, 0f, new SplittableRandom(3)));
        out.add(new BatchDrafter.Request(1, 20, 3, temperature, new SplittableRandom(4)));
        out.add(new BatchDrafter.Request(2, 33, 4, 0f, new SplittableRandom(5)));
        return out;
    }

    private static int[][] drafts(Path model, Path draft, boolean deviceTopK, Calls calls, float temperature)
            throws IOException {
        EngineConfig cfg = config().toBuilder().draftModelPath(draft.toString()).build();
        try (GGUFModel target = new GGUFModel(model.toString(), cfg)) {
            EngineModel draftModel = LLMModelFactory.openBatchDraftModel(target, cfg);
            EngineModel used = deviceTopK ? deviceTopK(draftModel, calls) : draftModel;
            // the drafter owns the draft model and closes it
            try (BatchModelDrafter drafter = new BatchModelDrafter(used, target, 0f, true)) {
                int[][] first = drafter.draftBatch(requests(temperature));
                int[][] second = drafter.draftBatch(requests(temperature));
                int[][] out = new int[first.length + second.length][];
                System.arraycopy(first, 0, out, 0, first.length);
                System.arraycopy(second, 0, out, first.length, second.length);
                return out;
            }
        }
    }

    @Test
    public void greedyDraftsReadTopCandidatesAndProposeTheWholeRowsTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path draft = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().seed(7).build())
                .writeTempFile();
        int[][] whole = drafts(model, draft, false, null, 0f);
        Calls calls = new Calls();
        int[][] top = drafts(model, draft, true, calls, 0f);
        assertEquals(whole.length, top.length);
        for (int i = 0; i < whole.length; i++) {
            assertArrayEquals("draft " + i, whole[i], top[i]);
        }
        assertTrue("the drafts read top candidates", calls.topK.get() > 0);
        assertEquals("no whole row is read", 0, calls.rows.get());
    }

    @Test
    public void aSampledDraftKeepsTheWholeRows() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path draft = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().seed(7).build())
                .writeTempFile();
        Calls calls = new Calls();
        drafts(model, draft, true, calls, 0.8f);
        assertEquals("a probabilistic draw needs the whole row", 0, calls.topK.get());
        assertTrue(calls.rows.get() > 0);
    }
}
