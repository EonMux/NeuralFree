package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's scheduling policies against vLLM's scheduler: {@code fcfs} refuses a nonzero
 * {@code priority} at submit and preempts the youngest running sequence; {@code priority} preempts the running
 * sequence of the highest {@code priority} value (among equals the latest arrival) and admits a waiting
 * submission of a lower value ahead of a preempted sequence of a higher one, as vLLM's priority waiting queue
 * orders both. Preemption never changes a result. Two sequences of 52 rows outgrow the 64 cells two slots share.
 */
public class BatchPrioritySchedulingTest {

    private static final int PROMPT = 12;
    private static final int LONG = 40;
    private static final int STEPS = 4_000;

    private static Path model;

    @BeforeClass
    public static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    private static EngineConfig config(SchedulingPolicy policy) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true).kvPageSize(4)
                .parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(false).schedulingPolicy(policy)
                .build()).build();
    }

    private static GenerationRequest request(int maxNewTokens, int priority) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens).ignoreEos(true).priority(priority)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 11 + i * 7) % 250;
        }
        return t;
    }

    private static BatchSequence submit(BatchEngine engine, int[] prompt, GenerationRequest request) {
        return engine.submit(prompt, request, (TokenCallback) null);
    }

    /** Pumps until every handle finished, recording the order they finished in. */
    private static List<BatchSequence> drain(BatchEngine engine, BatchSequence... handles) {
        List<BatchSequence> order = new ArrayList<>();
        for (int step = 0; step < STEPS && order.size() < handles.length; step++) {
            engine.pump();
            for (BatchSequence h : handles) {
                if (h.finished() && !order.contains(h)) {
                    order.add(h);
                }
            }
        }
        for (BatchSequence h : handles) {
            assertEquals("the request completed: " + h.error(), BatchSequence.State.DONE, h.state());
        }
        return order;
    }

    /** The tokens each prompt generates alone on an engine of {@code policy}, one after the other. */
    private static int[][] alone(SchedulingPolicy policy, int[][] prompts, int[] maxNew) throws IOException {
        EngineConfig cfg = config(policy);
        int[][] out = new int[prompts.length][];
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            for (int i = 0; i < prompts.length; i++) {
                BatchSequence seq = submit(engine, prompts[i], request(maxNew[i], 0));
                drain(engine, seq);
                assertEquals("a request running alone is never preempted", 0, seq.preemptions());
                out[i] = seq.result().tokens();
            }
        }
        return out;
    }

    @Test
    public void fcfsRefusesANonzeroPriorityAtSubmitByName() throws IOException {
        EngineConfig cfg = config(SchedulingPolicy.FCFS);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            assertEquals(SchedulingPolicy.FCFS, engine.schedulingPolicy());
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> submit(engine, prompt(1, PROMPT), request(2, 3)));
            assertTrue(e.getMessage(), e.getMessage().startsWith("priority 3"));
            assertTrue(e.getMessage(), e.getMessage().contains(MemoryConfig.PROP_SCHEDULING_POLICY));
            assertTrue("nothing queued", engine.idle());
            BatchSequence plain = submit(engine, prompt(1, PROMPT), request(2, 0));
            drain(engine, plain);
        }
    }

    @Test
    public void memoryPressurePreemptsTheRunningSequenceOfTheHighestPriorityValue() throws IOException {
        int[][] prompts = {prompt(2, PROMPT), prompt(3, PROMPT)};
        int[][] expected = alone(SchedulingPolicy.PRIORITY, prompts, new int[] {LONG, LONG});
        EngineConfig cfg = config(SchedulingPolicy.PRIORITY);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            assertEquals(SchedulingPolicy.PRIORITY, engine.schedulingPolicy());
            BatchSequence older = submit(engine, prompts[0], request(LONG, 5));
            BatchSequence urgent = submit(engine, prompts[1], request(LONG, 0));
            List<BatchSequence> order = drain(engine, older, urgent);
            assertTrue("the two outgrow the shared cells", engine.stats().preemptions() > 0);
            assertTrue("the older request of the higher value yields", older.preemptions() > 0);
            assertEquals("the lower value keeps running though it arrived later", 0, urgent.preemptions());
            assertEquals(Arrays.asList(urgent, older), order);
            assertEquals(engine.stats().preemptions(), older.preemptions());
            assertArrayEquals("preemption changes no result", expected[0], older.result().tokens());
            assertArrayEquals(expected[1], urgent.result().tokens());
        }
    }

    @Test
    public void fcfsAndEqualPrioritiesPreemptTheYoungestRunningSequence() throws IOException {
        for (SchedulingPolicy policy : SchedulingPolicy.values()) {
            EngineConfig cfg = config(policy);
            try (GGUFModel m = new GGUFModel(model.toString(), cfg);
                 BatchEngine engine = BatchEngine.forModel(m, cfg)) {
                BatchSequence first = submit(engine, prompt(4, PROMPT), request(LONG, 0));
                BatchSequence second = submit(engine, prompt(5, PROMPT), request(LONG, 0));
                List<BatchSequence> order = drain(engine, first, second);
                assertTrue(policy + ": the two outgrow the shared cells", engine.stats().preemptions() > 0);
                assertEquals(policy + ": the first arrival keeps running", 0, first.preemptions());
                assertTrue(policy + ": the later arrival yields", second.preemptions() > 0);
                assertEquals(policy.toString(), Arrays.asList(first, second), order);
            }
        }
    }

    @Test
    public void aWaitingLowerValueAdmitsAheadOfAPreemptedHigherValue() throws IOException {
        int[][] prompts = {prompt(6, PROMPT), prompt(7, PROMPT), prompt(8, 44)};
        int[] maxNew = {LONG, LONG, 4};
        int[][] expected = alone(SchedulingPolicy.PRIORITY, prompts, maxNew);
        EngineConfig cfg = config(SchedulingPolicy.PRIORITY);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence low = submit(engine, prompts[0], request(maxNew[0], 5));
            BatchSequence top = submit(engine, prompts[1], request(maxNew[1], 0));
            engine.pump();
            assertEquals(2, engine.running());
            // both slots are taken: the middle value queues behind the running pair
            BatchSequence middle = submit(engine, prompts[2], request(maxNew[2], 1));
            for (int step = 0; step < STEPS && low.preemptions() == 0; step++) {
                engine.pump();
            }
            assertTrue("the highest value yields first", low.preemptions() > 0);
            assertEquals(0, top.preemptions());
            for (int step = 0; step < STEPS && middle.state() == BatchSequence.State.QUEUED; step++) {
                engine.pump();
            }
            assertEquals(BatchSequence.State.RUNNING, middle.state());
            assertTrue("the slots freed once the lowest value finished", top.finished());
            assertEquals("the preempted higher value still waits: its recompute does not fit beside the"
                    + " admitted prompt", 1, engine.stats().preempted());
            List<BatchSequence> order = drain(engine, low, top, middle);
            assertEquals(Arrays.asList(top, middle, low), order);
            assertEquals(0, middle.preemptions());
            assertArrayEquals(expected[0], low.result().tokens());
            assertArrayEquals(expected[1], top.result().tokens());
            assertArrayEquals(expected[2], middle.result().tokens());
        }
    }
}
