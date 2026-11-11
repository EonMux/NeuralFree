package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The queue limit's placement over a two-replica pool of one sequence slot each, driven with generations that stay
 * waiting (never registered with an engine): a replica's free slot takes one waiting generation, so it holds a
 * request waiting only from the second on; a new request routed to a full replica goes to the other one and moves
 * its prompt's prefix affinity there; a request finds every replica full only when both are.
 */
class QueueLimitPlacementTest {

    private static EngineService pool;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(128, 1),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
    }

    @AfterAll
    static void close() {
        pool.close();
    }

    /** A generation that waits until the test ends it. */
    private static EngineReplica.Submission waitingGeneration(List<EngineReplica.Submission> created) {
        EngineReplica.Submission submission = new EngineReplica.Submission(null);
        created.add(submission);
        return submission;
    }

    private static int[] prompt(int seed) {
        return new int[] {5 + seed, 9, 13, 17, 21, 25};
    }

    @Test
    void aFullReplicasNewRequestGoesToTheOtherAndTakesItsPromptsAffinity() throws Exception {
        List<EngineReplica> replicas = pool.replicas();
        DpScheduler scheduler = new DpScheduler(replicas.toArray(new EngineReplica[0]), LoadBalance.SHORTEST_QUEUE,
                1);
        QueueLimit queue = scheduler.queue();
        List<EngineReplica.Submission> created = new ArrayList<>();
        try {
            int[] prompt = prompt(0);
            EngineReplica routed = scheduler.pick(prompt);
            EngineReplica other = routed == replicas.get(0) ? replicas.get(1) : replicas.get(0);

            queue.submit(null, routed, null, r -> new EngineService.Submission(waitingGeneration(created)));
            assertEquals(0, queue.waiting(routed), "the free slot takes the first waiting generation");
            queue.submit(null, routed, null, r -> new EngineService.Submission(waitingGeneration(created)));
            assertEquals(1, queue.waiting(routed), "the second waits for the slot");

            AtomicReference<EngineReplica> placed = new AtomicReference<>();
            queue.submit(null, routed, prompt, r -> {
                placed.set(r);
                return new EngineService.Submission(waitingGeneration(created));
            });
            assertSame(other, placed.get(), "a request routed to the full replica goes to the other");
            assertSame(other, scheduler.pick(prompt), "the prompt's affinity follows it to the other replica");

            queue.submit(null, other, null, r -> new EngineService.Submission(waitingGeneration(created)));
            assertEquals(1, queue.waiting(other));
            assertThrows(EngineSaturatedException.class, () -> queue.submit(null, routed, prompt(1),
                    r -> new EngineService.Submission(waitingGeneration(created))), "both replicas are full");
            assertNotSame(routed, other);
        } finally {
            for (EngineReplica.Submission submission : created) {
                submission.refused(new IllegalStateException("the test ended the generation"));
            }
        }
        assertEquals(0, queue.waiting(replicas.get(0)), "ended generations hold nothing");
        assertEquals(0, queue.waiting(replicas.get(1)));
    }
}
