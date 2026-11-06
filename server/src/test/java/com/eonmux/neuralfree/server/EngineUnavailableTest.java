package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A pause or sleep that takes effect after a route's entry check still refuses the request on the serving state:
 * {@link EngineService#submit} refuses it with an {@link EngineUnavailableException}, the replica's pump re-checks
 * at the registration, and {@link GenerationBatch} answers both with 503 instead of a 400 or a 500.
 */
class EngineUnavailableTest {

    /** A sink that drops the generated text; the tests read the terminal state. */
    private static final ReasoningFilter.Sink SINK = ReasoningFilter.contentTo((token, text) -> { });

    private static Path model;
    private static EngineService service;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2));
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** A short greedy request the engine finishes deterministically. */
    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(4).ignoreEos(true).build();
    }

    private static List<GenerationBatch.Job> job() {
        return List.of(new GenerationBatch.Job(new int[]{1, 2, 3}, request(), null, null, null));
    }

    @Test
    @Timeout(60)
    void aPauseAfterTheEntryCheckAnswers503() throws InterruptedException {
        service.pause(SleepControl.Mode.ABORT, false);
        try {
            WebApplicationException refused = assertThrows(WebApplicationException.class,
                    () -> GenerationBatch.start(service, job(), null));
            assertEquals(503, refused.getResponse().getStatus());
            assertTrue(String.valueOf(refused.getResponse().getEntity()).contains("/resume"),
                    String.valueOf(refused.getResponse().getEntity()));
        } finally {
            service.resume();
        }
        GenerationBatch served = GenerationBatch.start(service, job(), null);
        served.awaitAll();
        assertEquals(4, served.result(0).tokens().length, "the resumed engine serves again");
    }

    @Test
    @Timeout(60)
    void aPauseBetweenTheCheckAndTheRegistrationRefusesOnThePump() throws Exception {
        try (EngineReplica replica = new EngineReplica(0, 0, null, model.toString(),
                ServerTestSupport.pagedConfig(512, 2), ServerOptions.defaults(), () -> false, new LoraCatalog(),
                () -> { })) {
            replica.pauseOp(SleepControl.Mode.ABORT, false);
            // the replica's own submit skips the pool's entry check, as a request that raced the pause does
            EngineReplica.Submission refused = replica.submit(new int[]{1, 2, 3}, request(), SINK, null, null,
                    null, null);
            assertTrue(refused.await(30, TimeUnit.SECONDS));
            assertNull(refused.sequence(), "a paused replica never registers the request");
            assertInstanceOf(EngineUnavailableException.class, refused.refused());
            assertTrue(refused.refused().getMessage().contains("/resume"), refused.refused().getMessage());

            replica.resumeOp();
            EngineReplica.Submission served = replica.submit(new int[]{1, 2, 3}, request(), SINK, null, null,
                    null, null);
            assertTrue(served.await(30, TimeUnit.SECONDS));
            assertNull(served.refused());
            assertEquals(BatchSequence.State.DONE, served.sequence().state());
        }
    }
}
