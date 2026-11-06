package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fault-tolerance control routes over the in-process replica pool: the vLLM status shape, the 202 dispatch and
 * its validation, a retry that re-opens a replica whose wake failed, and a scale_down that drops a failed rank while
 * the deployment keeps its slot. The status reports the replicas' health and the instruction state.
 */
class FaultToleranceResourceTest {

    private static Path model;

    private EngineService pool;
    private FaultToleranceResource resource;

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @BeforeEach
    void open() throws IOException {
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(128, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        resource = new FaultToleranceResource(pool, new ClusterSwitches(true, true));
    }

    @AfterEach
    void close() {
        pool.close();
    }

    private static final TokenCallback IGNORED = (token, text) -> { };

    /** Waits up to {@code seconds} for {@code condition}. */
    private static void await(BooleanSupplier condition, long seconds, String what) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds);
        while (!condition.getAsBoolean()) {
            assertTrue(System.nanoTime() < deadline, what);
            Thread.sleep(10);
        }
    }

    private JsonView status() {
        return view(resource.status().getEntity());
    }

    /** Sleeps replica 1 at level 2 and removes the model file, so its wake fails with a real reopen error. */
    private Path breakReplicaOne() throws Exception {
        Path backup = Files.createTempFile("neuralfree-model-backup", ".gguf");
        Files.copy(model, backup, StandardCopyOption.REPLACE_EXISTING);
        EngineReplica replica = pool.replicas().get(1);
        replica.sleepOp(2, SleepControl.Mode.ABORT);
        Files.delete(model);
        assertThrows(IllegalStateException.class, () -> replica.wakeUp(List.of("weights")));
        return backup;
    }

    @Test
    void theStatusReportsEveryReplica() {
        JsonView status = status();
        assertEquals(1, status.i64("schema_version"));
        assertEquals(2, status.i64("total_engines"));
        assertEquals(2, size(status, "engines"));
        assertEquals(0, at(status, "engines", 0).i64("id"));
        assertEquals(1, at(status, "engines", 1).i64("id"));
        assertEquals("healthy", at(status, "engines", 0).str("status"));
        assertNull(at(status, "engines", 0).raw("fault_info"));
    }

    @Test
    void theApplyBodyIsValidatedByField() {
        assertEquals("'instruction' must be a non-empty string.", errorOf(() -> resource.apply("{}")));
        assertTrue(errorOf(() -> resource.apply("{\"instruction\": \"restart\", \"request_id\": \"r\"}"))
                .contains("Invalid instruction: 'restart'."));
        assertTrue(errorOf(() -> resource.apply("{\"instruction\": \"retry\"}"))
                .contains("'request_id' must be a non-empty string."));
        assertTrue(errorOf(() -> resource.apply(
                "{\"instruction\": \"retry\", \"request_id\": \"r\", \"params\": []}"))
                .contains("'params' must be an object."));
        assertTrue(errorOf(() -> resource.apply(
                "{\"instruction\": \"scale_down\", \"request_id\": \"r\", \"params\": {}}"))
                .contains("removed_dp_ranks"));
    }

    @Test
    void theRoutesAreOffWithoutTheSwitch() {
        FaultToleranceResource off = new FaultToleranceResource(pool, new ClusterSwitches(true, false));
        assertEquals(404, statusOf(off::status));
        assertEquals(404, statusOf(() -> off.apply("{\"instruction\": \"retry\", \"request_id\": \"r\"}")));
    }

    @Test
    void aRetryOfAHealthyPoolAnswersAcceptedAndRecordsTheReason() throws Exception {
        Response accepted = resource.apply("{\"instruction\": \"retry\", \"request_id\": \"round-1\"}");
        assertEquals(202, accepted.getStatus());
        assertEquals("round-1", view(accepted.getEntity()).str("request_id"));
        await(() -> "failed".equals(at(status(), "engines", 0).optStr("ft_state", null)), 10,
                "the healthy retry was never recorded");
        JsonView row = at(status(), "engines", 0);
        assertEquals("round-1", row.str("last_ft_request_id"));
        assertTrue(row.str("ft_error").contains("every replica is healthy"));
        assertEquals("healthy", row.str("status"));
    }

    @Test
    @Timeout(120)
    void aRetryReopensAReplicaWhoseWakeFailed() throws Exception {
        Path backup = breakReplicaOne();
        try {
            JsonView broken = at(status(), "engines", 1);
            assertEquals("unhealthy", broken.str("status"));
            assertTrue(broken.str("fault_info").contains("did not wake"), broken.str("fault_info"));

            Files.copy(backup, model, StandardCopyOption.REPLACE_EXISTING);
            Response accepted = resource.apply("{\"instruction\": \"retry\", \"request_id\": \"repair\"}");
            assertEquals(202, accepted.getStatus());
            await(() -> "healthy".equals(at(status(), "engines", 1).str("status")), 30,
                    "the replica never recovered");
            assertNull(at(status(), "engines", 1).raw("ft_state"));

            int[] prompt = pool.encode("hello after recovery", true, false);
            EngineService.Submission served = pool.submit(prompt, GenerationRequest.builder().maxNewTokens(4)
                    .ignoreEos(true).sampler(SamplerConfig.greedyDecoding()).build(), IGNORED);
            assertTrue(served.await(30, TimeUnit.SECONDS));
            assertNull(served.refused());
            assertNotNull(served.sequence());
        } finally {
            Files.copy(backup, model, StandardCopyOption.REPLACE_EXISTING);
            Files.deleteIfExists(backup);
        }
    }

    @Test
    @Timeout(120)
    void aScaleDownDropsAFailedRankAndTheDeploymentKeepsItsSlot() throws Exception {
        Path backup = breakReplicaOne();
        try {
            JsonView before = status();
            assertEquals("unhealthy", at(before, "engines", 1).str("status"));

            Response accepted = resource.apply(
                    "{\"instruction\": \"scale_down\", \"request_id\": \"down\", \"params\":"
                            + " {\"removed_dp_ranks\": [1]}}");
            assertEquals(202, accepted.getStatus());
            await(() -> pool.replicaCount() == 1, 30, "the failed rank never left the pool");
            assertEquals(2, pool.dataParallelSize(), "the deployment keeps the removed rank's slot");
            assertEquals(0, pool.replicaRank(0));
            JsonView after = status();
            assertEquals(1, after.i64("total_engines"));
            assertEquals(1, size(after, "engines"));
        } finally {
            Files.copy(backup, model, StandardCopyOption.REPLACE_EXISTING);
            Files.deleteIfExists(backup);
        }
    }

    @Test
    void aHealthyRankCannotBeRemoved() {
        assertTrue(errorOf(() -> resource.apply(
                "{\"instruction\": \"scale_down\", \"request_id\": \"down\", \"params\":"
                        + " {\"removed_dp_ranks\": [1]}}")).contains("which is healthy"));
        assertTrue(errorOf(() -> resource.apply(
                "{\"instruction\": \"scale_down\", \"request_id\": \"down\", \"params\":"
                        + " {\"removed_dp_ranks\": [7]}}")).contains("does not serve"));
        assertFalse(pool.replicas().isEmpty());
    }
}
