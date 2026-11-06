package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's process-reward step filter over {@code token_classify}: {@code step_tag_id} keeps only the rows of the
 * tokens carrying that id, {@code returned_token_ids} picks the classifier outputs those rows answer, either field
 * alone applies its half, and both refuse on any model but a per-token classifier — as does vLLM's combined
 * {@code embed&token_classify} task, which this engine offers no dual-head model for.
 */
public class PoolingStepFilterTest {

    private static EngineService processService;
    private static EngineService sequenceService;
    private static PoolingResource process;
    private static PoolingResource sequence;

    @BeforeAll
    static void open() throws IOException {
        Path processModel = SyntheticModels.processReward().writeTempFile();
        processService = new EngineService(processModel.toString(), config().build());
        Path sequenceModel = SyntheticModels.sequenceClassify("neg", "pos").writeTempFile();
        sequenceService = new EngineService(sequenceModel.toString(), config().build());
        process = new PoolingResource(processService);
        sequence = new PoolingResource(sequenceService);
    }

    @AfterAll
    static void close() {
        processService.close();
        sequenceService.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static List<?> rows(JsonView body) {
        return (List<?>) at(body, "data", 0).raw("data");
    }

    @Test
    void theStepTagKeepsOnlyItsTokenRows() {
        JsonView body = ok(process.pooling(
                "{\"input\": [5, 6, 7], \"task\": \"token_classify\", \"step_tag_id\": 6}"));
        List<?> kept = rows(body);
        assertEquals(1, kept.size(), "exactly the token with the step id keeps its row");
        assertEquals(2, ((List<?>) kept.get(0)).size(), "both step scores stay");
    }

    @Test
    void theReturnedIdsPickTheOutputColumns() {
        JsonView body = ok(process.pooling(
                "{\"input\": [5, 6, 7], \"task\": \"token_classify\", \"returned_token_ids\": [1, 0]}"));
        List<?> kept = rows(body);
        assertEquals(3, kept.size(), "every token keeps its row");
        for (Object row : kept) {
            assertEquals(2, ((List<?>) row).size(), "the two picked outputs, in the requested order");
        }
    }

    @Test
    void bothFieldsApplyTogether() {
        JsonView body = ok(process.pooling("{\"input\": [5, 6, 7], \"task\": \"token_classify\","
                + " \"step_tag_id\": 7, \"returned_token_ids\": [1]}"));
        List<?> kept = rows(body);
        assertEquals(1, kept.size());
        assertEquals(1, ((List<?>) kept.get(0)).size());
    }

    @Test
    void anOutOfRangeReturnedIdRefusesByName() {
        assertTrue(errorOf(() -> process.pooling(
                        "{\"input\": [5, 6], \"task\": \"token_classify\", \"returned_token_ids\": [2]}"))
                .contains("returned_token_ids"));
    }

    @Test
    void aNegativeStepTagRefuses() {
        assertEquals(400, statusOf(() -> process.pooling(
                "{\"input\": [5, 6], \"task\": \"token_classify\", \"step_tag_id\": -1}")));
    }

    @Test
    void theFieldsRefuseBesideANonTokenTask() {
        assertTrue(errorOf(() -> sequence.pooling(
                        "{\"input\": [5, 6], \"task\": \"classify\", \"step_tag_id\": 6}"))
                .contains("only to the token_classify task"));
        assertTrue(errorOf(() -> sequence.pooling(
                        "{\"input\": [5, 6], \"task\": \"classify\", \"returned_token_ids\": [0]}"))
                .contains("only to the token_classify task"));
    }

    @Test
    void theFieldsRefuseOnEveryOtherModel() {
        assertTrue(errorOf(() -> sequence.pooling("{\"input\": [5, 6], \"task\": \"token_classify\","
                + " \"step_tag_id\": 6}")).contains("step_tag_id"));
        assertTrue(errorOf(() -> sequence.pooling("{\"input\": [5, 6], \"returned_token_ids\": [0]}"))
                .contains("returned_token_ids"));
    }

    @Test
    void theCombinedTaskRefusesByWhatTheModelLacks() {
        assertTrue(errorOf(() -> process.pooling("{\"input\": [5, 6], \"task\": \"embed&token_classify\"}"))
                .contains("embed&token_classify"));
        assertTrue(errorOf(() -> sequence.pooling("{\"input\": [5, 6], \"task\": \"embed&token_classify\"}"))
                .contains("embed&token_classify"));
        assertTrue(errorOf(() -> process.pooling("{\"input\": [5, 6], \"task\": \"embed&token_classify\"}"))
                .contains("per-token"));
    }
}
