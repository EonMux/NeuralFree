package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.Test;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The dispatch shapes the cloud adapters share: llama.cpp's {@code path_to_gcp_format} aliases, the Vertex format
 * map, and the invocation type a SageMaker body names by its fields (vLLM's priority order).
 */
class CloudProtocolsTest {

    @Test
    void camelCaseMatchesLlamaCppsPathToGcpFormat() {
        assertEquals("chatCompletions", CloudProtocols.camelCase("/v1/chat/completions"));
        assertEquals("completions", CloudProtocols.camelCase("/v1/completions"));
        assertEquals("embeddings", CloudProtocols.camelCase("/v1/embeddings"));
        assertEquals("score", CloudProtocols.camelCase("/score"));
        assertEquals("v2Rerank", CloudProtocols.camelCase("/v2/rerank"));
        assertEquals("cohereV2Chat", CloudProtocols.camelCase("/cohere/v2/chat"));
        assertEquals("applyTemplate", CloudProtocols.camelCase("/apply-template"));
        assertEquals("applyTemplate", CloudProtocols.camelCase("/v1/apply_template"));
        assertEquals("tokenize", CloudProtocols.camelCase("/tokenize:param"), "stops before a path parameter");
    }

    @Test
    void vertexFormatsAcceptPathsAndAliases() {
        assertEquals("/v1/chat/completions", CloudProtocols.vertexRoute("/v1/chat/completions"));
        assertEquals("/v1/chat/completions", CloudProtocols.vertexRoute("chatCompletions"));
        assertEquals("/v2/rerank", CloudProtocols.vertexRoute("v2Rerank"));
        assertEquals("/score", CloudProtocols.vertexRoute("score"));
        assertEquals("/cohere/v2/chat", CloudProtocols.vertexRoute("cohereV2Chat"));
        assertNull(CloudProtocols.vertexRoute("nope"));
        assertNull(CloudProtocols.vertexRoute(null));
    }

    @Test
    void invocationsAreNamedByTheirFieldsInVllmsOrder() {
        assertInvocation(CloudProtocols.Invocation.CHAT, "{\"messages\": []}");
        assertInvocation(CloudProtocols.Invocation.COMPLETION, "{\"prompt\": \"hi\"}");
        assertInvocation(CloudProtocols.Invocation.RERANK,
                "{\"query\": \"q\", \"documents\": [\"d\"]}");
        assertInvocation(CloudProtocols.Invocation.SCORE, "{\"queries\": [\"q\"], \"documents\": [\"d\"]}");
        assertInvocation(CloudProtocols.Invocation.SCORE, "{\"data_1\": [\"q\"], \"data_2\": [\"d\"]}");
        assertInvocation(CloudProtocols.Invocation.EMBEDDING, "{\"input\": \"hi\"}");
        assertInvocation(CloudProtocols.Invocation.POOLING, "{\"input\": \"hi\", \"task\": \"embed\"}");
    }

    @Test
    void anUnknownBodyRefusesWithVllmsTypeList() {
        String message = errorOf(() -> CloudProtocols.requireInvocation(
                ServerJson.parse("{\"unrelated\": 1}"), null));
        assertTrue(message.contains("Cannot find suitable handler for request. Expected one of:"));
        assertTrue(message.contains("ChatCompletionRequest"));
        assertTrue(message.contains("PoolingRequest"));
        assertEquals(CloudProtocols.Invocation.CHAT, CloudProtocols.requireInvocation(
                ServerJson.parse("{\"messages\": [], \"input\": \"x\"}"), null),
                "the chat request wins in vLLM's priority order");
    }

    private static void assertInvocation(CloudProtocols.Invocation expected, String json) {
        JsonView body = ServerJson.parse(json);
        assertEquals(expected, CloudProtocols.invocationOf(body, null));
    }
}
