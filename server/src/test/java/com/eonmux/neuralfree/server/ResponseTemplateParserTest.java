package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's {@code beam} parser over the real engine: the synthetic llama carries a {@code tokenizer.response_template}
 * whose answer sits in an explicit {@code <a>} region and whose calls name their function in the opener, served with
 * {@code beam} as the reasoning and tool parser. Every chat answers under the grammar derived from the template — a
 * plain chat, a {@code response_format} framed as the answer, {@code tool_choice: none} and a required or named
 * choice — on the OpenAI and the Anthropic routes alike; a structural-tag output stays the request's own, a tool
 * guarantee the template cannot express is refused with a 400 naming the parameter, and
 * {@code neuralfree.beam.always.constrain=false} leaves an unconstrained chat free.
 */
class ResponseTemplateParserTest {

    private static final String TEMPLATE = "{\"start_anchor\": \"<s>\", \"fields\": {"
            + "\"thinking\": {\"open\": \"<t>\", \"close\": \"</t>\"},"
            + "\"content\": {\"open\": \"<a>\", \"close\": \"</a>\"},"
            + "\"tool_calls\": {\"open_pattern\": \"<call name=\\\"(?P<name>[\\\\w.-]+)\\\">\", \"close\": \"</call>\","
            + " \"repeats\": true, \"content\": \"json\", \"transform\": {\"type\": \"function\","
            + " \"function\": {\"name\": \"{name}\", \"arguments\": \"{content}\"}}}}}";
    private static final String TOOLS = "\"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"f\","
            + " \"parameters\": {\"type\": \"object\", \"properties\": {}}}}]";

    private static EngineService service;
    private static EngineService free;
    private static OpenAiResource openAi;
    private static AnthropicResource anthropic;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).str(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE, TEMPLATE).writeTempFile();
        EngineConfig engine = ServerTestSupport.pagedConfig(1024, 2).toBuilder().reasoningParser("beam")
                .toolParser("beam").build();
        service = new EngineService(model.toString(), engine);
        free = new EngineService(model.toString(), engine.toBuilder().beamAlwaysConstrain(false).build());
        openAi = new OpenAiResource(service);
        anthropic = new AnthropicResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
        free.close();
    }

    private static String chat(String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 4, \"temperature\": 0"
                + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    private static ConstraintSpec planned(EngineService on, String body) {
        return ChatPlan.of(on, ServerJson.parse(body)).constraint();
    }

    @Test
    void thePlainChatParsesWithTheModelsTemplate() {
        TemplateParsing parsing = service.reasoning().forRequest(java.util.Collections.emptyMap()).responseTemplate();
        assertNotNull(parsing);
        assertTrue(parsing.beam());
        assertEquals(200, openAi.chatCompletions(chat("")).getStatus());
        assertEquals(200, openAi.chatCompletions(chat(TOOLS)).getStatus());
    }

    @Test
    void everyChatFollowsTheTemplatesGrammar() {
        for (String extra : new String[]{"", TOOLS, "\"response_format\": {\"type\": \"json_object\"}",
                TOOLS + ", \"tool_choice\": \"none\"", TOOLS + ", \"tool_choice\": \"required\"",
                TOOLS + ", \"tool_choice\": {\"type\": \"function\", \"function\": {\"name\": \"f\"}}"}) {
            ConstraintSpec constraint = planned(service, chat(extra));
            assertNotNull(constraint, extra);
            assertEquals(ConstraintSpec.Kind.STRUCTURAL_TAG, constraint.kind(), extra);
            assertEquals(200, openAi.chatCompletions(chat(extra)).getStatus(), extra);
        }
        String messages = "{\"model\": \"m\", \"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\":"
                + " \"hi\"}], \"tools\": [{\"name\": \"f\", \"input_schema\": {\"type\": \"object\"}}],"
                + " \"tool_choice\": {\"type\": \"none\"}}";
        assertEquals(200, anthropic.messages(messages).getStatus());
    }

    @Test
    void theAlwaysOnGrammarCanBeTurnedOff() {
        assertNull(planned(free, chat("")));
        assertEquals(ConstraintSpec.Kind.STRUCTURAL_TAG,
                planned(free, chat(TOOLS + ", \"tool_choice\": \"required\"")).kind());
    }

    @Test
    void aStructuralTagOutputStaysTheRequestsOwn() {
        String tag = chat("\"response_format\": {\"type\": \"structural_tag\", \"format\": {\"type\": \"const_string\","
                + " \"value\": \"x\"}}");
        ConstraintSpec constraint = planned(service, tag);
        assertEquals(ConstraintSpec.Kind.STRUCTURAL_TAG, constraint.kind());
        assertTrue(constraint.text().contains("const_string"), constraint.text());
        assertFalse(constraint.text().contains("<a>"), "the template's framing is not derived: " + constraint.text());
        assertEquals(200, openAi.chatCompletions(tag).getStatus());
    }

    @Test
    void whatTheTemplateCannotExpressIsRefusedByName() {
        String both = chat(TOOLS + ", \"tool_choice\": \"required\", \"response_format\": {\"type\": \"json_object\"}");
        assertEquals(400, statusOf(() -> openAi.chatCompletions(both)));
        assertTrue(errorOf(() -> openAi.chatCompletions(both)).contains("'response_format'"));

        String spaced = chat("\"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get weather\"}}],"
                + " \"tool_choice\": \"required\"");
        assertEquals(400, statusOf(() -> openAi.chatCompletions(spaced)));
        assertTrue(errorOf(() -> openAi.chatCompletions(spaced)).contains("'tools'"));

        String unknown = "{\"model\": \"m\", \"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\":"
                + " \"hi\"}], \"tools\": [{\"name\": \"get weather\", \"input_schema\": {\"type\": \"object\"}}],"
                + " \"tool_choice\": {\"type\": \"any\"}}";
        assertEquals(400, statusOf(() -> anthropic.messages(unknown)));
        assertTrue(errorOf(() -> anthropic.messages(unknown)).contains("tool name 'get weather'"));
    }
}
