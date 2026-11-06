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

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Lark grammars over the real service: vLLM's {@code structured_outputs.grammar} and {@code guided_grammar}, the
 * chat route, and llama.cpp's {@code grammar} take Lark as well as GBNF (told apart as vLLM does), drive the
 * synthetic llama's sampling into the grammar's language, and a malformed Lark grammar is a 400 naming the problem.
 */
class LarkGrammarResourceTest {

    /** vLLM's {@code SAMPLE_SQL_LARK} ({@code tests/entrypoints/llm/test_struct_output_generate.py}). */
    private static final String SQL_LARK = "start: select_statement\\n"
            + "select_statement: \\\"SELECT\\\" column \\\"from\\\" table \\\"where\\\" condition\\n"
            + "column: \\\"col_1\\\" | \\\"col_2\\\"\\n"
            + "table: \\\"table_1\\\" | \\\"table_2\\\"\\n"
            + "condition: column \\\"=\\\" number\\n"
            + "number: \\\"1\\\" | \\\"2\\\"\\n";

    private static final String SQL = "SELECT(col_1|col_2)from(table_1|table_2)where(col_1|col_2)=(1|2)";

    /** A left-recursive sum over common numbers, with ignored blanks. */
    private static final String SUM_LARK = "start: sum\\nsum: sum \\\"+\\\" INT | INT\\n%import common.INT\\n"
            + "%import common.WS_INLINE\\n%ignore WS_INLINE";

    private static EngineService service;
    private static OpenAiResource openAi;
    private static LlamaCompletionResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
        openAi = new OpenAiResource(service);
        llama = new LlamaCompletionResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static JsonView firstChoice(Response response) {
        assertEquals(200, response.getStatus());
        return JsonView.of(((List<?>) view(response.getEntity()).raw("choices")).get(0), "choices[0]");
    }

    @Test
    void structuredOutputsGrammarInLarkConstrainsTheCompletion() {
        JsonView choice = firstChoice(openAi.completions("{\"prompt\": \"query:\", \"max_tokens\": 64,"
                + " \"temperature\": 0, \"structured_outputs\": {\"grammar\": \"" + SQL_LARK + "\"}}"));
        assertTrue(choice.str("text").matches(SQL), "a statement of the Lark grammar: " + choice.str("text"));
        assertEquals("stop", choice.str("finish_reason"));
    }

    @Test
    void guidedGrammarInLarkRewritesLeftRecursionAndIgnoresBlanks() {
        JsonView choice = firstChoice(openAi.completions("{\"prompt\": \"sum:\", \"max_tokens\": 24,"
                + " \"temperature\": 0, \"guided_grammar\": \"" + SUM_LARK + "\"}"));
        String text = choice.str("text");
        assertTrue(text.matches("[0-9]+([ \\t]*\\+[ \\t]*[0-9]+)*[ \\t]*"), "a sum of integers: " + text);
    }

    @Test
    void chatStructuredOutputsTakeLark() {
        JsonView choice = firstChoice(openAi.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"write sql\"}], \"max_tokens\": 64, \"temperature\": 0,"
                + " \"structured_outputs\": {\"grammar\": \"" + SQL_LARK + "\"}}"));
        String content = choice.obj("message").str("content");
        assertTrue(content.matches(SQL), "a statement of the Lark grammar: " + content);
    }

    @Test
    void llamaCppGrammarFieldTakesLarkToo() {
        JsonView body = view(llama.completion("{\"prompt\": \"query:\", \"n_predict\": 64, \"temperature\": 0,"
                + " \"grammar\": \"" + SQL_LARK + "\"}").getEntity());
        assertTrue(body.str("content").matches(SQL), "a statement of the Lark grammar: " + body.str("content"));
    }

    @Test
    void llamaCppLlguidanceHeaderSelectsTheLarkGrammar() {
        String headed = "%llguidance {}\\n" + SQL_LARK;
        JsonView body = view(llama.completion("{\"prompt\": \"query:\", \"n_predict\": 64, \"temperature\": 0,"
                + " \"grammar\": \"" + headed + "\"}").getEntity());
        assertTrue(body.str("content").matches(SQL), "a statement of the Lark grammar: " + body.str("content"));
    }

    @Test
    void aMalformedLlguidanceHeaderIsRefusedNamingIt() {
        String request = "{\"prompt\": \"x\", \"n_predict\": 4, \"grammar\": \"%llguidance {bad}\"}";
        assertEquals(400, statusOf(() -> llama.completion(request)));
        assertTrue(errorOf(() -> llama.completion(request)).contains("failed to parse %llguidance declaration"));
    }

    @Test
    void gbnfStillCompilesAsGbnf() {
        JsonView choice = firstChoice(openAi.completions("{\"prompt\": \"pick\", \"max_tokens\": 8,"
                + " \"temperature\": 0,"
                + " \"structured_outputs\": {\"grammar\": \"root ::= \\\"yes\\\" | \\\"no\\\"\"}}"));
        assertTrue(choice.str("text").equals("yes") || choice.str("text").equals("no"), choice.str("text"));
    }

    @Test
    void aMalformedLarkGrammarIsRefusedNamingTheProblem() {
        String missingStart = "{\"prompt\": \"x\", \"max_tokens\": 4, \"structured_outputs\":"
                + " {\"grammar\": \"rule: \\\"hello\\\"\"}}";
        assertEquals(400, statusOf(() -> openAi.completions(missingStart)));
        assertTrue(errorOf(() -> openAi.completions(missingStart)).contains("no start rule found"));
        String singleQuoted = "{\"prompt\": \"x\", \"max_tokens\": 4, \"guided_grammar\": \"start: 'hello'\"}";
        assertTrue(errorOf(() -> openAi.completions(singleQuoted)).contains("double quotes"));
        String priority = "{\"prompt\": \"x\", \"n_predict\": 4, \"grammar\": \"start.2: \\\"a\\\"\"}";
        assertEquals(400, statusOf(() -> llama.completion(priority)));
        assertTrue(errorOf(() -> llama.completion(priority)).contains("priorities are not supported"));
    }
}
