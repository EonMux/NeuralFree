package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The slot endpoints: the list reports the engine's slots with llama.cpp's status fields (the running or last
 * task's counters, params, next_token, prompt and generated text over the bare id and context); every action
 * answers llama.cpp's refusal while the
 * server runs without {@code --slot-save-path}; under it a slot saves the prefix its last request left, restores
 * into another slot and erases with llama.cpp's {@code n_saved} / {@code n_restored} / {@code n_erased} fields, an
 * unusable file answers llama.cpp's restore refusal, an engine without a prefix cache refuses by name, and every
 * malformed slot id, action and filename is refused by name.
 */
class SlotsResourceTest {

    @TempDir
    static Path saveDir;

    private static EngineService service;
    private static EngineService unloaded;
    private static SlotsResource slots;
    private static SlotsResource disabled;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, saveDir));
        new OpenAiResource(service).completions("{\"prompt\": \"hi\", \"max_tokens\": 1}");
        unloaded = new EngineService(null, ServerTestSupport.pagedConfig(256, 2));
        slots = new SlotsResource(service);
        disabled = new SlotsResource(unloaded);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    @SuppressWarnings("unchecked")
    void theListReportsEverySlotAndItsContext() {
        List<Object> list = (List<Object>) parsed(slots.slots(null).getEntity());
        assertEquals(2, list.size());
        assertEquals(0, ((Number) ((Map<String, Object>) list.get(0)).get("id")).intValue());
        assertEquals(1, ((Number) ((Map<String, Object>) list.get(1)).get("id")).intValue());
        assertEquals(256, ((Number) ((Map<String, Object>) list.get(1)).get("n_ctx")).intValue());
    }

    @Test
    @SuppressWarnings("unchecked")
    void theListAnswersFailOnNoSlotWhenASlotIsIdle() throws IOException {
        new OpenAiResource(service).completions("{\"prompt\": \"hi\", \"max_tokens\": 1}");
        List<Object> list = (List<Object>) parsed(slots.slots("true").getEntity());
        assertEquals(2, list.size(), "an idle slot answers the list");
    }

    @Test
    void theListRefusesAnUnloadedServer() {
        assertEquals(503, statusOf(() -> disabled.slots(null)));
    }

    @Test
    void withoutTheSavePathEveryActionAnswersLlamaCppsRefusal() {
        WebApplicationException e = assertThrows(WebApplicationException.class,
                () -> disabled.action(0, "save", "{\"filename\": \"slot.state\"}"));
        assertEquals(501, e.getResponse().getStatus());
        JsonView error = view(e.getResponse().getEntity()).obj("error");
        assertEquals(SlotsResource.DISABLED, error.str("message"));
        assertEquals("not_supported_error", error.str("type"));
        assertEquals(501, statusOf(() -> disabled.action(0, "restore", "{\"filename\": \"s\"}")));
        assertEquals(501, statusOf(() -> disabled.action(0, "erase", "{}")));
    }

    @Test
    void saveRestoreAndEraseAnswerLlamaCppsFields() throws IOException {
        JsonView saved = view(slots.action(0, "save", "{\"filename\": \"slot0.state\"}").getEntity());
        assertEquals(0, saved.i32("id_slot"));
        assertEquals("slot0.state", saved.str("filename"));
        int tokens = saved.i32("n_saved");
        assertTrue(tokens > 0, "the completion's prompt stays cached as slot 0's state");
        assertEquals(Files.size(saveDir.resolve("slot0.state")), saved.i64("n_written"));
        assertTrue(saved.obj("timings").f64("save_ms") >= 0);

        JsonView restored = view(slots.action(1, "restore", "{\"filename\": \"slot0.state\"}").getEntity());
        assertEquals(1, restored.i32("id_slot"));
        assertEquals(tokens, restored.i32("n_restored"));
        assertEquals(saved.i64("n_written"), restored.i64("n_read"));
        assertTrue(restored.obj("timings").f64("restore_ms") >= 0);

        JsonView erased = view(slots.action(1, "erase", "{}").getEntity());
        assertEquals(1, erased.i32("id_slot"));
        assertEquals(tokens, erased.i32("n_erased"));
        assertEquals(0, view(slots.action(1, "erase", "{}").getEntity()).i32("n_erased"));
    }

    @Test
    void theListReportsEachSlotsLastTask() throws IOException {
        new OpenAiResource(service).completions("{\"prompt\": \"hi there\", \"max_tokens\": 2}");
        List<Object> list = (List<Object>) parsed(slots.slots(null).getEntity());
        Map<String, Object> served = (Map<String, Object>) list.get(0);
        assertEquals(Boolean.FALSE, served.get("is_processing"));
        assertEquals(Boolean.FALSE, served.get("speculative"));
        assertTrue(((Number) served.get("id_task")).intValue() >= 0,
                "the finished request stays as the slot's task");
        int promptTokens = ((Number) served.get("n_prompt_tokens")).intValue();
        assertTrue(promptTokens > 0);
        assertEquals(promptTokens, ((Number) served.get("n_prompt_tokens_processed")).intValue(),
                "the whole prompt processed by the end");
        Map<String, Object> params = (Map<String, Object>) served.get("params");
        assertTrue(params.containsKey("temperature"));
        assertTrue(params.containsKey("samplers"));
        assertTrue(params.containsKey("n_predict"));
        List<?> next = (List<?>) served.get("next_token");
        Map<?, ?> token = (Map<?, ?>) next.get(0);
        assertEquals(Boolean.FALSE, token.get("has_next_token"));
        assertTrue(((Number) token.get("n_decoded")).intValue() >= 1, "the two generated tokens fed");
        assertTrue(((String) served.get("prompt")).contains("hi there"));
        assertFalse(((String) served.get("generated")).isEmpty());

        Map<String, Object> idle = (Map<String, Object>) list.get(1);
        assertEquals(256, ((Number) idle.get("n_ctx")).intValue());
        assertFalse(idle.containsKey("id_task"), "a slot with no task reports the bare fields only");
        assertFalse(idle.containsKey("prompt"));
        assertFalse(idle.containsKey("generated"));
    }

    @Test
    void anUnusableSlotFileAnswersLlamaCppsRestoreRefusal() throws IOException {
        WebApplicationException missing = assertThrows(WebApplicationException.class,
                () -> slots.action(1, "restore", "{\"filename\": \"absent.state\"}"));
        assertEquals(400, missing.getResponse().getStatus());
        JsonView error = view(missing.getResponse().getEntity()).obj("error");
        assertTrue(error.str("message").startsWith("Unable to restore slot"), error.str("message"));
        assertEquals("invalid_request_error", error.str("type"));
        Files.write(saveDir.resolve("foreign.state"), new byte[] {9, 9, 9, 9, 9, 9, 9, 9});
        assertTrue(errorOf(() -> slots.action(1, "restore", "{\"filename\": \"foreign.state\"}"))
                .contains("not a NeuralFree slot file"));
    }

    @Test
    void anEngineWithoutAPrefixCacheRefusesByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig noCache = ServerTestSupport.config()
                .memory(MemoryConfig.builder().nCtx(256).ubatch(8).parallel(2).prefixReuse(false)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        EngineService off = new EngineService(model.toString(), noCache,
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, saveDir));
        try {
            WebApplicationException e = assertThrows(WebApplicationException.class,
                    () -> new SlotsResource(off).action(0, "save", "{\"filename\": \"off.state\"}"));
            assertEquals(501, e.getResponse().getStatus());
            JsonView error = view(e.getResponse().getEntity()).obj("error");
            assertTrue(error.str("message").contains(MemoryConfig.PROP_PREFIX_REUSE), error.str("message"));
            assertEquals("not_supported_error", error.str("type"));
        } finally {
            off.close();
        }
    }

    @Test
    void malformedSlotIdsAndActionsAreRefusedByName() {
        assertEquals(400, statusOf(() -> slots.action(5, "save", "{\"filename\": \"s\"}")));
        assertEquals("Invalid slot ID", errorOf(() -> slots.action(5, "erase", "{}")));
        assertEquals("Invalid slot ID", errorOf(() -> slots.action(-1, "save", "{\"filename\": \"s\"}")));
        assertEquals("Invalid action", errorOf(() -> slots.action(0, "resize", "{}")));
        assertEquals("Invalid action", errorOf(() -> slots.action(0, null, "{}")));
    }

    @Test
    void malformedFilenamesAreRefusedByName() {
        assertEquals(400, statusOf(() -> slots.action(0, "save", "{}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save", "{\"filename\": \"../evil\"}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save", "{\"filename\": \"a/b\"}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save", "{\"filename\": \"state.\"}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save", "{\"filename\": \"a..b\"}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save", "{\"filename\": \" state\"}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save", "{\"filename\": 3}")));
        assertEquals("Invalid filename", errorOf(() -> slots.action(0, "save",
                "{\"filename\": \"" + "x".repeat(256) + "\"}")));
    }

    @Test
    void filenamesPassTheReferenceRules() {
        assertTrue(SlotsResource.validFilename("slot-0.state"));
        assertTrue(SlotsResource.validFilename("café.state"));
        assertFalse(SlotsResource.validFilename(""));
        assertFalse(SlotsResource.validFilename("a\\b"));
        assertFalse(SlotsResource.validFilename("a:b"));
        assertFalse(SlotsResource.validFilename("a*b"));
        assertFalse(SlotsResource.validFilename("a?b"));
        assertFalse(SlotsResource.validFilename("a\"b"));
        assertFalse(SlotsResource.validFilename("a<b"));
        assertFalse(SlotsResource.validFilename("state "));
        assertTrue(SlotsResource.validFilename(".state"));
        assertFalse(SlotsResource.validFilename("a\uFEFFb"));
    }

    @Test
    void thePropertyNamesTheSaveDirectory() throws IOException {
        assertNull(ServerOptions.fromSystemProperties().slotSavePath());
        System.setProperty(ServerOptions.SLOT_SAVE_PATH_PROPERTY, saveDir.toString());
        try {
            assertEquals(saveDir, ServerOptions.fromSystemProperties().slotSavePath());
            Path file = Files.write(saveDir.resolve("note.txt"), new byte[0]);
            System.setProperty(ServerOptions.SLOT_SAVE_PATH_PROPERTY, file.toString());
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    ServerOptions::fromSystemProperties);
            assertTrue(e.getMessage().contains(ServerOptions.SLOT_SAVE_PATH_PROPERTY), e.getMessage());
        } finally {
            System.clearProperty(ServerOptions.SLOT_SAVE_PATH_PROPERTY);
        }
    }
}
