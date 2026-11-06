package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The multi-model router over the real engine (llama.cpp's server_models): the directory scan and the presets
 * name the entries, a routed request opens its model's own service and serves on it, the autoload loads on the
 * first mention, the count limit evicts the least recently used idle model (never a serving one), the memory
 * margin unloads until the next load fits, and the registry answers POST and DELETE /models. The events of every
 * state change reach a subscriber.
 */
class ModelRouterTest {

    private static EngineConfig config;
    private static Path aGguf;
    private static Path bGguf;
    private static EngineService unloaded;

    @TempDir
    Path dir;

    /** The device figures the margin tests use: the models report a fixed hold, the devices a fixed total. */
    private ModelRouter.DeviceMemory memory(long totalBytes, long aBytes, long bBytes) {
        return new ModelRouter.DeviceMemory() {
            @Override
            public long totalBytes() {
                return totalBytes;
            }

            @Override
            public long heldBytes(EngineService service) {
                String path = String.valueOf(service.modelPath());
                return path.endsWith("a.gguf") ? aBytes : bBytes;
            }
        };
    }

    private ModelRouter router(int modelsMax, long marginBytes, boolean autoload,
                               ModelRouter.DeviceMemory memory) throws IOException {
        return new ModelRouter(dir, null, modelsMax, marginBytes, autoload, config,
                new ServerOptions(null, Duration.ofSeconds(2)), memory);
    }

    private ModelRouter router(int modelsMax, long marginBytes, boolean autoload) throws IOException {
        return router(modelsMax, marginBytes, autoload, memory(0, 0, 0));
    }

    private static JsonView body(String json) {
        return ServerJson.parse(json);
    }

    @BeforeAll
    static void open() throws IOException {
        config = pagedConfig(512, 2);
        aGguf = SyntheticModels.llama().writeTempFile();
        bGguf = SyntheticModels.llama().writeTempFile();
        unloaded = new EngineService(null, config);
    }

    @AfterAll
    static void close() {
        unloaded.close();
    }

    // ------------------------------------------------------------------ the registry

    @Test
    void theScanListsEveryDirectoryModel() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.copy(bGguf, dir.resolve("b.gguf"));
        ModelRouter router = router(4, 0, true);
        List<ModelRouter.Entry> entries = router.list();
        assertEquals(List.of("a", "b"), entries.stream().map(ModelRouter.Entry::name).toList());
        assertEquals(ModelRouter.Source.MODELS_DIR, entries.get(0).source());
        assertEquals(dir.resolve("a.gguf"), entries.get(0).path());
        assertEquals(ModelRouter.Status.UNLOADED, entries.get(0).status());
        assertFalse(entries.get(0).loadOnStartup());
    }

    @Test
    void aSplitModelListsOnceUnderItsFirstShard() throws IOException {
        Files.copy(aGguf, dir.resolve("big-00001-of-00002.gguf"));
        Files.copy(aGguf, dir.resolve("big-00002-of-00002.gguf"));
        ModelRouter router = router(4, 0, true);
        assertEquals(List.of("big"), router.list().stream().map(ModelRouter.Entry::name).toList());
        assertEquals(dir.resolve("big-00001-of-00002.gguf"), router.list().get(0).path());
    }

    @Test
    void thePresetsJoinTheRegistryWithTheirNamesAndAliases() throws IOException {
        Files.copy(aGguf, dir.resolve("gemma-q4.gguf"));
        ModelPreset preset = ModelPreset.parse(presetFile("[gemma]\npath = gemma-q4.gguf\nalias = gem\ntag ="
                + " local\nn_ctx = 128\n")).get(0);
        ModelRouter router = new ModelRouter(null, List.of(preset), 4, 0, true, config,
                new ServerOptions(null, Duration.ofSeconds(2)), memory(0, 0, 0));
        ModelRouter.Entry entry = router.list().get(0);
        assertEquals("gemma", entry.name());
        assertEquals(ModelRouter.Source.PRESET, entry.source());
        assertEquals(List.of("gem"), entry.aliases());
        assertEquals(List.of("local"), entry.tags());
        assertEquals(dir.resolve("gemma-q4.gguf"), entry.path());
        EngineService served = ModelRouter.target(router, null, body("{\"model\":\"gem\"}"), null);
        assertNotNull(served, "the alias routes to the preset's model");
        assertEquals(128, served.maxModelLength(), "the preset's n_ctx overrides the server's");
        served.close();
        router.close();
    }

    private Path presetFile(String ini) throws IOException {
        Path file = dir.resolve("models.ini");
        Files.writeString(file, ini);
        return file;
    }

    // ------------------------------------------------------------------ routing

    @Test
    @Timeout(120)
    void aRoutedRequestServesOnItsOwnModel() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.copy(bGguf, dir.resolve("b.gguf"));
        ModelRouter router = router(4, 0, true);
        try {
            EngineService serviceA = ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null);
            assertNotNull(serviceA);
            assertEquals(ModelRouter.Status.LOADED, router.list().get(0).status());
            assertEquals("a", serviceA.served().primary(), "the entry's responses carry its own name");
            EngineService serviceB = ModelRouter.target(router, unloaded, body("{\"model\":\"b\"}"), null);
            assertNotNull(serviceB);
            assertFalse(serviceA == serviceB);
            assertTrue(ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null) == serviceA,
                    "a second request serves on the open service");
        } finally {
            router.close();
        }
    }

    @Test
    void anUnknownModelA404AndAMissingNameA400() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        ModelRouter router = router(4, 0, true);
        WebApplicationException unknown = assertThrows(WebApplicationException.class,
                () -> ModelRouter.target(router, unloaded, body("{\"model\":\"zzz\"}"), null));
        assertEquals(404, unknown.getResponse().getStatus());
        WebApplicationException missing = assertThrows(WebApplicationException.class,
                () -> ModelRouter.target(router, unloaded, body("{}"), null));
        assertEquals(400, missing.getResponse().getStatus());
        assertTrue(missing.getResponse().getEntity().toString().contains("model name is missing"),
                missing.getResponse().getEntity().toString());
        router.close();
    }

    @Test
    @Timeout(120)
    void autoloadOffRefusesUntilTheModelLoads() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.copy(bGguf, dir.resolve("b.gguf"));
        ModelRouter router = router(4, 0, false);
        try {
            WebApplicationException refused = assertThrows(WebApplicationException.class,
                    () -> ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null));
            assertEquals(400, refused.getResponse().getStatus());
            assertTrue(refused.getResponse().getEntity().toString().contains("not loaded"),
                    refused.getResponse().getEntity().toString());
            assertEquals(400, statusOfAutoload(router, "b", "false"), "the request's own false refuses too");
            EngineService serviceA = router.load("a");
            assertNotNull(serviceA);
            assertTrue(ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null) == serviceA);
            assertTrue(ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), "false") == serviceA,
                    "a loaded model serves whatever the request's autoload asks");
            WebApplicationException badFlag = assertThrows(WebApplicationException.class,
                    () -> ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), "perhaps"));
            assertEquals(400, badFlag.getResponse().getStatus());
        } finally {
            router.close();
        }
    }

    /** The status a routed autoload-off request for {@code name} answers with. */
    private static int statusOfAutoload(ModelRouter router, String name, String autoloadParam) {
        try {
            ModelRouter.target(router, unloaded, body("{\"model\":\"" + name + "\"}"), autoloadParam);
            return 200;
        } catch (WebApplicationException e) {
            return e.getResponse().getStatus();
        }
    }

    @Test
    @Timeout(120)
    void theCountLimitEvictsTheLeastRecentlyUsedIdleModel() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.copy(bGguf, dir.resolve("b.gguf"));
        ModelRouter router = router(1, 0, true);
        try {
            EngineService serviceA = ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null);
            EngineService serviceB = ModelRouter.target(router, unloaded, body("{\"model\":\"b\"}"), null);
            assertEquals(ModelRouter.Status.UNLOADED, router.list().get(0).status(), "a gave its slot to b");
            assertEquals(ModelRouter.Status.LOADED, router.list().get(1).status());
            assertTrue(ModelRouter.target(router, unloaded, body("{\"model\":\"b\"}"), null) == serviceB);
            assertFalse(serviceA.loaded(), "the evicted engine closed");
        } finally {
            router.close();
        }
    }

    @Test
    @Timeout(120)
    void aServingModelKeepsItsSlotAndTheWaitRefusesPastTheBudget() throws IOException, InterruptedException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.copy(bGguf, dir.resolve("b.gguf"));
        ModelRouter router = router(1, 0, true);
        try {
            EngineService serviceA = ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null);
            GenerationBatch.start(serviceA, List.of(new GenerationBatch.Job(
                    new int[]{1, 2, 3}, GenerationRequest.builder().maxNewTokens(100000).ignoreEos(true).build(),
                    null, null, null)), null);
            serviceA.pause(SleepControl.Mode.KEEP, false);
            try {
                EngineUnavailableException busy = assertThrows(EngineUnavailableException.class,
                        () -> ModelRouter.target(router, unloaded, body("{\"model\":\"b\"}"), null));
                assertTrue(busy.getMessage().contains("serving"), busy.getMessage());
                assertEquals(ModelRouter.Status.LOADED, router.list().get(0).status(), "a kept its slot");
                assertEquals(ModelRouter.Status.UNLOADED, router.list().get(1).status());
            } finally {
                serviceA.resume();
                serviceA.abortAll();
                long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
                while (serviceA.load() > 0 && System.nanoTime() < until) {
                    TimeUnit.MILLISECONDS.sleep(20);
                }
            }
            assertEquals(0, serviceA.load(), "the aborted request finished");
            ModelRouter.target(router, unloaded, body("{\"model\":\"b\"}"), null);
            assertEquals(ModelRouter.Status.UNLOADED, router.list().get(0).status(), "an idle a evicts");
        } finally {
            router.close();
        }
    }

    @Test
    @Timeout(180)
    void theMemoryMarginUnloadsUntilTheNextLoadFits() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.copy(bGguf, dir.resolve("b.gguf"));
        ModelRouter tight = router(0, 50L * 1024 * 1024, true, memory(100L * 1024 * 1024, 60L * 1024 * 1024,
                30L * 1024 * 1024));
        try {
            ModelRouter.target(tight, unloaded, body("{\"model\":\"a\"}"), null);
            ModelRouter.target(tight, unloaded, body("{\"model\":\"b\"}"), null);
            assertEquals(ModelRouter.Status.UNLOADED, tight.list().get(0).status(), "b did not fit beside a");
            assertEquals(ModelRouter.Status.LOADED, tight.list().get(1).status());
        } finally {
            tight.close();
        }
        ModelRouter loose = router(0, 10L * 1024 * 1024, true, memory(100L * 1024 * 1024, 60L * 1024 * 1024,
                30L * 1024 * 1024));
        try {
            ModelRouter.target(loose, unloaded, body("{\"model\":\"a\"}"), null);
            ModelRouter.target(loose, unloaded, body("{\"model\":\"b\"}"), null);
            assertEquals(ModelRouter.Status.LOADED, loose.list().get(0).status(), "both fit within the margin");
            assertEquals(ModelRouter.Status.LOADED, loose.list().get(1).status());
        } finally {
            loose.close();
        }
    }

    // ------------------------------------------------------------------ POST and DELETE /models

    @Test
    @Timeout(120)
    void aRegisteredModelRoutesAndUnregisters() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        Files.createDirectories(dir.resolve("extra"));
        Path registered = dir.resolve("extra").resolve("registered.gguf");
        Files.copy(bGguf, registered);
        ModelRouter router = router(4, 0, true);
        try {
            ModelRouter.Entry entry = router.register(registered, List.of("reg"), List.of("mine"));
            assertEquals(ModelRouter.Source.REGISTERED, entry.source());
            assertEquals("registered", entry.name());
            assertEquals(2, router.list().size(), "the scan's a plus the registered one");
            EngineService service = ModelRouter.target(router, unloaded, body("{\"model\":\"reg\"}"), null);
            assertNotNull(service, "the alias routes");
            IllegalArgumentException absent = assertThrows(IllegalArgumentException.class,
                    () -> router.register(dir.resolve("absent.gguf"), List.of(), List.of()));
            assertTrue(absent.getMessage().contains(".gguf file"), absent.getMessage());
            Path another = dir.resolve("extra").resolve("another.gguf");
            Files.copy(bGguf, another);
            IllegalArgumentException clash = assertThrows(IllegalArgumentException.class,
                    () -> router.register(another, List.of("a"), List.of()));
            assertTrue(clash.getMessage().contains("alias"), clash.getMessage());
            IllegalArgumentException duplicate = assertThrows(IllegalArgumentException.class,
                    () -> router.register(registered, List.of(), List.of()));
            assertTrue(duplicate.getMessage().contains("already served"), duplicate.getMessage());
            router.unregister("registered");
            assertEquals(1, router.list().size());
            assertThrows(ModelRouter.UnknownModelException.class, () -> ModelRouter.target(router, unloaded,
                    body("{\"model\":\"registered\"}"), null));
            IllegalArgumentException managed = assertThrows(IllegalArgumentException.class,
                    () -> router.unregister("a"));
            assertTrue(managed.getMessage().contains(DIR), managed.getMessage());
        } finally {
            router.close();
        }
    }

    private static final String DIR = ModelRouter.DIR_PROPERTY;

    @Test
    @Timeout(120)
    void reloadPicksUpNewAndDroppedFiles() throws IOException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        ModelRouter router = router(4, 0, true);
        try {
            Files.copy(bGguf, dir.resolve("c.gguf"));
            router.reload();
            assertEquals(List.of("a", "c"), router.list().stream().map(ModelRouter.Entry::name).toList());
            Files.delete(dir.resolve("c.gguf"));
            router.reload();
            assertEquals(List.of("a"), router.list().stream().map(ModelRouter.Entry::name).toList());
        } finally {
            router.close();
        }
    }

    // ------------------------------------------------------------------ the events

    @Test
    @Timeout(120)
    void theEventsCarryEveryStateChange() throws IOException, InterruptedException {
        Files.copy(aGguf, dir.resolve("a.gguf"));
        ModelRouter router = router(4, 0, true);
        try {
            java.util.concurrent.LinkedBlockingQueue<ModelRouter.Event> events = router.subscribe();
            ModelRouter.target(router, unloaded, body("{\"model\":\"a\"}"), null);
            router.unload("a");
            ModelRouter.Event loading = events.poll(5, TimeUnit.SECONDS);
            ModelRouter.Event loaded = events.poll(5, TimeUnit.SECONDS);
            ModelRouter.Event unloadedEvent = events.poll(5, TimeUnit.SECONDS);
            assertNotNull(loading);
            assertEquals("a", loading.model());
            assertEquals("model_status", loading.type());
            assertEquals("loading", loading.data().get("status"));
            Map<String, Object> wire = new LinkedHashMap<>();
            assertNotNull(loaded);
            wire.putAll(loaded.wire());
            assertEquals("a", wire.get("model"));
            assertEquals("model_status", wire.get("event"));
            assertEquals("loaded", ((Map<?, ?>) wire.get("data")).get("status"));
            assertNotNull(unloadedEvent);
            assertEquals("unloaded", unloadedEvent.data().get("status"));
            router.unsubscribe(events);
        } finally {
            router.close();
        }
    }

    @Test
    @Timeout(120)
    void aFailedLoadRecordsTheFailureAndRetries() throws IOException {
        Path broken = dir.resolve("broken.gguf");
        Files.writeString(broken, "this is no GGUF file");
        ModelRouter router = router(4, 0, true);
        try {
            EngineUnavailableException failed = assertThrows(EngineUnavailableException.class,
                    () -> ModelRouter.target(router, unloaded, body("{\"model\":\"broken\"}"), null));
            assertTrue(failed.getMessage().contains("broken"), failed.getMessage());
            ModelRouter.Entry entry = router.list().get(0);
            assertEquals(ModelRouter.Status.UNLOADED, entry.status());
            assertNotNull(entry.error(), "the failure is recorded for GET /models");
            Files.copy(aGguf, broken, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            assertNotNull(ModelRouter.target(router, unloaded, body("{\"model\":\"broken\"}"), null), "a retry opens");
            assertNull(router.list().get(0).error());
        } finally {
            router.close();
        }
    }

    @Test
    void aRouterWithoutSourcesIsInactiveAndRoutesToTheFallback() throws IOException {
        ModelRouter inactive = assertDoesNotThrow(() -> new ModelRouter(null, null, 4, 0, true, config,
                new ServerOptions(null, Duration.ofSeconds(2)), memory(0, 0, 0)));
        assertFalse(inactive.active());
        assertEquals(unloaded, ModelRouter.target(inactive, unloaded, body("{\"model\":\"a\"}"), null),
                "an inactive router routes nowhere: the fallback serves");
        inactive.close();
    }
}
