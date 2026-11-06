package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The profiler configuration (vLLM's {@code --profiler-config}): off by default, {@code torch} enables the window
 * over its trace directory, vLLM's other kinds and any unhonoured field refuse by name, a missing directory refuses
 * at launch, the launch flag sets the property, the enabled configuration logs vLLM's warning once, and the off
 * configuration refuses the profile routes with the unregistered route's 404.
 */
public class ProfilerConfigTest {

    @TempDir
    Path dir;

    private static String refusalOf(Runnable call) {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class, call::run);
        return refused.getMessage();
    }

    @Test
    void theConfigurationIsOffByDefault() {
        ProfilerConfig off = ProfilerConfig.from(Map.of());
        assertFalse(off.enabled());
        assertNull(off.profiler());
        assertNull(off.directory());
        assertFalse(ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY, "  ")).enabled());
        assertFalse(ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY, "{\"profiler\": null}")).enabled());
    }

    @Test
    void torchEnablesTheWindowOverItsTraceDirectory() {
        ProfilerConfig config = ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY,
                "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"" + dir + "\"}"));
        assertTrue(config.enabled());
        assertEquals(ProfilerConfig.TORCH, config.profiler());
        assertEquals(dir.toAbsolutePath().normalize(), config.directory());
    }

    @Test
    void theExistingProfileDirectoryStandsInForTheJsonField() {
        ProfilerConfig config = ProfilerConfig.from(Map.of(
                ProfilerConfig.PROPERTY, "{\"profiler\": \"torch\"}",
                EngineProfiler.PROFILE_DIR_PROPERTY, dir.toString()));
        assertTrue(config.enabled());
        assertEquals(dir.toAbsolutePath().normalize(), config.directory());
    }

    @Test
    void vllmsOtherKindsRefuseByName() {
        for (String kind : new String[]{"cuda", "proton", "CUDA"}) {
            String message = refusalOf(() -> ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY,
                    "{\"profiler\": \"" + kind + "\", \"torch_profiler_dir\": \"" + dir + "\"}")));
            assertTrue(message.contains(kind), message);
            assertTrue(message.contains(ProfilerConfig.TORCH), message);
        }
        String unknown = refusalOf(() -> ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY,
                "{\"profiler\": \"nope\", \"torch_profiler_dir\": \"" + dir + "\"}")));
        assertTrue(unknown.contains("nope"), unknown);
        assertTrue(unknown.contains(ProfilerConfig.TORCH), unknown);
    }

    @Test
    void unhonouredFieldsRefuseByName() {
        String message = refusalOf(() -> ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY,
                "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"" + dir + "\","
                        + " \"proton_profiler_dir\": \"/tmp/x\"}")));
        assertTrue(message.contains("proton_profiler_dir"), message);
    }

    @Test
    void aTorchProfilerNeedsItsTraceDirectory() {
        String message = refusalOf(() -> ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY,
                "{\"profiler\": \"torch\"}")));
        assertTrue(message.contains(ProfilerConfig.TORCH_DIR), message);
        assertTrue(message.contains(EngineProfiler.PROFILE_DIR_PROPERTY), message);

        String orphan = refusalOf(() -> ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY,
                "{\"torch_profiler_dir\": \"" + dir + "\"}")));
        assertTrue(orphan.contains(ProfilerConfig.TORCH_DIR), orphan);
    }

    @Test
    void aMalformedValueRefusesByName() {
        for (String raw : new String[]{"torch", "{", "[]"}) {
            String message = refusalOf(() -> ProfilerConfig.from(Map.of(ProfilerConfig.PROPERTY, raw)));
            assertTrue(message.contains(ProfilerConfig.PROPERTY), raw + ": " + message);
        }
    }

    @Test
    void theLaunchFlagSetsTheProperty() {
        Map<String, String> properties = ServerLaunchArgs.parse("--profiler-config",
                "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"" + dir + "\"}").properties();
        assertTrue(ProfilerConfig.from(properties).enabled());
        assertEquals(ProfilerConfig.TORCH, ProfilerConfig.from(properties).profiler());
        assertEquals(dir.toAbsolutePath().normalize(), ProfilerConfig.from(properties).directory());
    }

    @Test
    void theLaunchFlagRefusesUnsupportedValuesByName() {
        String previous = System.getProperty(EngineProfiler.PROFILE_DIR_PROPERTY);
        System.clearProperty(EngineProfiler.PROFILE_DIR_PROPERTY);
        try {
            String cuda = refusalOf(() -> ServerLaunchArgs.parse("--profiler-config",
                    "{\"profiler\": \"cuda\", \"torch_profiler_dir\": \"" + dir + "\"}"));
            assertTrue(cuda.startsWith(ProfilerConfig.FLAG), cuda);
            assertTrue(cuda.contains("cuda"), cuda);
            String missing = refusalOf(() -> ServerLaunchArgs.parse("--profiler-config",
                    "{\"profiler\": \"torch\"}"));
            assertTrue(missing.contains(ProfilerConfig.TORCH_DIR), missing);
            String json = refusalOf(() -> ServerLaunchArgs.parse("--profiler-config", "torch"));
            assertTrue(json.contains(ProfilerConfig.PROPERTY), json);
        } finally {
            if (previous != null) {
                System.setProperty(EngineProfiler.PROFILE_DIR_PROPERTY, previous);
            }
        }
    }

    @Test
    void theEnabledConfigurationLogsVllmsWarningOnce() throws Exception {
        List<String> messages = new ArrayList<>();
        Logger logger = Logger.getLogger(ProfilerConfig.class.getName());
        Handler handler = new Handler() {
            @Override
            public void publish(LogRecord record) {
                messages.add(record.getMessage());
            }

            @Override
            public void flush() {
            }

            @Override
            public void close() {
            }
        };
        logger.addHandler(handler);
        try {
            new ProfilerConfig(Map.of(ProfilerConfig.PROPERTY,
                    "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"" + dir + "\"}"));
        } finally {
            logger.removeHandler(handler);
        }
        assertEquals(1, messages.size(), messages.toString());
        assertTrue(messages.get(0).contains(ProfilerConfig.TORCH), messages.get(0));
        assertTrue(messages.get(0).contains("local development"), messages.get(0));
    }

    @Test
    void theOffConfigurationRefusesTheProfileRoutes() {
        WebApplicationException refused = assertThrows(WebApplicationException.class,
                () -> ProfilerConfig.OFF.require());
        assertEquals(404, refused.getResponse().getStatus());
        JsonView body = view(refused.getResponse().getEntity());
        assertEquals("NotFoundError", body.obj("error").str("type"));
        String message = body.obj("error").str("message");
        assertTrue(message.contains(ProfilerConfig.FLAG), message);
        assertTrue(message.contains(ProfilerConfig.PROPERTY), message);
        assertEquals(ProfilerConfig.TORCH, ProfilerConfig.of(ProfilerConfig.TORCH, dir).profiler());
        assertThrows(IllegalArgumentException.class, () -> ProfilerConfig.of("cuda", dir));
    }
}
