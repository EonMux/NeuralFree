package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.grpc.AbortRequest;
import com.eonmux.neuralfree.grpc.ControlGrpc;
import com.eonmux.neuralfree.grpc.GetServerInfoRequest;
import com.eonmux.neuralfree.grpc.IsPausedRequest;
import com.eonmux.neuralfree.grpc.IsSleepingRequest;
import com.eonmux.neuralfree.grpc.LoadLoraRequest;
import com.eonmux.neuralfree.grpc.PauseGenerationRequest;
import com.eonmux.neuralfree.grpc.PauseMode;
import com.eonmux.neuralfree.grpc.ResumeGenerationRequest;
import com.eonmux.neuralfree.grpc.SleepRequest;
import com.eonmux.neuralfree.grpc.UnloadLoraRequest;
import com.eonmux.neuralfree.grpc.WakeUpRequest;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.quarkus.test.common.QuarkusTestResource;
import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import io.quarkus.test.common.http.TestHTTPResource;
import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.QuarkusTestProfile;
import io.quarkus.test.junit.TestProfile;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Control gates with the development switch on and sleep mode off: the sleep and wake calls answer the
 * sleep-mode refusal the HTTP routes answer ({@code INVALID_ARGUMENT}, the 400 of the sleep-mode gate), while the
 * other development calls — server info, abort, pause, resume and the state calls — reach their handlers. The LoRA
 * changes still answer {@code NOT_FOUND} for their own switch.
 */
@QuarkusTest
@QuarkusTestResource(value = GrpcControlGateSleepOffTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(GrpcControlGateSleepOffTest.Profile.class)
class GrpcControlGateSleepOffTest {

    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    private static ManagedChannel channel;
    private static ControlGrpc.ControlBlockingStub control;

    @TestHTTPResource
    URI base;

    @BeforeEach
    void connect() {
        if (channel == null) {
            channel = ManagedChannelBuilder.forAddress(base.getHost(), base.getPort()).usePlaintext().build();
            control = ControlGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(GrpcTestSupport.TIMEOUT_SECONDS, TimeUnit.SECONDS);
        }
    }

    @AfterAll
    static void close() {
        if (channel != null) {
            channel.shutdownNow();
            channel = null;
        }
        control = null;
        PREVIOUS.forEach((name, value) -> {
            if (value == null) {
                System.clearProperty(name);
            } else {
                System.setProperty(name, value);
            }
        });
    }

    @Test
    void theSleepOperationsRefuseWithoutSleepMode() {
        assertSleepModeRefusal(() -> control.sleep(SleepRequest.newBuilder().setLevel(0).build()));
        assertSleepModeRefusal(() -> control.wakeUp(WakeUpRequest.getDefaultInstance()));
    }

    @Test
    void theOtherDevelopmentCallsReachTheirHandlers() {
        assertFalse(control.isSleeping(IsSleepingRequest.getDefaultInstance()).getSleeping());
        assertFalse(control.getServerInfo(GetServerInfoRequest.getDefaultInstance()).getEngineVersion().isEmpty());
        control.abort(AbortRequest.newBuilder().addRequestIds("no-such-request").build());
        control.pauseGeneration(PauseGenerationRequest.newBuilder()
                .setMode(PauseMode.PAUSE_MODE_ABORT).build());
        try {
            assertTrue(control.isPaused(IsPausedRequest.getDefaultInstance()).getPaused());
        } finally {
            control.resumeGeneration(ResumeGenerationRequest.getDefaultInstance());
        }
        assertFalse(control.isPaused(IsPausedRequest.getDefaultInstance()).getPaused());
    }

    @Test
    void theLoraChangesStillNeedTheirSwitch() {
        assertNotFound(ServerLoras.RUNTIME_PROPERTY,
                () -> control.loadLora(LoadLoraRequest.newBuilder()
                        .setLoraName("adapter").setSourcePath("/tmp/adapter.gguf").build()));
        assertNotFound(ServerLoras.RUNTIME_PROPERTY,
                () -> control.unloadLora(UnloadLoraRequest.newBuilder().setLoraName("adapter").build()));
    }

    private static void assertSleepModeRefusal(Runnable call) {
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class, call::run);
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(refused).getCode(),
                GrpcTestSupport.description(refused));
        assertTrue(GrpcTestSupport.description(refused).contains(SleepModeSwitch.FLAG),
                GrpcTestSupport.description(refused));
        assertTrue(GrpcTestSupport.description(refused).contains(SleepModeSwitch.PROPERTY),
                GrpcTestSupport.description(refused));
    }

    private static void assertNotFound(String hint, Runnable call) {
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class, call::run);
        assertEquals(Status.Code.NOT_FOUND, Status.fromThrowable(refused).getCode(),
                GrpcTestSupport.description(refused));
        assertTrue(GrpcTestSupport.description(refused).contains(hint), GrpcTestSupport.description(refused));
    }

    /** Turns the gRPC surface and the development routes on, sleep mode off, before Quarkus starts. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            try {
                Path model = SyntheticModels.llama().writeTempFile();
                Map.of("neuralfree.server.model", model.toString(),
                                "neuralfree.backend", "cpu",
                                "neuralfree.ctx", "2048",
                                "neuralfree.parallel", "1",
                                "neuralfree.kv.paged", "true",
                                GrpcSwitch.PROPERTY, "true",
                                DevMode.PROPERTY, "true",
                                SleepModeSwitch.PROPERTY, "false",
                                ServerLoras.RUNTIME_PROPERTY, "false")
                        .forEach((key, value) -> PREVIOUS.put(key, System.setProperty(key, value)));
                return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so the sleep-mode-off application serves this class alone. */
    public static class Profile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
