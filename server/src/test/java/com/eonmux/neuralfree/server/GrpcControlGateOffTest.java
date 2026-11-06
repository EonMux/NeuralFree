package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.grpc.AbortRequest;
import com.eonmux.neuralfree.grpc.ControlGrpc;
import com.eonmux.neuralfree.grpc.GenerateRequest;
import com.eonmux.neuralfree.grpc.GetKvEventSourcesRequest;
import com.eonmux.neuralfree.grpc.GetModelInfoRequest;
import com.eonmux.neuralfree.grpc.GetServerInfoRequest;
import com.eonmux.neuralfree.grpc.InferenceGrpc;
import com.eonmux.neuralfree.grpc.IsPausedRequest;
import com.eonmux.neuralfree.grpc.IsSleepingRequest;
import com.eonmux.neuralfree.grpc.ListLorasRequest;
import com.eonmux.neuralfree.grpc.LoadLoraRequest;
import com.eonmux.neuralfree.grpc.PauseGenerationRequest;
import com.eonmux.neuralfree.grpc.ResumeGenerationRequest;
import com.eonmux.neuralfree.grpc.SleepRequest;
import com.eonmux.neuralfree.grpc.StoppingCriteria;
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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Control gates with every switch off (the launch defaults), the gRPC surface on: the calls whose HTTP routes
 * are vLLM's development routes answer {@code NOT_FOUND} naming {@code --dev-mode}, the LoRA load and unload answer
 * {@code NOT_FOUND} naming the runtime-updating switch, and the calls outside those gates — model discovery, LoRA
 * listing, KV event discovery and generation — still serve. A model is loaded, so a call that reached its handler
 * would answer its own result rather than an unloaded-engine refusal.
 */
@QuarkusTest
@QuarkusTestResource(value = GrpcControlGateOffTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(GrpcControlGateOffTest.Profile.class)
class GrpcControlGateOffTest {

    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    private static ManagedChannel channel;
    private static ControlGrpc.ControlBlockingStub control;
    private static InferenceGrpc.InferenceBlockingStub inference;

    @TestHTTPResource
    URI base;

    @BeforeEach
    void connect() {
        if (channel == null) {
            channel = ManagedChannelBuilder.forAddress(base.getHost(), base.getPort()).usePlaintext().build();
            control = ControlGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(GrpcTestSupport.TIMEOUT_SECONDS, TimeUnit.SECONDS);
            inference = InferenceGrpc.newBlockingStub(channel)
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
        inference = null;
        PREVIOUS.forEach((name, value) -> {
            if (value == null) {
                System.clearProperty(name);
            } else {
                System.setProperty(name, value);
            }
        });
    }

    @Test
    void theDevelopmentCallsAnswerNotFound() {
        assertNotFound(DevMode.FLAG,
                () -> control.getServerInfo(GetServerInfoRequest.getDefaultInstance()));
        assertNotFound(DevMode.FLAG,
                () -> control.abort(AbortRequest.newBuilder().addRequestIds("r1").build()));
        assertNotFound(DevMode.FLAG,
                () -> control.pauseGeneration(PauseGenerationRequest.getDefaultInstance()));
        assertNotFound(DevMode.FLAG,
                () -> control.resumeGeneration(ResumeGenerationRequest.getDefaultInstance()));
        assertNotFound(DevMode.FLAG,
                () -> control.isPaused(IsPausedRequest.getDefaultInstance()));
        assertNotFound(DevMode.FLAG,
                () -> control.sleep(SleepRequest.newBuilder().setLevel(0).build()));
        assertNotFound(DevMode.FLAG,
                () -> control.wakeUp(WakeUpRequest.getDefaultInstance()));
        assertNotFound(DevMode.FLAG,
                () -> control.isSleeping(IsSleepingRequest.getDefaultInstance()));
    }

    @Test
    void theLoraChangesAnswerNotFoundWithoutTheRuntimeSwitch() {
        assertNotFound(ServerLoras.RUNTIME_PROPERTY,
                () -> control.loadLora(LoadLoraRequest.newBuilder()
                        .setLoraName("adapter").setSourcePath("/tmp/adapter.gguf").build()));
        assertNotFound(ServerLoras.RUNTIME_PROPERTY,
                () -> control.unloadLora(UnloadLoraRequest.newBuilder().setLoraName("adapter").build()));
    }

    @Test
    void discoveryListingAndGenerationStayReachable() {
        assertEquals(true, control.getModelInfo(GetModelInfoRequest.getDefaultInstance())
                .getSupportsTextInput());
        assertEquals(0, control.listLoras(ListLorasRequest.getDefaultInstance()).getAdaptersCount());
        assertEquals(0, control.getKvEventSources(GetKvEventSourcesRequest.getDefaultInstance()).getSourcesCount());
        assertEquals(1, inference.generate(GenerateRequest.newBuilder()
                .setText("hello").setTemperature(0f)
                .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(2).setIgnoreEos(true))
                .build()).getOutputsCount());
    }

    private static void assertNotFound(String hint, Runnable call) {
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class, call::run);
        assertEquals(Status.Code.NOT_FOUND, Status.fromThrowable(refused).getCode(),
                GrpcTestSupport.description(refused));
        assertTrue(GrpcTestSupport.description(refused).contains(hint), GrpcTestSupport.description(refused));
    }

    /** Leaves the gRPC surface on and every Control switch off before Quarkus starts. */
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
                                DevMode.PROPERTY, "false",
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

    /** A profile of its own, so the gates-off application serves this class alone. */
    public static class Profile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
