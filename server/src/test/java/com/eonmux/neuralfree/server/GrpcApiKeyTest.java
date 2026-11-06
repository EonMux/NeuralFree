package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.grpc.GenerateRequest;
import com.eonmux.neuralfree.grpc.GenerateResponse;
import com.eonmux.neuralfree.grpc.InferenceGrpc;
import com.eonmux.neuralfree.grpc.StoppingCriteria;
import grpc.health.v1.HealthGrpc;
import grpc.health.v1.HealthOuterClass.HealthCheckRequest;
import grpc.health.v1.HealthOuterClass.HealthCheckResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Metadata;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.MetadataUtils;
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
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The API key on the gRPC surface: with {@code --grpc --api-key} every call needs the key in its {@code
 * authorization: Bearer <key>} metadata, whichever service it names. Inference and server reflection answer
 * {@code UNAUTHENTICATED} without it or with a wrong one and serve with the right one; the standard health Check
 * stays open, as the HTTP {@code /health} probe does.
 */
@QuarkusTest
@QuarkusTestResource(value = GrpcApiKeyTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(GrpcApiKeyTest.Profile.class)
class GrpcApiKeyTest {

    private static final String KEY = "grpc-secret";
    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    private static ManagedChannel channel;

    @TestHTTPResource
    URI base;

    @BeforeEach
    void connect() {
        if (channel == null) {
            channel = ManagedChannelBuilder.forAddress(base.getHost(), base.getPort()).usePlaintext().build();
        }
    }

    @AfterAll
    static void close() {
        if (channel != null) {
            channel.shutdownNow();
            channel = null;
        }
        PREVIOUS.forEach((name, value) -> {
            if (value == null) {
                System.clearProperty(name);
            } else {
                System.setProperty(name, value);
            }
        });
    }

    @Test
    void callsWithoutOrWithAWrongKeyAreUnauthenticated() {
        assertUnauthenticated(inference(null));
        assertUnauthenticated(inference(GrpcTestSupport.bearer("wrong")));
    }

    @Test
    void serverReflectionNeedsTheKeyToo() {
        StatusRuntimeException missing = assertThrows(StatusRuntimeException.class,
                () -> GrpcTestSupport.listServices(channel, null));
        assertEquals(Status.Code.UNAUTHENTICATED, Status.fromThrowable(missing).getCode());

        StatusRuntimeException wrong = assertThrows(StatusRuntimeException.class,
                () -> GrpcTestSupport.listServices(channel, GrpcTestSupport.bearer("wrong")));
        assertEquals(Status.Code.UNAUTHENTICATED, Status.fromThrowable(wrong).getCode());
    }

    @Test
    void theRightKeyServesGeneration() {
        GenerateResponse response = inference(GrpcTestSupport.bearer(KEY))
                .generate(GenerateRequest.newBuilder()
                        .setText("hello")
                        .setTemperature(0f)
                        .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(4).setIgnoreEos(true))
                        .build());
        assertEquals(1, response.getOutputsCount());
        assertTrue(response.getPromptInfo().getNumPromptTokens() > 0);
    }

    @Test
    void theRightKeyServesReflection() throws Exception {
        List<String> services = GrpcTestSupport.listServices(channel, GrpcTestSupport.bearer(KEY));
        assertTrue(services.contains("neuralfree.Inference"), services.toString());
        assertTrue(services.contains("neuralfree.Control"), services.toString());
    }

    @Test
    void theHealthCheckStaysOpen() {
        HealthGrpc.HealthBlockingStub health = HealthGrpc.newBlockingStub(channel)
                .withDeadlineAfter(GrpcTestSupport.TIMEOUT_SECONDS, TimeUnit.SECONDS);
        HealthCheckResponse response = health.check(HealthCheckRequest.getDefaultInstance());
        assertEquals(HealthCheckResponse.ServingStatus.SERVING, response.getStatus());

        Metadata wrong = GrpcTestSupport.bearer("wrong");
        HealthCheckResponse stillOpen = health
                .withInterceptors(MetadataUtils.newAttachHeadersInterceptor(wrong))
                .check(HealthCheckRequest.getDefaultInstance());
        assertEquals(HealthCheckResponse.ServingStatus.SERVING, stillOpen.getStatus());
    }

    private static InferenceGrpc.InferenceBlockingStub inference(Metadata headers) {
        InferenceGrpc.InferenceBlockingStub stub = InferenceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(GrpcTestSupport.TIMEOUT_SECONDS, TimeUnit.SECONDS);
        return headers == null ? stub : stub.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers));
    }

    private static void assertUnauthenticated(InferenceGrpc.InferenceBlockingStub inference) {
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(GenerateRequest.newBuilder().setText("hello").build()));
        assertEquals(Status.Code.UNAUTHENTICATED, Status.fromThrowable(refused).getCode());
        assertTrue(GrpcTestSupport.description(refused).contains(ApiKeys.REFUSAL_MESSAGE),
                GrpcTestSupport.description(refused));
    }

    /** Turns the gRPC surface and an API key on before Quarkus starts, over a synthetic model. */
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
                                ApiKeys.PROPERTY, KEY)
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

    /** A profile of its own, so the keyed application serves this class alone. */
    public static class Profile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
