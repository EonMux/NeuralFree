package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.grpc.GenerateRequest;
import com.eonmux.neuralfree.grpc.InferenceGrpc;
import grpc.health.v1.HealthGrpc;
import grpc.health.v1.HealthOuterClass.HealthCheckRequest;
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

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The gRPC surface with the launch default, {@code --grpc} off: generation, the standard health Check and server
 * reflection all answer the {@code UNIMPLEMENTED} of a service the server does not serve, so a plain HTTP launch
 * exposes no gRPC on its port. The launch names no model, so a call that reached a service would report the unloaded
 * engine instead; the refusal comes from the ingress before any service.
 */
@QuarkusTest
@QuarkusTestResource(value = GrpcDisabledTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(GrpcDisabledTest.Profile.class)
class GrpcDisabledTest {

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
    void generationIsUnreachableWithoutTheSwitch() {
        InferenceGrpc.InferenceBlockingStub inference = InferenceGrpc.newBlockingStub(channel)
                .withDeadlineAfter(GrpcTestSupport.TIMEOUT_SECONDS, TimeUnit.SECONDS);
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(GenerateRequest.newBuilder().setText("hello").build()));
        assertEquals(Status.Code.UNIMPLEMENTED, Status.fromThrowable(refused).getCode());
        assertTrue(GrpcTestSupport.description(refused).contains(GrpcSwitch.FLAG),
                GrpcTestSupport.description(refused));
        assertTrue(GrpcTestSupport.description(refused).contains(GrpcSwitch.PROPERTY),
                GrpcTestSupport.description(refused));
    }

    @Test
    void theHealthCheckIsUnreachableWithoutTheSwitch() {
        HealthGrpc.HealthBlockingStub health = HealthGrpc.newBlockingStub(channel)
                .withDeadlineAfter(GrpcTestSupport.TIMEOUT_SECONDS, TimeUnit.SECONDS);
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                () -> health.check(HealthCheckRequest.getDefaultInstance()));
        assertEquals(Status.Code.UNIMPLEMENTED, Status.fromThrowable(refused).getCode());
    }

    @Test
    void serverReflectionIsUnreachableWithoutTheSwitch() {
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                () -> GrpcTestSupport.listServices(channel, null));
        assertEquals(Status.Code.UNIMPLEMENTED, Status.fromThrowable(refused).getCode());
    }

    /** Leaves the gRPC switch off before Quarkus starts. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            PREVIOUS.put(GrpcSwitch.PROPERTY, System.setProperty(GrpcSwitch.PROPERTY, "false"));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so the default (gRPC-off) application serves this class alone. */
    public static class Profile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
