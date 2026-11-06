package com.eonmux.neuralfree.server;

import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.Status;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The ingress decisions without a transport: the surface is closed while the switch is off, the configured key is
 * required on generation and reflection, the standard health Check stays open, and the key parses from either the
 * {@code authorization} or the {@code x-api-key} metadata.
 */
class GrpcAccessInterceptorTest {

    private static final String GENERATE = "neuralfree.Inference/Generate";
    private static final String REFLECTION = "grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo";
    private static final String HEALTH_CHECK = "grpc.health.v1.Health/Check";
    private static final String HEALTH_WATCH = "grpc.health.v1.Health/Watch";

    @Test
    void theSurfaceIsClosedWhileTheSwitchIsOff() {
        GrpcAccessInterceptor ingress = new GrpcAccessInterceptor(GrpcSwitch.OFF, keys("secret"));
        for (String method : new String[]{GENERATE, REFLECTION, HEALTH_CHECK}) {
            RecordingCall call = new RecordingCall(method);
            AtomicBoolean reached = new AtomicBoolean();
            ingress.interceptCall(call, new Metadata(), handler(reached));
            assertEquals(Status.Code.UNIMPLEMENTED, call.closed.getCode(), method);
            assertFalse(reached.get(), method);
        }
    }

    @Test
    void withoutKeysEveryCallProceeds() {
        GrpcAccessInterceptor ingress = new GrpcAccessInterceptor(GrpcSwitch.ON, ApiKeys.from(Map.of(), new Properties()));
        RecordingCall call = new RecordingCall(GENERATE);
        AtomicBoolean reached = new AtomicBoolean();
        ingress.interceptCall(call, new Metadata(), handler(reached));
        assertNull(call.closed);
        assertTrue(reached.get());
    }

    @Test
    void aMissingOrWrongKeyIsUnauthenticated() {
        GrpcAccessInterceptor ingress = new GrpcAccessInterceptor(GrpcSwitch.ON, keys("secret"));
        assertUnauthenticated(ingress, GENERATE, new Metadata());
        assertUnauthenticated(ingress, GENERATE, bearer("wrong"));
        assertUnauthenticated(ingress, REFLECTION, new Metadata());
    }

    @Test
    void theRightKeyProceedsFromEitherHeader() {
        GrpcAccessInterceptor ingress = new GrpcAccessInterceptor(GrpcSwitch.ON, keys("secret"));
        assertProceeds(ingress, GENERATE, bearer("secret"));

        Metadata xApiKey = new Metadata();
        xApiKey.put(Metadata.Key.of("x-api-key", Metadata.ASCII_STRING_MARSHALLER), "secret");
        assertProceeds(ingress, REFLECTION, xApiKey);
    }

    @Test
    void onlyTheHealthCheckStaysOpen() {
        GrpcAccessInterceptor ingress = new GrpcAccessInterceptor(GrpcSwitch.ON, keys("secret"));
        assertProceeds(ingress, HEALTH_CHECK, new Metadata());
        assertUnauthenticated(ingress, HEALTH_WATCH, new Metadata());
    }

    @Test
    void everyKeyIsCompared() {
        ApiKeys both = ApiKeys.from(Map.of(ApiKeys.ENV, "first,second"), new Properties());
        GrpcAccessInterceptor ingress = new GrpcAccessInterceptor(GrpcSwitch.ON, both);
        assertProceeds(ingress, GENERATE, bearer("first"));
        assertProceeds(ingress, GENERATE, bearer("second"));
        assertUnauthenticated(ingress, GENERATE, bearer("third"));
    }

    private static ApiKeys keys(String key) {
        return ApiKeys.from(Map.of(ApiKeys.ENV, key), new Properties());
    }

    private static Metadata bearer(String key) {
        return GrpcTestSupport.bearer(key);
    }

    private static ServerCallHandler<String, String> handler(AtomicBoolean reached) {
        return (call, headers) -> {
            reached.set(true);
            return new ServerCall.Listener<String>() {
            };
        };
    }

    private static void assertProceeds(GrpcAccessInterceptor ingress, String method, Metadata headers) {
        RecordingCall call = new RecordingCall(method);
        AtomicBoolean reached = new AtomicBoolean();
        ingress.interceptCall(call, headers, handler(reached));
        assertNull(call.closed, method);
        assertTrue(reached.get(), method);
    }

    private static void assertUnauthenticated(GrpcAccessInterceptor ingress, String method, Metadata headers) {
        RecordingCall call = new RecordingCall(method);
        AtomicBoolean reached = new AtomicBoolean();
        ingress.interceptCall(call, headers, handler(reached));
        assertEquals(Status.Code.UNAUTHENTICATED, call.closed.getCode(), method);
        assertFalse(reached.get(), method);
    }

    /** A call that records how the ingress closed it. */
    private static final class RecordingCall extends ServerCall<String, String> {

        private static final MethodDescriptor.Marshaller<String> MARSHALLER =
                new MethodDescriptor.Marshaller<>() {
                    @Override
                    public InputStream stream(String value) {
                        return new ByteArrayInputStream(value.getBytes(StandardCharsets.UTF_8));
                    }

                    @Override
                    public String parse(InputStream stream) {
                        return "";
                    }
                };

        private final MethodDescriptor<String, String> method;
        private Status closed;

        RecordingCall(String fullMethodName) {
            this.method = MethodDescriptor.<String, String>newBuilder()
                    .setType(MethodDescriptor.MethodType.UNARY)
                    .setFullMethodName(fullMethodName)
                    .setRequestMarshaller(MARSHALLER)
                    .setResponseMarshaller(MARSHALLER)
                    .build();
        }

        @Override
        public void request(int numMessages) {
        }

        @Override
        public void sendHeaders(Metadata headers) {
        }

        @Override
        public void sendMessage(String message) {
        }

        @Override
        public void close(Status status, Metadata trailers) {
            closed = status;
        }

        @Override
        public boolean isCancelled() {
            return false;
        }

        @Override
        public MethodDescriptor<String, String> getMethodDescriptor() {
            return method;
        }
    }
}
