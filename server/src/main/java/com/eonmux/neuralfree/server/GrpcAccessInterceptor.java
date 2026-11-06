package com.eonmux.neuralfree.server;

import grpc.health.v1.HealthGrpc;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import io.quarkus.grpc.GlobalInterceptor;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

/**
 * The gRPC ingress of the server, the one interceptor every gRPC call passes: it closes the whole surface with
 * {@code UNIMPLEMENTED} while {@link GrpcSwitch} is off, and otherwise requires the HTTP API key on every call when
 * {@link ApiKeys} names one. The key travels in the {@code authorization: Bearer <key>} metadata (or
 * {@code x-api-key}), parses with {@link ApiKeys#presented} and compares in constant time with {@link ApiKeys#accepts}
 * as {@link ApiKeyFilter} does, so one launch's key set protects both transports; a call without it answers
 * {@code UNAUTHENTICATED}. Only the standard {@code grpc.health.v1.Health/Check} stays open, and only while the HTTP
 * {@code /health} probe is open ({@link ApiKeyFilter#open}), so a probe that works over HTTP works over gRPC.
 * Server reflection is intercepted like every service and needs the key with the rest.
 *
 * <p>The interceptor is global, so it also covers the gRPC health service and server reflection, and it runs before
 * the call reaches a service implementation. Immutable.</p>
 */
@GlobalInterceptor
@Singleton
public class GrpcAccessInterceptor implements ServerInterceptor {

    /** The method the standard health probe calls; the only one the API key does not cover. */
    static final String HEALTH_CHECK = HealthGrpc.SERVICE_NAME + "/Check";

    /** The metadata carrying the API key as {@code Bearer <key>} (OpenAI, llama.cpp, vLLM). */
    static final Metadata.Key<String> AUTHORIZATION =
            Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER);

    /** The metadata carrying the API key directly (Anthropic's {@code x-api-key}). */
    static final Metadata.Key<String> API_KEY =
            Metadata.Key.of("x-api-key", Metadata.ASCII_STRING_MARSHALLER);

    private final GrpcSwitch grpc;
    private final ApiKeys keys;

    /** The ingress of the launch's switches and keys. */
    @Inject
    public GrpcAccessInterceptor(GrpcSwitch grpc) {
        this(grpc, ApiKeys.from(System.getenv(), System.getProperties()));
    }

    /** Test and manual wiring over explicit switches and keys. */
    GrpcAccessInterceptor(GrpcSwitch grpc, ApiKeys keys) {
        this.grpc = grpc;
        this.keys = keys;
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call, Metadata headers,
                                                                 ServerCallHandler<ReqT, RespT> next) {
        Status refusal = refusal(call.getMethodDescriptor(), headers);
        if (refusal != null) {
            call.close(refusal, new Metadata());
            return new ServerCall.Listener<ReqT>() {
            };
        }
        return next.startCall(call, headers);
    }

    /** The status that closes {@code method}, or null when the call may proceed. */
    private Status refusal(MethodDescriptor<?, ?> method, Metadata headers) {
        if (!grpc.enabled()) {
            return Status.UNIMPLEMENTED.withDescription("the gRPC adapter is off: start the server with "
                    + GrpcSwitch.FLAG + " or -D" + GrpcSwitch.PROPERTY + "=true (vLLM serves gRPC only with"
                    + " vllm serve --grpc)");
        }
        if (!keys.enabled()) {
            return null;
        }
        if (ApiKeyFilter.open("health") && HEALTH_CHECK.equals(method.getFullMethodName())) {
            return null;
        }
        if (keys.accepts(ApiKeys.presented(headers.get(AUTHORIZATION), headers.get(API_KEY)))) {
            return null;
        }
        return Status.UNAUTHENTICATED.withDescription(ApiKeys.REFUSAL_MESSAGE);
    }
}
