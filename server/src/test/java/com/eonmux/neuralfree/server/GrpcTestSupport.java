package com.eonmux.neuralfree.server;

import grpc.reflection.v1alpha.Reflection.ServerReflectionRequest;
import grpc.reflection.v1alpha.Reflection.ServerReflectionResponse;
import grpc.reflection.v1alpha.Reflection.ServiceResponse;
import grpc.reflection.v1alpha.ServerReflectionGrpc;
import io.grpc.ManagedChannel;
import io.grpc.Metadata;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.MetadataUtils;
import io.grpc.stub.StreamObserver;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** Shared helpers of the gRPC tests: key metadata, status text and the server-reflection client calls. */
final class GrpcTestSupport {

    static final long TIMEOUT_SECONDS = 120;

    private GrpcTestSupport() {
    }

    /** The {@code authorization: Bearer <key>} metadata a keyed call carries. */
    static Metadata bearer(String key) {
        Metadata headers = new Metadata();
        headers.put(Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER), "Bearer " + key);
        return headers;
    }

    /** The description of a refused call, empty when the status carries none. */
    static String description(StatusRuntimeException error) {
        String description = io.grpc.Status.fromThrowable(error).getDescription();
        return description == null ? "" : description;
    }

    /**
     * Lists the services server reflection reports, calling without headers when {@code headers} is null.
     *
     * @throws StatusRuntimeException when the reflection call is refused
     */
    static List<String> listServices(ManagedChannel channel, Metadata headers) throws InterruptedException {
        ServerReflectionGrpc.ServerReflectionStub stub = ServerReflectionGrpc.newStub(channel)
                .withDeadlineAfter(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        if (headers != null) {
            stub = stub.withInterceptors(MetadataUtils.newAttachHeadersInterceptor(headers));
        }
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        LinkedBlockingQueue<ServerReflectionResponse> responses = new LinkedBlockingQueue<>();
        StreamObserver<ServerReflectionRequest> requests = stub.serverReflectionInfo(new StreamObserver<>() {
            @Override
            public void onNext(ServerReflectionResponse response) {
                responses.add(response);
            }

            @Override
            public void onError(Throwable error) {
                failure.set(error);
                done.countDown();
            }

            @Override
            public void onCompleted() {
                done.countDown();
            }
        });
        requests.onNext(ServerReflectionRequest.newBuilder().setListServices("*").build());
        requests.onCompleted();
        if (!done.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            throw new IllegalStateException("the reflection call did not answer within " + TIMEOUT_SECONDS + "s");
        }
        Throwable error = failure.get();
        if (error instanceof StatusRuntimeException status) {
            throw status;
        }
        if (error != null) {
            throw new IllegalStateException("the reflection call failed", error);
        }
        ServerReflectionResponse response = responses.poll();
        if (response == null || !response.hasListServicesResponse()) {
            throw new IllegalStateException("the reflection call answered no service list");
        }
        List<String> names = new ArrayList<>();
        for (ServiceResponse service : response.getListServicesResponse().getServiceList()) {
            names.add(service.getName());
        }
        return names;
    }
}
