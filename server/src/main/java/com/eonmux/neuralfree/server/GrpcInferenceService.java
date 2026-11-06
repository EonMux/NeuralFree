package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.grpc.GenerateRequest;
import com.eonmux.neuralfree.grpc.GenerateResponse;
import com.eonmux.neuralfree.grpc.InferenceGrpc;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import io.quarkus.grpc.GrpcService;
import io.smallrye.common.annotation.Blocking;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

import java.util.Map;

/**
 * The {@code neuralfree.Inference} gRPC service: vLLM's gRPC inference surface over the same engine the HTTP routes
 * use. {@code Generate} plans the request with {@link GrpcRequestPlan}, submits its sequences together through
 * {@link GenerationBatch} and answers them in one response; {@code GenerateStream} sends the prompt info first, then
 * each generated piece, and each sequence's finish info last.
 *
 * <p>The {@code model} field routes the request through the multi-model {@link ModelRouter} exactly as the HTTP
 * routes do. A client that cancels its call cancels its generations, and {@code Control.Abort} cancels them by
 * {@code request_id}. A refused request reports the status {@link GrpcFailures} maps from the same refusal the HTTP
 * surfaces answer.</p>
 */
@GrpcService
@Singleton
@Blocking
public class GrpcInferenceService extends InferenceGrpc.InferenceImplBase {

    @Inject
    EngineService service;

    /** The multi-model router; inactive outside the router mode. */
    @Inject
    ModelRouter router;

    @Inject
    GrpcInFlight inFlight;

    @Override
    public void generate(GenerateRequest request, StreamObserver<GenerateResponse> observer) {
        String id = request.getRequestId();
        GenerationBatch batch = null;
        boolean registered = false;
        try {
            EngineService target = target(request);
            GrpcRequestPlan plan = GrpcRequestPlan.of(target, request);
            batch = GenerationBatch.start(target, plan.jobs(), null,
                    plan.requestId().isEmpty() ? null : plan.requestId());
            inFlight.register(id, batch::cancel);
            registered = true;
            ServerCallStreamObserver<GenerateResponse> call = call(observer);
            if (call != null) {
                call.setOnCancelHandler(batch::cancel);
            }
            batch.awaitAll();
            GenerateResponse.Builder response = GenerateResponse.newBuilder()
                    .setPromptInfo(GrpcResponses.promptInfo(batch, plan));
            for (int i = 0; i < batch.size(); i++) {
                response.addOutputs(GrpcResponses.finalOutput(target, i, batch.result(i), batch, plan, false));
            }
            observer.onNext(response.build());
        } catch (Throwable e) {
            if (batch != null) {
                batch.cancel();
            }
            observer.onError(GrpcFailures.of(e));
            return;
        } finally {
            if (registered) {
                inFlight.remove(id);
            }
        }
        observer.onCompleted();
    }

    @Override
    public void generateStream(GenerateRequest request, StreamObserver<GenerateResponse> observer) {
        String id = request.getRequestId();
        GenerationBatch batch = null;
        boolean registered = false;
        boolean cancelled = false;
        try {
            EngineService target = target(request);
            GrpcRequestPlan plan = GrpcRequestPlan.of(target, request);
            batch = GenerationBatch.start(target, plan.jobs(), null,
                    plan.requestId().isEmpty() ? null : plan.requestId());
            inFlight.register(id, batch::cancel);
            registered = true;
            ServerCallStreamObserver<GenerateResponse> call = call(observer);
            if (call != null) {
                call.setOnCancelHandler(batch::cancel);
            }
            LlamaStreamChunks chunks = new LlamaStreamChunks(batch.size());
            boolean promptSent = false;
            int done = 0;
            while (done < batch.size()) {
                if (call != null && call.isCancelled()) {
                    cancelled = true;
                    batch.cancel();
                    break;
                }
                GenerationBatch.Event event = batch.next(50);
                if (event == null) {
                    continue;
                }
                if (!promptSent) {
                    observer.onNext(GrpcResponses.promptResponse(GrpcResponses.promptInfo(batch, plan)));
                    promptSent = true;
                }
                if (event instanceof GenerationBatch.Text text) {
                    sendChunk(observer, chunks.offer(text), plan);
                } else if (event instanceof GenerationBatch.Done terminal) {
                    done++;
                    sendChunk(observer, chunks.flush(terminal.choice()), plan);
                    GenerationResult result = batch.result(terminal.choice());
                    observer.onNext(GenerateResponse.newBuilder()
                            .addOutputs(GrpcResponses.finalOutput(target, terminal.choice(), result, batch, plan, true))
                            .build());
                }
            }
        } catch (Throwable e) {
            if (batch != null) {
                batch.cancel();
            }
            if (!cancelled) {
                observer.onError(GrpcFailures.of(e));
            }
            return;
        } finally {
            if (registered) {
                inFlight.remove(id);
            }
        }
        if (!cancelled) {
            observer.onCompleted();
        }
    }

    /** Sends one generated piece when it carries text, a token id or a log probability capture. */
    private static void sendChunk(StreamObserver<GenerateResponse> observer, GenerationBatch.Text piece,
                                  GrpcRequestPlan plan) {
        if (piece == null) {
            return;
        }
        boolean useful = plan.outputText() && !piece.text().isEmpty()
                || plan.outputTokenIds() && piece.tokenId() >= 0
                || (plan.outputLogprobs() || plan.outputCandidates()) && piece.logprobs() != null;
        if (useful) {
            observer.onNext(GenerateResponse.newBuilder()
                    .addOutputs(GrpcResponses.chunk(piece.choice(), piece, plan))
                    .build());
        }
    }

    /** The engine {@code request} serves on: the router's target for its model, else the one model of the server. */
    private EngineService target(GenerateRequest request) {
        return ModelRouter.target(router, service,
                JsonView.of(Map.of("model", request.getModel()), "request"), null);
    }

    /** {@code observer} as a server call when the transport exposes it, else null. */
    @SuppressWarnings("unchecked")
    private static ServerCallStreamObserver<GenerateResponse> call(StreamObserver<GenerateResponse> observer) {
        return observer instanceof ServerCallStreamObserver<?> server
                ? (ServerCallStreamObserver<GenerateResponse>) server : null;
    }
}
