package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.grpc.AbortRequest;
import com.eonmux.neuralfree.grpc.CandidateTokens;
import com.eonmux.neuralfree.grpc.ControlGrpc;
import com.eonmux.neuralfree.grpc.DecodingParameters;
import com.eonmux.neuralfree.grpc.FinishInfo;
import com.eonmux.neuralfree.grpc.GenerateRequest;
import com.eonmux.neuralfree.grpc.GenerateResponse;
import com.eonmux.neuralfree.grpc.GetKvEventSourcesRequest;
import com.eonmux.neuralfree.grpc.GetModelInfoRequest;
import com.eonmux.neuralfree.grpc.GetServerInfoRequest;
import com.eonmux.neuralfree.grpc.InferenceGrpc;
import com.eonmux.neuralfree.grpc.IsPausedRequest;
import com.eonmux.neuralfree.grpc.IsSleepingRequest;
import com.eonmux.neuralfree.grpc.KVCacheParameters;
import com.eonmux.neuralfree.grpc.ListLorasRequest;
import com.eonmux.neuralfree.grpc.LoadLoraRequest;
import com.eonmux.neuralfree.grpc.ModelInfo;
import com.eonmux.neuralfree.grpc.PauseGenerationRequest;
import com.eonmux.neuralfree.grpc.PauseMode;
import com.eonmux.neuralfree.grpc.ResumeGenerationRequest;
import com.eonmux.neuralfree.grpc.ResponseOptions;
import com.eonmux.neuralfree.grpc.SequenceOutput;
import com.eonmux.neuralfree.grpc.ServerInfo;
import com.eonmux.neuralfree.grpc.SleepRequest;
import com.eonmux.neuralfree.grpc.StoppingCriteria;
import com.eonmux.neuralfree.grpc.TokenIds;
import com.eonmux.neuralfree.grpc.WakeUpRequest;
import com.eonmux.neuralfree.grpc.WakeUpResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The gRPC serving adapter over the wire: a real server on the test HTTP port (the unified server carries gRPC on
 * the HTTP port), a real {@link EngineService} over a synthetic GGUF, and a grpc-java client. It exercises the
 * {@code Inference} surface's unary and streamed generation, its token-prompt, truncation and logprobs settings, its
 * refusals by name, the {@code Control} discovery, LoRA, KV event and lifecycle calls, and per-request aborts. The
 * launch turns the gRPC surface, the development routes and sleep mode on, so the lifecycle calls reach their
 * handlers; the gates themselves are exercised by {@link GrpcControlGateOffTest} and its siblings.
 */
@QuarkusTest
@QuarkusTestResource(value = GrpcInferenceTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(GrpcInferenceTest.Profile.class)
class GrpcInferenceTest {

    private static final long TIMEOUT_SECONDS = 120;

    /** The values the launch properties had before this class set them, restored before a later class reads them. */
    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    private static ManagedChannel channel;
    private static InferenceGrpc.InferenceBlockingStub inference;
    private static ControlGrpc.ControlBlockingStub control;

    @TestHTTPResource
    URI base;

    @BeforeEach
    void connect() {
        if (channel == null) {
            channel = ManagedChannelBuilder.forAddress(base.getHost(), base.getPort()).usePlaintext().build();
            inference = InferenceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            control = ControlGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        }
    }

    @AfterAll
    static void close() {
        if (channel != null) {
            channel.shutdownNow();
            channel = null;
        }
        inference = null;
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
    void generateAnswersTextTokenIdsAndAFinish() {
        GenerateResponse response = inference.generate(prompt("hello").setRequestId("gen-1")
                .setResponse(ResponseOptions.newBuilder().setOutputTokenIds(true).setOutputLogprobs(true))
                .build());
        assertTrue(response.hasPromptInfo());
        assertTrue(response.getPromptInfo().getNumPromptTokens() > 0);
        assertEquals(1, response.getOutputsCount());
        SequenceOutput output = response.getOutputs(0);
        assertEquals(0, output.getIndex());
        assertEquals(4, output.getNumTokens());
        assertEquals(4, output.getTokenIdsCount());
        assertEquals(4, output.getLogprobsCount());
        assertEquals(output.getTokenIds(0), output.getLogprobs(0).getTokenId());
        assertTrue(output.hasFinishInfo());
        assertEquals(FinishInfo.FinishReason.LENGTH, output.getFinishInfo().getFinishReason());
        assertEquals(4, output.getFinishInfo().getNumOutputTokens());
    }

    @Test
    void aTokenPromptContinuesLikeTheTextItEncodes() {
        GenerateResponse text = inference.generate(prompt("token prompt")
                .setResponse(ResponseOptions.newBuilder().setOutputTokenIds(true).setPromptTokenIds(true)).build());
        List<Integer> tokens = text.getPromptInfo().getTokenIdsList();
        assertFalse(tokens.isEmpty());
        GenerateResponse fromTokens = inference.generate(GenerateRequest.newBuilder()
                .setTokenIds(TokenIds.newBuilder().addAllIds(tokens))
                .setTemperature(0f)
                .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(4).setIgnoreEos(true))
                .setResponse(ResponseOptions.newBuilder().setOutputTokenIds(true))
                .build());
        assertEquals(text.getOutputs(0).getTokenIdsList(), fromTokens.getOutputs(0).getTokenIdsList());
        assertEquals(tokens.size(), fromTokens.getPromptInfo().getNumPromptTokens());
    }

    @Test
    void generateStreamSendsThePromptThePiecesAndTheFinish() {
        GenerateRequest request = prompt("stream me")
                .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(5).setIgnoreEos(true))
                .setResponse(ResponseOptions.newBuilder().setOutputTokenIds(true))
                .build();
        GenerateResponse whole = inference.generate(request);
        StringBuilder streamed = new StringBuilder();
        int pieces = 0;
        FinishInfo finish = null;
        boolean promptSeen = false;
        Iterator<GenerateResponse> stream = inference.generateStream(request);
        while (stream.hasNext()) {
            GenerateResponse response = stream.next();
            if (response.hasPromptInfo()) {
                assertFalse(promptSeen, "the prompt info comes once, first");
                promptSeen = true;
            }
            for (SequenceOutput output : response.getOutputsList()) {
                streamed.append(output.getText());
                if (output.hasFinishInfo()) {
                    finish = output.getFinishInfo();
                } else {
                    pieces++;
                }
            }
        }
        assertTrue(promptSeen);
        assertTrue(pieces > 0, "the stream carries generated pieces");
        assertNotNull(finish, "the stream ends with the finish info");
        assertEquals(FinishInfo.FinishReason.LENGTH, finish.getFinishReason());
        assertEquals(whole.getOutputs(0).getText(), streamed.toString());
        assertEquals(5, finish.getNumOutputTokens());
    }

    @Test
    void truncatePromptTokensKeepsTheLeadingTokens() {
        GenerateResponse whole = inference.generate(prompt("one two three").build());
        GenerateResponse cut = inference.generate(prompt("one two three")
                .setTruncatePromptTokens(1).setResponse(ResponseOptions.newBuilder().setPromptTokenIds(true)).build());
        assertEquals(1, cut.getPromptInfo().getNumPromptTokens());
        assertTrue(cut.getPromptInfo().getNumPromptTokens() < whole.getPromptInfo().getNumPromptTokens());
    }

    @Test
    void everySequenceAnswersWithItsOwnIndex() {
        GenerateResponse response = inference.generate(prompt("parallel")
                .setSampling(com.eonmux.neuralfree.grpc.RandomSampling.newBuilder()
                        .setNumSequences(2).setSeed(42))
                .setResponse(ResponseOptions.newBuilder().setOutputTokenIds(true))
                .build());
        assertEquals(2, response.getOutputsCount());
        assertEquals(0, response.getOutputs(0).getIndex());
        assertEquals(1, response.getOutputs(1).getIndex());
        for (SequenceOutput output : response.getOutputsList()) {
            assertEquals(4, output.getNumTokens());
            assertEquals(4, output.getTokenIdsCount());
            assertTrue(output.hasFinishInfo());
        }
    }

    @Test
    void promptLogprobsFollowThePromptTokens() {
        GenerateResponse response = inference.generate(prompt("hello world")
                .setResponse(ResponseOptions.newBuilder().setPromptTokenIds(true).setPromptLogprobs(true))
                .build());
        List<Integer> tokens = response.getPromptInfo().getTokenIdsList();
        assertEquals(response.getPromptInfo().getNumPromptTokens(), tokens.size());
        assertEquals(tokens.size() - 1, response.getPromptInfo().getLogprobsCount());
        for (var logprob : response.getPromptInfo().getLogprobsList()) {
            assertTrue(logprob.getTokenIndex() > 0);
            assertEquals(tokens.get(logprob.getTokenIndex()), logprob.getToken().getTokenId());
        }
    }

    @Test
    void outputCandidatesReportTheAlternatives() {
        GenerateResponse response = inference.generate(prompt("candidates")
                .setResponse(ResponseOptions.newBuilder().setOutputCandidates(
                        CandidateTokens.newBuilder().setTopN(2)))
                .build());
        assertEquals(4, response.getOutputs(0).getLogprobsCount());
        for (var logprob : response.getOutputs(0).getLogprobsList()) {
            assertEquals(2, logprob.getCandidatesCount());
        }
    }

    @Test
    void outputTextMayBeSuppressedWhileTokensStay() {
        GenerateResponse response = inference.generate(prompt("quiet")
                .setResponse(ResponseOptions.newBuilder().setOutputText(false).setOutputTokenIds(true))
                .build());
        assertEquals("", response.getOutputs(0).getText());
        assertEquals(4, response.getOutputs(0).getTokenIdsCount());
    }

    @Test
    void aStopStringEndsTheGeneration() {
        GenerateResponse first = inference.generate(GenerateRequest.newBuilder()
                .setText("stop")
                .setTemperature(0f)
                .setDecoding(DecodingParameters.newBuilder().addAllowedTokenIds(20))
                .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(1).setIgnoreEos(true))
                .build());
        String piece = first.getOutputs(0).getText();
        assertFalse(piece.isEmpty(), "the forced token renders as text");
        GenerateResponse stopped = inference.generate(GenerateRequest.newBuilder()
                .setText("stop")
                .setTemperature(0f)
                .setDecoding(DecodingParameters.newBuilder().addAllowedTokenIds(20))
                .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(8).setIgnoreEos(true)
                        .addStopStrings(piece))
                .build());
        FinishInfo finish = stopped.getOutputs(0).getFinishInfo();
        assertEquals(FinishInfo.FinishReason.STOP, finish.getFinishReason());
        assertEquals(piece, finish.getStopString());
    }

    @Test
    void cacheSaltsAreAcceptedAndInvalidOnesRefused() {
        assertEquals(1, inference.generate(prompt("salt")
                .setKv(KVCacheParameters.newBuilder().setCacheSalt("tenant-a")).build()).getOutputsCount());
        assertEquals(1, inference.generate(prompt("salt")
                .setKv(KVCacheParameters.newBuilder().setBypassPrefixCache(true)).build()).getOutputsCount());
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(prompt("salt")
                        .setKv(KVCacheParameters.newBuilder().setCacheSalt("bad/salt")).build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(refused).getCode());
        assertTrue(description(refused).contains("cache_salt"), description(refused));
    }

    @Test
    void unsupportedAndInvalidInputsAreRefusedByName() {
        StatusRuntimeException noPrompt = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(GenerateRequest.newBuilder().build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(noPrompt).getCode());
        assertTrue(description(noPrompt).contains("prompt"), description(noPrompt));

        StatusRuntimeException noText = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(prompt("hello")
                        .setResponse(ResponseOptions.newBuilder().setOutputText(false))
                        .setStopping(StoppingCriteria.newBuilder().addStopStrings("x")).build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(noText).getCode());
        assertTrue(description(noText).contains("stop_strings"), description(noText));

        StatusRuntimeException all = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(prompt("hello")
                        .setResponse(ResponseOptions.newBuilder()
                                .setOutputCandidates(CandidateTokens.newBuilder().setAll(true))).build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(all).getCode());
        assertTrue(description(all).contains("output_candidates.all"), description(all));

        StatusRuntimeException token = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(GenerateRequest.newBuilder()
                        .setTokenIds(TokenIds.newBuilder().addIds(1_000_000)).build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(token).getCode());
        assertTrue(description(token).contains("token_ids"), description(token));

        StatusRuntimeException lora = assertThrows(StatusRuntimeException.class,
                () -> inference.generate(prompt("hello").setLoraName("missing").build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(lora).getCode());
        assertTrue(description(lora).contains("lora_name"), description(lora));
    }

    @Test
    void controlDiscoversTheServerAndModel() {
        inference.generate(prompt("warm").build()); // builds the batch engine, so the slots are known
        ServerInfo server = control.getServerInfo(GetServerInfoRequest.getDefaultInstance());
        assertEquals(ServerVersion.VALUE, server.getEngineVersion());
        assertEquals(GrpcControlService.API_VERSION, server.getApiVersion());
        assertEquals(GrpcControlService.PROTO_VERSION, server.getProtoVersion());
        assertEquals(GrpcControlService.INSTANCE_ID, server.getInstanceId());
        assertEquals(1, server.getParallelism().getTensorParallelSize());
        assertEquals(1, server.getParallelism().getDataParallelSize());
        assertTrue(server.getMaxModelLen() > 0);
        assertTrue(server.getMaxRunningRequests() > 0);

        ModelInfo model = control.getModelInfo(GetModelInfoRequest.getDefaultInstance());
        assertFalse(model.getServedModelName().isEmpty());
        assertFalse(model.getModelId().isEmpty());
        assertTrue(model.getSupportsTextInput());
        assertTrue(model.getSupportsTokenIdsInput());
        assertFalse(model.getSupportsMultimodal());
        assertFalse(model.getSupportsLora());

        assertEquals(0, control.listLoras(ListLorasRequest.getDefaultInstance()).getAdaptersCount());
        assertEquals(0, control.getKvEventSources(GetKvEventSourcesRequest.getDefaultInstance()).getSourcesCount());
    }

    @Test
    void loraLoadingIsRefusedWithoutRuntimeUpdating() {
        StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                () -> control.loadLora(LoadLoraRequest.newBuilder()
                        .setLoraName("adapter").setSourcePath("/tmp/adapter.gguf").build()));
        assertEquals(Status.Code.NOT_FOUND, Status.fromThrowable(refused).getCode());
        assertTrue(description(refused).contains(ServerLoras.RUNTIME_PROPERTY), description(refused));
    }

    @Test
    void sleepRefusesGenerationAndWakeRestoresIt() {
        control.sleep(SleepRequest.newBuilder().setLevel(0).build());
        try {
            assertTrue(control.isSleeping(IsSleepingRequest.getDefaultInstance()).getSleeping());
            StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                    () -> inference.generate(prompt("hello").build()));
            assertEquals(Status.Code.UNAVAILABLE, Status.fromThrowable(refused).getCode());
        } finally {
            WakeUpResponse wake = control.wakeUp(WakeUpRequest.getDefaultInstance());
            assertEquals("awake", wake.getStatus());
        }
        assertFalse(control.isSleeping(IsSleepingRequest.getDefaultInstance()).getSleeping());
        assertEquals(1, inference.generate(prompt("after").build()).getOutputsCount());
    }

    @Test
    void pauseRefusesGenerationAndResumeRestoresIt() {
        control.pauseGeneration(PauseGenerationRequest.newBuilder()
                .setMode(PauseMode.PAUSE_MODE_ABORT).build());
        try {
            assertTrue(control.isPaused(IsPausedRequest.getDefaultInstance()).getPaused());
            StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                    () -> inference.generate(prompt("hello").build()));
            assertEquals(Status.Code.UNAVAILABLE, Status.fromThrowable(refused).getCode());
        } finally {
            control.resumeGeneration(ResumeGenerationRequest.getDefaultInstance());
        }
        assertFalse(control.isPaused(IsPausedRequest.getDefaultInstance()).getPaused());
        assertEquals(1, inference.generate(prompt("after").build()).getOutputsCount());
    }

    @Test
    void abortStopsARunningRequest() throws Exception {
        List<GenerateResponse> responses = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch done = new CountDownLatch(1);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        // Abort from another thread as soon as the request is in flight; the abort loop tolerates the window
        // before the stream registers the id, so the cancellation lands while the generation still has tokens left.
        Thread aborter = new Thread(() -> {
            while (done.getCount() > 0) {
                try {
                    control.abort(AbortRequest.newBuilder().addRequestIds("abort-me").build());
                } catch (StatusRuntimeException e) {
                    // an abort that races the shutdown of the stream is harmless
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }, "grpc-abort");
        aborter.setDaemon(true);
        aborter.start();
        InferenceGrpc.InferenceStub stub = InferenceGrpc.newStub(channel)
                .withDeadlineAfter(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        try {
            stub.generateStream(GenerateRequest.newBuilder()
                            .setRequestId("abort-me")
                            .setText("hello")
                            .setTemperature(0f)
                            .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(1_000_000).setIgnoreEos(true))
                            .setResponse(ResponseOptions.newBuilder().setOutputTokenIds(true))
                            .build(),
                    new StreamObserver<GenerateResponse>() {
                        @Override
                        public void onNext(GenerateResponse response) {
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
            assertTrue(done.await(TIMEOUT_SECONDS, TimeUnit.SECONDS), "the aborted stream ends");
        } finally {
            aborter.interrupt();
        }
        assertNull(failure.get(), () -> "the aborted stream failed: " + failure.get());
        FinishInfo finish = null;
        for (GenerateResponse response : responses) {
            for (SequenceOutput output : response.getOutputsList()) {
                if (output.hasFinishInfo()) {
                    finish = output.getFinishInfo();
                }
            }
        }
        assertNotNull(finish, "the aborted sequence reports its finish info");
        assertEquals(FinishInfo.FinishReason.ABORTED, finish.getFinishReason());
    }

    @Test
    void aDuplicateRequestIdIsRefusedWhileTheFirstRuns() throws Exception {
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch finished = new CountDownLatch(1);
        InferenceGrpc.InferenceStub stub = InferenceGrpc.newStub(channel)
                .withDeadlineAfter(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        stub.generateStream(GenerateRequest.newBuilder()
                        .setRequestId("duplicate")
                        .setText("hello")
                        .setTemperature(0f)
                        .setSampling(com.eonmux.neuralfree.grpc.RandomSampling.newBuilder().setNumSequences(16))
                        .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(1_000_000).setIgnoreEos(true))
                        .build(),
                new StreamObserver<GenerateResponse>() {
                    @Override
                    public void onNext(GenerateResponse response) {
                        started.countDown();
                    }

                    @Override
                    public void onError(Throwable error) {
                        finished.countDown();
                    }

                    @Override
                    public void onCompleted() {
                        finished.countDown();
                    }
                });
        assertTrue(started.await(TIMEOUT_SECONDS, TimeUnit.SECONDS), "the first request starts");
        Thread aborter = new Thread(() -> {
            while (finished.getCount() > 0) {
                try {
                    control.abort(AbortRequest.newBuilder().addRequestIds("duplicate").build());
                } catch (StatusRuntimeException e) {
                    // an abort that races the shutdown of the stream is harmless
                }
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }, "grpc-abort");
        aborter.setDaemon(true);
        try {
            StatusRuntimeException refused = assertThrows(StatusRuntimeException.class,
                    () -> inference.generate(prompt("hello").setRequestId("duplicate").build()));
            assertEquals(Status.Code.INVALID_ARGUMENT, Status.fromThrowable(refused).getCode());
            assertTrue(description(refused).contains("already in flight"), description(refused));
        } finally {
            aborter.start();
            assertTrue(finished.await(TIMEOUT_SECONDS, TimeUnit.SECONDS), "the first request ends");
        }
    }

    @Test
    void abortingAnUnknownIdIsHarmless() {
        control.abort(AbortRequest.newBuilder().addRequestIds("no-such-request").build());
    }

    private static GenerateRequest.Builder prompt(String text) {
        return GenerateRequest.newBuilder()
                .setText(text)
                .setTemperature(0f)
                .setStopping(StoppingCriteria.newBuilder().setMaxNewTokens(4).setIgnoreEos(true));
    }

    private static String description(StatusRuntimeException error) {
        String description = Status.fromThrowable(error).getDescription();
        return description == null ? "" : description;
    }

    /** Writes a synthetic llama GGUF and points the server at it before Quarkus starts. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {

        private final Map<String, String> backups = new LinkedHashMap<>();

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
                                SleepModeSwitch.PROPERTY, "true")
                        .forEach((key, value) -> {
                            backups.put(key, System.setProperty(key, value));
                            PREVIOUS.put(key, backups.get(key));
                        });
                return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public void stop() {
            backups.forEach((key, value) -> {
                if (value == null) {
                    System.clearProperty(key);
                } else {
                    System.setProperty(key, value);
                }
            });
        }
    }

    /** A profile of its own, so the gRPC-on application serves this class alone. */
    public static class Profile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
