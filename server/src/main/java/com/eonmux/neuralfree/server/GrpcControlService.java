package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventsConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisherConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.grpc.AbortRequest;
import com.eonmux.neuralfree.grpc.AbortResponse;
import com.eonmux.neuralfree.grpc.ControlGrpc;
import com.eonmux.neuralfree.grpc.GetKvEventSourcesRequest;
import com.eonmux.neuralfree.grpc.GetKvEventSourcesResponse;
import com.eonmux.neuralfree.grpc.GetModelInfoRequest;
import com.eonmux.neuralfree.grpc.GetServerInfoRequest;
import com.eonmux.neuralfree.grpc.IsPausedRequest;
import com.eonmux.neuralfree.grpc.IsPausedResponse;
import com.eonmux.neuralfree.grpc.IsSleepingRequest;
import com.eonmux.neuralfree.grpc.IsSleepingResponse;
import com.eonmux.neuralfree.grpc.KvEventSource;
import com.eonmux.neuralfree.grpc.ListLorasRequest;
import com.eonmux.neuralfree.grpc.ListLorasResponse;
import com.eonmux.neuralfree.grpc.LoadLoraRequest;
import com.eonmux.neuralfree.grpc.LoadLoraResponse;
import com.eonmux.neuralfree.grpc.LoraAdapter;
import com.eonmux.neuralfree.grpc.ModelInfo;
import com.eonmux.neuralfree.grpc.PauseGenerationRequest;
import com.eonmux.neuralfree.grpc.PauseGenerationResponse;
import com.eonmux.neuralfree.grpc.PauseMode;
import com.eonmux.neuralfree.grpc.ParallelismInfo;
import com.eonmux.neuralfree.grpc.ResumeGenerationRequest;
import com.eonmux.neuralfree.grpc.ResumeGenerationResponse;
import com.eonmux.neuralfree.grpc.ServerInfo;
import com.eonmux.neuralfree.grpc.SleepRequest;
import com.eonmux.neuralfree.grpc.SleepResponse;
import com.eonmux.neuralfree.grpc.UnloadLoraRequest;
import com.eonmux.neuralfree.grpc.UnloadLoraResponse;
import com.eonmux.neuralfree.grpc.WakeUpRequest;
import com.eonmux.neuralfree.grpc.WakeUpResponse;
import io.grpc.stub.StreamObserver;
import io.quarkus.grpc.GrpcService;
import io.smallrye.common.annotation.Blocking;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * The {@code neuralfree.Control} gRPC service: the serving control plane of vLLM's gRPC adapter over the same
 * {@link EngineService} the HTTP routes drive. It answers server and model discovery, cancels requests by id, loads
 * and unloads per-request LoRA adapters, reports the KV event transports, and runs the sleep, wake and pause
 * lifecycle. Every refusal becomes the gRPC status of the HTTP answer it parallels ({@link GrpcFailures}), so each
 * call applies exactly the switch its HTTP route applies through the same object: {@link DevMode} gates the server
 * info, abort, pause, resume and sleep routes, {@link SleepModeSwitch} gates the sleep and wake operations, and the
 * engine's runtime LoRA switch gates the adapter load and unload. Discovery, generation and adapter listing are not
 * development routes and stay reachable.
 *
 * <p>In the router mode a request's {@code model} selects the engine, exactly as the {@code ?model=} query of the
 * HTTP routes does; without it the single-model server's engine serves. Thread-safe: the service delegates to the
 * engine's own serialized operations.</p>
 */
@GrpcService
@Singleton
@Blocking
public class GrpcControlService extends ControlGrpc.ControlImplBase {

    /** The gRPC protocol version this schema defines. */
    static final String API_VERSION = "v1";
    /** The protobuf schema version reported by the discovery calls. */
    static final String PROTO_VERSION = "1";
    /** The KV event encoding: the engine's msgpack event batches, as vLLM's ZMQ publisher emits. */
    static final String KV_ENCODING = "msgpack";
    /** The KV event schema version, vLLM's {@code schema_version}. */
    static final int KV_SCHEMA_VERSION = 1;
    /** The server process's identity, one per JVM. */
    static final String INSTANCE_ID = UUID.randomUUID().toString();

    @Inject
    EngineService service;

    /** The multi-model router; inactive outside the router mode. */
    @Inject
    ModelRouter router;

    @Inject
    GrpcInFlight inFlight;

    /** The development-route switch; the server info, abort and lifecycle calls answer 404 while it is off. */
    @Inject
    DevMode devMode;

    /** The sleep-mode switch; the sleep and wake operations need it. */
    @Inject
    SleepModeSwitch sleepMode;

    @Override
    public void getServerInfo(GetServerInfoRequest request, StreamObserver<ServerInfo> observer) {
        try {
            devMode.require();
            EngineService target = target(request.getModel());
            int dataParallel = target.dataParallelSize();
            ServerInfo.Builder info = ServerInfo.newBuilder()
                    .setEngineVersion(ServerVersion.VALUE)
                    .setApiVersion(API_VERSION)
                    .setInstanceId(INSTANCE_ID)
                    .setProtoVersion(PROTO_VERSION)
                    .setMaxModelLen(target.loaded() ? target.maxModelLength() : 0)
                    .setMaxRunningRequests(target.slots())
                    .setParallelism(ParallelismInfo.newBuilder()
                            .setTensorParallelSize(1)
                            .setPipelineParallelSize(1)
                            .setDataParallelSize(dataParallel)
                            .setDataParallelRank(dataParallel > 0 ? target.replicaRank(0) : 0)
                            .setWorldSize(dataParallel));
            observer.onNext(info.build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void getModelInfo(GetModelInfoRequest request, StreamObserver<ModelInfo> observer) {
        try {
            EngineService target = target(request.getModel());
            target.requireLoaded();
            List<String> names = target.served().names();
            ModelInfo.Builder info = ModelInfo.newBuilder()
                    .setModelId(target.modelPath() == null ? "" : target.modelPath())
                    .setSupportsTextInput(true)
                    .setSupportsTokenIdsInput(true)
                    .setSupportsLora(target.loras().runtimeUpdating() || !target.loras().list().isEmpty())
                    .setSupportsMultimodal(target.media().enabled());
            if (!names.isEmpty()) {
                info.setServedModelName(names.get(0));
                for (int i = 1; i < names.size(); i++) {
                    info.addServedModelAliases(names.get(i));
                }
            }
            String reasoning = target.config().reasoningParser();
            if (reasoning != null && !reasoning.isBlank()) {
                info.setReasoningParser(reasoning);
            }
            String tools = target.config().toolParser();
            if (tools != null && !tools.isBlank()) {
                info.setToolCallParser(tools);
            }
            observer.onNext(info.build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void abort(AbortRequest request, StreamObserver<AbortResponse> observer) {
        try {
            devMode.require();
            inFlight.cancel(request.getRequestIdsList());
            observer.onNext(AbortResponse.getDefaultInstance());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void loadLora(LoadLoraRequest request, StreamObserver<LoadLoraResponse> observer) {
        try {
            EngineService target = target(request.getModel());
            target.loras().requireRuntimeUpdating();
            target.requireLoaded();
            String name = required("lora_name", request.getLoraName());
            String path = required("source_path", request.getSourcePath());
            target.loras().load(name, path, request.getLoadInplace());
            LoraAdapter adapter = adapter(target, name);
            if (adapter == null) {
                throw new IllegalStateException("the loaded adapter '" + name + "' did not reach the catalog");
            }
            observer.onNext(LoadLoraResponse.newBuilder().setAdapter(adapter).build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void unloadLora(UnloadLoraRequest request, StreamObserver<UnloadLoraResponse> observer) {
        try {
            EngineService target = target(request.getModel());
            target.loras().requireRuntimeUpdating();
            target.requireLoaded();
            String name = required("lora_name", request.getLoraName());
            LoraAdapter adapter = adapter(target, name);
            if (adapter == null) {
                throw new java.util.NoSuchElementException("The lora adapter '" + name + "' cannot be found.");
            }
            target.loras().unload(name);
            observer.onNext(UnloadLoraResponse.newBuilder().setAdapter(adapter).build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void listLoras(ListLorasRequest request, StreamObserver<ListLorasResponse> observer) {
        try {
            EngineService target = target(request.getModel());
            target.requireLoaded();
            ListLorasResponse.Builder response = ListLorasResponse.newBuilder();
            List<LoraCatalog.Adapter> adapters = target.loras().list();
            for (int i = 0; i < adapters.size(); i++) {
                response.addAdapters(wire(adapters.get(i), i));
            }
            observer.onNext(response.build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void getKvEventSources(GetKvEventSourcesRequest request,
                                  StreamObserver<GetKvEventSourcesResponse> observer) {
        try {
            EngineService target = target(request.getModel());
            GetKvEventSourcesResponse.Builder response = GetKvEventSourcesResponse.newBuilder();
            for (KvEventSource source : kvEventSources(target)) {
                response.addSources(source);
            }
            observer.onNext(response.build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void pauseGeneration(PauseGenerationRequest request, StreamObserver<PauseGenerationResponse> observer) {
        try {
            devMode.require();
            EngineService target = target(request.getModel());
            target.requireLoaded();
            boolean clearCache = !request.hasClearCache() || request.getClearCache();
            target.pause(mode(request.getMode()), clearCache);
            observer.onNext(PauseGenerationResponse.getDefaultInstance());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void resumeGeneration(ResumeGenerationRequest request,
                                 StreamObserver<ResumeGenerationResponse> observer) {
        try {
            devMode.require();
            EngineService target = target(request.getModel());
            target.resume();
            observer.onNext(ResumeGenerationResponse.getDefaultInstance());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void isPaused(IsPausedRequest request, StreamObserver<IsPausedResponse> observer) {
        try {
            devMode.require();
            EngineService target = target(request.getModel());
            observer.onNext(IsPausedResponse.newBuilder().setPaused(target.isPaused()).build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void sleep(SleepRequest request, StreamObserver<SleepResponse> observer) {
        try {
            devMode.require();
            sleepMode.requireSleep();
            EngineService target = target(request.getModel());
            requireModelOrSleep(target);
            int level = request.hasLevel() ? request.getLevel() : 1;
            target.sleep(level, mode(request.getMode()));
            observer.onNext(SleepResponse.newBuilder().setLevel(level).build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void wakeUp(WakeUpRequest request, StreamObserver<WakeUpResponse> observer) {
        try {
            devMode.require();
            sleepMode.requireSleep();
            EngineService target = target(request.getModel());
            requireModelOrSleep(target);
            String status = target.wakeUp(request.getTagsList());
            observer.onNext(WakeUpResponse.newBuilder().setStatus(status).build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    @Override
    public void isSleeping(IsSleepingRequest request, StreamObserver<IsSleepingResponse> observer) {
        try {
            devMode.require();
            EngineService target = target(request.getModel());
            observer.onNext(IsSleepingResponse.newBuilder().setSleeping(target.isSleeping()).build());
            observer.onCompleted();
        } catch (Throwable e) {
            observer.onError(GrpcFailures.of(e));
        }
    }

    // ------------------------------------------------------------------ helpers

    /** The engine a control call names: the router's target for {@code model}, else the one-model server's. */
    private EngineService target(String model) {
        return ModelRouter.targetQuery(router, service, model, null);
    }

    private static SleepControl.Mode mode(PauseMode mode) {
        switch (mode) {
            case PAUSE_MODE_WAIT:
                return SleepControl.Mode.WAIT;
            case PAUSE_MODE_KEEP:
                return SleepControl.Mode.KEEP;
            default:
                return SleepControl.Mode.ABORT;
        }
    }

    private static void requireModelOrSleep(EngineService target) {
        if (!target.loaded() && !target.isSleeping()) {
            target.requireLoaded();
        }
    }

    private static String required(String field, String value) {
        if (value == null || value.isBlank()) {
            throw GrpcFailures.invalid(field + " is required");
        }
        return value;
    }

    /** The wire form of the catalog entry {@code name}, or null when no adapter is named so. */
    private static LoraAdapter adapter(EngineService target, String name) {
        List<LoraCatalog.Adapter> adapters = target.loras().list();
        for (int i = 0; i < adapters.size(); i++) {
            if (adapters.get(i).name().equals(name)) {
                return wire(adapters.get(i), i);
            }
        }
        return null;
    }

    private static LoraAdapter wire(LoraCatalog.Adapter adapter, int id) {
        return LoraAdapter.newBuilder()
                .setLoraId(id)
                .setLoraName(adapter.name())
                .setSourcePath(adapter.path())
                .setScale(adapter.scale())
                .build();
    }

    /**
     * The KV event transports the engine publishes on: one source per local replica when the configured publisher
     * is the ZMQ transport (each rank's ports offset by its global data-parallel rank), else none.
     */
    private static List<KvEventSource> kvEventSources(EngineService target) {
        String spec = target.config().memory().kvEvents();
        if (spec == null || spec.isBlank()) {
            return List.of();
        }
        KvEventsConfig config;
        if (spec.startsWith("{")) {
            config = KvEventsConfig.parse(spec, MemoryConfig.PROP_KV_EVENTS);
        } else if (spec.equalsIgnoreCase(KvEventsConfig.ZMQ)) {
            config = KvEventsConfig.parse("{\"enable_kv_cache_events\":true}", MemoryConfig.PROP_KV_EVENTS);
        } else {
            return List.of(); // log, none or a custom publisher: no ZMQ endpoint to discover
        }
        ZmqEventPublisherConfig zmq = config.zmq();
        if (zmq == null) {
            return List.of();
        }
        List<KvEventSource> sources = new ArrayList<>();
        for (int i = 0; i < target.replicaCount(); i++) {
            int rank = target.replicaRank(i);
            KvEventSource.Builder source = KvEventSource.newBuilder()
                    .setTransport(KvEventsConfig.ZMQ)
                    .setEndpoint(ZmqEventPublisherConfig.offsetEndpointPort(zmq.endpoint(), rank))
                    .setTopic(zmq.topic())
                    .setDataParallelRank(rank)
                    .setEncoding(KV_ENCODING)
                    .setSchemaVersion(KV_SCHEMA_VERSION)
                    .setBufferSteps(zmq.bufferSteps())
                    .setHwm(zmq.hwm())
                    .setMaxQueueSize(zmq.maxQueueSize());
            if (zmq.replayEndpoint() != null) {
                source.setReplayEndpoint(ZmqEventPublisherConfig.offsetEndpointPort(zmq.replayEndpoint(), rank));
            }
            sources.add(source.build());
        }
        return sources;
    }
}
