package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.server.tools.McpServerConfig;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The server's own settings, beside the engine's {@code EngineConfig}: the served model names ({@link ServedModels}),
 * the request budget, the longest one request may wait on the engine before it is cancelled and refused with 503
 * (llama.cpp's {@code --timeout}, there a socket timeout), the suffix-prefix-middle infill order (llama.cpp's
 * {@code --spm-infill}), the Responses API store size (vLLM's {@code VLLM_ENABLE_RESPONSES_API_STORE}, here
 * bounded), the slot save directory (llama.cpp's {@code --slot-save-path}), which turns the slot save and restore
 * endpoints on, the idle sleep (llama.cpp's {@code --sleep-idle-seconds}: after so many seconds without work each
 * engine replica releases the model and sleeps, and the next request wakes it), and the data-parallel layout: how
 * many engine replicas serve the one model and, optionally, which device each opens on (vLLM's
 * {@code --data-parallel-size} with one rank per device), where they sit among the deployment's global ranks and
 * how a request picks one ({@link DpLayout}), whether a request may supply its own {@code chat_template}
 * (vLLM's {@code --trust-request-chat-template}, off by default as there), the default chat template
 * variables every chat request renders with unless it sets them itself (vLLM's
 * {@code --default-chat-template-kwargs}, llama.cpp's {@code --chat-template-kwargs}), the distribution the
 * OpenAI endpoints' logprobs score (vLLM's {@code --logprobs-mode}, {@code raw_logprobs} by default as there), and
 * the OTLP endpoint request traces are exported to (vLLM's {@code --otlp-traces-endpoint}); null keeps per-request
 * OpenTelemetry tracing off; and the most generation requests one replica holds waiting to run (SGLang's
 * {@code --max-queued-requests}).
 * Immutable.
 *
 * @param servedNames      the comma-separated served names, or null to serve the model path
 * @param requestTimeout   the request budget, positive
 * @param spmInfill        whether {@code /infill} puts the suffix before the prefix
 * @param responseStore    how many Responses API responses the server keeps, 0 for none
 * @param slotSavePath     the directory slot files are written to, or null when slot persistence is off
 * @param sleepIdleSeconds the idle seconds before each replica sleeps, negative for never
 * @param replicas         how many engine replicas serve the model, at least 1
 * @param devices          the per-replica device-name filters ({@code neuralfree.device} substrings), or null or
 *                         empty for one replica on the default device; when set, one entry per replica
 * @param trustRequestChatTemplate whether a request's {@code chat_template} renders in place of the model's
 *                         template; when false such a request is refused with 400
 * @param dataParallel     the global rank layout and the load-balance method; null holds every rank here
 *                         ({@link DpLayout#internal}); the local replicas must fit inside the global size from the
 *                         start rank
 * @param chatTemplateKwargs the default chat template variables ({@link ChatTemplateKwargs}); null for none
 * @param logprobsMode     the {@link LogprobsMode} the OpenAI endpoints' logprobs and prompt logprobs score; null
 *                         is {@link LogprobsMode#RAW_LOGPROBS}
 * @param tools            the enabled built-in server tools ({@code all} or the names); null or empty keeps
 *                         {@code /tools} off
 * @param mcpServers       the MCP tool servers of the launch config; null or empty for none
 * @param otlpTracesEndpoint the OTLP endpoint ({@code host:port} for grpc, an {@code http(s)://} URL for
 *                         http/protobuf) every request's {@code llm_request} span is exported to; null keeps
 *                         request tracing off
 * @param watermark        the engine's text watermark ({@link WatermarkConfig}, vLLM's {@code --watermark-config});
 *                         null serves unwatermarked generation, and a request may still switch it off with its
 *                         {@code watermarking} field
 * @param kvTransfer       the disaggregated prefill/decode connector (vLLM's {@code --kv-transfer-config}), or
 *                         null for an instance that transfers no KV
 * @param maxQueuedRequests the most generation requests one replica holds waiting to run (SGLang's
 *                         {@code --max-queued-requests}): a request past it is refused with 429 and
 *                         {@code Retry-After}; 0 queues without limit
 */
record ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                     Path slotSavePath, int sleepIdleSeconds, int replicas, List<String> devices,
                     boolean trustRequestChatTemplate, DpLayout dataParallel,
                     Map<String, Object> chatTemplateKwargs, LogprobsMode logprobsMode,
                     List<String> tools, List<McpServerConfig> mcpServers, String otlpTracesEndpoint,
                     WatermarkConfig watermark, KvTransferConfig kvTransfer, int maxQueuedRequests) {

    /** The request budget in seconds. */
    static final String TIMEOUT_PROPERTY = "neuralfree.server.request.timeout";

    /** The default request budget. */
    static final Duration DEFAULT_TIMEOUT = Duration.ofMinutes(30);

    /** The longest request budget: a year, far inside the nanosecond clock's range. */
    static final Duration MAX_TIMEOUT = Duration.ofDays(365);

    /** Whether {@code /infill} uses the suffix-prefix-middle order. */
    static final String SPM_INFILL_PROPERTY = "neuralfree.server.spm.infill";

    /** How many Responses API responses the server keeps for retrieval and {@code previous_response_id}. */
    static final String RESPONSE_STORE_PROPERTY = "neuralfree.server.responses.store";

    /** The directory the slot save and restore endpoints write session files into (llama.cpp's flag of the same
     *  name); unset or blank keeps them off. */
    static final String SLOT_SAVE_PATH_PROPERTY = "neuralfree.server.slot.save.path";

    /** The idle seconds before the engine sleeps; negative keeps it resident forever. */
    static final String SLEEP_IDLE_PROPERTY = "neuralfree.server.sleep.idle.seconds";

    /** The default idle sleep: never. */
    static final int DEFAULT_SLEEP_IDLE_SECONDS = -1;

    /** The most generation requests one replica holds waiting to run; 0 (the default) for no limit. */
    static final String MAX_QUEUED_PROPERTY = "neuralfree.server.max.queued.requests";

    /** How many engine replicas serve the model (vLLM's {@code --data-parallel-size}). */
    static final String REPLICAS_PROPERTY = "neuralfree.server.replicas";

    /** The comma-separated per-replica device-name filters, one engine replica per entry. */
    static final String DEVICES_PROPERTY = "neuralfree.server.devices";

    /** Whether requests may supply their own {@code chat_template} (vLLM's {@code --trust-request-chat-template}). */
    static final String TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY = "neuralfree.server.trust.request.chat.template";

    /** The launch flag that sets {@link #TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY}. */
    static final String TRUST_REQUEST_CHAT_TEMPLATE_FLAG = "--trust-request-chat-template";

    /** The distribution the OpenAI endpoints' logprobs score (vLLM's {@code --logprobs-mode}). */
    static final String LOGPROBS_MODE_PROPERTY = "neuralfree.server.logprobs.mode";

    /** The launch flag that sets {@link #LOGPROBS_MODE_PROPERTY}. */
    static final String LOGPROBS_MODE_FLAG = "--logprobs-mode";

    /** The comma-separated built-in server tools ({@code all} for every one), or none when unset (llama.cpp's
     *  {@code --tools}). */
    static final String TOOLS_PROPERTY = "neuralfree.server.tools";

    /** The launch flag that sets {@link #TOOLS_PROPERTY}. */
    static final String TOOLS_FLAG = "--tools";

    /** The file with the {@code mcpServers} JSON the MCP tool servers are configured from (llama.cpp's
     *  {@code --mcp-servers-config}). */
    static final String MCP_CONFIG_PROPERTY = "neuralfree.server.mcp.config";

    /** The launch flag that sets {@link #MCP_CONFIG_PROPERTY}. */
    static final String MCP_CONFIG_FLAG = "--mcp-servers-config";

    /** The inline {@code mcpServers} JSON beside {@link #MCP_CONFIG_PROPERTY} (llama.cpp's
     *  {@code --mcp-servers-json}). */
    static final String MCP_JSON_PROPERTY = "neuralfree.server.mcp.json";

    /** The launch flag that sets {@link #MCP_JSON_PROPERTY}. */
    static final String MCP_JSON_FLAG = "--mcp-servers-json";

    /** The OTLP endpoint request traces are exported to (vLLM's {@code --otlp-traces-endpoint}). */
    static final String OTLP_TRACES_ENDPOINT_PROPERTY = "neuralfree.server.otlp.traces.endpoint";

    /** The launch flag that sets {@link #OTLP_TRACES_ENDPOINT_PROPERTY}. */
    static final String OTLP_TRACES_ENDPOINT_FLAG = "--otlp-traces-endpoint";

    /** The launch flag that sets the engine watermark ({@link WatermarkConfig#PROP_CONFIG}, vLLM's flag of the
     *  same name): a JSON object with the watermark's fields; its key may instead come from the
     *  {@link WatermarkConfig#ENV_KEY} or {@link WatermarkConfig#ENV_KEY_FILE} environment variables. */
    static final String WATERMARK_CONFIG_FLAG = "--watermark-config";

    /** The disaggregated prefill/decode connector (vLLM's {@code --kv-transfer-config}): a JSON object with the
     *  {@link KvTransferConfig} fields, unset or blank for no KV transfer. The JSON may carry the connector's
     *  shared token; the {@link KvTransferConfig#ENV_TOKEN} or {@link KvTransferConfig#ENV_TOKEN_FILE}
     *  environment variables supply it instead, and refuse beside a JSON token. */
    static final String KV_TRANSFER_CONFIG_PROPERTY = KvTransferConfig.SERVER_PROPERTY;

    /** The launch flag that sets {@link #KV_TRANSFER_CONFIG_PROPERTY}. */
    static final String KV_TRANSFER_CONFIG_FLAG = KvTransferConfig.FLAG;

    ServerOptions {
        Objects.requireNonNull(requestTimeout, "requestTimeout");
        if (requestTimeout.isNegative() || requestTimeout.isZero()) {
            throw new IllegalArgumentException(TIMEOUT_PROPERTY + " must be positive, got " + requestTimeout);
        }
        if (requestTimeout.compareTo(MAX_TIMEOUT) > 0) {
            throw new IllegalArgumentException(TIMEOUT_PROPERTY + " must be at most " + MAX_TIMEOUT.toSeconds()
                    + " seconds, got " + requestTimeout.toSeconds());
        }
        if (responseStore < 0) {
            throw new IllegalArgumentException(RESPONSE_STORE_PROPERTY + " must be 0 or more, got " + responseStore);
        }
        if (slotSavePath != null && !Files.isDirectory(slotSavePath)) {
            throw new IllegalArgumentException(SLOT_SAVE_PATH_PROPERTY + " must name a directory, got " + slotSavePath);
        }
        if (maxQueuedRequests < 0) {
            throw new IllegalArgumentException(MAX_QUEUED_PROPERTY + " must be 0 or more, got " + maxQueuedRequests);
        }
        if (replicas < 1) {
            throw new IllegalArgumentException(REPLICAS_PROPERTY + " must be 1 or more, got " + replicas);
        }
        if (devices != null) {
            for (int i = 0; i < devices.size(); i++) {
                if (devices.get(i) == null || devices.get(i).isBlank()) {
                    throw new IllegalArgumentException(DEVICES_PROPERTY + " entry " + i + " is empty");
                }
            }
            if (devices.size() != replicas) {
                throw new IllegalArgumentException(DEVICES_PROPERTY + " holds " + devices.size()
                        + " entries but " + REPLICAS_PROPERTY + " is " + replicas + "; they must match");
            }
            devices = List.copyOf(devices);
        }
        if (dataParallel == null) {
            dataParallel = DpLayout.internal(replicas);
        }
        if (logprobsMode == null) {
            logprobsMode = LogprobsMode.RAW_LOGPROBS;
        }
        if (dataParallel.startRank() + (long) replicas > dataParallel.size()) {
            throw new IllegalArgumentException("the " + replicas + " local replica(s) from " + DpLayout.RANK_PROPERTY
                    + " " + dataParallel.startRank() + " pass " + DpLayout.SIZE_PROPERTY + " "
                    + dataParallel.size() + "; the ranks must lie inside [0, " + DpLayout.SIZE_PROPERTY + ")");
        }
        chatTemplateKwargs = chatTemplateKwargs == null ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(chatTemplateKwargs));
        if (tools != null) {
            for (int i = 0; i < tools.size(); i++) {
                if (tools.get(i) == null || tools.get(i).isBlank()) {
                    throw new IllegalArgumentException(TOOLS_PROPERTY + " entry " + i + " is empty");
                }
            }
            tools = List.copyOf(tools);
        }
        mcpServers = mcpServers == null ? null : List.copyOf(mcpServers);
        otlpTracesEndpoint = otlpTracesEndpoint == null || otlpTracesEndpoint.isBlank()
                ? null : otlpTracesEndpoint.trim();
    }

    /** The seventeen settings with no limit on the requests waiting per replica. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  Path slotSavePath, int sleepIdleSeconds, int replicas, List<String> devices,
                  boolean trustRequestChatTemplate, DpLayout dataParallel, Map<String, Object> chatTemplateKwargs,
                  LogprobsMode logprobsMode, List<String> tools, List<McpServerConfig> mcpServers,
                  String otlpTracesEndpoint, WatermarkConfig watermark, KvTransferConfig kvTransfer) {
        this(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath, sleepIdleSeconds, replicas,
                devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs, logprobsMode, tools, mcpServers,
                otlpTracesEndpoint, watermark, kvTransfer, 0);
    }

    /** The twelve settings with no server tools enabled (the {@code /tools} endpoint off) and tracing off. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  Path slotSavePath, int sleepIdleSeconds, int replicas, List<String> devices,
                  boolean trustRequestChatTemplate, DpLayout dataParallel, Map<String, Object> chatTemplateKwargs,
                  LogprobsMode logprobsMode) {
        this(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath, sleepIdleSeconds, replicas,
                devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs, logprobsMode, null, null, null,
                null, null);
    }

    /** The eleven settings with the default logprobs mode ({@link LogprobsMode#RAW_LOGPROBS}). */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  Path slotSavePath, int sleepIdleSeconds, int replicas, List<String> devices,
                  boolean trustRequestChatTemplate, DpLayout dataParallel, Map<String, Object> chatTemplateKwargs) {
        this(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath, sleepIdleSeconds, replicas,
                devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs, null);
    }

    /** The nine settings with every data-parallel rank served here, routed by the default method, no default chat
     *  template variables and the default logprobs mode. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  Path slotSavePath, int sleepIdleSeconds, int replicas, List<String> devices,
                  boolean trustRequestChatTemplate) {
        this(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath, sleepIdleSeconds, replicas,
                devices, trustRequestChatTemplate, null, null);
    }

    /** The eight settings with request-supplied chat templates refused. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  Path slotSavePath, int sleepIdleSeconds, int replicas, List<String> devices) {
        this(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath, sleepIdleSeconds, replicas,
                devices, false);
    }

    /** {@code servedNames} and {@code requestTimeout}: one default replica, slots off and no idle sleep. */
    ServerOptions(String servedNames, Duration requestTimeout) {
        this(servedNames, requestTimeout, false, 0, null, DEFAULT_SLEEP_IDLE_SECONDS, 1, null);
    }

    /** {@code servedNames}, {@code requestTimeout}, {@code spmInfill} and {@code responseStore}: one default
     *  replica, slots off and no idle sleep. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore) {
        this(servedNames, requestTimeout, spmInfill, responseStore, null, DEFAULT_SLEEP_IDLE_SECONDS, 1, null);
    }

    /** The four settings with slots on at {@code slotSavePath}: one default replica and no idle sleep. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  Path slotSavePath) {
        this(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath, DEFAULT_SLEEP_IDLE_SECONDS, 1, null);
    }

    /** The four settings with the idle sleep at {@code sleepIdleSeconds}: one default replica and slots off. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  int sleepIdleSeconds) {
        this(servedNames, requestTimeout, spmInfill, responseStore, null, sleepIdleSeconds, 1, null);
    }

    /** The four settings with the data-parallel layout at {@code replicas} and {@code devices}: slots off and no
     *  idle sleep. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  int replicas, List<String> devices) {
        this(servedNames, requestTimeout, spmInfill, responseStore, null, DEFAULT_SLEEP_IDLE_SECONDS, replicas,
                devices);
    }

    /** {@link #ServerOptions(String, Duration, boolean, int, int, List)} with the global rank layout and the
     *  request template trust switch. */
    ServerOptions(String servedNames, Duration requestTimeout, boolean spmInfill, int responseStore,
                  int replicas, List<String> devices, boolean trustRequestChatTemplate, DpLayout dataParallel) {
        this(servedNames, requestTimeout, spmInfill, responseStore, null, DEFAULT_SLEEP_IDLE_SECONDS, replicas,
                devices, trustRequestChatTemplate, dataParallel, null);
    }

    /** The served model path and the default budget. */
    static ServerOptions defaults() {
        return new ServerOptions(null, DEFAULT_TIMEOUT);
    }

    /** A copy keeping at most {@code responses} Responses API responses. */
    ServerOptions withResponseStore(int responses) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responses, slotSavePath, sleepIdleSeconds,
                replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs, logprobsMode,
                tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy that accepts ({@code trust} true) or refuses request-supplied chat templates. */
    ServerOptions withTrustRequestChatTemplate(boolean trust) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trust, dataParallel, chatTemplateKwargs, logprobsMode,
                tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy whose chat requests render with {@code kwargs} as their default template variables. */
    ServerOptions withChatTemplateKwargs(Map<String, Object> kwargs) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, kwargs, logprobsMode,
                tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy placing the local replicas at {@code layout}'s ranks and routing by its method. */
    ServerOptions withDataParallel(DpLayout layout) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, layout, chatTemplateKwargs,
                logprobsMode, tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy whose OpenAI endpoints score logprobs in {@code mode}. */
    ServerOptions withLogprobsMode(LogprobsMode mode) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs,
                mode, tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy serving the model under {@code names}, the comma-separated list {@link ServedModels} parses. */
    ServerOptions withServedNames(String names) {
        return new ServerOptions(names, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs,
                logprobsMode, tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy whose per-request spans are exported to {@code endpoint}; null turns request tracing off. */
    ServerOptions withOtlpTracesEndpoint(String endpoint) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs,
                logprobsMode, tools, mcpServers, endpoint, watermark, kvTransfer, maxQueuedRequests);
    }

    /** A copy serving every request under {@code config}'s watermark; null serves unwatermarked generation. */
    ServerOptions withWatermark(WatermarkConfig config) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs,
                logprobsMode, tools, mcpServers, otlpTracesEndpoint, config, kvTransfer, maxQueuedRequests);
    }

    /** A copy that serves disaggregated prefill/decode through {@code config}'s connector; null transfers no KV. */
    ServerOptions withKvTransfer(KvTransferConfig config) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs,
                logprobsMode, tools, mcpServers, otlpTracesEndpoint, watermark, config, maxQueuedRequests);
    }

    /** A copy holding at most {@code requests} generation requests waiting per replica; 0 for no limit. */
    ServerOptions withMaxQueuedRequests(int requests) {
        return new ServerOptions(servedNames, requestTimeout, spmInfill, responseStore, slotSavePath,
                sleepIdleSeconds, replicas, devices, trustRequestChatTemplate, dataParallel, chatTemplateKwargs,
                logprobsMode, tools, mcpServers, otlpTracesEndpoint, watermark, kvTransfer, requests);
    }

    /** The global data-parallel rank local replica {@code index} serves. */
    int globalRank(int index) {
        return dataParallel.startRank() + index;
    }

    /**
     * The options of the {@code neuralfree.server.*} system properties. The device list implies the replica count
     * (one replica per entry) when {@code neuralfree.server.replicas} is unset; an explicit count must match it.
     * The global data-parallel size defaults to the replica count and the start rank to 0.
     *
     * @throws IllegalArgumentException when the timeout, the store size, the idle seconds, the replica count, the
     *                                  data-parallel size or rank is not a whole number, the slot save path does
     *                                  not name a directory, the replica count is not 1 or more, a device entry is
     *                                  empty, the ranks pass the data-parallel size, the load-balance method or
     *                                  the logprobs mode is unknown, {@link ChatTemplateKwargs#PROPERTY} is not
     *                                  a JSON object, a tools entry is blank, the MCP config file does not read
     *                                  or either MCP source does not parse, or {@link WatermarkConfig#PROP_CONFIG}
     *                                  is not a valid watermark configuration
     */
    static ServerOptions fromSystemProperties() {
        return fromSystemProperties(System.getenv());
    }

    /**
     * {@link #fromSystemProperties()} over an explicit environment, so the secret sources are testable: the
     * watermark reads its key and the KV transfer its token from {@code env} when their properties do not carry
     * them.
     */
    static ServerOptions fromSystemProperties(Map<String, String> env) {
        long timeout = wholeNumber(TIMEOUT_PROPERTY, DEFAULT_TIMEOUT.toSeconds());
        long store = wholeNumber(RESPONSE_STORE_PROPERTY, 0);
        if (store > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(RESPONSE_STORE_PROPERTY + " is too large: " + store);
        }
        long idle = wholeNumber(SLEEP_IDLE_PROPERTY, DEFAULT_SLEEP_IDLE_SECONDS);
        if (idle > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(SLEEP_IDLE_PROPERTY + " is too large: " + idle);
        }
        long maxQueued = wholeNumber(MAX_QUEUED_PROPERTY, 0);
        if (maxQueued > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(MAX_QUEUED_PROPERTY + " is too large: " + maxQueued);
        }
        List<String> devices = devices(System.getProperty(DEVICES_PROPERTY));
        long fallbackReplicas = devices == null ? 1 : devices.size();
        long replicas = wholeNumber(REPLICAS_PROPERTY, fallbackReplicas);
        if (replicas > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(REPLICAS_PROPERTY + " is too large: " + replicas);
        }
        return new ServerOptions(System.getProperty(ServedModels.PROPERTY), Duration.ofSeconds(timeout),
                Boolean.parseBoolean(System.getProperty(SPM_INFILL_PROPERTY, "false").trim()), (int) store,
                slotSaveProperty(), (int) idle, (int) replicas, devices,
                Boolean.parseBoolean(System.getProperty(TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY, "false").trim()),
                dataParallelProperties((int) replicas), ChatTemplateKwargs.fromProperties(System.getProperties()),
                logprobsModeProperty(), toolsProperty(), mcpServersProperty(), otlpTracesProperty(),
                WatermarkConfig.fromProperties(System.getProperties(), env), kvTransferProperty(env),
                (int) maxQueued);
    }

    /**
     * The {@link KvTransferConfig} of {@link #KV_TRANSFER_CONFIG_PROPERTY} (vLLM's {@code --kv-transfer-config}
     * JSON), or null when it is unset or blank. The token may come from the environment.
     *
     * @throws IllegalArgumentException      when the JSON is malformed, a field is unknown or out of range, or a
     *                                       token source is malformed or given twice
     * @throws UnsupportedOperationException for a vLLM field this build does not run
     */
    static KvTransferConfig kvTransferProperty() {
        return kvTransferProperty(System.getenv());
    }

    /**
     * {@link #kvTransferProperty()} over an explicit environment, so the token sources are testable.
     */
    static KvTransferConfig kvTransferProperty(Map<String, String> env) {
        String raw = System.getProperty(KV_TRANSFER_CONFIG_PROPERTY);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        return KvTransferConfig.parse(raw, KV_TRANSFER_CONFIG_FLAG, env);
    }

    /** The comma-separated {@link #TOOLS_PROPERTY} entries, or null when it is unset or blank. */
    private static List<String> toolsProperty() {
        String raw = System.getProperty(TOOLS_PROPERTY);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        List<String> out = new ArrayList<>();
        for (String entry : raw.split(",")) {
            out.add(entry.trim());
        }
        return out;
    }

    /** The trimmed {@link #OTLP_TRACES_ENDPOINT_PROPERTY}, or null when it is unset or blank. */
    private static String otlpTracesProperty() {
        String raw = System.getProperty(OTLP_TRACES_ENDPOINT_PROPERTY);
        return raw == null || raw.isBlank() ? null : raw.trim();
    }

    /**
     * The MCP servers of {@link #MCP_CONFIG_PROPERTY} (a file) and {@link #MCP_JSON_PROPERTY} (inline JSON),
     * the file first; a name named twice keeps its first entry. Null when neither source is set.
     */
    private static List<McpServerConfig> mcpServersProperty() {
        String file = System.getProperty(MCP_CONFIG_PROPERTY);
        String inline = System.getProperty(MCP_JSON_PROPERTY);
        if ((file == null || file.isBlank()) && (inline == null || inline.isBlank())) {
            return null;
        }
        List<McpServerConfig> out = new ArrayList<>();
        if (file != null && !file.isBlank()) {
            out.addAll(parseMcp(readFile(file), "MCP config file '" + file + "'"));
        }
        if (inline != null && !inline.isBlank()) {
            out.addAll(parseMcp(inline, MCP_JSON_PROPERTY));
        }
        List<McpServerConfig> deduped = new ArrayList<>();
        for (McpServerConfig config : out) {
            if (deduped.stream().noneMatch(kept -> kept.name().equals(config.name()))) {
                deduped.add(config);
            }
        }
        return deduped;
    }

    /** The text of the {@code path} (the MCP config file), failed naming the property. */
    private static String readFile(String path) {
        try {
            return Files.readString(Path.of(path));
        } catch (IOException | UncheckedIOException e) {
            throw new IllegalArgumentException(MCP_CONFIG_PROPERTY + " does not read: " + path, e);
        }
    }

    /** The servers of an {@code mcpServers} JSON document, failed naming the {@code source}. */
    private static List<McpServerConfig> parseMcp(String json, String source) {
        try {
            return McpServerConfig.parse(json, source);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(source + ": " + e.getMessage(), e);
        }
    }

    /** The mode {@link #LOGPROBS_MODE_PROPERTY} names, or {@link LogprobsMode#RAW_LOGPROBS} when unset or blank. */
    private static LogprobsMode logprobsModeProperty() {
        String raw = System.getProperty(LOGPROBS_MODE_PROPERTY);
        if (raw == null || raw.isBlank()) {
            return LogprobsMode.RAW_LOGPROBS;
        }
        try {
            return LogprobsMode.parse(raw);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(LOGPROBS_MODE_PROPERTY + ": " + e.getMessage(), e);
        }
    }

    /**
     * The {@link DpLayout} of its properties: the size defaults to {@code replicas} (at least 1, so a replica count
     * below 1 is refused under its own property), the rank to 0.
     */
    private static DpLayout dataParallelProperties(int replicas) {
        long size = wholeNumber(DpLayout.SIZE_PROPERTY, Math.max(replicas, 1));
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(DpLayout.SIZE_PROPERTY + " is too large: " + size);
        }
        long rank = wholeNumber(DpLayout.RANK_PROPERTY, 0);
        if (rank > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(DpLayout.RANK_PROPERTY + " is too large: " + rank);
        }
        String balance = System.getProperty(DpLayout.BALANCE_PROPERTY);
        LoadBalance method = balance == null || balance.isBlank() ? LoadBalance.SHORTEST_QUEUE
                : LoadBalance.parse(balance, DpLayout.BALANCE_PROPERTY);
        return new DpLayout((int) size, (int) rank, method);
    }

    /** The directory {@link #SLOT_SAVE_PATH_PROPERTY} names, or null when it is unset or blank. */
    private static Path slotSaveProperty() {
        String raw = System.getProperty(SLOT_SAVE_PATH_PROPERTY);
        if (raw == null || raw.isBlank()) {
            return null;
        }
        return Path.of(raw.trim());
    }

    /** The comma-separated {@code raw} as one trimmed entry per replica, or null when unset or empty. */
    private static List<String> devices(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        List<String> out = new ArrayList<>();
        for (String entry : raw.split(",")) {
            out.add(entry.trim());
        }
        return out;
    }

    private static long wholeNumber(String property, long fallback) {
        String raw = System.getProperty(property);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return Long.parseLong(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(property + " must be a whole number, got '" + raw + "'", e);
        }
    }

    /** The budget as the refusal message names it. */
    String budgetText() {
        return requestTimeout.toSeconds() + "-second";
    }
}
