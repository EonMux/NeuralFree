package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateDetector;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.chat.TemplateCaps;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LogprobsMode;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.batch.PrefixHashAlgo;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventsConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConfig;
import com.eonmux.neuralfree.gguf.mm.MmEncoderTpMode;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;
import com.eonmux.neuralfree.gguf.video.VideoPruner;
import com.eonmux.neuralfree.server.tools.McpServerConfig;
import com.eonmux.neuralfree.server.tools.ToolsRegistry;
import com.eonmux.neuralfree.utils.json.Json;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;

/**
 * The parsed arguments of the server launch command: llama-server's and {@code vllm serve}'s flags, each mapped to
 * the {@code neuralfree.*} or {@code quarkus.*} system property the server reads, so {@code --port 8081} and
 * {@code -Dquarkus.http.port=8081} configure the same setting and a flag overrides a pre-set property. llama.cpp's
 * device list ({@code -dev}, {@code --device}) selects the GPUs the model loads over
 * ({@link EngineConfig#PROP_GPU_DEVICES}, {@link EngineConfig#PROP_DEVICE}, and a layer split for two or more
 * selectors), and {@code --rpc} adds remote compute servers as layer targets
 * ({@link EngineConfig#PROP_RPC}), as it does on the CLI. The multi-GPU launch flags ({@link ParallelLaunchFlags})
 * add llama-server's split flags and vLLM's parallel flags beside them: {@code -sm}/{@code --split-mode},
 * {@code -ts}/{@code --tensor-split}, {@code -mg}/{@code --main-gpu}, {@code -tp}/{@code --tensor-parallel-size},
 * {@code -pp}/{@code --pipeline-parallel-size}, {@code --enable-expert-parallel} and the EPLB set, each through the
 * CLI's own parsers. Other vLLM spellings and flags are accepted beside llama.cpp's ({@code --served-model-name},
 * {@code --max-model-len},
 * {@code --ssl-keyfile} / {@code --ssl-certfile}, {@code --trust-request-chat-template},
 * {@code --chat-template-content-format}, {@code --default-chat-template-kwargs}, the parser selection
 * {@code --reasoning-parser} and {@code --tool-call-parser} (with vLLM's {@code --enable-auto-tool-choice}),
 * {@code --logits-processors} (one or more custom logits processor factory class names),
 * {@code --scheduling-policy},
 * {@code --logprobs-mode},
 * llama.cpp's self-extend and chunk-reuse defaults {@code --grp-attn-n} / {@code -gan}, {@code --grp-attn-w} /
 * {@code -gaw} and {@code --cache-reuse} (the request fields' defaults when a request names none),
 * {@code --otlp-traces-endpoint},
 * {@code --watermark-config} (the text watermark, in vLLM's JSON form),
 * {@code --kv-events-config}, {@code --kv-transfer-config}, {@code --prefix-caching-hash-algo}, the
 * {@code --data-parallel-*} set, the multi-LoRA
 * {@code --lora-modules},
 * {@code --max-loras}, {@code --max-lora-rank}, {@code --max-cpu-loras}, {@code --enable-lora}, the pooler
 * config's pooling type, {@code --override-pooler-config} / {@code --pooler-config}, the projector sources
 * ({@code --mmproj-url} / {@code -mmu} and the {@code --mmproj-auto} / {@code --no-mmproj} switch family), the
 * encoder scale-out ({@code --mm-encoder-tp-mode} and {@code --mmproj-devices} for the encoder's rank group,
 * {@code --ec-transfer-config} for a shared encoder cache), the media
 * loader options {@code --media-io-kwargs}, the video flags {@code --video-fps}, {@code --video-timestamp-interval},
 * {@code --video-ffmpeg-dir}, {@code --video-pruning-rate} and {@code --video-pruning-method}, the KV offload
 * tiers' {@code --cache-disk} / {@code --cache-disk-dir} beside {@code -cram}), and SGLang's
 * {@code --dp-size} and {@code --load-balance-method} ({@link DpLaunchFlags}), and llama-server's speculative
 * decoding flags ({@link SpecLaunchFlags}), and its web serving flags: the UI switch ({@code --no-ui}), the static
 * directory ({@code --path}), the UI defaults ({@code --ui-config}), the route prefix ({@code --api-prefix}),
 * {@code --reuse-port} and the CORS set ({@link CorsLaunchFlags}), vLLM's CORS spellings
 * ({@code --allowed-origins}, {@code --allowed-methods}, {@code --allowed-headers} and
 * {@code --allow-credentials}) included, and its router flags ({@code --models-dir},
 * {@code --models-preset}, {@code --models-max}, {@code --models-memory-margin}, {@code --models-autoload} /
 * {@code --no-models-autoload}) onto the {@code neuralfree.server.models.*} properties; an unknown flag, a missing
 * value or a value that cannot work (a port outside {@code [1, 65535]}, a missing file, a malformed KV events
 * config, contradicting data-parallel flags) is refused naming the flag. The development-endpoint switch
 * ({@code --dev-mode}, vLLM's {@code VLLM_SERVER_DEV_MODE}) is a flag of its own here ({@link DevMode}), and so are
 * the sleep-mode switch ({@code --enable-sleep-mode}, {@link SleepModeSwitch}), the gRPC serving switch
 * ({@code --grpc}, {@link GrpcSwitch}), vLLM's batch-invariant mode ({@code --batch-invariant},
 * {@code neuralfree.batch.invariant}), vLLM's profiler configuration
 * ({@code --profiler-config}, {@link ProfilerConfig}), and the distributed-serving control switches
 * ({@code --enable-elastic-ep} and {@code --enable-fault-tolerance}, {@link ClusterSwitches}). Immutable.
 *
 * @param properties  the system properties to set before the server starts, in flag order
 * @param help        whether {@code --help} was asked for; no other flag is then read
 * @param listDevices whether {@code --list-devices} was asked for; the server then prints the devices and does
 *                    not start, as llama-server's common argument does
 */
record ServerLaunchArgs(Map<String, String> properties, boolean help, boolean listDevices) {

    /** The pooler config keys {@link #poolerConfig} reads. */
    private static final List<String> POOLER_KEYS = List.of("pooling_type", "seq_pooling_type", "tok_pooling_type");

    /** The Quarkus socket option of {@code --reuse-port}, exposed as a runtime HTTP property. */
    private static final String REUSE_PORT_PROPERTY = "quarkus.http.so-reuse-port";

    /** A {@code --logprobs-mode} value, checked against vLLM's four modes. */
    private static String logprobsMode(String flag, String value) {
        try {
            return LogprobsMode.parse(value).label();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
    }

    /** A {@code --watermark-config} value, checked to parse as a watermark over {@code env} before the server
     *  starts; the environment may supply the key the JSON omits. */
    private static String watermarkConfig(String flag, String json, Map<String, String> env) {
        Properties check = new Properties();
        check.setProperty(WatermarkConfig.PROP_CONFIG, json);
        try {
            WatermarkConfig.fromProperties(check, env);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
        return json;
    }

    /** Parses launch {@code args} with this process's environment; {@code --help} short-circuits like llama.cpp's. */
    static ServerLaunchArgs parse(String... args) {
        return parse(System.getenv(), args);
    }

    /** Parses launch {@code args} over an explicit {@code env}, so the watermark key sources are testable.
     *
     * @throws IllegalArgumentException when an argument names no flag or carries an unusable value
     */
    static ServerLaunchArgs parse(Map<String, String> env, String... args) {
        Map<String, String> properties = new LinkedHashMap<>();
        List<String> apiKeys = new ArrayList<>();
        List<String> loras = new ArrayList<>();
        List<String> modules = new ArrayList<>();
        String sslKey = null;
        String sslCert = null;
        DpLaunchFlags dp = new DpLaunchFlags();
        ParallelLaunchFlags parallel = new ParallelLaunchFlags();
        CorsLaunchFlags cors = null;
        Boolean noMmproj = null;
        Boolean mmprojOffload = null;
        String mmprojDevice = null;
        String mmprojUrl = null;
        Map<String, Object> templateKwargs = null;
        String toolCallParser = null;
        boolean autoToolChoice = false;
        int i = 0;
        while (i < args.length) {
            String flag = args[i++];
            int consumed = dataParallel(dp, args, flag, i);
            if (consumed >= 0) {
                i += consumed;
                continue;
            }
            if (ParallelLaunchFlags.VALUED.contains(flag)) {
                parallel.value(flag, value(args, flag, i++));
                continue;
            }
            if (ParallelLaunchFlags.SWITCHES.contains(flag)) {
                parallel.toggle(flag);
                continue;
            }
            if (SpecLaunchFlags.handles(flag)) {
                SpecLaunchFlags.refuseUnsupported(flag);
                if (SpecLaunchFlags.isSwitch(flag)) {
                    SpecLaunchFlags.applySwitch(properties, flag,
                            i < args.length && !args[i].startsWith("-") ? args[i] : null);
                    continue;
                }
                SpecLaunchFlags.apply(properties, flag, value(args, flag, i++));
                continue;
            }
            if (CorsLaunchFlags.handles(flag)) {
                cors = cors == null ? new CorsLaunchFlags() : cors;
                cors = cors.with(flag, flag.endsWith("-credentials") ? "" : value(args, flag, i++));
                continue;
            }
            switch (flag) {
                case "-h", "--help" -> { return new ServerLaunchArgs(Map.of(), true, false); }
                case "--list-devices" -> { return new ServerLaunchArgs(Map.of(), false, true); }
                case "-m", "--model" -> properties.put("neuralfree.server.model", value(args, flag, i++));
                case "--host" -> properties.put("quarkus.http.host", value(args, flag, i++));
                case AllowedToolHosts.FLAG -> properties.put(AllowedToolHosts.PROPERTY,
                        AllowedToolHosts.validate(value(args, flag, i++), flag));
                case "--port" -> port(properties, value(args, flag, i++));
                case "-dev", "-device", "--device" -> device(properties, flag, value(args, flag, i++));
                case "--rpc" -> rpc(properties, flag, value(args, flag, i++));
                case "--numa" -> properties.put(NumaStrategy.PROPERTY, numa(flag, value(args, flag, i++)));
                case "-lm", "--load-mode" -> loadMode(properties, flag, value(args, flag, i++));
                case "--mlock" -> mlock(properties, flag);
                case "--no-mmap" -> setMmap(properties, false);
                case "-a", "--alias", "--served-model-name" ->
                        properties.put(ServedModels.PROPERTY, value(args, flag, i++));
                case "--api-key" -> apiKeys.add(key(value(args, flag, i++)));
                case "--api-key-file" ->
                        properties.put(ApiKeys.FILE_PROPERTY, keyFile(value(args, flag, i++)));
                case DevMode.FLAG -> properties.put(DevMode.PROPERTY, "true");
                case ClusterSwitches.ELASTIC_FLAG -> properties.put(ClusterSwitches.ELASTIC_PROPERTY, "true");
                case ClusterSwitches.FAULT_FLAG -> properties.put(ClusterSwitches.FAULT_PROPERTY, "true");
                case SleepModeSwitch.FLAG -> properties.put(SleepModeSwitch.PROPERTY, "true");
                case GrpcSwitch.FLAG -> properties.put(GrpcSwitch.PROPERTY, "true");
                case ProfilerConfig.FLAG -> profilerConfig(properties, value(args, flag, i++));
                case "-c", "--ctx-size", "--max-model-len" -> ctx(properties, value(args, flag, i++));
                case "-np", "--parallel" -> parallel(properties, value(args, flag, i++));
                case "--context-shift" -> properties.put(MemoryConfig.PROP_CONTEXT_SHIFT, "true");
                case "--no-context-shift" -> properties.put(MemoryConfig.PROP_CONTEXT_SHIFT, "false");
                case "--pooling" -> pooling(properties, value(args, flag, i++));
                case "--override-pooler-config", "--pooler-config" ->
                        poolerConfig(properties, flag, value(args, flag, i++));
                case "--jinja" -> properties.put(ChatTemplateSource.JINJA_PROPERTY, "true");
                case "--no-jinja" -> properties.put(ChatTemplateSource.JINJA_PROPERTY, "false");
                case "--metrics" -> { }
                case "--chat-template" ->
                        properties.put(ChatTemplateSource.OVERRIDE_PROPERTY, chatTemplate(value(args, flag, i++)));
                case "--chat-template-file" ->
                        properties.put(ChatTemplateSource.OVERRIDE_FILE_PROPERTY, templateFile(value(args, flag, i++)));
                case "--chat-template-kwargs", "--default-chat-template-kwargs" ->
                        templateKwargs = templateKwargs(templateKwargs, flag, value(args, flag, i++));
                case ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_FLAG ->
                        properties.put(ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY, "true");
                case "--chat-template-content-format" -> properties.put(ChatFormatter.CONTENT_FORMAT_PROPERTY,
                        contentFormat(value(args, flag, i++)));
                case "--reasoning-parser" -> properties.put(EngineConfig.PROP_REASONING_PARSER,
                        reasoningParser(flag, value(args, flag, i++)));
                case "--logits-processors", "--logits_processors" ->
                        i = logitsProcessors(args, i, properties);
                case "--tool-call-parser" -> {
                    toolCallParser = toolParser(flag, value(args, flag, i++));
                    properties.put(EngineConfig.PROP_TOOL_PARSER, toolCallParser);
                }
                case "--enable-auto-tool-choice" -> autoToolChoice = true;
                case "--reasoning-preserve" -> properties.put(ChatFormatter.PRESERVE_REASONING_PROPERTY, "true");
                case "--no-reasoning-preserve" -> properties.put(ChatFormatter.PRESERVE_REASONING_PROPERTY, "false");
                case ServerOptions.LOGPROBS_MODE_FLAG -> properties.put(ServerOptions.LOGPROBS_MODE_PROPERTY,
                        logprobsMode(flag, value(args, flag, i++)));
                case ServerOptions.OTLP_TRACES_ENDPOINT_FLAG -> properties.put(
                        ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY, value(args, flag, i++));
                case ServerOptions.WATERMARK_CONFIG_FLAG -> properties.put(WatermarkConfig.PROP_CONFIG,
                        watermarkConfig(flag, value(args, flag, i++), env));
                case "--lora" -> loras.add(adapter(value(args, flag, i++)));
                case "--lora-scaled" -> {
                    String path = value(args, flag, i++);
                    loras.add(adapter(path) + ":" + scale(value(args, flag, i++)));
                }
                case "--lora-modules" -> i = loraModules(args, i, modules);
                case "--enable-lora" -> properties.put(LoraConfig.PROP_ENABLE, "true");
                case "--batch-invariant" -> properties.put(BatchInvariance.PROPERTY, "true");
                case "-cmoed", "--spec-draft-cpu-moe", "--cpu-moe-draft" ->
                        properties.put(EngineConfig.PROP_DRAFT_CPU_MOE, "true");
                case "-td", "--spec-draft-threads", "--threads-draft" ->
                        draftThreads(properties, flag, value(args, flag, i++));
                case "-tbd", "--spec-draft-threads-batch", "--threads-batch-draft" ->
                        draftThreads(properties, flag, value(args, flag, i++));
                case "--max-loras" -> properties.put("neuralfree.lora.max",
                        whole(flag, value(args, flag, i++), 1, Integer.MAX_VALUE).toString());
                case "--max-lora-rank" -> properties.put("neuralfree.lora.max.rank",
                        whole(flag, value(args, flag, i++), 1, Integer.MAX_VALUE).toString());
                case "--max-cpu-loras" -> properties.put("neuralfree.lora.cache",
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--ssl-key-file", "--ssl-keyfile" -> sslKey = sslFile(flag, value(args, flag, i++));
                case "--ssl-cert-file", "--ssl-certfile" -> sslCert = sslFile(flag, value(args, flag, i++));
                case "-nkvo", "--no-kv-offload" -> properties.put(EngineConfig.PROP_KV_OFFLOAD, "false");
                case "-kvo", "--kv-offload" -> properties.put(EngineConfig.PROP_KV_OFFLOAD, "true");
                case "-cram", "--cache-ram" -> properties.put(MemoryConfig.PROP_CACHE_RAM,
                        whole(flag, value(args, flag, i++), -1, Integer.MAX_VALUE).toString());
                case "-cdisk", "--cache-disk" -> properties.put(MemoryConfig.PROP_CACHE_DISK_MIB,
                        whole(flag, value(args, flag, i++), -1, Integer.MAX_VALUE).toString());
                case "--cache-disk-dir" -> properties.put(MemoryConfig.PROP_CACHE_DISK_DIR,
                        directory(flag, value(args, flag, i++)));
                case "--slot-save-path" ->
                        properties.put(ServerOptions.SLOT_SAVE_PATH_PROPERTY, directory(flag, value(args, flag, i++)));
                case "--models-dir" ->
                        properties.put(ModelRouter.DIR_PROPERTY, directory(flag, value(args, flag, i++)));
                case "--models-preset" ->
                        properties.put(ModelRouter.PRESET_PROPERTY, regularFile(flag, value(args, flag, i++)));
                case "--models-max" -> properties.put(ModelRouter.MAX_PROPERTY,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--models-memory-margin" -> properties.put(ModelRouter.MARGIN_PROPERTY,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--models-autoload" -> properties.put(ModelRouter.AUTOLOAD_PROPERTY, "true");
                case "--no-models-autoload" -> properties.put(ModelRouter.AUTOLOAD_PROPERTY, "false");
                case "--kv-events-config" ->
                        properties.put(MemoryConfig.PROP_KV_EVENTS, kvEvents(value(args, flag, i++)));
                case ServerOptions.KV_TRANSFER_CONFIG_FLAG ->
                        properties.put(ServerOptions.KV_TRANSFER_CONFIG_PROPERTY,
                                kvTransferConfig(value(args, flag, i++), env));
                case "--scheduling-policy" -> properties.put(MemoryConfig.PROP_SCHEDULING_POLICY,
                        schedulingPolicy(value(args, flag, i++)));
                case "--prefix-caching-hash-algo" -> properties.put(MemoryConfig.PROP_PREFIX_HASH_ALGO,
                        hashAlgo(flag, value(args, flag, i++)));
                case "--ctx-checkpoints", "--swa-checkpoints", "-ctxcp" ->
                        properties.put(MemoryConfig.PROP_CTX_CHECKPOINTS,
                                whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--checkpoint-min-step", "-cms" -> properties.put(MemoryConfig.PROP_CHECKPOINT_MIN_STEP,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--cache-reuse" -> properties.put(OpenAiRequests.PROP_CACHE_REUSE,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--grp-attn-n", "-gan" -> properties.put(OpenAiRequests.PROP_GRP_ATTN_N,
                        whole(flag, value(args, flag, i++), 1, Integer.MAX_VALUE).toString());
                case "--grp-attn-w", "-gaw" -> properties.put(OpenAiRequests.PROP_GRP_ATTN_W,
                        whole(flag, value(args, flag, i++), 1, Integer.MAX_VALUE).toString());
                case "-mm", "--mmproj" ->
                        properties.put(MediaOptions.MMPROJ_PROPERTY, regularFile(flag, value(args, flag, i++)));
                case "-mmu", "--mmproj-url" -> mmprojUrl = mmprojUrl(flag, value(args, flag, i++));
                case "--mmproj-auto" -> noMmproj = false;
                case "--no-mmproj", "--no-mmproj-auto" -> noMmproj = true;
                case "--media-path", "--allowed-local-media-path" ->
                        properties.put(MediaOptions.LOCAL_PATH_PROPERTY, directory(flag, value(args, flag, i++)));
                case "--allowed-media-domains" ->
                        properties.put(MediaOptions.DOMAINS_PROPERTY, value(args, flag, i++));
                case "--limit-mm-per-prompt" ->
                        properties.put(MediaOptions.LIMITS_PROPERTY, limits(value(args, flag, i++)));
                case "--image-min-tokens" -> properties.put(MediaOptions.IMAGE_MIN_TOKENS_PROPERTY,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--image-max-tokens" -> properties.put(MediaOptions.IMAGE_MAX_TOKENS_PROPERTY,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--mm-processor-cache-gb" ->
                        properties.put(MediaOptions.CACHE_GB_PROPERTY, cacheGb(flag, value(args, flag, i++)));
                case "--media-io-kwargs" ->
                        properties.put(MediaOptions.IO_KWARGS_PROPERTY, ioKwargs(value(args, flag, i++)));
                case "--video-fps" -> properties.put(MediaOptions.VIDEO_FPS_PROPERTY,
                        videoFps(flag, value(args, flag, i++)));
                case "--video-timestamp-interval" -> properties.put(MediaOptions.VIDEO_TIMESTAMP_PROPERTY,
                        whole(flag, value(args, flag, i++), 0, Integer.MAX_VALUE).toString());
                case "--video-ffmpeg-dir" ->
                        properties.put(MediaOptions.VIDEO_FFMPEG_DIR_PROPERTY,
                                directory(flag, value(args, flag, i++)));
                case "--video-pruning-rate" -> properties.put(MediaOptions.VIDEO_PRUNING_RATE_PROPERTY,
                        videoPruningRate(flag, value(args, flag, i++)));
                case "--video-pruning-method" -> properties.put(MediaOptions.VIDEO_PRUNING_METHOD_PROPERTY,
                        videoPruningMethod(flag, value(args, flag, i++)));
                case "--mmproj-offload" -> mmprojOffload = true;
                case "--no-mmproj-offload" -> mmprojOffload = false;
                case "-mmdev", "--mmproj-device" -> {
                    mmprojDevice = mmprojDevice(flag, value(args, flag, i++));
                    mmprojOffload = mmprojDevice != null;
                }
                case "--mmproj-devices" ->
                        properties.put(MmprojOptions.PROP_DEVICES, mmprojDevices(flag, value(args, flag, i++)));
                case "--mm-encoder-tp-mode" ->
                        properties.put(MmprojOptions.PROP_TP_MODE, mmEncoderTpMode(flag, value(args, flag, i++)));
                case "--ec-transfer-config" ->
                        ecTransferConfig(properties, value(args, flag, i++));
                case ServerOptions.TOOLS_FLAG -> properties.put(ServerOptions.TOOLS_PROPERTY, tools(value(args, flag, i++)));
                case ServerOptions.MCP_CONFIG_FLAG ->
                        properties.put(ServerOptions.MCP_CONFIG_PROPERTY, regularFile(flag, value(args, flag, i++)));
                case ServerOptions.MCP_JSON_FLAG -> {
                    String json = value(args, flag, i++);
                    mcpJson(json);
                    properties.put(ServerOptions.MCP_JSON_PROPERTY, json);
                }
                case "--tools-runtime" -> throw new IllegalArgumentException("--tools-runtime is not supported"
                        + " by this build: the tools run in the server process (basic isolation), not in a"
                        + " container or over ssh");
                case "-ag", "--agent" -> properties.put(ServerOptions.TOOLS_PROPERTY, "all");
                case "--ui", "--webui" -> properties.put(UiResource.WEBUI_PROPERTY, "true");
                case "--no-ui", "--no-webui" -> properties.put(UiResource.WEBUI_PROPERTY, "false");
                case "--path" -> properties.put(UiResource.PATH_PROPERTY, directory(flag, value(args, flag, i++)));
                case "--ui-config", "--webui-config" -> properties.put(UiResource.CONFIG_PROPERTY,
                        uiConfig(flag, value(args, flag, i++)));
                case "--ui-config-file", "--webui-config-file" -> properties.put(UiResource.CONFIG_PROPERTY,
                        uiConfigFile(flag, value(args, flag, i++)));
                case "--api-prefix" -> properties.put(ApiPrefixRouter.PROPERTY,
                        ApiPrefixRouter.canonical(flag, value(args, flag, i++)));
                case "--reuse-port" -> properties.put(REUSE_PORT_PROPERTY, "true");
                default -> positional(properties, flag);
            }
        }
        if (!apiKeys.isEmpty()) {
            properties.put(ApiKeys.PROPERTY, String.join(",", apiKeys));
        }
        if (!loras.isEmpty()) {
            properties.put("neuralfree.lora", String.join(",", loras));
        }
        if (mmprojUrl != null) {
            properties.put(MediaOptions.MMPROJ_URL_PROPERTY, mmprojUrl);
        }
        if (noMmproj != null) {
            // the last flag of the projector switch family wins (llama.cpp's no_mmproj): "--no-mmproj" and
            // "--no-mmproj-auto" blank the explicit sources too, "--mmproj-auto" re-enables the search
            if (noMmproj) {
                properties.put(MediaOptions.MMPROJ_PROPERTY, "");
                properties.put(MediaOptions.MMPROJ_URL_PROPERTY, "");
                properties.put(MediaOptions.MMPROJ_AUTO_PROPERTY, "false");
            } else {
                properties.put(MediaOptions.MMPROJ_AUTO_PROPERTY, "true");
            }
        }
        if (mmprojOffload != null) {
            properties.put(MmprojOptions.PROP_OFFLOAD, mmprojOffload.toString());
        }
        if (mmprojDevice != null) {
            properties.put(MmprojOptions.PROP_DEVICE, mmprojDevice);
        }
        if (!modules.isEmpty()) {
            properties.put(ServerLoras.MODULES_PROPERTY, String.join(",", modules));
        }
        if (templateKwargs != null) {
            properties.put(ChatTemplateKwargs.PROPERTY, ChatTemplateKwargs.toJson(templateKwargs));
        }
        if ("false".equals(properties.get(ChatTemplateSource.JINJA_PROPERTY))) {
            verifyNoJinja(properties);
        }
        ssl(properties, sslKey, sslCert);
        tracing(properties);
        parallel.resolveInto(properties);
        dp.resolveInto(properties);
        encoderScaleOut(properties);
        if (autoToolChoice && toolCallParser == null) {
            throw new IllegalArgumentException("--enable-auto-tool-choice requires --tool-call-parser");
        }
        if (cors != null) {
            cors.resolveInto(properties);
        }
        SpecLaunchFlags.validate(properties);
        vertexPort(properties);
        return new ServerLaunchArgs(Collections.unmodifiableMap(properties), false, false);
    }

    /**
     * Vertex AI's {@code AIP_HTTP_PORT}: a prediction container announces the port it serves on, and llama.cpp
     * overrides its own port with it ({@code gcp_params}). The override applies only in prediction mode
     * ({@code AIP_MODE=PREDICTION}), as llama.cpp's does, and refuses a value outside the port range by name.
     */
    private static void vertexPort(Map<String, String> properties) {
        vertexPort(properties, System.getenv());
    }

    /** {@link #vertexPort(Map)} over an explicit environment, so the override is testable. */
    static void vertexPort(Map<String, String> properties, Map<String, String> env) {
        if (!"PREDICTION".equalsIgnoreCase(env.get(VertexCompat.MODE_ENV))) {
            return;
        }
        String value = env.get("AIP_HTTP_PORT");
        if (value == null || value.isBlank()) {
            return;
        }
        properties.put("quarkus.http.port", whole("AIP_HTTP_PORT", value, 1, 65_535).toString());
    }

    /**
     * Records {@code flag} into {@code dp} when it is a data-parallel flag, its value the argument at {@code at},
     * and answers how many arguments past the flag it consumed; -1 for any other flag.
     *
     * @throws IllegalArgumentException for a refused coordination flag, a missing or unusable value
     */
    private static int dataParallel(DpLaunchFlags dp, String[] args, String flag, int at) {
        if (DpLaunchFlags.VALUED.contains(flag)) {
            dp.value(flag, value(args, flag, at));
            return 1;
        }
        if (DpLaunchFlags.SWITCHES.contains(flag)) {
            dp.toggle(flag);
            return 0;
        }
        if (DpLaunchFlags.REFUSED.contains(flag)) {
            throw DpLaunchFlags.refusal(flag);
        }
        return -1;
    }

    /**
     * {@code earlier} with the template variables of {@code flag}'s JSON object over it: repeated flags merge,
     * a later flag's variable replacing an earlier one's, as llama.cpp's {@code --chat-template-kwargs} does.
     *
     * @throws IllegalArgumentException when {@code json} is not a JSON object, naming {@code flag}
     */
    private static Map<String, Object> templateKwargs(Map<String, Object> earlier, String flag, String json) {
        Map<String, Object> merged = earlier == null ? new LinkedHashMap<>() : new LinkedHashMap<>(earlier);
        merged.putAll(ChatTemplateKwargs.parse(json, flag));
        return merged;
    }

    /**
     * The values of vLLM's {@code --logits-processors} flag: one or more factory class names (each argument may
     * also hold a comma-separated list), stored comma-separated under
     * {@link EngineConfig#PROP_LOGITS_PROCESSORS}. Returns the index after the last value.
     */
    private static int logitsProcessors(String[] args, int at, Map<String, String> properties) {
        List<String> names = new ArrayList<>();
        int i = at;
        while (i < args.length && !args[i].startsWith("-")) {
            for (String part : args[i].split(",", -1)) {
                String name = part.trim();
                if (name.isEmpty()) {
                    throw new IllegalArgumentException("--logits-processors has an empty class name in '"
                            + args[i] + "'");
                }
                names.add(name);
            }
            i++;
        }
        if (names.isEmpty()) {
            throw new IllegalArgumentException("--logits-processors needs at least one factory class name");
        }
        properties.put(EngineConfig.PROP_LOGITS_PROCESSORS, String.join(",", names));
        return i;
    }

    /** The value of {@code flag}, the argument at {@code at}, refused when absent. */
    private static String value(String[] args, String flag, int at) {
        if (at >= args.length) {
            throw new IllegalArgumentException(flag + " needs a value");
        }
        String value = args[at];
        if (value.isBlank()) {
            throw new IllegalArgumentException(flag + " needs a value, got blank");
        }
        return value.trim();
    }

    /** The model of an unnamed positional argument (vLLM's {@code serve <model>}), refused when named twice. */
    private static void positional(Map<String, String> properties, String arg) {
        if (arg.startsWith("-")) {
            throw new IllegalArgumentException("unknown flag '" + arg + "'; run with --help for the flags");
        }
        String previous = properties.putIfAbsent("neuralfree.server.model", arg);
        if (previous != null) {
            throw new IllegalArgumentException("the model is named twice: '" + previous + "' and '" + arg + "'");
        }
    }

    /** The {@code neuralfree.scheduling.policy} of vLLM's {@code --scheduling-policy}: fcfs or priority. */
    private static String schedulingPolicy(String value) {
        return SchedulingPolicy.parse(value, "--scheduling-policy").wireName();
    }

    /**
     * The {@code neuralfree.prefix.hash.algo} of vLLM's {@code --prefix-caching-hash-algo}: sha256, sha256_cbor,
     * xxhash or xxhash_cbor, kept in vLLM's lowercase spelling.
     */
    private static String hashAlgo(String flag, String value) {
        return PrefixHashAlgo.parse(value, flag).algoName();
    }

    private static void port(Map<String, String> properties, String value) {
        properties.put("quarkus.http.port", whole("--port", value, 1, 65_535).toString());
    }

    /** The key file of {@code --api-key-file}, refused when it is not a regular file. */
    private static String keyFile(String value) {
        if (!Files.isRegularFile(Path.of(value))) {
            throw new IllegalArgumentException("--api-key-file is not a regular file: " + value);
        }
        return value;
    }

    private static void ctx(Map<String, String> properties, String value) {
        properties.put("neuralfree.ctx", whole("--ctx-size", value, 0, Integer.MAX_VALUE).toString());
    }

    private static void parallel(Map<String, String> properties, String value) {
        properties.put("neuralfree.parallel", whole("--parallel", value, 1, 256).toString());
    }

    /**
     * llama.cpp's {@code --numa}: the CPU pool's thread placement ({@link NumaStrategy#PROPERTY}). The pool reads
     * the property when it starts, so the flag must precede the first CPU computation, which a launch always does.
     *
     * @throws IllegalArgumentException naming the flag when the strategy is unknown
     */
    private static String numa(String flag, String value) {
        try {
            return NumaStrategy.parse(value).label();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
    }

    /**
     * llama.cpp's {@code -lm}/{@code --load-mode}: the process-wide tensor read mode ({@link TensorReadMode}).
     * {@code auto} leaves the configured mode alone.
     *
     * @throws IllegalArgumentException naming the flag for an unknown mode
     */
    private static void loadMode(Map<String, String> properties, String flag, String value) {
        String v = value.trim().toLowerCase(Locale.ROOT);
        TensorReadMode mode;
        switch (v) {
            case "auto": return;
            case "none": mode = TensorReadMode.READ; break;
            case "mmap": mode = TensorReadMode.MMAP; break;
            case "mlock": mode = TensorReadMode.MLOCK; break;
            case "mmap+mlock": mode = TensorReadMode.MMAP_MLOCK; break;
            case "dio": mode = TensorReadMode.DIRECT; break;
            default:
                throw new IllegalArgumentException(flag + " needs auto, none, mmap, mlock, mmap+mlock or dio,"
                        + " not '" + value + "'");
        }
        properties.put(TensorReadMode.PROP_READ_MODE, mode.flagName());
    }

    /** {@code --mlock}: pin the model's tensor pages in RAM on top of the configured load mode. */
    private static void mlock(Map<String, String> properties, String flag) {
        try {
            properties.put(TensorReadMode.PROP_READ_MODE, effectiveReadMode(properties).withMlock(true).flagName());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
    }

    /** {@code --no-mmap}: read tensors into process buffers, keeping a pin of the configured load mode. */
    private static void setMmap(Map<String, String> properties, boolean mmap) {
        properties.put(TensorReadMode.PROP_READ_MODE, effectiveReadMode(properties).withMmap(mmap).flagName());
    }

    /** The read mode the launch so far selects: the system properties with the accumulated launch flags over them. */
    private static TensorReadMode effectiveReadMode(Map<String, String> properties) {
        Properties merged = new Properties();
        merged.putAll(System.getProperties());
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            merged.setProperty(entry.getKey(), entry.getValue());
        }
        return TensorReadMode.fromProperties(merged);
    }

    /**
     * llama.cpp's {@code --device}: the GPUs the model's load runs over. The list is read and validated by the CLI's
     * own reader ({@link EngineLoadFlags#deviceSelection}), so both surfaces refuse the same lists. One selector sets
     * the device list ({@link EngineConfig#PROP_GPU_DEVICES}); one name substring also sets the one-device load's GPU
     * ({@link EngineConfig#PROP_DEVICE}); two or more split by layer; and {@code none} runs the CPU engine.
     *
     * @throws IllegalArgumentException naming the flag for an empty or mixed device list
     */
    private static void device(Map<String, String> properties, String flag, String value) {
        String selection;
        try {
            selection = EngineLoadFlags.deviceSelection(flag, value);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
        if (selection == null) {
            return;
        }
        // this flag is the only writer of the four load properties: a later --device replaces an earlier one
        properties.remove(EngineConfig.PROP_BACKEND);
        properties.remove(EngineConfig.PROP_GPU_DEVICES);
        properties.remove(EngineConfig.PROP_DEVICE);
        properties.remove(EngineConfig.PROP_SPLIT_MODE);
        if (selection.equals("none")) {
            if (hasRpcServers(properties)) {
                throw new IllegalArgumentException(flag + " none keeps the model off every GPU and cannot carry the"
                        + " remote servers of --rpc");
            }
            properties.put(EngineConfig.PROP_BACKEND, "cpu");
            return;
        }
        properties.put(EngineConfig.PROP_GPU_DEVICES, selection);
        String[] selectors = selection.split(",");
        if (selectors.length == 1 && !EngineConfig.isDeviceIndex(selectors[0])) {
            properties.put(EngineConfig.PROP_DEVICE, selectors[0]);
        } else if (selectors.length > 1 || hasRpcServers(properties)) {
            properties.put(EngineConfig.PROP_SPLIT_MODE, "layer");
        }
    }

    /**
     * Whether remote compute servers are configured, either by an earlier {@code --rpc} of this launch or by a
     * preset {@code neuralfree.rpc} system property (the {@code neuralfree serve} relaunch forwards it in the
     * secret properties file, so it never appears among the arguments).
     */
    private static boolean hasRpcServers(Map<String, String> properties) {
        if (properties.containsKey(EngineConfig.PROP_RPC)) {
            return true;
        }
        String preset = System.getProperty(EngineConfig.PROP_RPC);
        return preset != null && !preset.trim().isEmpty();
    }

    /**
     * llama.cpp's {@code --rpc}: comma-separated {@code host:port[@key]} remote compute servers a layer split adds
     * as devices. Every entry is validated here, naming the flag, so the server refuses a malformed endpoint before
     * it loads; the load property is {@link EngineConfig#PROP_RPC}.
     *
     * @throws IllegalArgumentException naming the flag for a malformed entry, or beside {@code --device none}
     */
    private static void rpc(Map<String, String> properties, String flag, String value) {
        if ("cpu".equals(properties.get(EngineConfig.PROP_BACKEND))) {
            throw new IllegalArgumentException(flag + " cannot be combined with the CPU engine (--device none)");
        }
        for (String entry : value.split(",", -1)) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new IllegalArgumentException(flag + " " + RpcServerSpec.masked(value)
                        + " has an empty server entry");
            }
            try {
                RpcServerSpec.parse(s);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(flag + " " + RpcServerSpec.masked(value) + ": "
                        + e.getMessage(), e);
            }
        }
        properties.put(EngineConfig.PROP_RPC, value);
        properties.put(EngineConfig.PROP_SPLIT_MODE, "layer");
    }

    /**
     * The pooling type override of {@code --pooling} (llama.cpp's flag): a {@link PoolingKind} name, refused naming
     * the flag.
     */
    private static void pooling(Map<String, String> properties, String value) {
        properties.put("neuralfree.pooling", poolingValue("--pooling", value));
    }

    /**
     * The pooling type override of vLLM's pooler config ({@code --override-pooler-config}, the newer
     * {@code --pooler-config} spelling): a JSON object naming the pooling by {@code pooling_type}
     * ({@code CLS | LAST | MEAN | ALL}, and {@code RANK} as llama.cpp names it, case-insensitive), or by
     * {@code seq_pooling_type} ({@code CLS | LAST | MEAN}) or {@code tok_pooling_type} ({@code ALL}, per-token
     * pooling), as vllm/config/pooler.py resolves them. vLLM refuses {@code pooling_type} beside either of the
     * others; the engine runs one pooling, so the sequence and token types together refuse too, and so does any
     * other key (this server takes no pooler setting beyond the pooling type). An empty object keeps the defaults.
     */
    private static void poolerConfig(Map<String, String> properties, String flag, String value) {
        Object parsed;
        try {
            parsed = Json.parse(value, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(flag + " is not a JSON object: " + e.getMessage());
        }
        if (!(parsed instanceof Map<?, ?> config)) {
            throw new IllegalArgumentException(flag + " is a JSON object with a pooling_type, got "
                    + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
        }
        for (Object key : config.keySet()) {
            if (!POOLER_KEYS.contains(String.valueOf(key))) {
                throw new IllegalArgumentException(flag + " key '" + key + "' is not supported; only "
                        + String.join(", ", POOLER_KEYS) + " are (this server takes no pooler setting beyond the"
                        + " pooling type)");
            }
        }
        String pooling = poolerString(flag, config, "pooling_type");
        String seq = poolerString(flag, config, "seq_pooling_type");
        String tok = poolerString(flag, config, "tok_pooling_type");
        if (pooling != null && (seq != null || tok != null)) {
            throw new IllegalArgumentException(flag + ": cannot set both pooling_type and "
                    + (seq != null ? "seq_pooling_type" : "tok_pooling_type"));
        }
        if (seq != null && tok != null) {
            throw new IllegalArgumentException(flag + ": seq_pooling_type and tok_pooling_type together are not"
                    + " supported; the engine runs one pooling type, so name one of them");
        }
        if (seq != null) {
            pooling = poolerType(flag, "seq_pooling_type", seq, "CLS", "LAST", "MEAN");
        } else if (tok != null) {
            pooling = poolerType(flag, "tok_pooling_type", tok, "ALL", "STEP");
        }
        if (pooling != null) {
            properties.put("neuralfree.pooling", poolingValue(flag, pooling));
        }
    }

    /** The string value of pooler config key {@code key}, or null when absent or JSON null; refused otherwise. */
    private static String poolerString(String flag, Map<?, ?> config, String key) {
        Object v = config.get(key);
        if (v == null) {
            return null;
        }
        if (!(v instanceof String s)) {
            throw new IllegalArgumentException(flag + " " + key + " is a string, got " + v);
        }
        return s;
    }

    /** {@code value} of pooler key {@code key} when it is one of {@code allowed} (case-insensitive); refused else. */
    private static String poolerType(String flag, String key, String value, String... allowed) {
        for (String a : allowed) {
            if (a.equalsIgnoreCase(value.trim())) {
                return a;
            }
        }
        throw new IllegalArgumentException(flag + " " + key + " '" + value + "' is invalid; use "
                + String.join(", ", allowed));
    }

    /** The {@code neuralfree.pooling} value of one pooling name, refused naming the flag, stored canonical. */
    private static String poolingValue(String flag, String value) {
        String canonical;
        try {
            canonical = PoolingKind.parse(value).name().toLowerCase(Locale.ROOT);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage());
        }
        return canonical;
    }

    /** The key of {@code --api-key}, refused when blank (a blank key would disable nothing and match nothing). */
    private static String key(String value) {
        if (value.isBlank()) {
            throw new IllegalArgumentException("--api-key must not be blank");
        }
        return value;
    }

    /** The {@code --chat-template-content-format} value, refused naming the flag unless auto, string or openai. */
    private static String contentFormat(String value) {
        try {
            return TemplateCaps.ContentFormat.parse(value).value();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("--chat-template-content-format must be auto, string or openai, got '"
                    + value + "'");
        }
    }

    /**
     * The {@code --reasoning-parser} name (vLLM's flag): {@code auto}, {@code none} or one of the names
     * {@link ReasoningParsers} registers, refused by name otherwise.
     */
    private static String reasoningParser(String flag, String value) {
        if (!value.equalsIgnoreCase("auto") && !value.equalsIgnoreCase("none")
                && ReasoningParsers.forName(value) == null) {
            throw new IllegalArgumentException(flag + ": unknown reasoning parser '" + value + "'; use auto, none"
                    + " or one of " + ReasoningParsers.names());
        }
        return value;
    }

    /**
     * The {@code --tool-call-parser} name (vLLM's flag): {@code auto}, {@code none} or one of the names
     * {@link ToolParserRegistry} registers, refused by name otherwise.
     */
    private static String toolParser(String flag, String value) {
        if (!value.equalsIgnoreCase("auto") && !value.equalsIgnoreCase("none")
                && ToolParserRegistry.forName(value) == null) {
            throw new IllegalArgumentException(flag + ": unknown tool-call parser '" + value + "'; use auto, none"
                    + " or one of " + ToolParserRegistry.names());
        }
        return value;
    }

    /**
     * The template override of {@code --chat-template}, refused when it is neither a built-in name nor Jinja source
     * and names no template file ({@link ChatTemplateSource#fromOverride}).
     */
    private static String chatTemplate(String value) {
        try {
            ChatTemplateSource.fromOverride(value);
        } catch (IllegalArgumentException | UncheckedIOException e) {
            throw new IllegalArgumentException("--chat-template: " + e.getMessage(), e);
        }
        return value;
    }

    /** The template override file of {@code --chat-template-file}, refused when it is not a regular file. */
    private static String templateFile(String value) {
        Path file = Path.of(value);
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException("--chat-template-file is not a regular file: " + value);
        }
        return value;
    }

    /**
     * The {@code --no-jinja} check llama.cpp runs at parse time (common/arg.cpp:940-946): a template
     * override whose text no built-in template is detected from ({@link ChatTemplateDetector#detect})
     * is refused here naming the template; without an override the model's own template is detected
     * when the tokenizer is built.
     */
    private static void verifyNoJinja(Map<String, String> properties) {
        String inline = properties.get(ChatTemplateSource.OVERRIDE_PROPERTY);
        String file = properties.get(ChatTemplateSource.OVERRIDE_FILE_PROPERTY);
        if (inline == null && file == null) {
            return;
        }
        String raw;
        try {
            raw = inline != null ? ChatTemplateSource.overrideValueText(inline)
                    : ChatTemplateSource.overrideText(Path.of(file));
        } catch (UncheckedIOException e) {
            throw new IllegalArgumentException("--no-jinja: the chat template override does not read: "
                    + e.getMessage(), e);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("--no-jinja: " + e.getMessage(), e);
        }
        if (ChatTemplateDetector.detect(raw) == null) {
            throw new IllegalArgumentException("--no-jinja: " + ChatTemplateDetector.refusal(raw));
        }
    }

    /** One {@code neuralfree.lora} adapter path ({@code --lora}, {@code --lora-scaled}). */
    private static String adapter(String value) {
        if (value.contains(":")) {
            throw new IllegalArgumentException(
                    "--lora names a path, not path:scale; use --lora-scaled for a scale");
        }
        return value;
    }

    /**
     * The adapters of {@code --lora-modules} (vLLM's per-request adapters loaded at startup): every argument from
     * {@code at} up to the next flag, each {@code name=path} or vLLM's JSON {@code {"name", "path"}}, added to
     * {@code out} as {@code name=path}; returns the index after them.
     *
     * @throws IllegalArgumentException when no adapter follows, an entry is malformed, or its path does not exist
     */
    private static int loraModules(String[] args, int at, List<String> out) {
        int i = at;
        while (i < args.length && !args[i].startsWith("-")) {
            String entry = args[i++].trim();
            String module = entry.startsWith("{") ? jsonModule(entry) : entry;
            int eq = module.indexOf('=');
            if (eq <= 0 || eq == module.length() - 1) {
                throw new IllegalArgumentException("--lora-modules entry '" + entry + "' is not name=path or"
                        + " {\"name\": ..., \"path\": ...}");
            }
            String name = module.substring(0, eq).trim();
            String path = module.substring(eq + 1).trim();
            ServerLoras.requireName(name);
            if (path.indexOf(',') >= 0) {
                throw new IllegalArgumentException("--lora-modules: the path of " + name + " holds ','");
            }
            if (!Files.exists(Path.of(path))) {
                throw new IllegalArgumentException("--lora-modules: the adapter path of " + name
                        + " does not exist: " + path);
            }
            out.add(name + "=" + path);
        }
        if (i == at) {
            throw new IllegalArgumentException("--lora-modules needs at least one name=path adapter");
        }
        return i;
    }

    /** The {@code name=path} of a JSON module; {@code base_model_name} and other keys refuse by name. */
    private static String jsonModule(String entry) {
        Object parsed;
        try {
            parsed = Json.parse(entry, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("--lora-modules entry is not valid JSON: " + e.getMessage());
        }
        if (!(parsed instanceof Map<?, ?> module)) {
            throw new IllegalArgumentException("--lora-modules JSON entry must be an object");
        }
        for (Object key : module.keySet()) {
            if (!"name".equals(key) && !"path".equals(key)) {
                throw new IllegalArgumentException("--lora-modules: the key '" + key + "' is not supported; an"
                        + " adapter's parent is the served model");
            }
        }
        if (!(module.get("name") instanceof String name) || !(module.get("path") instanceof String path)) {
            throw new IllegalArgumentException("--lora-modules JSON entry needs string \"name\" and \"path\"");
        }
        return name + "=" + path;
    }

    /** The {@code --lora-scaled} scale, a number. */
    private static String scale(String value) {
        try {
            Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("--lora-scaled needs a number as the scale, got '" + value + "'");
        }
        return value;
    }

    /** The {@code --ssl-*-file} file, refused when it is not a regular file. */
    private static String sslFile(String flag, String value) {
        if (!Files.isRegularFile(Path.of(value))) {
            throw new IllegalArgumentException(flag + " is not a regular file: " + value);
        }
        return value;
    }

    /** TLS serves on the launch port itself, as llama.cpp's {@code --ssl-*-file} does, and plain HTTP goes off. */
    private static void ssl(Map<String, String> properties, String key, String cert) {
        if (key != null && cert != null) {
            properties.put("quarkus.http.ssl.certificate.key-files", key);
            properties.put("quarkus.http.ssl.certificate.files", cert);
            properties.put("quarkus.http.ssl-port", properties.getOrDefault("quarkus.http.port", "8080"));
            properties.put("quarkus.http.insecure-requests", "disabled");
            return;
        }
        if (key != null) {
            throw new IllegalArgumentException("--ssl-key-file needs --ssl-cert-file with it");
        }
        if (cert != null) {
            throw new IllegalArgumentException("--ssl-cert-file needs --ssl-key-file with it");
        }
    }

    /**
     * The OpenTelemetry runtime of an endpoint flag ({@code --otlp-traces-endpoint}, the pre-set property
     * included): the SDK ships disabled, so the flag turns it on and points its exporter at the endpoint.
     * {@code OTEL_EXPORTER_OTLP_TRACES_PROTOCOL} picks the exporter protocol (vLLM reads the same variable),
     * {@code grpc} by default as Quarkus's is.
     */
    private static void tracing(Map<String, String> properties) {
        String endpoint = properties.get(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY);
        if (endpoint == null) {
            return;
        }
        properties.put("quarkus.otel.sdk.disabled", "false");
        properties.put("quarkus.otel.exporter.otlp.traces.endpoint", endpoint);
    }

    /** The file of {@code flag}, refused when it is not a regular file. */
    private static String regularFile(String flag, String value) {
        if (!Files.isRegularFile(Path.of(value))) {
            throw new IllegalArgumentException(flag + " is not a regular file: " + value);
        }
        return value;
    }

    /** The {@code --tools} list: {@code all} or a comma-separated set of built-in names, refused otherwise. */
    private static String tools(String value) {
        if (value.equals("all")) {
            return value;
        }
        for (String name : value.split(",")) {
            if (!ToolsRegistry.isBuiltin(name.trim())) {
                throw new IllegalArgumentException(ServerOptions.TOOLS_FLAG + ": unknown tool '" + name.trim()
                        + "'. available tools: " + String.join(", ", ToolsRegistry.builtinNames())
                        + ", or all");
            }
        }
        return value;
    }

    /** The inline {@code --mcp-servers-json}, validated now so a bad document fails the launch by name. */
    private static void mcpJson(String json) {
        try {
            McpServerConfig.parse(json, ServerOptions.MCP_JSON_FLAG);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(ServerOptions.MCP_JSON_FLAG + ": " + e.getMessage(), e);
        }
    }

    /** The {@code --limit-mm-per-prompt} JSON object, validated now so a bad value fails the launch by name. */
    private static String limits(String value) {
        MediaLimits.parse(value);
        return value;
    }

    /** The {@code --media-io-kwargs} JSON object, validated now so a bad value fails the launch by name. */
    private static String ioKwargs(String value) {
        MediaIoOptions.parse(value);
        return value;
    }

    /** The GiB count of {@code --mm-processor-cache-gb}, validated now so a bad value fails the launch by name. */
    private static String cacheGb(String flag, String value) {
        try {
            MediaOptions.cacheBytes(value);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + " must be a non-negative number of GiB, got '" + value + "'", e);
        }
        return value.trim();
    }

    /** The {@code --video-fps} rate: a non-negative finite number, 0 taking the source rate. */
    private static String videoFps(String flag, String value) {
        double fps;
        try {
            fps = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " must be a number, got '" + value + "'", e);
        }
        if (!(fps >= 0) || Double.isNaN(fps) || Double.isInfinite(fps)) {
            throw new IllegalArgumentException(flag + " must be a non-negative number, got '" + value + "'");
        }
        return value;
    }

    /** The {@code --video-pruning-rate}: a number in {@code [0, 1)}. */
    private static String videoPruningRate(String flag, String value) {
        double rate;
        try {
            rate = Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " must be a number in [0, 1), got '" + value + "'", e);
        }
        if (!(rate >= 0) || rate >= 1 || Double.isNaN(rate) || Double.isInfinite(rate)) {
            throw new IllegalArgumentException(flag + " must be a number in [0, 1), got '" + value + "'");
        }
        return value;
    }

    /** The {@code --video-pruning-method}: vLLM's {@code evs} or {@code vidcom2}, in canonical form. */
    private static String videoPruningMethod(String flag, String value) {
        try {
            return VideoPruner.Method.of(value).key();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
    }

    /**
     * The vision encoder device of {@code --mmproj-device}, or null for {@value MmprojOptions#NO_DEVICE} (the
     * encoder stays on the CPU, as llama.cpp's {@code -mmdev none} turns the offload off).
     */
    private static String mmprojDevice(String flag, String value) {
        MmprojOptions options;
        try {
            options = MmprojOptions.onDevice(value);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
        return options.device();
    }

    /**
     * The comma-separated device selectors of {@code --mmproj-devices}, the encoder's rank group in canonical form
     * (one trimmed selector per rank, none repeated).
     */
    private static String mmprojDevices(String flag, String value) {
        MmprojOptions options;
        try {
            options = MmprojOptions.defaults().withTpMode(MmEncoderTpMode.DATA)
                    .withDevices(Arrays.asList(value.split(",", -1)));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
        return String.join(",", options.devices());
    }

    /** The encoder parallelism of {@code --mm-encoder-tp-mode}, vLLM's {@code weights} or {@code data}. */
    private static String mmEncoderTpMode(String flag, String value) {
        try {
            return MmEncoderTpMode.parse(value).label();
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(flag + ": " + e.getMessage(), e);
        }
    }

    /**
     * The shared encoder cache of {@code --ec-transfer-config} (vLLM's JSON form of the encoder cache connector):
     * its shared directory and role become {@value EncoderCacheConfig#PROP_DIR} and
     * {@value EncoderCacheConfig#PROP_ROLE}.
     */
    private static void ecTransferConfig(Map<String, String> properties, String json) {
        EncoderCacheConfig config = EncoderCacheConfig.parseTransferConfig(json, "--ec-transfer-config");
        properties.put(EncoderCacheConfig.PROP_DIR, config.directory().toString());
        properties.put(EncoderCacheConfig.PROP_ROLE, config.role().label());
    }

    /**
     * Refuses {@code --mmproj-devices} without {@code --mm-encoder-tp-mode data}: weights mode runs the encoder
     * whole on one device, so the rest of a rank group would never be used. A mode set as a system property
     * counts, as the launch flags override the properties.
     */
    private static void encoderScaleOut(Map<String, String> properties) {
        if (!properties.containsKey(MmprojOptions.PROP_DEVICES)) {
            return;
        }
        String mode = properties.containsKey(MmprojOptions.PROP_TP_MODE)
                ? properties.get(MmprojOptions.PROP_TP_MODE) : System.getProperty(MmprojOptions.PROP_TP_MODE);
        if (mode == null || !MmEncoderTpMode.DATA.label().equalsIgnoreCase(mode.trim())) {
            throw new IllegalArgumentException("--mmproj-devices names the encoder's rank group, which only"
                    + " --mm-encoder-tp-mode data spreads the images over; the encoder runs whole on one device"
                    + " under weights mode — give --mm-encoder-tp-mode data, or one device with --mmproj-device");
        }
    }

    /** The projector URL of {@code --mmproj-url}, http or https, refused naming the flag otherwise. */
    private static String mmprojUrl(String flag, String value) {
        String lower = value.toLowerCase(Locale.ROOT);
        if (!lower.startsWith("http://") && !lower.startsWith("https://")) {
            throw new IllegalArgumentException(flag + " must be an http or https URL, got '" + value + "'");
        }
        return value;
    }

    /** The directory of {@code flag} ({@code --slot-save-path}, the local media path), refused when it is not one. */
    private static String directory(String flag, String value) {
        if (!Files.isDirectory(Path.of(value))) {
            throw new IllegalArgumentException(flag + " is not a directory: " + value);
        }
        return value;
    }

    /**
     * The JSON of {@code --ui-config} (llama.cpp's {@code ui_config_json}), the defaults the bundled UI applies
     * under its stored user settings, validated now so a bad value fails the launch by name.
     */
    private static String uiConfig(String flag, String json) {
        try {
            Json.parse(json, Json.Limits.UNTRUSTED);
        } catch (RuntimeException e) {
            throw new IllegalArgumentException(flag + " is not valid JSON: " + e.getMessage(), e);
        }
        return json.trim();
    }

    /** The file contents of {@code --ui-config-file}, validated like {@link #uiConfig}, refused when unreadable. */
    private static String uiConfigFile(String flag, String value) {
        Path file = Path.of(value);
        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException(flag + " is not a regular file: " + value);
        }
        try {
            return uiConfig(flag, Files.readString(file, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new IllegalArgumentException(flag + " does not read: " + value, e);
        }
    }

    /**
     * The {@code neuralfree.kv.events} publisher of {@code --kv-events-config}, vLLM's {@code KVEventsConfig} JSON:
     * {@code enable_kv_cache_events} (default false) turns publishing on; {@code publisher} picks the publisher —
     * unset resolves to {@code zmq} as vLLM's does, {@code null} for none, {@code log}, {@code zmq}, or the class
     * name of a {@code KvEventPublisher}. The property answers the publisher's name, or the config JSON itself when
     * it carries the ZMQ transport fields ({@code endpoint}, {@code replay_endpoint}, {@code buffer_steps},
     * {@code hwm}, {@code max_queue_size}, {@code topic}), which only a {@code zmq} publisher may name.
     */
    private static String kvEvents(String json) {
        String flag = "--kv-events-config";
        KvEventsConfig config = KvEventsConfig.parse(json, flag);
        if (!config.enabled()) {
            return "none";
        }
        if (config.zmq() == null) {
            return config.publisher();
        }
        return config.hasTransportFields() ? json : KvEventsConfig.ZMQ;
    }

    /**
     * The JSON of vLLM's {@code --kv-transfer-config} ({@link KvTransferConfig}), validated now — over {@code env},
     * which may supply the shared token — so a bad field or token source fails the launch by name; the connector
     * itself is created with the engine, where one this build does not ship (NIXL, Mooncake, LMCache) refuses by
     * name.
     */
    private static String kvTransferConfig(String json, Map<String, String> env) {
        String flag = ServerOptions.KV_TRANSFER_CONFIG_FLAG;
        KvTransferConfig.parse(json, flag, env);
        return json;
    }

    /**
     * The JSON of vLLM's {@code --profiler-config} ({@link ProfilerConfig}), validated now against the launch's
     * {@code neuralfree.server.profile.dir} fallback so a bad value fails the launch by name.
     */
    private static void profilerConfig(Map<String, String> properties, String json) {
        Properties merged = new Properties();
        merged.putAll(System.getProperties());
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            merged.setProperty(entry.getKey(), entry.getValue());
        }
        merged.setProperty(ProfilerConfig.PROPERTY, json);
        try {
            ProfilerConfig.from(merged);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(ProfilerConfig.FLAG + ": " + e.getMessage(), e);
        }
        properties.put(ProfilerConfig.PROPERTY, json);
    }

    /** The whole number {@code flag} accepts, inside {@code [min, max]}. */
    static Long whole(String flag, String value, long min, long max) {
        long number;
        try {
            number = Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " must be a whole number, got '" + value + "'");
        }
        if (number < min || number > max) {
            throw new IllegalArgumentException(flag + " must be between " + min + " and " + max + ", got " + number);
        }
        return number;
    }

    /**
     * The CPU thread pool size of {@code --threads-draft} / {@code --threads-batch-draft} (llama.cpp {@code -td},
     * {@code -tbd}): the engine runs the target's and the draft model's host tensors on one pool, which
     * {@code neuralfree.cpu.threads} sizes, so both flags set that property and must agree.
     */
    private static void draftThreads(Map<String, String> properties, String flag, String value) {
        Long n = whole(flag, value, 1, Integer.MAX_VALUE);
        String previous = properties.get(CpuParallel.THREADS_PROPERTY);
        if (previous != null && !previous.equals(n.toString())) {
            throw new IllegalArgumentException(flag + " " + n + " conflicts with the " + CpuParallel.THREADS_PROPERTY
                    + "=" + previous + " an earlier draft-threads flag set: the engine runs the target's and the"
                    + " draft model's host tensors on one CPU thread pool, so the draft thread flags must agree");
        }
        properties.put(CpuParallel.THREADS_PROPERTY, n.toString());
    }
}
