package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryRequirement;
import com.eonmux.neuralfree.demo.LLMDemoOptions.Mode;
import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.PocketTtsSession;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.BufferedOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

/**
 * Command-line front end of the GGUF engine: text generation (console chat or one prompt, optionally speculative),
 * embeddings, classification, reranking, masked diffusion and audio models. Models open through
 * {@link LLMModelFactory}; {@code --help} lists the options ({@link LLMDemoOptions}).
 */
public final class LLMDemo {

    private static final List<String> statsLog = new ArrayList<>();
    private static final String SAMPLE_EMBED_TEXT = "The quick brown fox jumps over the lazy dog.";
    private static final double MIB = 1024.0 * 1024.0;

    private LLMDemo() {
    }

    public static void main(String[] args) {
        LLMDemoOptions opts;
        try {
            opts = LLMDemoOptions.parse(args);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.println("Run with --help for the list of options.");
            System.exit(2);
            return;
        }
        if (opts.help) {
            for (String line : LLMDemoOptions.usage()) {
                System.out.println(line);
            }
            return;
        }
        String modelPath = resolveModelPath(opts);
        if (modelPath == null || !new java.io.File(modelPath).isFile()) {
            System.err.println(modelPath == null ? "No model specified." : "Model file not found: " + modelPath);
            System.err.println("Specify a GGUF model with one of:");
            System.err.println("  --model <path>            (e.g. ./gradlew runLLMDemo -Pargs=\"--model /path/to/model.gguf\")");
            System.err.println("  export NEURALFREE_MODEL=<path>");
            System.err.println("  -Dneuralfree.model=<path>");
            System.exit(2);
            return;
        }
        if (opts.noMmap) {
            System.setProperty(TensorReadMode.PROP_DISABLE_MMAP, "true");
        }
        if (opts.engine.deviceFilter() != null) {
            System.setProperty(EngineConfig.PROP_DEVICE, opts.engine.deviceFilter());
        }

        System.out.println("=== LLM Demo - NeuralFree GGUF Inference ===");
        System.out.println("Model: " + modelPath);
        System.out.println("Acceleration: " + (opts.engine.backend().usesGpu()
                ? "GPU (" + opts.engine.backend().name().toLowerCase(Locale.ROOT) + ")" : "CPU"));
        System.out.println();

        try {
            runSession(modelPath, opts);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            String m = e.getMessage();
            if (m != null && (m.contains("CL_MEM_OBJECT_ALLOCATION_FAILURE")
                    || m.contains("CL_OUT_OF_RESOURCES")
                    || m.contains("CL_OUT_OF_HOST_MEMORY")
                    || m.contains("out of memory"))) {
                System.err.println("=> Out of GPU memory for this model on this device.");
                System.err.println("   Try `--cpu`, a smaller quantization, or a smaller model.");
                System.err.println("   You can also lower the KV-cache size with `--ctx <N>` or `--kv-type q8_0`.");
            }
            e.printStackTrace();
            logStats("ERROR: " + e.getMessage());
            System.exit(1);
        } finally {
            if (opts.statsFile != null) {
                writeStatsToFile(opts.statsFile);
            }
        }
    }

    /**
     * The model path: the command line, else {@code -Dneuralfree.model}, else {@code NEURALFREE_MODEL}; null when none
     * of them gives one.
     */
    private static String resolveModelPath(LLMDemoOptions opts) {
        if (opts.modelPath != null) {
            return opts.modelPath;
        }
        String path = System.getProperty("neuralfree.model");
        if (path == null || path.isEmpty()) {
            path = System.getenv("NEURALFREE_MODEL");
        }
        return path != null && !path.isEmpty() ? path : null;
    }

    // ================================================================== session

    private static void runSession(String modelPath, LLMDemoOptions opts) throws IOException {
        Mode mode = resolveMode(modelPath, opts);
        EngineConfig cfg = opts.engineConfig(mode);
        HWAcceleration hw = null;
        ComputeBackend backend = null;
        if (cfg.backend().usesGpu()) {
            ACCELERATION_MODE accel = cfg.backend().toAccelerationMode();
            System.out.println("Initializing GPU backend...");
            try {
                Config.setAccelerationMode(accel);
                hw = HWAcceleration.forMode(accel);
                backend = hw.createBackend(accel, KernelLibraryRequirement.of(KernelLibraries.LLM));
                System.out.println("Backend: " + backend.backendType() + " (" + backend.capabilities().getDeviceClass() + ")");
                System.out.println("Device: " + backend.capabilities().getDeviceName());
                System.out.println("VRAM: " + (backend.capabilities().getTotalVramBytes() / (1024 * 1024)) + " MB");
                System.out.println();
            } catch (RuntimeException | LinkageError e) {
                System.err.println("GPU initialization failed: " + e.getMessage());
                System.err.println("Falling back to CPU...");
                if (hw != null) {
                    hw.shutdown();
                    hw = null;
                }
                backend = null;
                cfg = cfg.toBuilder().backend(EngineConfig.BackendMode.CPU).build();
            }
        }
        try (EngineModel model = LLMModelFactory.open(modelPath, cfg, backend)) {
            if (hw != null && model instanceof GGUFModel) {
                System.out.println("GPU load failed (reason logged above); running on the CPU engine.");
                hw.shutdown();
                hw = null;
                backend = null;
            }
            describeModel(modelPath, model, cfg, backend);
            if (opts.memoryReportOnly) {
                return;
            }
            if (opts.benchmark || opts.benchStreaming) {
                runBenchmark(model, opts);
                return;
            }
            dispatch(model, mode == Mode.AUTO ? autoMode(model.spec(), opts) : mode, opts, cfg);
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
        }
    }

    /**
     * {@code --mode}, or for AUTO the mode of the model ({@link #autoMode}, read from its metadata), so the pooled
     * modes get their ubatch before the model loads.
     */
    private static Mode resolveMode(String modelPath, LLMDemoOptions opts) {
        if (opts.mode != Mode.AUTO) {
            return opts.mode;
        }
        try {
            ModelSpec spec = LLMModelFactory.inspect(modelPath, opts.engine.memory());
            return autoMode(spec, opts);
        } catch (IOException | RuntimeException e) {
            System.err.println("Cannot read the model's architecture ahead of loading (" + e.getMessage()
                    + "); the mode is chosen after loading.");
            return Mode.AUTO;
        }
    }

    /**
     * The AUTO mode of a model, as the llama.cpp tools pick their output from the head rather than the architecture:
     * a decision head classifies; RANK pooling reranks {@code --document} against the prompt, or classifies the
     * prompt without a document; any other pooling type set by the GGUF embeds (llama.cpp embeddings of any model,
     * such as a qwen3 embedding model with LAST pooling); otherwise the architecture's support class decides.
     */
    static Mode autoMode(ModelSpec spec, LLMDemoOptions opts) {
        if (DecisionScorer.present(spec)) {
            return Mode.CLASSIFY;
        }
        PoolingKind pooling = spec.head().pooling();
        if (pooling == PoolingKind.RANK) {
            return opts.document != null ? Mode.RERANK : Mode.CLASSIFY;
        }
        if (pooling != PoolingKind.NONE) {
            return Mode.EMBED;
        }
        return modeOf(spec.supportClass());
    }

    private static Mode modeOf(SupportClass support) {
        switch (support) {
            case EMBED:
                return Mode.EMBED;
            case CLS:
                return Mode.CLASSIFY;
            case DIFF:
                return Mode.DIFFUSION;
            case AUDIO:
                return Mode.AUDIO;
            default:
                return Mode.GENERATE;
        }
    }

    private static void dispatch(EngineModel model, Mode mode, LLMDemoOptions opts, EngineConfig cfg)
            throws IOException {
        System.out.println("Mode: " + mode.name().toLowerCase(Locale.ROOT));
        System.out.println();
        switch (mode) {
            case EMBED:
                runEmbed(model, opts);
                break;
            case CLASSIFY:
                runClassify(model, opts);
                break;
            case RERANK:
                runRerank(model, opts);
                break;
            case DIFFUSION:
                runDiffusion(model, opts);
                break;
            case AUDIO:
                runAudio(model, opts, cfg);
                break;
            default:
                runGenerate(model, opts, cfg);
                break;
        }
    }

    private static void describeModel(String modelPath, EngineModel model, EngineConfig cfg, ComputeBackend backend) {
        model.printInfo();
        System.out.println("Support class: " + model.supportClass());
        if (!cfg.loras().isEmpty()) {
            System.out.println("LoRA adapters: " + cfg.loras());
        }
        printMemoryPlan(LLMModelFactory.memoryPlan(model));
        if (backend != null && model instanceof GGUFModelGPU) {
            System.out.printf(Locale.ROOT, "Device memory: %.1f MiB allocated of %.1f MiB%n",
                    backend.getAllocatedBytes() / MIB, backend.capabilities().getTotalVramBytes() / MIB);
        }
        System.out.println();

        boolean gpu = model instanceof GGUFModelGPU;
        logStats("=== LLM Demo Session (" + (gpu ? "GPU" : "CPU") + ") ===");
        logStats("Model: " + modelPath);
        if (gpu && backend != null) {
            logStats("Backend: " + backend.backendType() + " (" + backend.capabilities().getDeviceClass() + ")");
            logStats("Device: " + backend.capabilities().getDeviceName());
        }
        logStats("Support class: " + model.supportClass());
        logModelInfo(model);
    }

    /** The memory plan's byte breakdown by category. */
    private static void printMemoryPlan(MemoryPlan plan) {
        if (plan == null) {
            System.out.println("Memory: none (forward passes run without a cache)");
            return;
        }
        if (plan.totalBytes() == 0) {
            System.out.printf(Locale.ROOT, "Memory plan: %s, no caches (context %d, ubatch %d = longest input of one pass)%n",
                    plan.kind(), plan.nCtx(), plan.ubatch());
            return;
        }
        long k = 0;
        long v = 0;
        long cross = 0;
        long indexer = 0;
        long msa = 0;
        long compressedCache = 0;
        long compressedState = 0;
        Set<String> kvTypes = new LinkedHashSet<>();
        Set<String> indexerTypes = new LinkedHashSet<>();
        for (LayerMemoryPlan l : plan.layers()) {
            KvLayout kv = l.kv();
            if (l.ownsKv() && kv != null) {
                k += kv.kBytes();
                v += kv.vBytes();
                kvTypes.add(kv.dtypeK() + "/" + (kv.vAliasesK() ? "(K)" : kv.dtypeV()));
            }
            if (l.crossKv() != null) {
                cross += l.crossKv().totalBytes();
            }
            if (l.indexer() != null) {
                indexer += l.indexer().bytes();
                indexerTypes.add(String.valueOf(l.indexer().dtype()));
            }
            if (l.msa() != null) {
                msa += l.msa().bytes();
            }
            for (LayerMemoryPlan.CompressedKvPlan c : l.compressed()) {
                compressedCache += c.cacheBytes();
                compressedState += c.stateBytes();
            }
        }
        System.out.printf(Locale.ROOT, "Memory plan: %s, context %d, ubatch %d, %d layers, %d KV groups%n",
                plan.kind(), plan.nCtx(), plan.ubatch(), plan.layers().size(), plan.groups().size());
        printBytes("K cache", k, kvTypes.isEmpty() ? "" : "types " + kvTypes);
        printBytes("V cache", v, "");
        printBytes("cross-attention KV", cross, "");
        printBytes("recurrent state", plan.stateBytes(), "");
        printBytes("state snapshots", plan.snapshotBytes(), "speculative rollback copies");
        printBytes("sparse indexer", indexer, indexerTypes.isEmpty() ? ""
                : indexerTypes + "; rows follow neuralfree.kv.type: 136 B (q8_0), 256 B (f16), 512 B (f32)"
                + " per 128 values");
        printBytes("MSA cache", msa, "");
        printBytes("compressed cache", compressedCache, "");
        printBytes("compressed state", compressedState, "");
        printBytes("positions", plan.positionBytes(), "");
        System.out.printf(Locale.ROOT, "  %-20s %10.1f MiB%n", "total", plan.totalBytes() / MIB);
        logStats(String.format(Locale.ROOT, "Memory: %s, ctx %d, ubatch %d, %.1f MiB", plan.kind(), plan.nCtx(),
                plan.ubatch(), plan.totalBytes() / MIB));
    }

    private static void printBytes(String label, long bytes, String note) {
        if (bytes == 0) {
            return;
        }
        System.out.printf(Locale.ROOT, "  %-20s %10.1f MiB%s%n", label, bytes / MIB, note.isEmpty() ? "" : "  " + note);
    }

    private static void runBenchmark(EngineModel model, LLMDemoOptions opts) {
        if (!(model instanceof GGUFModelGPU)) {
            throw new UnsupportedOperationException("--benchmark and --bench-streaming profile the GPU engine;"
                    + " the model runs on the CPU");
        }
        GGUFModelGPU gpu = (GGUFModelGPU) model;
        GGUFTokenizer tokenizer = gpu.tokenizer();
        int tokenId = tokenizer.bosId() >= 0 ? tokenizer.bosId() : 1;
        if (opts.benchmark) {
            gpu.runBenchmark(tokenId);
        } else {
            gpu.benchmarkStreamingOverlap(tokenId);
        }
    }

    // ================================================================== generation

    /** One way of generating: the engine loop, a speculative session, or a model's own generate. */
    private interface Generator {
        GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                  String renderedPrompt) throws IOException;

        void reset();
    }

    private static void runGenerate(EngineModel model, LLMDemoOptions opts, EngineConfig cfg) throws IOException {
        SpeculativeSession session = cfg.speculative() ? LLMModelFactory.speculate(model, cfg) : null;
        try {
            Generator generator;
            if (session != null) {
                System.out.println("Speculative decoding: " + session.drafterKind() + " drafter, draft max "
                        + session.draftMax() + ", min " + session.draftMin() + ", p_min " + cfg.draftPMin());
                logStats("Speculative: " + session.drafterKind() + ", draft max " + session.draftMax());
                generator = new Generator() {
                    @Override
                    public GenerationResult generate(int[] prompt, GenerationRequest request,
                                                     ReasoningFilter.Sink sink, String rendered) {
                        return session.generate(prompt, request, sink, rendered);
                    }

                    @Override
                    public void reset() {
                        session.reset();
                    }
                };
            } else {
                generator = loopGenerator(model, LLMModelFactory.effectiveConfig(model, cfg));
            }
            boolean chatTemplate = !opts.raw && model.supportClass() != SupportClass.ENC_DEC;
            String prompt = opts.prompt != null ? opts.prompt : opts.nonInteractive ? readStdin() : null;
            if (prompt != null) {
                runSinglePrompt(model, generator, prompt, opts, chatTemplate);
            } else {
                runConsoleChat(model, generator, opts.systemPrompt, opts.generation, opts.debugPrompt,
                        opts.thinking, opts.showAllChannels);
            }
            if (session != null && session.lastStats() != null) {
                printSpeculativeStats(session.lastStats());
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    private static Generator loopGenerator(EngineModel model, EngineConfig effective) {
        GenerationLoop loop = GenerationLoop.forModel(model, effective);
        return new Generator() {
            @Override
            public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                             String rendered) {
                return loop.generate(prompt, request, sink, rendered);
            }

            @Override
            public void reset() {
                model.resetKVCache();
            }
        };
    }

    /** A model's own {@link LLMModel#generate(int[], GenerationRequest, com.eonmux.neuralfree.gguf.TokenCallback)}. */
    private static Generator modelGenerator(LLMModel model) {
        return new Generator() {
            @Override
            public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                             String rendered) throws IOException {
                return model.generate(prompt, request,
                        (id, text) -> sink.accept(ReasoningFilter.Channel.CONTENT, id, text));
            }

            @Override
            public void reset() {
                model.resetKVCache();
            }
        };
    }

    private static void printSpeculativeStats(SpeculativeDecoder.Stats s) {
        String line = String.format(Locale.ROOT, "Speculative: %d rounds, %d drafted, %d accepted (%.1f%%)",
                s.rounds(), s.drafted(), s.accepted(), 100.0 * s.acceptanceRate());
        System.out.println("[" + line + "]");
        logStats(line);
    }

    private static String readStdin() throws IOException {
        StringBuilder sb = new StringBuilder();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            if (sb.length() > 0) {
                sb.append('\n');
            }
            sb.append(line);
        }
        String prompt = sb.toString().trim();
        if (prompt.isEmpty()) {
            throw new IllegalArgumentException("--non-interactive without --prompt reads the prompt from standard"
                    + " input, which was empty");
        }
        return prompt;
    }

    /**
     * Prints routed output: content as is, and with {@code showAll} the reasoning and tool-call channels under a
     * header at each channel change. Collects the content for the chat history.
     */
    private static final class ChannelPrinter implements ReasoningFilter.Sink {
        private final boolean showAll;
        private final boolean trace;
        private final StringBuilder content = new StringBuilder();
        private ReasoningFilter.Channel current = ReasoningFilter.Channel.CONTENT;
        private int pieces;

        ChannelPrinter(boolean showAll, boolean trace) {
            this.showAll = showAll;
            this.trace = trace;
        }

        @Override
        public void accept(ReasoningFilter.Channel channel, int tokenId, String text) {
            pieces++;
            if (trace) {
                System.err.println("[GEN] piece=" + pieces + " channel=" + channel + " id=" + tokenId + " text='"
                        + text.replace("\n", "\\n") + "'");
            }
            if (channel == ReasoningFilter.Channel.CONTENT) {
                content.append(text);
            } else if (!showAll) {
                return;
            }
            if (channel != current) {
                System.out.print("\n[" + channel.name().toLowerCase(Locale.ROOT).replace('_', ' ') + "]\n");
                current = channel;
            }
            System.out.print(text);
            System.out.flush();
        }

        String content() {
            return content.toString();
        }
    }

    private static void runSinglePrompt(EngineModel model, Generator generator, String prompt, LLMDemoOptions opts,
                                        boolean chatTemplate) throws IOException {
        System.out.println("Prompt: " + prompt);
        System.out.print("Generating: ");
        System.out.flush();

        GGUFTokenizer tokenizer = model.tokenizer();
        String fullPrompt = chatTemplate ? tokenizer.formatChat(opts.systemPrompt, prompt, opts.thinking) : prompt;
        int[] promptTokens = tokenizer.encode(fullPrompt);
        if (opts.debugPrompt) {
            printDebugPrompt(fullPrompt, promptTokens, tokenizer);
        }
        ChannelPrinter printer = new ChannelPrinter(opts.showAllChannels, opts.debugPrompt);
        GenerationResult result = generator.generate(promptTokens, opts.generation, printer, fullPrompt);
        System.out.println();
        System.out.println();
        String statLine = statLine(promptTokens.length, result);
        System.out.println("[" + statLine + "]");
        logStats(statLine);
        logStats("  Prompt: " + prompt);
        logStats("  Response: " + printer.content());
        logStats("");
    }

    /** Prompt size, generated tokens, stop reason and the prefill / decode rates of one generation. */
    private static String statLine(int promptTokens, GenerationResult result) {
        GenerationResult.Timings t = result.timings();
        double totalSec = t.totalNanos() / 1e9;
        double totalSpeed = totalSec > 0 ? result.tokenCount() / totalSec : 0;
        if (!t.split()) {
            return String.format(Locale.ROOT, "Prompt: %d tokens | Generated: %d tokens (%s) | Total: %.2fs (%.2f tok/s)",
                    promptTokens, result.tokenCount(), result.stopReason(), totalSec, totalSpeed);
        }
        double prefillSec = t.promptNanos() / 1e9;
        double decodeSec = t.generationNanos() / 1e9;
        return String.format(Locale.ROOT,
                "Prompt: %d tokens (%d evaluated) | Generated: %d tokens (%s) | Prefill: %.2fs (%.2f tok/s)"
                        + " | Decode: %.2fs (%.2f tok/s) | Total: %.2fs (%.2f tok/s)",
                promptTokens, t.promptTokens(), result.tokenCount(), result.stopReason(), prefillSec,
                rate(t.promptTokensPerSecond()), decodeSec, rate(t.generationTokensPerSecond()), totalSec, totalSpeed);
    }

    private static double rate(double r) {
        return Double.isNaN(r) ? 0 : r;
    }

    private static void runConsoleChat(LLMModel model, Generator generator, String systemPrompt,
                                       GenerationRequest request, boolean debugPrompt,
                                       boolean enableThinking, boolean showAllChannels) throws IOException {
        Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8.name());
        GGUFTokenizer tokenizer = model.tokenizer();
        List<Map<String, String>> conversationHistory = new ArrayList<>();
        SamplerConfig sampler = request.sampler();

        System.out.println("Chat ready. Type your message (or 'quit' to exit, 'clear' to reset):");
        System.out.println(String.format(Locale.ROOT,
                "Settings: temperature=%.2f, top_p=%.2f, top_k=%d, min_p=%.3f, typical_p=%.2f, max_tokens=%d,"
                        + " repeat_penalty=%.2f, presence_penalty=%.2f, frequency_penalty=%.2f, repeat_last_n=%d,"
                        + " seed=%d%s%s",
                sampler.temperature(), sampler.topP(), sampler.topK(), sampler.minP(), sampler.typicalP(),
                request.maxNewTokens(), sampler.repeatPenalty(), sampler.presencePenalty(),
                sampler.frequencyPenalty(), sampler.penaltyLastN(), sampler.seed(),
                request.ignoreEos() ? ", ignore_eos" : "",
                request.stopStrings().isEmpty() ? "" : ", stop=" + request.stopStrings()));
        System.out.println();

        while (true) {
            System.out.print("You: ");
            System.out.flush();
            if (!scanner.hasNextLine()) {
                break;
            }
            String input = scanner.nextLine().trim();
            if (input.equalsIgnoreCase("quit") || input.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye!");
                break;
            }
            if (input.equalsIgnoreCase("clear") || input.equalsIgnoreCase("reset")) {
                generator.reset();
                conversationHistory.clear();
                System.out.println("Conversation reset.\n");
                continue;
            }
            if (input.isEmpty()) {
                continue;
            }

            Map<String, String> sysMsg = message("system", systemPrompt);
            Map<String, String> userMsg = message("user", input);
            String fullPrompt = tokenizer.formatChat(conversation(sysMsg, conversationHistory, userMsg), enableThinking);
            int[] promptTokens = tokenizer.encode(fullPrompt);
            if (promptTokens.length >= model.kvCacheContext()) {
                System.out.println("[Context full: " + promptTokens.length + " tokens >= "
                        + model.kvCacheContext() + " KV cache capacity. Trimming oldest turns.]");
                while (promptTokens.length >= model.kvCacheContext() && conversationHistory.size() >= 2) {
                    conversationHistory.remove(0);
                    conversationHistory.remove(0);
                    fullPrompt = tokenizer.formatChat(conversation(sysMsg, conversationHistory, userMsg),
                            enableThinking);
                    promptTokens = tokenizer.encode(fullPrompt);
                }
                if (promptTokens.length >= model.kvCacheContext()) {
                    System.out.println("[Still too long after trimming. Starting fresh turn.]");
                    conversationHistory.clear();
                    fullPrompt = tokenizer.formatChat(systemPrompt, input, enableThinking);
                    promptTokens = tokenizer.encode(fullPrompt);
                }
            }
            if (debugPrompt) {
                printDebugPrompt(fullPrompt, promptTokens, tokenizer);
            }

            System.out.print("Assistant: ");
            System.out.flush();
            ChannelPrinter printer = new ChannelPrinter(showAllChannels, debugPrompt);
            GenerationResult result = generator.generate(promptTokens, request, printer, fullPrompt);
            System.out.println();
            System.out.println();

            conversationHistory.add(message("user", input));
            conversationHistory.add(message("assistant", printer.content()));

            String statLine = statLine(promptTokens.length, result);
            System.out.println("[" + statLine + "]");
            System.out.println();
            logStats(statLine);
            String truncInput = input.length() > 50 ? input.substring(0, 50) + "..." : input;
            String resp = printer.content();
            String truncResp = resp.length() > 100 ? resp.substring(0, 100) + "..." : resp;
            logStats("  Input: " + truncInput);
            logStats("  Output: " + truncResp);
            logStats("");
        }
    }

    private static Map<String, String> message(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /**
     * The messages of one chat turn; an empty system prompt adds no system message, as llama.cpp's CLI adds one only
     * for a non-empty {@code --system-prompt} (tools/cli/cli-context.cpp:279).
     */
    private static List<Map<String, String>> conversation(Map<String, String> system,
                                                          List<Map<String, String>> history,
                                                          Map<String, String> user) {
        List<Map<String, String>> messages = new ArrayList<>(history.size() + 2);
        String systemContent = system.get("content");
        if (systemContent != null && !systemContent.isEmpty()) {
            messages.add(system);
        }
        messages.addAll(history);
        messages.add(user);
        return messages;
    }

    // ================================================================== embeddings, classifiers, reranking

    private static String promptOrSample(LLMDemoOptions opts, String what) {
        if (opts.prompt != null && !opts.prompt.isEmpty()) {
            return opts.prompt;
        }
        System.out.println("[" + what + "] No --prompt provided; using sample: " + SAMPLE_EMBED_TEXT);
        return SAMPLE_EMBED_TEXT;
    }

    /** Token types of the input: {@code --token-types}, which must give one per token, or null. */
    private static int[] tokenTypes(LLMDemoOptions opts, int nTokens) {
        if (opts.tokenTypes == null) {
            return null;
        }
        if (opts.tokenTypes.length != nTokens) {
            throw new IllegalArgumentException("--token-types gives " + opts.tokenTypes.length
                    + " segment ids for " + nTokens + " tokens");
        }
        return opts.tokenTypes.clone();
    }

    private static void runEmbed(EngineModel model, LLMDemoOptions opts) throws IOException {
        String text = promptOrSample(opts, "embed");
        int[] tokens = model.tokenizer().encode(text);
        int[] types = tokenTypes(opts, tokens.length);
        System.out.println("Prompt: " + text);
        System.out.println("Tokens: " + tokens.length);
        System.out.println("Pooling: " + model.poolingType());

        long t0 = System.nanoTime();
        if (model.poolingType() == LLMModel.PoolingType.NONE && types == null) {
            float[][] rows = model.encodeTokens(tokens);
            long dt = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println();
            System.out.println("Per-token embeddings: " + rows.length + " x " + (rows.length > 0 ? rows[0].length : 0)
                    + "  time=" + dt + "ms");
            for (int r = 0; r < Math.min(rows.length, 4); r++) {
                System.out.println("  token " + r + ": norm=" + String.format(Locale.ROOT, "%.4f", norm(rows[r]))
                        + "  preview = " + preview(rows[r], opts.embedPreview));
            }
            logStats("Per-token embeddings: " + rows.length + " rows, encode time: " + dt + "ms");
            writeRows(opts.audioOut, rows);
            return;
        }
        float[] pooled = model.encode(tokens, types);
        long dt = (System.nanoTime() - t0) / 1_000_000L;
        double norm = norm(pooled);
        System.out.println();
        System.out.println("Embedding: dim=" + pooled.length + "  norm=" + String.format(Locale.ROOT, "%.4f", norm)
                + "  time=" + dt + "ms");
        System.out.println("  preview = " + preview(pooled, opts.embedPreview));
        logStats("Embedding dim: " + pooled.length);
        logStats("Embedding norm: " + String.format(Locale.ROOT, "%.4f", norm));
        logStats("Encode time: " + dt + "ms");
    }

    private static void runClassify(EngineModel model, LLMDemoOptions opts) throws IOException {
        String text = promptOrSample(opts, "classify");
        int[] tokens = model.tokenizer().encode(text);
        System.out.println("Prompt: " + text);
        System.out.println("Tokens: " + tokens.length);
        long t0 = System.nanoTime();
        float[] scores;
        if (model instanceof GGUFModelGPU) {
            scores = ((GGUFModelGPU) model).classify(tokens, opts.decisionOrder);
        } else if (model instanceof GGUFModel) {
            scores = ((GGUFModel) model).classify(tokens, opts.decisionOrder);
        } else if (opts.decisionOrder == null) {
            scores = model.classify(tokens);
        } else {
            throw new UnsupportedOperationException("--decision-order needs a model opened by LLMModelFactory");
        }
        long dt = (System.nanoTime() - t0) / 1_000_000L;
        String[] labels = model.classLabels();
        System.out.println();
        System.out.println("Scores (" + scores.length + ", time=" + dt + "ms):");
        for (int s = 0; s < scores.length; s++) {
            String label = s < labels.length && labels[s] != null && !labels[s].isEmpty() ? labels[s] : "#" + s;
            System.out.printf(Locale.ROOT, "  %-24s %+.6f%n", label, scores[s]);
        }
        logStats("Classification: " + scores.length + " scores, time " + dt + "ms");
    }

    /**
     * Relevance of {@code --document} for the prompt, scored on the model's rerank input
     * ({@code ClassifierHead.rerankInput}): its {@code tokenizer.chat_template.rerank} prompt filled with the query
     * and document, or the two packed with the vocabulary's BOS / EOS / SEP.
     */
    private static void runRerank(EngineModel model, LLMDemoOptions opts) throws IOException {
        if (opts.prompt == null || opts.document == null) {
            throw new IllegalArgumentException("--mode rerank needs the query as --prompt and the document as"
                    + " --document");
        }
        int[] input = ClassifierHead.rerankInput(model.tokenizer(), opts.prompt, opts.document);
        boolean prompted = model.tokenizer().rerankTemplate() != null;
        long t0 = System.nanoTime();
        float score = model.rerank(opts.prompt, opts.document);
        long dt = (System.nanoTime() - t0) / 1_000_000L;
        System.out.println("Query: " + opts.prompt);
        System.out.println("Document: " + opts.document);
        System.out.println("Input: " + input.length + " tokens"
                + (prompted ? " (" + GGUFTokenizer.KEY_RERANK_TEMPLATE + ")" : " (query and document packed)"));
        System.out.printf(Locale.ROOT, "Relevance: %+.6f  time=%dms%n", score, dt);
        logStats(String.format(Locale.ROOT, "Rerank score: %+.6f, time %dms", score, dt));
    }

    private static double norm(float[] v) {
        double sq = 0;
        for (float x : v) {
            sq += (double) x * x;
        }
        return Math.sqrt(sq);
    }

    private static String preview(float[] v, int n) {
        int shown = Math.min(n, v.length);
        StringBuilder sb = new StringBuilder("[ ");
        for (int i = 0; i < shown; i++) {
            sb.append(String.format(Locale.ROOT, "%+.4f", v[i]));
            if (i + 1 < shown) {
                sb.append(", ");
            }
        }
        if (shown < v.length) {
            sb.append(shown > 0 ? ", ..." : "...");
        }
        return sb.append(" ]").toString();
    }

    // ================================================================== diffusion and audio

    /** Masked-diffusion generation (llama.cpp diffusion-cli: the chat-formatted prompt, canvas of one ubatch). */
    private static void runDiffusion(EngineModel model, LLMDemoOptions opts) throws IOException {
        String prompt = opts.prompt != null ? opts.prompt : opts.nonInteractive ? readStdin() : null;
        if (prompt == null) {
            throw new IllegalArgumentException("--mode diffusion needs --prompt (or --non-interactive with standard"
                    + " input)");
        }
        GGUFTokenizer tokenizer = model.tokenizer();
        String fullPrompt = opts.raw ? prompt : tokenizer.formatChat(opts.systemPrompt, prompt, opts.thinking);
        int[] tokens = tokenizer.encode(fullPrompt);
        if (opts.debugPrompt) {
            printDebugPrompt(fullPrompt, tokens, tokenizer);
        }
        System.out.println("Prompt: " + prompt);
        System.out.println("Diffusion: " + opts.diffusion);
        System.out.print("Generating: ");
        System.out.flush();
        StringBuilder text = new StringBuilder();
        long t0 = System.nanoTime();
        int[] canvas = model.generateDiffusion(tokens, opts.diffusion, (id, piece) -> {
            text.append(piece);
            System.out.print(piece);
            System.out.flush();
        });
        double sec = (System.nanoTime() - t0) / 1e9;
        System.out.println();
        System.out.println();
        String line = String.format(Locale.ROOT, "Prompt: %d tokens | Canvas: %d tokens | Steps: %d | Total: %.2fs",
                tokens.length, canvas.length, opts.diffusion.steps(), sec);
        System.out.println("[" + line + "]");
        logStats(line);
        logStats("  Prompt: " + prompt);
        logStats("  Response: " + text);
    }

    /**
     * Audio models: WavTokenizer decodes codes to spectral features (waveform synthesis by iSTFT is not part of the
     * engine), pockettts returns its backbone hidden states, and token-generating speech models (qwen3tts) generate
     * codec tokens from the raw prompt.
     */
    private static void runAudio(EngineModel model, LLMDemoOptions opts, EngineConfig cfg) throws IOException {
        Arch arch = model.spec().arch();
        if (arch == Arch.WAVTOKENIZER_DEC) {
            int[] codes = opts.audioCodes != null ? opts.audioCodes.clone() : parseCodes(opts.prompt);
            long t0 = System.nanoTime();
            float[][] features = model.decodeAudioCodes(codes);
            long dt = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("Codes: " + codes.length);
            System.out.println("Features: " + features.length + " frames x " + (features.length > 0 ? features[0].length : 0)
                    + "  time=" + dt + "ms (spectral features; no iSTFT waveform synthesis)");
            if (features.length > 0) {
                System.out.println("  frame 0 = " + preview(features[0], opts.embedPreview));
            }
            logStats("WavTokenizer: " + codes.length + " codes -> " + features.length + " frames, " + dt + "ms");
            writeRows(opts.audioOut, features);
            return;
        }
        if (arch == Arch.POCKETTTS) {
            String text = promptOrSample(opts, "pockettts");
            int[] tokens = model.tokenizer().encode(text);
            try (PocketTtsSession session = PocketTtsSession.open(model)) {
                session.reset();
                long t0 = System.nanoTime();
                float[][] rows = session.prefillAll(tokens, 0);
                long dt = (System.nanoTime() - t0) / 1_000_000L;
                System.out.println("Tokens: " + tokens.length);
                System.out.println("Backbone hidden states: " + rows.length + " x " + session.width() + "  time=" + dt
                        + "ms (input of the pockettts flow network)");
                if (rows.length > 0) {
                    System.out.println("  last = " + preview(rows[rows.length - 1], opts.embedPreview));
                }
                logStats("pockettts: " + rows.length + " hidden rows, " + dt + "ms");
                writeRows(opts.audioOut, rows);
            }
            return;
        }
        String prompt = opts.prompt != null ? opts.prompt : opts.nonInteractive ? readStdin() : null;
        if (prompt == null) {
            throw new IllegalArgumentException("--mode audio on " + model.architecture() + " needs --prompt");
        }
        GenerationLoop loop = GenerationLoop.forModel(model, LLMModelFactory.effectiveConfig(model, cfg));
        int[] tokens = model.tokenizer().encode(prompt);
        GenerationResult result = loop.generate(tokens, opts.generation, (id, text) -> { });
        int[] ids = result.tokens();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(ids.length, 32); i++) {
            sb.append(i == 0 ? "" : ", ").append(ids[i]);
        }
        System.out.println("Generated " + ids.length + " codec tokens (" + result.stopReason() + "): [" + sb
                + (ids.length > 32 ? ", ..." : "") + "]");
        String line = statLine(tokens.length, result);
        System.out.println("[" + line + "]");
        logStats(line);
    }

    private static int[] parseCodes(String text) {
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("WavTokenizer decoding needs codes: --audio-codes <csv> or a --prompt"
                    + " of integers");
        }
        String[] parts = text.trim().split("[,\\s]+");
        int[] codes = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                codes[i] = Integer.parseInt(parts[i]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("audio code '" + parts[i] + "' is not an integer", e);
            }
        }
        return codes;
    }

    /** Writes {@code rows} row-major as little-endian float32 to {@code path}; nothing when the path is null. */
    private static void writeRows(String path, float[][] rows) throws IOException {
        if (path == null) {
            return;
        }
        int width = rows.length > 0 ? rows[0].length : 0;
        ByteBuffer buf = ByteBuffer.allocate(Math.max(1, width) * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            for (float[] row : rows) {
                if (row.length != width) {
                    throw new IllegalStateException("rows have different widths: " + row.length + " vs " + width);
                }
                buf.clear();
                for (float v : row) {
                    buf.putFloat(v);
                }
                out.write(buf.array(), 0, width * Float.BYTES);
            }
        }
        System.out.println("Wrote " + rows.length + " x " + width + " float32 (little-endian) to " + path);
    }

    // ================================================================== diagnostics and stats

    private static void logModelInfo(LLMModel model) {
        logStats("Architecture: " + model.architecture());
        logStats("Context Length: " + model.contextLength() + " (KV cache: " + model.kvCacheContext() + ")");
        logStats("Embedding: " + model.embeddingLength());
        logStats("Layers: " + model.layerCount());
        logStats("Heads: " + model.headCount() + " / KV: " + model.headCountKv());
        logStats("Vocab Size: " + model.vocabSize());
        logStats("");
    }

    private static void printDebugPrompt(String fullPrompt, int[] tokens, GGUFTokenizer tokenizer) {
        System.out.println();
        System.out.println("=== DEBUG: Formatted Prompt ===");
        System.out.println(fullPrompt);
        System.out.println("=== END Formatted Prompt ===");
        System.out.println("Prompt length: " + fullPrompt.length() + " chars, " + tokens.length + " tokens");
        int show = Math.min(40, tokens.length);
        System.out.println("First " + show + " tokens:");
        for (int i = 0; i < show; i++) {
            System.out.println("  [" + i + "] id=" + tokens[i] + " raw='" + tokenizer.getToken(tokens[i]) + "'");
        }
        if (tokens.length > 40) {
            System.out.println("  ...");
        }
        int lastStart = Math.max(show, tokens.length - 20);
        if (lastStart > show) {
            System.out.println("Last " + (tokens.length - lastStart) + " tokens:");
            for (int i = lastStart; i < tokens.length; i++) {
                System.out.println("  [" + i + "] id=" + tokens[i] + " raw='" + tokenizer.getToken(tokens[i]) + "'");
            }
        }
        int startId = tokenizer.getTokenId("<|start|>");
        int messageId = tokenizer.getTokenId("<|message|>");
        int endId = tokenizer.getTokenId("<|end|>");
        int channelId = tokenizer.getTokenId("<|channel|>");
        if (startId >= 0 || messageId >= 0 || endId >= 0 || channelId >= 0) {
            System.out.println("Control token IDs: <|start|>=" + startId + " <|message|>=" + messageId
                    + " <|end|>=" + endId + " <|channel|>=" + channelId);
            int ctrlCount = 0;
            for (int t : tokens) {
                if (t == startId || t == messageId || t == endId || t == channelId) {
                    ctrlCount++;
                }
            }
            System.out.println("Control tokens in encoded prompt: " + ctrlCount
                    + " (expect >=8 for system+developer+user+assistant)");
        }
        System.out.println("===");
        System.out.println();
    }

    private static void logStats(String line) {
        statsLog.add(line);
    }

    /** Appends the statistics of this session to {@code statsFile}. */
    private static void writeStatsToFile(String statsFile) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(statsFile, true))) {
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
            writer.write("\n========================================\n");
            writer.write("Session: " + timestamp + "\n");
            writer.write("========================================\n");
            for (String line : statsLog) {
                writer.write(line + "\n");
            }
            System.out.println("Stats written to " + statsFile);
        } catch (IOException e) {
            System.err.println("Failed to write stats: " + e.getMessage());
        }
    }

    /**
     * Console chat on an open model with the demo's default settings (the {@code -Dneuralfree.*} properties, else
     * the demo defaults of {@link LLMDemoOptions}): engine models run the generation loop with every channel shown,
     * other models their own {@code generate}.
     *
     * @throws IllegalArgumentException when a {@code neuralfree.*} property is invalid
     */
    public static void run(LLMModel model) throws IOException {
        LLMDemoOptions opts;
        try {
            opts = LLMDemoOptions.parse(new String[0]);
        } catch (UsageException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
        Generator generator;
        if (model instanceof EngineModel) {
            EngineModel em = (EngineModel) model;
            generator = loopGenerator(em, LLMModelFactory.effectiveConfig(em, opts.engine));
        } else {
            generator = modelGenerator(model);
        }
        runConsoleChat(model, generator, opts.systemPrompt, opts.generation, false, false, true);
    }
}
