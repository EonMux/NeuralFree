package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryRequirement;
import com.eonmux.neuralfree.cli.CacheListing;
import com.eonmux.neuralfree.cli.DeviceListing;
import com.eonmux.neuralfree.cli.Launcher;
import com.eonmux.neuralfree.demo.LLMDemoOptions.Mode;
import com.eonmux.neuralfree.demo.LLMDemoOptions.UsageException;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.hub.HubCache;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.AudioDecoder;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.audio.WavWriter;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClip;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.PocketTtsClip;
import com.eonmux.neuralfree.gguf.speech.PocketTtsSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsClip;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.LookaheadDecoder;
import com.eonmux.neuralfree.gguf.engine.PocketTtsSession;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.SpeculativeDecoder;
import com.eonmux.neuralfree.gguf.engine.SpeculativeSession;
import com.eonmux.neuralfree.gguf.gpu.pipeline.GpuDevices;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.mm.MmprojSources;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.BufferedOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
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
        if (opts.version) {
            System.out.println(Launcher.versionLine());
            return;
        }
        if (opts.listDevices) {
            DeviceListing.print(System.out);
            return;
        }
        if (opts.cacheList) {
            try {
                CacheListing.print(System.out, new HubCache(HubCache.defaultRoot()));
            } catch (IOException | IllegalStateException | IllegalArgumentException e) {
                System.err.println("Error: " + e.getMessage());
                System.exit(1);
            }
            return;
        }
        if (opts.serverBase != null) {
            // Client mode: the server holds the model; nothing loads here.
            try {
                int code = RemoteChat.run(opts);
                if (code != 0) {
                    System.exit(code);
                }
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                System.exit(1);
            }
            return;
        }
        String modelPath = resolveModelPath(opts);
        if (modelPath == null) {
            System.err.println("No model specified.");
            printModelOptions();
            System.exit(2);
            return;
        }
        try {
            ModelSources.requireOpenable(modelPath);
        } catch (FileNotFoundException e) {
            System.err.println("Model not found: " + modelPath);
            printModelOptions();
            System.exit(2);
            return;
        }
        if (opts.exportGguf != null) {
            try {
                exportGguf(modelPath, opts);
                return;
            } catch (Exception e) {
                System.err.println("Error: " + e.getMessage());
                e.printStackTrace();
                System.exit(1);
                return;
            }
        }
        if (opts.noMmap) {
            // turns the configured mode into its non-mapped form, keeping a --mlock pin (mmap+mlock -> mlock)
            System.setProperty(TensorReadMode.PROP_READ_MODE,
                    TensorReadMode.current().withMmap(false).flagName());
        }
        if (opts.noJinja) {
            System.setProperty(ChatTemplateSource.JINJA_PROPERTY, "false");
        }
        if (opts.engine.deviceFilter() != null) {
            System.setProperty(EngineConfig.PROP_DEVICE, opts.engine.deviceFilter());
        }

        // the projector resolves (and downloads) before the model opens, whose media turns need paged memory
        java.nio.file.Path projector = resolveProjector(opts, Paths.get(modelPath));
        if (projector != null) {
            System.out.println("mmproj: " + projector);
            if (System.getProperty(MemoryConfig.PROP_KV_PAGED) == null) {
                System.setProperty(MemoryConfig.PROP_KV_PAGED, "true");
                System.out.println("Media turns run the batch engine; enabling paged memory ("
                        + MemoryConfig.PROP_KV_PAGED + "=true)");
            }
        }

        System.out.println("=== LLM Demo - NeuralFree GGUF Inference ===");
        System.out.println("Model: " + modelPath);
        System.out.println("Acceleration: " + (opts.engine.backend().usesGpu()
                ? "GPU (" + opts.engine.backend().name().toLowerCase(Locale.ROOT) + ")" : "CPU"));
        System.out.println();

        try {
            runSession(modelPath, opts, projector);
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
     * The model path: the {@code -hf} Hugging Face Hub download (the file the Hub cache
     * holds for the repo expression), else the command line, else {@code -Dneuralfree.model}, else
     * {@code NEURALFREE_MODEL}; null when none of them gives one.
     */
    private static String resolveModelPath(LLMDemoOptions opts) {
        if (opts.hfRepo != null) {
            try {
                return com.eonmux.neuralfree.demo.tools.DownloadCommand.downloadModel(
                        opts.hfRepo, opts.hfFile);
            } catch (java.io.IOException | java.lang.InterruptedException e) {
                System.err.println("Error: " + e.getMessage());
                System.exit(1);
                return null;
            } catch (com.eonmux.neuralfree.demo.tools.DownloadCommand.UsageException e) {
                System.err.println("Error: " + e.getMessage());
                System.exit(2);
                return null;
            }
        }
        if (opts.modelPath != null) {
            return opts.modelPath;
        }
        String path = System.getProperty("neuralfree.model");
        if (path == null || path.isEmpty()) {
            path = System.getenv("NEURALFREE_MODEL");
        }
        return path != null && !path.isEmpty() ? path : null;
    }

    /** How a missing model is reported: a GGUF file, a Hugging Face model directory or a safetensors file. */
    private static void printModelOptions() {
        System.err.println("Specify a model (GGUF file, Hugging Face model directory, or .safetensors file) with one of:");
        System.err.println(
                "  --model <path>            (e.g. ./gradlew runLLMDemo -Pargs=\"--model /path/to/model.gguf\")");
        System.err.println("  export NEURALFREE_MODEL=<path>");
        System.err.println("  -Dneuralfree.model=<path>");
    }

    /**
     * The projector of a local multimodal run, llama.cpp's sources in order (common/arg.cpp): the explicit
     * {@code --mmproj}, the {@code --mmproj-url} download into the hub cache, or the {@code --mmproj-auto}
     * sibling — for a {@code -hf} model the repository's mmproj sidecar llama.cpp downloads with the model,
     * else the {@code *mmproj*.gguf} beside the model file. Null when only the automatic search ran and found
     * none, which is a text-only run; media files with no projector anywhere end the run naming the flags. The
     * audio mode keeps {@code --mmproj} for its speech generator and resolves nothing.
     */
    private static java.nio.file.Path resolveProjector(LLMDemoOptions opts, java.nio.file.Path modelPath) {
        if (opts.mode == Mode.AUDIO) {
            // the audio mode keeps --mmproj for its speech generator; the projector sources configure a
            // text model's media run
            return null;
        }
        java.nio.file.Path resolved = null;
        try {
            resolved = MmprojSources.resolve("--mmproj", opts.mmproj, opts.mmprojUrl, !opts.noMmproj, modelPath,
                    HubCache.defaultRoot());
            if (resolved == null && !opts.noMmproj && opts.mmproj == null && opts.mmprojUrl == null
                    && opts.hfRepo != null) {
                resolved = com.eonmux.neuralfree.demo.tools.DownloadCommand.downloadMmproj(
                        opts.hfRepo, opts.hfFile);
            }
        } catch (java.io.IOException | InterruptedException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
            return null;
        } catch (com.eonmux.neuralfree.demo.tools.DownloadCommand.UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(2);
            return null;
        }
        if (resolved == null && !opts.media.isEmpty()) {
            System.err.println("Error: --image/--audio/--video need the projector of a multimodal model; give"
                    + " --mmproj or --mmproj-url, or let --mmproj-auto (the default) find the mmproj GGUF beside"
                    + " the model file or in the -hf repository (--no-mmproj turns the projector off)");
            System.exit(1);
            return null;
        }
        return resolved;
    }

    // ================================================================== export

    /**
     * Outputs the model's GGUF view to {@code --export-gguf} ({@link GgufWriter}, with the {@code --export-mode} and
     * {@code --export-overwrite} options and the Hugging Face flags' {@link ModelSourceOptions}); main exits after the
     * write, without generating.
     */
    private static void exportGguf(String modelPath, LLMDemoOptions opts) throws IOException {
        System.out.println("=== LLM Demo - GGUF export ===");
        Path out = Paths.get(opts.exportGguf);
        GgufWriter.Options options = new GgufWriter.Options(opts.exportMode, opts.exportOverwrite,
                GGUFFileParser.DEFAULT_ALIGNMENT);
        try (ModelSource src = ModelSources.open(modelPath, opts.engine.sourceOptions())) {
            GgufWriter.write(src, out, options);
            System.out.println("Exported '" + modelPath + "' to '" + out + "'.");
        }
    }

    // ================================================================== session

    private static void runSession(String modelPath, LLMDemoOptions opts, java.nio.file.Path projector)
            throws IOException {
        Mode mode = resolveMode(modelPath, opts);
        EngineConfig cfg = opts.engineConfig(mode);
        HWAcceleration hw = null;
        ComputeBackend backend = null;
        // a layer split opens a backend per device itself and reports its stages; it shares no backend
        if (cfg.backend().usesGpu() && !cfg.split()) {
            ACCELERATION_MODE accel = cfg.backend().toAccelerationMode();
            System.out.println("Initializing GPU backend...");
            try {
                Config.setAccelerationMode(accel);
                hw = HWAcceleration.forMode(accel);
                backend = cfg.picksMainGpu() ? GpuDevices.openMainGpu(hw, cfg).backend()
                        : hw.createBackend(accel, KernelLibraryRequirement.of(KernelLibraries.LLM));
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
                printFittedArguments(model);
                return;
            }
            if (opts.benchmark || opts.benchStreaming) {
                runBenchmark(model, opts);
                return;
            }
            dispatch(model, mode == Mode.AUTO ? autoMode(model.spec(), opts) : mode, opts, cfg, projector);
        } finally {
            if (hw != null) {
                hw.shutdown();
            }
        }
    }

    /**
     * {@code --mode}, or for AUTO the mode of the model ({@link #autoMode}, read from its metadata, with the
     * {@code --pooling} override applied), so the pooled modes get their ubatch before the model loads.
     */
    private static Mode resolveMode(String modelPath, LLMDemoOptions opts) {
        if (opts.mode != Mode.AUTO) {
            return opts.mode;
        }
        try {
            ModelSpec spec = LLMModelFactory.inspect(modelPath, opts.engine);
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
     * prompt without a document; any other pooling type set by the GGUF or {@code --pooling} embeds (llama.cpp
     * embeddings of any model, such as a qwen3 embedding model with LAST pooling); otherwise the architecture's
     * support class decides.
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

    private static void dispatch(EngineModel model, Mode mode, LLMDemoOptions opts, EngineConfig cfg,
                                 java.nio.file.Path projector) throws IOException {
        String modeName = mode.name().toLowerCase(Locale.ROOT);
        if (mode != Mode.GENERATE && opts.promptCache != null) {
            // an AUTO mode that resolves away from generate reaches here; the explicit ones parse-refused
            throw new IllegalArgumentException("--prompt-cache runs the generate mode; the model resolves to "
                    + modeName + ", which holds no sequence to cache");
        }
        if (mode != Mode.GENERATE) {
            // only the text generation mode runs the request's sampler chain
            opts.requireNoDecodingConstraints("--mode " + modeName);
        }
        if (mode != Mode.GENERATE && opts.multimodalInputGiven()) {
            // an AUTO mode that resolves away from generate reaches here; the explicit ones parse-refused
            throw new IllegalArgumentException("--mode " + modeName + " reads text only; --mmproj, --mmproj-url"
                    + " and --image/--audio/--video belong to the generate mode");
        }
        System.out.println("Mode: " + modeName);
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
                runGenerate(model, opts, cfg, projector);
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
        } else if (!gpu && model.gpuFallbackReason() != null) {
            logStats("GPU fallback: " + model.gpuFallbackReason());
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

    /**
     * The reusable arguments of the load that just ran ({@code --vram-budget}, llama.cpp's
     * {@code llama_params_fit} output): a GPU load's fitted context and placement ({@link
     * GGUFModelGPU#fittedArguments}), or, on the CPU engine, a context fitted below the training one. A load
     * that needed no fitting prints nothing.
     */
    private static void printFittedArguments(EngineModel model) {
        String args = null;
        if (model instanceof GGUFModelGPU) {
            args = ((GGUFModelGPU) model).fittedArguments();
        } else if (model instanceof GGUFModel) {
            int train = model.spec().hparams().nCtxTrain();
            int nCtx = ((GGUFModel) model).forwardBackend().memory().plan().nCtx();
            if (train > 0 && nCtx < train) {
                args = "-c " + nCtx;
            }
        }
        if (args != null) {
            System.out.println("Fitted arguments: " + args);
        }
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

    private static void runGenerate(EngineModel model, LLMDemoOptions opts, EngineConfig cfg,
                                    java.nio.file.Path projector) throws IOException {
        boolean chatTemplate = !opts.raw && model.supportClass() != SupportClass.ENC_DEC;
        String prompt = opts.prompt != null ? opts.prompt
                : opts.nonInteractive && !opts.completionInteractive() ? readStdin() : null;
        if (projector != null) {
            // the media turns run the batch engine over the projector's context (mtmd-cli's single prompt and
            // chat loop); the text-only turn generator and the prompt cache play no part
            try (LocalMedia media = LocalMedia.open(model, cfg, opts, projector)) {
                media.printBanner(System.out);
                System.out.println();
                if (prompt != null) {
                    runMediaPrompt(model, media, prompt, opts, chatTemplate);
                } else {
                    LocalMedia.run(media, opts, LLMDemo::logStats);
                }
            }
            return;
        }
        PromptCache cache = PromptCache.open(model, opts);
        SpeculativeSession session = cfg.speculative() ? LLMModelFactory.speculate(model, cfg) : null;
        LookaheadDecoder lookahead = cfg.lookaheadOptions().enabled() ? LookaheadDecoder.forModel(model, cfg) : null;
        try {
            TurnGenerator generator;
            if (lookahead != null) {
                System.out.println("Lookahead decoding: window " + lookahead.window() + ", n-gram "
                        + lookahead.ngramSize() + ", max n-grams " + lookahead.maxNgrams() + " ("
                        + lookahead.sequences() + " sequences)");
                logStats("Lookahead: window " + lookahead.window() + ", n-gram " + lookahead.ngramSize()
                        + ", max n-grams " + lookahead.maxNgrams());
                generator = TurnGenerator.lookahead(model, lookahead);
            } else if (session != null) {
                SamplingDefaults defaults = SamplingDefaults.forModel(model, cfg);
                System.out.println("Speculative decoding: " + session.drafterKind() + " drafter, draft max "
                        + session.draftMax() + ", min " + session.draftMin() + ", p_min " + cfg.draftPMin());
                logStats("Speculative: " + session.drafterKind() + ", draft max " + session.draftMax());
                generator = new TurnGenerator() {
                    @Override
                    public GenerationResult generate(int[] prompt, GenerationRequest request,
                                                     ReasoningFilter.Sink sink, String rendered) {
                        return session.generate(prompt, request, sink, rendered);
                    }

                    @Override
                    public GenerationRequest effective(GenerationRequest request) {
                        return request.withModelDefaults(defaults);
                    }

                    @Override
                    public void reset() {
                        session.reset();
                    }
                };
            } else {
                generator = TurnGenerator.loop(model, LLMModelFactory.effectiveConfig(model, cfg));
            }
            if (cache != null) {
                generator = cache.wrap(generator);
            }
            if (opts.completionInteractive()) {
                CompletionRepl.run(model, generator, opts);
            } else if (prompt != null) {
                runSinglePrompt(model, generator, prompt, opts, chatTemplate);
            } else {
                LocalChat.run(model, generator, opts, LLMDemo::logStats);
            }
            if (session != null && session.lastStats() != null) {
                printSpeculativeStats(session.lastStats());
            }
            if (lookahead != null && lookahead.lastStats() != null) {
                printLookaheadStats(lookahead.lastStats());
            }
        } finally {
            if (session != null) {
                session.close();
            }
        }
    }

    private static void printLookaheadStats(LookaheadDecoder.Stats s) {
        String line = String.format(Locale.ROOT, "Lookahead: %d steps, %d emitted, %d accepted (%.2f per step)",
                s.rounds(), s.emitted(), s.accepted(), s.acceptanceRate());
        System.out.println("[" + line + "]");
        logStats(line);
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

    /** One prompt, one answer; with {@code -o} the turn is also written to the transcript. */
    private static void runSinglePrompt(EngineModel model, TurnGenerator generator, String prompt,
                                        LLMDemoOptions opts, boolean chatTemplate) throws IOException {
        Transcript transcript = opts.outFile == null ? null : Transcript.open(opts.outFile);
        try {
            System.out.println("Prompt: " + prompt);
            System.out.print("Generating: ");
            System.out.flush();
            GGUFTokenizer tokenizer = model.tokenizer();
            String fullPrompt = chatTemplate ? LocalChat.singleTurnPrompt(tokenizer, opts.systemPrompt, prompt,
                    opts.thinking, opts.templateKwargs) : prompt;
            int[] promptTokens = tokenizer.encode(fullPrompt);
            if (opts.debugPrompt) {
                printDebugPrompt(fullPrompt, promptTokens, tokenizer);
            }
            ChannelPrinter printer = new ChannelPrinter(System.out, opts.showAllChannels, opts.debugPrompt);
            GenerationRequest request = chatTemplate
                    ? LocalChat.templated(opts.generation, opts.thinking, opts.templateKwargs) : opts.generation;
            GenerationResult result = generator.generate(promptTokens, request, printer, fullPrompt);
            System.out.println();
            System.out.println();
            String statLine = LocalChat.statLine(promptTokens.length, result);
            System.out.println("[" + statLine + "]");
            logStats(statLine);
            logStats("  Prompt: " + prompt);
            logStats("  Response: " + printer.content());
            logStats("");
            if (transcript != null) {
                transcript.userTurn(prompt);
                transcript.assistant(printer.content(), printer.reasoning());
            }
        } finally {
            if (transcript != null) {
                transcript.close();
            }
        }
    }

    /**
     * One media prompt, one answer (mtmd-cli's single shot): the chat template renders the turn, the media
     * backend splices the {@code --image}/{@code --audio}/{@code --video} files into the rendered markers, and
     * the batch engine generates over the whole prompt.
     */
    private static void runMediaPrompt(EngineModel model, LocalMedia media, String prompt, LLMDemoOptions opts,
                                       boolean chatTemplate) throws IOException {
        Transcript transcript = opts.outFile == null ? null : Transcript.open(opts.outFile);
        try {
            System.out.println("Prompt: " + prompt);
            System.out.print("Generating: ");
            System.out.flush();
            GGUFTokenizer tokenizer = model.tokenizer();
            String fullPrompt = chatTemplate ? LocalChat.singleTurnPrompt(tokenizer, opts.systemPrompt, prompt,
                    opts.thinking, opts.templateKwargs) : prompt;
            if (opts.debugPrompt) {
                System.out.println("--- rendered prompt ---");
                System.out.println(fullPrompt);
                System.out.println("--- end ---");
                System.out.println();
            }
            ChannelPrinter printer = new ChannelPrinter(System.out, opts.showAllChannels, opts.debugPrompt);
            GenerationRequest request = chatTemplate
                    ? LocalChat.templated(opts.generation, opts.thinking, opts.templateKwargs) : opts.generation;
            GenerationResult result = media.turn(fullPrompt, request, printer);
            System.out.println();
            System.out.println();
            String statLine = LocalChat.statLine(result.timings().promptTokens(), result);
            System.out.println("[" + statLine + "]");
            logStats(statLine);
            logStats("  Prompt: " + prompt);
            logStats("  Response: " + printer.content());
            logStats("");
            if (transcript != null) {
                transcript.userTurn(prompt);
                transcript.assistant(printer.content(), printer.reasoning());
            }
        } finally {
            if (transcript != null) {
                transcript.close();
            }
        }
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
        String fullPrompt = opts.raw ? prompt
                : LocalChat.singleTurnPrompt(tokenizer, opts.systemPrompt, prompt, opts.thinking, opts.templateKwargs);
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
     * Audio models: WavTokenizer decodes codes to spectral features, which a {@code --audio-out} file ending in
     * {@code .wav} turns into the synthesized waveform (the model's declared speech geometry, the
     * {@code --audio-fft/--audio-hop/--audio-rate} overrides filling what the model does not declare); any other
     * {@code --audio-out} name keeps the float32 feature rows. Pockettts returns its backbone hidden states, or,
     * with {@code --mmproj}, speaks the prompt through the flow network and mimi decoder of its generator file.
     * An lfm3a backbone speaks the prompt through the MaskGIT depthformer and liquid-codec decoder of its
     * generator file. Token-generating speech models (qwen3tts) generate codec tokens from the raw prompt.
     */
    private static void runAudio(EngineModel model, LLMDemoOptions opts, EngineConfig cfg) throws IOException {
        Arch arch = model.spec().arch();
        if (arch == Arch.WAVTOKENIZER_DEC) {
            int[] codes = opts.audioCodes != null ? opts.audioCodes.clone() : parseCodes(opts.prompt);
            long t0 = System.nanoTime();
            boolean wave = opts.audioOut != null && opts.audioOut.toLowerCase(Locale.ROOT).endsWith(".wav");
            if (wave) {
                AudioClip clip = synthesize(model, opts, codes);
                long dt = (System.nanoTime() - t0) / 1_000_000L;
                System.out.println("Codes: " + codes.length);
                System.out.println("Waveform: " + clip.length() + " samples @ " + clip.sampleRate() + " Hz  time="
                        + dt + "ms");
                WavWriter.write(Path.of(opts.audioOut), clip.samples(), clip.sampleRate());
                System.out.println("Wrote " + opts.audioOut + " (16-bit PCM mono WAV)");
                logStats("wavtokenizer: " + codes.length + " codes -> " + clip.length() + " samples, " + dt + "ms");
                return;
            }
            float[][] features = model.decodeAudioCodes(codes);
            long dt = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("Codes: " + codes.length);
            System.out.println("Features: " + features.length + " frames x " + (features.length > 0 ? features[0].length : 0)
                    + "  time=" + dt + "ms (spectral frames; name --audio-out *.wav for the waveform)");
            if (features.length > 0) {
                System.out.println("  frame 0 = " + preview(features[0], opts.embedPreview));
            }
            logStats("WavTokenizer: " + codes.length + " codes -> " + features.length + " frames, " + dt + "ms");
            writeRows(opts.audioOut, features);
            return;
        }
        if (arch == Arch.QWEN3TTS) {
            runQwen3Tts(model, opts, cfg);
            return;
        }
        if (arch == Arch.LFM3) {
            runLfm3a(model, opts, cfg);
            return;
        }
        if (arch == Arch.POCKETTTS) {
            if (opts.mmproj != null) {
                runPocketTts(model, opts, cfg);
                return;
            }
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
        String line = LocalChat.statLine(tokens.length, result);
        System.out.println("[" + line + "]");
        logStats(line);
    }

    /**
     * The reference waveform of {@code --audio-speaker}, decoded through the shared audio decoder; null without the
     * flag.
     */
    private static AudioClip readSpeakerReference(LLMDemoOptions opts) throws IOException {
        if (opts.audioSpeaker == null) {
            return null;
        }
        return AudioDecoder.decode(Files.readAllBytes(Path.of(opts.audioSpeaker)), opts.audioSpeaker);
    }

    /**
     * qwen3tts: speaks the prompt through the generator file's code predictor and code2wav decoder, the pipeline
     * of llama.cpp's mtmd helper ({@code --mode audio} on a qwen3tts backbone). The {@code --audio-out} file
     * receives the waveform as 16-bit PCM mono WAV; the command's sampler settings (seed, top-k, top-p) drive the
     * codec sampling.
     */
    private static void runQwen3Tts(EngineModel model, LLMDemoOptions opts, EngineConfig cfg) throws IOException {
        String text = opts.prompt != null ? opts.prompt : opts.nonInteractive ? readStdin() : null;
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("--mode audio on " + model.architecture()
                    + " needs --prompt with the text to speak");
        }
        if (opts.mmproj == null) {
            throw new IllegalArgumentException("qwen3tts text-to-speech needs --mmproj <generator.gguf>: the code"
                    + " predictor and code2wav decoder that turn the backbone's codes into audio");
        }
        SamplerConfig sampler = opts.generation.sampler();
        SpeechOptions speech = SpeechOptions.defaults()
                .withLanguage(opts.audioLang)
                .withVoice(opts.voice)
                .withSpeakerReference(readSpeakerReference(opts))
                .withSeed(sampler.seed())
                .withTopK(sampler.topK() > 0 ? sampler.topK() : SpeechOptions.DEFAULT_TOP_K)
                .withTopP(sampler.topP());
        long t0 = System.nanoTime();
        AudioClip clip;
        try (Qwen3TtsClip generatorFile = Qwen3TtsClip.open(opts.mmproj, cfg.sourceOptions());
             Qwen3TtsSpeechGenerator generator = Qwen3TtsSpeechGenerator.open(model, generatorFile)) {
            clip = generator.synthesize(text, speech);
        }
        long dt = (System.nanoTime() - t0) / 1_000_000L;
        System.out.println("Text: " + text.length() + " chars");
        System.out.println("Waveform: " + clip.length() + " samples @ " + clip.sampleRate() + " Hz  time=" + dt
                + "ms");
        if (opts.audioOut != null) {
            WavWriter.write(Path.of(opts.audioOut), clip.samples(), clip.sampleRate());
            System.out.println("Wrote " + opts.audioOut + " (16-bit PCM mono WAV)");
        } else {
            System.out.println("Name --audio-out <file>.wav to write the waveform");
        }
        logStats("qwen3tts: " + text.length() + " chars -> " + clip.length() + " samples, " + dt + "ms");
    }

    /**
     * pockettts: speaks the prompt through the generator file's flow network and mimi decoder, the pipeline of
     * llama.cpp's mtmd helper ({@code --mode audio} with {@code --mmproj} on a pockettts backbone). The
     * {@code --audio-out} file receives the waveform as 16-bit PCM mono WAV; the seed drives the flow noise and
     * the local pack's temperature, while top-k/top-p do not apply because the decoder is continuous.
     */
    private static void runPocketTts(EngineModel model, LLMDemoOptions opts, EngineConfig cfg) throws IOException {
        String text = opts.prompt != null ? opts.prompt : opts.nonInteractive ? readStdin() : null;
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("--mode audio on " + model.architecture()
                    + " needs --prompt with the text to speak");
        }
        SamplerConfig sampler = opts.generation.sampler();
        SpeechOptions speech = SpeechOptions.defaults()
                .withLanguage(opts.audioLang)
                .withVoice(opts.voice)
                .withSpeakerReference(readSpeakerReference(opts))
                .withSeed(sampler.seed())
                .withTopK(sampler.topK() > 0 ? sampler.topK() : SpeechOptions.DEFAULT_TOP_K)
                .withTopP(sampler.topP());
        long t0 = System.nanoTime();
        AudioClip clip;
        try (PocketTtsClip generatorFile = PocketTtsClip.open(opts.mmproj, cfg.sourceOptions());
             PocketTtsSpeechGenerator generator = PocketTtsSpeechGenerator.open(model, generatorFile)) {
            clip = generator.synthesize(text, speech);
        }
        long dt = (System.nanoTime() - t0) / 1_000_000L;
        System.out.println("Text: " + text.length() + " chars");
        System.out.println("Waveform: " + clip.length() + " samples @ " + clip.sampleRate() + " Hz  time=" + dt
                + "ms");
        if (opts.audioOut != null) {
            WavWriter.write(Path.of(opts.audioOut), clip.samples(), clip.sampleRate());
            System.out.println("Wrote " + opts.audioOut + " (16-bit PCM mono WAV)");
        } else {
            System.out.println("Name --audio-out <file>.wav to write the waveform");
        }
        logStats("pockettts: " + text.length() + " chars -> " + clip.length() + " samples, " + dt + "ms");
    }

    /**
     * lfm3a: speaks the prompt through the generator file's MaskGIT depthformer and liquid-codec decoder, the
     * pipeline of llama.cpp's mtmd helper ({@code --mode audio} on an lfm3a backbone). The {@code --audio-out}
     * file receives the waveform as 16-bit PCM mono WAV. The sampling uses the generator file's model defaults;
     * the command's seed drives the draws and --audio-lang/--voice fill the tts template.
     */
    private static void runLfm3a(EngineModel model, LLMDemoOptions opts, EngineConfig cfg) throws IOException {
        String text = opts.prompt != null ? opts.prompt : opts.nonInteractive ? readStdin() : null;
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("--mode audio on " + model.architecture()
                    + " needs --prompt with the text to speak");
        }
        if (opts.mmproj == null) {
            throw new IllegalArgumentException("lfm3a text-to-speech needs --mmproj <generator.gguf>: the MaskGIT"
                    + " depthformer and liquid-codec decoder that turn the backbone's hidden states into audio");
        }
        SpeechOptions speech = SpeechOptions.defaults()
                .withLanguage(opts.audioLang)
                .withVoice(opts.voice)
                .withSpeakerReference(readSpeakerReference(opts))
                .withSeed(opts.generation.sampler().seed());
        long t0 = System.nanoTime();
        AudioClip clip;
        try (Lfm3aGenClip generatorFile = Lfm3aGenClip.open(opts.mmproj, cfg.sourceOptions());
             Lfm3aSpeechGenerator generator = Lfm3aSpeechGenerator.open(model, generatorFile)) {
            clip = generator.synthesize(text, speech);
        }
        long dt = (System.nanoTime() - t0) / 1_000_000L;
        System.out.println("Text: " + text.length() + " chars");
        System.out.println("Waveform: " + clip.length() + " samples @ " + clip.sampleRate() + " Hz  time=" + dt
                + "ms");
        if (opts.audioOut != null) {
            WavWriter.write(Path.of(opts.audioOut), clip.samples(), clip.sampleRate());
            System.out.println("Wrote " + opts.audioOut + " (16-bit PCM mono WAV)");
        } else {
            System.out.println("Name --audio-out <file>.wav to write the waveform");
        }
        logStats("lfm3a: " + text.length() + " chars -> " + clip.length() + " samples, " + dt + "ms");
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

    /**
     * The waveform of {@code codes}: the model's declared speech geometry
     * ({@link EngineModel#speechFormat()}, the GGUF keys {@code audio.sample_rate}, {@code audio.n_fft} and
     * {@code audio.hop_length}) with the {@code --audio-fft/--audio-hop/--audio-rate} flags filling what the
     * model does not declare; a missing piece refuses naming both sources.
     */
    private static AudioClip synthesize(EngineModel model, LLMDemoOptions opts, int[] codes) throws IOException {
        SpeechFormat declared = model.speechFormat();
        if (declared == null && (opts.audioFft == 0 || opts.audioHop == 0 || opts.audioRate == 0)) {
            throw new IllegalArgumentException("the model declares no speech decoding geometry: give the GGUF keys"
                    + " audio.sample_rate, audio.n_fft and audio.hop_length (all three), or the flags"
                    + " --audio-rate, --audio-fft and --audio-hop");
        }
        if (declared == null) {
            return new SpeechFormat(opts.audioRate, opts.audioFft, opts.audioHop).decode(model.decodeAudioCodes(codes));
        }
        int rate = opts.audioRate > 0 ? opts.audioRate : declared.sampleRate();
        int fft = opts.audioFft > 0 ? opts.audioFft : declared.nFft();
        int hop = opts.audioHop > 0 ? opts.audioHop : declared.hopLength();
        SpeechFormat format = rate == declared.sampleRate() && fft == declared.nFft() && hop == declared.hopLength()
                ? declared : new SpeechFormat(rate, fft, hop);
        return format.decode(model.decodeAudioCodes(codes));
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
        PromptDebug.print(System.out, fullPrompt, tokens, tokenizer);
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
        TurnGenerator generator;
        if (model instanceof EngineModel) {
            EngineModel em = (EngineModel) model;
            generator = TurnGenerator.loop(em, LLMModelFactory.effectiveConfig(em, opts.engine));
        } else {
            generator = TurnGenerator.ofModel(model);
        }
        LocalChat.run(model, generator, opts, LLMDemo::logStats);
    }
}
