package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardBackend;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeightUpdater;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ClassifierOutput;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.DiffusionGenerator;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LoraSpec;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.engine.QuantizedView;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.lora.LoraRegistry;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.PoolingOperator;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.audio.CpuWavTokenizer;
import com.eonmux.neuralfree.gguf.ops.audio.TtsBackbone;
import com.eonmux.neuralfree.gguf.ops.encdec.CpuEncDecBackend;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.ControlVector;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionBody;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.ops.heads.ModernBertDecisionHead;
import com.eonmux.neuralfree.gguf.ops.heads.PoolingHead;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraBackend;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * A GGUF model run on the CPU by the spec-driven engine.
 *
 * <p>Construction: the {@link ModelSource} ({@link ModelSources} with the configuration's
 * {@link EngineConfig#sourceOptions()}), tokenizer, {@link GgufKv} and {@link TensorCatalog},
 * {@link ModelSpecBuilder}; the memory settings the model runs with ({@link MemoryConfig#forModel}: models that
 * evaluate their whole input in one step get a ubatch of their whole context, models without memory at most their
 * context); the {@link CpuWeights} read in place from the mapped file; for models with memory the context
 * {@link ContextFitter} chooses within the Java heap, beside the NextN / MTP draft caches and a configured draft
 * model's host memory ({@link LLMModelFactory#draftCompanion}), its {@link MemoryPlan} ({@link MemoryPlanner}) and the
 * {@link HeapModelMemory}; the {@link CpuScratch} of the plan's ubatch (of the settings' ubatch without memory), the
 * {@link CpuForwardContext}, the {@link CpuBlockRegistry} of {@link CpuFactoryCatalog} and the backend:
 * {@link CpuEncDecBackend} for encoder-decoder models, {@link CpuForwardBackend} otherwise (wrapped for the
 * graniteswitch adapter router and the qwen3tts logits layout), or the {@link CpuWavTokenizer} decoder for
 * wavtokenizer-dec.</p>
 *
 * <p>Use: autoregressive generation through {@link GenerationLoop} (the legacy {@code generate} overloads map onto
 * {@link GenerationRequest}); embeddings, classification and reranking through {@link ForwardBackend#hiddenRows}
 * and the heads; masked diffusion through {@link DiffusionGenerator}; audio codes through the wavtokenizer decoder.
 * The scalar getters describe trunk layer 0. Not thread-safe. {@link #close()} releases the backend, the memory,
 * the weights and the source.</p>
 */
public final class GGUFModel implements EngineModel {

    private static final Logger LOGGER = LogManager.getLogger(GGUFModel.class);

    private final ModelSource source;
    private final GGUFTokenizer tokenizer;
    private final EngineConfig config;
    private final ModelSpec spec;
    private final CpuWeights weights;
    private final MemoryPlan plan;
    private final HeapModelMemory memory;
    private final CpuForwardContext context;
    private final CpuBlockRegistry registry;
    /** The token forward pass; rebuilt in place by a live weight update's rebind ({@link #rebindForwardStack}). */
    private ForwardBackend backend;
    private final CpuWavTokenizer wavTokenizer;
    /** Speech decoding geometry of the optional audio keys; null when the model declares none. */
    private final SpeechFormat speechFormat;
    private final LoraAttachment loras;
    /** The per-request adapters the attachment serves; null without a token forward pass. */
    private final LoraRegistry loraRegistry;
    private final SamplingDefaults samplingDefaults;
    /** The control vectors the load applied over the forward spec, or null; the quantized view refuses a copy that would run a different graph. */
    private final ControlVector controlVectors;
    /** Why the model factory fell back from a GPU load to this CPU engine, or null when it did not. */
    private final String gpuFallbackReason;

    private GenerationLoop generationLoop;
    private PoolingHead poolingHead;
    private DecisionScorer decisionScorer;
    private PoolingOperator poolingOp;
    private boolean closed;
    private final MoeCacheRanking ranking;

    /** Loads the model with the engine settings of the system properties. */
    public GGUFModel(String path) throws IOException {
        this(path, EngineConfig.fromSystemProperties());
    }

    /**
     * Loads the model.
     *
     * @param config the requested settings; {@link #engineConfig()} returns them with the ubatch the model runs with
     * @throws IOException                          when the file or its tensor data cannot be read
     * @throws ContextFitter.ContextBudgetException when no allowed context fits the Java heap
     * @throws UnsupportedOperationException        for draft-only architectures or blocks the CPU cannot run
     */
    public GGUFModel(String path, EngineConfig config) throws IOException {
        this(path, config, null);
    }

    /**
     * Loads the model, reporting a GPU fallback.
     *
     * @param config           the requested settings; {@link #engineConfig()} returns them with the ubatch the model
     *                         runs with
     * @param gpuFallbackReason why the model factory fell back from a GPU load to this CPU engine, or null when it
     *                         did not ({@link #gpuFallbackReason()} reports it)
     * @throws IOException                          when the file or its tensor data cannot be read
     * @throws ContextFitter.ContextBudgetException when no allowed context fits the Java heap
     * @throws UnsupportedOperationException        for draft-only architectures or blocks the CPU cannot run
     */
    public GGUFModel(String path, EngineConfig config, String gpuFallbackReason) throws IOException {
        this.gpuFallbackReason = gpuFallbackReason;
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(config, "config");
        this.source = ModelSources.open(path, config.sourceOptions(), null, config.activationQuant());
        Deque<AutoCloseable> opened = new ArrayDeque<>();
        try {
            this.tokenizer = new GGUFTokenizer(source);
            GgufKv kv = GgufKv.of(source);
            this.samplingDefaults = SamplingDefaults.read(kv);
            TensorCatalog tc = TensorCatalog.of(source);
            ModelSpec resolved = ModelSpecBuilder.build(kv, tc, config.memory(), 0, config.pooling());
            rejectStandalone(resolved);
            this.ranking = MoeCacheRanking.open(Paths.get(path), MoeCacheRanking.maxExperts(resolved),
                    config.moeCache().ranking());
            opened.push(ranking);
            MemoryConfig mc = config.memory().forModel(resolved);

            this.weights = CpuWeights.open(source, tc,
                    InputTransformLoader.load(source, kv, tc, config.activationQuant()));
            opened.push(weights);
            CpuScratch scratch = CpuScratch.forModel(resolved, mc.ubatch());

            if (resolved.memoryKind() != MemoryKind.NONE) {
                ContextFitter.Companion drafter =
                        LLMModelFactory.draftCompanion(config, resolved, true, (draft, placed) -> 0L)
                                .plus(LLMModelFactory.twoLevelCompanion(config, source, false));
                int specCtx = mc.autoContext() ? resolved.hparams().nCtxTrain() : mc.nCtx();
                ModelSpec fitted = resolved;
                int nCtx = HeapMemory.fitContext(budget -> ContextFitter.fit(fitted, mc, budget, drafter), specCtx);
                if (nCtx != specCtx) {
                    resolved = ModelSpecBuilder.build(kv, tc, mc, nCtx, config.pooling());
                }
                this.plan = MemoryPlanner.planForContext(resolved, mc, nCtx);
                if (plan.ubatch() != scratch.ubatch()) {
                    scratch = CpuScratch.forModel(resolved, plan.ubatch());
                }
                this.memory = HeapModelMemory.allocate(plan, resolved, null);
                opened.push(memory);
            } else {
                this.plan = null;
                this.memory = null;
            }
            this.spec = resolved;
            this.config = config.toBuilder().memory(mc.toBuilder().ubatch(scratch.ubatch()).build()).build();

            ModelSpec runSpec = DecisionBody.forwardSpec(resolved);
            this.controlVectors = controlVectors(config, runSpec);
            if (this.controlVectors != null) {
                runSpec = ControlVector.applied(runSpec);
            }
            CpuForwardContext.Builder cb = CpuForwardContext.builder(runSpec, weights, scratch);
            if (memory != null) {
                cb.memory(memory);
            }
            cb.heat(ranking.heat());
            this.context = cb.build();

            if (resolved.arch() == Arch.WAVTOKENIZER_DEC) {
                if (!config.loras().isEmpty()) {
                    throw new UnsupportedOperationException(resolved.arch().ggufName() + " has no token forward pass"
                            + " to apply LoRA adapters to");
                }
                this.registry = null;
                this.backend = null;
                this.wavTokenizer = CpuWavTokenizer.bind(context);
                this.speechFormat = SpeechFormat.ofGguf(kv);
                this.loras = null;
                this.loraRegistry = null;
            } else {
                CpuFactoryCatalog.checkRouting(runSpec);
                SwitchedLoraExtra.Router router = CpuFactoryCatalog.usesSwitchedLora(runSpec)
                        ? SwitchedLoraExtra.Router.of(resolved.hparams()) : null;
                this.registry = CpuFactoryCatalog.defaults(router, controlVectors);
                registry.validate(runSpec);
                this.backend = openBackend(context, registry, router, tokenizer);
                opened.push(backend);
                this.wavTokenizer = null;
                this.speechFormat = null;
                this.loras = attachLoras(config, weights, context);
                opened.push(loras);
                this.loraRegistry = LoraRegistry.open(weights, config.sourceOptions(), config.loraConfig());
                loras.serve(loraRegistry);
            }
        } catch (IOException | RuntimeException | Error e) {
            closeAll(opened, e);
            closeSource(source, e);
            throw e;
        }
        LOGGER.info("{} loaded on the CPU: {} ({})", spec.arch().ggufName(), spec.supportClass(),
                plan != null ? plan : "no memory");
        if (config.speculativeOptions().w12()) {
            LOGGER.info("{}: {}=true has no CPU path; the weights keep their stored type (the lossless W12"
                            + " streaming decodes in the GPU projection kernels)", spec.arch().ggufName(),
                    SpeculativeOptions.PROP_W12);
        }
    }

    // ------------------------------------------------------------------ construction

    /**
     * Loads the configured control vectors ({@link EngineConfig#controlVectors()}, llama.cpp
     * {@code --control-vector} / {@code --control-vector-scaled} with {@code --control-vector-layer-range}) against
     * the bound model {@code spec}; null when none is configured.
     *
     * @throws IOException                   when a vector file cannot be read
     * @throws GgufFormatException           when a file is not a control vector in the llama.cpp file form or its
     *                                       direction rows are not exactly the model's {@code n_embd} wide
     * @throws IllegalArgumentException      for a layer range outside the model or a scale that is not finite
     * @throws UnsupportedOperationException for the WavTokenizer decoder
     */
    private static ControlVector controlVectors(EngineConfig config, ModelSpec spec) throws IOException {
        if (config.controlVectors().isEmpty()) {
            return null;
        }
        if (spec.arch() == Arch.WAVTOKENIZER_DEC) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no token forward pass to apply"
                    + " control vectors to");
        }
        ControlVector set = ControlVector.load(spec, config.controlVectors(), config.controlVectorLayerStart(),
                config.controlVectorLayerEnd());
        LOGGER.info("{}: {} control vector(s) applied over layers {}..{}: {}", spec.arch().ggufName(),
                set.list().size(), set.list().get(0).layerStart(), set.list().get(0).layerEnd(), set.list());
        return set;
    }

    /**
     * Attaches the configured LoRA adapters ({@link EngineConfig#loras()}, llama.cpp {@code --lora} /
     * {@code --lora-scaled}) to the bound blocks of {@code ctx}, applied to every row; the attachment also serves
     * the per-request adapters of the model's {@link LoraRegistry}.
     *
     * @throws IOException                   when an adapter file cannot be read
     * @throws UnsupportedOperationException for a refused adapter or adapted product
     * @throws IllegalStateException         when an adapter tensor targets a product the model does not run
     */
    private static LoraAttachment attachLoras(EngineConfig config, CpuWeights weights, CpuForwardContext ctx)
            throws IOException {
        List<LoraSpec> specs = config.loras();
        List<LoraAttachment.Binding> bindings = new ArrayList<>(specs.size());
        for (LoraSpec s : specs) {
            bindings.add(LoraAttachment.Binding.load(s.path(), s.scale(), weights, config.sourceOptions()));
        }
        LoraAttachment att = LoraAttachment.cpu(bindings, ctx.projectionDeltas());
        if (!bindings.isEmpty()) {
            LOGGER.info("{}: {} LoRA adapter(s) attached, {} deltas: {}", ctx.spec().arch().ggufName(),
                    bindings.size(), att.deltaCount(), bindings);
        }
        return att;
    }

    /** Draft-only architectures run beside a target model, never on their own. */
    private static void rejectStandalone(ModelSpec spec) {
        if (spec.supportClass() == SupportClass.DRAFT) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " is a draft model; load it beside its"
                    + " target model as a speculative drafter (DraftModelLoader)");
        }
    }

    private static ForwardBackend openBackend(CpuForwardContext ctx, CpuBlockRegistry registry,
                                              SwitchedLoraExtra.Router router, GGUFTokenizer tokenizer) {
        ModelSpec s = ctx.spec();
        ForwardBackend b = s.supportClass() == SupportClass.ENC_DEC
                ? new CpuEncDecBackend(ctx, registry, CpuFactoryCatalog.headPreprocessors(), tokenizer.bosId())
                : new CpuForwardBackend(ctx, registry, CpuFactoryCatalog.headPreprocessors());
        try {
            if (router != null) {
                b = SwitchedLoraBackend.wrap(b, router);
            }
            if (s.arch() == Arch.QWEN3TTS) {
                b = TtsBackbone.qwen3tts(b, s);
            }
            return b;
        } catch (RuntimeException e) {
            try {
                b.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    // ------------------------------------------------------------------ engine API

    @Override
    public ModelSpec spec() {
        return spec;
    }

    /** The sampling defaults of the model's metadata ({@link SamplingDefaults#read}). */
    @Override
    public SamplingDefaults modelSamplingDefaults() {
        return samplingDefaults;
    }

    /** The registry of the per-request adapters; null for wavtokenizer-dec, which runs no tokens. */
    @Override
    public LoraRegistry loraRegistry() {
        requireOpen();
        return loraRegistry;
    }

    /**
     * @throws UnsupportedOperationException for wavtokenizer-dec, which decodes audio codes
     *                                       ({@link #decodeAudioCodes}) instead of running tokens
     */
    @Override
    public ForwardBackend forwardBackend() {
        requireOpen();
        if (backend == null) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no token forward pass; decode"
                    + " audio codes with decodeAudioCodes");
        }
        return backend;
    }

    /** The CPU context the backend runs on (the target context of a NextN drafter). */
    public CpuForwardContext forwardContext() {
        return context;
    }

    /**
     * Opens the quantized target view of two-level verification ({@code neuralfree.draft.tlv}) over this model's
     * memory: a {@link QCopySource} restates this model's projection matrices at the copy's bits, a second
     * {@link CpuWeights} and forward context run them beside this model's passes through the same block registry,
     * and the returned view owns the weights and the backend it built. The copy reads and writes this model's KV
     * cache, so a Q round's rows sit exactly where the exact round's would.
     *
     * @throws UnsupportedOperationException naming {@code neuralfree.draft.tlv} for the WavTokenizer decoder (no
     *                                       token pass to copy), a target with LoRA adapters or control vectors
     *                                       (the copy's passes would run a different graph than the target's),
     *                                       and for the targets {@link QCopySource} refuses (routed experts, no
     *                                       matrix the copy undercuts)
     */
    @Override
    public QuantizedView openQuantizedView(int bits) throws IOException {
        requireOpen();
        if (backend == null) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " opens a quantized copy of the"
                    + " target's projection matrices to verify a draft tree against, but " + spec.arch().ggufName()
                    + " runs no token forward pass to copy");
        }
        if (!config.loras().isEmpty()) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " copies the target's projection"
                    + " matrices, whose passes would run without the LoRA adapters the target's run; clear the"
                    + " adapters or " + SpeculativeOptions.PROP_TLV);
        }
        if (controlVectors != null) {
            throw new UnsupportedOperationException(SpeculativeOptions.PROP_TLV + " copies the target's projection"
                    + " matrices, whose passes would run without the control vectors the target's apply; clear"
                    + " the control vectors or " + SpeculativeOptions.PROP_TLV);
        }
        ModelSpec runSpec = DecisionBody.forwardSpec(spec);
        QCopySource qSource = QCopySource.open(source, bits);
        QuantizedView.Build build = new QuantizedView.Build();
        try {
            TensorCatalog qCatalog = TensorCatalog.of(qSource);
            CpuWeights qWeights = CpuWeights.open(qSource, qCatalog,
                    InputTransformLoader.load(qSource, GgufKv.of(source), qCatalog, config.activationQuant()));
            build.owns(qWeights);
            CpuScratch qScratch = CpuScratch.forModel(spec, plan.ubatch());
            CpuForwardContext.Builder qb = CpuForwardContext.builder(runSpec, qWeights, qScratch)
                    .memory(memory)
                    .heat(ranking.heat());
            SwitchedLoraExtra.Router router = CpuFactoryCatalog.usesSwitchedLora(runSpec)
                    ? SwitchedLoraExtra.Router.of(spec.hparams()) : null;
            CpuBlockRegistry qRegistry = CpuFactoryCatalog.defaults(router, null);
            qRegistry.validate(runSpec);
            ForwardBackend qBackend = openBackend(qb.build(), qRegistry, router, tokenizer);
            build.owns(qBackend);
            LOGGER.info("{}: quantized target view for two-level verification: {} restated matrices, {} bytes"
                            + " beside the target's weights", spec.arch().ggufName(), qSource.getTensorCount(),
                    qSource.plannedBytes());
            return new QuantizedView(qBackend, build.owned(),
                    "quantized target view (" + bits + "-bit rows) of " + source.getFilePath(),
                    qSource.plannedBytes(), bits);
        } catch (RuntimeException e) {
            throw build.close(e);
        }
    }

    /** The host weights: token embeddings and output norms for the speech generator. */
    @Override
    public CpuWeights cpuWeights() {
        requireOpen();
        return weights;
    }

    @Override
    public WeightUpdater weightUpdater() {
        requireOpen();
        if (backend == null) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " has no token forward pass; its audio"
                    + " tensors take no live weight update");
        }
        return new CpuWeightUpdater(weights, name(), this::rebindForwardStack);
    }

    /**
     * Rebuilds the forward pass over the context's weights, so the tensors a live weight update replaced bind into
     * it: every block re-resolves its tensors from the weights and re-serves its delta sites, and the LoRA
     * adapters registered on them apply to the new bindings. The memory and its sequences carry over untouched.
     * Called between forward passes, on the model's own thread.
     */
    private void rebindForwardStack() {
        ModelSpec runSpec = DecisionBody.forwardSpec(spec);
        SwitchedLoraExtra.Router router = CpuFactoryCatalog.usesSwitchedLora(runSpec)
                ? SwitchedLoraExtra.Router.of(spec.hparams()) : null;
        ForwardBackend fresh = openBackend(context, registry, router, tokenizer);
        ForwardBackend old = backend;
        backend = fresh;
        generationLoop = null;
        poolingHead = null;
        decisionScorer = null;
        poolingOp = null;
        old.close();
    }

    /**
     * The CPU blocks of this model (for a NextN drafter over its own MTP layers).
     *
     * @throws UnsupportedOperationException for wavtokenizer-dec
     */
    public CpuBlockRegistry blockRegistry() {
        if (registry == null) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " runs no CPU blocks");
        }
        return registry;
    }

    /** The memory plan; null for models without memory. */
    public MemoryPlan memoryPlan() {
        return plan;
    }

    /** The engine settings the model was loaded with; its memory settings carry the ubatch the model runs with. */
    public EngineConfig engineConfig() {
        return config;
    }

    /** Why the model factory fell back from a GPU load to this CPU engine, or null when it did not. */
    @Override
    public String gpuFallbackReason() {
        return gpuFallbackReason;
    }

    // ------------------------------------------------------------------ generation

    @Override
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        requireOpen();
        return loop().generate(prompt, request, callback);
    }

    @Override
    public GenerationResult generateEmbeds(float[][] promptEmbeds, GenerationRequest request, TokenCallback callback) {
        requireOpen();
        return loop().generate(EmbedsPrompt.ofRows(promptEmbeds), request, callback);
    }

    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, null);
    }

    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                          TokenCallback callback) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, 1.0f, 0, 0.0f, 0, callback);
    }

    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                          float repetitionPenalty, int repeatWindow, TokenCallback callback) throws IOException {
        return generate(tokens, maxNewTokens, temperature, topP, repetitionPenalty, repeatWindow, 0.0f, 0, callback);
    }

    /**
     * Generates through {@link GenerationLoop} with the sampler {@link EngineConfig#legacySampler} maps these
     * parameters to over the configured sampler defaults (seed, typical-p, presence and frequency penalties, greedy),
     * the same chain the GPU engine builds; {@code repeatWindow} 0 or -1 penalizes the whole context.
     *
     * @return the prompt followed by the generated tokens, or only the generated tokens for encoder-decoder models
     * @throws IllegalArgumentException for empty tokens, a negative token budget or out-of-range sampling values
     */
    @Override
    public int[] generate(int[] tokens, int maxNewTokens, float temperature, float topP,
                          float repetitionPenalty, int repeatWindow, float minP, int topK,
                          TokenCallback callback) throws IOException {
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("Input tokens must not be empty");
        }
        if (maxNewTokens < 0) {
            throw new IllegalArgumentException("maxNewTokens must be >= 0, got " + maxNewTokens);
        }
        boolean encDec = isEncoderDecoder();
        if (maxNewTokens == 0) {
            return encDec ? new int[0] : tokens.clone();
        }
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(config.legacySampler(temperature, topP, repetitionPenalty, repeatWindow, minP, topK))
                .build();
        GenerationResult result = generate(tokens, request, callback);
        return encDec ? result.tokens() : result.withPrompt(tokens);
    }

    @Override
    public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP)
            throws IOException {
        return generateWithPrompt(prompt, maxNewTokens, temperature, topP, null);
    }

    @Override
    public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                    TokenCallback callback) throws IOException {
        return generate(tokenizer.encode(prompt), maxNewTokens, temperature, topP, callback);
    }

    @Override
    public int[] generateWithPrompt(String prompt, int maxNewTokens, float temperature, float topP,
                                    float repetitionPenalty, int repeatWindow,
                                    TokenCallback callback) throws IOException {
        return generate(tokenizer.encode(prompt), maxNewTokens, temperature, topP, repetitionPenalty, repeatWindow,
                callback);
    }

    @Override
    public int[] generateDiffusion(int[] prompt, DiffusionConfig diffusion, TokenCallback callback) {
        requireOpen();
        return DiffusionGenerator.forModel(this, config).generate(prompt, diffusion, callback);
    }

    /** Empties the memory (KV caches, recurrent states and the adapter router history). */
    @Override
    public void resetKVCache() {
        requireOpen();
        if (backend != null) {
            backend.reset();
        }
    }

    /**
     * Writes this model's session state to {@code out} ({@code SessionFile.save}, llama.cpp's
     * {@code llama_state_save_file}): the stored tokens and every KV group's rows and recurrent state, over either
     * backend. A later {@link #loadSession} into a model of the same plan continues the sequence exactly, and the
     * generation loop's prefix reuse then skips the shared tokens of a matching prompt.
     *
     * @throws UnsupportedOperationException when the memory is outside the session file's scope (a paged memory,
     *                                       DSV4 compressed caches, the sparse side caches, cross-attention caches
     *                                       or caches that read a target model), naming it
     * @throws IOException                   when the write fails
     */
    @Override
    public void saveSession(Path out) throws IOException {
        sessionMemory("save").saveSession(out);
    }

    /**
     * Reads the session file at {@code in} into this model's memory, resetting it first ({@code
     * SessionFile.load}, llama.cpp's {@code llama_state_load_file}).
     *
     * @throws UnsupportedOperationException as {@link #saveSession}
     * @throws IOException                   for an unreadable file, a foreign format or a plan mismatch
     */
    @Override
    public void loadSession(Path in) throws IOException {
        sessionMemory("load").loadSession(in);
    }

    /** The memory a session action moves, refused by name for a model without one. */
    private ModelMemory sessionMemory(String action) {
        requireOpen();
        ModelMemory memory = forwardBackend().memory();
        if (memory == null) {
            throw new UnsupportedOperationException("session " + action + " refused: " + architecture()
                    + " holds no KV cache memory");
        }
        return memory;
    }

    private GenerationLoop loop() {
        if (generationLoop == null) {
            generationLoop = GenerationLoop.forModel(this, config);
        }
        return generationLoop;
    }

    // ------------------------------------------------------------------ embeddings and heads

    @Override
    public boolean isEncoder() {
        return EncodeGate.isEncoder(spec);
    }

    @Override
    public boolean isEncoderDecoder() {
        return spec.supportClass() == SupportClass.ENC_DEC;
    }

    @Override
    public LLMModel.PoolingType poolingType() {
        return spec.head().pooling().toPoolingType();
    }

    @Override
    public float[] encode(int[] tokens) throws IOException {
        return encode(tokens, null);
    }

    /**
     * The pooled embedding (MEAN / CLS / LAST), the ranking scores (RANK) or every token's embedding concatenated
     * (NONE), through the dense projections. Embedding models and models whose GGUF sets a pooling type encode.
     *
     * @throws UnsupportedOperationException for generative models without pooling and for decision models
     *                                       ({@link #classify})
     */
    @Override
    public float[] encode(int[] tokens, int[] tokenTypes) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkEncode(spec);
        PoolingHead head = poolingHead();
        float[][] hidden = hidden(tokens, tokenTypes);
        if (head.pooling() != PoolingKind.NONE) {
            return head.pool(hidden);
        }
        if (head.sparsePooling() != null) {
            return head.sparseEmbed(hidden, tokens);
        }
        return flatten(head.poolTokens(hidden, tokens), head.tokenEmbeddingWidth());
    }

    /**
     * {@link #encode(int[])} over a media prompt: the prompt's text and media rows pool as one sequence, the media
     * rows where the projector placed them. The memory must be paged
     * ({@code neuralfree.parallel > 1 or neuralfree.kv.paged=true}); the contiguous form refuses by name.
     *
     * @throws UnsupportedOperationException for generative models without pooling, for decision models
     *                                       ({@link #classify}), for a sparse embedding, and off a paged memory
     */
    @Override
    public float[] encodeMedia(MediaPrompt prompt) throws IOException {
        requireOpen();
        Objects.requireNonNull(prompt, "prompt");
        EncodeGate.checkEncode(spec);
        PoolingHead head = poolingHead();
        if (head.sparsePooling() != null) {
            throw new UnsupportedOperationException("a sparse embedding encodes token inputs, not a media prompt");
        }
        float[][] hidden = hidden(prompt);
        if (head.pooling() != PoolingKind.NONE) {
            return head.pool(hidden);
        }
        return flatten(head.poolTokens(hidden), head.tokenEmbeddingWidth());
    }

    /**
     * Per-token embeddings of NONE pooling: {@code [tokens.length][width]} through the dense projections; for
     * pockettts the backbone's {@code t_embd} rows.
     *
     * @throws UnsupportedOperationException for a decision model ({@link #classify}), and when the model pools per
     *                                       sequence or has no per-token embedding output
     */
    @Override
    public float[][] encodeTokens(int[] tokens) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkEncodeTokens(spec);
        return poolingHead().poolTokens(hidden(tokens, null), tokens);
    }

    /**
     * The pooled output row of every token sequence of one batch (MEAN / CLS / LAST pooling, the RANK classifier
     * scores): a paged memory runs the sequences as multi-sequence forwards from position 0, packed to its
     * {@code neuralfree.parallel} sequences and its context ({@link PagedEvaluations#hiddenSequences}); an encoder
     * planned without memory packs them into memoryless batches of one ubatch, each sequence attending only itself
     * ({@link PagedEvaluations#memorylessSequences}); any other form embeds them one after another.
     *
     * @throws UnsupportedOperationException for a decision model, and for NONE pooling
     * @throws IllegalArgumentException      when the batch or one of its sequences is empty, or a sequence is longer
     *                                       than one sequence of the memory holds
     */
    @Override
    public float[][] encodeBatch(int[][] sequences) throws IOException {
        requireOpen();
        PagedEvaluations.checkSequences(sequences);
        EncodeGate.checkEncodeBatch(spec);
        ForwardBackend b = forwardBackend();
        boolean packs = PagedEvaluations.packsMemoryless(b);
        if (!PagedEvaluations.paged(b) && !packs) {
            float[][] out = new float[sequences.length][];
            for (int i = 0; i < sequences.length; i++) {
                out[i] = encode(sequences[i]);
            }
            return out;
        }
        PoolingHead head = poolingHead();
        float[][][] hidden;
        if (packs) {
            hidden = PagedEvaluations.memorylessSequences(b, sequences);
        } else {
            b.reset();
            hidden = PagedEvaluations.hiddenSequences(b, sequences);
        }
        float[][] out = new float[sequences.length][];
        for (int s = 0; s < sequences.length; s++) {
            out[s] = head.pool(hidden[s]);
        }
        return out;
    }

    @Override
    public float[] classify(int[] tokens) throws IOException {
        return classify(tokens, null);
    }

    /**
     * Classifier outputs of one input:
     * <ul>
     *   <li>a pooled classifier (RANK or another pooled type): the classifier scores, one per
     *       {@link #classLabels()} entry;</li>
     *   <li>vllm-decision2 and clef: one score per token (row {@code i} scores option {@code i});</li>
     *   <li>modern-bert and lfm2 decision models: {@code [t][it]} row-major,
     *       {@link ModernBertDecisionHead#N_DECISION_TYPES} scores per token.</li>
     * </ul>
     *
     * @param order one decision code per token for the vllm-decision2 and clef heads, or null
     * @throws UnsupportedOperationException when the model has no classifier or decision head
     * @throws IllegalArgumentException      when an order is given to a head that reads none, or its length differs
     *                                       from the token count
     */
    @Override
    public float[] classify(int[] tokens, int[] order) throws IOException {
        requireTokens(tokens);
        DecisionScorer scorer = decisionScorer();
        if (scorer != null) {
            scorer.checkOrder(tokens.length, order);
            return scorer.score(hidden(tokens, null), tokens, order);
        }
        if (order != null) {
            throw new IllegalArgumentException(spec.arch().ggufName() + " reads no decision order");
        }
        if (!EncodeGate.hasSequenceClassifier(spec)) {
            throw new UnsupportedOperationException("classification needs a RANK classifier, another sequence"
                    + " classifier or a decision head; "
                    + spec.arch().ggufName() + " pools by " + spec.head().pooling()
                    + (EncodeGate.hasTokenClassifier(spec) ? " and classifies per token (use tokenClassify)" : ""));
        }
        return poolingHead().classify(hidden(tokens, null)).scores();
    }

    @Override
    public float[][] tokenClassify(int[] tokens) throws IOException {
        requireTokens(tokens);
        EncodeGate.checkTokenClassify(spec);
        return poolingHead().classifyTokens(hidden(tokens, null), tokens);
    }

    @Override
    public String[] classLabels() {
        requireOpen();
        if (DecisionScorer.present(spec)
                || (!EncodeGate.hasSequenceClassifier(spec) && !EncodeGate.hasTokenClassifier(spec))) {
            return new String[0];
        }
        return poolingHead().classifier().labels();
    }

    /**
     * The sequence classifier's output for one input: scores with their labels and the head's normalized flag. A
     * decision model refuses, unless it reads its fixed question over the pooled classifier output
     * ({@link EncodeGate#hasDecisionClassifier}, the pooled decision read).
     *
     * @throws UnsupportedOperationException when the model has neither a sequence classifier nor the pooled decision
     *                                       read
     */
    @Override
    public ClassifierOutput classifyOutput(int[] tokens) throws IOException {
        return classifyOutput(tokens, true);
    }

    /**
     * {@link #classifyOutput(int[])} with the classifier's final probability activation applied or skipped
     * (vLLM's {@code use_activation}); {@code false} answers the raw head output, which is not normalized. A
     * decision model refuses, unless it reads its fixed question over the pooled classifier output.
     *
     * @throws UnsupportedOperationException when the model has neither a sequence classifier nor the pooled decision
     *                                       read
     */
    @Override
    public ClassifierOutput classifyOutput(int[] tokens, boolean activation) throws IOException {
        requireTokens(tokens);
        if (!EncodeGate.hasDecisionClassifier(spec)) {
            if (DecisionScorer.present(spec)) {
                throw new UnsupportedOperationException(spec.arch().ggufName() + " is a decision model; its"
                        + " decision scores come from classify, not a classifier output");
            }
            if (!EncodeGate.hasSequenceClassifier(spec)) {
                throw new UnsupportedOperationException("a classifier output needs a RANK classifier or another"
                        + " sequence classifier; " + spec.arch().ggufName() + " pools by " + spec.head().pooling());
            }
        }
        return poolingHead().classify(hidden(tokens, null), activation);
    }

    /**
     * {@link #classifyOutput(int[])} over a media prompt: the sequence classifier's scores over the pooled row of the
     * prompt's text and media rows, as vLLM's multimodal rerankers score a media prompt. The memory must be paged
     * ({@code neuralfree.parallel > 1 or neuralfree.kv.paged=true}); the contiguous form refuses by name.
     *
     * @throws UnsupportedOperationException when the model has neither a sequence classifier nor the pooled decision
     *                                       read, or the memory is not paged
     */
    @Override
    public ClassifierOutput classifyOutputMedia(MediaPrompt prompt) throws IOException {
        return classifyOutputMedia(prompt, true);
    }

    /**
     * {@link #classifyOutputMedia(MediaPrompt)} with the classifier's final probability activation applied or
     * skipped (vLLM's {@code use_activation}); {@code false} answers the raw head output, which is not
     * normalized. The memory must be paged
     * ({@code neuralfree.parallel > 1 or neuralfree.kv.paged=true}); the contiguous form refuses by name.
     *
     * @throws UnsupportedOperationException when the model has neither a sequence classifier nor the pooled decision
     *                                       read, or the memory is not paged
     */
    @Override
    public ClassifierOutput classifyOutputMedia(MediaPrompt prompt, boolean activation) throws IOException {
        requireOpen();
        Objects.requireNonNull(prompt, "prompt");
        if (!EncodeGate.hasDecisionClassifier(spec)) {
            if (DecisionScorer.present(spec)) {
                throw new UnsupportedOperationException(spec.arch().ggufName() + " is a decision model; its"
                        + " decision scores come from classify, not a classifier output");
            }
            if (!EncodeGate.hasSequenceClassifier(spec)) {
                throw new UnsupportedOperationException("a classifier output needs a RANK classifier or another"
                        + " sequence classifier; " + spec.arch().ggufName() + " pools by " + spec.head().pooling());
            }
        }
        return poolingHead().classify(hidden(prompt), activation);
    }

    /**
     * The next-token logits over the whole vocabulary after {@code tokens}, from position 0 with the memory
     * emptied (the generation state is discarded), as vLLM's generative scoring reads them. A paged memory runs
     * the forward as a one-sequence batch, which is then dropped; a contiguous one through {@code prefill}.
     *
     * @throws UnsupportedOperationException when the model does not generate token logits
     */
    @Override
    public float[] nextLogits(int[] tokens) throws IOException {
        requireTokens(tokens);
        if (spec.supportClass() != SupportClass.GEN) {
            throw new UnsupportedOperationException("next-token logits need a generative model; "
                    + spec.arch().ggufName() + " is a " + spec.supportClass() + " model");
        }
        ForwardBackend b = forwardBackend();
        if (b.memory() == null) {
            // a non-causal generative model (an lfm3 d1 checkpoint) evaluates the whole prompt in one step
            return b.forwardAll(tokens, new int[]{tokens.length - 1})[0].clone();
        }
        b.reset();
        if (PagedEvaluations.paged(b)) {
            return PagedEvaluations.nextLogits(b, tokens);
        }
        return b.prefill(tokens, 0).clone();
    }

    /**
     * {@link #nextLogits(int[])} over a media prompt: the prompt's chunks run as consecutive forwards on the paged
     * memory and the last row's logits are read. Any other memory form refuses by name.
     *
     * @throws UnsupportedOperationException when the model does not generate token logits, or the memory is not
     *                                       paged
     */
    @Override
    public float[] nextLogits(MediaPrompt prompt) throws IOException {
        requireOpen();
        Objects.requireNonNull(prompt, "prompt");
        if (spec.supportClass() != SupportClass.GEN) {
            throw new UnsupportedOperationException("next-token logits need a generative model; "
                    + spec.arch().ggufName() + " is a " + spec.supportClass() + " model");
        }
        ForwardBackend b = forwardBackend();
        if (b.memory() != null) {
            b.reset();
        }
        if (PagedEvaluations.paged(b)) {
            return PagedEvaluations.nextLogits(b, prompt);
        }
        if (b.memory() == null) {
            return PagedEvaluations.memorylessNextLogits(b, prompt);
        }
        throw new UnsupportedOperationException("a media prompt needs paged memory (neuralfree.parallel > 1"
                + " or neuralfree.kv.paged=true); " + spec.arch().ggufName() + " holds a contiguous memory");
    }

    /**
     * {@link #classify(int[], int[])} over a media prompt: the hidden rows of the prompt's text and media rows,
     * scored by the decision head. Media rows carry no token id; only the heads that read text spans (clef's
     * lexical term) consult them, and those spans are text.
     *
     * @throws UnsupportedOperationException when the model has no decision head, or the memory is not paged
     */
    @Override
    public float[] classifyMedia(MediaPrompt prompt, int[] order) throws IOException {
        requireOpen();
        Objects.requireNonNull(prompt, "prompt");
        DecisionScorer scorer = decisionScorer();
        if (scorer == null) {
            throw new UnsupportedOperationException("classification needs a decision head; "
                    + spec.arch().ggufName() + " has none");
        }
        ForwardBackend b = forwardBackend();
        if (!PagedEvaluations.paged(b) && b.memory() != null) {
            throw new UnsupportedOperationException("a media prompt needs paged memory (neuralfree.parallel > 1"
                    + " or neuralfree.kv.paged=true); " + spec.arch().ggufName() + " holds a contiguous memory");
        }
        int[] tokens = b.memory() == null ? PagedEvaluations.tokens(prompt) : PagedEvaluations.tokens(b, prompt);
        scorer.checkOrder(tokens.length, order);
        return scorer.score(hidden(prompt), tokens, order);
    }

    /**
     * Relevance of a tokenized {@code document} for a tokenized {@code query}: the first classifier output over the
     * pair packed with the vocabulary's special tokens ({@link ClassifierHead#rerankInput(GGUFTokenizer, int[],
     * int[])}).
     *
     * @throws UnsupportedOperationException when the model has no sequence classifier, or it has a rerank prompt,
     *                                       which needs the text ({@link #rerank(String, String)})
     */
    @Override
    public float rerank(int[] query, int[] document) throws IOException {
        requireOpen();
        EncodeGate.checkRerank(spec);
        return poolingHead().rerank(hidden(ClassifierHead.rerankInput(tokenizer, query, document), null));
    }

    /**
     * Relevance of {@code document} for {@code query}: the first classifier output over the model's rerank input
     * ({@link ClassifierHead#rerankInput(GGUFTokenizer, String, String)}: the {@code tokenizer.chat_template.rerank}
     * prompt when the model has one, else the packed pair), as the llama.cpp server scores it.
     *
     * @throws UnsupportedOperationException when the model has no sequence classifier
     */
    @Override
    public float rerank(String query, String document) throws IOException {
        requireOpen();
        EncodeGate.checkRerank(spec);
        return poolingHead().rerank(hidden(ClassifierHead.rerankInput(tokenizer, query, document), null));
    }

    @Override
    public float[][] decodeAudioCodes(int[] codes) {
        requireOpen();
        if (wavTokenizer == null) {
            throw new UnsupportedOperationException("audio code decoding needs wavtokenizer-dec, not "
                    + spec.arch().ggufName());
        }
        return wavTokenizer.decode(codes);
    }

    @Override
    public SpeechFormat speechFormat() {
        return speechFormat;
    }

    @Override
    public SpeechFormat requireSpeechFormat() {
        requireOpen();
        if (wavTokenizer == null) {
            throw new UnsupportedOperationException("speech synthesis needs wavtokenizer-dec, not "
                    + spec.arch().ggufName());
        }
        if (speechFormat == null) {
            throw new UnsupportedOperationException("the model declares no speech decoding geometry: give the GGUF"
                    + " keys " + SpeechFormat.SAMPLE_RATE_KEY + ", "
                    + SpeechFormat.N_FFT_KEY + " and "
                    + SpeechFormat.HOP_LENGTH_KEY + " (all three) to synthesize"
                    + " waveforms");
        }
        return speechFormat;
    }

    @Override
    public AudioClip synthesizeSpeech(int[] codes) {
        SpeechFormat format = requireSpeechFormat();
        return format.decode(wavTokenizer.decode(codes));
    }

    @Override
    public void synthesizeSpeech(int[] codes, Consumer<AudioClip> sink) {
        SpeechFormat format = requireSpeechFormat();
        format.decode(wavTokenizer.decode(codes), sink);
    }

    /**
     * Final hidden rows from position 0, emptying the memory first (the generation state is discarded); a paged
     * memory runs them as a one-sequence batch ({@link PagedEvaluations#hiddenRows}).
     */
    private float[][] hidden(int[] tokens, int[] tokenTypes) {
        requireTokens(tokens);
        ForwardBackend b = forwardBackend();
        if (b.memory() != null) {
            b.reset();
        }
        if (PagedEvaluations.paged(b)) {
            return PagedEvaluations.hiddenRows(b, PagedEvaluations.untyped(tokens, tokenTypes));
        }
        return b.hiddenRows(tokens, tokenTypes);
    }

    /**
     * {@link #hidden(int[], int[])} over a media prompt: the chunks run as consecutive batch forwards on the paged
     * memory ({@link PagedEvaluations#hiddenRows(ForwardBackend, MediaPrompt)}); a model planned without memory runs
     * the whole prompt as one non-causal batch ({@link PagedEvaluations#memorylessRows}); any other memory form
     * refuses by name, since it has no batch path for embedding rows.
     */
    private float[][] hidden(MediaPrompt prompt) {
        ForwardBackend b = forwardBackend();
        if (b.memory() != null) {
            b.reset();
        }
        if (PagedEvaluations.paged(b)) {
            return PagedEvaluations.hiddenRows(b, prompt);
        }
        if (b.memory() == null) {
            return PagedEvaluations.memorylessRows(b, prompt);
        }
        throw new UnsupportedOperationException("a media prompt needs paged memory (neuralfree.parallel > 1"
                + " or neuralfree.kv.paged=true); " + spec.arch().ggufName() + " holds a contiguous memory");
    }

    /** One vector of every {@code rows} row laid end to end, each {@code width} values. */
    private static float[] flatten(float[][] rows, int width) {
        float[] out = new float[Math.multiplyExact(rows.length, width)];
        for (int t = 0; t < rows.length; t++) {
            System.arraycopy(rows[t], 0, out, t * width, width);
        }
        return out;
    }

    private PoolingHead poolingHead() {
        requireOpen();
        if (poolingHead == null) {
            poolingHead = PoolingHead.of(spec, weights);
        }
        return poolingHead;
    }

    /** The pooling operator over the host heads (vLLM's pooler over a batched step). */
    @Override
    public PoolingOperator poolingOperator() {
        if (poolingOp == null) {
            poolingOp = PoolingOperator.of(spec, this::poolingHead, this::decisionScorer);
        }
        return poolingOp;
    }

    /** The decision head of clef, vllm-decision2, kev and the modern-bert / lfm2 decision models, or null. */
    private DecisionScorer decisionScorer() {
        requireOpen();
        if (decisionScorer == null) {
            decisionScorer = DecisionScorer.forModel(spec, weights, tokenizer);
        }
        return decisionScorer;
    }

    private void requireTokens(int[] tokens) {
        requireOpen();
        Objects.requireNonNull(tokens, "tokens");
        if (tokens.length == 0) {
            throw new IllegalArgumentException("tokens must not be empty");
        }
    }

    // ------------------------------------------------------------------ description

    @Override
    public GGUFTokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public String architecture() {
        return spec.arch().ggufName();
    }

    @Override
    public String name() {
        String n = spec.hparams().name();
        return n != null ? n : architecture();
    }

    @Override
    public int vocabSize() {
        return spec.hparams().nVocab();
    }

    /** The training context ({@code n_ctx_train}). */
    @Override
    public int contextLength() {
        return spec.hparams().nCtxTrain();
    }

    /** The planned context of the memory, or the most tokens one pass takes for models without memory. */
    @Override
    public int kvCacheContext() {
        return plan != null ? plan.nCtx() : context.scratch().ubatch();
    }

    /**
     * Width of an embedding output row: the {@code n_embd_out} of the head projection or the embedding output when
     * the model has one ({@link com.eonmux.neuralfree.gguf.spec.HeadSpec#embeddingWidth}), else {@code n_embd}, as
     * llama.cpp's {@code llama_model_n_embd_out}.
     */
    @Override
    public int embeddingLength() {
        int nEmbd = spec.hparams().nEmbd();
        return spec.head() != null ? spec.head().embeddingWidth(nEmbd) : nEmbd;
    }

    /** Trunk blocks ({@code block_count} less the NextN blocks). */
    @Override
    public int layerCount() {
        return spec.hparams().nLayer();
    }

    @Override
    public int headCount() {
        return spec.hparams().nHead(0);
    }

    @Override
    public int headCountKv() {
        return spec.hparams().nHeadKv(0);
    }

    @Override
    public int feedForwardLength() {
        return spec.hparams().nFf(0);
    }

    /** Epsilon of the block norm of trunk layer 0 (RMSNorm or LayerNorm, as the architecture uses). */
    @Override
    public float rmsNormEps() {
        LayerSpec first = firstLayer();
        return first != null ? first.norm().eps() : spec.hparams().normRmsEps();
    }

    @Override
    public float ropeFreqBase() {
        return spec.hparams().ropeFreqBase(0);
    }

    /** RoPE frequency scale of trunk layer 0. */
    public float ropeFreqScale() {
        return spec.hparams().ropeFreqScale(0);
    }

    @Override
    public int headDim() {
        return spec.hparams().headK(0);
    }

    @Override
    public int kvDim() {
        return spec.hparams().nEmbdKGqa(0);
    }

    private LayerSpec firstLayer() {
        if (!spec.layers().isEmpty()) {
            return spec.layers().get(0);
        }
        if (!spec.encoderLayers().isEmpty()) {
            return spec.encoderLayers().get(0);
        }
        return spec.nextnLayers().isEmpty() ? null : spec.nextnLayers().get(0);
    }

    @Override
    public void printInfo() {
        System.out.println("=== GGUF Model Info ===");
        System.out.println("Architecture: " + architecture());
        System.out.println("Name: " + name());
        System.out.println("Support class: " + supportClass());
        System.out.println("Vocab size: " + vocabSize());
        System.out.println("Context length: " + contextLength() + " (KV cache: " + kvCacheContext() + ")");
        System.out.println("Embedding length: " + embeddingLength());
        System.out.println("Layers: " + layerCount());
        System.out.println("Heads: " + headCount());
        System.out.println("KV Heads: " + headCountKv());
        System.out.println("FFN length: " + feedForwardLength());
        System.out.println("Norm eps: " + rmsNormEps());
        System.out.println("RoPE freq base: " + ropeFreqBase());
        System.out.println("Tensors: " + source.getTensorCount());
        System.out.println("Memory: " + (plan != null ? plan : "none"));
        if (gpuFallbackReason != null) {
            System.out.println("Backend: CPU (" + gpuFallbackReason + ")");
        }
    }

    // ------------------------------------------------------------------ lifecycle

    /** Releases the decoder or backend, the memory, the weights and the source; idempotent. */
    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        Deque<AutoCloseable> owned = new ArrayDeque<>();
        owned.push(weights);
        if (memory != null) {
            owned.push(memory);
        }
        if (backend != null) {
            owned.push(backend);
        }
        if (loras != null) {
            owned.push(loras);
        }
        if (wavTokenizer != null) {
            owned.push(wavTokenizer);
        }
        owned.push(ranking);
        IOException failure = new IOException("closing the " + architecture() + " model failed");
        closeAll(owned, failure);
        closeSource(source, failure);
        if (failure.getSuppressed().length > 0) {
            throw failure;
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("the " + architecture() + " model is closed");
        }
    }

    /** Closes the stack top-down, adding every failure to {@code sink}. */
    private static void closeAll(Deque<AutoCloseable> stack, Throwable sink) {
        while (!stack.isEmpty()) {
            try {
                stack.pop().close();
            } catch (Exception e) {
                sink.addSuppressed(e);
            }
        }
    }

    private static void closeSource(ModelSource source, Throwable sink) {
        try {
            source.close();
        } catch (IOException | RuntimeException e) {
            sink.addSuppressed(e);
        }
    }
}
