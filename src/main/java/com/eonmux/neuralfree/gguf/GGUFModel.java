package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.block.CpuBlockRegistry;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardBackend;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.DiffusionGenerator;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.audio.CpuWavTokenizer;
import com.eonmux.neuralfree.gguf.ops.audio.TtsBackbone;
import com.eonmux.neuralfree.gguf.ops.encdec.CpuEncDecBackend;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionBody;
import com.eonmux.neuralfree.gguf.ops.heads.DecisionScorer;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.ops.heads.ModernBertDecisionHead;
import com.eonmux.neuralfree.gguf.ops.heads.PoolingHead;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAttachment;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraBackend;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;

/**
 * A GGUF model run on the CPU by the spec-driven engine.
 *
 * <p>Construction: parser, tokenizer, {@link GgufKv} and {@link TensorCatalog}, {@link ModelSpecBuilder}; the
 * memory settings the model runs with ({@link MemoryConfig#forModel}: models that evaluate their whole input in one
 * step get a ubatch of their whole context, models without memory at most their context); the {@link CpuWeights}
 * read in place from the mapped file; for models
 * with memory the context {@link ContextFitter} chooses within the Java heap, beside the NextN / MTP draft caches and
 * a configured draft model's host memory ({@link LLMModelFactory#draftCompanion}), its {@link MemoryPlan}
 * ({@link MemoryPlanner}) and the {@link HeapModelMemory}; the {@link CpuScratch} of the plan's ubatch (of the
 * settings' ubatch without memory), the
 * {@link CpuForwardContext}, the {@link CpuBlockRegistry} of {@link CpuFactoryCatalog} and the backend:
 * {@link CpuEncDecBackend} for encoder-decoder models, {@link CpuForwardBackend} otherwise (wrapped for the
 * graniteswitch adapter router and the qwen3tts logits layout), or the {@link CpuWavTokenizer} decoder for
 * wavtokenizer-dec.</p>
 *
 * <p>Use: autoregressive generation through {@link GenerationLoop} (the legacy {@code generate} overloads map onto
 * {@link GenerationRequest}); embeddings, classification and reranking through {@link ForwardBackend#hiddenRows}
 * and the heads; masked diffusion through {@link DiffusionGenerator}; audio codes through the wavtokenizer decoder.
 * The scalar getters describe trunk layer 0. Not thread-safe. {@link #close()} releases the backend, the memory,
 * the weights and the parser.</p>
 */
public final class GGUFModel implements EngineModel {

    private static final Logger LOGGER = LogManager.getLogger(GGUFModel.class);

    /**
     * Share of the free heap kept out of the memory budget for the temporaries the blocks reserve when they bind
     * and for the per-call results.
     */
    private static final double HEAP_HEADROOM_FRACTION = 0.125;
    /** Smallest heap headroom kept out of the memory budget. */
    private static final long MIN_HEAP_HEADROOM_BYTES = 64L << 20;

    private final GGUFFileParser parser;
    private final GGUFTokenizer tokenizer;
    private final EngineConfig config;
    private final ModelSpec spec;
    private final CpuWeights weights;
    private final MemoryPlan plan;
    private final HeapModelMemory memory;
    private final CpuForwardContext context;
    private final CpuBlockRegistry registry;
    private final ForwardBackend backend;
    private final CpuWavTokenizer wavTokenizer;
    private final LoraAttachment loras;

    private GenerationLoop generationLoop;
    private PoolingHead poolingHead;
    private DecisionScorer decisionScorer;
    private boolean closed;

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
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(config, "config");
        this.parser = new GGUFFileParser(path);
        Deque<AutoCloseable> opened = new ArrayDeque<>();
        try {
            this.tokenizer = new GGUFTokenizer(parser);
            GgufKv kv = GgufKv.of(parser);
            TensorCatalog tc = TensorCatalog.of(parser);
            ModelSpec resolved = ModelSpecBuilder.build(kv, tc, config.memory());
            rejectStandalone(resolved);
            MemoryConfig mc = config.memory().forModel(resolved);

            this.weights = CpuWeights.open(parser);
            opened.push(weights);
            CpuScratch scratch = CpuScratch.forModel(resolved, mc.ubatch());

            if (resolved.memoryKind() != MemoryKind.NONE) {
                ContextFitter.Companion drafter =
                        LLMModelFactory.draftCompanion(config, resolved, true, (draft, placed) -> 0L);
                int nCtx = ContextFitter.fit(resolved, mc, heapBudget(), drafter);
                int specCtx = mc.autoContext() ? resolved.hparams().nCtxTrain() : mc.nCtx();
                if (nCtx != specCtx) {
                    resolved = ModelSpecBuilder.build(kv, tc, mc, nCtx);
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
            CpuForwardContext.Builder cb = CpuForwardContext.builder(runSpec, weights, scratch);
            if (memory != null) {
                cb.memory(memory);
            }
            this.context = cb.build();

            if (resolved.arch() == Arch.WAVTOKENIZER_DEC) {
                if (!config.loras().isEmpty()) {
                    throw new UnsupportedOperationException(resolved.arch().ggufName() + " has no token forward pass"
                            + " to apply LoRA adapters to");
                }
                this.registry = null;
                this.backend = null;
                this.wavTokenizer = CpuWavTokenizer.bind(context);
                this.loras = null;
            } else {
                CpuFactoryCatalog.checkRouting(runSpec);
                SwitchedLoraExtra.Router router = CpuFactoryCatalog.usesSwitchedLora(runSpec)
                        ? SwitchedLoraExtra.Router.of(resolved.hparams()) : null;
                this.registry = CpuFactoryCatalog.defaults(router);
                registry.validate(runSpec);
                this.backend = openBackend(context, registry, router, tokenizer);
                opened.push(backend);
                this.wavTokenizer = null;
                this.loras = attachLoras(config, weights, context);
                if (loras != null) {
                    opened.push(loras);
                }
            }
        } catch (IOException | RuntimeException | Error e) {
            closeAll(opened, e);
            closeParser(parser, e);
            throw e;
        }
        LOGGER.info("{} loaded on the CPU: {} ({})", spec.arch().ggufName(), spec.supportClass(),
                plan != null ? plan : "no memory");
    }

    // ------------------------------------------------------------------ construction

    /**
     * Attaches the configured LoRA adapters ({@link EngineConfig#loras()}, llama.cpp {@code --lora} /
     * {@code --lora-scaled}) to the bound blocks of {@code ctx}; null when none is configured.
     *
     * @throws IOException                   when an adapter file cannot be read
     * @throws UnsupportedOperationException for a refused adapter or adapted product
     * @throws IllegalStateException         when an adapter tensor targets a product the model does not run
     */
    private static LoraAttachment attachLoras(EngineConfig config, CpuWeights weights, CpuForwardContext ctx)
            throws IOException {
        List<EngineConfig.LoraSpec> specs = config.loras();
        if (specs.isEmpty()) {
            return null;
        }
        List<LoraAttachment.Binding> bindings = new ArrayList<>(specs.size());
        for (EngineConfig.LoraSpec s : specs) {
            bindings.add(LoraAttachment.Binding.load(s.path(), s.scale(), weights));
        }
        LoraAttachment att = LoraAttachment.cpu(bindings, ctx.projectionDeltas());
        LOGGER.info("{}: {} LoRA adapter(s) attached, {} deltas: {}", ctx.spec().arch().ggufName(), bindings.size(),
                att.deltaCount(), bindings);
        return att;
    }

    /** Draft-only architectures run beside a target model, never on their own. */
    private static void rejectStandalone(ModelSpec spec) {
        if (spec.supportClass() == SupportClass.DRAFT) {
            throw new UnsupportedOperationException(spec.arch().ggufName() + " is a draft model; load it beside its"
                    + " target model as a speculative drafter (DraftModelLoader)");
        }
    }

    /**
     * Bytes the memory plan may take from the Java heap: the free heap less the headroom, or
     * {@link ContextFitter#UNLIMITED} when the heap has no maximum.
     */
    private static long heapBudget() {
        Runtime rt = Runtime.getRuntime();
        long max = rt.maxMemory();
        if (max == Long.MAX_VALUE) {
            return ContextFitter.UNLIMITED;
        }
        long free = max - (rt.totalMemory() - rt.freeMemory());
        long headroom = Math.max(MIN_HEAP_HEADROOM_BYTES, (long) (free * HEAP_HEADROOM_FRACTION));
        return Math.max(0L, free - headroom);
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

    // ------------------------------------------------------------------ generation

    @Override
    public GenerationResult generate(int[] prompt, GenerationRequest request, TokenCallback callback) {
        requireOpen();
        return loop().generate(prompt, request, callback);
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
        float[][] rows = head.poolTokens(hidden);
        int width = head.outputWidth();
        float[] out = new float[Math.multiplyExact(rows.length, width)];
        for (int t = 0; t < rows.length; t++) {
            System.arraycopy(rows[t], 0, out, t * width, width);
        }
        return out;
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
        return poolingHead().poolTokens(hidden(tokens, null));
    }

    @Override
    public float[] classify(int[] tokens) throws IOException {
        return classify(tokens, null);
    }

    /**
     * Classifier outputs of one input:
     * <ul>
     *   <li>RANK pooling: the classifier scores, one per {@link #classLabels()} entry;</li>
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
        PoolingKind pooling = spec.head().pooling();
        if (pooling != PoolingKind.RANK) {
            throw new UnsupportedOperationException("classification needs a RANK classifier or a decision head; "
                    + spec.arch().ggufName() + " pools by " + pooling);
        }
        return poolingHead().classify(hidden(tokens, null)).scores();
    }

    @Override
    public String[] classLabels() {
        requireOpen();
        if (spec.head().pooling() != PoolingKind.RANK || DecisionScorer.present(spec)) {
            return new String[0];
        }
        return poolingHead().classifier().labels();
    }

    /**
     * Relevance of a tokenized {@code document} for a tokenized {@code query}: the first classifier output over the
     * pair packed with the vocabulary's special tokens ({@link ClassifierHead#rerankInput(GGUFTokenizer, int[],
     * int[])}).
     *
     * @throws UnsupportedOperationException when the model's pooling is not RANK, or the model has a rerank prompt,
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
     * @throws UnsupportedOperationException when the model's pooling is not RANK
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

    /** Final hidden rows from position 0, emptying the memory first (the generation state is discarded). */
    private float[][] hidden(int[] tokens, int[] tokenTypes) {
        requireTokens(tokens);
        ForwardBackend b = forwardBackend();
        if (b.memory() != null) {
            b.reset();
        }
        return b.hiddenRows(tokens, tokenTypes);
    }

    private PoolingHead poolingHead() {
        requireOpen();
        if (poolingHead == null) {
            poolingHead = PoolingHead.of(spec, weights);
        }
        return poolingHead;
    }

    /** The decision head of clef, vllm-decision2 and the modern-bert / lfm2 decision models, or null. */
    private DecisionScorer decisionScorer() {
        requireOpen();
        if (decisionScorer == null) {
            decisionScorer = DecisionScorer.forModel(spec, weights);
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
        System.out.println("Tensors: " + parser.getTensorCount());
        System.out.println("Memory: " + (plan != null ? plan : "none"));
    }

    // ------------------------------------------------------------------ lifecycle

    /** Releases the decoder or backend, the memory, the weights and the parser; idempotent. */
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
        IOException failure = new IOException("closing the " + architecture() + " model failed");
        closeAll(owned, failure);
        closeParser(parser, failure);
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

    private static void closeParser(GGUFFileParser parser, Throwable sink) {
        try {
            parser.close();
        } catch (IOException | RuntimeException e) {
            sink.addSuppressed(e);
        }
    }
}
