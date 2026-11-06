package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.engine.draft.DraftProbs;
import com.eonmux.neuralfree.gguf.engine.watermark.DraftWatermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Batch;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader.Pick;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Rows;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead.Source;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.List;
import java.util.Objects;

/**
 * The vLLM "concat-fc" drafters ({@code EagleMistralLarge3ForCausalLM}, {@code EagleLlama4ForCausalLM},
 * {@code EagleMiniCPMForCausalLM}, {@code EagleCohereForCausalLM}), one runtime for the four families.
 *
 * <p>Graph: {@code g = fc . [embed(token) | target hidden]}, the family's decoder layers over {@code g} as their
 * residual stream, the family's output norm and LM head. Unlike EAGLE-3 there is no {@code attn_norm} /
 * {@code attn_norm_2} split: the fusion projection is the layer input. The families change only what the resolved
 * {@link LayerSpec}s carry (Mistral Large 3's MLA attention and query temperature, Llama 4's chunked local attention
 * and NoPE temperature, MiniCPM's depth-scaled residual) and three scalars:
 * <ul>
 *   <li>{@code embedding_scale} multiplies the token embedding before the fusion (MiniCPM {@code scale_emb});</li>
 *   <li>{@code output_hidden_scale} multiplies the hidden state handed to the LM head and to the next step (MiniCPM
 *       {@code dim_model_base / hidden_size});</li>
 *   <li>{@code logit_scale} multiplies the draft logits (Cohere).</li>
 * </ul>
 * MiniCPM also normalizes the two fusion parts with {@code enc.input_norm1} / {@code enc.input_norm2} before the
 * concatenation, and applies no final norm: vLLM's {@code EagleMiniCPMModel.forward} returns the layer output scaled
 * by {@code scale_width}. The other three apply the draft's output norm; the returned post-norm state is the next
 * step's target hidden, as vLLM's {@code (hidden_states, hidden_states)} return feeds back.
 *
 * <p>The target hidden is vLLM's {@code hidden_states}: the target's post-output-norm final state. The engine taps
 * the final residual stream before its head, so the runtime applies the target's own output norm (kind, epsilon,
 * weight and bias) to the tapped rows on the host, as {@link Gemma4AssistantDraftModel} does.
 *
 * <p>Protocol (positions are the target's): the decoder row at position {@code P} is the pair
 * {@code (token[P + 1], hidden[P])}. A target call over {@code [p0, p0 + m)} rewrites the row at {@code p0 - 1} when
 * the hidden row before the call is known, then the rows {@code p0 .. p0 + m - 2}; its hidden rows become the verify
 * rows and the last one pending at {@code p0 + m - 1}. {@code accept(n)} makes hidden row
 * {@code min(n, m - 1)} pending. A draft starts with {@code (lastToken, pending)} and continues with each drafted
 * token and the previous step's hidden state one position later, dropping the decoder's cache rows from the step's
 * position first. {@link #beginGeneration(int)} keeps a stored prompt prefix's rows and re-evaluates its last one.
 *
 * <p>The drafter has no {@code d2t} of its own unless the checkpoint reduces its vocabulary; a draft head that
 * predicts fewer tokens than the target vocabulary needs one ({@link #mapping()}).
 */
public final class EagleFcDraftModel implements DraftModel, DraftProbs {

    private final MtpHead head;
    private final ModelSpec draft;
    private final ModelSpec target;
    private final MemoryConfig memoryConfig;
    private final DraftModelLoader.Options options;
    private final List<LayerSpec> layers;
    private final Arch arch;
    private final int nEmbd;
    private final int nTarget;
    private final float eps;
    private final NormKind finalKind;
    private final float finalEps;
    private final boolean finalNorm;
    private final float embdScale;
    private final float hiddenScale;
    private final float logitScale;
    private final Source embdSource;
    private final String embdTable;
    private final Source headSource;
    private final String headWeight;
    private final String outputNorm;
    private final String outputNormBias;
    private final String inputNorm1;
    private final String inputNorm2;
    private final EagleFusion fusion;
    private final int targetLayer;
    private final int draftVocab;

    private ForwardBackend backend;
    private DraftModelLoader.TargetFeed feed;
    private DraftModelLoader.HiddenRows hidden;
    private MtpHead.Memory memory;
    private MtpHead.Stack stack;
    private int[] d2t;
    private int chunk;
    private float[] targetNorm;
    private float[] targetNormBias;
    private NormKind targetNormKind;
    private float targetNormEps;
    private Rows e;
    private Rows gIn;
    private Rows cat;
    private Rows x;
    private Rows hn;
    private Rows normBiasRows;
    private float[] normBiasTiled;
    private float[] logits;

    /** The draft-side watermark of the current generation, or null for the ordinary greedy pick. */
    private DraftWatermarker draftWatermark;
    private SplittableRandom draftRng;
    private float draftTemperature = 1f;
    /** The proposal distributions of the last draft's emitted positions, by position (null entries for greedy). */
    private final List<float[]> draftProposals = new ArrayList<>();
    /** The tokens the last draft emitted. */
    private int draftCount;

    /**
     * @throws IllegalArgumentException when the drafter is not a concat-fc family, does not fit the target (hidden
     *                                  size, tap layer, tensors, vocabulary), or a scale is invalid
     */
    EagleFcDraftModel(MtpHead head, ModelSpec draft, MemoryConfig memoryConfig, DraftModelLoader.Options options) {
        this.head = Objects.requireNonNull(head, "head");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.memoryConfig = Objects.requireNonNull(memoryConfig, "memoryConfig");
        this.options = Objects.requireNonNull(options, "options");
        this.target = head.target();
        this.arch = draft.arch();
        if (!isConcatFc(arch)) {
            throw new IllegalArgumentException(arch.ggufName() + " is not a concat-fc drafter");
        }
        ModelHparams hp = draft.hparams();
        List<LayerSpec> all = draft.layers();
        if (all.isEmpty()) {
            throw new IllegalArgumentException(arch.ggufName() + ": the drafter has no decoder layer");
        }
        this.layers = all;

        this.nEmbd = hp.nEmbd();
        this.nTarget = hp.targetHiddenSize();
        if (nTarget != target.hparams().nEmbd()) {
            throw new IllegalArgumentException(arch.ggufName() + ": target_hidden_size is " + nTarget + ", the"
                    + " target " + target.arch().ggufName() + " has n_embd " + target.hparams().nEmbd());
        }
        int[] taps = hp.targetLayers();
        if (taps.length != 1) {
            throw new IllegalArgumentException(arch.ggufName() + ": a concat-fc drafter reads exactly one target"
                    + " layer, got " + Arrays.toString(taps));
        }
        this.targetLayer = taps[0];
        if (targetLayer != target.nLayer()) {
            throw new IllegalArgumentException(arch.ggufName() + ": target layer " + targetLayer + " is not the"
                    + " target's final stream (" + target.nLayer() + "), whose output norm the fusion expects");
        }
        this.eps = hp.normRmsEps() > 0f ? hp.normRmsEps() : hp.normEps();
        this.embdScale = hp.embeddingScale() != 0f ? hp.embeddingScale() : 1f;
        this.hiddenScale = hp.outputHiddenScale() != 0f ? hp.outputHiddenScale() : 1f;
        this.logitScale = hp.logitScale() == 0f ? 1f : hp.logitScale();
        boolean miniCpm = arch == Arch.EAGLE_MINICPM;
        this.finalNorm = !miniCpm;
        if (finalNorm) {
            if (hp.normEps() > 0f) {
                this.finalKind = NormKind.LAYER;
                this.finalEps = hp.normEps();
            } else {
                this.finalKind = all.get(0).norm().kind();
                this.finalEps = all.get(0).norm().eps();
            }
        } else {
            this.finalKind = null;
            this.finalEps = 0f;
        }
        String outputName = TensorNames.weight(TensorNames.OUTPUT);
        String embdName = TensorNames.weight(TensorNames.TOKEN_EMBD);
        if (head.has(Source.DRAFT, embdName)) {
            this.embdSource = Source.DRAFT;
            this.embdTable = embdName;
        } else {
            this.embdSource = Source.TARGET;
            this.embdTable = requiredTarget(target.embedding().tokenEmbd(), "token embeddings (own or the target's)");
        }
        if (head.has(Source.DRAFT, outputName)) {
            this.headSource = Source.DRAFT;
            this.headWeight = outputName;
        } else {
            this.headSource = Source.TARGET;
            this.headWeight = requiredTarget(target.head().output(), "an output projection (own or the target's)");
        }
        this.outputNorm = head.has(Source.DRAFT, TensorNames.weight(TensorNames.OUTPUT_NORM))
                ? TensorNames.weight(TensorNames.OUTPUT_NORM) : null;
        this.outputNormBias = outputNorm != null && head.has(Source.DRAFT, TensorNames.bias(TensorNames.OUTPUT_NORM))
                ? TensorNames.bias(TensorNames.OUTPUT_NORM) : null;
        if (finalNorm && outputNorm == null) {
            throw new IllegalArgumentException(arch.ggufName() + ": the drafter needs "
                    + TensorNames.weight(TensorNames.OUTPUT_NORM));
        }
        this.inputNorm1 = optional(TensorNames.weight(TensorNames.ENC_INPUT_NORM1));
        this.inputNorm2 = optional(TensorNames.weight(TensorNames.ENC_INPUT_NORM2));
        this.fusion = EagleFusion.bind(head, Source.DRAFT, required(TensorNames.weight(TensorNames.FC)),
                optional(TensorNames.bias(TensorNames.FC)), 2 * nEmbd, nEmbd);
        long[] embdShape = head.shape(embdSource, embdTable);
        if (embdShape[0] != nEmbd) {
            throw new IllegalArgumentException(arch.ggufName() + ": " + embdTable + " rows hold " + embdShape[0]
                    + " floats, the decoder n_embd is " + nEmbd);
        }
        this.draftVocab = Math.toIntExact(head.shape(headSource, headWeight)[1]);
    }

    /** True for the four concat-fc drafter architectures. */
    static boolean isConcatFc(Arch arch) {
        return arch == Arch.EAGLE_MISTRAL_LARGE3 || arch == Arch.EAGLE_LLAMA4 || arch == Arch.EAGLE_MINICPM
                || arch == Arch.EAGLE_COHERE;
    }

    private String required(String tensor) {
        if (!head.has(Source.DRAFT, tensor)) {
            throw new IllegalArgumentException(arch.ggufName() + ": the drafter needs " + tensor);
        }
        return tensor;
    }

    private String optional(String tensor) {
        return head.has(Source.DRAFT, tensor) ? tensor : null;
    }

    private String requiredTarget(TensorRef ref, String what) {
        if (ref == null || !head.has(Source.TARGET, ref.name())) {
            throw new IllegalArgumentException(arch.ggufName() + ": the decoder needs " + what);
        }
        return ref.name();
    }

    @Override
    public void bindTarget(EngineModel model) {
        Objects.requireNonNull(model, "target");
        if (backend != null) {
            throw new IllegalStateException("the " + arch.ggufName() + " drafter is already bound");
        }
        ForwardBackend fb = model.forwardBackend();
        ModelMemory tm = fb.memory();
        if (tm == null) {
            throw new IllegalArgumentException(target.arch().ggufName() + ": the target runs without memory");
        }
        this.d2t = mapping();
        MemoryPlan plan = MemoryPlanner.planForContext(draft, memoryConfig, tm.plan().nCtx());
        this.chunk = Math.min(head.ubatch(), plan.ubatch());
        this.memory = head.ownMemory(draft, plan);
        this.stack = head.stack(draft, Source.DRAFT, memory, layers);
        readTargetNorm();
        this.fusion.allocate();
        this.e = head.rows("eagle_fc.e", nEmbd);
        this.gIn = head.rows("eagle_fc.g_in", nEmbd);
        this.cat = head.rows("eagle_fc.cat", 2 * nEmbd);
        this.x = head.rows("eagle_fc.x", nEmbd);
        this.hn = head.rows("eagle_fc.hn", nEmbd);
        if (finalNorm && outputNormBias != null) {
            float[] bias = head.vector(Source.DRAFT, outputNormBias);
            if (bias.length != nEmbd) {
                throw new IllegalArgumentException(outputNormBias + " holds " + bias.length + " values, n_embd is "
                        + nEmbd);
            }
            this.normBiasRows = head.rows("eagle_fc.norm_bias", nEmbd);
            this.normBiasTiled = new float[Math.multiplyExact(head.ubatch(), nEmbd)];
            for (int r = 0; r < head.ubatch(); r++) {
                System.arraycopy(bias, 0, normBiasTiled, r * nEmbd, nEmbd);
            }
        }
        this.logits = new float[draftVocab];
        this.hidden = new DraftModelLoader.HiddenRows(nEmbd);
        this.feed = new DraftModelLoader.TargetFeed(new int[]{targetLayer}, nEmbd, tm);
        fb.setHiddenTap(feed);
        this.backend = fb;
    }

    /**
     * The target's output norm, which turns the tapped final residual stream into vLLM's {@code hidden_states}.
     *
     * @throws UnsupportedOperationException when the target normalizes its output in groups
     */
    private void readTargetNorm() {
        if (target.head().outputNormGroups() > 1) {
            throw new UnsupportedOperationException("the target " + target.arch().ggufName() + " normalizes its"
                    + " output in " + target.head().outputNormGroups() + " groups; a concat-fc drafter reads one"
                    + " hidden row");
        }
        this.targetNormKind = target.layer(0).norm().kind();
        this.targetNormEps = target.layer(0).norm().eps();
        TensorRef weight = target.head().outputNorm();
        TensorRef bias = target.head().outputNormBias();
        if (weight == null && !target.layer(0).norm().weightless()) {
            throw new IllegalArgumentException(target.arch().ggufName() + ": the target has no output norm");
        }
        this.targetNorm = weight != null ? head.vector(Source.TARGET, weight.name()) : null;
        this.targetNormBias = bias != null ? head.vector(Source.TARGET, bias.name()) : null;
        if (targetNorm != null && targetNorm.length != nTarget) {
            throw new IllegalArgumentException(weight + " holds " + targetNorm.length + " values, the target"
                    + " hidden is " + nTarget);
        }
    }

    /** Draft id to target id: {@code d2t} when present, else the identity. */
    private int[] mapping() {
        int targetVocab = Math.toIntExact(target.head().vocabRows());
        TensorRef ref = draft.head().d2t();
        if (ref == null) {
            if (draftVocab != targetVocab) {
                throw new IllegalArgumentException(arch.ggufName() + ": the draft vocabulary has " + draftVocab
                        + " tokens, the target " + targetVocab + ", and there is no d2t table");
            }
            return null;
        }
        int[] map = head.integers(Source.DRAFT, ref.name());
        if (map.length != draftVocab) {
            throw new IllegalArgumentException(arch.ggufName() + ": d2t has " + map.length + " entries for "
                    + draftVocab + " draft tokens");
        }
        for (int i = 0; i < map.length; i++) {
            if (map[i] < 0 || map[i] >= targetVocab) {
                throw new IllegalArgumentException(arch.ggufName() + ": d2t[" + i + "] = " + map[i] + " is outside"
                        + " the target vocabulary of " + targetVocab);
            }
        }
        return map;
    }


    @Override
    public int[] draft(int lastToken, int n) {
        return draft(lastToken, n, 1f);
    }

    @Override
    public int[] draft(int lastToken, int n, float temperature) {
        requireBound();
        if (n < 0) {
            throw new IllegalArgumentException("draft length " + n + " is negative");
        }
        draftProposals.clear();
        draftCount = 0;
        draftTemperature = temperature;
        if (draftWatermark != null && temperature > 0f) {
            draftWatermark.sync(backend.memory().sequence().textTokens(), lastToken);
            draftWatermark.beginRound();
        }
        sync();
        int pos = hidden.pendingPos();
        if (pos < 0 || n == 0) {
            return new int[0];
        }
        int count = Math.min(n, memory.memory().plan().nCtx() - pos);
        if (count <= 0) {
            return new int[0];
        }
        memory.align(pos);
        float[] hRow = hidden.pending().clone();
        int[] out = new int[count];
        int made = 0;
        int[] one = new int[]{lastToken};
        for (int i = 0; i < count; i++) {
            decode(pos + i, one, hRow, 1);
            hiddenState(1);
            head.logits(headSource, headWeight, hn, 0, logits);
            if (logitScale != 1f) {
                scaleLogits(logits, logitScale);
            }
            boolean watermarked = draftWatermark != null && draftTemperature > 0f;
            Pick p;
            if (watermarked) {
                float[] row = d2t == null ? logits : DraftModelLoader.scatter(logits, draftVocab, d2t);
                p = DraftModelLoader.pick(row, row.length, draftWatermark, draftTemperature, draftRng::nextDouble);
            } else {
                p = DraftModelLoader.pick(logits, draftVocab);
            }
            if (p.p < options.pMin()) {
                break;
            }
            int id;
            if (watermarked) {
                draftProposals.add(p.proposal);
                id = p.id;
            } else {
                id = d2t != null ? d2t[p.id] : p.id;
            }
            out[made++] = id;
            one[0] = id;
            if (i + 1 < count) {
                head.readRow(hn, 0, hRow);
            }
        }
        draftCount = made;
        return Arrays.copyOf(out, made);
    }

    /** {@code logits *= scale}: the draft configuration's {@code logit_scale} (vLLM's logits processor). */
    private static void scaleLogits(float[] logits, float scale) {
        for (int i = 0; i < logits.length; i++) {
            logits[i] *= scale;
        }
    }

    @Override
    public boolean supportsWatermarking() {
        return true;
    }

    @Override
    public void watermark(WatermarkConfig config, int promptLength, long seed) {
        this.draftWatermark = DraftWatermarker.of(config);
        if (draftWatermark != null) {
            draftWatermark.reset(promptLength);
        }
        this.draftRng = draftWatermark == null ? null : new SplittableRandom(seed);
    }

    @Override
    public float[] draftProbs(int position) {
        if (position < 0 || position >= draftCount) {
            throw new IllegalStateException("no drafted token at position " + position);
        }
        float[] row = position < draftProposals.size() ? draftProposals.get(position) : null;
        return row == null ? null : row.clone();
    }

    @Override
    public void accept(int nAccepted) {
        requireBound();
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count " + nAccepted + " is negative");
        }
        sync();
        hidden.accept(nAccepted);
    }

    @Override
    public int beginGeneration(int reused) {
        if (reused < 0) {
            throw new IllegalArgumentException("reused prefix length " + reused + " is negative");
        }
        if (backend == null) {
            return 0;
        }
        feed.clear();
        hidden.clear();
        if (reused == 0) {
            memory.reset();
            return 0;
        }
        if (!keepPrefix(reused)) {
            return reused;
        }
        // row P pairs token[P + 1] with hidden[P]: re-evaluating the last stored position rewrites its row against
        // the new first tail token and supplies the hidden row for the row after it
        return 1;
    }

    /** Keeps this drafter's decoder rows of positions {@code [0, reused)}; false when they are gone or cannot drop. */
    private boolean keepPrefix(int reused) {
        ModelMemory m = memory.memory();
        int past = m.sequence().nPast();
        int drop = past - reused;
        if (past < reused || (drop > 0 && !m.canRollback(drop))) {
            memory.reset();
            return false;
        }
        if (drop > 0) {
            m.rollback(drop);
        }
        return true;
    }

    @Override
    public boolean readsTargetHidden() {
        return true;
    }

    @Override
    public void reset() {
        draftProposals.clear();
        draftCount = 0;
        if (backend == null) {
            return;
        }
        feed.clear();
        memory.reset();
        hidden.clear();
    }

    @Override
    public void close() {
        try {
            if (backend != null) {
                backend.setHiddenTap(HiddenTap.NONE);
            }
        } finally {
            backend = null;
            head.close();
        }
    }

    private void requireBound() {
        if (backend == null) {
            throw new IllegalStateException("the " + arch.ggufName() + " drafter is not bound to a target");
        }
    }

    private void sync() {
        for (Batch b : feed.drain()) {
            process(b);
        }
    }

    /**
     * Turns a target call's tapped rows into the target's post-output-norm hidden state and writes the call's
     * decoder rows.
     */
    private void process(Batch b) {
        int m = b.size();
        float[] rows = b.rows[0];
        CpuOps.norm(targetNormKind, targetNormEps, rows, 0, rows, 0, m, nTarget, targetNorm, targetNormBias);
        float[] before = hidden.at(b.pos0 - 1);
        if (before != null) {
            memory.align(b.pos0 - 1);
            decode(b.pos0 - 1, new int[]{b.tokens[0]}, before, 1);
        } else {
            memory.align(b.pos0);
        }
        float[] pairs = new float[chunk * nEmbd];
        for (int k = 0; k < m - 1; k += chunk) {
            int len = Math.min(chunk, m - 1 - k);
            System.arraycopy(rows, k * nEmbd, pairs, 0, len * nEmbd);
            decode(b.pos0 + k, Arrays.copyOfRange(b.tokens, k + 1, k + 1 + len), pairs, len);
        }
        hidden.record(b.pos0, rows, m);
    }

    /** The decoder layers over {@code len} pairs at {@code pos0 ..}, leaving the raw layer output in {@code x}. */
    private void decode(int pos0, int[] tokens, float[] hiddenRows, int len) {
        head.embed(embdSource, embdTable, tokens, len, e);
        if (embdScale != 1f) {
            head.scale(e, embdScale, len);
        }
        if (inputNorm1 != null) {
            head.norm(NormKind.RMS, Source.DRAFT, inputNorm1, eps, e, e, len);
        }
        head.write(gIn, hiddenRows, len);
        if (inputNorm2 != null) {
            head.norm(NormKind.RMS, Source.DRAFT, inputNorm2, eps, gIn, gIn, len);
        }
        head.concat(e, gIn, cat, len);
        fusion.project(cat, x, len);
        stack.run(pos0, len, tokens, x);
    }

    /**
     * The hidden state handed to the LM head and to the next step: the draft's output norm (none for MiniCPM) and
     * {@code output_hidden_scale}, over {@code n} rows.
     */
    private void hiddenState(int n) {
        if (finalNorm) {
            head.norm(finalKind, Source.DRAFT, outputNorm, finalEps, x, hn, n);
            if (normBiasRows != null) {
                head.write(normBiasRows, normBiasTiled, n);
                head.add(hn, normBiasRows, hn, n);
            }
        } else {
            head.copyRows(x, 0, hn, 0, n);
        }
        if (hiddenScale != 1f) {
            head.scale(hn, hiddenScale, n);
        }
    }

    @Override
    public String toString() {
        return "EagleFcDraftModel{" + arch.ggufName() + ", target layer " + targetLayer + ", n_embd " + nEmbd
                + ", draft vocab " + draftVocab + (d2t != null ? ", d2t" : "") + '}';
    }
}
