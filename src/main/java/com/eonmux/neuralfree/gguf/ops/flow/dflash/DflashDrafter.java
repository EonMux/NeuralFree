package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.DraftModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * Block-diffusion speculative drafter for dflash draft models (DFlash, DFlash2, DSpark, LiLiCorr and the
 * DeepSeek-V4 DSpark backbone), following llama.cpp's {@code common_speculative_impl_draft_dflash}
 * (common/speculative.cpp:981-1429) and models/dflash.cpp.
 *
 * <p>Protocol:
 * <ul>
 *   <li>{@link #bindTarget} installs a {@link HiddenTap} on the target that gathers the inputs of the
 *       {@code target_layers}. After every target forward call those rows are fused by the encoder
 *       ({@code rms(fc . features) * enc.output_norm}, dflash.cpp:856-858) and injected into the draft KV cache at
 *       the positions the call appended, {@code nPast - nRows ..} (speculative.cpp process()).</li>
 *   <li>{@link #draft} runs the noise block {@code [lastToken, mask, ...]} at positions {@code nPast ..} (the
 *       position the sampled token will take) and reads the draft: greedy top tokens with the {@code p_min} stop
 *       (DFlash), confidence-truncated Markov-biased tokens (DSpark), or the lattice walk (DFlash2, LiLiCorr);
 *       drafts shorter than {@code n_min} are dropped (speculative.cpp draft()). The block's cache rows are removed
 *       again, as the server's {@code seq_rm} after drafting.</li>
 *   <li>Rows of rejected tokens are dropped by position: every injection and every draft first forgets the cached
 *       positions at or after its start, which equals llama.cpp's {@code seq_rm(pos, -1)} after acceptance, so
 *       {@link #accept} keeps no state (speculative.cpp:1424-1426).</li>
 * </ul>
 * The drafter runs on the CPU over memory-mapped weights; the target's {@code token_embd} and {@code output} are
 * used when the draft has none (dflash.cpp:918-926, 1052-1061). Not thread-safe: the target's forward thread
 * drives the tap and must also call {@link #draft}.
 */
public final class DflashDrafter implements DraftModel {

    /** Rows per encoder pass of an injection, llama.cpp's default draft {@code n_ubatch}. */
    static final int INJECT_CHUNK = 512;

    /**
     * Drafting options ({@code common_params_speculative_draft} n_min / p_min), the draft KV type and whether a
     * quantized draft cache uses the Hadamard rotation ({@code neuralfree.kv.rotation}, llama.cpp's
     * {@code attn_rot}). Immutable.
     */
    public static final class Options {
        private static final Options DEFAULTS = new Options(0, 0.0f, KvDType.F16, true);

        private final int nMin;
        private final float pMin;
        private final KvDType kvType;
        private final boolean kvRotation;

        private Options(int nMin, float pMin, KvDType kvType, boolean kvRotation) {
            this.nMin = nMin;
            this.pMin = pMin;
            this.kvType = kvType;
            this.kvRotation = kvRotation;
        }

        /** {@code n_min = 0}, {@code p_min = 0}, F16 cache, rotation enabled (llama.cpp defaults). */
        public static Options defaults() {
            return DEFAULTS;
        }

        /**
         * Options with the KV rotation enabled.
         *
         * @throws IllegalArgumentException for a negative {@code nMin} or a {@code pMin} outside {@code [0, 1]}
         */
        public static Options of(int nMin, float pMin, KvDType kvType) {
            return of(nMin, pMin, kvType, true);
        }

        /**
         * @param kvRotation rotate the rows of a quantized draft cache (no effect on F16 / F32 caches)
         * @throws IllegalArgumentException for a negative {@code nMin} or a {@code pMin} outside {@code [0, 1]}
         */
        public static Options of(int nMin, float pMin, KvDType kvType, boolean kvRotation) {
            if (nMin < 0) {
                throw new IllegalArgumentException("draft n_min must be non-negative, got " + nMin);
            }
            if (!(pMin >= 0.0f && pMin <= 1.0f)) {
                throw new IllegalArgumentException("draft p_min must be in [0, 1], got " + pMin);
            }
            return new Options(nMin, pMin, Objects.requireNonNull(kvType, "kvType"), kvRotation);
        }

        public int nMin() { return nMin; }
        public float pMin() { return pMin; }
        public KvDType kvType() { return kvType; }
        public boolean kvRotation() { return kvRotation; }

        @Override
        public String toString() {
            return "DflashDrafter.Options{nMin=" + nMin + ", pMin=" + pMin + ", kv=" + kvType
                    + (kvRotation ? "" : ", no rotation") + '}';
        }
    }

    private final DflashSpec spec;
    private final ModelHparams hp;
    private final CpuWeights draft;
    private final boolean ownsDraft;
    private final Options options;
    private final DflashMatrix tokEmbd;
    private final DflashMatrix output;
    private final DflashMatrix fc;
    private final float[] encNorm;
    private final int targetEmbd;
    private final int[] d2t;
    private final DflashMarkovHead markov;
    private final DflashMatrix selectorPrev;
    private final DflashMatrix selectorNext;
    private final DflashMatrix selectorHidden;
    private final DflashLilicorr lilicorr;
    private final float[] lilicorrRows;
    private final int[] lilicorrPos;

    private ForwardBackend backend;
    private DflashKvCache cache;
    private DflashStack stack;
    private DflashTargetFeatures features;

    /**
     * Drafter over an opened draft model.
     *
     * @param draft     the draft model's weights
     * @param ownsDraft close {@code draft} with this drafter
     * @param target    the target model's weights, for the shared {@code token_embd} / {@code output}; may be
     *                  null when the draft carries both
     * @throws IllegalArgumentException when the draft is not a supported dflash model or needs a missing tensor
     */
    public DflashDrafter(ModelHparams hp, GgufKv kv, CpuWeights draft, boolean ownsDraft, CpuWeights target,
                         Options options) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.ownsDraft = ownsDraft;
        this.options = Objects.requireNonNull(options, "options");
        this.spec = DflashSpec.of(hp, kv, draft);
        final int nEmbd = spec.nEmbd();

        this.tokEmbd = firstOf(TensorNames.weight(TensorNames.TOKEN_EMBD), draft, target, null);
        if (tokEmbd.cols() != nEmbd || tokEmbd.rows() < spec.nVocab()) {
            throw new IllegalArgumentException("dflash drafter: token embeddings " + tokEmbd + " do not cover "
                    + spec.nVocab() + " tokens of width " + nEmbd);
        }
        this.output = firstOf(TensorNames.weight(TensorNames.OUTPUT), draft, target,
                TensorNames.weight(TensorNames.TOKEN_EMBD)).expect(spec.nVocabDraft(), nEmbd);
        this.fc = DflashMatrix.require(draft, TensorNames.weight(TensorNames.FC));
        if (fc.rows() != nEmbd || fc.cols() % spec.nTargets() != 0) {
            throw new IllegalArgumentException("dflash drafter: " + fc + " does not map " + spec.nTargets()
                    + " target layers to width " + nEmbd);
        }
        this.targetEmbd = fc.cols() / spec.nTargets();
        this.encNorm = DflashBackbone.vector(draft.floats(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM)), nEmbd,
                TensorNames.ENC_OUTPUT_NORM);
        this.d2t = spec.hasD2t() ? readD2t(draft, spec.nVocab()) : null;

        if (spec.dspark() && options.pMin() > 0.0f && !spec.confidenceHeadMeta()) {
            throw new IllegalArgumentException("DSpark draft has no confidence head: please set the draft p_min to 0");
        }
        this.markov = spec.dspark() ? new DflashMarkovHead(spec, draft, d2t) : null;
        if (markov != null && options.pMin() > 0.0f && !markov.hasConfidence()) {
            throw new IllegalArgumentException("DSpark draft declares a confidence head but has no "
                    + TensorNames.weight(TensorNames.DSPARK_CONF_PROJ) + ": please set the draft p_min to 0");
        }
        if (spec.dflash2()) {
            final int rank = spec.selectorRank();
            this.selectorPrev = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DFLASH_SELECTOR_PREV))
                    .expect(spec.nVocab(), rank);
            this.selectorNext = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DFLASH_SELECTOR_NEXT))
                    .expect(spec.nVocab(), rank);
            this.selectorHidden = DflashMatrix.require(draft, TensorNames.weight(TensorNames.DFLASH_SELECTOR_HIDDEN))
                    .expect(rank, nEmbd);
        } else {
            this.selectorPrev = null;
            this.selectorNext = null;
            this.selectorHidden = null;
        }
        if (spec.lilicorr()) {
            this.lilicorr = new DflashLilicorr(spec, draft, tokEmbd);
            this.lilicorrRows = new float[(spec.blockSize() + 1) * nEmbd];
            this.lilicorrPos = new int[spec.blockSize() + 1];
            Arrays.fill(lilicorrPos, -1);
        } else {
            this.lilicorr = null;
            this.lilicorrRows = null;
            this.lilicorrPos = null;
        }
    }

    /**
     * Opens a draft GGUF; the drafter owns and closes its weights.
     *
     * @param target the target model's weights for the shared tensors, or null when the draft carries them
     * @throws IOException              when the draft's tensor data cannot be mapped
     * @throws IllegalArgumentException when the file is not a supported dflash draft
     */
    public static DflashDrafter open(GGUFFileParser draftParser, CpuWeights target, Options options)
            throws IOException {
        Objects.requireNonNull(draftParser, "draftParser");
        GgufKv kv = GgufKv.of(draftParser);
        ModelHparams hp = HparamsLoader.load(kv);
        CpuWeights weights = CpuWeights.open(draftParser);
        try {
            return new DflashDrafter(hp, kv, weights, true, target, options);
        } catch (RuntimeException e) {
            try {
                weights.close();
            } catch (IOException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
    }

    /** The draft model's description. */
    public DflashSpec spec() {
        return spec;
    }

    /**
     * The spec of a dflash draft GGUF, read without binding a drafter; the draft's tensors are mapped only while it
     * is read.
     *
     * @throws IOException              when the draft's tensor data cannot be mapped
     * @throws IllegalArgumentException when the file is not a supported dflash draft
     */
    public static DflashSpec readSpec(GGUFFileParser draftParser) throws IOException {
        Objects.requireNonNull(draftParser, "draftParser");
        GgufKv kv = GgufKv.of(draftParser);
        ModelHparams hp = HparamsLoader.load(kv);
        try (CpuWeights weights = CpuWeights.open(draftParser)) {
            return DflashSpec.of(hp, kv, weights);
        }
    }

    /**
     * Bytes of the host KV cache a drafter of {@code spec} allocates when it binds to a target with a context of
     * {@code nCtx} tokens and stores its rows as {@code kvType}: the context plus one block per full-attention
     * layer, a ring per sliding-window layer.
     *
     * @throws IllegalArgumentException when the context is not positive
     */
    public static long cacheBytes(DflashSpec spec, int nCtx, KvDType kvType) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(kvType, "kvType");
        if (nCtx <= 0) {
            throw new IllegalArgumentException("dflash cache of a context of " + nCtx + " tokens");
        }
        return DflashKvCache.bytes(cacheCapacity(spec, nCtx), spec.blockSize(), spec.nSwa(), swaLayers(spec),
                spec.dsv4(), kvHeads(spec), spec.headDim(), kvType);
    }

    /** Positions a full-attention layer of the cache holds: the target context plus one noise block. */
    private static int cacheCapacity(DflashSpec spec, int nCtx) {
        return Math.addExact(nCtx, spec.blockSize());
    }

    private static boolean[] swaLayers(DflashSpec spec) {
        boolean[] swa = new boolean[spec.nLayer()];
        for (int il = 0; il < swa.length; il++) {
            swa[il] = spec.isSwa(il);
        }
        return swa;
    }

    private static int[] kvHeads(DflashSpec spec) {
        int[] heads = new int[spec.nLayer()];
        for (int il = 0; il < heads.length; il++) {
            heads[il] = spec.nHeadKv(il);
        }
        return heads;
    }

    // ------------------------------------------------------------------ DraftModel

    @Override
    public void bindTarget(EngineModel target) {
        Objects.requireNonNull(target, "target");
        if (backend != null) {
            throw new IllegalStateException("dflash drafter is already bound to a target");
        }
        ModelHparams thp = target.spec().hparams();
        if (thp.nEmbd() != targetEmbd) {
            throw new IllegalArgumentException("dflash draft fc expects target width " + targetEmbd + ", target "
                    + thp.arch().ggufName() + " has " + thp.nEmbd());
        }
        if (thp.nVocab() != spec.nVocab()) {
            throw new IllegalArgumentException("dflash draft vocabulary " + spec.nVocab() + " differs from the target's "
                    + thp.nVocab());
        }
        for (int layer : spec.targetLayers()) {
            if (layer < 0 || layer > thp.nLayer()) {
                throw new IllegalArgumentException("dflash target layer " + layer + " is outside the target's tap points"
                        + " [0, " + thp.nLayer() + "]");
            }
        }
        ForwardBackend fb = Objects.requireNonNull(target.forwardBackend(), "target forward backend");
        final int nCtx = fb.memory().sequence().nCtx();
        DflashKvCache kvCache = new DflashKvCache(cacheCapacity(spec, nCtx), spec.blockSize(), spec.nSwa(),
                swaLayers(spec), spec.causal(), spec.dsv4(), kvHeads(spec), spec.headDim(), options.kvType(),
                options.kvRotation());
        this.stack = spec.dsv4()
                ? new DflashDsv4Backbone(spec, hp.ropeType(), draft, tokEmbd, output, kvCache)
                : new DflashBackbone(spec, hp, draft, tokEmbd, output, kvCache, nCtx);
        this.cache = kvCache;
        this.features = new DflashTargetFeatures(spec.targetLayers(), targetEmbd, this::inject);
        fb.setHiddenTap(features);
        this.backend = fb;
    }

    @Override
    public int[] draft(int lastToken, int n) {
        requireBound();
        if (lastToken < 0 || lastToken >= spec.nVocab()) {
            throw new IllegalArgumentException("token " + lastToken + " is outside the vocabulary of "
                    + spec.nVocab());
        }
        final int nMax = Math.min(n, spec.nDraftMax());
        final int nMin = Math.min(options.nMin(), spec.nDraftMax());
        if (nMax <= 0) {
            return new int[0];
        }
        final int pos0 = backend.memory().sequence().nPast();
        final int nBlock = spec.blockTokens(nMax);
        if (nBlock <= 0 || !cache.fits(pos0, nBlock) || (lilicorr != null && pos0 < 1)) {
            return new int[0];
        }
        int[] tokens = new int[nBlock];
        Arrays.fill(tokens, spec.maskTokenId());
        tokens[0] = lastToken;

        cache.trim(pos0);
        DflashStack.Block block;
        try {
            block = stack.forward(tokens, pos0);
        } finally {
            cache.trim(pos0);
        }
        float[] logits = targetLogits(block);
        float[] conf = markov != null ? markov.apply(logits, block.tEmbd(), nBlock, lastToken) : null;

        int[] result;
        if (spec.dflash2()) {
            DflashSelector.Lattice lattice = DflashSelector.dflash2(logits, spec.nVocab(), block.tEmbd(),
                    spec.nEmbd(), nBlock, lastToken, spec.selectorTopK(), spec.graphBlockSize(), selectorPrev,
                    selectorNext, selectorHidden);
            result = DflashSelector.walk(lattice, Math.min(nBlock, lattice.nRows()) - 1, options.pMin());
        } else if (lilicorr != null) {
            DflashSelector.Lattice lattice = lilicorr.score(logits, block.tEmbd(), lilicorrAnchor(pos0 - 1));
            result = DflashSelector.walk(lattice, nMax, options.pMin());
        } else if (spec.dspark()) {
            result = readDspark(logits, options.pMin() > 0.0f ? conf : null, nBlock);
        } else {
            result = readGreedy(logits, nBlock);
        }
        return result.length < nMin ? new int[0] : result;
    }

    /**
     * Records verification of the last draft. The cache needs no update here: rows of rejected tokens lie at or
     * after the target's next position and are dropped by the next injection or draft.
     *
     * @throws IllegalArgumentException for a negative count
     */
    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must be non-negative, got " + nAccepted);
        }
    }

    @Override
    public void reset() {
        if (features != null) {
            features.discard();
        }
        if (cache != null) {
            cache.clear();
        }
        if (lilicorrPos != null) {
            Arrays.fill(lilicorrPos, -1);
            Arrays.fill(lilicorrRows, 0.0f);
        }
    }

    @Override
    public void close() {
        if (backend != null) {
            backend.setHiddenTap(HiddenTap.NONE);
            backend = null;
        }
        if (ownsDraft) {
            try {
                draft.close();
            } catch (IOException e) {
                throw new UncheckedIOException("failed to close the dflash draft weights", e);
            }
        }
    }

    // ------------------------------------------------------------------ injection

    /** {@link DflashTargetFeatures.Sink}: fuses and injects the rows of one target forward call. */
    private void inject(float[] feats, int nRows) {
        SequenceState seq = backend.memory().sequence();
        final int pos0 = seq.nPast() - nRows;
        if (pos0 < 0 || !cache.fits(pos0, nRows)) {
            throw new IllegalStateException("dflash drafter: " + nRows + " target rows do not end at the target's"
                    + " position " + seq.nPast() + " within the draft cache of " + cache.capacity() + " positions");
        }
        cache.trim(pos0);
        final int nEmbd = spec.nEmbd();
        final int width = features.rowWidth();
        for (int off = 0; off < nRows; off += INJECT_CHUNK) {
            final int n = Math.min(INJECT_CHUNK, nRows - off);
            float[] g = new float[n * nEmbd];
            fc.matmul(feats, off * width, g, 0, n);
            CpuOps.rmsNorm(g, 0, g, 0, n, nEmbd, encNorm, spec.rmsEps());
            stack.inject(g, n, pos0 + off);
            if (lilicorr != null) {
                keepLilicorrRows(g, n, pos0 + off);
            }
        }
    }

    /**
     * Keeps the fused rows LiLiCorr reads as its anchor: rows of the last {@code block_size + 1} positions of the
     * chunk, in slot {@code pos % (block_size + 1)} (speculative.cpp:1229-1245).
     */
    private void keepLilicorrRows(float[] g, int n, int pos0) {
        final int nEmbd = spec.nEmbd();
        final int slots = lilicorrPos.length;
        final int posMin = Math.max(0, pos0 + n - 1 - spec.blockSize());
        for (int t = 0; t < n; t++) {
            final int pos = pos0 + t;
            if (pos < posMin) {
                continue;
            }
            final int slot = pos % slots;
            System.arraycopy(g, t * nEmbd, lilicorrRows, slot * nEmbd, nEmbd);
            lilicorrPos[slot] = pos;
        }
    }

    /** The stored fused row of slot {@code pos % (block_size + 1)} (speculative.cpp:1257-1262). */
    private float[] lilicorrAnchor(int pos) {
        final int nEmbd = spec.nEmbd();
        final int slot = pos % lilicorrPos.length;
        return Arrays.copyOfRange(lilicorrRows, slot * nEmbd, (slot + 1) * nEmbd);
    }

    // ------------------------------------------------------------------ readout

    /**
     * Logits of every block row in the target vocabulary: for DFlash2 first {@code * f_logit_scale} and the final
     * softcap (dflash.cpp:1065-1076), then the {@code d2t} scatter with {@code -inf} elsewhere (dflash.cpp:1078-1092).
     */
    private float[] targetLogits(DflashStack.Block block) {
        float[] head = block.logits();
        if (spec.dflash2()) {
            if (spec.logitScale() != 0.0f) {
                for (int i = 0; i < head.length; i++) {
                    head[i] *= spec.logitScale();
                }
            }
            final float cap = spec.finalSoftcap();
            if (cap > 0.0f) {
                for (int i = 0; i < head.length; i++) {
                    head[i] = (float) Math.tanh(head[i] * (1.0f / cap)) * cap;
                }
            }
        }
        if (d2t == null) {
            return head;
        }
        final int nVocab = spec.nVocab();
        final int rows = block.vocab();
        float[] full = new float[block.nTokens() * nVocab];
        Arrays.fill(full, Float.NEGATIVE_INFINITY);
        for (int t = 0; t < block.nTokens(); t++) {
            for (int r = 0; r < rows; r++) {
                full[t * nVocab + d2t[r]] = head[t * rows + r];
            }
        }
        return full;
    }

    /**
     * DFlash: the top token of block positions {@code 1 ..}, stopping when its probability among the
     * {@value DflashSpec#SAMPLER_TOP_K} best candidates is below {@code p_min} (speculative.cpp:1386-1414).
     */
    private int[] readGreedy(float[] logits, int nBlock) {
        final int nVocab = spec.nVocab();
        int[] out = new int[nBlock - 1];
        int n = 0;
        for (int i = 1; i < nBlock; i++) {
            final int off = i * nVocab;
            final int top = CpuOps.argmax(logits, off, nVocab);
            if (options.pMin() > 0.0f && topProbability(logits, off, nVocab) < options.pMin()) {
                break;
            }
            out[n++] = top;
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * DSpark: the top token of positions {@code 0 ..} (anchor-first) or {@code 1 ..}, stopping at the first
     * confidence below {@code p_min} (speculative.cpp:1351-1384).
     */
    private int[] readDspark(float[] logits, float[] conf, int nBlock) {
        final int nVocab = spec.nVocab();
        final int iBeg = spec.sampleFromAnchor() ? 0 : 1;
        int[] out = new int[Math.max(0, nBlock - iBeg)];
        int n = 0;
        for (int i = iBeg; i < nBlock; i++) {
            if (conf != null && conf[i] < options.pMin()) {
                break;
            }
            out[n++] = CpuOps.argmax(logits, i * nVocab, nVocab);
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * Probability of the best candidate after {@code top_k(10)} and the distribution sampler's softmax:
     * {@code 1 / sum_k exp(l_k - l_max)} over the ten largest logits.
     */
    static float topProbability(float[] logits, int off, int nVocab) {
        final int k = Math.min(DflashSpec.SAMPLER_TOP_K, nVocab);
        int[] idx = new int[k];
        CpuOps.topK(logits, off, nVocab, k, idx, 0);
        final float max = logits[off + idx[0]];
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            sum += (float) Math.exp(logits[off + idx[j]] - max);
        }
        return 1.0f / sum;
    }

    // ------------------------------------------------------------------ setup helpers

    private void requireBound() {
        if (backend == null) {
            throw new IllegalStateException("dflash drafter is not bound to a target (bindTarget)");
        }
    }

    /**
     * The tensor {@code name} of the draft, else of the target, else {@code fallback} of the target.
     *
     * @throws IllegalArgumentException when none exists
     */
    private static DflashMatrix firstOf(String name, CpuWeights draft, CpuWeights target, String fallback) {
        DflashMatrix m = DflashMatrix.orNull(draft, name);
        if (m == null) {
            m = DflashMatrix.orNull(target, name);
        }
        if (m == null && fallback != null) {
            m = DflashMatrix.orNull(target, fallback);
        }
        if (m == null) {
            throw new IllegalArgumentException("dflash drafter needs '" + name + "' from the draft or the target"
                    + (target == null ? " (no target weights were given)" : ""));
        }
        return m;
    }

    /**
     * {@code d2t}: the target id of each draft-head row, an I64 vector (dflash.cpp:128-135, 1084).
     *
     * @throws IllegalArgumentException for another element type or an id outside the vocabulary
     */
    private static int[] readD2t(CpuWeights draft, int nVocab) {
        CpuTensorRef t = draft.get(TensorNames.D2T);
        if (t.type() != GGMLType.I64) {
            throw new IllegalArgumentException("dflash d2t must be I64, got " + t.type());
        }
        final long count = t.rows() * t.cols();
        if (count > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("dflash d2t holds " + count + " entries");
        }
        int[] ids = new int[(int) count];
        int i = 0;
        for (long r = 0; r < t.rows(); r++) {
            ByteBuffer b = t.rowBuffer(r);
            final int base = t.rowOffset(r);
            for (long c = 0; c < t.cols(); c++) {
                final long id = b.getLong(base + (int) (8 * c));
                if (id < 0 || id >= nVocab) {
                    throw new IllegalArgumentException("dflash d2t entry " + i + " maps to token " + id
                            + " outside the vocabulary of " + nVocab);
                }
                ids[i++] = (int) id;
            }
        }
        return ids;
    }

    @Override
    public String toString() {
        return "DflashDrafter{" + spec + ", " + options + (backend != null ? ", bound" : "") + '}';
    }
}
