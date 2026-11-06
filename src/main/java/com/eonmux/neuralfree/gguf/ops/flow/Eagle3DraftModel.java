package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
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
import java.util.Objects;

/**
 * The EAGLE-3 drafter (llama.cpp src/models/eagle3.cpp), driven like llama.cpp common/speculative.cpp
 * {@code common_speculative_impl_draft_eagle3} (515-982).
 *
 * <p>Encoder (eagle3.cpp:122-141): the target's hidden states at its three {@code target_layers} tap points,
 * concatenated in that order, optionally RMS-normed as a whole by {@code enc.output_norm} ({@code norm_before_fc}),
 * then each tap RMS-normed by its own {@code fc_norm} weight when the drafter keeps them, then
 * {@code g = fc . features}. The two norms follow vLLM's order ({@code combine_hidden_states},
 * llama_eagle3.py:405-418): the whole-input norm first, then the per-tap norms. Decoder (eagle3.cpp:146-300), one
 * layer over caches of its own, for the pair {@code (tok, g)}: {@code e = attn_norm(embed(tok))},
 * {@code gn = attn_norm_2(g)}; attention
 * over {@code [e | gn]}; {@code ffn_inp = attn + (norm_before_residual ? gn : g)}; {@code h = ffn(ffn_norm(ffn_inp)) +
 * ffn_inp}, the pre-norm state that is the next step's {@code g}; logits {@code = output . output_norm(h)}, scaled by
 * the drafter's optional {@code logit_scale}, with draft id {@code i} standing for target id {@code d2t[i]}. The token
 * table is the drafter's own when it has one, else the target's; the LM head likewise.
 *
 * <p>Protocol (positions are the target's): the decoder row at position {@code P} is the pair
 * {@code (token[P + 1], g[P])}. A target call over {@code [p0, p0 + m)} writes {@code (token[p0], g[p0 - 1])} at
 * {@code p0 - 1} when {@code g[p0 - 1]} is known, and {@code (token[p0 + k + 1], g[p0 + k])} at {@code p0 + k} for
 * {@code k < m - 1}; its {@code g} rows become the verify rows and the last one pending at {@code p0 + m - 1}.
 * {@code accept(n)} makes {@code g[p0 + min(n, m - 1)]} pending. A draft starts with {@code (lastToken, pending)} at
 * the pending position and continues with each drafted token and the decoder's own {@code h} one position later.
 * Steps at a position first drop the decoder's cache rows from that position on. A generation that reuses a stored
 * prompt prefix ({@link #beginGeneration(int)}) keeps the rows of those positions and re-evaluates the last one,
 * whose row pairs the new first tail token with its {@code g}.
 */
public final class Eagle3DraftModel implements DraftModel, DraftProbs {

    private final MtpHead head;
    private final ModelSpec draft;
    private final ModelSpec target;
    private final MemoryConfig memoryConfig;
    private final DraftModelLoader.Options options;
    private final LayerSpec layer;
    private final int nEmbd;
    private final int nTarget;
    private final int[] targetLayers;
    private final int nEnc;
    private final float eps;
    private final boolean normBeforeFc;
    private final boolean normBeforeResidual;
    private final float logitScale;
    /** The per-tap encoder norms of an {@code fc_norm} drafter, or null when the drafter keeps none. */
    private final String[] fcNorm;
    private final Source embdSource;
    private final String embdTable;
    private final Source headSource;
    private final String headWeight;
    private final String outputNorm;
    private final String attnNorm;
    private final String attnNorm2;
    private final String ffnNorm;
    private final EagleFusion fusion;
    private final String encNorm;
    private final int draftVocab;

    private ForwardBackend backend;
    private DraftModelLoader.TargetFeed feed;
    private DraftModelLoader.HiddenRows hidden;
    private MtpHead.Memory memory;
    private MtpHead.Blocks blocks;
    private int[] d2t;
    private int chunk;
    private Rows enc;
    private Rows[] encParts;
    private Rows[] encPartNorms;
    private Rows encTmp;
    private Rows g;
    private Rows gIn;
    private Rows gNorm;
    private Rows e;
    private Rows cat;
    private Rows attn;
    private Rows ffnInp;
    private Rows ffnIn;
    private Rows h;
    private Rows hn;
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
     * @throws IllegalArgumentException when the drafter is not an eagle3 model or does not fit the target (hidden
     *                                  size, tap layers, tensors, vocabulary)
     */
    Eagle3DraftModel(MtpHead head, ModelSpec draft, MemoryConfig memoryConfig, DraftModelLoader.Options options) {
        this.head = Objects.requireNonNull(head, "head");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.memoryConfig = Objects.requireNonNull(memoryConfig, "memoryConfig");
        this.options = Objects.requireNonNull(options, "options");
        this.target = head.target();
        if (draft.arch() != Arch.EAGLE3) {
            throw new IllegalArgumentException(draft.arch().ggufName() + " is not an eagle3 drafter");
        }
        ModelHparams hp = draft.hparams();
        if (draft.nLayer() != 1) {
            throw new IllegalArgumentException("eagle3: the drafter has exactly one decoder layer, got "
                    + draft.nLayer() + " (eagle3.cpp:156)");
        }
        this.layer = draft.layer(0);
        this.nEmbd = hp.nEmbd();
        this.nTarget = hp.targetHiddenSize();
        if (nTarget != target.hparams().nEmbd()) {
            throw new IllegalArgumentException("eagle3: target_hidden_size is " + nTarget + ", the target "
                    + target.arch().ggufName() + " has n_embd " + target.hparams().nEmbd());
        }
        this.targetLayers = hp.targetLayers();
        for (int id : targetLayers) {
            if (id < 0 || id > target.nLayer()) {
                throw new IllegalArgumentException("eagle3: target layer " + id + " is outside [0, "
                        + target.nLayer() + "] (speculative.cpp:601-608)");
            }
        }
        this.nEnc = targetLayers.length * nTarget;
        this.eps = hp.normRmsEps();
        this.normBeforeFc = hp.normBeforeFc();
        this.normBeforeResidual = hp.normBeforeResidual();
        this.logitScale = hp.logitScale() == 0f ? 1f : hp.logitScale();
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
        this.outputNorm = required(TensorNames.weight(TensorNames.OUTPUT_NORM));
        this.attnNorm = required(blk(TensorNames.ATTN_NORM));
        this.attnNorm2 = required(blk(TensorNames.ATTN_NORM_2));
        this.ffnNorm = required(blk(TensorNames.FFN_NORM));
        this.encNorm = normBeforeFc ? required(TensorNames.weight(TensorNames.ENC_OUTPUT_NORM)) : null;
        this.fusion = EagleFusion.bind(head, Source.DRAFT, required(TensorNames.weight(TensorNames.FC)), null,
                nEnc, nEmbd);
        this.fcNorm = fcNorms();
        long[] embdShape = head.shape(embdSource, embdTable);
        if (embdShape[0] != nEmbd) {
            throw new IllegalArgumentException("eagle3: " + embdTable + " rows hold " + embdShape[0]
                    + " floats, the decoder n_embd is " + nEmbd);
        }
        this.draftVocab = Math.toIntExact(head.shape(headSource, headWeight)[1]);
    }

    private static String blk(String base) {
        return TensorNames.layerName(0, TensorNames.weight(base));
    }

    /** {@code logits *= scale}: the draft configuration's {@code logit_scale} (vLLM's logits processor). */
    private static void scaleLogits(float[] logits, float scale) {
        for (int i = 0; i < logits.length; i++) {
            logits[i] *= scale;
        }
    }

    /**
     * The per-tap encoder norms ({@code fc_norm.<part>.weight}) of a vLLM EAGLE-3 drafter, or null when the
     * checkpoint keeps none. They are applied after {@code norm_before_fc}, whose whole-input norm the same drafts
     * may also carry.
     *
     * @throws IllegalArgumentException when one tap's norm is missing or does not hold one value per target tap dim
     */
    private String[] fcNorms() {
        if (!head.has(Source.DRAFT, TensorNames.fcNorm(0))) {
            return null;
        }
        if (targetLayers.length != 3) {
            throw new IllegalArgumentException("eagle3: fc_norm describes the " + targetLayers.length
                    + " target taps, the three-layer EAGLE-3 fusion expects three");
        }
        String[] out = new String[targetLayers.length];
        for (int k = 0; k < out.length; k++) {
            String name = TensorNames.fcNorm(k);
            if (!head.has(Source.DRAFT, name)) {
                throw new IllegalArgumentException("eagle3: the drafter has fc_norm weights but no " + name);
            }
            long[] shape = head.shape(Source.DRAFT, name);
            if (shape.length != 1 || shape[0] != nTarget) {
                throw new IllegalArgumentException("eagle3: " + name + " has shape " + Arrays.toString(shape)
                        + ", expected one value per target hidden dim (" + nTarget + ")");
            }
            out[k] = name;
        }
        return out;
    }

    private String required(String tensor) {
        if (!head.has(Source.DRAFT, tensor)) {
            throw new IllegalArgumentException("eagle3: the drafter needs " + tensor);
        }
        return tensor;
    }

    private String requiredTarget(TensorRef ref, String what) {
        if (ref == null || !head.has(Source.TARGET, ref.name())) {
            throw new IllegalArgumentException("eagle3: the decoder needs " + what);
        }
        return ref.name();
    }

    @Override
    public void bindTarget(EngineModel model) {
        Objects.requireNonNull(model, "target");
        if (backend != null) {
            throw new IllegalStateException("the eagle3 drafter is already bound");
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
        this.blocks = head.blocks(draft, Source.DRAFT, memory, layer);
        this.fusion.allocate();
        this.enc = head.rows("eagle3.enc", nEnc);
        if (fcNorm != null) {
            this.encParts = new Rows[fcNorm.length];
            this.encPartNorms = new Rows[fcNorm.length];
            for (int k = 0; k < fcNorm.length; k++) {
                this.encParts[k] = head.rows("eagle3.enc_part" + k, nTarget);
                this.encPartNorms[k] = head.rows("eagle3.enc_part_norm" + k, nTarget);
            }
            this.encTmp = head.rows("eagle3.enc_tmp", 2 * nTarget);
        }
        this.g = head.rows("eagle3.g", nEmbd);
        this.gIn = head.rows("eagle3.g_in", nEmbd);
        this.gNorm = head.rows("eagle3.g_norm", nEmbd);
        this.e = head.rows("eagle3.e", nEmbd);
        this.cat = head.rows("eagle3.cat", 2 * nEmbd);
        this.attn = head.rows("eagle3.attn", nEmbd);
        this.ffnInp = head.rows("eagle3.ffn_inp", nEmbd);
        this.ffnIn = head.rows("eagle3.ffn_in", nEmbd);
        this.h = head.rows("eagle3.h", nEmbd);
        this.hn = head.rows("eagle3.hn", nEmbd);
        this.logits = new float[draftVocab];
        this.hidden = new DraftModelLoader.HiddenRows(nEmbd);
        this.feed = new DraftModelLoader.TargetFeed(targetLayers, nTarget, tm);
        fb.setHiddenTap(feed);
        this.backend = fb;
    }

    /** Draft id to target id: {@code d2t} when present (eagle3.cpp:286-299), else the identity. */
    private int[] mapping() {
        int targetVocab = Math.toIntExact(target.head().vocabRows());
        TensorRef ref = draft.head().d2t();
        if (ref == null) {
            if (draftVocab != targetVocab) {
                throw new IllegalArgumentException("eagle3: the draft vocabulary has " + draftVocab + " tokens, the"
                        + " target " + targetVocab + ", and there is no d2t table");
            }
            return null;
        }
        int[] map = head.integers(Source.DRAFT, ref.name());
        if (map.length != draftVocab) {
            throw new IllegalArgumentException("eagle3: d2t has " + map.length + " entries for " + draftVocab
                    + " draft tokens");
        }
        for (int i = 0; i < map.length; i++) {
            if (map[i] < 0 || map[i] >= targetVocab) {
                throw new IllegalArgumentException("eagle3: d2t[" + i + "] = " + map[i] + " is outside the target"
                        + " vocabulary of " + targetVocab);
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
        float[] gRow = hidden.pending().clone();
        int[] out = new int[count];
        int made = 0;
        int[] one = new int[]{lastToken};
        for (int i = 0; i < count; i++) {
            decode(pos + i, one, gRow, 1);
            head.norm(NormKind.RMS, Source.DRAFT, outputNorm, eps, h, hn, 1);
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
                head.readRow(h, 0, gRow);
            }
        }
        draftCount = made;
        return Arrays.copyOf(out, made);
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
        // row P pairs token[P + 1] with g[P]: re-evaluating the last stored position rewrites its row against the
        // new first tail token and supplies g for the row after it
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
            throw new IllegalStateException("the eagle3 drafter is not bound to a target");
        }
    }

    private void sync() {
        for (Batch b : feed.drain()) {
            process(b);
        }
    }

    /** Encodes a target call's features and writes its decoder rows (speculative.cpp eagle3 process). */
    private void process(Batch b) {
        int m = b.size();
        float[] gAll = encode(b, m);
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
            System.arraycopy(gAll, k * nEmbd, pairs, 0, len * nEmbd);
            decode(b.pos0 + k, Arrays.copyOfRange(b.tokens, k + 1, k + 1 + len), pairs, len);
        }
        hidden.record(b.pos0, gAll, m);
    }

    /** {@code g} rows of a target call: {@code fc . [fc_norm]([enc_norm](features))}, read back to the host. */
    private float[] encode(Batch b, int m) {
        float[] gAll = new float[Math.multiplyExact(m, nEmbd)];
        float[] gChunk = new float[chunk * nEmbd];
        float[] features = new float[chunk * nEnc];
        float[] part = fcNorm == null ? null : new float[chunk * nTarget];
        for (int c = 0; c < m; c += chunk) {
            int len = Math.min(chunk, m - c);
            writeFeatures(b, c, len, features);
            if (fcNorm != null) {
                writeFcNorms(len, features, part);
            }
            fusion.project(enc, g, len);
            head.read(g, len, gChunk);
            System.arraycopy(gChunk, 0, gAll, c * nEmbd, len * nEmbd);
        }
        return gAll;
    }

    /**
     * The fusion input of {@code len} rows at {@code c}: the target taps concatenated in tap order into {@code enc},
     * RMS-normed as a whole when {@code norm_before_fc} is set. Leaves the rows in {@code rows}: the raw features, or
     * the whole-input normed features when they are read back for {@link #writeFcNorms}.
     */
    private void writeFeatures(Batch b, int c, int len, float[] rows) {
        for (int t = 0; t < len; t++) {
            for (int k = 0; k < targetLayers.length; k++) {
                System.arraycopy(b.rows[k], (c + t) * nTarget, rows, t * nEnc + k * nTarget, nTarget);
            }
        }
        head.write(enc, rows, len);
        if (normBeforeFc) {
            head.norm(NormKind.RMS, Source.DRAFT, encNorm, eps, enc, enc, len);
            if (fcNorm != null) {
                head.read(enc, len, rows);
            }
        }
    }

    /**
     * The per-tap {@code fc_norm} of the fusion rows in {@code rows} ({@code len} rows of {@code nEnc} floats),
     * concatenated in tap order into {@code enc}.
     */
    private void writeFcNorms(int len, float[] rows, float[] part) {
        for (int k = 0; k < fcNorm.length; k++) {
            for (int t = 0; t < len; t++) {
                System.arraycopy(rows, t * nEnc + k * nTarget, part, t * nTarget, nTarget);
            }
            head.write(encParts[k], part, len);
            head.norm(NormKind.RMS, Source.DRAFT, fcNorm[k], eps, encParts[k], encPartNorms[k], len);
        }
        head.concat(encPartNorms[0], encPartNorms[1], encTmp, len);
        head.concat(encTmp, encPartNorms[2], enc, len);
    }

    /** The decoder layer over {@code len} pairs at {@code pos0 ..}, leaving the pre-norm state in {@code h}. */
    private void decode(int pos0, int[] tokens, float[] gRows, int len) {
        head.embed(embdSource, embdTable, tokens, len, e);
        head.norm(NormKind.RMS, Source.DRAFT, attnNorm, eps, e, e, len);
        head.write(gIn, gRows, len);
        head.norm(NormKind.RMS, Source.DRAFT, attnNorm2, eps, gIn, gNorm, len);
        head.concat(e, gNorm, cat, len);
        blocks.mixer(pos0, len, tokens, cat, attn);
        head.add(attn, normBeforeResidual ? gNorm : gIn, ffnInp, len);
        head.norm(NormKind.RMS, Source.DRAFT, ffnNorm, eps, ffnInp, ffnIn, len);
        blocks.ffn(len, ffnIn, h);
        head.add(h, ffnInp, h, len);
    }

    @Override
    public String toString() {
        return "Eagle3DraftModel{targets " + Arrays.toString(targetLayers) + ", n_embd " + nEmbd + ", draft vocab "
                + draftVocab + (d2t != null ? ", d2t" : "") + '}';
    }
}
