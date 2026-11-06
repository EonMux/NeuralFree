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
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * The EAGLE-1 drafter (vLLM llama_eagle.py and mistral_eagle.py, sglang srt/models/llama_eagle.py, the SafeAILab
 * EAGLE fused drafter of cnets.py:427), driven like llama.cpp common/speculative.cpp drives its eagle drafters.
 *
 * <p>Fusion (llama_eagle.py:127-131): {@code g = fc . [embed(tok) | hIn]}, where {@code hIn} is the hidden row of
 * the drafter's single {@code target_layer} tap the step fuses. A DeepSeek-MTP-shaped checkpoint (vLLM's
 * {@code deepseek_eagle.EagleDeepseekV3ForCausalLM}, the {@code EagleDeepSeekMTPModel} name) normalizes each half
 * before the fusion, {@code g = fc . [enorm(embed(tok)) | hnorm(hIn)]} (deepseek_eagle.py:86-89): the drafter reads
 * both weights when the file carries them. One standard decoder layer over caches of its own:
 * the attention runs on {@code g} when the checkpoint keeps no input norm (llama_eagle.py:52-61, cnets.py:427) and
 * on its norm otherwise (mistral_eagle.py keeps {@code input_layernorm}); the attention's residual is the raw
 * {@code g} either way, so {@code ffn_inp = attn + g} and {@code h = ffn(ffn_norm(ffn_inp)) + ffn_inp}. The logits
 * are {@code head . hn} with {@code hn = output_norm(h)} when the checkpoint keeps the final norm (mistral_eagle.py
 * :109-129) and {@code hn = h} otherwise (llama_eagle.py keeps none); {@code hn} is the next step's {@code hIn}.
 * There is no d2t: the draft vocabulary is the target's, and the embeddings and LM head are the drafter's own when
 * the checkpoint carries them, else the target's (vLLM shares them from the target model, mistral_eagle.py's
 * {@code load_eagle_model}).
 *
 * <p>Protocol (positions are the target's): the decoder row at position {@code P} is the pair
 * {@code (token[P], h[P - 1])} — vLLM's {@code shift_input_ids} kernel pairs every drafter token with the previous
 * position's hidden state (v1/spec_decode/utils.py:441-499) — and the row predicts {@code token[P + 1]}. A target
 * call over {@code [p0, p0 + m)} writes those rows for {@code k = 0 .. m - 1} from its tokens and the hidden rows
 * {@code h[p0 - 1 .. p0 + m - 2]}: the first fuses the row {@code h[p0 - 1]} a previous call left, or a zero row
 * when none exists (vLLM skips that first token; here the row is written so the caches stay dense for the
 * attention). Its last hidden row {@code h[p0 + m - 1]} stays pending. {@code accept(n)} makes
 * {@code h[p0 + min(n, m - 1)]} pending. A draft continues from the pending row: its first step at
 * {@code pending + 1} fuses it with {@code lastToken}, and every later step fuses the previous step's {@code hn}
 * with the token it predicted, one position later each. Steps at a position first drop the decoder's cache rows
 * from that position on. A generation that reuses a stored prompt prefix ({@link #beginGeneration(int)}) keeps the
 * decoder rows of those positions and re-evaluates the last one: its row is already correct, so that call only
 * supplies the hidden row the first new row fuses.
 */
public final class Eagle1DraftModel implements DraftModel, DraftProbs {

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
    private final String attnNorm;
    private final String outputNorm;
    private final String ffnNorm;
    private final String fc;
    /** The embedding-side fusion input norm, or null when the checkpoint normalizes neither half. */
    private final String encEnorm;
    /** The target-hidden-side fusion input norm, present exactly when {@link #encEnorm} is. */
    private final String encHnorm;
    private final Source embdSource;
    private final String embdTable;
    private final Source headSource;
    private final String headWeight;
    private final int draftVocab;

    private ForwardBackend backend;
    private DraftModelLoader.TargetFeed feed;
    private DraftModelLoader.HiddenRows hidden;
    private MtpHead.Memory memory;
    private MtpHead.Blocks blocks;
    private int chunk;
    private Rows e;
    private Rows hIn;
    private Rows fused;
    private Rows g;
    private Rows gNorm;
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
     * @throws IllegalArgumentException when the drafter is not an eagle model or does not fit the target (hidden
     *                                  size, tap layer, tensors, vocabulary)
     */
    Eagle1DraftModel(MtpHead head, ModelSpec draft, MemoryConfig memoryConfig, DraftModelLoader.Options options) {
        this.head = Objects.requireNonNull(head, "head");
        this.draft = Objects.requireNonNull(draft, "draft");
        this.memoryConfig = Objects.requireNonNull(memoryConfig, "memoryConfig");
        this.options = Objects.requireNonNull(options, "options");
        this.target = head.target();
        if (draft.arch() != Arch.EAGLE) {
            throw new IllegalArgumentException(draft.arch().ggufName() + " is not an eagle drafter");
        }
        ModelHparams hp = draft.hparams();
        if (draft.nLayer() != 1) {
            throw new IllegalArgumentException("eagle: the drafter holds exactly one decoder layer, got "
                    + draft.nLayer() + " (llama_eagle.py builds one)");
        }
        this.layer = draft.layer(0);
        this.nEmbd = hp.nEmbd();
        this.nTarget = hp.targetHiddenSize();
        if (nTarget != target.hparams().nEmbd()) {
            throw new IllegalArgumentException("eagle: target_hidden_size is " + nTarget + ", the target "
                    + target.arch().ggufName() + " has n_embd " + target.hparams().nEmbd());
        }
        this.targetLayers = hp.targetLayers();
        if (targetLayers[0] < 0 || targetLayers[0] > target.nLayer()) {
            throw new IllegalArgumentException("eagle: target layer " + targetLayers[0] + " is outside [0, "
                    + target.nLayer() + "]");
        }
        this.nEnc = nEmbd + nTarget;
        this.eps = hp.normRmsEps();
        this.attnNorm = head.has(Source.DRAFT, blk(TensorNames.ATTN_NORM)) ? blk(TensorNames.ATTN_NORM) : null;
        this.outputNorm = head.has(Source.DRAFT, TensorNames.weight(TensorNames.OUTPUT_NORM))
                ? TensorNames.weight(TensorNames.OUTPUT_NORM) : null;
        this.ffnNorm = required(blk(TensorNames.FFN_NORM));
        this.fc = required(TensorNames.weight(TensorNames.FC));
        this.encEnorm = fusionNorm(TensorNames.ENC_ENORM);
        this.encHnorm = fusionNorm(TensorNames.ENC_HNORM);
        if ((encEnorm == null) != (encHnorm == null)) {
            throw new IllegalArgumentException("eagle: the drafter holds "
                    + (encEnorm != null ? encEnorm : encHnorm) + " but no "
                    + (encEnorm != null ? encHnorm : encEnorm) + "; a fusion drafter normalizes both halves or"
                    + " neither");
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
        long[] fcShape = head.shape(Source.DRAFT, fc);
        if (fcShape[0] != nEnc || fcShape[1] != nEmbd) {
            throw new IllegalArgumentException("eagle: fc has shape " + Arrays.toString(fcShape) + ", expected ["
                    + nEnc + ", " + nEmbd + "]");
        }
        long[] embdShape = head.shape(embdSource, embdTable);
        if (embdShape[0] != nEmbd) {
            throw new IllegalArgumentException("eagle: " + embdTable + " rows hold " + embdShape[0]
                    + " floats, the decoder n_embd is " + nEmbd);
        }
        this.draftVocab = Math.toIntExact(head.shape(headSource, headWeight)[1]);
        int targetVocab = Math.toIntExact(target.head().vocabRows());
        if (draftVocab != targetVocab) {
            throw new IllegalArgumentException("eagle: the draft vocabulary has " + draftVocab + " tokens, the"
                    + " target " + targetVocab + " (an eagle drafter shares the target's vocabulary)");
        }
    }

    private static String blk(String base) {
        return TensorNames.layerName(0, TensorNames.weight(base));
    }

    private String required(String tensor) {
        if (!head.has(Source.DRAFT, tensor)) {
            throw new IllegalArgumentException("eagle: the drafter needs " + tensor);
        }
        return tensor;
    }

    /**
     * The stored fusion input norm {@code base + ".weight"}, or null when the checkpoint keeps none.
     *
     * @throws IllegalArgumentException when it does not hold one value per fused hidden dim
     */
    private String fusionNorm(String base) {
        String name = TensorNames.weight(base);
        if (!head.has(Source.DRAFT, name)) {
            return null;
        }
        long[] shape = head.shape(Source.DRAFT, name);
        if (shape.length != 1 || shape[0] != nEmbd) {
            throw new IllegalArgumentException("eagle: " + name + " has shape " + Arrays.toString(shape)
                    + ", expected one value per fused hidden dim (" + nEmbd + ")");
        }
        return name;
    }

    private String requiredTarget(TensorRef ref, String what) {
        if (ref == null || !head.has(Source.TARGET, ref.name())) {
            throw new IllegalArgumentException("eagle: the decoder needs " + what);
        }
        return ref.name();
    }

    @Override
    public void bindTarget(EngineModel model) {
        Objects.requireNonNull(model, "target");
        if (backend != null) {
            throw new IllegalStateException("the eagle drafter is already bound");
        }
        ForwardBackend fb = model.forwardBackend();
        ModelMemory tm = fb.memory();
        if (tm == null) {
            throw new IllegalArgumentException(target.arch().ggufName() + ": the target runs without memory");
        }
        MemoryPlan plan = MemoryPlanner.planForContext(draft, memoryConfig, tm.plan().nCtx());
        this.chunk = Math.min(head.ubatch(), plan.ubatch());
        this.memory = head.ownMemory(draft, plan);
        this.blocks = head.blocks(draft, Source.DRAFT, memory, layer);
        this.e = head.rows("eagle1.e", nEmbd);
        this.hIn = head.rows("eagle1.h_in", nTarget);
        this.fused = head.rows("eagle1.fused", nEnc);
        this.g = head.rows("eagle1.g", nEmbd);
        this.gNorm = head.rows("eagle1.g_norm", nEmbd);
        this.attn = head.rows("eagle1.attn", nEmbd);
        this.ffnInp = head.rows("eagle1.ffn_inp", nEmbd);
        this.ffnIn = head.rows("eagle1.ffn_in", nEmbd);
        this.h = head.rows("eagle1.h", nEmbd);
        this.hn = head.rows("eagle1.hn", nEmbd);
        this.logits = new float[draftVocab];
        this.hidden = new DraftModelLoader.HiddenRows(nTarget);
        this.feed = new DraftModelLoader.TargetFeed(targetLayers, nTarget, tm);
        fb.setHiddenTap(feed);
        this.backend = fb;
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
        int pending = hidden.pendingPos();
        if (pending < 0 || n == 0) {
            return new int[0];
        }
        int pos0 = pending + 1;
        int count = Math.min(n, memory.memory().plan().nCtx() - pos0);
        if (count <= 0) {
            return new int[0];
        }
        memory.align(pos0);
        float[] hRow = hidden.pending().clone();
        Rows state = outputNorm != null ? hn : h;
        int[] out = new int[count];
        int made = 0;
        int[] one = new int[]{lastToken};
        for (int i = 0; i < count; i++) {
            decode(pos0 + i, one, hRow, 1);
            head.logits(headSource, headWeight, state, 0, logits);
            boolean watermarked = draftWatermark != null && draftTemperature > 0f;
            Pick p = watermarked
                    ? DraftModelLoader.pick(logits, draftVocab, draftWatermark, draftTemperature,
                            draftRng::nextDouble)
                    : DraftModelLoader.pick(logits, draftVocab);
            if (p.p < options.pMin()) {
                break;
            }
            if (watermarked) {
                draftProposals.add(p.proposal);
            }
            out[made++] = p.id;
            one[0] = p.id;
            if (i + 1 < count) {
                head.readRow(state, 0, hRow);
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
        // the decoder re-evaluates the last stored position: this cache already holds its decoder row, and the
        // call's hidden row is the one the first new row fuses
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
            throw new IllegalStateException("the eagle drafter is not bound to a target");
        }
    }

    private void sync() {
        for (Batch b : feed.drain()) {
            process(b);
        }
    }

    /**
     * Writes a target call's decoder rows (speculative.cpp's eagle process): row {@code p0 + k} pairs the call's
     * token at that position with the previous position's hidden row, and the call's last row stays pending. A first
     * call after a reused prefix re-evaluates the last stored position, whose row the caches already hold, so it is
     * kept and only the rows from the next position on are written.
     */
    private void process(Batch b) {
        int m = b.size();
        float[] rows = b.rows[0];
        float[] before = hidden.at(b.pos0 - 1);
        int start = b.pos0;
        if (before == null && memory.memory().sequence().nPast() > b.pos0) {
            // the retained row at b.pos0 pairs the same token with the same hidden row, and this call carries the
            // hidden row the row at b.pos0 + 1 fuses
            start = b.pos0 + 1;
        }
        memory.align(start);
        float[] hChunk = new float[chunk * nTarget];
        for (int c = start - b.pos0; c < m; c += chunk) {
            int len = Math.min(chunk, m - c);
            for (int t = 0; t < len; t++) {
                int k = c + t;
                // a first call with no previous row fuses a zero row (vLLM skips the token; the row keeps
                // the caches dense); every other row k reads h[p0 + k - 1]
                if (k == 0 && before != null) {
                    System.arraycopy(before, 0, hChunk, 0, nTarget);
                } else if (k > 0) {
                    System.arraycopy(rows, (k - 1) * nTarget, hChunk, t * nTarget, nTarget);
                }
            }
            decode(b.pos0 + c, Arrays.copyOfRange(b.tokens, c, c + len), hChunk, len);
        }
        hidden.record(b.pos0, rows, m);
    }

    /** The decoder layer over {@code len} pairs at {@code pos0 ..}, leaving the state row in {@code h}/{@code hn}. */
    private void decode(int pos0, int[] tokens, float[] hRows, int len) {
        head.embed(embdSource, embdTable, tokens, len, e);
        head.write(hIn, hRows, len);
        if (encEnorm != null) {
            head.norm(NormKind.RMS, Source.DRAFT, encEnorm, eps, e, e, len);
            head.norm(NormKind.RMS, Source.DRAFT, encHnorm, eps, hIn, hIn, len);
        }
        head.concat(e, hIn, fused, len);
        head.project(Source.DRAFT, fc, fused, g, len);
        Rows aIn = g;
        if (attnNorm != null) {
            head.norm(NormKind.RMS, Source.DRAFT, attnNorm, eps, g, gNorm, len);
            aIn = gNorm;
        }
        blocks.mixer(pos0, len, tokens, aIn, attn);
        head.add(attn, g, ffnInp, len);
        head.norm(NormKind.RMS, Source.DRAFT, ffnNorm, eps, ffnInp, ffnIn, len);
        blocks.ffn(len, ffnIn, h);
        head.add(h, ffnInp, h, len);
        if (outputNorm != null) {
            head.norm(NormKind.RMS, Source.DRAFT, outputNorm, eps, h, hn, len);
        }
    }

    @Override
    public String toString() {
        return "Eagle1DraftModel{target layer " + targetLayers[0] + ", n_embd " + nEmbd + ", draft vocab " + draftVocab
                + (attnNorm != null ? ", attn_norm" : "") + (outputNorm != null ? ", output_norm" : "")
                + (encEnorm != null ? ", enc norms" : "") + '}';
    }
}
