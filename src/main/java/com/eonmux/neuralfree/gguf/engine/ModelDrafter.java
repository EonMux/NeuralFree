package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.draft.DraftProbs;
import com.eonmux.neuralfree.gguf.engine.watermark.DraftWatermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceState;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Speculative drafting with a smaller ordinary language model that shares the target's vocabulary (llama.cpp
 * {@code common_speculative_impl_draft_simple}, common/speculative.cpp:219-512, with
 * {@code common_speculative_are_compatible}, :107-176).
 *
 * <p>Per {@link #draft}: the draft model's memory is brought to the target's stored tokens (the longest common
 * prefix is kept, the rest rolled back, the missing target tokens prefilled), then {@code lastToken} is decoded at the
 * target's next position and tokens are drafted greedily ({@link DraftModelLoader#pick}): each step takes the most
 * likely token and its probability among the 10 most likely ones (the drafter's top-k 10 sampler), stops when that
 * probability is below {@code p_min}, and decodes the token to continue unless the draft is complete. Drafted tokens
 * stay in the draft memory; the next draft drops the ones the target rejected by the same prefix comparison, so
 * {@link #accept} keeps no state.
 *
 * <p>Two vocabularies that share token ids are used as they are (llama.cpp
 * {@code common_speculative_are_compatible}); with heterogeneous-vocabulary drafting ({@link #ModelDrafter(EngineModel,
 * float, boolean)}) the two may differ, and a {@link DraftVocabMapping} translates the target's tokens into the
 * draft's, masks the draft logits to the token-level intersection and translates the proposals back.
 *
 * <p><b>Multi-branch (tree) drafting.</b> When both memories are paged and hold more than one sequence, this
 * drafter implements {@link TreeDraftModel}: it evaluates one draft sequence per branch and splits a branch into a
 * new one when a secondary candidate's probability exceeds {@code pSplit} (llama.cpp's
 * {@code examples/speculative/speculative.cpp}:{@code n_seq_dft} branches and {@code p_draft_split}), returning the
 * {@link DraftTree} the decoder verifies. One draft sequence evaluates one branch per depth; a split copies the
 * branch's pages to a fresh sequence (copy-on-write), so the tree costs no token copies.
 *
 * <p>The drafter owns the draft model and closes it. Not thread-safe: it runs on the target's forward thread.
 */
public final class ModelDrafter implements TreeDraftModel, DraftProbs {

    /** Largest vocabulary size difference llama.cpp accepts ({@code SPEC_VOCAB_MAX_SIZE_DIFFERENCE}). */
    static final int MAX_VOCAB_DIFFERENCE = 128;
    /** First token id whose text must match ({@code SPEC_VOCAB_CHECK_START_TOKEN_ID}). */
    static final int VOCAB_CHECK_START = 5;
    /** Largest candidate set a tree branch splits from (llama.cpp's draft sampler top-k, its {@code f < 8} loop). */
    private static final int MAX_CANDIDATES = 10;

    private final EngineModel draft;
    private final float pMin;
    private final boolean heterogeneousVocab;
    private final double pSplit;
    private ForwardBackend target;
    private ForwardBackend drafter;
    private DraftVocabMapping vocab;
    private int targetVocab;
    /** Draft sequences the bound target's paged memory holds, 0 when the drafter drafts linearly. */
    private int branches;
    private boolean closed;
    /** The draft-side watermark of the current generation, or null for the ordinary greedy pick. */
    private DraftWatermarker draftWatermark;
    private SplittableRandom draftRng;
    /** The proposal rows of the last draft, by position (null entries for a greedy pick). */
    private float[][] proposalRows = new float[0][];

    /**
     * A drafter of vocabularies that share token ids.
     *
     * @param draft the draft model, owned by this drafter
     * @param pMin  drafting stops at a token whose top-10 probability is below this ({@code --draft-p-min})
     * @throws IllegalArgumentException for a {@code pMin} outside {@code [0, 1]}
     */
    public ModelDrafter(EngineModel draft, float pMin) {
        this(draft, pMin, false, SpeculativeOptions.DEFAULT_P_SPLIT);
    }

    /**
     * @param draft              the draft model, owned by this drafter
     * @param pMin               drafting stops at a token whose top-10 probability is below this
     *                           ({@code --draft-p-min})
     * @param heterogeneousVocab whether to draft across two vocabularies by their token-level intersection
     *                           (vLLM {@code use_heterogeneous_vocab}), instead of requiring shared token ids
     * @throws IllegalArgumentException for a {@code pMin} outside {@code [0, 1]}
     */
    public ModelDrafter(EngineModel draft, float pMin, boolean heterogeneousVocab) {
        this(draft, pMin, heterogeneousVocab, SpeculativeOptions.DEFAULT_P_SPLIT);
    }

    /**
     * @param pSplit the probability above which a secondary draft candidate splits a branch into a new one
     *               ({@code --spec-draft-p-split}; only paged multi-sequence drafting reads it)
     * @throws IllegalArgumentException for a {@code pMin} outside {@code [0, 1]} or a {@code pSplit} outside it too
     */
    public ModelDrafter(EngineModel draft, float pMin, boolean heterogeneousVocab, double pSplit) {
        this.draft = Objects.requireNonNull(draft, "draft");
        if (!(pMin >= 0f && pMin <= 1f)) {
            throw new IllegalArgumentException("draft p_min must be in [0, 1], got " + pMin);
        }
        if (!(pSplit >= 0.0 && pSplit <= 1.0)) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_P_SPLIT + " must be in [0, 1], got " + pSplit);
        }
        this.pMin = pMin;
        this.heterogeneousVocab = heterogeneousVocab;
        this.pSplit = pSplit;
    }

    /** The draft branch split probability ({@code --spec-draft-p-split}). */
    public double pSplit() {
        return pSplit;
    }

    /** Draft sequences a tree proposal may use; 0 when this drafter proposes one linear chain. */
    public int branches() {
        return branches;
    }

    /** The draft model. */
    public EngineModel draftModel() {
        return draft;
    }

    /**
     * @throws IllegalArgumentException when the draft model cannot draft for the target: it does not generate
     *                                  autoregressively with memory, or the vocabularies are incompatible (without
     *                                  heterogeneous-vocabulary drafting)
     */
    @Override
    public void bindTarget(EngineModel targetModel) {
        Objects.requireNonNull(targetModel, "target");
        ensureOpen();
        if (target != null) {
            throw new IllegalStateException("the draft model is already bound to a target");
        }
        if (draft.supportClass() != SupportClass.GEN) {
            throw new IllegalArgumentException("draft model " + draft.architecture() + " (" + draft.supportClass()
                    + ") does not generate text autoregressively");
        }
        ForwardBackend tb = targetModel.forwardBackend();
        ForwardBackend db = draft.forwardBackend();
        if (tb.memory() == null || db.memory() == null) {
            throw new IllegalArgumentException("speculative drafting needs model memory on both the target ("
                    + targetModel.architecture() + ") and the draft model (" + draft.architecture() + ")");
        }
        if (heterogeneousVocab) {
            this.vocab = DraftVocabMapping.of(targetModel.tokenizer(), targetModel.vocabSize(), draft.tokenizer(),
                    draft.vocabSize());
        } else {
            String incompatible = vocabMismatch(targetModel.tokenizer(), draft.tokenizer());
            if (incompatible != null) {
                throw new IllegalArgumentException("draft model " + draft.name() + " cannot draft for "
                        + targetModel.name() + ": " + incompatible);
            }
            this.vocab = DraftVocabMapping.identity(targetModel.vocabSize(), draft.vocabSize());
        }
        this.target = tb;
        this.drafter = db;
        this.targetVocab = targetModel.vocabSize();
        this.branches = treeBranches(tb.memory(), db.memory());
    }

    /**
     * The draft sequences a tree proposal may use: the target's paged sequences when both memories are paged and
     * hold the same count, 0 for a contiguous target or a single sequence. A paged target beside a draft model
     * without paged memory cannot draft at all: the target's per-depths forwards address its sequences.
     *
     * @throws IllegalArgumentException when the target memory is paged and the draft model's is not, or their
     *                                  sequence counts differ
     */
    private static int treeBranches(ModelMemory target, ModelMemory drafter) {
        boolean targetPaged = target.pagedSequences() != null;
        boolean draftPaged = drafter.pagedSequences() != null;
        if (!targetPaged) {
            return 0;
        }
        if (!draftPaged) {
            throw new IllegalArgumentException("the target runs paged memory with " + target.pagedSequences()
                    .maxSequences() + " sequences; the draft model needs paged memory of the same size to draft"
                    + " for it (neuralfree.parallel / neuralfree.draft.*)");
        }
        int tn = target.pagedSequences().maxSequences();
        int dn = drafter.pagedSequences().maxSequences();
        if (tn != dn) {
            throw new IllegalArgumentException("the target's paged memory holds " + tn + " sequences, the draft"
                    + " model's " + dn + "; multi-branch drafting runs one draft sequence per target sequence");
        }
        return tn;
    }

    /**
     * Why two vocabularies cannot share token ids, or null when they can (llama.cpp
     * {@code common_speculative_are_compatible}): the same tokenizer type, the same BOS / EOS insertion and ids, sizes
     * within {@value #MAX_VOCAB_DIFFERENCE}, and the same text for every token from id {@value #VOCAB_CHECK_START}.
     */
    public static String vocabMismatch(GGUFTokenizer target, GGUFTokenizer draft) {
        TokenizerVocab tv = target.vocab();
        TokenizerVocab dv = draft.vocab();
        if (tv.type() != dv.type()) {
            return "tokenizer types differ (" + tv.type() + " vs " + dv.type() + ")";
        }
        if (tv.addBos() != dv.addBos() || (tv.addBos() && target.bosId() != draft.bosId())) {
            return "BOS handling differs (add " + tv.addBos() + "/" + dv.addBos() + ", id " + target.bosId() + "/"
                    + draft.bosId() + ")";
        }
        if (tv.addEos() != dv.addEos() || (tv.addEos() && target.eosId() != draft.eosId())) {
            return "EOS handling differs (add " + tv.addEos() + "/" + dv.addEos() + ", id " + target.eosId() + "/"
                    + draft.eosId() + ")";
        }
        int nt = tv.size();
        int nd = dv.size();
        if (Math.abs(nt - nd) > MAX_VOCAB_DIFFERENCE) {
            return "vocabulary sizes " + nt + " and " + nd + " differ by more than " + MAX_VOCAB_DIFFERENCE;
        }
        for (int i = VOCAB_CHECK_START; i < Math.min(nt, nd); i++) {
            if (!tv.text(i).equals(dv.text(i))) {
                return "token " + i + " differs ('" + tv.text(i) + "' vs '" + dv.text(i) + "')";
            }
        }
        return null;
    }

    @Override
    public boolean supportsWatermarking() {
        return vocab == null || vocab.isIdentity();
    }

    @Override
    public void watermark(WatermarkConfig config, int promptLength, long seed) {
        this.draftWatermark = DraftWatermarker.of(config);
        this.draftRng = draftWatermark == null ? null : new SplittableRandom(seed);
    }

    @Override
    public float[] draftProbs(int position) {
        if (position < 0 || position >= proposalRows.length) {
            throw new IllegalStateException("no drafted token at position " + position);
        }
        return proposalRows[position] == null ? null : proposalRows[position].clone();
    }

    @Override
    public int[] draft(int lastToken, int n) {
        return draft(lastToken, n, 1f);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Under a draft watermark ({@link #watermark}) and a sampling request, each proposal is the keyed Gumbel
     * draw from the temperature-scaled draft distribution instead of the greedy argmax, and
     * {@link #draftProbs(int)} reports the distribution it was drawn from.
     */
    @Override
    public int[] draft(int lastToken, int n, float temperature) {
        ensureBound();
        if (n <= 0) {
            return new int[0];
        }
        SequenceState ts = target.memory().sequence();
        int pos = ts.nPast();
        int count = Math.min(n, drafter.memory().sequence().nCtx() - pos);
        if (count <= 0 || lastToken < 0 || lastToken >= targetVocab || !vocab.canStartDraft(lastToken)) {
            proposalRows = new float[0][];
            return new int[0];
        }
        boolean watermarked = draftWatermark != null && temperature > 0f;
        // a greedy draft over an identity vocabulary picks straight off the device's top-k readback: the drafter
        // reads its candidates back per step instead of the whole row, and the mapping cannot change the pick
        boolean devicePick = !watermarked && vocab.isIdentity() && drafter.topKOnDevice();
        try {
            sync(ts, pos);
            if (watermarked) {
                draftWatermark.sync(ts.textTokens(), lastToken);
                draftWatermark.beginRound();
            }
            int draftToken = vocab.toDraft(lastToken);
            float[] logits = devicePick ? null : drafter.decode(draftToken, pos);
            int[] out = new int[count];
            float[][] rows = new float[count][];
            int len = 0;
            while (len < count) {
                DraftModelLoader.Pick pick;
                if (devicePick) {
                    pick = DraftModelLoader.pick(
                            drafter.decodeTopKOnDevice(draftToken, pos + len, DraftModelLoader.DRAFT_TOP_K), 0);
                } else {
                    vocab.constrain(logits);
                    pick = watermarked
                            ? DraftModelLoader.pick(logits, logits.length, draftWatermark, temperature,
                                    draftRng::nextDouble)
                            : DraftModelLoader.pick(logits, logits.length);
                }
                int targetToken = vocab.toTarget(pick.id);
                if (pick.p < pMin || targetToken < 0) {
                    break;
                }
                out[len] = targetToken;
                rows[len] = pick.proposal;
                len++;
                if (len < count) {
                    if (devicePick) {
                        draftToken = pick.id;
                    } else {
                        logits = drafter.decode(pick.id, pos + len);
                    }
                }
            }
            proposalRows = Arrays.copyOf(rows, len);
            return Arrays.copyOf(out, len);
        } catch (RuntimeException e) {
            try {
                drafter.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    @Override
    public DraftTree draftTree(int lastToken, int maxTokens, int branches, float temperature, SplittableRandom rng) {
        return draftTree(lastToken, maxTokens, branches, Integer.MAX_VALUE, temperature, rng);
    }

    /**
     * The proposal, bounded to the target rows one single-pass verify lays it out over: at most
     * {@code maxTokens} depths long ({@code maxTokens} nodes for the linear chain of a contiguous memory), and
     * a node whose addition would exceed {@code maxRows} nodes (the root excluded — the verify batch adds it)
     * does not enter the branching tree, so the masked verify step's shape stays within one ubatch.
     */
    @Override
    public DraftTree draftTree(int lastToken, int maxTokens, int branches, int maxRows, float temperature,
                               SplittableRandom rng) {
        ensureBound();
        Objects.requireNonNull(rng, "rng");
        if (maxTokens <= 0) {
            return new DraftTree(java.util.Collections.emptyList());
        }
        int nodeBudget = Math.min(maxTokens, maxRows);
        try {
            if (target.memory().pagedSequences() == null) {
                return linearTree(draft(lastToken, nodeBudget));
            }
            return pagedTree(lastToken, maxTokens, Math.max(1, maxRows),
                    Math.max(1, Math.min(branches, this.branches)), temperature, rng);
        } catch (RuntimeException e) {
            try {
                drafter.reset();
            } catch (RuntimeException resetFailure) {
                e.addSuppressed(resetFailure);
            }
            throw e;
        }
    }

    /** The decoder verifies this drafter's branching proposals in one masked forward ({@code TreeVerify}). */
    @Override
    public boolean singlePassVerify() {
        return true;
    }

    /** The linear tree of a contiguous proposal. */
    private static DraftTree linearTree(int[] tokens) {
        List<DraftTree.Node> nodes = new ArrayList<>(tokens.length);
        for (int i = 0; i < tokens.length; i++) {
            nodes.add(new DraftTree.Node(tokens[i], i == 0 ? -1 : i - 1, i + 1, 0));
        }
        return new DraftTree(nodes);
    }

    /** One draft branch while a tree proposal runs. */
    private static final class Branch {
        final int id;
        int lastNode = -1;
        float[] logits;

        Branch(int id) {
            this.id = id;
        }
    }

    /** One row of a depth's draft batch: the branch it evaluates and the token it produces. */
    private static final class Row {
        final Branch branch;
        final int token;
        final int pos;

        Row(Branch branch, int token, int pos) {
            this.branch = branch;
            this.token = token;
            this.pos = pos;
        }
    }

    /** The top draft candidates of a row, ids in draft vocabulary, probabilities normalized over them. */
    private static final class Candidates {
        final int[] ids;
        final float[] probs;

        Candidates(int[] ids, float[] probs) {
            this.ids = ids;
            this.probs = probs;
        }
    }

    /**
     * The multi-branch proposal of llama.cpp's speculative example: each draft sequence evaluates one branch per
     * depth, a branch splits into a new sequence when a secondary candidate's probability exceeds
     * {@link #pSplit()}, and every depth's rows run in one forward batch. The proposal runs at most
     * {@code maxDepth} depths and holds at most {@code maxNodes} nodes.
     */
    private DraftTree pagedTree(int lastToken, int maxDepth, int maxNodes, int branches, float temperature,
                                SplittableRandom rng) {
        SequenceMemory dseqs = drafter.memory().sequences();
        SequenceMemory tseqs = target.memory().sequences();
        SequenceState ts = tseqs.sequence(0);
        int pos = ts.nPast();
        if (lastToken < 0 || lastToken >= targetVocab || !vocab.canStartDraft(lastToken)) {
            return new DraftTree(java.util.Collections.emptyList());
        }
        syncPaged(dseqs, ts, pos);
        float[] logits = decodeRow(vocab.toDraft(lastToken), pos, 0);
        List<DraftTree.Node> nodes = new ArrayList<>();
        List<Branch> active = new ArrayList<>();
        Branch root = new Branch(0);
        root.logits = logits;
        active.add(root);
        int branchCount = 1;
        for (int depth = 0; depth < maxDepth && !active.isEmpty(); depth++) {
            List<Branch> next = new ArrayList<>();
            List<Row> rows = new ArrayList<>();
            for (Branch branch : active) {
                if (nodes.size() >= maxNodes) {
                    break; // the verify width is spent; deeper depths would only add more rows
                }
                Candidates cur = candidates(branch.logits, temperature);
                if (cur.ids.length == 0) {
                    continue;
                }
                // with several draft sequences the chain follows the most likely candidate (llama.cpp
                // examples/speculative:579); a single sequence samples it
                int pick = branches > 1 || temperature <= 0f ? 0 : sample(cur, rng);
                int chosen = vocab.toTarget(cur.ids[pick]);
                if (chosen < 0 || cur.probs[pick] < pMin) {
                    continue;
                }
                int parent = branch.lastNode;
                for (int f = 1; f < 8 && f < cur.ids.length && branchCount < branches
                        && nodes.size() < maxNodes; f++) {
                    if (!(cur.probs[f] > pSplit)) {
                        break;
                    }
                    int split = vocab.toTarget(cur.ids[f]);
                    if (split < 0) {
                        continue;
                    }
                    Branch child = new Branch(branchCount++);
                    dseqs.remove(child.id, 0, -1);
                    dseqs.copy(branch.id, child.id, 0, -1);
                    child.lastNode = nodes.size();
                    nodes.add(new DraftTree.Node(split, parent, depth + 1, child.id));
                    rows.add(new Row(child, split, pos + 1 + depth));
                    next.add(child);
                }
                if (nodes.size() >= maxNodes) {
                    break; // the splits spent the budget before the chain's own node
                }
                branch.lastNode = nodes.size();
                nodes.add(new DraftTree.Node(chosen, parent, depth + 1, branch.id));
                rows.add(new Row(branch, chosen, pos + 1 + depth));
                next.add(branch);
            }
            if (rows.isEmpty()) {
                break;
            }
            ForwardBatch.Builder batch = ForwardBatch.builder(rows.size());
            for (Row row : rows) {
                batch.add(row.token, row.pos, row.branch.id, true);
            }
            float[][] out = drafter.forwardBatch(batch.build());
            if (out.length != rows.size()) {
                throw new IllegalStateException("the draft model returned " + out.length + " rows for a batch of "
                        + rows.size());
            }
            for (int i = 0; i < rows.size(); i++) {
                rows.get(i).branch.logits = out[i];
            }
            // llama.cpp walks the draft sequences in id order at every depth, so sequence 0's chain is first
            next.sort((a, b) -> Integer.compare(a.id, b.id));
            active = next;
        }
        return new DraftTree(nodes);
    }

    /** Decodes one draft-vocabulary token at {@code pos} on draft sequence {@code seq} and returns its logits. */
    private float[] decodeRow(int token, int pos, int seq) {
        ForwardBatch batch = ForwardBatch.builder(1).add(token, pos, seq, true).build();
        float[][] rows = drafter.forwardBatch(batch);
        if (rows.length != 1) {
            throw new IllegalStateException("the draft model returned " + rows.length + " rows for one token");
        }
        return rows[0];
    }

    /** Brings draft sequence 0 to the target's tokens {@code [0, pos)} and empties every other draft sequence. */
    private void syncPaged(SequenceMemory dseqs, SequenceState ts, int pos) {
        SequenceState ds = dseqs.sequence(0);
        int stored = ds.nPast();
        int keep = 0;
        int limit = Math.min(stored, pos);
        while (keep < limit && ds.token(keep) == vocab.toDraft(ts.token(keep))) {
            keep++;
        }
        if (keep < stored) {
            if (dseqs.removeRefusal(0, keep, -1) != null) {
                keep = 0;
                dseqs.remove(0, 0, -1);
            } else {
                dseqs.remove(0, keep, -1);
            }
        }
        if (keep < pos) {
            int[] missing = new int[pos - keep];
            for (int i = 0; i < missing.length; i++) {
                missing[i] = vocab.toDraft(ts.token(keep + i));
            }
            ForwardBatch.Builder batch = ForwardBatch.builder(missing.length);
            for (int i = 0; i < missing.length; i++) {
                batch.add(missing[i], keep + i, 0, false);
            }
            drafter.forwardBatch(batch.build());
        }
        for (int s = 1; s < branches; s++) {
            dseqs.remove(s, 0, -1);
        }
    }

    /**
     * The top candidates of {@code logits} after the draft vocabulary's mask, as draft ids with probabilities
     * normalized over the kept candidates: softmax at the request's temperature, or at 1 when it drafts greedily.
     */
    private Candidates candidates(float[] logits, float temperature) {
        vocab.constrain(logits);
        int k = 0;
        for (float v : logits) {
            if (!Float.isNaN(v) && v != Float.NEGATIVE_INFINITY) {
                k++;
            }
        }
        k = Math.min(k, MAX_CANDIDATES);
        if (k == 0) {
            return new Candidates(new int[0], new float[0]);
        }
        int[] ids = new int[k];
        float[] vals = new float[k];
        int filled = 0;
        for (int i = 0; i < logits.length; i++) {
            float v = logits[i];
            if (Float.isNaN(v) || v == Float.NEGATIVE_INFINITY) {
                continue;
            }
            int at;
            if (filled < k) {
                at = filled++;
            } else if (v > vals[k - 1]) {
                at = k - 1;
            } else {
                continue;
            }
            ids[at] = i;
            vals[at] = v;
            while (at > 0 && vals[at - 1] < vals[at]) {
                float tv = vals[at - 1];
                vals[at - 1] = vals[at];
                vals[at] = tv;
                int ti = ids[at - 1];
                ids[at - 1] = ids[at];
                ids[at] = ti;
                at--;
            }
        }
        double t = temperature > 0f ? temperature : 1.0;
        double[] e = new double[filled];
        double sum = 0.0;
        for (int i = 0; i < filled; i++) {
            e[i] = Math.exp((vals[i] - vals[0]) / t);
            sum += e[i];
        }
        float[] probs = new float[filled];
        for (int i = 0; i < filled; i++) {
            probs[i] = (float) (e[i] / sum);
        }
        return new Candidates(Arrays.copyOf(ids, filled), probs);
    }

    /** Samples an index of {@code cur} from its probabilities with {@code rng}. */
    private static int sample(Candidates cur, SplittableRandom rng) {
        double u = rng.nextDouble();
        double run = 0.0;
        for (int i = 0; i < cur.probs.length; i++) {
            run += cur.probs[i];
            if (u < run) {
                return i;
            }
        }
        return cur.probs.length - 1;
    }

    /** Verification keeps no drafter state: the next tree re-aligns the draft memory by prefix. */
    @Override
    public void acceptTree(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted depth count must be non-negative, got " + nAccepted);
        }
    }

    /**
     * Brings the draft memory to the target's tokens {@code [0, pos)}: keeps the common prefix, drops the rest (or
     * empties the memory when it cannot drop it) and prefills the missing target tokens, translated to the draft's
     * vocabulary (a token outside the token-level intersection becomes the draft's unknown token).
     */
    private void sync(SequenceState ts, int pos) {
        ModelMemory dm = drafter.memory();
        SequenceState ds = dm.sequence();
        int stored = ds.nPast();
        int keep = 0;
        int limit = Math.min(stored, pos);
        while (keep < limit && ds.token(keep) == vocab.toDraft(ts.token(keep))) {
            keep++;
        }
        int drop = stored - keep;
        if (drop > 0) {
            if (drafter.canRollback(drop)) {
                drafter.rollback(drop);
            } else {
                drafter.reset();
                keep = 0;
            }
        }
        if (keep < pos) {
            int[] missing = new int[pos - keep];
            for (int i = 0; i < missing.length; i++) {
                missing[i] = vocab.toDraft(ts.token(keep + i));
            }
            drafter.prefill(missing, keep);
        }
    }

    /** Verification keeps no drafter state: the next {@link #draft} re-aligns the draft memory by prefix. */
    @Override
    public void accept(int nAccepted) {
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must be non-negative, got " + nAccepted);
        }
    }

    @Override
    public void reset() {
        proposalRows = new float[0][];
        if (drafter != null && !closed) {
            drafter.reset();
        }
    }

    /** Closes the draft model. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        target = null;
        drafter = null;
        try {
            draft.close();
        } catch (IOException e) {
            throw new UncheckedIOException("closing the draft model " + draft.name() + " failed", e);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the draft model drafter is closed");
        }
    }

    private void ensureBound() {
        ensureOpen();
        if (target == null) {
            throw new IllegalStateException("the draft model drafter is not bound to a target");
        }
    }

    @Override
    public String toString() {
        return "ModelDrafter{" + draft.name() + " (" + draft.architecture() + "), pMin=" + pMin
                + (heterogeneousVocab ? ", heterogeneous vocab" : "") + '}';
    }
}
