package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.DraftVocabMapping;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ModelDrafter;
import com.eonmux.neuralfree.gguf.engine.watermark.DraftWatermarker;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SequenceMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.logging.Logger;

/**
 * Speculative drafting with a smaller language model of the target's vocabulary for every sequence of a
 * {@link BatchEngine} (vLLM's draft-model proposer, llama.cpp {@code common_speculative_impl_draft_simple} per
 * server slot): the draft model runs over paged memory of its own with one sequence per target slot, so each
 * sequence keeps its draft KV in the draft's page tables and every step drafts all sequences in shared forward
 * batches through {@link ForwardBackend#forwardBatch}.
 *
 * <p>Per {@link #draftBatch}: each requested sequence's draft rows are first brought to its target text tokens
 * (the longest common prefix stays, the rest leaves the draft sequence, the missing target tokens join the first
 * batch as rows without logits), then the first batch evaluates every sequence's sampled token at its next
 * position, and each further batch the token every still-drafting sequence picked. A sequence picks either
 * greedily, the argmax and its probability among the 10 most likely tokens ({@link DraftModelLoader#pick},
 * llama.cpp's drafter top-k 10), or — with probabilistic drafting (vLLM {@code draft_sample_method=probabilistic})
 * and a request that samples — by a draw from the draft distribution {@code q = softmax(logits / T)} at the
 * request's temperature {@code T}. Drafting a sequence stops at its limit, at a token whose probability is below
 * {@code p_min}, or at a token outside the target vocabulary. A probabilistic draw's proposal distribution
 * ({@link #draftProbs}) is {@code q} restricted to the tokens that would have been proposed (target vocabulary,
 * probability at least {@code p_min}) and renormalized: the distribution a proposed token was drawn from, which
 * keeps rejection sampling lossless; a greedy pick has none (one-hot).
 *
 * <p>Drafted rows stay in the draft memory; the next draft drops the ones the target rejected by the same prefix
 * comparison, so {@link #accept} keeps no state. A step whose draft rows do not fit the draft memory drops the
 * requested sequences' draft rows and proposes nothing for them that step (they re-prefill at their next draft).
 *
 * <p>Two vocabularies that share token ids are used as they are (llama.cpp
 * {@code common_speculative_are_compatible}); with heterogeneous-vocabulary drafting the two may differ, and a
 * {@link DraftVocabMapping} translates the target's tokens into the draft's, masks the draft logits to the token-level
 * intersection and translates the proposals back. vLLM restricts that to greedy drafting, and so does this drafter:
 * the proposal distributions of {@code probabilistic} drafting index the target vocabulary directly.
 *
 * <p>The drafter owns the draft model and closes it. Not thread-safe: it runs on the engine's thread.
 */
public final class BatchModelDrafter implements BatchDrafter {

    private static final Logger LOGGER = Logger.getLogger(BatchModelDrafter.class.getName());

    private final EngineModel draft;
    private final ForwardBackend backend;
    private final PagedSequences draftSequences;
    private final SequenceMemory target;
    private final DraftVocabMapping vocab;
    private final int targetVocab;
    private final float pMin;
    private final boolean probabilistic;
    /** Per sequence, the branching factor of its tree proposals; 0 drafts chains only. */
    private final int treeBranches;
    /** Per sequence, the draft position its last tree proposal's rows start at, or -1 (the rows a sync drops). */
    private final int[] treeRowStart;
    /** Per sequence, the proposal rows of its last draft by position (null rows for deterministic picks). */
    private final float[][][] probs;
    /** Per sequence, the draft-side watermarker of its request and the configuration it was built from. */
    private final DraftWatermarker[] draftWatermarks;
    private final WatermarkConfig[] draftWatermarkConfigs;
    private boolean closed;

    /**
     * A drafter of vocabularies that share token ids.
     *
     * @param draft         the draft model, over paged memory with at least the target's sequences; owned by this
     *                      drafter
     * @param targetModel   the target whose paged sequences the drafts follow
     * @param pMin          drafting stops at a token whose probability is below this ({@code --draft-p-min})
     * @param probabilistic whether sampling requests draft by a draw from the draft distribution
     */
    public BatchModelDrafter(EngineModel draft, EngineModel targetModel, float pMin, boolean probabilistic) {
        this(draft, targetModel, pMin, probabilistic, false);
    }

    /**
     * @param draft              the draft model, over paged memory with at least the target's sequences; owned by this
     *                           drafter
     * @param targetModel        the target whose paged sequences the drafts follow
     * @param pMin               drafting stops at a token whose probability is below this ({@code --draft-p-min})
     * @param probabilistic      whether sampling requests draft by a draw from the draft distribution
     * @param heterogeneousVocab whether to draft across two vocabularies by their token-level intersection (vLLM
     *                           {@code use_heterogeneous_vocab}); greedy drafting only
     * @throws IllegalArgumentException when the draft model does not generate autoregressively, its memory is not
     *                                  paged or holds fewer sequences than the target's, the vocabularies are
     *                                  incompatible, {@code pMin} is outside {@code [0, 1]}, or heterogeneous
     *                                  drafting is combined with probabilistic drafting
     */
    public BatchModelDrafter(EngineModel draft, EngineModel targetModel, float pMin, boolean probabilistic,
                             boolean heterogeneousVocab) {
        this(draft, targetModel, pMin, probabilistic, heterogeneousVocab, 0);
    }

    /**
     * @param draft              the draft model, over paged memory with at least the target's sequences; owned by this
     *                           drafter
     * @param targetModel        the target whose paged sequences the drafts follow
     * @param pMin               drafting stops at a token whose probability is below this ({@code --draft-p-min})
     * @param probabilistic      whether sampling requests draft by a draw from the draft distribution
     * @param heterogeneousVocab whether to draft across two vocabularies by their token-level intersection (vLLM
     *                           {@code use_heterogeneous_vocab}); greedy drafting only
     * @param treeBranches       the branching factor of the tree proposals of {@link #draftTreeBatch}
     *                           ({@code neuralfree.draft.tree.branches}); 0 proposes chains only, 1 a linear tree
     * @throws IllegalArgumentException when the draft model does not generate autoregressively, its memory is not
     *                                  paged or holds fewer sequences than the target's, the vocabularies are
     *                                  incompatible, {@code pMin} is outside {@code [0, 1]}, heterogeneous
     *                                  drafting is combined with probabilistic drafting, or {@code treeBranches}
     *                                  is negative
     * @throws UnsupportedOperationException when {@code treeBranches} is positive and the draft model's attention
     *                                       takes no part in a tree-masked batch
     */
    public BatchModelDrafter(EngineModel draft, EngineModel targetModel, float pMin, boolean probabilistic,
                             boolean heterogeneousVocab, int treeBranches) {
        this.draft = Objects.requireNonNull(draft, "draft");
        Objects.requireNonNull(targetModel, "target");
        if (!(pMin >= 0f && pMin <= 1f)) {
            throw new IllegalArgumentException("draft p_min must be in [0, 1], got " + pMin);
        }
        if (treeBranches < 0) {
            throw new IllegalArgumentException("draft tree branches must not be negative, got " + treeBranches);
        }
        if (heterogeneousVocab && probabilistic) {
            throw new IllegalArgumentException("heterogeneous-vocabulary drafting supports greedy drafting only; use"
                    + " draft_sample_method=greedy (or unset) with heterogeneous vocabularies");
        }
        if (draft.supportClass() != SupportClass.GEN) {
            throw new IllegalArgumentException("draft model " + draft.architecture() + " (" + draft.supportClass()
                    + ") does not generate text autoregressively");
        }
        if (treeBranches >= 1) {
            BatchEngineFactory.checkBatchTreeAttention("draft model " + draft.name(), draft.spec());
        }
        this.target = pagedOf(targetModel, "target");
        this.draftSequences = pagedOf(draft, "draft model");
        if (draftSequences.maxSequences() < target.maxSequences()) {
            throw new IllegalArgumentException("the draft model's paged memory holds " + draftSequences.maxSequences()
                    + " sequences, the target's " + target.maxSequences());
        }
        if (heterogeneousVocab) {
            this.vocab = DraftVocabMapping.of(targetModel.tokenizer(), targetModel.vocabSize(), draft.tokenizer(),
                    draft.vocabSize());
        } else {
            String incompatible = ModelDrafter.vocabMismatch(targetModel.tokenizer(), draft.tokenizer());
            if (incompatible != null) {
                throw new IllegalArgumentException("draft model " + draft.name() + " cannot draft for "
                        + targetModel.name() + ": " + incompatible);
            }
            this.vocab = DraftVocabMapping.identity(targetModel.vocabSize(), draft.vocabSize());
        }
        this.backend = draft.forwardBackend();
        this.targetVocab = targetModel.vocabSize();
        this.pMin = pMin;
        this.probabilistic = probabilistic;
        this.treeBranches = treeBranches;
        this.treeRowStart = new int[target.maxSequences()];
        Arrays.fill(this.treeRowStart, -1);
        this.probs = new float[target.maxSequences()][][];
        this.draftWatermarks = new DraftWatermarker[target.maxSequences()];
        this.draftWatermarkConfigs = new WatermarkConfig[target.maxSequences()];
    }

    private static PagedSequences pagedOf(EngineModel model, String role) {
        ModelMemory memory = model.forwardBackend().memory();
        PagedSequences paged = memory == null ? null : memory.pagedSequences();
        if (paged == null) {
            throw new IllegalArgumentException("the " + role + " " + model.name() + " runs without paged memory; a"
                    + " batch draft model drafts every sequence over paged memory");
        }
        return paged;
    }

    /** The draft model. */
    public EngineModel draftModel() {
        return draft;
    }

    /** A greedy draft of one sequence ({@link #draftBatch} over a single request). */
    @Override
    public int[] draft(int seq, int lastToken, int n) {
        if (n <= 0) {
            return new int[0];
        }
        return draftBatch(Collections.singletonList(new Request(seq, lastToken, n, 0f, new SplittableRandom(0))))[0];
    }

    /** One requested sequence while its draft runs. */
    private static final class Drafting {
        final Request request;
        /** The request's index in the step's request list. */
        final int index;
        final int base;
        final int count;
        final int[] out;
        final float[][] rows;
        /** The sequence's draft-side watermarker, or null for the ordinary pick. */
        final DraftWatermarker watermark;
        int len;
        /** The next token to decode, in the draft vocabulary. */
        int nextDraftToken;

        Drafting(Request request, int index, int base, int count, DraftWatermarker watermark) {
            this.request = request;
            this.index = index;
            this.base = base;
            this.count = count;
            this.out = new int[count];
            this.rows = new float[count][];
            this.watermark = watermark;
            this.nextDraftToken = request.lastToken();
        }
    }

    @Override
    public int[][] draftBatch(List<Request> requests) {
        ensureOpen();
        int[][] answers = new int[requests.size()][];
        List<Drafting> running = new ArrayList<>();
        List<int[]> missing = new ArrayList<>();
        int rows = 0;
        for (int i = 0; i < answers.length; i++) {
            Request r = requests.get(i);
            checkSequence(r.seq());
            probs[r.seq()] = null;
            answers[i] = new int[0];
            int[] history = target.sequence(r.seq()).textTokens();
            int count = Math.min(r.limit(), draftSequences.sequence(r.seq()).nCtx() - history.length);
            if (count <= 0 || r.lastToken() < 0 || r.lastToken() >= targetVocab
                    || !vocab.canStartDraft(r.lastToken())) {
                continue;
            }
            DraftWatermarker watermark = watermarkFor(r, history);
            if (watermark != null) {
                watermark.beginRound();
            }
            int[] tail = sync(r.seq(), history);
            Drafting drafting = new Drafting(r, i, history.length, count, watermark);
            drafting.nextDraftToken = vocab.toDraft(r.lastToken());
            running.add(drafting);
            missing.add(tail);
            rows += tail.length + 1;
        }
        if (running.isEmpty()) {
            return answers;
        }
        try {
            run(running, missing, rows);
        } catch (MemoryFullException e) {
            LOGGER.fine("the draft rows of " + running.size() + " sequences do not fit the draft memory: " + e);
            for (Drafting d : running) {
                draftSequences.remove(d.request.seq(), 0, -1);
                d.len = 0;
            }
        }
        for (Drafting d : running) {
            answers[d.index] = Arrays.copyOf(d.out, d.len);
            probs[d.request.seq()] = Arrays.copyOf(d.rows, d.len);
        }
        return answers;
    }

    /** The empty proposal of a request whose draft produced nothing. */
    private static final TreeProposal NO_TREE = emptyProposal();

    /** Rows one masked draft forward holds at most: the per-row ancestor mask's reach. */
    private static final int MAX_TREE_ROWS = 64;

    private static TreeProposal emptyProposal() {
        return new TreeProposal(new DraftTree(new ArrayList<>()), new double[0]);
    }

    /** One requested sequence while its tree draft runs: its nodes in breadth-first build order. */
    private static final class TreeSeq {
        final Request request;
        /** The request's index in the step's request list. */
        final int index;
        /** The draft position the root row (the pending token) stores at. */
        final int base;
        /** The node cap the request's limit and the draft context set. */
        final int budget;
        /** The draft tokens between the kept prefix and the root row, stored before the root row runs. */
        final int[] tail;
        final List<Integer> tokens = new ArrayList<>();
        /** Each node's parent index, -1 at depth 1 (a child of the root row's candidates). */
        final List<Integer> parents = new ArrayList<>();
        final List<Integer> depths = new ArrayList<>();
        /** Cumulative path log-probabilities, node 0's first. */
        final List<Double> scores = new ArrayList<>();
        /** The logits rows of the current depth's nodes, parallel to {@code [frontierFrom, tokens.size())}. */
        float[][] frontier;
        /** The index of the current depth's first node. */
        int frontierFrom;
        /** The node depth the current frontier sits at. */
        int depth;

        TreeSeq(Request request, int index, int base, int budget, int[] tail) {
            this.request = request;
            this.index = index;
            this.base = base;
            this.budget = budget;
            this.tail = tail;
        }
    }

    /**
     * The tree form of {@link #draftBatch} (SpecPilot's batched tree drafting over one draft model, arXiv
     * 2611.02501): every requested sequence expands its own ragged draft tree breadth-first, all sequences' rows of
     * one depth in shared masked forward batches. A depth's rows re-run the sequence's whole tree under the exact
     * per-row ancestor masks — a branch sees only its own path among the tree rows (the rest of the tree leaves the
     * fresh block by the mask), the committed prefix by position — because the draft follows one sequence per target
     * slot, whose causal rule alone would show a branch its aunts. Each node's path score is its parent's plus the
     * candidate's log-softmax over the parent row, so scores descend along any path and the breadth-first build
     * order sorts stable into the best-first node list the verify budget cuts (SpecPilot's {@code ddtree}).
     *
     * <p>Per node, the top {@code treeBranches} candidates extend (distinct target ids, best first); no sample is
     * drawn — the target's verify walk picks among the branches, so a tree draft stays exact whatever the request
     * samples. The rows of one chunk share one masked forward: the requests group so their tree rows fit the
     * draft's ubatch and the 64-row mask, while the synced tails and root rows ride the plain root batch the
     * backend splits by its ubatch. Each sequence's tree rows store past its synced prefix and leave at the
     * next sync, the way the chain drafter's rows do.
     *
     * @throws IllegalStateException when a forward returns the wrong row count or a closed drafter is asked
     */
    @Override
    public TreeProposal[] draftTreeBatch(List<Request> requests) {
        ensureOpen();
        TreeProposal[] answers = new TreeProposal[requests.size()];
        List<TreeSeq> ready = new ArrayList<>();
        int rowCap = Math.min(MAX_TREE_ROWS, draftUbatch());
        for (int i = 0; i < requests.size(); i++) {
            Request r = requests.get(i);
            checkSequence(r.seq());
            probs[r.seq()] = null;
            answers[i] = NO_TREE;
            int[] history = target.sequence(r.seq()).textTokens();
            int budget = Math.min(Math.min(r.limit(), rowCap - 1),
                    draftSequences.sequence(r.seq()).nCtx() - history.length - 1);
            if (budget <= 0 || r.lastToken() < 0 || r.lastToken() >= targetVocab
                    || !vocab.canStartDraft(r.lastToken())) {
                continue;
            }
            int[] tail = sync(r.seq(), history);
            ready.add(new TreeSeq(r, i, history.length, budget, tail));
        }
        int from = 0;
        while (from < ready.size()) {
            int to = from;
            int rows = 0;
            while (to < ready.size() && rows + ready.get(to).budget <= rowCap) {
                rows += ready.get(to).budget;
                to++;
            }
            buildTrees(ready.subList(from, to), answers);
            from = to;
        }
        return answers;
    }

    /** The draft model's ubatch, the bound on one masked draft forward's rows. */
    private int draftUbatch() {
        return draft.forwardBackend().memory().plan().ubatch();
    }

    /**
     * Runs one chunk's breadth-first tree build: the root rows' logits select the depth-1 nodes, then every depth's
     * rows forward masked and their logits select the next depth's candidates. A chunk whose rows leave the draft
     * memory drafts nothing for its sequences (they re-sync at their next draft), as {@link #draftBatch} does.
     */
    private void buildTrees(List<TreeSeq> chunk, TreeProposal[] answers) {
        int rows = 0;
        for (TreeSeq t : chunk) {
            rows += t.tail.length + 1;
        }
        ForwardBatch.Builder roots = ForwardBatch.builder(rows);
        for (TreeSeq t : chunk) {
            int pos = t.base - t.tail.length;
            for (int j = 0; j < t.tail.length; j++) {
                roots.add(t.tail[j], pos + j, t.request.seq(), false);
            }
            roots.add(vocab.toDraft(t.request.lastToken()), t.base, t.request.seq(), true);
        }
        float[][] rootLogits;
        try {
            rootLogits = backend.forwardBatch(roots.build());
            if (rootLogits.length != chunk.size()) {
                throw new IllegalStateException("the draft model returned " + rootLogits.length + " rows for a"
                        + " batch of " + chunk.size());
            }
            for (int i = 0; i < chunk.size(); i++) {
                expand(chunk.get(i), new float[][] {rootLogits[i]}, -1);
            }
            while (chunk.stream().anyMatch(t -> t.frontier != null && t.frontier.length > 0)) {
                forwardTreeRows(chunk);
                boolean grew = false;
                for (TreeSeq t : chunk) {
                    grew |= expand(t, t.frontier, t.frontierFrom);
                }
                if (!grew) {
                    break;
                }
            }
        } catch (MemoryFullException e) {
            LOGGER.fine("the tree rows of " + chunk.size() + " sequences do not fit the draft memory: " + e);
            for (TreeSeq t : chunk) {
                draftSequences.remove(t.request.seq(), t.base + 1, -1);
                treeRowStart[t.request.seq()] = -1;
            }
            chunk.clear();
        }
        for (TreeSeq t : chunk) {
            answers[t.index] = proposalOf(t);
            treeRowStart[t.request.seq()] = t.tokens.isEmpty() ? -1 : t.base + 1;
        }
        for (TreeSeq t : chunk) {
            if (answers[t.index].tree().size() == 0) {
                draftSequences.remove(t.request.seq(), t.base + 1, -1);
                treeRowStart[t.request.seq()] = -1;
            }
        }
    }

    /**
     * Forwards every sequence's tree rows of one chunk under the exact ancestor masks and hands the current
     * depth's logits rows to the sequences' frontiers. The rows re-store each depth (the previous depth's cells
     * leave first), so the draft KV always holds one copy of the tree.
     */
    private void forwardTreeRows(List<TreeSeq> chunk) {
        List<TreeSeq> active = new ArrayList<>(chunk.size());
        int rows = 0;
        for (TreeSeq t : chunk) {
            if (t.frontier != null && t.frontier.length > 0) {
                active.add(t);
                draftSequences.remove(t.request.seq(), t.base + 1, -1);
                rows += t.tokens.size();
            }
        }
        if (active.isEmpty()) {
            return;
        }
        ForwardBatch.Builder batch = ForwardBatch.builder(rows);
        long[] mask = new long[rows];
        int row = 0;
        for (TreeSeq t : active) {
            for (int i = 0; i < t.tokens.size(); i++) {
                batch.add(vocab.toDraft(t.tokens.get(i)), t.base + 1 + i, t.request.seq(), true);
                long bits = 0L;
                for (int j = i; j >= 0; j = t.parents.get(j)) {
                    bits |= 1L << j;
                }
                mask[row++] = bits;
            }
        }
        float[][] out = backend.forwardBatch(batch.treeMask(mask).build());
        if (out.length != rows) {
            throw new IllegalStateException("the draft model returned " + out.length + " rows for a batch of "
                    + rows);
        }
        int at = 0;
        for (TreeSeq t : active) {
            int n = t.tokens.size();
            t.frontier = new float[n - t.frontierFrom][];
            for (int i = t.frontierFrom; i < n; i++) {
                t.frontier[i - t.frontierFrom] = out[at + i];
            }
            at += n;
        }
    }

    /**
     * Extends {@code t} with the children of {@code rows}' logits rows: per row, the top {@code treeBranches}
     * candidates (distinct target ids, best first) score the parent's path plus their log-softmax. {@code
     * parentFrom} names the node index of the first row's parent, or -1 for the root row (its children sit at
     * depth 1, parents -1). False when the frontier is empty or the tree is spent.
     */
    private boolean expand(TreeSeq t, float[][] rows, int parentFrom) {
        if (rows == null || rows.length == 0 || t.tokens.size() >= t.budget) {
            t.frontier = null;
            return false;
        }
        int grew = 0;
        for (int f = 0; f < rows.length && t.tokens.size() < t.budget; f++) {
            int parent = parentFrom < 0 ? -1 : parentFrom + f;
            int[] ids = treeCandidates(rows[f], Math.min(treeBranches, t.budget - t.tokens.size()));
            for (int id : ids) {
                t.tokens.add(vocab.toTarget(id));
                t.parents.add(parent);
                t.depths.add(parent < 0 ? 1 : t.depths.get(parent) + 1);
                t.scores.add(parent < 0 ? 0.0 : t.scores.get(parent) + logSoftmaxAt(rows[f], id));
                grew++;
            }
        }
        if (grew == 0) {
            t.frontier = null;
            return false;
        }
        t.frontierFrom = t.tokens.size() - grew;
        t.depth = t.depths.get(t.frontierFrom);
        return true;
    }

    /**
     * The top {@code k} draft ids of a logits row after the draft vocabulary's mask, best first, target-id
     * duplicates keeping their highest-ranked occurrence.
     */
    private int[] treeCandidates(float[] logits, int k) {
        vocab.constrain(logits);
        Integer[] order = new Integer[logits.length];
        int n = 0;
        for (int i = 0; i < logits.length; i++) {
            float v = logits[i];
            if (!Float.isNaN(v) && v != Float.NEGATIVE_INFINITY) {
                order[n++] = i;
            }
        }
        java.util.Arrays.sort(order, 0, n, (a, b) -> Float.compare(logits[b], logits[a]));
        Set<Integer> seen = new HashSet<>();
        int[] ids = new int[Math.min(k, n)];
        int kept = 0;
        for (int i = 0; i < n && kept < ids.length; i++) {
            int targetToken = vocab.toTarget(order[i]);
            if (targetToken < 0 || !seen.add(targetToken)) {
                continue;
            }
            ids[kept++] = order[i];
        }
        return kept == ids.length ? ids : Arrays.copyOf(ids, kept);
    }

    /** The log-softmax of draft id {@code id} over a row's finite logits. */
    private static double logSoftmaxAt(float[] logits, int id) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (!Float.isNaN(v) && v != Float.NEGATIVE_INFINITY) {
                max = Math.max(max, v);
            }
        }
        if (max == Double.NEGATIVE_INFINITY) {
            throw new IllegalStateException("the draft logits hold no finite value");
        }
        double sum = 0.0;
        for (float v : logits) {
            if (!Float.isNaN(v) && v != Float.NEGATIVE_INFINITY) {
                sum += Math.exp(v - max);
            }
        }
        return (logits[id] - max) - Math.log(sum);
    }

    /**
     * The proposal of a built tree: the nodes sorted stable by path score descending (the best-first order the
     * verify budget cuts, parents ahead of children — a child never outranks its parent), their parent indices
     * remapped into the sorted order.
     */
    private static TreeProposal proposalOf(TreeSeq t) {
        int n = t.tokens.size();
        if (n == 0) {
            return NO_TREE;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> {
            int byScore = Double.compare(t.scores.get(b), t.scores.get(a));
            return byScore != 0 ? byScore : Integer.compare(a, b);
        });
        int[] remap = new int[n];
        for (int pos = 0; pos < n; pos++) {
            remap[order[pos]] = pos;
        }
        List<DraftTree.Node> nodes = new ArrayList<>(n);
        double[] scores = new double[n];
        for (int pos = 0; pos < n; pos++) {
            int old = order[pos];
            int parent = t.parents.get(old);
            nodes.add(new DraftTree.Node(t.tokens.get(old), parent < 0 ? -1 : remap[parent], t.depths.get(old), 0));
            scores[pos] = t.scores.get(old);
        }
        return new TreeProposal(new DraftTree(nodes), scores);
    }

    @Override
    public boolean proposesTrees() {
        return treeBranches >= 1;
    }

    /**
     * Brings draft sequence {@code seq} to the target's text tokens: keeps the common prefix of the target's tokens
     * translated to the draft vocabulary, drops the rest (all of it when the memory refuses the partial removal) and
     * returns the missing target tokens translated to the draft vocabulary. A tree proposal's rows never keep: the
     * masked rows' keys miss the aunts a linear continuation attends, so the prefix stops before them.
     */
    private int[] sync(int seq, int[] history) {
        int[] stored = draftSequences.sequence(seq).tokens();
        int keep = 0;
        int limit = Math.min(stored.length, history.length);
        while (keep < limit && stored[keep] == vocab.toDraft(history[keep])) {
            keep++;
        }
        if (treeRowStart[seq] >= 0 && keep > treeRowStart[seq]) {
            keep = Math.min(keep, treeRowStart[seq]);
        }
        if (keep < stored.length) {
            if (draftSequences.removeRefusal(seq, keep, -1) != null) {
                keep = 0;
            }
            draftSequences.remove(seq, keep, -1);
        }
        int[] missing = new int[history.length - keep];
        for (int i = 0; i < missing.length; i++) {
            missing[i] = vocab.toDraft(history[keep + i]);
        }
        return missing;
    }

    /** The draft batches of one step: the synced tails with every sampled token, then one row per drafting step. */
    private void run(List<Drafting> running, List<int[]> missing, int rows) {
        ForwardBatch.Builder first = ForwardBatch.builder(rows);
        for (int i = 0; i < running.size(); i++) {
            Drafting d = running.get(i);
            int[] tail = missing.get(i);
            int pos = d.base - tail.length;
            for (int t = 0; t < tail.length; t++) {
                first.add(tail[t], pos + t, d.request.seq(), false);
            }
            first.add(d.nextDraftToken, d.base, d.request.seq(), true);
        }
        List<Drafting> step = running;
        float[][] logits = backend.forwardBatch(first.build());
        while (true) {
            List<Drafting> next = new ArrayList<>(step.size());
            for (int i = 0; i < step.size(); i++) {
                Drafting d = step.get(i);
                if (pick(d, logits[i]) && d.len < d.count) {
                    next.add(d);
                }
            }
            if (next.isEmpty()) {
                return;
            }
            ForwardBatch.Builder b = ForwardBatch.builder(next.size());
            for (Drafting d : next) {
                b.add(d.nextDraftToken, d.base + d.len, d.request.seq(), true);
            }
            logits = backend.forwardBatch(b.build());
            step = next;
        }
    }

    /**
     * The sequence's draft-side watermarker for {@code request}, synced to {@code history} and ready for a round;
     * null when the request does not watermark, drafts greedily, or asks for a target-only watermark. A changed
     * configuration rebuilds the watermarker.
     */
    private DraftWatermarker watermarkFor(Request request, int[] history) {
        WatermarkConfig config = request.watermark();
        if (config == null || !(request.temperature() > 0f)) {
            return null;
        }
        int seq = request.seq();
        if (!config.equals(draftWatermarkConfigs[seq])) {
            draftWatermarkConfigs[seq] = config;
            draftWatermarks[seq] = DraftWatermarker.of(config);
            if (draftWatermarks[seq] != null) {
                draftWatermarks[seq].reset(request.promptLength());
            }
        }
        DraftWatermarker watermark = draftWatermarks[seq];
        if (watermark != null) {
            watermark.sync(history, request.lastToken());
        }
        return watermark;
    }

    /** Picks the next draft token of {@code d} from its logits row; false when its draft ends here. */
    private boolean pick(Drafting d, float[] logits) {
        int targetToken;
        float[] row = null;
        float temperature = d.request.temperature();
        if (d.watermark != null) {
            vocab.constrain(logits);
            DraftModelLoader.Pick p = DraftModelLoader.pick(logits, logits.length, d.watermark, temperature,
                    d.request.rng()::nextDouble);
            targetToken = vocab.toTarget(p.id);
            if (targetToken < 0 || p.p < pMin) {
                return false;
            }
            d.nextDraftToken = p.id;
            row = p.proposal;
        } else if (probabilistic && temperature > 0f) {
            float[] q = softmax(logits, temperature);
            int token = sample(q, d.request.rng().nextDouble());
            targetToken = vocab.toTarget(token);
            if (targetToken < 0 || q[token] < pMin) {
                return false;
            }
            d.nextDraftToken = token;
            row = admissible(q);
        } else {
            vocab.constrain(logits);
            DraftModelLoader.Pick p = DraftModelLoader.pick(logits, logits.length);
            targetToken = vocab.toTarget(p.id);
            if (p.p < pMin || targetToken < 0) {
                return false;
            }
            d.nextDraftToken = p.id;
        }
        d.out[d.len] = targetToken;
        d.rows[d.len] = row;
        d.len++;
        return true;
    }

    /** {@code softmax(logits / t)} over the draft vocabulary. */
    static float[] softmax(float[] logits, float t) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        if (max == Double.NEGATIVE_INFINITY) {
            throw new IllegalStateException("the draft logits hold no finite value");
        }
        double[] e = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            e[i] = Math.exp((logits[i] - max) / t);
            sum += e[i];
        }
        float[] q = new float[logits.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) (e[i] / sum);
        }
        return q;
    }

    /** The token of {@code q} whose cumulative probability first reaches {@code u * sum(q)}. */
    static int sample(float[] q, double u) {
        double total = 0.0;
        for (float v : q) {
            total += v;
        }
        double target = u * total;
        double run = 0.0;
        int last = -1;
        for (int i = 0; i < q.length; i++) {
            if (q[i] > 0f) {
                last = i;
                run += q[i];
                if (run > target) {
                    return i;
                }
            }
        }
        return last;
    }

    /**
     * {@code q} over the target vocabulary, restricted to the tokens a draw proposes (probability at least
     * {@code p_min}) and renormalized: the distribution of a proposed token.
     */
    private float[] admissible(float[] q) {
        float[] row = new float[targetVocab];
        double sum = 0.0;
        for (int i = 0; i < Math.min(q.length, targetVocab); i++) {
            if (q[i] >= pMin && q[i] > 0f) {
                row[i] = q[i];
                sum += q[i];
            }
        }
        for (int i = 0; i < row.length; i++) {
            row[i] = (float) (row[i] / sum);
        }
        return row;
    }

    @Override
    public float[] draftProbs(int seq, int position) {
        checkSequence(seq);
        float[][] rows = probs[seq];
        if (rows == null || position < 0 || position >= rows.length) {
            throw new IllegalStateException("sequence " + seq + " has no drafted token at position " + position);
        }
        return rows[position] == null ? null : rows[position].clone();
    }

    /** Verification keeps no drafter state: the next draft re-aligns the draft memory by prefix. */
    @Override
    public void accept(int seq, int nAccepted) {
        checkSequence(seq);
        if (nAccepted < 0) {
            throw new IllegalArgumentException("accepted count must not be negative, got " + nAccepted);
        }
    }

    /** Drops the draft rows and proposal rows of {@code seq}. */
    @Override
    public void reset(int seq) {
        checkSequence(seq);
        probs[seq] = null;
        draftWatermarks[seq] = null;
        draftWatermarkConfigs[seq] = null;
        if (!closed) {
            draftSequences.remove(seq, 0, -1);
        }
    }

    @Override
    public boolean supportsWatermarking() {
        return vocab.isIdentity();
    }

    /** Closes the draft model. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        try {
            draft.close();
        } catch (IOException e) {
            throw new UncheckedIOException("closing the draft model " + draft.name() + " failed", e);
        }
    }

    private void checkSequence(int seq) {
        if (seq < 0 || seq >= probs.length) {
            throw new IndexOutOfBoundsException("sequence " + seq + " outside the target's " + probs.length
                    + " sequences");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the batch draft model drafter is closed");
        }
    }

    @Override
    public String toString() {
        return "BatchModelDrafter{" + draft.name() + " (" + draft.architecture() + "), pMin=" + pMin
                + (probabilistic ? ", probabilistic" : "")
                + (vocab.isIdentity() ? "" : ", heterogeneous vocab, intersection=" + vocab.intersectionSize())
                + '}';
    }
}
