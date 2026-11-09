package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SessionFile;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.CpPageShard;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.memory.paged.RowMove;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The mirror that keeps every rank's shard KV cache in step with the model device's memory: it replays the step
 * uploads, the sequence appends, the paged steps and page copies, the shifts and the resets onto the rank memories,
 * so the rank caches hold exactly the positions the main sequence holds. {@link TensorParallelGroup} owns the
 * ranks; the mirror owns the step bookkeeping the replay needs — the current step window, the last applied one,
 * and the paged page-ownership shard of a context-parallel split.
 */
final class TpMemoryMirror {

    /** Container magic of a tensor-parallel session ("NFTP"). */
    private static final int SESSION_MAGIC = 0x4E465450;
    private static final int SESSION_VERSION = 1;

    private final TensorParallelGroup group;
    /** The current step's first position ({@code beforeStep}); the append records split their rows from it. */
    private int stepPos0 = -1;
    /** Page owner bookkeeping of a context-parallel paged split ({@link CpPageShard}), or null otherwise. */
    private CpPageShard pageShard;
    private int lastStepPos0 = -1;
    private int lastStepTokens = -1;

    TpMemoryMirror(TensorParallelGroup group) {
        this.group = group;
    }

    /**
     * Uploads the current step's positions to every rank's step inputs, once per step. A tensor split's rank
     * caches address their rows through these buffers, exactly as the main memory does. A context split's rank
     * caches hold only their own context block, so each rank's step inputs store the positions of that block and
     * leave the other rows at {@code -1}; the host mirrors keep the whole sequence for rollback and reset.
     */
    void uploadStepInputs(int pos0, int nTokens) {
        if (group.attentionPlan() == null) {
            return;
        }
        if (pos0 == lastStepPos0 && nTokens == lastStepTokens) {
            return;
        }
        this.stepPos0 = pos0;
        for (int rank = 0; rank < group.rankCount(); rank++) {
            int skip = group.prefillSplitOffset(rank, pos0, nTokens);
            if (skip < nTokens) {
                group.rankMemory(rank).stepInputs().upload(pos0 + skip, nTokens - skip);
            }
        }
        lastStepPos0 = pos0;
        lastStepTokens = nTokens;
    }

    /** Records a step's tokens on every rank's sequence record: the rank's own rows under the prefill query split. */
    void appendStep(int[] tokens, int off, int n) {
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            int skip = group.prefillSplitOffset(rank, stepPos0, n);
            if (skip < n) {
                group.rankMemory(rank).sequence().append(tokens, off + skip, n - skip);
            }
        }
    }

    /** Records a step of embedding rows, carrying {@code token}, on every rank's sequence record. */
    void appendEmbeddingStep(int n, int token) {
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            int skip = group.prefillSplitOffset(rank, stepPos0, n);
            if (skip < n) {
                group.rankMemory(rank).sequence().appendEmbeddings(n - skip, token);
            }
        }
    }

    /** True when every rank can drop the last {@code nTokens} positions. */
    boolean canRollback(int nTokens) {
        if (group.attentionPlan() == null) {
            return true;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            if (!group.rankMemory(rank).canRollback(nTokens)) {
                return false;
            }
        }
        return true;
    }

    /** Empties every rank's cache. */
    void reset() {
        lastStepPos0 = -1;
        lastStepTokens = -1;
        if (group.attentionPlan() == null) {
            return;
        }
        if (pageShard != null) {
            pageShard.reset();
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            group.rankMemory(rank).reset();
        }
    }

    /** Drops the last {@code nTokens} positions from every rank's cache. */
    void rollback(int nTokens) {
        lastStepPos0 = -1;
        lastStepTokens = -1;
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            group.rankMemory(rank).rollback(nTokens);
        }
    }

    /** Notes the coordinator's packed step window: the appends split the step's rows from it, and the next
     * non-trunk step upload re-runs (the trunk uploads packed windows the plain upload would otherwise see
     * as already applied). */
    void noteStepWindow(int pos0, int nTokens) {
        this.stepPos0 = pos0;
        this.lastStepPos0 = -1;
        this.lastStepTokens = -1;
    }

    /**
     * Uploads a paged step to every rank cache, so each rank's paged attention kernels read the page tables, cell
     * rows and positions the main memory's bookkeeping prepared. A tensor split replays the main step verbatim:
     * {@code mainMirror} (the model device's host {@code kv_pos} mirror after it applied the step's copies,
     * removals and trims) installs on every rank before its own step upload. A context-parallel paged split
     * translates instead ({@link CpPageShard#translateStep}): every rank receives the step over its own page
     * ownership, and the main memory's authoritative mirror ({@code mainPaged}) drives the reconcile.
     */
    void beginPagedStep(PagedStep step, int[] positions, int[][] ropeSections, boolean nonCausal,
                        PagedSequences mainPaged, int[][] mainMirror) {
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            DeviceModelMemory memory = group.rankMemory(rank);
            try {
                if (pageShard != null) {
                    PagedStep rankStep = pageShard.translateStep(step, mainPaged, memory, rank);
                    memory.beginPagedStep(rankStep, positions, ropeSections, nonCausal);
                } else {
                    if (mainMirror != null) {
                        memory.installPagedMirror(mainMirror);
                    }
                    memory.beginPagedStep(step, positions, ropeSections, nonCausal);
                }
            } catch (IllegalArgumentException | IllegalStateException | UnsupportedOperationException e) {
                throw new UnsupportedOperationException(group.rankName(rank) + " cannot run the paged step: "
                        + e.getMessage() + "; the rank's page pools and the main memory's bookkeeping must hold"
                        + " the same page geometry", e);
            }
        }
    }

    /** Clears the paged step from every rank cache. */
    void endPagedStep() {
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            group.rankMemory(rank).endPagedStep();
        }
    }

    /**
     * Runs one page copy of the copy-on-write protocol on every rank cache. A context-parallel paged split
     * translates the copy into the rank's own page ids ({@link CpPageShard#translateCopy}); a tensor split replays
     * the main copy verbatim.
     */
    void copyPage(PageCopy copy, PagedSequences mainPaged) {
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            DeviceModelMemory memory = group.rankMemory(rank);
            if (pageShard != null) {
                PageCopy rankCopy = pageShard.translateCopy(copy, mainPaged, rank);
                if (rankCopy != null) {
                    memory.copyPage(rankCopy);
                }
            } else {
                memory.copyPage(copy);
            }
        }
    }

    /**
     * Why a per-sequence context shift over the rank caches would refuse, or null: a context-parallel paged split
     * refuses it by name (the shift's row moves change a page's first stored position, which the rank ownership
     * cannot follow); a tensor split needs every rank cache to hold paged sequences whose sequence view can move
     * the rows.
     */
    String sequenceShiftRefusal() {
        if (group.attentionPlan() == null) {
            return null;
        }
        if (pageShard != null) {
            return "a context-parallel paged split assigns every page to the rank owning its first stored"
                    + " position, and a sequence shift moves rows between pages; run the shift on one device";
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            DeviceModelMemory memory = group.rankMemory(rank);
            if (memory == null) {
                continue;
            }
            if (memory.pagedSequences() == null) {
                return group.rankName(rank) + " holds no paged sequences";
            }
            String refusal = memory.sequences().shiftRefusal();
            if (refusal != null) {
                return group.rankName(rank) + ": " + refusal;
            }
        }
        return null;
    }

    /**
     * Applies a shift the model device planned to every rank cache: the shift's page copies run, each rank installs
     * the plan's per-row RoPE deltas (its mixers re-rotate by them) and the model device's post-shift {@code kv_pos}
     * mirror, and {@link #moveShiftedRows} later carries the {@code moves} on every rank. The rank caches keep no
     * paged sequence bookkeeping of their own, so they take the plan rather than recomputing it.
     */
    void applySequenceShift(int[][] deltas, List<RowMove> moves,
                            List<PageCopy> copies, int[][] mirror) {
        if (group.attentionPlan() == null) {
            return;
        }
        lastStepPos0 = -1;
        lastStepTokens = -1;
        for (int rank = 0; rank < group.rankCount(); rank++) {
            DeviceModelMemory memory = group.rankMemory(rank);
            for (PageCopy copy : copies) {
                memory.copyPage(copy);
            }
            if (deltas != null) {
                memory.installPagedShiftDeltas(deltas);
            }
            memory.installPagedMirror(mirror);
        }
    }

    /** Completes the rank shifts after the mixers re-rotated the moved keys. */
    void moveShiftedRows(List<RowMove> moves) {
        if (group.attentionPlan() == null) {
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            group.rankMemory(rank).moveShiftedRows(moves);
        }
    }

    /**
     * Refuses a paged main memory whose step geometry the rank caches cannot address, and builds the context
     * split's page shard. A tensor split needs every rank's page pools to hold the same page size, page counts and
     * sequence slots as the memory that prepares the steps; a context-parallel split gives every rank pools of its
     * own size (the shard's share), so only the page size and the slot count must agree, and the shard's
     * page-ownership bookkeeping is built from the main plan and the rank plans.
     *
     * @throws UnsupportedOperationException naming the rank and the differing geometry
     */
    void verifyPagedMemory(ModelMemory main) {
        if (main == null || main.pagedSequences() == null) {
            return;
        }
        MemoryPlan mainPlan = main.plan();
        if (group.attentionPlan().contextParallel()) {
            MemoryPlan rankPlan = group.rankMemoryPlan(0);
            if (rankPlan == null || group.rankMemory(0) == null
                    || group.rankMemory(0).pagedSequences() == null) {
                throw new UnsupportedOperationException(group.rankName(0) + " does not hold paged caches, but the"
                        + " model serves multi-sequence batches");
            }
            if (mainPlan.group(0).pageSize() != rankPlan.group(0).pageSize()) {
                throw new UnsupportedOperationException("split-mode context cannot serve paged batches: "
                        + group.rankName(0) + " holds page size " + rankPlan.group(0).pageSize() + " to the model"
                        + " device's " + mainPlan.group(0).pageSize());
            }
            if (mainPlan.config() != null && rankPlan.config() != null
                    && mainPlan.config().parallel() != rankPlan.config().parallel()) {
                throw new UnsupportedOperationException("split-mode context cannot serve paged batches: "
                        + group.rankName(0) + " holds " + rankPlan.config().parallel() + " sequence slot(s) to "
                        + mainPlan.config().parallel());
            }
            pageShard = CpPageShard.of(mainPlan, rankPlan, group.rankCount());
            return;
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            DeviceModelMemory memory = group.rankMemory(rank);
            MemoryPlan rankPlan = memory == null ? null : memory.plan();
            if (rankPlan == null || memory.pagedSequences() == null) {
                throw new UnsupportedOperationException(group.rankName(rank) + " does not hold paged caches, but the"
                        + " model serves multi-sequence batches");
            }
            String mismatch = pagedMismatch(mainPlan, rankPlan);
            if (mismatch != null) {
                throw new UnsupportedOperationException("split-mode tensor cannot serve paged batches: "
                        + group.rankName(rank) + " " + mismatch + "; keep one KV layout for every rank or use"
                        + " neuralfree.split.mode=layer");
            }
        }
    }

    /** The difference between two paged plans' step geometry, or null when they address the same steps. */
    private static String pagedMismatch(MemoryPlan main, MemoryPlan rank) {
        if (main.groups().size() != rank.groups().size()) {
            return "holds " + rank.groups().size() + " KV group(s) to the model device's " + main.groups().size();
        }
        for (int g = 0; g < main.groups().size(); g++) {
            MemoryPlan.KvGroup a = main.group(g);
            MemoryPlan.KvGroup b = rank.group(g);
            if (a.pageSize() != b.pageSize()) {
                return "holds page size " + b.pageSize() + " to " + a.pageSize() + " in KV group " + g;
            }
            if (a.pages() != b.pages()) {
                return "holds " + b.pages() + " page(s) to " + a.pages() + " in KV group " + g;
            }
            if (a.rows() != b.rows() || a.ring() != b.ring() || !sameWindow(a.window(), b.window())) {
                return "holds KV group " + g + " (" + b.rows() + " rows, ring " + b.ring() + ", window "
                        + b.window() + ") to the model device's (" + a.rows() + " rows, ring " + a.ring()
                        + ", window " + a.window() + ")";
            }
        }
        if (main.config() != null && rank.config() != null
                && main.config().parallel() != rank.config().parallel()) {
            return "holds " + rank.config().parallel() + " sequence slot(s) to " + main.config().parallel();
        }
        return null;
    }

    /** True when two KV groups keep the same sliding window (the planner's group identity). */
    private static boolean sameWindow(com.eonmux.neuralfree.gguf.memory.KvWindow a,
                                      com.eonmux.neuralfree.gguf.memory.KvWindow b) {
        return a.type() == b.type() && a.nSwa() == b.nSwa() && a.causal() == b.causal();
    }

    String contextShiftRefusal() {
        if (group.attentionPlan() == null) {
            return null;
        }
        if (group.attentionPlan().contextParallel()) {
            return "a context split keeps one context block per rank, so a shift that moves positions across a block"
                    + " boundary cannot be carried by the rank caches; disable the context shift";
        }
        for (int rank = 0; rank < group.rankCount(); rank++) {
            DeviceModelMemory memory = group.rankMemory(rank);
            if (memory == null) {
                continue;
            }
            MemoryPlan plan = memory.plan();
            if (!plan.groups().isEmpty() && plan.group(0).paged()) {
                return "a tensor-parallel split keeps a paged KV cache per rank; shift its sequences through"
                        + " shiftSequence";
            }
        }
        return null;
    }

    /**
     * Runs the contiguous context shift (llama.cpp K-shift) on every rank cache: drops positions
     * {@code [nKeep, nKeep + nDiscard)} and moves the later rows down, so each rank's {@code shift_delta} addresses
     * the cells its mixer re-rotates. The mixers re-rotate afterwards; on failure every rank cache is reset, because
     * the moved rows can no longer be trusted.
     *
     * @throws UnsupportedOperationException when a rank cache cannot shift
     * @throws IllegalArgumentException      for a range beyond the stored positions
     */
    void shiftPositions(int nKeep, int nDiscard) {
        String refusal = contextShiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        if (group.attentionPlan() == null) {
            return;
        }
        lastStepPos0 = -1;
        lastStepTokens = -1;
        try {
            for (int rank = 0; rank < group.rankCount(); rank++) {
                group.rankMemory(rank).shiftPositions(nKeep, nDiscard);
            }
        } catch (RuntimeException e) {
            resetAfterShift(e);
            throw e;
        }
    }

    private void resetAfterShift(RuntimeException failure) {
        for (int rank = 0; rank < group.rankCount(); rank++) {
            try {
                group.rankMemory(rank).reset();
            } catch (RuntimeException resetFailure) {
                failure.addSuppressed(resetFailure);
            }
        }
    }

    /**
     * Writes the session of the main memory and of every rank's shard caches to {@code out}: the container holds the
     * main sequence first, then rank 0's shard memory and every peer's, each in the {@link SessionFile} format. A
     * load restores every rank, so the next decode continues the saved sequence with the same rows on every device.
     * A context-parallel paged split refuses by name: the rank page images follow the main page tables at run time
     * and have no session form of their own.
     */
    void saveSession(ModelMemory main, Path out) throws IOException {
        refuseSessionOverPageShard();
        List<ModelMemory> memories = sessionMemories(main);
        try (DataOutputStream stream = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(out)))) {
            stream.writeInt(SESSION_MAGIC);
            stream.writeInt(SESSION_VERSION);
            stream.writeInt(memories.size());
            for (ModelMemory memory : memories) {
                SessionFile.save(memory, stream);
            }
            stream.flush();
        }
    }

    /** Reads a tensor-parallel session written by {@link #saveSession}, resetting the step bookkeeping first. */
    void loadSession(ModelMemory main, Path in) throws IOException {
        refuseSessionOverPageShard();
        lastStepPos0 = -1;
        lastStepTokens = -1;
        List<ModelMemory> memories = sessionMemories(main);
        try (DataInputStream stream = new DataInputStream(new BufferedInputStream(Files.newInputStream(in)))) {
            if (stream.readInt() != SESSION_MAGIC) {
                throw new IOException(in + " is not a NeuralFree tensor-parallel session file");
            }
            int version = stream.readInt();
            if (version != SESSION_VERSION) {
                throw new IOException(in + " holds tensor-parallel session format " + version + ", this build reads "
                        + SESSION_VERSION);
            }
            int count = stream.readInt();
            if (count != memories.size()) {
                throw new IOException(in + " holds " + count + " session(s), this model needs " + memories.size());
            }
            for (int i = 0; i < memories.size(); i++) {
                SessionFile.load(memories.get(i), stream, in + " session " + i);
            }
        }
    }

    /** Why a session save or load would refuse under a context-parallel paged split. */
    private void refuseSessionOverPageShard() {
        if (pageShard != null) {
            throw new UnsupportedOperationException("a context-parallel paged split saves no session: the rank"
                    + " page images follow the main memory's page tables at run time, so only the main"
                    + " memory's cache would be saved and the split would start from rows the ranks cannot"
                    + " address; save the session without the context split");
        }
    }

    private List<ModelMemory> sessionMemories(ModelMemory main) {
        List<ModelMemory> memories = new ArrayList<>();
        memories.add(Objects.requireNonNull(main, "main"));
        memories.addAll(group.rankMemories());
        return memories;
    }

    /** True while a context-parallel paged split runs (its page-ownership shard is built). */
    boolean hasPageShard() {
        return pageShard != null;
    }

    /** Drops the paged shard and the step bookkeeping: the group is closing. */
    void discard() {
        pageShard = null;
        lastStepPos0 = -1;
        lastStepTokens = -1;
    }
}
