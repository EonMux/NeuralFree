package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.SessionFile;
import com.eonmux.neuralfree.gguf.memory.StateTransfer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.util.List;
import java.util.Objects;

/**
 * How cached blocks of a paged memory leave for the host and come back ({@link BlockImage}): the memory's page
 * bookkeeping, its {@link StateTransfer} and the identity a slot file records — the page size, the KV group count and
 * every owning cache layer's group, row widths and dtypes. A block's rows are all its state only when the memory
 * keeps nothing per sequence beside its pages, so a memory with recurrent state, DSV4 compressed caches, sparse side
 * caches, cross-attention caches or target-model caches holds no layout; {@link #refusal(ModelMemory)} names why.
 */
final class BlockLayout {

    private final PagedSequences paged;
    private final List<StateTransfer.Cache> caches;

    BlockLayout(PagedSequences paged, List<StateTransfer.Cache> caches) {
        this.paged = paged;
        this.caches = caches;
    }

    /**
     * The layout of {@code memory}'s blocks.
     *
     * @throws UnsupportedOperationException carrying {@link #refusal(ModelMemory)} when its blocks cannot leave it
     */
    static BlockLayout of(ModelMemory memory) {
        String why = refusal(memory);
        if (why != null) {
            throw new UnsupportedOperationException(why);
        }
        return new BlockLayout(memory.pagedSequences(), memory.stateTransfer().caches());
    }

    /** Why the cached blocks of {@code memory} cannot move to the host and back, or null when they can. */
    static String refusal(ModelMemory memory) {
        Objects.requireNonNull(memory, "memory");
        if (memory.pagedSequences() == null) {
            return memory.getClass().getSimpleName() + " holds no paged sequences";
        }
        MemoryPlan plan = memory.plan();
        if (plan.hasRecurrentState()) {
            return "the recurrent state of a sequence is not part of its cached blocks";
        }
        if (plan.hasCompressed()) {
            return "the DSV4 compressed caches of a sequence are not part of its cached blocks";
        }
        for (LayerMemoryPlan lp : plan.layers()) {
            String why = SessionFile.layerRefusal(lp);
            if (why != null) {
                return why;
            }
        }
        try {
            memory.stateTransfer();
        } catch (UnsupportedOperationException e) {
            return e.getMessage();
        }
        return null;
    }

    /** The memory's page bookkeeping. */
    PagedSequences paged() {
        return paged;
    }

    /** Rows per page. */
    int pageSize() {
        return paged.pageSize();
    }

    /** The owning cache layers, in the memory's transfer order. */
    List<StateTransfer.Cache> caches() {
        return caches;
    }

    /**
     * The identity a slot file records and checks: page size, KV group count, then per cache layer its layer,
     * group, K/V dtype codes and widths, whether V aliases K and its KV head count (which sizes the per-token-head
     * scale planes).
     */
    int[] identity() {
        int[] out = new int[2 + 8 * caches.size()];
        out[0] = paged.pageSize();
        out[1] = paged.groups();
        int at = 2;
        for (StateTransfer.Cache c : caches) {
            KvLayout l = c.layout();
            out[at++] = c.layer();
            out[at++] = c.group();
            out[at++] = l.dtypeK().code();
            out[at++] = l.dtypeV().code();
            out[at++] = l.widthK();
            out[at++] = l.widthV();
            out[at++] = l.vAliasesK() ? 1 : 0;
            out[at++] = l.nHeadKv();
        }
        return out;
    }

    /** What entry {@code i} of {@link #identity()} records, for a mismatch message. */
    String identityField(int i) {
        if (i == 0) {
            return "the page size";
        }
        if (i == 1) {
            return "the KV group count";
        }
        String[] fields = {"layer", "group", "K dtype", "V dtype", "K width", "V width", "latent V", "KV heads"};
        int c = (i - 2) / fields.length;
        return "cache layer " + c + " " + fields[(i - 2) % fields.length];
    }
}
