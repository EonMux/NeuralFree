package com.eonmux.neuralfree.gguf.gpu.weights;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Segmented LRU over the expert slots of one cache group, the policy of llama.cpp's {@code moe_cache_slru}
 * (src/llama-moe-cache.cpp:17-155): probationary and protected segments over one shared slot pool, hits promoted to
 * the protected half, misses evicting from the probation tail.
 *
 * <p>A slot holds the expert {@code key / nExpert} of layer {@code key / nExpert} ({@code -1} when empty);
 * {@code slotOf[layer][expert]} mirrors the residency ({@code -1} when the expert holds no slot). {@link #plan}
 * gives every demanded expert of one step a slot: hits move to the protected MRU end, misses take victims from the
 * probation tail first, and an expert the step demands is never evicted while the step is being filled. One expert
 * per demand is counted once. Unlike the reference, the caller keeps working when {@link #plan} reports
 * {@code false} (more distinct experts demanded than there are slots): the unrouted demands are simply misses whose
 * outputs the host computes.</p>
 *
 * <p>A slot can be lent out ({@link #lend}, Avalon's prefetch zones): its expert is evicted and the slot leaves both
 * segments, so no demand or admission takes it until {@link #restore} returns it empty at the probation LRU end.</p>
 *
 * <p>Sizes are fixed at construction. Not thread-safe; the caller runs one {@link #plan} per step.</p>
 */
public final class MoeCacheSlru {

    /** A miss assigned a slot: the expert of layer {@code layer} takes {@code slot}. */
    public static final class Fill {
        public final int layer;
        public final int expert;
        public final int slot;

        Fill(int layer, int expert, int slot) {
            this.layer = layer;
            this.expert = expert;
            this.slot = slot;
        }

        @Override
        public String toString() {
            return "fill{layer " + layer + ", expert " + expert + " -> slot " + slot + "}";
        }
    }

    private final int nLayer;
    private final int nExpert;
    private final int nSlots;
    private final int[][] slotOf;
    private final int[] keyOf;
    /** Slot to the left / right neighbour in the segment list ({@code -1} at the ends). */
    private final int[] prev;
    private final int[] next;
    private final int[] head = {-1, -1};
    private final int[] tail = {-1, -1};
    private final boolean[] inProtected;
    private int nProtected;
    /** Generation marks of the experts the current {@link #plan} demands. */
    private final int[] seen;
    private int seenGen;
    private final int[] uniq;
    private int nUniq;
    /** Slots whose contents are still being written; never victims and never promoted. */
    private final boolean[] pending;
    /** Slots lent out of the pool: in neither segment, never victims. */
    private final boolean[] lent;

    /**
     * A policy over {@code nSlots} slots shared by {@code nLayer} layers of {@code nExpert} experts each; every
     * slot starts empty and probationary.
     *
     * @throws IllegalArgumentException for a non-positive size
     */
    public MoeCacheSlru(int nLayer, int nExpert, int nSlots) {
        if (nLayer < 1 || nExpert < 1 || nSlots < 1) {
            throw new IllegalArgumentException("SLRU sizes must be positive, got layers " + nLayer + ", experts "
                    + nExpert + ", slots " + nSlots);
        }
        this.nLayer = nLayer;
        this.nExpert = nExpert;
        this.nSlots = nSlots;
        this.slotOf = new int[nLayer][];
        for (int il = 0; il < nLayer; il++) {
            int[] m = new int[nExpert];
            Arrays.fill(m, -1);
            slotOf[il] = m;
        }
        this.keyOf = new int[nSlots];
        Arrays.fill(keyOf, -1);
        this.prev = new int[nSlots];
        this.next = new int[nSlots];
        for (int s = 0; s < nSlots; s++) {
            prev[s] = s - 1;
            next[s] = s + 1 < nSlots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = nSlots - 1;
        this.inProtected = new boolean[nSlots];
        this.seen = new int[nExpert];
        this.uniq = new int[nExpert];
        this.pending = new boolean[nSlots];
        this.lent = new boolean[nSlots];
    }

    public int slots() {
        return nSlots;
    }

    /** Slots in the pool that hold no expert (free for a fill without an eviction); lent slots do not count. */
    public int emptySlots() {
        int n = 0;
        for (int s = 0; s < nSlots; s++) {
            if (keyOf[s] < 0 && !lent[s]) {
                n++;
            }
        }
        return n;
    }

    /** Slot of layer {@code il}'s expert {@code e}, or {@code -1} when it holds none. */
    public int slotOf(int il, int e) {
        checkLayer(il);
        checkExpert(e);
        return slotOf[il][e];
    }

    /** True when the slot's fill is still in flight. */
    public boolean pending(int slot) {
        checkSlot(slot);
        return pending[slot];
    }

    /**
     * Gives a slot to every distinct expert of layer {@code il} the {@code ids} demand, evicting from the
     * probation segment first and never an expert of this demand or one still being written; hits are promoted
     * before the victims are chosen. {@code fills} receives one {@link Fill} per miss, in ascending expert order
     * (so consecutive experts usually take consecutive slots and their uploads can be merged).
     *
     * @return the number of hits, or {@code -1} when the demand selects more distinct experts than there are
     *         slots (nothing is assigned)
     * @throws IllegalArgumentException for an id outside {@code [0, nExpert)}
     */
    public int plan(int il, int[] ids, int nIds, List<Fill> fills) {
        checkLayer(il);
        checkIds(ids, nIds);
        fills.clear();
        if (++seenGen == 0) {
            Arrays.fill(seen, 0);
            seenGen = 1;
        }
        nUniq = 0;
        for (int i = 0; i < nIds; i++) {
            int e = ids[i];
            checkExpert(e);
            if (seen[e] != seenGen) {
                seen[e] = seenGen;
                uniq[nUniq++] = e;
            }
        }
        if (nUniq > nSlots) {
            return -1;
        }
        int[] slots = slotOf[il];
        int hits = 0;
        for (int i = 0; i < nUniq; i++) {
            int s = slots[uniq[i]];
            if (s >= 0) {
                touch(s);
                hits++;
            }
        }
        for (int i = 0; i < nUniq; i++) {
            int e = uniq[i];
            if (slots[e] >= 0) {
                continue;
            }
            int s = victim(il);
            if (s < 0) {
                // every slot is held by a pending fill or by this demand; the rest stay misses
                break;
            }
            if (keyOf[s] >= 0) {
                slotOf[keyOf[s] / nExpert][keyOf[s] % nExpert] = -1;
            }
            keyOf[s] = il * nExpert + e;
            slots[e] = s;
            move(s, false);
            fills.add(new Fill(il, e, s));
        }
        return hits;
    }

    /**
     * Static admission without a demand, for a ranking preload or a prefill histogram flush: expert {@code e} of
     * layer {@code il} takes a victim slot when it holds none and one is free to take, entering at the probation
     * MRU end; a hit or a pending slot is left alone.
     *
     * @return the assigned slot, or {@code -1} when the expert holds a slot, is pending, or no victim exists
     */
    public int admit(int il, int e) {
        checkLayer(il);
        checkExpert(e);
        int[] slots = slotOf[il];
        if (slots[e] >= 0) {
            return -1;
        }
        int s = victim(il);
        if (s < 0) {
            return -1;
        }
        if (keyOf[s] >= 0) {
            slotOf[keyOf[s] / nExpert][keyOf[s] % nExpert] = -1;
        }
        keyOf[s] = il * nExpert + e;
        slots[e] = s;
        move(s, false);
        return s;
    }

    /** Marks a slot's fill as started; a pending slot is never a victim until {@link #fillDone}. */
    public void fillStarted(int slot) {
        checkSlot(slot);
        pending[slot] = true;
    }

    /** Marks a slot's fill complete. */
    public void fillDone(int slot) {
        checkSlot(slot);
        pending[slot] = false;
    }

    /**
     * Lends a slot out of the pool: its expert, if any, is evicted and the slot leaves both segments until
     * {@link #restore}.
     *
     * @return the evicted key ({@code layer * nExpert + expert}), or {@code -1} when the slot was empty
     * @throws IllegalStateException when the slot is already lent or its fill is still in flight
     */
    public int lend(int slot) {
        checkSlot(slot);
        if (lent[slot]) {
            throw new IllegalStateException("slot " + slot + " is already lent");
        }
        if (pending[slot]) {
            throw new IllegalStateException("slot " + slot + " is still being filled; it cannot be lent");
        }
        int key = keyOf[slot];
        if (key >= 0) {
            slotOf[key / nExpert][key % nExpert] = -1;
            keyOf[slot] = -1;
        }
        unlink(slot);
        lent[slot] = true;
        return key;
    }

    /**
     * Returns a lent slot to the pool, empty, at the probation LRU end (the next victim).
     *
     * @throws IllegalStateException when the slot is not lent
     */
    public void restore(int slot) {
        checkSlot(slot);
        if (!lent[slot]) {
            throw new IllegalStateException("slot " + slot + " is not lent");
        }
        lent[slot] = false;
        inProtected[slot] = false;
        prev[slot] = -1;
        next[slot] = head[0];
        if (head[0] >= 0) {
            prev[head[0]] = slot;
        } else {
            tail[0] = slot;
        }
        head[0] = slot;
    }

    /** True when the slot is lent out of the pool. */
    public boolean lent(int slot) {
        checkSlot(slot);
        return lent[slot];
    }

    /** Removes slot {@code s} from its segment list. */
    private void unlink(int s) {
        int seg = inProtected[s] ? 1 : 0;
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[seg] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[seg] = prev[s];
        }
        prev[s] = -1;
        next[s] = -1;
        if (inProtected[s]) {
            inProtected[s] = false;
            nProtected--;
        }
    }

    /**
     * The slot to evict: the least recently used slot of the probation segment, then of the protected segment,
     * whose expert is not demanded now, not of layer {@code il}'s current fill set, empty, or still being written;
     * {@code -1} when every slot is held by pending or demanded experts.
     */
    private int victim(int il) {
        for (int segment = 0; segment < 2; segment++) {
            for (int s = head[segment]; s >= 0; s = next[s]) {
                int key = keyOf[s];
                if (pending[s]) {
                    continue;
                }
                if (key < 0 || key / nExpert != il || seen[key % nExpert] != seenGen) {
                    return s;
                }
            }
        }
        return -1;
    }

    /** Moves slot {@code s} to the MRU end of its segment, or of the protected segment when {@code protect}. */
    private void move(int s, boolean protect) {
        int from = inProtected[s] ? 1 : 0;
        int to = protect ? 1 : 0;
        if (from == to && s == tail[to]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }
        prev[s] = tail[to];
        next[s] = -1;
        if (tail[to] >= 0) {
            next[tail[to]] = s;
        } else {
            head[to] = s;
        }
        tail[to] = s;
        inProtected[s] = protect;
        nProtected += (protect ? 1 : 0) - from;
    }

    /** Promotes a hit to the protected MRU end, demoting the protected head when the half outgrows half the pool. */
    private void touch(int s) {
        move(s, true);
        if (nProtected > nSlots / 2) {
            move(head[1], false);
        }
    }

    private void checkIds(int[] ids, int nIds) {
        if (ids == null) {
            throw new IllegalArgumentException("ids must not be null");
        }
        if (nIds < 0 || nIds > ids.length) {
            throw new IllegalArgumentException("demand length " + nIds + " outside [0, " + ids.length + "]");
        }
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= nLayer) {
            throw new IllegalArgumentException("layer " + il + " outside [0, " + nLayer + ")");
        }
    }

    private void checkExpert(int e) {
        if (e < 0 || e >= nExpert) {
            throw new IllegalArgumentException("expert " + e + " outside [0, " + nExpert + ")");
        }
    }

    private void checkSlot(int s) {
        if (s < 0 || s >= nSlots) {
            throw new IllegalArgumentException("slot " + s + " outside [0, " + nSlots + ")");
        }
    }

    @Override
    public String toString() {
        StringBuilder b = new StringBuilder("MoeCacheSlru{slots=").append(nSlots)
                .append(", layers=").append(nLayer).append(", experts=").append(nExpert)
                .append(", protected=").append(nProtected).append(", probation=[");
        for (int s = head[0]; s >= 0; s = next[s]) {
            b.append(keyOf[s] < 0 ? '-' : keyOf[s]);
            if (next[s] >= 0) {
                b.append(',');
            }
        }
        return b.append(']').append('}').toString();
    }

    /** Fills list helper: a fresh mutable list of the right element type. */
    public static List<Fill> newFillList() {
        return new ArrayList<>();
    }
}
