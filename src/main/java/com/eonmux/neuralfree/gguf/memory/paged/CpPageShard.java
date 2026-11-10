package com.eonmux.neuralfree.gguf.memory.paged;

import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;

import java.util.Arrays;
import java.util.Objects;

/**
 * The page-granular context-parallel shard of a paged multi-sequence memory (vLLM's decode context parallelism
 * over block tables, with the interleave at one page): every rank keeps its own {@link PagePool} of
 * {@code ceil(mainPages / ranks) + parallel} pages per KV group, and a main page belongs to rank
 * {@code (firstStoredPosition / pageSize) % ranks}. The interleave equals vLLM's
 * {@code dcp_kv_cache_interleave_size} taken at the page size, so the partition is the page-table analogue of the
 * contiguous context block, and it stays stable because a page's first stored position never changes under the
 * supported operations: appends fill rows bottom-up, removals clear from the back (a whole-tail-page removal
 * frees the page), copy-on-write copies preserve positions, and the position edits are refused.
 *
 * <p>Each step the shard reconciles against the main memory's host {@code kv_pos} mirror: a page whose rows are
 * all empty is freed on its rank; a live page registers a rank page on its owner; a mapping whose recorded first
 * position no longer matches the mirror is the main pool recycling a page id, and remaps. Every rank then receives
 * a translated {@link PagedStep} over its own {@code PagedSequences} — rank-local page tables (the owned pages in
 * order), rank-local slot counts ({@code seqSlots} is the rank's key length per sequence), cells and
 * {@code kv_pos} rows — so the rank's paged kernels run unchanged. A row landing on another rank's page carries
 * cell {@code -1}: the rank stores nothing there, its query still attends over this rank's keys, and the per-rank
 * partials merge through the log-sum-exp exactly as in the contiguous split, so the merged output stays the
 * single-device attention.</p>
 *
 * <p>Refused by name, because they move a page's rows across positions or break the page's consecutive stored
 * rows: media / embedding segments (several rows at one base position), a sequence's next position jumping over
 * its page's next slots, KV-chunk reuse and its shifted-row page copies, page imports at non-identity positions,
 * and the per-sequence shifts. Not thread-safe; the forward thread drives it.</p>
 */
public final class CpPageShard {

    private final int ranks;
    private final int pageSize;
    /** Main-plan KV group -> the rank-plan group holding the same layers' cache, or -1 when the rank holds none. */
    private final int[] rankGroupOfMain;
    /** Pages of each main group's pool (the image arrays' bound). */
    private final int[] mainGroupPages;
    /** Pages of each rank group's pool. */
    private final int[] rankGroupPages;
    /** Per main group and rank: the rank's page pool (sized by the rank group's pages). */
    private final PagePool[][] pools;
    /** Per main group, rank and main page: the rank page holding its rows, or -1 when unmapped. */
    private final int[][][] imageOf;
    /** Per main group and main page: the page's first stored position, or -1 while empty. */
    private final int[][] firstPos;
    /** Sequence slots the plans share (the per-sequence margin of every rank pool). */
    private final int sequences;

    private CpPageShard(MemoryPlan mainPlan, MemoryPlan rankPlan, int ranks) {
        this.ranks = ranks;
        this.pageSize = mainPlan.group(0).pageSize();
        this.sequences = mainPlan.config() != null ? mainPlan.config().parallel() : 0;
        int mainGroups = mainPlan.groups().size();
        int rankGroups = rankPlan.groups().size();
        this.rankGroupOfMain = new int[mainGroups];
        Arrays.fill(rankGroupOfMain, -1);
        this.mainGroupPages = new int[mainGroups];
        for (int g = 0; g < mainGroups; g++) {
            mainGroupPages[g] = mainPlan.group(g).pages();
        }
        this.rankGroupPages = new int[rankGroups];
        for (int rg = 0; rg < rankGroups; rg++) {
            rankGroupPages[rg] = rankPlan.group(rg).pages();
        }
        // the correspondence follows the layers: a rank group's cache holds a main group's layers' rows
        for (int rg = 0; rg < rankGroups; rg++) {
            for (int il = 0; il < rankPlan.layers().size(); il++) {
                if (rankPlan.layer(il).kvGroup() == rg) {
                    rankGroupOfMain[mainPlan.layer(il).kvGroup()] = rg;
                    break;
                }
            }
        }
        this.pools = new PagePool[mainGroups][ranks];
        this.imageOf = new int[mainGroups][ranks][];
        for (int g = 0; g < mainGroups; g++) {
            if (rankGroupOfMain[g] < 0) {
                continue;
            }
            for (int rank = 0; rank < ranks; rank++) {
                pools[g][rank] = new PagePool(pageSize, rankGroupPages[rankGroupOfMain[g]]);
                imageOf[g][rank] = new int[mainGroupPages[g]];
                Arrays.fill(imageOf[g][rank], -1);
            }
        }
        this.firstPos = new int[mainGroups][];
        for (int g = 0; g < mainGroups; g++) {
            firstPos[g] = new int[mainGroupPages[g]];
            Arrays.fill(firstPos[g], -1);
        }
    }

    /**
     * Builds the shard over the main plan and one rank plan (a paged context split gives every rank the same
     * plan; the runtime page ownership differentiates them).
     *
     * @throws IllegalArgumentException when the plans are not both paged or disagree on page size or slot count
     */
    public static CpPageShard of(MemoryPlan mainPlan, MemoryPlan rankPlan, int ranks) {
        Objects.requireNonNull(mainPlan, "mainPlan");
        Objects.requireNonNull(rankPlan, "rankPlan");
        if (ranks < 2) {
            throw new IllegalArgumentException("a context page shard needs at least two ranks, got " + ranks);
        }
        MemoryPlan.KvGroup mainFirst = mainPlan.group(0);
        MemoryPlan.KvGroup rankFirst = rankPlan.group(0);
        if (!mainFirst.paged() || !rankFirst.paged()) {
            throw new IllegalArgumentException("a context page shard needs paged plans; got " + mainFirst + " and "
                    + rankFirst);
        }
        if (mainFirst.pageSize() != rankFirst.pageSize()) {
            throw new IllegalArgumentException("the rank plan's page size " + rankFirst.pageSize()
                    + " must match the main plan's " + mainFirst.pageSize());
        }
        if (mainPlan.config() != null && rankPlan.config() != null
                && mainPlan.config().parallel() != rankPlan.config().parallel()) {
            throw new IllegalArgumentException("the rank plan holds " + rankPlan.config().parallel()
                    + " sequence slot(s) to the main plan's " + mainPlan.config().parallel());
        }
        return new CpPageShard(mainPlan, rankPlan, ranks);
    }

    /**
     * Reconciles the rank's page images against the main memory's authoritative host mirror and returns the rank's
     * translated step: the same segments with rank-local tables, cells, slot counts and {@code kv_pos} rows.
     *
     * @param mainStep   the main memory's committed step ({@link PagedSequences#prepare} + {@link PagedSequences#commit})
     * @param mainPaged  the main memory's paged sequences, whose host mirror the commit keeps current
     * @param rankMemory the rank's device memory the translated step belongs to
     * @param rank       this rank's index
     * @throws UnsupportedOperationException for a step the shard cannot carry (media segments sharing positions)
     *                                      or when a rank pool is exhausted
     */
    public PagedStep translateStep(PagedStep mainStep, PagedSequences mainPaged, DeviceModelMemory rankMemory,
                                   int rank) {
        Objects.requireNonNull(mainStep, "mainStep");
        Objects.requireNonNull(mainPaged, "mainPaged");
        Objects.requireNonNull(rankMemory, "rankMemory");
        checkRank(rank);
        int[][] mirror = mainPaged.kvPos();
        if (mirror.length < mainStep.groups()) {
            throw new IllegalArgumentException("the main mirror covers " + mirror.length + " group(s), the step "
                    + mainStep.groups());
        }
        refuseSharedPositionSegments(mainStep);
        int n = mainStep.nTokens();
        PagedStep.Builder b = PagedStep.builder(rankMemory.pagedSequences(), n);
        int nSeg = mainStep.nSeg();
        int[] segSeq = new int[nSeg];
        int[] segStart = new int[nSeg];
        int[] segLen = new int[nSeg];
        int[] segPos0 = new int[nSeg];
        for (int seg = 0; seg < nSeg; seg++) {
            segSeq[seg] = mainStep.segSeq(seg);
            segStart[seg] = mainStep.segStart(seg);
            segLen[seg] = mainStep.segLen(seg);
            segPos0[seg] = mainStep.segPos0(seg);
        }
        b.segments(segSeq, segStart, segLen, segPos0);
        for (int g = 0; g < mainStep.groups(); g++) {
            int rg = g < rankGroupOfMain.length ? rankGroupOfMain[g] : -1;
            if (rg < 0) {
                continue;
            }
            translateGroup(g, rg, mainStep, mirror[g], rank, b);
        }
        return b.build();
    }

    /** Refuses a step whose rows share positions (media / embedding chunks): the shard stores one position per row. */
    private static void refuseSharedPositionSegments(PagedStep mainStep) {
        for (int seg = 0; seg < mainStep.nSeg(); seg++) {
            if (mainStep.segLen(seg) > 1
                    && mainStep.segPosMax(seg) != mainStep.segPos0(seg) + mainStep.segLen(seg) - 1) {
                throw new UnsupportedOperationException("a context-parallel paged batch cannot carry segment " + seg
                        + " (sequence " + mainStep.segSeq(seg) + "), whose " + mainStep.segLen(seg)
                        + " rows share the base position " + mainStep.segPos0(seg)
                        + "; media and embedding rows need one stored position per row");
            }
        }
    }

    /** Reconciles one main group against the mirror and writes the rank's translated rows into {@code b}. */
    private void translateGroup(int g, int rg, PagedStep mainStep, int[] mirror, int rank, PagedStep.Builder b) {
        int pages = mainGroupPages[g];
        int[] mainCells = mainStep.cellIdx(g);
        int[] image = imageOf[g][rank];
        PagePool pool = pools[g][rank];
        int[] recorded = firstPos[g];

        reconcile(g, mirror, image, pool, recorded, rank);

        int n = mainStep.nTokens();
        for (int row = 0; row < n; row++) {
            b.cell(rg, row, rankCell(mainCells[row], image, recorded, pages, rank));
        }
        int[] kvRows = new int[pages * pageSize];
        int[] kvValues = new int[pages * pageSize];
        int kvCount = 0;
        int maxSeqs = mainStep.maxSequences();
        for (int seq = 0; seq < maxSeqs; seq++) {
            int owned = 0;
            int rankSlots = 0;
            int tableLen = mainStep.pageTableLength(g, seq);
            for (int k = 0; k < tableLen; k++) {
                int p = mainStep.pageTableEntry(g, seq, k);
                if (p < 0 || p >= pages || recorded[p] < 0 || image[p] < 0 || ownerOf(recorded[p]) != rank) {
                    continue;
                }
                b.pageTable(rg, seq, owned, image[p]);
                owned++;
                int expected = recorded[p];
                for (int r = 0; r < pageSize; r++) {
                    int pos = mirror[p * pageSize + r];
                    if (pos >= 0) {
                        if (pos != expected) {
                            throw new UnsupportedOperationException("a context-parallel paged batch cannot carry"
                                    + " sequence " + seq + "'s page " + p + " (KV group " + rg + "): its rows"
                                    + " hold the positions " + mirror[p * pageSize] + ".. but row " + r + " holds "
                                    + pos + " instead of the consecutive " + expected + "; the shard assigns a"
                                    + " page to the rank of its first stored position, so a page's stored rows"
                                    + " must be consecutive: media and embedding rows share one base position,"
                                    + " and a position jump leaves a hole, either of which the shard cannot"
                                    + " partition");
                        }
                        kvRows[kvCount] = image[p] * pageSize + r;
                        kvValues[kvCount] = pos;
                        kvCount++;
                        rankSlots++;
                        expected++;
                    }
                }
            }
            b.seqSlots(rg, seq, rankSlots);
        }
        if (kvCount > 0) {
            b.kvPos(rg, Arrays.copyOf(kvRows, kvCount), Arrays.copyOf(kvValues, kvCount));
        }
    }

    /** Refreshes the recorded first positions, frees emptied pages and registers the live ones of this rank. */
    private void reconcile(int g, int[] mirror, int[] image, PagePool pool, int[] recorded, int rank) {
        int pages = mainGroupPages[g];
        for (int p = 0; p < pages; p++) {
            int fp = mirror[p * pageSize];
            if (fp < 0) {
                if (image[p] >= 0) {
                    pool.unref(image[p]);
                    image[p] = -1;
                }
                recorded[p] = -1;
                continue;
            }
            if (image[p] >= 0 && recorded[p] >= 0 && recorded[p] != fp) {
                // the main pool recycled this page id for a page with other positions
                pool.unref(image[p]);
                image[p] = -1;
            }
            recorded[p] = fp;
            if (ownerOf(fp) == rank && image[p] < 0) {
                image[p] = takePage(g, rank, pool, p);
            }
        }
    }

    /** Allocates the rank page of one owned main page. */
    private int takePage(int g, int rank, PagePool pool, int p) {
        int fresh = pool.alloc();
        if (fresh < 0) {
            throw new UnsupportedOperationException("rank " + rank + "'s page pool of KV group "
                    + rankGroupOfMain[g] + " (" + pool.pages() + " pages) is exhausted while sharding the paged"
                    + " memory; the pool holds the main pages split over the ranks plus one page of margin per"
                    + " sequence slot");
        }
        return fresh;
    }

    /** The rank cell of one main cell, or -1 when the row lands on another rank's page. */
    private int rankCell(int mainCell, int[] image, int[] recorded, int pages, int rank) {
        if (mainCell < 0) {
            return -1;
        }
        int p = mainCell / pageSize;
        if (p >= pages || recorded[p] < 0 || image[p] < 0 || ownerOf(recorded[p]) != rank) {
            return -1;
        }
        return image[p] * pageSize + mainCell % pageSize;
    }

    /**
     * Translates one device page copy for one rank, or returns null when the rank holds no cache for the copy's
     * group. The source page must already be registered (a copy always follows a committed step that mapped it);
     * a copy-on-write copy preserves positions, so the destination registers with the source's first stored
     * position — the main memory's authoritative mirror holds it already ({@link PagedSequences} mirrors a copy
     * when the copy is queued). A shifted-row copy (KV-chunk reuse) is refused by name.
     */
    public PageCopy translateCopy(PageCopy copy, PagedSequences mainPaged, int rank) {
        Objects.requireNonNull(copy, "copy");
        Objects.requireNonNull(mainPaged, "mainPaged");
        checkRank(rank);
        int g = copy.group();
        int rg = g < rankGroupOfMain.length ? rankGroupOfMain[g] : -1;
        if (rg < 0) {
            return null;
        }
        if (copy.srcRow0() != copy.dstRow0()) {
            throw new UnsupportedOperationException("a context-parallel paged batch cannot carry the shifted-row"
                    + " page copy " + copy + ": it lands rows at the destination's own slots, which the shard's"
                    + " position-based page ownership cannot follow (KV-chunk reuse)");
        }
        int[] mirror = mainPaged.kvPos()[g];
        int[] image = imageOf[g][rank];
        int src = copy.srcPage();
        int dst = copy.dstPage();
        int srcImage = image[src];
        if (srcImage < 0) {
            if (firstPos[g][src] >= 0 && ownerOf(firstPos[g][src]) == rank) {
                throw new IllegalStateException("page copy " + copy + " reads a page the shard has not registered"
                        + " (rank " + rank + ", KV group " + rg + "); copies follow a committed step's tables");
            }
            /* The rank holds no rows of the source page: the copy-on-write lands on its owner only. */
            return null;
        }
        int fp = dst * pageSize < mirror.length ? mirror[dst * pageSize] : -1;
        if (fp < 0) {
            fp = firstPos[g][src];
        }
        int dstImage = image[dst];
        if (dstImage >= 0 && firstPos[g][dst] != fp) {
            pools[g][rank].unref(dstImage);
            dstImage = -1;
        }
        if (dstImage < 0) {
            dstImage = pools[g][rank].alloc();
            if (dstImage < 0) {
                throw new UnsupportedOperationException("rank " + rank + "'s page pool of KV group " + rg + " ("
                        + pools[g][rank].pages() + " pages) is exhausted while copying main page " + src
                        + " to " + dst);
            }
        }
        firstPos[g][dst] = fp;
        image[dst] = dstImage;
        return PageCopy.ofRange(rg, srcImage, dstImage, copy.srcRow0(), copy.rows());
    }

    /** Empties every rank's shard: all mappings and pools reset (a full-memory reset). */
    public void reset() {
        for (int g = 0; g < mainGroupPages.length; g++) {
            if (rankGroupOfMain[g] < 0) {
                continue;
            }
            for (int rank = 0; rank < ranks; rank++) {
                pools[g][rank] = new PagePool(pageSize, rankGroupPages[rankGroupOfMain[g]]);
                Arrays.fill(imageOf[g][rank], -1);
            }
            Arrays.fill(firstPos[g], -1);
        }
    }

    /** The rank page holding main page {@code mainPage} of main group {@code g} on {@code rank}, or -1. */
    public int imageOf(int g, int rank, int mainPage) {
        if (g >= imageOf.length || rank >= ranks || mainPage >= mainGroupPages[g]) {
            return -1;
        }
        return imageOf[g][rank][mainPage];
    }

    /** The first stored position recorded for main page {@code p} of group {@code g}, or -1. */
    public int firstPosOf(int g, int p) {
        return g < firstPos.length && p >= 0 && p < firstPos[g].length ? firstPos[g][p] : -1;
    }

    /** The rank group's pool size behind main group {@code g}, or -1 when the rank holds no cache for it. */
    public int rankPagesOfMainGroup(int g) {
        int rg = g < rankGroupOfMain.length ? rankGroupOfMain[g] : -1;
        return rg >= 0 ? rankGroupPages[rg] : -1;
    }

    /** The owner rank of the position range starting at {@code position} (the shard's assignment rule). */
    public int ownerOf(int position) {
        return (position / pageSize) % ranks;
    }

    /** The rank count of the split. */
    public int ranks() {
        return ranks;
    }

    private void checkRank(int rank) {
        if (rank < 0 || rank >= ranks) {
            throw new IllegalArgumentException("rank " + rank + " is outside the " + ranks + " rank(s)");
        }
    }

    @Override
    public String toString() {
        return "CpPageShard{" + ranks + " ranks, page size " + pageSize + ", " + sequences + " sequence slot(s)}";
    }
}
