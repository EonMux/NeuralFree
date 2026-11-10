package com.eonmux.neuralfree.gguf.memory.paged;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The compaction of a {@link PagedSequences} sequence after a context shift: the live rows of each KV group slide
 * down over the dead slots below them, in slot order, onto a running write cursor, so every live row ends at the
 * slot equal to the number of live rows below it. The dead slots may form any number of runs: an interior removal
 * leaves one, and a sliding-window group's leading page keeps the rows its trim cleared as further holes.
 *
 * <p>The cells move with the rows: the {@code kv_pos} mirror follows on the host, one absolute set report and one
 * clear report per group name every cell by its final value, and the backend carries the keys and values through
 * the queued {@link RowMove}s. A move is a maximal run that stays contiguous in both its source and its destination
 * cells and never overlaps itself; the moves are queued in slot order and must run in that order, since a later
 * move's destination can be an earlier move's source (a slide shorter than a run). The pages that receive rows
 * privatize first, so a shared page is never written; a vacated tail page leaves the list, and one another holder
 * shares keeps its rows for that holder. Not thread-safe; used by the thread that drives the owning memory.
 */
final class PagedCompaction {

    private final PagedSequences owner;

    PagedCompaction(PagedSequences owner) {
        this.owner = Objects.requireNonNull(owner, "owner");
    }

    /** Compacts the sequence's rows of KV group {@code g}. */
    void compactGroup(int g, int seq) {
        int pageSize = owner.pageSize();
        int cursor = owner.slots()[g][seq];
        int[] kvPos = owner.kvPos()[g];
        PageList list = owner.lists()[seq][g];
        int live = 0;
        int firstDst = -1;
        for (int s = 0; s < cursor; s++) {
            if (kvPos[owner.cellOf(list, s)] >= 0) {
                if (s != live && firstDst < 0) {
                    firstDst = live;
                }
                live++;
            }
        }
        if (live == cursor) {
            return;
        }
        int keptPages = (live + pageSize - 1) / pageSize;
        if (firstDst >= 0) {
            for (int k = firstDst / pageSize; k < keptPages; k++) {
                owner.privatize(g, seq, k);
            }
            list = owner.lists()[seq][g];
            moveRows(g, seq, list, cursor, keptPages);
        }
        owner.slots()[g][seq] = live;
        while (list.size() > keptPages) {
            owner.sharing().unref(false, g, list.page(list.size() - 1));
            list.removeBack(1);
        }
    }

    /**
     * Moves every live slot below {@code cursor} onto the write cursor, queueing the coalesced row moves and the
     * group's two mirror reports. A source on a vacated tail page another holder shares keeps its row.
     */
    private void moveRows(int g, int seq, PageList list, int cursor, int keptPages) {
        int pageSize = owner.pageSize();
        int[] kvPos = owner.kvPos()[g];
        PagePool pool = owner.pools()[g];
        List<RowMove> moves = owner.pendingRowMoveList();
        int[] touched = new int[2 * cursor];
        int nTouched = 0;
        int runSrc = 0;
        int runDst = 0;
        int runLen = 0;
        int w = 0;
        for (int s = 0; s < cursor; s++) {
            int src = owner.cellOf(list, s);
            int pos = kvPos[src];
            if (pos < 0) {
                continue;
            }
            if (s == w) {
                w++;
                continue;
            }
            int dst = owner.cellOf(list, w);
            int page = s / pageSize;
            if (page >= keptPages && pool.refCount(list.page(page)) > 1) {
                owner.usedRows()[g]++;
            } else {
                kvPos[src] = -1;
                touched[nTouched++] = src;
            }
            kvPos[dst] = pos;
            touched[nTouched++] = dst;
            if (runLen > 0 && src == runSrc + runLen && dst == runDst + runLen && runLen < Math.abs(runSrc - runDst)) {
                runLen++;
            } else {
                if (runLen > 0) {
                    moves.add(RowMove.of(seq, g, runSrc, runDst, runLen));
                }
                runSrc = src;
                runDst = dst;
                runLen = 1;
            }
            w++;
        }
        if (runLen > 0) {
            moves.add(RowMove.of(seq, g, runSrc, runDst, runLen));
        }
        reportFinal(owner.pendingReportList(), g, kvPos, touched, nTouched);
    }

    /**
     * Reports every touched cell once by its final {@code kv_pos} value: one absolute set report for the cells that
     * end live, one clear report for those that end empty. A cell may be touched twice (one move's source and a
     * later move's destination); the final mirror value decides.
     */
    static void reportFinal(List<RowReport> reports, int g, int[] kvPos, int[] touched, int n) {
        int[] cells = Arrays.copyOf(touched, n);
        Arrays.sort(cells);
        int[] setRows = new int[n];
        int[] setValues = new int[n];
        int[] clearRows = new int[n];
        int sets = 0;
        int clears = 0;
        for (int i = 0; i < n; i++) {
            int cell = cells[i];
            if (i > 0 && cells[i - 1] == cell) {
                continue;
            }
            if (kvPos[cell] >= 0) {
                setRows[sets] = cell;
                setValues[sets++] = kvPos[cell];
            } else {
                clearRows[clears++] = cell;
            }
        }
        if (sets > 0) {
            reports.add(RowReport.sets(g, Arrays.copyOf(setRows, sets), Arrays.copyOf(setValues, sets)));
        }
        if (clears > 0) {
            reports.add(RowReport.clears(g, Arrays.copyOf(clearRows, clears)));
        }
    }
}
