package com.eonmux.neuralfree.gguf.ops.sparse;

import java.util.Objects;

/**
 * The two-level candidate selection of a DeepSeek-V4.1 indexer source (models/deepseek41.cpp:588-611): level one
 * scores blocks of {@code blockSize} consecutive compressed rows by the best row score, always keeps the block that
 * holds the newest visible row, and returns the best {@code topKBlocks} blocks; level two restricts a later source's
 * own row scores to those blocks before its row top-k. The score arithmetic itself belongs to
 * {@link CpuDsaIndexer}; this class only folds rows into blocks and masks.
 */
public final class Dsv41Selection {

    private Dsv41Selection() {
    }

    /**
     * Blocks of {@code nKv} rows, {@code ceil(nKv / blockSize)}.
     *
     * @throws IllegalArgumentException for a non-positive block size or negative row count
     */
    public static int blockCount(int nKv, int blockSize) {
        if (blockSize < 1 || nKv < 0) {
            throw new IllegalArgumentException("block selection needs a positive block size and non-negative rows,"
                    + " got " + blockSize + " / " + nKv);
        }
        return (nKv + blockSize - 1) / blockSize;
    }

    /**
     * Selects one query's candidate blocks into {@code blocks[blocksOff .. blocksOff + k)}: block {@code b} scores
     * the maximum row score over its rows, the block of the newest visible row ({@code (nVis - 1) / blockSize}) gets
     * {@code +1e30} so a partly filled block is always kept, and the best {@code k = min(topKBlocks, blockCount)}
     * blocks are kept (deepseek41.cpp:588-597).
     *
     * @param scores      row scores of the query, {@code nKv} values at {@code scoresOff}; unselected entries of
     *                    {@code blockScores} are overwritten
     * @param blockScores scratch of at least {@code blockCount} values
     * @return {@code k = min(topKBlocks, blockCount)}; blocks whose every row is invisible (all scores
     *         {@code -inf}) come back as {@code -1}
     * @throws IllegalArgumentException when an array is too short or {@code nVis} exceeds {@code nKv}
     */
    public static int blockTopK(float[] scores, int scoresOff, int nKv, float[] blockScores, int blockSize,
                                int topKBlocks, int nVis, int[] blocks, int blocksOff) {
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(blockScores, "blockScores");
        Objects.requireNonNull(blocks, "blocks");
        int nBlk = blockCount(nKv, blockSize);
        if (nVis < 0 || nVis > nKv) {
            throw new IllegalArgumentException("block selection needs 0 <= n_visible <= n_kv, got " + nVis + " / "
                    + nKv);
        }
        if (topKBlocks < 1 || topKBlocks > SparseKernels.TOPK_MAX) {
            throw new IllegalArgumentException("block selection needs a positive top-k block count up to "
                    + SparseKernels.TOPK_MAX + ", got " + topKBlocks);
        }
        if (scoresOff < 0 || (long) scoresOff + nKv > scores.length) {
            throw new IllegalArgumentException("block selection needs " + nKv + " scores at " + scoresOff + ", got "
                    + scores.length);
        }
        if (blockScores.length < nBlk || blocksOff < 0 || blocks.length - blocksOff < Math.min(topKBlocks, nBlk)) {
            throw new IllegalArgumentException("block selection needs " + nBlk + " block scores and "
                    + Math.min(topKBlocks, nBlk) + " slots at " + blocksOff);
        }
        for (int b = 0; b < nBlk; b++) {
            blockScores[b] = Float.NEGATIVE_INFINITY;
        }
        for (int row = 0; row < nKv; row++) {
            int b = row / blockSize;
            float s = scores[scoresOff + row];
            if (s > blockScores[b]) {
                blockScores[b] = s;
            }
        }
        if (nVis > 0) {
            blockScores[(nVis - 1) / blockSize] += 1e30f;
        }
        int k = Math.min(topKBlocks, nBlk);
        CpuTopK.topKRow(blockScores, 0, nBlk, k, blocks, blocksOff);
        return k;
    }

    /**
     * Masks every row of {@code scores} whose block is not one of {@code blocks[blocksOff ..
     * blocksOff + blocksStride)} to {@code -inf} (deepseek41.cpp:598-611): a later source's row scores restricted to
     * an earlier source's candidate blocks.
     *
     * @param blockMask scratch of at least {@code blockCount(nKv, blockSize)} entries
     * @throws IllegalArgumentException when an array is too short
     */
    public static void restrictToBlocks(float[] scores, int scoresOff, int nKv, int blockSize, int[] blocks,
                                        int blocksOff, int blocksStride, int[] blockMask) {
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(blocks, "blocks");
        Objects.requireNonNull(blockMask, "blockMask");
        int nBlk = blockCount(nKv, blockSize);
        if (blocksStride < 0 || blocksOff < 0 || (long) blocksOff + blocksStride > blocks.length) {
            throw new IllegalArgumentException("block list holds " + blocks.length + " entries, needs "
                    + blocksStride + " at " + blocksOff);
        }
        if (blockMask.length < nBlk) {
            throw new IllegalArgumentException("block mask holds " + blockMask.length + " entries, needs " + nBlk);
        }
        if (scoresOff < 0 || (long) scoresOff + nKv > scores.length) {
            throw new IllegalArgumentException("block mask needs " + nKv + " scores at " + scoresOff + ", got "
                    + scores.length);
        }
        for (int b = 0; b < nBlk; b++) {
            blockMask[b] = 0;
        }
        for (int j = 0; j < blocksStride; j++) {
            int b = blocks[blocksOff + j];
            if (b >= 0 && b < nBlk) {
                blockMask[b] = 1;
            }
        }
        for (int row = 0; row < nKv; row++) {
            if (blockMask[row / blockSize] == 0) {
                scores[scoresOff + row] = Float.NEGATIVE_INFINITY;
            }
        }
    }
}
