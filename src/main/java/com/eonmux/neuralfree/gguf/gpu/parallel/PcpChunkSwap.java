package com.eonmux.neuralfree.gguf.gpu.parallel;


/**
 * vLLM's DualChunkSwap partition of one prefill's query rows over the ranks of a prefill context-parallel split
 * ({@code vllm/v1/attention/backends/utils.py}): the {@code queryLen} rows cut into {@code 2 * ranks} consecutive
 * chunks of {@code ceil(queryLen / (2 * ranks))} rows, and rank {@code r} owns chunks {@code r} and
 * {@code 2 * ranks - 1 - r} — the symmetric pair that loads a causal attention evenly, since the first chunk of
 * every rank attends few keys and its mirror chunk many. Every chunk maps to exactly one rank, so the owned rows
 * of the ranks partition the query: rank {@code r} runs its norms, projections, rope and FFN over its packed rows
 * only, and the other ranks' rows reach it through the exchange before its attention runs over the whole query.
 *
 * <p>A query shorter than the chunk grid leaves the trailing chunks empty, so a rank may own zero rows; its rows
 * array is empty and its packed step is skipped. Immutable and thread-safe; one instance serves one step.</p>
 */
public final class PcpChunkSwap {

    private final int queryLen;
    private final int ranks;
    private final int chunkCount;
    private final int chunkSize;
    /** Rows of each rank's first chunk ({@code 0} when empty). */
    private final int[] firstLen;
    /** Rows of each rank's second chunk ({@code 0} when empty). */
    private final int[] secondLen;

    private PcpChunkSwap(int queryLen, int ranks) {
        this.queryLen = queryLen;
        this.ranks = ranks;
        this.chunkCount = 2 * ranks;
        this.chunkSize = (queryLen + chunkCount - 1) / chunkCount;
        this.firstLen = new int[ranks];
        this.secondLen = new int[ranks];
        for (int rank = 0; rank < ranks; rank++) {
            firstLen[rank] = chunkLen(firstChunk(rank));
            secondLen[rank] = chunkLen(secondChunk(rank));
        }
    }

    /**
     * The partition of {@code queryLen} rows over {@code ranks} ranks.
     *
     * @throws IllegalArgumentException when the query is empty or a rank count is not positive
     */
    public static PcpChunkSwap of(int queryLen, int ranks) {
        if (queryLen < 1) {
            throw new IllegalArgumentException("a prefill context-parallel step carries at least one query row, got "
                    + queryLen);
        }
        if (ranks < 1) {
            throw new IllegalArgumentException("a prefill context-parallel split has at least one rank, got " + ranks);
        }
        return new PcpChunkSwap(queryLen, ranks);
    }

    /** The rows of the partitioned query. */
    public int queryLen() {
        return queryLen;
    }

    /** The ranks of the split. */
    public int ranks() {
        return ranks;
    }

    /** The chunks of the partition ({@code 2 * ranks}). */
    public int chunkCount() {
        return chunkCount;
    }

    /** Rows per chunk; the trailing chunks of a short query are empty. */
    public int chunkSize() {
        return chunkSize;
    }

    /** First row of {@code chunk}, at most {@link #queryLen()}. */
    public int chunkStart(int chunk) {
        checkChunk(chunk);
        return Math.min(queryLen, chunk * chunkSize);
    }

    /** Rows of {@code chunk} (0 for a trailing chunk of a short query). */
    public int chunkLen(int chunk) {
        checkChunk(chunk);
        long end = (chunk + 1) * (long) chunkSize;
        return Math.min(queryLen, (int) Math.min((long) Integer.MAX_VALUE, end)) - chunkStart(chunk);
    }

    /** The low chunk of rank {@code rank}'s pair ({@code rank} itself). */
    public int firstChunk(int rank) {
        return checkRank(rank);
    }

    /** The high chunk of rank {@code rank}'s pair ({@code 2 * ranks - 1 - rank}). */
    public int secondChunk(int rank) {
        checkRank(rank);
        return chunkCount - 1 - rank;
    }

    /** First row of rank {@code rank}'s packed rows; only defined when {@link #ownedCount(int)} is positive. */
    public int ownedStart(int rank) {
        checkRank(rank);
        if (firstLen[rank] + secondLen[rank] == 0) {
            throw new IllegalStateException("rank " + rank + " owns no row of this " + queryLen + "-row query");
        }
        return chunkStart(firstChunk(rank));
    }

    /** Rows of rank {@code rank}'s packed partition (0 when both of its chunks are empty). */
    public int ownedCount(int rank) {
        checkRank(rank);
        return firstLen[rank] + secondLen[rank];
    }

    /** Rank {@code rank}'s packed rows: its chunk pair concatenated in chunk order; empty when it owns none. */
    public int[] rows(int rank) {
        checkRank(rank);
        int[] out = new int[firstLen[rank] + secondLen[rank]];
        int first = chunkStart(firstChunk(rank));
        for (int i = 0; i < firstLen[rank]; i++) {
            out[i] = first + i;
        }
        int second = chunkStart(secondChunk(rank));
        for (int i = 0; i < secondLen[rank]; i++) {
            out[firstLen[rank] + i] = second + i;
        }
        return out;
    }

    /** The chunk {@code row} lies in. */
    public int chunkOfRow(int row) {
        checkRow(row);
        return Math.min(chunkCount - 1, row / chunkSize);
    }

    /** The one rank whose chunk pair holds {@code row}. */
    public int rankOfRow(int row) {
        int chunk = chunkOfRow(row);
        return chunk < ranks ? chunk : chunkCount - 1 - chunk;
    }

    /** Whether rank {@code rank}'s chunk pair holds {@code row}. */
    public boolean ownsRow(int rank, int row) {
        return rankOfRow(row) == checkRank(rank);
    }

    /**
     * Index of query row {@code row} in its owner rank's packed rows (the inverse of {@link #rows(int)}): a row of
     * the owner's first chunk sits before its second chunk's rows, which follow in chunk order.
     */
    public int packedIndexOfRow(int row) {
        checkRow(row);
        int rank = rankOfRow(row);
        int second = chunkStart(secondChunk(rank));
        if (row < second) {
            return row - chunkStart(firstChunk(rank));
        }
        return firstLen[rank] + (row - second);
    }

    private int checkRank(int rank) {
        if (rank < 0 || rank >= ranks) {
            throw new IndexOutOfBoundsException("rank " + rank + " outside [0, " + ranks + ")");
        }
        return rank;
    }

    private void checkChunk(int chunk) {
        if (chunk < 0 || chunk >= chunkCount) {
            throw new IndexOutOfBoundsException("chunk " + chunk + " outside [0, " + chunkCount + ")");
        }
    }

    private void checkRow(int row) {
        if (row < 0 || row >= queryLen) {
            throw new IndexOutOfBoundsException("query row " + row + " outside [0, " + queryLen + ")");
        }
    }
}
