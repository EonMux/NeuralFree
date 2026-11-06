package com.eonmux.neuralfree.server;

import java.util.Objects;

/**
 * Where this server's replicas sit in the data-parallel deployment, and how it routes among them: the global
 * data-parallel size (vLLM's {@code --data-parallel-size}), the global rank of its first replica (vLLM's
 * {@code --data-parallel-rank}, or {@code --data-parallel-start-rank} with {@code --data-parallel-size-local}), and
 * the load-balance method ({@link LoadBalance}). Local replica {@code i} serves global rank {@code startRank + i}.
 * When the server holds every rank the balancing is internal; when other servers hold the remaining ranks an
 * external balancer spreads requests over the servers (vLLM's external and hybrid load-balancing modes). Immutable;
 * {@link ServerOptions} checks the layout against its replica count.
 *
 * @param size      the global data-parallel size, at least 1
 * @param startRank the global rank of the first local replica, 0 or more
 * @param balance   how a request picks a local replica
 */
record DpLayout(int size, int startRank, LoadBalance balance) {

    /** The global data-parallel size, beside {@link ServerOptions#REPLICAS_PROPERTY}'s local count. */
    static final String SIZE_PROPERTY = "neuralfree.server.dp.size";

    /** The global rank of the first local replica. */
    static final String RANK_PROPERTY = "neuralfree.server.dp.rank";

    /** The load-balance method name ({@link LoadBalance#parse}). */
    static final String BALANCE_PROPERTY = "neuralfree.server.load.balance";

    DpLayout {
        Objects.requireNonNull(balance, "balance");
        if (size < 1) {
            throw new IllegalArgumentException(SIZE_PROPERTY + " must be 1 or more, got " + size);
        }
        if (startRank < 0) {
            throw new IllegalArgumentException(RANK_PROPERTY + " must be 0 or more, got " + startRank);
        }
    }

    /** Every one of {@code replicas} ranks served here, routed by the default method. */
    static DpLayout internal(int replicas) {
        return new DpLayout(replicas, 0, LoadBalance.SHORTEST_QUEUE);
    }

    /**
     * Whether an external balancer spreads the deployment over several servers: this server holds fewer than
     * {@code size} ranks.
     */
    boolean externalLb(int replicas) {
        return replicas < size;
    }

    /** The balancing mode as the endpoints report it: {@code internal}, {@code external} or {@code hybrid}. */
    String mode(int replicas) {
        if (!externalLb(replicas)) {
            return "internal";
        }
        return replicas == 1 ? "external" : "hybrid";
    }
}
