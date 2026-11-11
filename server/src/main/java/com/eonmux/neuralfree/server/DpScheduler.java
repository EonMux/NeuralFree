package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.media.MediaKey;

import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The data-parallel scheduler ({@link EngineService}): routes each request to one replica by SGLang's
 * {@code data_parallel_controller} load-balance methods ({@link LoadBalance}):
 *
 * <ul>
 *   <li>{@code shortest_queue}, the default: the replica with the least work takes the request (over the
 *       replicas' published queue and running counts), round robin breaks the ties among idle replicas, and a
 *       replica that recently served a prompt whose prefix the request extends wins such a tie outright — the
 *       prefix cache of that replica would reuse the rows a fresh replica would prefill again;</li>
 *   <li>{@code round_robin}: each request to the next replica in turn;</li>
 *   <li>{@code minimum_tokens}: the replica with the fewest prompt tokens in flight (submitted and not yet
 *       terminal), ties to the fewest requests in flight, then round robin.</li>
 * </ul>
 *
 * <p>The affinity tables hold the first {@code PREFIX_TOKENS} tokens of the last routed prompts, bounded per
 * replica; they approximate the engines' prefix caches, which the scheduler cannot read from another thread. A media
 * prompt routes by {@link #routingRows its rows' prefix-cache keys}, so a conversation's next turn favours the
 * replica that cached its images.
 * Routing is safe from any thread: it reads only published snapshots, atomic counters and synchronized tables.
 * Generation submissions pass the scheduler's {@link QueueLimit} after the routing.
 */
final class DpScheduler {

    /** The first tokens of a routed prompt the affinity tables keep for prefix matching. */
    private static final int PREFIX_TOKENS = 64;

    /** The routed prompts one replica remembers for affinity. */
    private static final int TABLE_ENTRIES = 64;

    private volatile EngineReplica[] replicas;
    private final LoadBalance method;
    private final AtomicInteger roundRobin = new AtomicInteger();
    /** The recent prompt prefixes per replica, oldest first; guarded by {@code table}. */
    private int[][][] affinity;
    /** Guards the affinity tables together with a {@link #replace} of the replica set. */
    private final Object table = new Object();
    /** The limit on the generation requests waiting per replica. */
    private final QueueLimit queue;

    DpScheduler(EngineReplica[] replicas, LoadBalance method) {
        this(replicas, method, 0);
    }

    /** A scheduler whose submissions pass a limit of {@code maxQueuedRequests} waiting requests per replica. */
    DpScheduler(EngineReplica[] replicas, LoadBalance method, int maxQueuedRequests) {
        this.replicas = replicas.clone();
        this.method = Objects.requireNonNull(method, "method");
        this.affinity = new int[replicas.length][0][];
        this.queue = new QueueLimit(this, maxQueuedRequests);
    }

    /**
     * Routes over {@code next} from now on: an elastic scale or a fault-tolerance rank removal swaps the replica
     * set, and the affinity tables restart empty because the pool positions no longer name the same replicas.
     */
    void replace(EngineReplica[] next) {
        Objects.requireNonNull(next, "next");
        if (next.length == 0) {
            throw new IllegalArgumentException("the scheduler needs at least one replica");
        }
        synchronized (table) {
            this.affinity = new int[next.length][0][];
            this.replicas = next.clone();
        }
    }

    /**
     * The routing view of a media prompt's first {@code PREFIX_TOKENS} rows, over the keys the prefix cache matches
     * rows by: a text row's token; a media row its item's key ({@link MediaChunk#item()}, the client's uuid or a
     * content digest) mixed with the chunk's piece of the item and the row's index in the chunk; a prompt-embedding
     * row the leading 32 bits of its own SHA-256. A keyed row has its sign bit set, so it never equals a token.
     */
    static int[] routingRows(MediaPrompt prompt) {
        int n = Math.min(prompt.nTokens(), PREFIX_TOKENS);
        int[] out = new int[n];
        int row = 0;
        for (MediaChunk c : prompt.chunks()) {
            if (row >= n) {
                break;
            }
            int take = Math.min(c.nTokens(), n - row);
            if (c.isText()) {
                System.arraycopy(c.tokens(), 0, out, row, take);
            } else if (c.isMediaItem()) {
                long bits = c.item().bits() ^ (0xC2B2AE3D27D4EB4FL * (c.piece() + 1L));
                int item = (int) (bits ^ (bits >>> 32));
                for (int i = 0; i < take; i++) {
                    out[row + i] = (item ^ (i * 0x9E3779B9)) | Integer.MIN_VALUE;
                }
            } else {
                MessageDigest digest = MediaKey.sha256();
                for (int i = 0; i < take; i++) {
                    c.digestRows(digest, i, i + 1);
                    byte[] d = digest.digest();
                    out[row + i] = ((d[0] & 0xFF) << 24 | (d[1] & 0xFF) << 16 | (d[2] & 0xFF) << 8 | (d[3] & 0xFF))
                            | Integer.MIN_VALUE;
                }
            }
            row += take;
        }
        return out;
    }

    /** The per-replica limit on waiting generation requests the submissions pass through. */
    QueueLimit queue() {
        return queue;
    }

    /**
     * Moves {@code prompt}'s affinity from {@code from}, the replica {@link #pick} chose, to {@code to}, where the
     * queue limit placed the request: the entry the pick remembered under {@code from} is dropped and the prompt is
     * remembered under {@code to}, so the next request extending the prompt follows it to the replica whose prefix
     * cache holds it. A replica no longer in the pool, or a method without affinity, changes nothing.
     */
    void rerouted(EngineReplica from, EngineReplica to, int[] prompt) {
        if (method != LoadBalance.SHORTEST_QUEUE) {
            return;
        }
        int[] prefix = Arrays.copyOf(prompt, Math.min(prompt.length, PREFIX_TOKENS));
        synchronized (table) {
            int source = indexOf(from);
            int target = indexOf(to);
            if (source < 0 || target < 0) {
                return;
            }
            int[][] entries = affinity[source];
            for (int i = entries.length - 1; i >= 0; i--) {
                if (Arrays.equals(entries[i], prefix)) {
                    int[][] kept = new int[entries.length - 1][];
                    System.arraycopy(entries, 0, kept, 0, i);
                    System.arraycopy(entries, i + 1, kept, i, entries.length - i - 1);
                    affinity[source] = kept;
                    break;
                }
            }
            remember(target, prompt);
        }
    }

    /** The pool position of {@code replica}, or -1 when it is not in the pool; guarded by {@code table}. */
    private int indexOf(EngineReplica replica) {
        EngineReplica[] current = replicas;
        for (int i = 0; i < current.length; i++) {
            if (current[i] == replica) {
                return i;
            }
        }
        return -1;
    }

    /** The replicas the scheduler routes over, in pool order, as of the call. */
    List<EngineReplica> routed() {
        return List.of(replicas);
    }

    /** The method the scheduler routes by. */
    LoadBalance method() {
        return method;
    }

    /**
     * The replica for a request with {@code prompt} (null for a routing-free call, as the pooling forwards and the
     * embedding-row prompts) by the scheduler's method.
     */
    EngineReplica pick(int[] prompt) {
        EngineReplica[] current = replicas;
        return switch (method) {
            case SHORTEST_QUEUE -> shortestQueue(current, prompt);
            case ROUND_ROBIN -> current[mod(roundRobin.getAndIncrement(), current.length)];
            case MINIMUM_TOKENS -> minimumTokens(current);
        };
    }

    /**
     * The fewest prompt tokens in flight, ties to the fewest requests in flight (SGLang's {@code total_tokens}
     * order), round robin among the replicas still tied.
     */
    private EngineReplica minimumTokens(EngineReplica[] current) {
        long leastTokens = Long.MAX_VALUE;
        int leastLoad = Integer.MAX_VALUE;
        int[] picks = new int[current.length];
        int candidates = 0;
        for (int i = 0; i < current.length; i++) {
            long tokens = current[i].promptTokensInFlight();
            int load = current[i].load();
            if (tokens < leastTokens || (tokens == leastTokens && load < leastLoad)) {
                leastTokens = tokens;
                leastLoad = load;
                candidates = 0;
            }
            if (tokens == leastTokens && load == leastLoad) {
                picks[candidates++] = i;
            }
        }
        return current[picks[mod(roundRobin.getAndIncrement(), candidates)]];
    }

    /**
     * The least-loaded replica, the one with the longest remembered prefix match among the least loaded, round
     * robin when none matches.
     */
    private EngineReplica shortestQueue(EngineReplica[] current, int[] prompt) {
        int least = Integer.MAX_VALUE;
        int[] picks = new int[current.length];
        int candidates = 0;
        for (int i = 0; i < current.length; i++) {
            BatchStats s = current[i].stats();
            int load = s.running() + s.waiting();
            if (load < least) {
                least = load;
                candidates = 0;
            }
            if (load == least) {
                picks[candidates++] = i;
            }
        }
        if (prompt == null) {
            return current[picks[mod(roundRobin.getAndIncrement(), candidates)]];
        }
        int best = -1;
        int bestLen = 0;
        for (int i = 0; i < candidates; i++) {
            int len = longestRemembered(picks[i], prompt);
            if (len > bestLen) {
                bestLen = len;
                best = picks[i];
            }
        }
        int chosen = best >= 0 ? best : picks[mod(roundRobin.getAndIncrement(), candidates)];
        remember(chosen, prompt);
        return current[chosen];
    }

    /** The longest remembered prefix length replica {@code r} holds for {@code prompt}. */
    private int longestRemembered(int r, int[] prompt) {
        synchronized (table) {
            int[][] entries = affinity[r];
            int bestLen = 0;
            for (int i = 0; i < entries.length; i++) {
                int len = commonLength(entries[i], prompt);
                if (len > bestLen) {
                    bestLen = len;
                }
            }
            return bestLen;
        }
    }

    /** Remembers the prompt's leading tokens under replica {@code r}, evicting the oldest when the table is full. */
    private void remember(int r, int[] prompt) {
        int[] prefix = Arrays.copyOf(prompt, Math.min(prompt.length, PREFIX_TOKENS));
        synchronized (table) {
            int[][] entries = affinity[r];
            if (entries.length >= TABLE_ENTRIES) {
                int[][] shifted = new int[TABLE_ENTRIES][];
                System.arraycopy(entries, 1, shifted, 0, TABLE_ENTRIES - 1);
                shifted[TABLE_ENTRIES - 1] = prefix;
                affinity[r] = shifted;
                return;
            }
            int[][] grown = Arrays.copyOf(entries, entries.length + 1);
            grown[entries.length] = prefix;
            affinity[r] = grown;
        }
    }

    /** The longest common prefix length of the two arrays. */
    private static int commonLength(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            if (a[i] != b[i]) {
                return i;
            }
        }
        return n;
    }

    /** The non-negative remainder, for a counter that may wrap. */
    private static int mod(int value, int bound) {
        return Math.floorMod(value, bound);
    }
}
