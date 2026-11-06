package com.eonmux.neuralfree.server;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The data-parallel launch flags of {@link ServerLaunchArgs}: vLLM's {@code --data-parallel-*} set and SGLang's
 * {@code --dp-size} and {@code --load-balance-method}, collected while the arguments are read and resolved together
 * into the {@code neuralfree.server.replicas}, {@code .devices}, {@code .dp.size}, {@code .dp.rank} and
 * {@code .load.balance} properties once all are known, as vLLM resolves them:
 *
 * <ul>
 *   <li>a size alone serves every rank here behind the internal balancer;</li>
 *   <li>a rank ({@code --data-parallel-rank}, external-LB mode) serves that one rank here, other servers serve the
 *       others and an external balancer spreads the requests over them;</li>
 *   <li>a local size ({@code --data-parallel-size-local}, with {@code --data-parallel-start-rank}) serves that many
 *       consecutive ranks here behind the internal balancer, the rest elsewhere (hybrid mode).</li>
 * </ul>
 *
 * <p>The flags vLLM uses to coordinate ranks across processes ({@code --data-parallel-address}, {@code -rpc-port},
 * {@code -backend}, {@code --data-parallel-multi-port-external-lb}) are refused by name: the replicas of this server
 * decode independently and need no coordination. A parse accumulator, used by one parse only.
 */
final class DpLaunchFlags {

    /** The flags taking a value. */
    static final Set<String> VALUED = Set.of("-dp", "--data-parallel-size", "--dp-size", "-dpl",
            "--data-parallel-size-local", "-dpn", "--data-parallel-rank", "-dpr", "--data-parallel-start-rank",
            "--data-parallel-devices", "--dp-devices", "--load-balance-method");

    /** The switches. */
    static final Set<String> SWITCHES = Set.of("-dpe", "--data-parallel-external-lb", "-dph",
            "--data-parallel-hybrid-lb");

    /** The cross-process coordination flags, refused. */
    static final Set<String> REFUSED = Set.of("-dpa", "--data-parallel-address", "-dpp", "--data-parallel-rpc-port",
            "-dpb", "--data-parallel-backend", "-dpm", "--data-parallel-multi-port-external-lb");

    private static final String SIZE = "--data-parallel-size";
    private static final String LOCAL = "--data-parallel-size-local";
    private static final String RANK = "--data-parallel-rank";
    private static final String START = "--data-parallel-start-rank";
    private static final String EXTERNAL = "--data-parallel-external-lb";
    private static final String HYBRID = "--data-parallel-hybrid-lb";
    private static final String DEVICES = "--data-parallel-devices";

    private Integer size;
    private Integer local;
    private Integer rank;
    private Integer start;
    private boolean external;
    private boolean hybrid;
    private List<String> devices;
    private LoadBalance balance;

    /** Records the value of one {@link #VALUED} flag, refused when it is not a usable value. */
    void value(String flag, String value) {
        switch (flag) {
            case "-dp", SIZE, "--dp-size" -> size = count(SIZE, value);
            case "-dpl", LOCAL -> local = count(LOCAL, value);
            case "-dpn", RANK -> rank = rank(RANK, value);
            case "-dpr", START -> start = rank(START, value);
            case DEVICES, "--dp-devices" -> devices = devices(value);
            case "--load-balance-method" -> balance = LoadBalance.parse(value, flag);
            default -> throw new IllegalArgumentException("not a data-parallel flag: " + flag);
        }
    }

    /** Records one {@link #SWITCHES} flag. */
    void toggle(String flag) {
        switch (flag) {
            case "-dpe", EXTERNAL -> external = true;
            case "-dph", HYBRID -> hybrid = true;
            default -> throw new IllegalArgumentException("not a data-parallel switch: " + flag);
        }
    }

    /** The refusal of a {@link #REFUSED} flag. */
    static IllegalArgumentException refusal(String flag) {
        return new IllegalArgumentException(flag + " coordinates data-parallel ranks across processes (vLLM); the"
                + " replicas of this server decode independently, so give each server its ranks with "
                + SIZE + " and " + RANK + " (or " + LOCAL + " and " + START + ") and balance over the servers");
    }

    /**
     * Puts the resolved properties into {@code properties}; nothing when no data-parallel flag was given.
     *
     * @throws IllegalArgumentException when the flags contradict each other, naming them
     */
    void resolveInto(Map<String, String> properties) {
        Integer replicas;
        if (rank != null || external) {
            replicas = external();
        } else if (local != null) {
            replicas = hybrid();
        } else {
            replicas = internal();
        }
        if (replicas != null) {
            properties.put(ServerOptions.REPLICAS_PROPERTY, replicas.toString());
            properties.put(DpLayout.SIZE_PROPERTY, Integer.toString(size == null ? replicas : size));
            properties.put(DpLayout.RANK_PROPERTY, Integer.toString(firstRank()));
        }
        if (devices != null) {
            if (replicas != null && devices.size() != replicas) {
                throw new IllegalArgumentException(DEVICES + " names " + devices.size() + " device(s) but the"
                        + " data-parallel flags place " + replicas + " replica(s) here; give one device per replica");
            }
            properties.put(ServerOptions.DEVICES_PROPERTY, String.join(",", devices));
        }
        if (balance != null) {
            properties.put(DpLayout.BALANCE_PROPERTY, balance.label());
        }
    }

    /** The external-LB layout: one replica serving {@code rank} of {@code size}. */
    private Integer external() {
        if (rank == null) {
            throw new IllegalArgumentException(EXTERNAL + " needs " + RANK + ": each server behind the external"
                    + " balancer serves the one rank it names");
        }
        if (hybrid) {
            throw new IllegalArgumentException(HYBRID + " and " + EXTERNAL + " (or " + RANK + ") cannot be"
                    + " enabled together");
        }
        if (start != null) {
            throw new IllegalArgumentException(RANK + " and " + START + " both name this server's first rank;"
                    + " an external-LB server takes " + RANK + " only");
        }
        if (local != null && local != 1) {
            throw new IllegalArgumentException("an external data-parallel rank serves one replica: " + LOCAL
                    + " must be 1 or omitted, got " + local);
        }
        int total = size == null ? 1 : size;
        if (rank >= total) {
            throw new IllegalArgumentException(RANK + " " + rank + " must be below " + SIZE + " " + total);
        }
        return 1;
    }

    /** The hybrid layout: {@code local} consecutive ranks from {@code start} served here. */
    private Integer hybrid() {
        int total = size == null ? local : size;
        if (local > total) {
            throw new IllegalArgumentException(LOCAL + " " + local + " exceeds " + SIZE + " " + total);
        }
        int first = start == null ? 0 : start;
        if ((long) first + local > total) {
            throw new IllegalArgumentException(START + " " + first + " with " + LOCAL + " " + local
                    + " passes " + SIZE + " " + total + "; the ranks must lie inside [0, " + total + ")");
        }
        return local;
    }

    /** The internal layout: every rank of {@code size} served here; null when no size was given. */
    private Integer internal() {
        if (hybrid) {
            throw new IllegalArgumentException(HYBRID + " needs " + LOCAL + ", the number of ranks this server holds");
        }
        if (start != null) {
            throw new IllegalArgumentException(START + " needs " + LOCAL + ", the number of ranks this server holds");
        }
        return size;
    }

    /** The global rank of the first local replica. */
    private int firstRank() {
        if (rank != null) {
            return rank;
        }
        return start == null ? 0 : start;
    }

    private static int count(String flag, String value) {
        return ServerLaunchArgs.whole(flag, value, 1, Integer.MAX_VALUE).intValue();
    }

    private static int rank(String flag, String value) {
        return ServerLaunchArgs.whole(flag, value, 0, Integer.MAX_VALUE - 1).intValue();
    }

    /** The comma-separated device-name filters, refused when an entry is empty. */
    private static List<String> devices(String value) {
        List<String> out = new ArrayList<>();
        String[] entries = value.split(",", -1);
        for (int i = 0; i < entries.length; i++) {
            String entry = entries[i].trim();
            if (entry.isEmpty()) {
                throw new IllegalArgumentException(DEVICES + " entry " + i + " is empty");
            }
            out.add(entry);
        }
        return List.copyOf(out);
    }
}
