package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The data-parallel launch flags ({@link DpLaunchFlags}): vLLM's {@code --data-parallel-*} set and SGLang's
 * {@code --dp-size} and {@code --load-balance-method} resolve into the layout properties as vLLM resolves its
 * internal, external-LB and hybrid modes, and contradicting or unsupported flags are refused by name.
 */
class ServerLaunchArgsDpTest {

    private static Map<String, String> parse(String... args) {
        return ServerLaunchArgs.parse(args).properties();
    }

    private static String refusalOf(String... args) {
        return assertThrows(IllegalArgumentException.class, () -> ServerLaunchArgs.parse(args)).getMessage();
    }

    @Test
    void aSizeAloneServesEveryRankHere() {
        Map<String, String> expected = Map.of(
                ServerOptions.REPLICAS_PROPERTY, "2",
                DpLayout.SIZE_PROPERTY, "2",
                DpLayout.RANK_PROPERTY, "0");
        assertEquals(expected, parse("-dp", "2"));
        assertEquals(expected, parse("--data-parallel-size", "2"));
        assertEquals(expected, parse("--dp-size", "2"), "SGLang's spelling");
    }

    @Test
    void aRankServesThatOneRankBehindAnExternalBalancer() {
        Map<String, String> expected = Map.of(
                ServerOptions.REPLICAS_PROPERTY, "1",
                DpLayout.SIZE_PROPERTY, "4",
                DpLayout.RANK_PROPERTY, "2");
        assertEquals(expected, parse("--data-parallel-size", "4", "--data-parallel-rank", "2"));
        assertEquals(expected, parse("-dp", "4", "-dpn", "2", "-dpe"));
        assertEquals(expected, parse("-dpn", "2", "-dpl", "1", "-dp", "4"), "a local size of 1 is the rank's own");
    }

    @Test
    void aLocalSizeServesConsecutiveRanksFromTheStartRank() {
        assertEquals(Map.of(
                ServerOptions.REPLICAS_PROPERTY, "2",
                DpLayout.SIZE_PROPERTY, "4",
                DpLayout.RANK_PROPERTY, "2"),
                parse("-dp", "4", "--data-parallel-size-local", "2", "--data-parallel-start-rank", "2", "-dph"));
        assertEquals(Map.of(
                ServerOptions.REPLICAS_PROPERTY, "3",
                DpLayout.SIZE_PROPERTY, "3",
                DpLayout.RANK_PROPERTY, "0"),
                parse("-dpl", "3"), "a local size alone serves every rank");
    }

    @Test
    void theDeviceListAndTheMethodMapToTheirProperties() {
        assertEquals(Map.of(ServerOptions.DEVICES_PROPERTY, "NVIDIA GeForce RTX 5080,AMD Radeon"),
                parse("--data-parallel-devices", " NVIDIA GeForce RTX 5080 , AMD Radeon "),
                "the list alone implies the replica count");
        Map<String, String> both = parse("-dp", "2", "--dp-devices", "RTX 5080,Radeon",
                "--load-balance-method", "round_robin");
        assertEquals("2", both.get(ServerOptions.REPLICAS_PROPERTY));
        assertEquals("RTX 5080,Radeon", both.get(ServerOptions.DEVICES_PROPERTY));
        assertEquals("round_robin", both.get(DpLayout.BALANCE_PROPERTY));
        assertEquals("minimum_tokens", parse("--load-balance-method", "TOTAL_TOKENS").get(DpLayout.BALANCE_PROPERTY));
        assertEquals("shortest_queue", parse("--load-balance-method", "auto").get(DpLayout.BALANCE_PROPERTY));
        assertEquals("shortest_queue",
                parse("--load-balance-method", "total_requests").get(DpLayout.BALANCE_PROPERTY));
    }

    @Test
    void withoutDataParallelFlagsNoLayoutPropertyIsSet() {
        Map<String, String> properties = parse("-m", "model.gguf", "--port", "8081");
        assertFalse(properties.containsKey(ServerOptions.REPLICAS_PROPERTY));
        assertFalse(properties.containsKey(DpLayout.SIZE_PROPERTY));
        assertFalse(properties.containsKey(DpLayout.RANK_PROPERTY));
    }

    @Test
    void unusableValuesAreRefusedByFlag() {
        assertEquals("--data-parallel-size must be between 1 and 2147483647, got 0", refusalOf("-dp", "0"));
        assertEquals("--data-parallel-size must be a whole number, got 'two'", refusalOf("--dp-size", "two"));
        assertEquals("--data-parallel-rank must be between 0 and 2147483646, got -1", refusalOf("-dpn", "-1"));
        assertEquals("-dpl needs a value", refusalOf("-dpl"), "the refusal names the spelling given");
        assertEquals("--data-parallel-size-local needs a value", refusalOf("--data-parallel-size-local"));
        assertEquals("--data-parallel-devices entry 1 is empty", refusalOf("--dp-devices", "a,,b"));
        assertEquals("--load-balance-method must be shortest_queue, round_robin or minimum_tokens (also auto,"
                + " total_requests, total_tokens), got 'fastest'", refusalOf("--load-balance-method", "fastest"));
    }

    @Test
    void contradictingFlagsAreRefusedByName() {
        assertEquals("--data-parallel-rank 4 must be below --data-parallel-size 4",
                refusalOf("-dp", "4", "-dpn", "4"));
        assertEquals("--data-parallel-rank 1 must be below --data-parallel-size 1", refusalOf("-dpn", "1"));
        assertEquals("--data-parallel-external-lb needs --data-parallel-rank: each server behind the external"
                + " balancer serves the one rank it names", refusalOf("-dp", "2", "-dpe"));
        assertEquals("an external data-parallel rank serves one replica: --data-parallel-size-local must be 1 or"
                + " omitted, got 2", refusalOf("-dp", "4", "-dpn", "0", "-dpl", "2"));
        assertEquals("--data-parallel-hybrid-lb and --data-parallel-external-lb (or --data-parallel-rank) cannot"
                + " be enabled together", refusalOf("-dp", "4", "-dpn", "0", "-dph"));
        assertEquals("--data-parallel-rank and --data-parallel-start-rank both name this server's first rank; an"
                + " external-LB server takes --data-parallel-rank only", refusalOf("-dp", "4", "-dpn", "1", "-dpr", "1"));
        assertEquals("--data-parallel-size-local 3 exceeds --data-parallel-size 2", refusalOf("-dp", "2", "-dpl", "3"));
        assertEquals("--data-parallel-start-rank 3 with --data-parallel-size-local 2 passes --data-parallel-size 4;"
                + " the ranks must lie inside [0, 4)", refusalOf("-dp", "4", "-dpl", "2", "-dpr", "3"));
        assertEquals("--data-parallel-hybrid-lb needs --data-parallel-size-local, the number of ranks this server"
                + " holds", refusalOf("-dp", "4", "-dph"));
        assertEquals("--data-parallel-start-rank needs --data-parallel-size-local, the number of ranks this server"
                + " holds", refusalOf("-dp", "4", "-dpr", "2"));
        assertEquals("--data-parallel-devices names 1 device(s) but the data-parallel flags place 2 replica(s)"
                + " here; give one device per replica", refusalOf("-dp", "2", "--dp-devices", "RTX 5080"));
    }

    @Test
    void crossProcessCoordinationFlagsAreRefusedByName() {
        for (String flag : new String[]{"--data-parallel-address", "-dpa", "--data-parallel-rpc-port", "-dpp",
                "--data-parallel-backend", "-dpb", "--data-parallel-multi-port-external-lb", "-dpm"}) {
            String message = refusalOf("-dp", "2", flag, "x");
            assertEquals(flag + " coordinates data-parallel ranks across processes (vLLM); the replicas of this"
                    + " server decode independently, so give each server its ranks with --data-parallel-size and"
                    + " --data-parallel-rank (or --data-parallel-size-local and --data-parallel-start-rank) and"
                    + " balance over the servers", message);
        }
    }

    @Test
    void theResolvedPropertiesOpenTheLayoutTheFlagsDescribe() {
        Map<String, String> properties = parse("-dp", "4", "-dpl", "2", "-dpr", "2",
                "--load-balance-method", "minimum_tokens");
        Map<String, String> saved = new java.util.HashMap<>();
        for (String key : new String[]{ServerOptions.REPLICAS_PROPERTY, ServerOptions.DEVICES_PROPERTY,
                DpLayout.SIZE_PROPERTY, DpLayout.RANK_PROPERTY, DpLayout.BALANCE_PROPERTY}) {
            saved.put(key, System.getProperty(key));
            System.clearProperty(key);
        }
        try {
            properties.forEach(System::setProperty);
            ServerOptions options = ServerOptions.fromSystemProperties();
            assertEquals(2, options.replicas());
            assertEquals(new DpLayout(4, 2, LoadBalance.MINIMUM_TOKENS), options.dataParallel());
            assertEquals(3, options.globalRank(1));
        } finally {
            saved.forEach((key, value) -> {
                if (value == null) {
                    System.clearProperty(key);
                } else {
                    System.setProperty(key, value);
                }
            });
        }
    }
}
