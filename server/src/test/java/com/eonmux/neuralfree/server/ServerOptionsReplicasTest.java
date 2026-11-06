package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The data-parallel layout options ({@code neuralfree.server.replicas}, {@code .devices}, {@code .dp.size},
 * {@code .dp.rank} and {@code .load.balance}): parsing and refusals.
 */
class ServerOptionsReplicasTest {

    private static final String[] PROPERTIES = {ServerOptions.REPLICAS_PROPERTY, ServerOptions.DEVICES_PROPERTY,
            DpLayout.SIZE_PROPERTY, DpLayout.RANK_PROPERTY, DpLayout.BALANCE_PROPERTY};

    private final Map<String, String> saved = new HashMap<>();

    @BeforeEach
    void save() {
        for (String property : PROPERTIES) {
            saved.put(property, System.getProperty(property));
            System.clearProperty(property);
        }
    }

    @AfterEach
    void restore() {
        saved.forEach((property, value) -> {
            if (value == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, value);
            }
        });
    }

    @Test
    void oneDefaultReplicaWithoutTheProperties() {
        ServerOptions options = ServerOptions.fromSystemProperties();
        assertEquals(1, options.replicas());
        assertNull(options.devices());
        assertEquals(new DpLayout(1, 0, LoadBalance.SHORTEST_QUEUE), options.dataParallel(),
                "every rank served here, by the default method");
    }

    @Test
    void theGlobalSizeDefaultsToTheReplicaCount() {
        System.setProperty(ServerOptions.REPLICAS_PROPERTY, "3");
        ServerOptions options = ServerOptions.fromSystemProperties();
        assertEquals(new DpLayout(3, 0, LoadBalance.SHORTEST_QUEUE), options.dataParallel());
        assertEquals("internal", options.dataParallel().mode(options.replicas()));
    }

    @Test
    void anExternalRankAndAHybridSliceParse() {
        System.setProperty(DpLayout.SIZE_PROPERTY, "4");
        System.setProperty(DpLayout.RANK_PROPERTY, " 3 ");
        System.setProperty(DpLayout.BALANCE_PROPERTY, "Round_Robin");
        ServerOptions external = ServerOptions.fromSystemProperties();
        assertEquals(1, external.replicas());
        assertEquals(new DpLayout(4, 3, LoadBalance.ROUND_ROBIN), external.dataParallel());
        assertEquals("external", external.dataParallel().mode(1));
        assertEquals(3, external.globalRank(0));

        System.setProperty(ServerOptions.REPLICAS_PROPERTY, "2");
        System.setProperty(DpLayout.RANK_PROPERTY, "2");
        ServerOptions hybrid = ServerOptions.fromSystemProperties();
        assertEquals("hybrid", hybrid.dataParallel().mode(hybrid.replicas()));
        assertEquals(2, hybrid.globalRank(0));
        assertEquals(3, hybrid.globalRank(1));
    }

    @Test
    void ranksPastTheGlobalSizeAreRefusedByName() {
        System.setProperty(ServerOptions.REPLICAS_PROPERTY, "2");
        System.setProperty(DpLayout.SIZE_PROPERTY, "4");
        System.setProperty(DpLayout.RANK_PROPERTY, "3");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertEquals("the 2 local replica(s) from " + DpLayout.RANK_PROPERTY + " 3 pass " + DpLayout.SIZE_PROPERTY
                + " 4; the ranks must lie inside [0, " + DpLayout.SIZE_PROPERTY + ")", e.getMessage());
        System.setProperty(DpLayout.RANK_PROPERTY, "0");
        System.setProperty(DpLayout.SIZE_PROPERTY, "1");
        assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties,
                "a global size below the local replica count");
    }

    @Test
    void unusableLayoutValuesAreRefusedByName() {
        System.setProperty(DpLayout.SIZE_PROPERTY, "four");
        assertEquals(DpLayout.SIZE_PROPERTY + " must be a whole number, got 'four'",
                assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties).getMessage());
        System.setProperty(DpLayout.SIZE_PROPERTY, "0");
        assertEquals(DpLayout.SIZE_PROPERTY + " must be 1 or more, got 0",
                assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties).getMessage());
        System.clearProperty(DpLayout.SIZE_PROPERTY);
        System.setProperty(DpLayout.RANK_PROPERTY, "-1");
        assertEquals(DpLayout.RANK_PROPERTY + " must be 0 or more, got -1",
                assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties).getMessage());
        System.clearProperty(DpLayout.RANK_PROPERTY);
        System.setProperty(DpLayout.BALANCE_PROPERTY, "random");
        assertEquals(DpLayout.BALANCE_PROPERTY + " must be shortest_queue, round_robin or minimum_tokens (also"
                + " auto, total_requests, total_tokens), got 'random'",
                assertThrows(IllegalArgumentException.class, ServerOptions::fromSystemProperties).getMessage());
    }

    @Test
    void theReplicaCountAndTheDevicesParse() {
        System.setProperty(ServerOptions.REPLICAS_PROPERTY, " 2 ");
        System.setProperty(ServerOptions.DEVICES_PROPERTY, " NVIDIA GeForce , Intel(R) Arc ");
        ServerOptions options = ServerOptions.fromSystemProperties();
        assertEquals(2, options.replicas());
        assertEquals(List.of("NVIDIA GeForce", "Intel(R) Arc"), options.devices());
    }

    @Test
    void devicesAloneSetTheReplicaCount() {
        System.clearProperty(ServerOptions.REPLICAS_PROPERTY);
        System.setProperty(ServerOptions.DEVICES_PROPERTY, " NVIDIA GeForce RTX 5080 , AMD Radeon ");
        ServerOptions options = ServerOptions.fromSystemProperties();
        assertEquals(2, options.replicas(), "one replica per device entry");
        assertEquals(List.of("NVIDIA GeForce RTX 5080", "AMD Radeon"), options.devices());
    }

    @Test
    void anExplicitReplicaCountOffTheDeviceCountIsRefusedByName() {
        System.setProperty(ServerOptions.REPLICAS_PROPERTY, "3");
        System.setProperty(ServerOptions.DEVICES_PROPERTY, "NVIDIA GeForce RTX 5080, AMD Radeon");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertEquals(ServerOptions.DEVICES_PROPERTY + " holds 2 entries but " + ServerOptions.REPLICAS_PROPERTY
                + " is 3; they must match", e.getMessage());
    }

    @Test
    void aReplicaCountBelowOneIsRefusedByName() {
        System.setProperty(ServerOptions.REPLICAS_PROPERTY, "0");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertEquals(ServerOptions.REPLICAS_PROPERTY + " must be 1 or more, got 0", e.getMessage());
    }

    @Test
    void anEmptyDeviceEntryIsRefusedByName() {
        System.setProperty(ServerOptions.DEVICES_PROPERTY, "NVIDIA GeForce, , Intel");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertEquals(ServerOptions.DEVICES_PROPERTY + " entry 1 is empty", e.getMessage());
    }

    @Test
    void aDeviceCountOffTheReplicaCountIsRefusedByName() {
        System.setProperty(ServerOptions.REPLICAS_PROPERTY, "2");
        System.setProperty(ServerOptions.DEVICES_PROPERTY, "NVIDIA GeForce RTX 5080");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                ServerOptions::fromSystemProperties);
        assertEquals(ServerOptions.DEVICES_PROPERTY + " holds 1 entries but " + ServerOptions.REPLICAS_PROPERTY
                + " is 2; they must match", e.getMessage());
    }

    @Test
    void theRecordKeepsTheLayout() {
        ServerOptions options = new ServerOptions(null, Duration.ofMinutes(1), false, 0, 2,
                List.of("NVIDIA GeForce RTX 5080", "Intel(R) Arc"));
        assertEquals(2, options.replicas());
        assertEquals(List.of("NVIDIA GeForce RTX 5080", "Intel(R) Arc"), options.devices());
        assertEquals(options.replicas(), options.withResponseStore(3).replicas());
        assertEquals(options.devices(), options.withResponseStore(3).devices());
        DpLayout layout = new DpLayout(8, 4, LoadBalance.MINIMUM_TOKENS);
        ServerOptions placed = options.withDataParallel(layout);
        assertEquals(layout, placed.dataParallel());
        assertEquals(layout, placed.withResponseStore(3).withTrustRequestChatTemplate(true).dataParallel(),
                "the copies keep the layout");
        assertEquals(5, placed.globalRank(1));
    }
}
