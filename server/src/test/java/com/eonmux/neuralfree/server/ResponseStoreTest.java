package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The response store's bounds: replay positions at the integer edge, eviction around running responses. */
class ResponseStoreTest {

    private static ResponseStore.Entry finished(String id) {
        ResponseStore.Entry e = new ResponseStore.Entry(id, List.of(), Map.of("id", id), null);
        e.finish(Map.of("id", id), List.of());
        return e;
    }

    @Test
    void aReplayPastTheLastEventIsEmptyEvenAtTheIntegerEdge() throws InterruptedException {
        ResponseStore.Entry e = new ResponseStore.Entry("r", List.of(), Map.of(), null);
        e.event("a", Map.of("sequence_number", 0));
        e.event("b", Map.of("sequence_number", 1));
        e.finish(Map.of(), List.of());
        assertEquals(2, e.eventsAfter(-1, 0).size());
        assertEquals(1, e.eventsAfter(0, 0).size());
        assertTrue(e.eventsAfter(Integer.MAX_VALUE, 0).isEmpty(), "no wrap-around to the whole log");
        assertTrue(e.drained(Integer.MAX_VALUE));
    }

    @Test
    void aNewResponseStaysWhenRunningOnesHoldTheOtherPlaces() {
        ResponseStore store = new ResponseStore(2);
        store.put(new ResponseStore.Entry("run1", List.of(), Map.of(), null));
        store.put(new ResponseStore.Entry("run2", List.of(), Map.of(), null));
        store.put(finished("done"));
        assertNotNull(store.get("done"), "the response just stored is retrievable");
        store.put(finished("next"));
        assertEquals(List.of("run1", "run2", "next"), store.ids(), "the older finished entry goes first");
    }

    @Test
    void theRequestBudgetIsBounded() {
        assertThrows(IllegalArgumentException.class,
                () -> new ServerOptions(null, ServerOptions.MAX_TIMEOUT.plus(Duration.ofSeconds(1))));
        assertEquals(ServerOptions.MAX_TIMEOUT, new ServerOptions(null, ServerOptions.MAX_TIMEOUT).requestTimeout());
    }
}
