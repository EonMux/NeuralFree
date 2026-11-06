package com.eonmux.neuralfree.gguf.engine.batch;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The scheduling policies' rules, as vLLM's scheduler applies them: names parse case-insensitively and refuse
 * naming the setting, {@code fcfs} takes priority 0 only, and the service order is arrival under {@code fcfs} and
 * {@code (priority, arrival)} under {@code priority} (vLLM's {@code Request.__lt__}).
 */
public class SchedulingPolicyTest {

    @Test
    public void namesParseCaseInsensitivelyAndRefuseNamingTheSetting() {
        assertEquals(SchedulingPolicy.FCFS, SchedulingPolicy.parse("fcfs", "k"));
        assertEquals(SchedulingPolicy.PRIORITY, SchedulingPolicy.parse(" PRIORITY ", "k"));
        assertEquals("fcfs", SchedulingPolicy.FCFS.wireName());
        assertEquals("priority", SchedulingPolicy.PRIORITY.wireName());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SchedulingPolicy.parse("round_robin", "--scheduling-policy"));
        assertTrue(e.getMessage(), e.getMessage().startsWith("--scheduling-policy=round_robin"));
        assertThrows(IllegalArgumentException.class, () -> SchedulingPolicy.parse(null, "k"));
        assertThrows(IllegalArgumentException.class, () -> SchedulingPolicy.parse("", "k"));
    }

    @Test
    public void fcfsTakesPriorityZeroOnlyAndPriorityTakesEveryValue() {
        assertNull(SchedulingPolicy.FCFS.priorityRefusal(0));
        String refusal = SchedulingPolicy.FCFS.priorityRefusal(-2);
        assertNotNull(refusal);
        assertTrue(refusal, refusal.startsWith("priority -2"));
        assertTrue(refusal, refusal.contains("neuralfree.scheduling.policy=priority"));
        assertNotNull(SchedulingPolicy.FCFS.priorityRefusal(Integer.MAX_VALUE));
        assertNull(SchedulingPolicy.PRIORITY.priorityRefusal(0));
        assertNull(SchedulingPolicy.PRIORITY.priorityRefusal(Integer.MIN_VALUE));
        assertNull(SchedulingPolicy.PRIORITY.priorityRefusal(7));
    }

    @Test
    public void fcfsServesByArrivalAlone() {
        assertTrue(SchedulingPolicy.FCFS.servesBefore(0, 1, 0, 2));
        assertFalse(SchedulingPolicy.FCFS.servesBefore(0, 2, 0, 1));
        assertTrue("the priority plays no part", SchedulingPolicy.FCFS.servesBefore(9, 1, -9, 2));
    }

    @Test
    public void priorityServesTheLowestValueFirstAndTiesByArrival() {
        assertTrue(SchedulingPolicy.PRIORITY.servesBefore(-1, 5, 0, 1));
        assertFalse(SchedulingPolicy.PRIORITY.servesBefore(3, 1, 2, 9));
        assertTrue("ties by arrival", SchedulingPolicy.PRIORITY.servesBefore(4, 1, 4, 2));
        assertFalse("ties by arrival", SchedulingPolicy.PRIORITY.servesBefore(4, 2, 4, 1));
        assertFalse("a request is not served before itself", SchedulingPolicy.PRIORITY.servesBefore(4, 2, 4, 2));
        assertTrue("the full int range orders without overflow",
                SchedulingPolicy.PRIORITY.servesBefore(Integer.MIN_VALUE, 9, Integer.MAX_VALUE, 1));
    }
}
