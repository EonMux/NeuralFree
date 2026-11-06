package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The arrival schedule against vLLM's {@code get_request}: infinite rate sends everything at once,
 * infinite burstiness spaces the requests by {@code 1/rate}, a sampled run is rescaled to exactly
 * {@code n/rate} seconds, and the linear and exponential ramp-ups rate every request as the reference
 * does.
 */
public class ArrivalScheduleTest {

    private static final double EPS = 1e-9;

    @Test
    public void anInfiniteRateSendsEveryRequestAtOnce() {
        ArrivalSchedule schedule = ArrivalSchedule.of(5, Double.POSITIVE_INFINITY, 1.0,
                null, null, null, new Random(0));
        for (int i = 0; i < schedule.size(); i++) {
            assertEquals(0.0, schedule.delayAt(i), EPS);
        }
    }

    @Test
    public void anInfiniteBurstinessSpacesTheRequestsConstantly() {
        ArrivalSchedule schedule = ArrivalSchedule.of(4, 10.0, Double.POSITIVE_INFINITY,
                null, null, null, new Random(0));
        assertEquals(0.1, schedule.delayAt(0), EPS);
        assertEquals(0.2, schedule.delayAt(1), EPS);
        assertEquals(0.3, schedule.delayAt(2), EPS);
        assertEquals(0.4, schedule.delayAt(3), EPS);
    }

    @Test
    public void aSampledRunEndsAtTheRequestedRate() {
        ArrivalSchedule schedule = ArrivalSchedule.of(8, 4.0, 1.0, null, null, null, new Random(7));
        assertEquals(8.0 / 4.0, schedule.delayAt(7), 1e-9);
        for (int i = 1; i < schedule.size(); i++) {
            assertTrue("the delays accumulate", schedule.delayAt(i) >= schedule.delayAt(i - 1));
        }
    }

    @Test
    public void theLinearRampUpRatesEveryRequest() {
        ArrivalSchedule schedule = ArrivalSchedule.of(4, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                "linear", 1, 4, new Random(0));
        assertEquals(1.0, schedule.rateAt(0), EPS);
        assertEquals(2.0, schedule.rateAt(1), EPS);
        assertEquals(3.0, schedule.rateAt(2), EPS);
        assertEquals(4.0, schedule.rateAt(3), EPS);
        // no rescaling under a ramp-up: the constant burstiness gaps are raw
        assertEquals(1.0, schedule.delayAt(0), EPS);
        assertEquals(1.0 + 0.5 + 1.0 / 3.0, schedule.delayAt(2), EPS);
    }

    @Test
    public void theExponentialRampUpRatesEveryRequest() {
        ArrivalSchedule schedule = ArrivalSchedule.of(3, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                "exponential", 1, 4, new Random(0));
        assertEquals(1.0, schedule.rateAt(0), EPS);
        assertEquals(2.0, schedule.rateAt(1), EPS);
        assertEquals(4.0, schedule.rateAt(2), EPS);
    }

    @Test
    public void aNonPositiveBurstinessIsRefused() {
        try {
            ArrivalSchedule.of(2, 1.0, 0.0, null, null, null, new Random(0));
            fail("a positive burstiness is required");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("positive burstiness"));
        }
    }

    @Test
    public void aNonPositiveConstantRateIsRefused() {
        try {
            ArrivalSchedule.of(2, 0.0, 1.0, null, null, null, new Random(0));
            fail("a positive request rate is required");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("non-positive request rate"));
        }
    }
}
