package com.eonmux.neuralfree.acceleration.ocl;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import org.junit.Assume;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.IntToDoubleFunction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Pins the device-selection policy.
 *
 * <p>The fixtures are measured values of a hybrid laptop (Arrow Lake iGPU plus
 * RTX 5080 Laptop GPU): the integrated GPU leads on every reported attribute yet
 * is 123x slower, so a ranking by attributes alone picks the wrong device.</p>
 */
public class CLDeviceRankingTest {

    /** Intel(R) Graphics on Arrow Lake-S: 244 GFLOP/s measured. */
    private static CLDeviceRanking.Profile arrowLakeIgpu() {
        return new CLDeviceRanking.Profile(
                "Intel(R) Graphics", "Intel(R) Corporation", Boolean.TRUE, 64, 1900);
    }

    /** NVIDIA GeForce RTX 5080 Laptop GPU: 30048 GFLOP/s measured. */
    private static CLDeviceRanking.Profile rtx5080() {
        return new CLDeviceRanking.Profile(
                "NVIDIA GeForce RTX 5080 Laptop GPU", "NVIDIA Corporation",
                Boolean.FALSE, 60, 1500);
    }

    /** A probe that fails the test if it is called at all. */
    private static IntToDoubleFunction neverProbed() {
        return index -> {
            fail("the probe ran for candidate " + index
                    + ", but the class test should already have decided");
            return Double.NaN;
        };
    }

    @Test
    public void discreteGpuBeatsABetterLookingIntegratedOne() {
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(arrowLakeIgpu(), rtx5080());

        // Index 1: enumeration lists the iGPU first. One discrete device among
        // integrated ones is decided by the device class, without a probe.
        assertEquals(1, CLDeviceRanking.rank(profiles, neverProbed()));
    }

    @Test
    public void enumerationOrderDoesNotDecide() {
        // Same two devices, listed the other way round. A ranking that quietly
        // fell back to "first wins" would pass the test above and fail this one.
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(rtx5080(), arrowLakeIgpu());
        assertEquals(0, CLDeviceRanking.rank(profiles, neverProbed()));
    }

    @Test
    public void aDeviceThatWouldNotAnswerTheQueryStaysInTheRunning() {
        // CL_DEVICE_HOST_UNIFIED_MEMORY is deprecated in OpenCL 2.0, so a
        // conforming ICD may refuse it. Unknown must mean unknown: this device
        // has to reach the probe, not be discarded as if it were integrated.
        CLDeviceRanking.Profile unknown =
                new CLDeviceRanking.Profile("Mystery Accelerator", "Someone", null, 8, 1000);
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(arrowLakeIgpu(), unknown);

        List<Integer> probed = new ArrayList<>();
        int winner = CLDeviceRanking.rank(profiles, index -> {
            probed.add(index);
            return index == 1 ? 500.0 : 244.0;
        });

        assertEquals(1, winner);
        assertEquals("both candidates should have been offered to the probe",
                Arrays.asList(0, 1), probed);
    }

    @Test
    public void measurementDecidesBetweenTwoDiscreteDevices() {
        // Attributes say the first is far ahead: 96 compute units at 2100 MHz
        // against 60 at 1500. Measurement says otherwise, and measurement wins.
        CLDeviceRanking.Profile paperTiger = new CLDeviceRanking.Profile(
                "Vendor X Accelerator", "Vendor X", Boolean.FALSE, 96, 2100);
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(paperTiger, rtx5080());

        int winner = CLDeviceRanking.rank(profiles, index -> index == 0 ? 900.0 : 30048.0);
        assertEquals(1, winner);
    }

    @Test
    public void oneUnmeasurableCandidateSendsEveryCandidateBackToAttributes() {
        CLDeviceRanking.Profile otherDiscrete = new CLDeviceRanking.Profile(
                "Radeon RX 7900 XTX", "Advanced Micro Devices, Inc.", Boolean.FALSE, 96, 2500);
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(otherDiscrete, rtx5080());

        // Ranking a measured device against an unmeasured one would be comparing
        // a number to the absence of one, so the whole comparison is abandoned.
        // Attributes then give AMD 96 x 64 x 2500 = 15.4M against NVIDIA
        // 60 x 128 x 1500 = 11.5M.
        int winner = CLDeviceRanking.rank(profiles, index -> index == 0 ? Double.NaN : 30048.0);
        assertEquals(0, winner);
    }

    @Test
    public void aProbeThatThrowsIsTreatedAsUnmeasured() {
        CLDeviceRanking.Profile otherDiscrete = new CLDeviceRanking.Profile(
                "Radeon RX 7900 XTX", "Advanced Micro Devices, Inc.", Boolean.FALSE, 96, 2500);
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(otherDiscrete, rtx5080());

        int winner = CLDeviceRanking.rank(profiles, index -> {
            throw new RuntimeException("driver fell over");
        });
        assertEquals("attribute ranking should still produce an answer", 0, winner);
    }

    @Test
    public void attributeFallbackNormalisesComputeUnitsAcrossVendors() {
        // Both marked discrete so the class test cannot decide, and no probe, so
        // this exercises the lane table alone. Raw compute units and raw clock
        // both favour Intel; lanes per compute unit is what stops that being the
        // answer.
        CLDeviceRanking.Profile intelDiscrete = new CLDeviceRanking.Profile(
                "Intel(R) Arc(tm) A770", "Intel(R) Corporation", Boolean.FALSE, 64, 1900);
        List<CLDeviceRanking.Profile> profiles = Arrays.asList(intelDiscrete, rtx5080());

        assertEquals(1, CLDeviceRanking.rank(profiles, null));
    }

    @Test
    public void anExplicitChoiceOverridesEverything() {
        String previous = System.getProperty(CLDeviceRanking.DEVICE_PROPERTY);
        System.setProperty(CLDeviceRanking.DEVICE_PROPERTY, "intel");
        try {
            List<CLDeviceRanking.Profile> profiles = Arrays.asList(arrowLakeIgpu(), rtx5080());
            // The slower device, because it was asked for by name.
            assertEquals(0, CLDeviceRanking.rank(profiles, neverProbed()));
        } finally {
            restore(CLDeviceRanking.DEVICE_PROPERTY, previous);
        }
    }

    @Test
    public void anExplicitChoiceThatMatchesNothingFallsThroughToRanking() {
        String previous = System.getProperty(CLDeviceRanking.DEVICE_PROPERTY);
        System.setProperty(CLDeviceRanking.DEVICE_PROPERTY, "no such device");
        try {
            List<CLDeviceRanking.Profile> profiles = Arrays.asList(arrowLakeIgpu(), rtx5080());
            // A typo must not become "run on whatever came first".
            assertEquals(1, CLDeviceRanking.rank(profiles, neverProbed()));
        } finally {
            restore(CLDeviceRanking.DEVICE_PROPERTY, previous);
        }
    }

    @Test
    public void theBenchmarkCanBeSwitchedOff() {
        String previous = System.getProperty(CLDeviceRanking.NO_BENCHMARK_PROPERTY);
        System.setProperty(CLDeviceRanking.NO_BENCHMARK_PROPERTY, "true");
        try {
            CLDeviceRanking.Profile intelDiscrete = new CLDeviceRanking.Profile(
                    "Intel(R) Arc(tm) A770", "Intel(R) Corporation", Boolean.FALSE, 64, 1900);
            List<CLDeviceRanking.Profile> profiles = Arrays.asList(intelDiscrete, rtx5080());
            // neverProbed() is the assertion: with the switch on, no dispatch.
            assertEquals(1, CLDeviceRanking.rank(profiles, neverProbed()));
        } finally {
            restore(CLDeviceRanking.NO_BENCHMARK_PROPERTY, previous);
        }
    }

    @Test
    public void emptyInputIsNotAnError() {
        assertNull(CLDeviceRanking.fastest(null));
        assertNull(CLDeviceRanking.fastest(Collections.<CLDevice>emptyList()));
        assertEquals(-1, CLDeviceRanking.rank(Collections.<CLDeviceRanking.Profile>emptyList(), null));
    }

    @Test
    public void theProbeMeasuresSomethingOnRealHardware() {
        // Guarded, not skipped-by-default: a host with no OpenCL device is a
        // legitimate place to run this suite, but a host that has one must prove
        // the probe works, because every multi-GPU decision rests on it.
        HWAcceleration hw = new HWAcceleration();
        List<CLDevice> devices = hw.getCLDevices();
        Assume.assumeFalse("no OpenCL device on this host", devices.isEmpty());

        double gflops = CLDeviceRanking.probe(devices.get(0));
        assertTrue("probe returned " + gflops + " GFLOP/s, which is not a rate",
                gflops > 0 && !Double.isInfinite(gflops));

        // fastest() must agree with itself across calls, or a run could end up
        // split across two devices.
        assertEquals(CLDeviceRanking.fastest(devices), CLDeviceRanking.fastest(devices));
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }
}
