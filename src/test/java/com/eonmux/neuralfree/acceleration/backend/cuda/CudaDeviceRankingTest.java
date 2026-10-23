package com.eonmux.neuralfree.acceleration.backend.cuda;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The CUDA device-ranking policy over synthetic profiles: multiprocessors times clock, integrated
 * devices below discrete ones, unmeasured devices last, enumeration order on ties, and the forced
 * choice by name. No device is needed.
 */
public class CudaDeviceRankingTest {

    private String savedDevice;
    private String savedGlobal;

    @Before
    public void clearDeviceFilters() {
        savedDevice = System.getProperty(CudaDeviceRanking.DEVICE_PROPERTY);
        savedGlobal = System.getProperty(CudaDeviceRanking.GLOBAL_DEVICE_PROPERTY);
        System.clearProperty(CudaDeviceRanking.DEVICE_PROPERTY);
        System.clearProperty(CudaDeviceRanking.GLOBAL_DEVICE_PROPERTY);
    }

    @After
    public void restoreDeviceFilters() {
        restore(CudaDeviceRanking.DEVICE_PROPERTY, savedDevice);
        restore(CudaDeviceRanking.GLOBAL_DEVICE_PROPERTY, savedGlobal);
    }

    @Test
    public void emptyOrNullListHasNoWinner() {
        assertEquals(-1, CudaDeviceRanking.rank(Collections.<CudaDeviceRanking.Profile>emptyList()));
        assertEquals(-1, CudaDeviceRanking.rank(null));
        assertNull(CudaDeviceRanking.fastest(null));
    }

    @Test
    public void moreMultiprocessorsTimesClockWins() {
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                discrete("small", 20, 1_800_000),
                discrete("large", 84, 2_200_000),
                discrete("medium", 46, 2_500_000));

        assertEquals(1, CudaDeviceRanking.rank(p));
    }

    @Test
    public void integratedRanksBelowAnyDiscreteDevice() {
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                new CudaDeviceRanking.Profile("Orin", true, 1024, 3_000_000),
                discrete("tiny", 2, 1_000_000));

        assertEquals(1, CudaDeviceRanking.rank(p));
        assertTrue(CudaDeviceRanking.score(p.get(0)) < 0);
    }

    @Test
    public void unknownIntegratedFlagCountsAsDiscrete() {
        CudaDeviceRanking.Profile unknown = new CudaDeviceRanking.Profile("dGPU", null, 10, 1_000_000);

        assertEquals(10.0 * 1_000_000, CudaDeviceRanking.score(unknown), 0.0);
    }

    @Test
    public void unmeasuredDeviceRanksLastButStaysEligible() {
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                new CudaDeviceRanking.Profile("broken", false, CudaDeviceRanking.UNKNOWN, 2_000_000),
                new CudaDeviceRanking.Profile("integrated", true, 4, 1_000_000));

        assertEquals(1, CudaDeviceRanking.rank(p));
        assertEquals(Double.NEGATIVE_INFINITY, CudaDeviceRanking.score(p.get(0)), 0.0);
        assertEquals(0, CudaDeviceRanking.rank(Collections.singletonList(p.get(0))));
    }

    @Test
    public void tiesKeepEnumerationOrder() {
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                discrete("first", 30, 1_500_000),
                discrete("second", 30, 1_500_000));

        assertEquals(0, CudaDeviceRanking.rank(p));
    }

    @Test
    public void forcedNameWinsCaseInsensitively() {
        System.setProperty(CudaDeviceRanking.DEVICE_PROPERTY, "  rtx 3050 ");
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                discrete("NVIDIA GeForce RTX 5080", 84, 2_600_000),
                discrete("NVIDIA GeForce RTX 3050 Laptop", 16, 1_500_000));

        assertEquals(1, CudaDeviceRanking.rank(p));
    }

    @Test
    public void globalFilterAppliesWhenTheCudaFilterIsUnset() {
        System.setProperty(CudaDeviceRanking.GLOBAL_DEVICE_PROPERTY, "a100");
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                discrete("H100", 132, 1_980_000),
                discrete("A100-SXM4", 108, 1_410_000));

        assertEquals(1, CudaDeviceRanking.rank(p));
    }

    @Test
    public void filterMatchingNothingFallsBackToTheRanking() {
        System.setProperty(CudaDeviceRanking.DEVICE_PROPERTY, "no such device");
        List<CudaDeviceRanking.Profile> p = Arrays.asList(
                discrete("slow", 10, 1_000_000),
                discrete("fast", 80, 2_000_000));

        assertEquals(1, CudaDeviceRanking.rank(p));
    }

    @Test
    public void nullNameIsReplaced() {
        assertEquals("<unnamed>", new CudaDeviceRanking.Profile(null, false, 1, 1).name);
    }

    private static CudaDeviceRanking.Profile discrete(String name, int multiprocessors, int clockKhz) {
        return new CudaDeviceRanking.Profile(name, false, multiprocessors, clockKhz);
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
