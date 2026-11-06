package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The measured tuning tables of the certified exact-argmax lane: {@code auto} serves exactly the
 * measured pair (OpenCL Q8_0 on this machine's NVIDIA RTX 5080, and only above the head size whose
 * live decode loop measured the lane faster; every other measured pair lost, and nothing unmeasured
 * enters), {@code on} reaches every pair the lane's kernels cover, and the candidate cap reads the new
 * property with the draft-lane name as an alias.
 */
public class CertifiedArgmaxTuningTest {

    private String originalKmax;
    private String originalLegacyKmax;

    @Before
    public void rememberProperties() {
        originalKmax = System.getProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        originalLegacyKmax = System.getProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY);
    }

    @After
    public void restoreProperties() {
        restore(CertifiedArgmaxTuning.KMAX_PROPERTY, originalKmax);
        restore(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY, originalLegacyKmax);
    }

    private static void restore(String property, String value) {
        if (value == null) {
            System.clearProperty(property);
        } else {
            System.setProperty(property, value);
        }
    }

    /**
     * The measured {@code auto} table: OpenCL Q8_0 only, and only at heads large enough to amortize the
     * lane's fixed per-token cost (above it the 0.6B-class heads measured the live decode loop
     * neutral-to-slower even though their isolated readouts won).
     */
    @Test
    public void autoServesExactlyTheMeasuredPairs() {
        assertTrue(CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.Q8_0));
        assertFalse("the 0.6B F16 head won its readout but lost the live decode loop in both runs",
                CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.F16));
        assertFalse("BF16 measured a loss on the only faster backend",
                CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.BF16));
        for (BackendType backend : new BackendType[]{BackendType.CUDA, BackendType.VULKAN, BackendType.HIP,
                BackendType.METAL}) {
            for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0}) {
                assertFalse("no measured win on " + backend + " for " + type,
                        CertifiedArgmaxTuning.screenFaster(backend, type));
            }
        }
        assertFalse("unmeasured backends keep the exact readout",
                CertifiedArgmaxTuning.screenFaster(BackendType.SYCL, GGMLType.F16));
        assertEquals(Set.of(GGMLType.Q8_0), CertifiedArgmaxTuning.fasterTypes(BackendType.OPENCL));
        assertTrue(CertifiedArgmaxTuning.fasterTypes(BackendType.CUDA).isEmpty());
        for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0}) {
            assertFalse("no CPU type measured the screen faster", CertifiedArgmaxTuning.cpuFaster(type));
        }
        assertTrue(CertifiedArgmaxTuning.cpuFasterTypes().isEmpty());
    }

    /** The auto lane only serves the measured pairs from {@link CertifiedArgmaxTuning#HEAD_MIN_ELEMENTS} up. */
    @Test
    public void autoServesTheMeasuredPairsOnlyAboveTheSizeGate() {
        long gate = CertifiedArgmaxTuning.HEAD_MIN_ELEMENTS;
        long fourBHead = 151665L * 2560;
        long smallHead = 151669L * 1024;
        assertTrue(fourBHead + " element head sits above the gate", fourBHead > gate);
        assertTrue(smallHead + " element head sits below the gate", smallHead < gate);
        assertTrue(CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.Q8_0, fourBHead));
        assertTrue(CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.Q8_0, gate));
        assertFalse("the 0.6B-class Q8_0 head measured the live loop neutral-to-slower",
                CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.Q8_0, smallHead));
        assertFalse("the type gate still applies above the size gate",
                CertifiedArgmaxTuning.screenFaster(BackendType.OPENCL, GGMLType.F16, fourBHead));
        assertFalse("the backend gate still applies above the size gate",
                CertifiedArgmaxTuning.screenFaster(BackendType.CUDA, GGMLType.Q8_0, fourBHead));
    }

    /** The {@code on} table: the lane's kernels cover F16, BF16 and Q8_0 on every GPU backend it ships. */
    @Test
    public void onReachesEveryPairTheKernelsCover() {
        for (BackendType backend : new BackendType[]{BackendType.OPENCL, BackendType.CUDA, BackendType.VULKAN,
                BackendType.HIP, BackendType.METAL}) {
            for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0}) {
                assertTrue(CertifiedArgmaxTuning.screenCapable(backend, type));
            }
            assertEquals(Set.of(GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0),
                    CertifiedArgmaxTuning.capableTypes(backend));
            assertFalse("F32 has no screen", CertifiedArgmaxTuning.screenCapable(backend, GGMLType.F32));
        }
        assertFalse("SYCL has no certified kernels in this build",
                CertifiedArgmaxTuning.screenCapable(BackendType.SYCL, GGMLType.F16));
        assertTrue(CertifiedArgmaxTuning.screenCapable(BackendType.CUDA, GGMLType.Q8_0));
        assertTrue(CertifiedArgmaxTuning.cpuCapable(GGMLType.F16));
        assertTrue(CertifiedArgmaxTuning.cpuCapable(GGMLType.BF16));
        assertEquals(Set.of(GGMLType.F16, GGMLType.BF16), CertifiedArgmaxTuning.cpuCapableTypes());
        assertFalse("the CPU Q8_0 vec-dot quantizes the activations, which the bound does not cover",
                CertifiedArgmaxTuning.cpuCapable(GGMLType.Q8_0));
    }

    /** The mode gates the pairs through {@link CertifiedArgmaxMode#permitsScreen}. */
    @Test
    public void theModeGatesTheTunedPairs() {
        long bigHead = CertifiedArgmaxTuning.HEAD_MIN_ELEMENTS;
        assertTrue("auto serves the measured OpenCL pair above the size gate",
                CertifiedArgmaxMode.AUTO.permitsScreen(BackendType.OPENCL, GGMLType.Q8_0, bigHead));
        assertFalse("auto keeps the sub-gate heads on the exact readout",
                CertifiedArgmaxMode.AUTO.permitsScreen(BackendType.OPENCL, GGMLType.Q8_0, bigHead - 1));
        assertFalse("auto keeps the CUDA losses off the lane",
                CertifiedArgmaxMode.AUTO.permitsScreen(BackendType.CUDA, GGMLType.F16, bigHead));
        assertFalse(CertifiedArgmaxMode.OFF.permitsScreen(BackendType.OPENCL, GGMLType.Q8_0, bigHead));
        assertTrue("on forces wherever the kernels exist, any size",
                CertifiedArgmaxMode.ON.permitsScreen(BackendType.CUDA, GGMLType.F16, 1));
        assertFalse(CertifiedArgmaxMode.OFF.permitsCpuScreen(GGMLType.F16));
        assertTrue(CertifiedArgmaxMode.ON.permitsCpuScreen(GGMLType.BF16));
        assertFalse(CertifiedArgmaxMode.ON.permitsCpuScreen(GGMLType.Q8_0));
        assertFalse("auto keeps the CPU head exact", CertifiedArgmaxMode.AUTO.permitsCpuScreen(GGMLType.F16));
    }

    /** The tables are unmodifiable. */
    @Test
    public void theTablesAreUnmodifiable() {
        assertThrows(UnsupportedOperationException.class,
                () -> CertifiedArgmaxTuning.fasterTypes(BackendType.OPENCL).add(GGMLType.BF16));
        assertThrows(UnsupportedOperationException.class,
                () -> CertifiedArgmaxTuning.capableTypes(BackendType.CUDA).add(GGMLType.F32));
        assertThrows(UnsupportedOperationException.class,
                () -> CertifiedArgmaxTuning.cpuFasterTypes().add(GGMLType.F16));
        assertThrows(UnsupportedOperationException.class,
                () -> CertifiedArgmaxTuning.cpuCapableTypes().add(GGMLType.Q8_0));
    }

    /** The cap defaults to {@link CertifiedArgmaxTuning#DEFAULT_KMAX} when no property is set. */
    @Test
    public void theCapDefaultsToTheTunedValue() {
        System.clearProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        System.clearProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY);
        assertEquals(CertifiedArgmaxTuning.DEFAULT_KMAX, CertifiedArgmaxTuning.kmax());
    }

    /** The cap reads the new property, and the draft-lane property when the new one is unset. */
    @Test
    public void theCapReadsTheNewPropertyThenTheLegacyAlias() {
        System.clearProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY);
        System.setProperty(CertifiedArgmaxTuning.KMAX_PROPERTY, "64");
        assertEquals(64, CertifiedArgmaxTuning.kmax());
        System.clearProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        System.setProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY, "16");
        assertEquals(16, CertifiedArgmaxTuning.kmax());
        System.setProperty(CertifiedArgmaxTuning.KMAX_PROPERTY, "8");
        assertEquals("the new property wins when both are set", 8, CertifiedArgmaxTuning.kmax());
    }

    /** A bad cap value names the property that was read. */
    @Test
    public void aBadCapNamesThePropertyThatWasRead() {
        System.clearProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY);
        System.setProperty(CertifiedArgmaxTuning.KMAX_PROPERTY, "0");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, CertifiedArgmaxTuning::kmax);
        assertTrue(e.getMessage(), e.getMessage().contains(CertifiedArgmaxTuning.KMAX_PROPERTY));
        System.clearProperty(CertifiedArgmaxTuning.KMAX_PROPERTY);
        System.setProperty(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY, "zero");
        IllegalArgumentException legacy = assertThrows(IllegalArgumentException.class,
                CertifiedArgmaxTuning::kmax);
        assertTrue(legacy.getMessage(), legacy.getMessage().contains(CertifiedArgmaxTuning.KMAX_LEGACY_PROPERTY));
    }
}
