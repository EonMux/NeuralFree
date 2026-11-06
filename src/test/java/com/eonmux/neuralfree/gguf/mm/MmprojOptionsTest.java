package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The device choice of the vision encoder against llama.cpp's {@code --mmproj-offload} / {@code --no-mmproj-offload}
 * and {@code --mmproj-device} rules (common/arg.cpp): offload on by default, {@code none} keeps the encoder on the
 * CPU, one device only, and the mmproj offload follows a CPU-only backend choice unless a device is named.
 */
public class MmprojOptionsTest {

    private static Properties props(String... kv) {
        Properties p = new Properties();
        for (int i = 0; i < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return p;
    }

    @Test
    public void offloadIsOnByDefault() {
        MmprojOptions o = MmprojOptions.fromProperties(props());
        assertTrue(o.offload());
        assertNull(o.device());
        assertEquals(EngineConfig.BackendMode.AUTO, o.backend());
        assertEquals(MmprojOptions.defaults(), o);
    }

    @Test
    public void noOffloadAndDeviceNoneRunOnTheCpu() {
        assertFalse(MmprojOptions.fromProperties(props(MmprojOptions.PROP_OFFLOAD, "false")).offload());
        assertFalse(MmprojOptions.fromProperties(props(MmprojOptions.PROP_DEVICE, "none")).offload());
        assertFalse(MmprojOptions.fromProperties(props(MmprojOptions.PROP_DEVICE, "NONE")).offload());
        assertFalse(MmprojOptions.onDevice(" none ").offload());
        assertFalse(MmprojOptions.fromProperties(props(MmprojOptions.PROP_OFFLOAD, "false",
                MmprojOptions.PROP_DEVICE, "1")).offload());
    }

    @Test
    public void aDeviceSelectorIsKept() {
        MmprojOptions o = MmprojOptions.fromProperties(props(MmprojOptions.PROP_DEVICE, " RTX "));
        assertTrue(o.offload());
        assertEquals("RTX", o.device());
        assertEquals("1", MmprojOptions.onDevice("1").device());
    }

    @Test
    public void theBackendModeIsFollowed() {
        MmprojOptions cuda = MmprojOptions.fromProperties(props(EngineConfig.PROP_BACKEND, "cuda"));
        assertTrue(cuda.offload());
        assertEquals(EngineConfig.BackendMode.CUDA, cuda.backend());
        // a CPU-only engine keeps the encoder on the CPU unless a device is named
        assertFalse(MmprojOptions.fromProperties(props(EngineConfig.PROP_BACKEND, "cpu")).offload());
        MmprojOptions named = MmprojOptions.fromProperties(props(EngineConfig.PROP_BACKEND, "cpu",
                MmprojOptions.PROP_DEVICE, "0"));
        assertTrue(named.offload());
        assertEquals(EngineConfig.BackendMode.AUTO, named.backend());
        assertFalse(MmprojOptions.defaults().withBackend(EngineConfig.BackendMode.CPU).offload());
        assertNull(MmprojOptions.onDevice("0").withOffload(false).device());
    }

    @Test
    public void malformedValuesAreRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_OFFLOAD, "yes")));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_OFFLOAD));
        e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_DEVICE, "0,1")));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_DEVICE));
        assertTrue(e.getMessage().contains("only one device"));
        assertThrows(IllegalArgumentException.class, () -> MmprojOptions.onDevice("  "));
        assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(EngineConfig.PROP_BACKEND, "tpu")));
    }

    @Test
    public void theEncoderParallelismModeIsWeightsUnlessDataIsAsked() {
        assertEquals(MmEncoderTpMode.WEIGHTS, MmprojOptions.defaults().tpMode());
        assertEquals(MmEncoderTpMode.WEIGHTS, MmprojOptions.cpu().tpMode());
        assertEquals(MmEncoderTpMode.DATA,
                MmprojOptions.fromProperties(props(MmprojOptions.PROP_TP_MODE, "data")).tpMode());
        assertEquals("data", MmEncoderTpMode.parse(" DATA ").label());
        assertEquals(MmEncoderTpMode.DATA, MmEncoderTpMode.parse("data"));
        assertTrue(MmprojOptions.defaults().devices().isEmpty());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_TP_MODE, "tensor")));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_TP_MODE));
    }

    @Test
    public void aRankGroupIsParsedForDataParallelEncoding() {
        MmprojOptions o = MmprojOptions.fromProperties(props(MmprojOptions.PROP_TP_MODE, "data",
                MmprojOptions.PROP_DEVICES, " 0 , RTX "));
        assertEquals(Arrays.asList("0", "RTX"), o.devices());
        assertTrue(o.offload());
        assertEquals(MmEncoderTpMode.DATA, o.tpMode());
        assertEquals(Collections.singletonList("1"), MmprojOptions.defaults().withDevices(Arrays.asList("1"))
                .devices());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_TP_MODE, "data",
                        MmprojOptions.PROP_DEVICES, "0,,1")));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_DEVICES));
        e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.defaults().withDevices(Arrays.asList("0", "0")));
        assertTrue(e.getMessage().contains("repeats"));
        assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.defaults().withDevices(Arrays.asList("0,1")));
    }

    @Test
    public void weightsModeRefusesARankGroupAndACpuEncoderIsOneRank() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_DEVICES, "0,1")));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_DEVICES));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_TP_MODE));
        MmprojOptions data = MmprojOptions.defaults().withTpMode(MmEncoderTpMode.DATA)
                .withDevices(Arrays.asList("0", "1"));
        assertThrows(IllegalArgumentException.class, () -> data.withTpMode(MmEncoderTpMode.WEIGHTS));
        e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_OFFLOAD, "false",
                        MmprojOptions.PROP_TP_MODE, "data", MmprojOptions.PROP_DEVICES, "0,1")));
        assertTrue(e.getMessage().contains("CPU"));
    }

    @Test
    public void aDeviceAndARankGroupAreNeverGivenTogether() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MmprojOptions.fromProperties(props(MmprojOptions.PROP_DEVICE, "0",
                        MmprojOptions.PROP_DEVICES, "1")));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_DEVICE));
        assertTrue(e.getMessage().contains(MmprojOptions.PROP_DEVICES));
        MmprojOptions data = MmprojOptions.defaults().withTpMode(MmEncoderTpMode.DATA)
                .withDevices(Arrays.asList("0", "1"));
        MmprojOptions cpu = data.withBackend(EngineConfig.BackendMode.CPU);
        assertFalse(cpu.offload());
        assertTrue(cpu.devices().isEmpty());
        assertTrue(data.withOffload(false).devices().isEmpty());
        assertEquals(data, MmprojOptions.defaults().withTpMode(MmEncoderTpMode.DATA)
                .withDevices(Arrays.asList("0", "1")));
    }
}
