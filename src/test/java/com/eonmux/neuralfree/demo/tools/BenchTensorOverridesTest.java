package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.TensorOverride;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The llama.cpp buffer types of {@link BenchTensorOverrides} map onto the engine's placement targets: the device
 * buffers of every {@link TensorOverride#deviceTargets()} backend, plus llama.cpp's {@code ROCm<n>} (the HIP
 * build's name for CUDA buffers) and {@code MTL<n>} / {@code Metal<n>} (Metal), each with an optional device
 * index; a buffer type with no engine counterpart is refused by name.
 */
public class BenchTensorOverridesTest {

    private static TensorOverride first(String value) throws BenchCommand.UsageException {
        List<BenchTensorOverrides> members = BenchTensorOverrides.parse("-ot", value);
        assertEquals("one sweep member", 1, members.size());
        assertEquals("one override", 1, members.get(0).overrides().size());
        return members.get(0).overrides().get(0);
    }

    @Test
    public void rocmBuffersMapToHip() throws Exception {
        for (String bufferType : new String[] {"ROCm0", "rocm1", "ROCm"}) {
            TensorOverride o = first("exps=" + bufferType);
            assertFalse(bufferType + " is a device rule", o.host());
            assertEquals(bufferType + " maps onto hip", "hip", o.target());
        }
    }

    @Test
    public void metalAndMtlBuffersMapToMetal() throws Exception {
        for (String bufferType : new String[] {"Metal0", "metal", "MTL1", "mtl0"}) {
            TensorOverride o = first("exps=" + bufferType);
            assertFalse(bufferType + " is a device rule", o.host());
            assertEquals(bufferType + " maps onto metal", "metal", o.target());
        }
    }

    @Test
    public void theBackendBuffersKeepTheirTargetNames() throws Exception {
        assertEquals("cuda", first("exps=CUDA0").target());
        assertEquals("vulkan", first("exps=Vulkan2").target());
        assertEquals("opencl", first("exps=OpenCL0").target());
        assertEquals("hip", first("exps=HIP0").target());
        assertTrue("CPU stays the host", first("exps=CPU").host());
    }

    @Test
    public void theAcceleratorApiBuffersMapOntoTheirTargets() throws Exception {
        assertEquals("sycl", first("exps=SYCL0").target());
        assertEquals("xpu", first("exps=XPU1").target());
        assertEquals("cann", first("exps=CANN0").target());
        assertEquals("musa", first("exps=MUSA0").target());
        assertEquals("webgpu", first("exps=WebGPU").target());
        assertEquals("zdnn", first("exps=ZDNN3").target());
        assertEquals("zendnn", first("exps=ZenDNN").target());
    }

    @Test
    public void theMemberPrintsTheReferenceBufferNames() throws Exception {
        List<BenchTensorOverrides> members = BenchTensorOverrides.parse("-ot", "exps=ROCm0;blk\\.0\\..*=MTL1");
        assertEquals("exps=ROCm0;blk\\.0\\..*=MTL1", members.get(0).text());
        assertEquals("the engine flag carries the targets", "exps=hip,blk\\.0\\..*=metal",
                members.get(0).engineArg());
    }

    @Test
    public void aBufferTypeWithoutAnEngineCounterpartIsRefusedByName() {
        for (String bufferType : new String[] {"CUDA_Host", "MTL0_Private"}) {
            try {
                BenchTensorOverrides.parse("-ot", "exps=" + bufferType);
                throw new AssertionError("accepted buffer type " + bufferType);
            } catch (BenchCommand.UsageException expected) {
                assertTrue("the refusal names the buffer type: " + expected.getMessage(),
                        expected.getMessage().contains(bufferType));
                assertTrue("the refusal suggests the accelerator APIs: " + expected.getMessage(),
                        expected.getMessage().contains("SYCL"));
            }
        }
    }
}
