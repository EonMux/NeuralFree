package com.eonmux.neuralfree.acceleration.backend.ocl;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * The OpenCL backend reports subgroups only when its OpenCL C 1.2 programs can enable them, and
 * matches extension names exactly.
 */
public class OclSubgroupCapabilityTest {

    @Test
    public void intelSubgroupsEnableSubgroupPrograms() {
        assertTrue(OclComputeBackend.programsCanUseSubgroups(
                "cl_khr_fp16 cl_khr_subgroups cl_intel_subgroups cl_intel_required_subgroup_size"));
        assertTrue(OclComputeBackend.programsCanUseSubgroups("cl_intel_subgroups"));
    }

    @Test
    public void khrSubgroupsAloneDoNotEnableSubgroupPrograms() {
        assertFalse("cl_khr_subgroups needs OpenCL C 2.0; CL1.2 programs fall back to local memory",
                OclComputeBackend.programsCanUseSubgroups("cl_khr_fp64 cl_khr_subgroups cl_amd_device_attribute_query"));
        assertFalse(OclComputeBackend.programsCanUseSubgroups(""));
        assertFalse(OclComputeBackend.programsCanUseSubgroups(null));
    }

    @Test
    public void extensionNamesMatchExactly() {
        assertFalse(OclComputeBackend.hasExtension("cl_intel_subgroups_short cl_intel_subgroups_char",
                "cl_intel_subgroups"));
        assertTrue(OclComputeBackend.hasExtension("  cl_khr_fp16\tcl_khr_fp64 ", "cl_khr_fp64"));
        assertFalse(OclComputeBackend.hasExtension("cl_khr_fp64", ""));
    }
}
