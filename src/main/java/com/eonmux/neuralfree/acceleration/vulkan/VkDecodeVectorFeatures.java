package com.eonmux.neuralfree.acceleration.vulkan;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VkBaseOutStructure;

/**
 * {@code VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV} of {@code VK_NV_cooperative_matrix_decode_vector},
 * which the bound Vulkan headers do not declare: {@code {VkStructureType sType; void* pNext; VkBool32
 * cooperativeMatrixDecodeVector;}}, laid out as the C struct on a 64-bit target (the pointer 8-byte aligned, the
 * struct padded to a multiple of 8). The feature lets cooperative-matrix tensor loads call a decode function that
 * returns a vector of consecutive elements (SPIR-V {@code CooperativeMatrixDecodeVectorNV}, GLSL
 * {@code GL_NV_cooperative_matrix_decode_vector}). The struct is allocated as a {@link VkBaseOutStructure} view, so it
 * joins a feature {@code pNext} chain like the bound structs.
 */
public final class VkDecodeVectorFeatures {

    /** The device extension name. */
    public static final String EXTENSION_NAME = "VK_NV_cooperative_matrix_decode_vector";
    /** {@code VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV}. */
    public static final int STRUCTURE_TYPE = 1000689000;
    /** Byte size of the struct. */
    static final int SIZEOF = 24;
    /** Byte offset of {@code cooperativeMatrixDecodeVector}. */
    static final int FEATURE_OFFSET = 16;

    private VkDecodeVectorFeatures() {
    }

    /**
     * A zeroed struct on {@code stack} with its {@code sType} set and {@code cooperativeMatrixDecodeVector} set to
     * {@code enable}: a query struct (false) or the enabling struct of device creation (true).
     */
    public static VkBaseOutStructure calloc(MemoryStack stack, boolean enable) {
        long address = stack.ncalloc(Long.BYTES, 1, SIZEOF);
        MemoryUtil.memPutInt(address, STRUCTURE_TYPE);
        MemoryUtil.memPutInt(address + FEATURE_OFFSET, enable ? 1 : 0);
        return VkBaseOutStructure.create(address);
    }

    /** {@code cooperativeMatrixDecodeVector} of a struct from {@link #calloc} after a feature query. */
    public static boolean decodeVector(VkBaseOutStructure struct) {
        return MemoryUtil.memGetInt(struct.address() + FEATURE_OFFSET) != 0;
    }
}
