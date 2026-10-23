package com.eonmux.neuralfree.acceleration.cuda;

import java.util.*;

import static jcuda.driver.CUdevice_attribute.*;


/**
 * Helpers for describing CUDA devices: attribute names and descriptions, and
 * decoding of the driver's NUL-terminated strings.
 *
 * <p>Derived from the JCudaDeviceQuery sample of JCuda (MIT License, Copyright (c) 2008-2016
 * Marco Hutter); see THIRD-PARTY-NOTICES.md.</p>
 */
public class CUDA_Utils {
    /**
     * Returns a short description of the given CUdevice_attribute constant
     *
     * @param attribute The CUdevice_attribute constant
     * @return A short description of the given constant
     */
    protected static String getAttributeDescription(int attribute)
    {
        switch (attribute)
        {
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
                return "Maximum number of threads per block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
                return "Maximum x-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
                return "Maximum y-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
                return "Maximum z-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
                return "Maximum x-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
                return "Maximum y-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
                return "Maximum z-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
                return "Maximum shared memory per thread block in bytes";
            case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
                return "Total constant memory on the device in bytes";
            case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
                return "Warp size in threads";
            case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
                return "Maximum pitch in bytes allowed for memory copies";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
                return "Maximum number of 32-bit registers per thread block";
            case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
                return "Clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
                return "Alignment requirement";
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
                return "Number of multiprocessors on the device";
            case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
                return "Whether there is a run time limit on kernels";
            case CU_DEVICE_ATTRIBUTE_INTEGRATED:
                return "Device is integrated with host memory";
            case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
                return "Device can map host memory into CUDA address space";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
                return "Compute mode";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
                return "Maximum 1D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
                return "Maximum 2D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
                return "Maximum 2D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
                return "Maximum 3D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
                return "Maximum 3D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:
                return "Maximum 3D texture depth";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH:
                return "Maximum 2D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:
                return "Maximum 2D layered texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:
                return "Maximum layers in a 2D layered texture";
            case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
                return "Alignment requirement for surfaces";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
                return "Device can execute multiple kernels concurrently";
            case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
                return "Device has ECC support enabled";
            case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
                return "PCI bus ID of the device";
            case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
                return "PCI device ID of the device";
            case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
                return "Device is using TCC driver model";
            case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
                return "Peak memory clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
                return "Global memory bus width in bits";
            case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
                return "Size of L2 cache in bytes";
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
                return "Maximum resident threads per multiprocessor";
            case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
                return "Number of asynchronous engines";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
                return "Device shares a unified address space with the host";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH:
                return "Maximum 1D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:
                return "Maximum layers in a 1D layered texture";
            case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
                return "PCI domain ID of the device";
        }
        return "(UNKNOWN ATTRIBUTE)";
    }

    /**
     * Returns a list of all CUdevice_attribute constants
     *
     * @return A list of all CUdevice_attribute constants
     */
    protected static List<Integer> getAttributes()
    {
        List<Integer> list = new ArrayList<Integer>();
        list.addAll(Arrays.asList(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
        		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
        		CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
        		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
        		CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        		CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        		CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
        		CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        		CU_DEVICE_ATTRIBUTE_MAX_PITCH,
        		CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
        		CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
        		CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        		CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
        		CU_DEVICE_ATTRIBUTE_INTEGRATED,
        		CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
        		CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS,
        		CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
        		CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
        		CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
        		CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
        		CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
        		CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
        		CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
        		CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
        		CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        		CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        		CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
        		CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH,
        		CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS,
        		CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID));

        return list;
    }

    /**
     * Creates a String from a zero-terminated string in a byte array
     *
     * @param bytes
     *            The byte array
     * @return The String
     */
    protected static String createString(byte bytes[])
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bytes.length; i++)
        {
            char c = (char)bytes[i];
            if (c == 0)
            {
                break;
            }
            sb.append(c);
        }
        return sb.toString();
    }
}
