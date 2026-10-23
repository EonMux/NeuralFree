package com.eonmux.neuralfree.acceleration;

import org.jocl.*;

import static org.jocl.CL.*;

/**
 * Standalone OpenCL device probe run as a subprocess.
 * <p>
 * Safely detects broken OpenCL drivers (e.g., NVIDIA ICD with corrupted
 * dispatch tables) that cause SIGSEGV when {@code clGetDeviceInfo} is called.
 * <p>
 * When run in a subprocess, a native crash only kills the probe process,
 * not the main JVM. The parent reads the exit code to determine which
 * OpenCL platforms are safe to use.
 * <p>
 * Exit codes:
 * <ul>
 *   <li>0 - All probed devices are accessible</li>
 *   <li>1 - Catchable error (classpath issue, JOCL not found, etc.)</li>
 *   <li>128+ - Signal kill (SIGSEGV=139) indicating a broken driver</li>
 * </ul>
 *
 * Usage: {@code java OpenCLProbe [platformIndex]}
 */
public final class OpenCLProbe {

    public static void main(String[] args) {
        int targetPlatform = -1;
        if (args.length > 0) {
            try {
                targetPlatform = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.exit(1);
                return;
            }
        }

        try {
            CL.setExceptionsEnabled(false);

            int[] numPlatforms = new int[1];
            if (clGetPlatformIDs(0, null, numPlatforms) != CL_SUCCESS) {
                // No OpenCL runtime at all — not a driver crash issue
                System.exit(0);
                return;
            }
            if (numPlatforms[0] <= 0) {
                System.exit(0);
                return;
            }

            cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
            if (clGetPlatformIDs(platforms.length, platforms, null) != CL_SUCCESS) {
                System.exit(1);
                return;
            }

            for (int i = 0; i < platforms.length; i++) {
                if (targetPlatform >= 0 && i != targetPlatform) continue;

                // Query platform name first (handled by ICD loader, unlikely to crash)
                long[] platNameSize = new long[1];
                int err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, null, platNameSize);
                if (err == CL_SUCCESS && platNameSize[0] > 0 && platNameSize[0] <= 4096) {
                    byte[] nameBuf = new byte[(int) platNameSize[0]];
                    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameBuf.length,
                            Pointer.to(nameBuf), null);
                    System.out.println("PLATFORM:" + i + ":" + new String(nameBuf).trim());
                }

                int[] numDevices = new int[1];
                err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, null, numDevices);
                if (err != CL_SUCCESS || numDevices[0] <= 0) continue;

                cl_device_id[] devices = new cl_device_id[numDevices[0]];
                err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numDevices[0], devices, null);
                if (err != CL_SUCCESS) continue;

                for (int j = 0; j < devices.length; j++) {
                    if (devices[j] == null) continue;

                    // This clGetDeviceInfo call is the one that can SIGSEGV
                    // on broken drivers with corrupted ICD dispatch tables.
                    // That's the whole reason this class runs in a subprocess.
                    long[] size = new long[1];
                    err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, null, size);
                    if (err != CL_SUCCESS || size[0] <= 0 || size[0] > 4096) {
                        System.out.println("DEVICE:" + i + ":" + j + ":QUERY_FAILED:" + err);
                        continue;
                    }

                    byte[] devNameBuf = new byte[(int) size[0]];
                    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, devNameBuf.length,
                            Pointer.to(devNameBuf), null);
                    System.out.println("DEVICE:" + i + ":" + j + ":OK:" +
                            new String(devNameBuf).trim());
                }
            }

            System.exit(0);
        } catch (UnsatisfiedLinkError e) {
            // JOCL native library not available in this subprocess
            System.err.println("JOCL native library not found: " + e.getMessage());
            System.exit(1);
        } catch (Throwable t) {
            System.err.println("Probe error: " + t.getMessage());
            System.exit(1);
        }
    }
}
