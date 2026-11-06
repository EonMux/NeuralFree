package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.backend.WorkSizes;

import java.util.Arrays;

/**
 * Translates a kernel dispatch's work sizes into Vulkan work-group counts within the device's
 * {@code maxComputeWorkGroupCount}: the count of a one-dimensional dispatch above the x limit is
 * folded into y and the kernel's {@code nf_wg_id()} recovers the linear group id, while
 * multi-dimensional counts are never folded.
 */
final class VkDispatchLimits {

    private VkDispatchLimits() {
    }

    /**
     * Group counts {@code ceil(global / local3)} padded to three dimensions.
     *
     * @throws IllegalArgumentException when a count exceeds the device limits after folding
     */
    static long[] groupCounts(String kernel, long[] global, int[] local3, long[] maxGroups) {
        long[] local = new long[global.length];
        for (int d = 0; d < global.length; d++) {
            local[d] = local3[d];
        }
        long[] groups = WorkSizes.pad3(WorkSizes.groupCounts(global, local), 1);
        if (groups[0] > maxGroups[0]) {
            if (global.length != 1) {
                throw new IllegalArgumentException(kernel + ": " + groups[0] + " work-groups in x exceed "
                        + "maxComputeWorkGroupCount[0] = " + maxGroups[0] + "; only one-dimensional dispatches are "
                        + "folded (global " + Arrays.toString(global) + ", groups " + Arrays.toString(groups) + ")");
            }
            long folded = (groups[0] - 1) / maxGroups[0] + 1;
            groups[0] = maxGroups[0];
            groups[1] = folded;
        }
        for (int d = 0; d < 3; d++) {
            if (groups[d] > maxGroups[d]) {
                throw new IllegalArgumentException(kernel + ": work-group counts " + Arrays.toString(groups)
                        + " exceed maxComputeWorkGroupCount " + Arrays.toString(maxGroups));
            }
        }
        return groups;
    }
}
