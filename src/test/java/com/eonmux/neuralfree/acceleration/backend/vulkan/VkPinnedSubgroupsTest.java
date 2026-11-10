package com.eonmux.neuralfree.acceleration.backend.vulkan;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT;
import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_FRAGMENT_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_BALLOT_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_BASIC_BIT;
import static org.lwjgl.vulkan.VK11.VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

/**
 * The pinned-subgroup rules of the Vulkan backend ({@link VkPinnedSubgroups}) over device facts built without a
 * device: the refusal {@link VkProgramCache} applies before it loads a pinned module (operations, size control, full
 * subgroups, required-size stages, the subgroup range, the width itself) and the local-size rule
 * {@link VkKernelHandle} applies before it selects a pinned pipeline.
 */
public class VkPinnedSubgroupsTest {

    private static final int ALL_OPERATIONS = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT
            | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;

    /** The Arrow-Lake iGPU under Mesa ANV: 8 to 32 lanes, size control on compute, 64 subgroups per work-group. */
    private static VkPinnedSubgroups anv() {
        return new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, ALL_OPERATIONS, true, true,
                VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 64);
    }

    private static void assertRefuses(String what, VkPinnedSubgroups device, String expectedPart) {
        String reason = device.refusal(16);
        assertNotNull(what + " was accepted", reason);
        assertTrue(what + ": " + reason, reason.contains(expectedPart));
    }

    @Test
    public void aDeviceWithEveryFactMayPinAnyWidthInItsRange() {
        for (int size : new int[]{8, 16, 32}) {
            assertNull("width " + size, anv().refusal(size));
        }
        VkPinnedSubgroups exact = new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, ALL_OPERATIONS, true, true,
                VK_SHADER_STAGE_COMPUTE_BIT, 16, 16, 64);
        assertNull(exact.refusal(16));
    }

    @Test
    public void aWidthOutsideTheRangeIsRefused() {
        assertNotNull(anv().refusal(4));
        assertNotNull(anv().refusal(64));
        VkPinnedSubgroups nvidia = new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, ALL_OPERATIONS, true, true,
                VK_SHADER_STAGE_COMPUTE_BIT, 32, 32, 32);
        assertRefuses("a 32-lane-only device", nvidia, "32 to 32 lanes");
    }

    @Test
    public void aWidthThatIsNotAPowerOfTwoIsRefused() {
        for (int size : new int[]{0, -16, 12, 24}) {
            String reason = anv().refusal(size);
            assertNotNull("width " + size, reason);
            assertTrue(reason, reason.contains("power of two"));
        }
    }

    @Test
    public void missingOperationsAreRefused() {
        int noArithmetic = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
        assertRefuses("no arithmetic", new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, noArithmetic, true, true,
                VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 64), "subgroup operations");
        int noBallot = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        assertRefuses("no ballot", new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, noBallot, true, true,
                VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 64), "subgroup operations");
        assertRefuses("no compute subgroups", new VkPinnedSubgroups(VK_SHADER_STAGE_FRAGMENT_BIT, ALL_OPERATIONS,
                true, true, VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 64), "subgroup operations");
    }

    @Test
    public void sizeControlOffIsRefused() {
        assertRefuses("size control off", new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, ALL_OPERATIONS, false,
                true, VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 64), "subgroupSizeControl");
    }

    @Test
    public void fullSubgroupsOffIsRefused() {
        assertRefuses("full subgroups off", new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, ALL_OPERATIONS, true,
                false, VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 64), "computeFullSubgroups");
    }

    @Test
    public void requiredSizeStagesWithoutComputeAreRefused() {
        assertRefuses("no compute required-size stage", new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT,
                ALL_OPERATIONS, true, true, VK_SHADER_STAGE_FRAGMENT_BIT, 8, 32, 64), "requiredSubgroupSizeStages");
    }

    @Test
    public void aLocalSizeOfWholeSubgroupsWithinTheLimitIsAccepted() {
        assertNull(anv().localSizeRefusal(16, new int[]{64, 1, 1}));
        assertNull(anv().localSizeRefusal(16, new int[]{128, 1, 1}));
        assertNull("16 x 64 invocations is the limit", anv().localSizeRefusal(16, new int[]{1024, 1, 1}));
        assertNull(anv().localSizeRefusal(16, new int[]{16, 4, 2}));
    }

    @Test
    public void aLocalSizeThatSplitsASubgroupIsRefused() {
        assertNotNull(anv().localSizeRefusal(16, new int[]{8, 1, 1}));
        assertNotNull(anv().localSizeRefusal(16, new int[]{24, 1, 1}));
        assertNotNull("local x alone must hold whole subgroups", anv().localSizeRefusal(16, new int[]{8, 2, 1}));
    }

    @Test
    public void aLocalSizeAboveTheSubgroupLimitIsRefused() {
        VkPinnedSubgroups fewSubgroups = new VkPinnedSubgroups(VK_SHADER_STAGE_COMPUTE_BIT, ALL_OPERATIONS, true,
                true, VK_SHADER_STAGE_COMPUTE_BIT, 8, 32, 4);
        assertNull(fewSubgroups.localSizeRefusal(16, new int[]{64, 1, 1}));
        String reason = fewSubgroups.localSizeRefusal(16, new int[]{128, 1, 1});
        assertNotNull(reason);
        assertTrue(reason, reason.contains("at most 64 invocations"));
        assertNotNull(fewSubgroups.localSizeRefusal(16, new int[]{64, 2, 1}));
    }

    @Test
    public void anInvalidLocalSizeOrWidthIsAnArgumentError() {
        for (int[] bad : new int[][]{null, {64, 1}, {0, 1, 1}, {64, -1, 1}}) {
            try {
                anv().localSizeRefusal(16, bad);
                fail("accepted local size " + Arrays.toString(bad));
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("local size"));
            }
        }
        try {
            anv().localSizeRefusal(0, new int[]{64, 1, 1});
            fail("accepted a pinned width of 0");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("positive"));
        }
    }
}
