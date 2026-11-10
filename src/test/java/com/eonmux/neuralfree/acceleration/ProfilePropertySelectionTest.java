package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A malformed {@code -Dneuralfree.gpu.profile} fails backend selection by naming the property, in AUTO and in an
 * explicit mode alike, instead of making every GPU backend look unusable: AUTO skips a backend whose creation fails,
 * and every GPU backend reads the timing mode while it is created.
 */
public class ProfilePropertySelectionTest {

    private String previous;

    @Before
    public void setUp() {
        previous = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        System.setProperty(KernelTiming.PROFILE_PROPERTY, "yes");
    }

    @After
    public void tearDown() {
        if (previous == null) {
            System.clearProperty(KernelTiming.PROFILE_PROPERTY);
        } else {
            System.setProperty(KernelTiming.PROFILE_PROPERTY, previous);
        }
    }

    @Test
    public void aMalformedProfileModeFailsTheSelectionByName() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.OPENCL, BackendType.VULKAN));
        try {
            for (ACCELERATION_MODE mode : new ACCELERATION_MODE[]{ACCELERATION_MODE.AUTO, ACCELERATION_MODE.VULKAN,
                ACCELERATION_MODE.OPENCL}) {
                IllegalArgumentException e = assertThrows(mode.name(), IllegalArgumentException.class,
                        () -> hw.createBackend(mode, BackendRequirement.ANY));
                assertTrue(mode + ": " + e.getMessage(), e.getMessage().contains("-D" + KernelTiming.PROFILE_PROPERTY
                        + " must be true, false or step, got 'yes'"));
            }
        } finally {
            hw.shutdown();
        }
    }
}
