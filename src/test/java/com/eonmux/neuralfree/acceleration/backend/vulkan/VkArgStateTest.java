package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Argument validation of {@link VkArgState} against the build-time {@link KernelSignature} when the
 * optimiser stripped a binding from the module: the signature, not the module, decides how Java
 * arguments map to bindings and push-constant offsets, with the messages {@link KernelSignature}
 * defines for every backend. Layout checks run before any buffer is touched, so no device is needed.
 */
public class VkArgStateTest {

    private static final int[] NONE = {};

    /** Module with bindings 0 and 2 (binding 1 stripped) and push members int at 0, float at 4. */
    private static SpirvModuleInfo strippedModule() {
        SpirvTestModule m = new SpirvTestModule("main");
        int f32 = m.typeFloat(32);
        int i32 = m.typeInt(32, true);
        m.storageBuffer(0, 0, f32, 4, new int[]{SpirvTestModule.DECO_NON_WRITABLE}, NONE);
        m.storageBuffer(0, 2, f32, 4, NONE, NONE);
        m.pushBlock(i32, f32);
        m.localSize(64, 1, 1);
        return SpirvReflector.reflect(m.bytes());
    }

    private static KernelSignature threeBuffersIntFloat() {
        return KernelSignature.parse("3", "110", "i,f");
    }

    @Test
    public void strippedBindingStillCountsItsBufferArgument() {
        VkArgState state = new VkArgState("K", strippedModule(), threeBuffersIntFloat());
        state.setAll(new Object[]{buffer(), 7, buffer(), 1.5f, buffer()});

        assertTrue(state.isComplete());
        assertEquals(8, state.pushSize());
    }

    @Test
    public void missingArgumentLeavesTheStateIncomplete() {
        VkArgState state = new VkArgState("K", strippedModule(), threeBuffersIntFloat());
        state.setAll(new Object[]{buffer(), 7, buffer(), 1.5f});

        assertFalse(state.isComplete());
    }

    @Test
    public void scalarOfTheWrongKindIsNamed() {
        VkArgState state = new VkArgState("K", strippedModule(), threeBuffersIntFloat());
        state.setAll(new Object[]{buffer(), 1.5f, buffer(), 7, buffer()});
        try {
            state.materialize();
            fail("a float where the signature declares int32 must be rejected");
        } catch (IllegalArgumentException e) {
            assertEquals("K: arg 1 (Float) does not match scalar 0 (int32)", e.getMessage());
        }
    }

    @Test
    public void extraScalarIsNamed() {
        VkArgState state = new VkArgState("K", strippedModule(), threeBuffersIntFloat());
        state.setAll(new Object[]{buffer(), 7, buffer(), 1.5f, 3, buffer()});
        try {
            state.materialize();
            fail("a third scalar must be rejected");
        } catch (IllegalArgumentException e) {
            assertEquals("K: arg 4 (Integer) exceeds the 2 scalar arguments of the signature", e.getMessage());
        }
    }

    @Test
    public void missingBufferIsReportedWithBothCounts() {
        VkArgState state = new VkArgState("K", strippedModule(), threeBuffersIntFloat());
        state.setAll(new Object[]{buffer(), 7, buffer(), 1.5f});
        try {
            state.materialize();
            fail("two buffers for a three-buffer signature must be rejected");
        } catch (IllegalArgumentException e) {
            assertEquals("K: got 2 buffers and 2 scalars, signature expects 3 buffers and 2 scalars",
                    e.getMessage());
        }
    }

    @Test
    public void eightByteScalarIsRejected() {
        VkArgState state = new VkArgState("K", strippedModule(), threeBuffersIntFloat());
        state.setAll(new Object[]{buffer(), 7L, buffer(), 1.5f, buffer()});
        try {
            state.materialize();
            fail("a Long where the signature declares int32 must be rejected");
        } catch (IllegalArgumentException e) {
            assertEquals("K: arg 1 (Long) does not match scalar 0 (int32)", e.getMessage());
        }
    }

    @Test
    public void moduleBindingBeyondTheSignatureIsRefused() {
        try {
            new VkArgState("K", strippedModule(), KernelSignature.parse("2", "10", "i,f"));
            fail("binding 2 does not exist in a two-buffer signature");
        } catch (UnsupportedKernelException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("different builds"));
        }
    }

    @Test
    public void pushMemberOfAnotherKindThanTheSignatureIsRefused() {
        try {
            new VkArgState("K", strippedModule(), KernelSignature.parse("3", "110", "f,f"));
            fail("an int push member against a float signature scalar must be refused");
        } catch (UnsupportedKernelException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("push-constant member 0"));
        }
    }

    @Test
    public void legacyModuleWithoutSignatureUsesTheReflectedMembers() {
        VkArgState state = new VkArgState("K", strippedModule(), null);
        state.setAll(new Object[]{buffer(), buffer(), buffer(), 7, 1.5f});

        assertTrue(state.isComplete());
        state.setAll(new Object[]{buffer(), buffer(), buffer(), 1.5f, 7});
        try {
            state.materialize();
            fail("a float against an int push member must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("does not match push-constant member 0"));
        }
    }

    private static GpuBuffer buffer() {
        return new GpuBuffer() {
            @Override
            public long sizeInBytes() {
                return 4;
            }

            @Override
            public long bufferId() {
                return 1;
            }

            @Override
            public void close() {
            }
        };
    }
}
