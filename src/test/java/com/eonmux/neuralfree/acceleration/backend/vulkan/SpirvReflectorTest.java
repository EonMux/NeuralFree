package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link SpirvReflector} on hand-assembled compute modules ({@link SpirvTestModule}): aliased typed
 * views of one binding, bindings the optimiser stripped, push-constant members, specialisation
 * constants and the local-size forms, and the resources the backend refuses.
 */
public class SpirvReflectorTest {

    private static final int[] NONE = {};

    @Test
    public void aliasedViewsOfOneBindingMergeTheirAccess() {
        SpirvTestModule m = new SpirvTestModule("main").capability(SpirvModuleInfo.CAP_STORAGE_BUFFER_8BIT_ACCESS);
        int f32 = m.typeFloat(32);
        int u32 = m.typeInt(32, false);
        int u8 = m.typeInt(8, false);
        // binding 0: a float view and a byte view, both read-only (variable and member forms)
        m.storageBuffer(0, 0, f32, 4, new int[]{SpirvTestModule.DECO_NON_WRITABLE}, NONE);
        m.storageBuffer(0, 0, u8, 1, NONE, new int[]{SpirvTestModule.DECO_NON_WRITABLE});
        // binding 1: a read-only view aliased with a writable one
        m.storageBuffer(0, 1, u32, 4, new int[]{SpirvTestModule.DECO_NON_WRITABLE}, NONE);
        m.storageBuffer(0, 1, f32, 4, NONE, NONE);
        // binding 2: write-only
        m.storageBuffer(0, 2, f32, 4, new int[]{SpirvTestModule.DECO_NON_READABLE}, NONE);
        m.localSize(64, 1, 1);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        assertEquals("main", info.entryPoint());
        assertEquals(3, info.bindingCount());
        assertEquals(3, info.bindingSlots());
        assertBinding(info, 0, true, false);
        assertBinding(info, 1, true, true);
        assertBinding(info, 2, false, true);
        assertTrue(info.requiresStorage8());
    }

    @Test
    public void strippedBindingLeavesAGapButKeepsTheOtherNumbers() {
        SpirvTestModule m = new SpirvTestModule("main");
        int f32 = m.typeFloat(32);
        m.storageBuffer(0, 0, f32, 4, new int[]{SpirvTestModule.DECO_NON_WRITABLE}, NONE);
        m.storageBuffer(0, 2, f32, 4, NONE, NONE);
        m.localSize(32, 1, 1);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        assertEquals(2, info.bindingCount());
        assertEquals(3, info.bindingSlots());
        assertTrue(info.hasBinding(0));
        assertFalse(info.hasBinding(1));
        assertNull(info.binding(1));
        assertBinding(info, 2, true, true);
    }

    @Test
    public void pushConstantMembersKeepKindsOffsetsAndSize() {
        SpirvTestModule m = new SpirvTestModule("main");
        int i32 = m.typeInt(32, true);
        int u32 = m.typeInt(32, false);
        int f32 = m.typeFloat(32);
        m.pushBlock(i32, f32, u32);
        m.localSize(1, 1, 1);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        List<SpirvModuleInfo.PushMember> members = info.pushMembers();
        assertEquals(3, members.size());
        assertMember(members.get(0), 0, SpirvModuleInfo.PushMember.Kind.I32);
        assertMember(members.get(1), 4, SpirvModuleInfo.PushMember.Kind.F32);
        assertMember(members.get(2), 8, SpirvModuleInfo.PushMember.Kind.U32);
        assertEquals(12, info.pushConstantSize());
        assertEquals(0, info.bindingCount());
    }

    @Test
    public void moduleWithoutPushBlockHasEmptyPushRange() {
        SpirvTestModule m = new SpirvTestModule("main");
        m.storageBuffer(0, 0, m.typeFloat(32), 4, NONE, NONE);
        m.localSize(64, 1, 1);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        assertEquals(0, info.pushConstantSize());
        assertTrue(info.pushMembers().isEmpty());
    }

    @Test
    public void localSizeIdReadsSpecIdsAndDefaults() {
        SpirvTestModule m = new SpirvTestModule("main");
        int u32 = m.typeInt(32, false);
        int x = m.specConstant(u32, 64, 0);
        int y = m.specConstant(u32, 1, 1);
        int z = m.specConstant(u32, 1, 2);
        m.specConstant(u32, 3072, 3);
        m.localSizeId(x, y, z);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        assertArrayEquals(new int[]{64, 1, 1}, info.localSize());
        assertArrayEquals(new int[]{0, 1, 2}, info.localSizeSpecIds());
        assertTrue(info.hasSpecLocalSize());
        Map<Integer, Integer> expected = new HashMap<>();
        expected.put(0, 64);
        expected.put(1, 1);
        expected.put(2, 1);
        expected.put(3, 3072);
        assertEquals(expected, info.specDefaults());
        assertTrue(info.isLocalSizeSpecId(0));
        assertFalse(info.isLocalSizeSpecId(3));
        assertTrue(info.declaresSpecId(3));
    }

    @Test
    public void workgroupSizeBuiltInMixesSpecialisedAndFixedDimensions() {
        SpirvTestModule m = new SpirvTestModule("main");
        int u32 = m.typeInt(32, false);
        int vec3 = m.typeVector(u32, 3);
        int x = m.specConstant(u32, 128, 0);
        int y = m.constant(u32, 2);
        int z = m.constant(u32, 1);
        m.workgroupSize(vec3, x, y, z);
        m.localSize(1, 1, 1);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        assertArrayEquals(new int[]{128, 2, 1}, info.localSize());
        assertArrayEquals(new int[]{0, -1, -1}, info.localSizeSpecIds());
        assertTrue(info.hasAnySpecLocalSize());
        assertFalse(info.hasSpecLocalSize());
    }

    @Test
    public void literalLocalSizeIsFixed() {
        SpirvTestModule m = new SpirvTestModule("kernel_entry");
        m.localSize(256, 1, 1);

        SpirvModuleInfo info = SpirvReflector.reflect(m.bytes());

        assertEquals("kernel_entry", info.entryPoint());
        assertArrayEquals(new int[]{256, 1, 1}, info.localSize());
        assertArrayEquals(new int[]{-1, -1, -1}, info.localSizeSpecIds());
        assertFalse(info.hasAnySpecLocalSize());
    }

    @Test
    public void bigEndianModuleReflectsLikeLittleEndian() {
        SpirvTestModule m = new SpirvTestModule("main");
        int f32 = m.typeFloat(32);
        m.storageBuffer(0, 0, f32, 4, new int[]{SpirvTestModule.DECO_NON_WRITABLE}, NONE);
        m.storageBuffer(0, 1, f32, 4, NONE, NONE);
        m.pushBlock(m.typeInt(32, true));
        m.localSize(64, 1, 1);

        SpirvModuleInfo little = SpirvReflector.reflect(m.bytes(ByteOrder.LITTLE_ENDIAN));
        SpirvModuleInfo big = SpirvReflector.reflect(m.bytes(ByteOrder.BIG_ENDIAN));

        assertTrue(little.sameInterface(big));
        assertArrayEquals(little.localSize(), big.localSize());
    }

    @Test
    public void reflectionLeavesTheBufferPositionAlone() {
        SpirvTestModule m = new SpirvTestModule("main");
        m.localSize(1, 1, 1);
        byte[] bytes = m.bytes();
        ByteBuffer padded = ByteBuffer.allocate(bytes.length + 8);
        padded.position(8);
        padded.put(bytes);
        padded.position(8);

        SpirvModuleInfo info = SpirvReflector.reflect(padded);

        assertEquals("main", info.entryPoint());
        assertEquals(8, padded.position());
        assertEquals(padded.capacity(), padded.limit());
    }

    @Test
    public void descriptorSetOtherThanZeroIsRefused() {
        SpirvTestModule m = new SpirvTestModule("main");
        m.storageBuffer(1, 0, m.typeFloat(32), 4, NONE, NONE);
        m.localSize(1, 1, 1);
        try {
            SpirvReflector.reflect(m.bytes());
            fail("descriptor set 1 must be refused");
        } catch (UnsupportedKernelException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("descriptor set 1"));
        }
    }

    @Test
    public void uniformBufferIsRefused() {
        SpirvTestModule m = new SpirvTestModule("main");
        int f32 = m.typeFloat(32);
        int block = m.typeStruct(f32);
        m.decorate(block, SpirvTestModule.DECO_BLOCK);
        m.memberDecorate(block, 0, SpirvTestModule.DECO_OFFSET, 0);
        int var = m.variable(m.typePointer(SpirvTestModule.SC_UNIFORM, block), SpirvTestModule.SC_UNIFORM);
        m.decorate(var, SpirvTestModule.DECO_DESCRIPTOR_SET, 0);
        m.decorate(var, SpirvTestModule.DECO_BINDING, 0);
        m.localSize(1, 1, 1);
        try {
            SpirvReflector.reflect(m.bytes());
            fail("a uniform buffer must be refused");
        } catch (UnsupportedKernelException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("uniform buffer"));
        }
    }

    @Test
    public void storageBufferWithoutBindingIsMalformed() {
        SpirvTestModule m = new SpirvTestModule("main");
        int f32 = m.typeFloat(32);
        int array = m.typeRuntimeArray(f32, 4);
        int block = m.typeStruct(array);
        m.decorate(block, SpirvTestModule.DECO_BLOCK);
        m.memberDecorate(block, 0, SpirvTestModule.DECO_OFFSET, 0);
        int var = m.variable(m.typePointer(SpirvTestModule.SC_STORAGE_BUFFER, block), SpirvTestModule.SC_STORAGE_BUFFER);
        m.decorate(var, SpirvTestModule.DECO_DESCRIPTOR_SET, 0);
        m.localSize(1, 1, 1);
        try {
            SpirvReflector.reflect(m.bytes());
            fail("a storage buffer without Binding must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("lacks a Binding"));
        }
    }

    @Test
    public void wrongMagicIsRejected() {
        SpirvTestModule m = new SpirvTestModule("main");
        m.localSize(1, 1, 1);
        byte[] bytes = m.bytes();
        bytes[0] ^= 0x55;
        try {
            SpirvReflector.reflect(bytes);
            fail("a corrupt magic word must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("magic"));
        }
    }

    @Test
    public void truncatedInstructionIsRejected() {
        SpirvTestModule m = new SpirvTestModule("main");
        m.localSize(1, 1, 1);
        int[] words = m.words();
        // header (5) + OpCapability (2) + OpMemoryModel (3), then the 5-word OpEntryPoint cut after 2 words
        int[] cut = Arrays.copyOf(words, 12);
        ByteBuffer b = ByteBuffer.allocate(cut.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int w : cut) {
            b.putInt(w);
        }
        try {
            SpirvReflector.reflect(b.array());
            fail("a truncated instruction must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("truncated or malformed"));
        }
    }

    @Test
    public void moduleWithoutComputeEntryPointIsRejected() {
        int[] header = {SpirvTestModule.MAGIC, SpirvTestModule.VERSION_1_5, 0, 4, 0};
        ByteBuffer b = ByteBuffer.allocate(header.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int w : header) {
            b.putInt(w);
        }
        try {
            SpirvReflector.reflect(b.array());
            fail("a module without a GLCompute entry point must be rejected");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("no GLCompute entry point"));
        }
    }

    @Test
    public void subgroupAndPlainVariantsWithEqualInterfacesMatch() {
        SpirvModuleInfo plain = SpirvReflector.reflect(variant(false));
        SpirvModuleInfo sg = SpirvReflector.reflect(variant(true));

        assertTrue(plain.sameInterface(sg));
        assertFalse(plain.requiresSubgroupArithmetic());
        assertTrue(sg.requiresSubgroupArithmetic());
    }

    /** One kernel interface (two buffers, two scalars, specialised local size), optionally with subgroup arithmetic. */
    private static byte[] variant(boolean subgroups) {
        SpirvTestModule m = new SpirvTestModule("main");
        if (subgroups) {
            m.capability(SpirvModuleInfo.CAP_GROUP_NON_UNIFORM_ARITHMETIC);
        }
        int f32 = m.typeFloat(32);
        int i32 = m.typeInt(32, true);
        int u32 = m.typeInt(32, false);
        m.storageBuffer(0, 0, f32, 4, new int[]{SpirvTestModule.DECO_NON_WRITABLE}, NONE);
        m.storageBuffer(0, 1, f32, 4, NONE, NONE);
        m.pushBlock(i32, f32);
        m.localSizeId(m.specConstant(u32, 64, 0), m.specConstant(u32, 1, 1), m.specConstant(u32, 1, 2));
        return m.bytes();
    }

    private static void assertBinding(SpirvModuleInfo info, int binding, boolean readable, boolean writable) {
        SpirvModuleInfo.BindingInfo b = info.binding(binding);
        assertNotNull("binding " + binding + " present", b);
        assertEquals("binding " + binding + " readable", readable, b.readable());
        assertEquals("binding " + binding + " writable", writable, b.writable());
    }

    private static void assertMember(SpirvModuleInfo.PushMember member, int offset,
                                     SpirvModuleInfo.PushMember.Kind kind) {
        assertEquals(offset, member.offset());
        assertEquals(4, member.size());
        assertEquals(kind, member.kind());
    }
}
