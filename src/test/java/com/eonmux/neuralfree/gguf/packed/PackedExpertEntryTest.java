package com.eonmux.neuralfree.gguf.packed;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The packed-expert entry layout against Avalon's {@code pack-experts.py} arithmetic: the role pad
 * ({@code role_pad}), the 256-byte-aligned role offsets of an entry built from standard tensors,
 * and the {@code avalon.experts_packed.blk.N} values read back into the same entry, with every
 * malformed array refused with the layer and key named.
 */
public class PackedExpertEntryTest {

    private static GGUFTensorInfo tensor(String name, GGMLType type, long... ne) {
        return new GGUFTensorInfo(name, ne, type, 0L);
    }

    /** The three expert tensors of a layer with n_embd 64, n_ff_exp 32 and 4 experts, F32. */
    private static List<GGUFTensorInfo> experts64x32x4() {
        return Arrays.asList(
                tensor("blk.0.ffn_up_exps.weight", GGMLType.F32, 64, 32, 4),
                tensor("blk.0.ffn_gate_exps.weight", GGMLType.F32, 64, 32, 4),
                tensor("blk.0.ffn_down_exps.weight", GGMLType.F32, 32, 64, 4));
    }

    @Test
    public void rolePadMatchesPackExpertsArithmetic() {
        // pack-experts.py role_pad: over = (round ne0 up to 512 elements - ne0) in whole blocks,
        // padded to 256 bytes.
        assertEquals(1792L, PackedExpertEntry.rolePad(GGMLType.F32, 64));   // 448 elements * 4 bytes
        assertEquals(0L, PackedExpertEntry.rolePad(GGMLType.F32, 512));
        assertEquals(1024L, PackedExpertEntry.rolePad(GGMLType.F16, 48));   // 464 elements * 2 bytes
        assertEquals(256L, PackedExpertEntry.rolePad(GGMLType.Q8_0, 480));  // 1 block of 34 bytes
        assertEquals(2048L, PackedExpertEntry.rolePad(GGMLType.F32, 32));   // 480 elements * 4 bytes
    }

    @Test
    public void entryLayoutMatchesPackExpertsLayout() throws IOException {
        // up/gate: 64*32*4 = 8192 data bytes, pad 1792; down: 32*64*4 = 8192, pad 2048.
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(3, experts64x32x4());
        assertEquals(3, entry.layer());
        assertEquals(4L, entry.nExpert());
        assertEquals(0L, entry.roles().get(0).offset());
        assertEquals(8192L, entry.roles().get(0).entryBytes());
        assertEquals(1792L, entry.roles().get(0).pad());
        assertEquals(9984L, entry.roles().get(1).offset());
        assertEquals(19968L, entry.roles().get(2).offset());
        assertEquals(2048L, entry.roles().get(2).pad());
        assertEquals(30208L, entry.stride());
        assertEquals("blk.3.ffn_exps_packed", entry.packedTensorName());
        assertEquals("avalon.experts_packed.blk.3", entry.metadataKey());
    }

    @Test
    public void metadataValuesRoundTrip() throws IOException {
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(0, experts64x32x4());
        long[] expected = {30208L, 4L,
                GGMLType.F32.getId(), 64L, 32L, 0L,
                GGMLType.F32.getId(), 64L, 32L, 9984L,
                GGMLType.F32.getId(), 32L, 64L, 19968L};
        assertArrayEquals(expected, entry.metadataValues());
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, entry.stride(), entry.nExpert());
        PackedExpertEntry read = PackedExpertEntry.fromMetadata(0, entry.metadataValues(), packed);
        assertEquals(entry, read);
        assertEquals(entry.metadataKey(), read.metadataKey());
        assertArrayEquals(entry.metadataValues(), read.metadataValues());
    }

    @Test
    public void roleLookupByStandardSuffix() throws IOException {
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(2, experts64x32x4());
        assertEquals("ffn_gate_exps", suffix(entry.roleForSuffix("ffn_gate_exps")));
        assertNull(entry.roleForSuffix("ffn_gate_up_exps"));
        assertNull(entry.roleForSuffix("ffn_up.weight"));
        assertEquals("blk.2.ffn_up_exps.weight", entry.roles().get(0).standardTensorName(2));
    }

    private static String suffix(PackedExpertEntry.Role role) {
        return "ffn_" + role.name() + "_exps";
    }

    @Test
    public void refusesWrongValueCount() throws IOException {
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, 30208, 4);
        long[] short13 = Arrays.copyOf(new long[14], 13);
        IOException e = assertThrows(IOException.class,
                () -> PackedExpertEntry.fromMetadata(0, short13, packed));
        assertTrue(e.getMessage(), e.getMessage().contains("holds 13 values"));
    }

    @Test
    public void refusesStrideNotMatchingTensorBytes() throws IOException {
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(0, experts64x32x4());
        long[] values = entry.metadataValues();
        values[0] = 30209L;
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, 30208, 4);
        IOException e = assertThrows(IOException.class,
                () -> PackedExpertEntry.fromMetadata(0, values, packed));
        assertTrue(e.getMessage(), e.getMessage().contains("expected stride 30209"));
    }

    @Test
    public void refusesRoleBeyondStride() throws IOException {
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(0, experts64x32x4());
        long[] values = entry.metadataValues();
        values[5 + 4 * 2] = entry.stride() - 4; // the down role's 8192 bytes no longer fit
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, entry.stride(), entry.nExpert());
        IOException e = assertThrows(IOException.class,
                () -> PackedExpertEntry.fromMetadata(0, values, packed));
        assertTrue(e.getMessage(), e.getMessage().contains("does not fit the entry stride"));
    }

    @Test
    public void acceptsRolesInAnyOrderThatFitTheStride() throws IOException {
        // Avalon's loader checks only that every role fits its entry (llama-model-loader.cpp:760)
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(0, experts64x32x4());
        long[] values = entry.metadataValues();
        long gateOffset = values[5 + 4];
        values[5 + 4] = 0L;
        values[5] = gateOffset;
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, entry.stride(), entry.nExpert());
        PackedExpertEntry read = PackedExpertEntry.fromMetadata(0, values, packed);
        assertEquals(gateOffset, read.roles().get(0).offset());
        assertEquals(0L, read.roles().get(1).offset());
    }

    @Test
    public void refusesStrideTimesExpertsOverflow() {
        long[] values = new long[PackedExpertEntry.METADATA_VALUES];
        values[0] = Long.MAX_VALUE / 2;
        values[1] = 4;
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, 30208, 4);
        IOException e = assertThrows(IOException.class, () -> PackedExpertEntry.fromMetadata(0, values, packed));
        assertTrue(e.getMessage(), e.getMessage().contains("expected stride"));
    }

    @Test
    public void refusesUnknownTypeId() throws IOException {
        PackedExpertEntry entry = PackedExpertEntry.fromStandardTensors(0, experts64x32x4());
        long[] values = entry.metadataValues();
        values[2] = 9999;
        GGUFTensorInfo packed = tensor("blk.0.ffn_exps_packed", GGMLType.I8, entry.stride(), entry.nExpert());
        IOException e = assertThrows(IOException.class,
                () -> PackedExpertEntry.fromMetadata(0, values, packed));
        assertTrue(e.getMessage(), e.getMessage().contains("type id 9999"));
    }

    @Test
    public void refusesExpertCountMismatchBetweenRoles() throws IOException {
        List<GGUFTensorInfo> experts = Arrays.asList(
                tensor("blk.0.ffn_up_exps.weight", GGMLType.F32, 64, 32, 4),
                tensor("blk.0.ffn_gate_exps.weight", GGMLType.F32, 64, 32, 5),
                tensor("blk.0.ffn_down_exps.weight", GGMLType.F32, 32, 64, 4));
        IOException e = assertThrows(IOException.class,
                () -> PackedExpertEntry.fromStandardTensors(0, experts));
        assertTrue(e.getMessage(), e.getMessage().contains("experts, but an earlier role holds 4"));
    }

    @Test
    public void refusesNonExpertShape() throws IOException {
        IOException e = assertThrows(IOException.class, () -> PackedExpertEntry.fromStandardTensors(0,
                Arrays.asList(
                        tensor("blk.0.ffn_up_exps.weight", GGMLType.F32, 64, 32),
                        tensor("blk.0.ffn_gate_exps.weight", GGMLType.F32, 64, 32),
                        tensor("blk.0.ffn_down_exps.weight", GGMLType.F32, 32, 64))));
        assertTrue(e.getMessage(), e.getMessage().contains("3-D"));
        assertNotEquals("", e.getMessage());
    }
}
