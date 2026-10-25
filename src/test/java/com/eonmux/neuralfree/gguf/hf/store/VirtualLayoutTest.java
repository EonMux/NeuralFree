package com.eonmux.neuralfree.gguf.hf.store;

import com.eonmux.neuralfree.gguf.DataSegment;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.hf.arch.HfForceType;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.store.MaterializePolicyTest.Counting;
import com.eonmux.neuralfree.gguf.hf.tensor.DtypePolicy;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlanBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link VirtualLayout}: stored tensors keep their unaligned file offsets under {@code V_f = Σ align64(len)}, derived
 * tensors follow at {@code P}, each slot aligned to 64 in plan order.
 */
public class VirtualLayoutTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    static TensorPlanBuilder builder() {
        return new TensorPlanBuilder(new DtypePolicy(n -> false, OutType.SOURCE), HfForceType.NONE);
    }

    private static long offsetOf(VirtualLayout l, String name) {
        for (GGUFTensorInfo t : l.tensors()) {
            if (t.name().equals(name)) {
                return t.offset();
            }
        }
        throw new AssertionError(name + " is not in the layout");
    }

    /** File a holds 100 bytes at V 0, file b 70 bytes at V 128: P = 128 + 128 = 256. */
    private static List<DataSegment> twoShards() {
        return Arrays.asList(new DataSegment("a.safetensors", 8 + 120, 0, 100),
                new DataSegment("b.safetensors", 8 + 88, 128, 70));
    }

    @Test
    public void storedTensorsKeepUnalignedOffsetsAndDerivedOnesFollowAligned() {
        TensorPlanBuilder b = builder();
        b.add("blk.0.attn_q.weight", Counting.stored(4, 0, 2, 3));
        b.add("blk.0.attn_k.weight", Counting.stored(132, 1, 2, 3));
        b.add("blk.0.ffn_down_exps.weight", ExpertTransforms.transposeLast2(new Counting(2, 1, 5, 3)));
        b.add("blk.0.attn_v.weight", new Counting(3, 2, 2));
        b.add("blk.0.attn_output.weight", Counting.stored(52, 4, 2, 3));
        TensorPlan plan = b.build();

        VirtualLayout l = VirtualLayout.of(twoShards(), plan);
        assertEquals(256, l.producedBase());
        assertEquals("unaligned begin in file a", 4, offsetOf(l, "blk.0.attn_q.weight"));
        assertEquals("unaligned begin in file b", 132, offsetOf(l, "blk.0.attn_k.weight"));
        assertEquals(52, offsetOf(l, "blk.0.attn_output.weight"));
        assertEquals("the first derived tensor starts at P", 256, offsetOf(l, "blk.0.ffn_down_exps.weight"));
        assertEquals("256 + 60 rounded up to 64", 320, offsetOf(l, "blk.0.attn_v.weight"));
        assertEquals(320 + 16, l.totalBytes());

        ProducedSegments p = l.produced();
        assertEquals(2, p.size());
        assertEquals("plan order", Arrays.asList("blk.0.ffn_down_exps.weight", "blk.0.attn_v.weight"),
                Arrays.asList(p.tensors().get(0).ggufName(), p.tensors().get(1).ggufName()));
        for (int i = 0; i < p.size(); i++) {
            assertEquals(0, p.offset(i) % VirtualLayout.ALIGNMENT);
        }
        assertEquals(1, p.indexOf("blk.0.attn_v.weight"));
        assertEquals(-1, p.indexOf("blk.0.attn_q.weight"));
        assertEquals(256, p.base());
        assertEquals(l.totalBytes(), p.end());
        assertEquals("the infos follow the plan order", plan.names().size(), l.tensors().size());
        assertEquals("blk.0.attn_q.weight", l.tensors().get(0).name());
    }

    @Test
    public void withoutDerivedTensorsTheEndIsP() {
        TensorPlanBuilder b = builder();
        b.add("blk.0.attn_q.weight", Counting.stored(0, 0, 2, 3));
        VirtualLayout l = VirtualLayout.of(twoShards(), b.build());
        assertEquals(256, l.producedBase());
        assertEquals(256, l.totalBytes());
        assertEquals(0, l.produced().size());
    }

    @Test
    public void withoutFilesDerivedTensorsStartAtZero() {
        TensorPlanBuilder b = builder();
        b.add("blk.0.a.weight", new Counting(0, 3, 5));
        b.add("blk.0.b.weight", new Counting(0, 1, 2));
        VirtualLayout l = VirtualLayout.of(Collections.emptyList(), b.build());
        assertEquals(0, l.producedBase());
        assertEquals(0, offsetOf(l, "blk.0.a.weight"));
        assertEquals(64, offsetOf(l, "blk.0.b.weight"));
        assertEquals(72, l.totalBytes());
    }

    @Test
    public void storedBytesOutsideEveryFileAreRejected() {
        TensorPlanBuilder b = builder();
        b.add("blk.0.attn_q.weight", Counting.stored(90, 0, 2, 3));
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> VirtualLayout.of(twoShards(), b.build()));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.0.attn_q.weight"));
        TensorPlanBuilder gap = builder();
        gap.add("blk.0.attn_k.weight", Counting.stored(110, 0, 2, 3));
        assertThrows("the gap between files holds no tensor", IllegalStateException.class,
                () -> VirtualLayout.of(twoShards(), gap.build()));
    }

    @Test
    public void overlappingFilesAreRejected() {
        List<DataSegment> overlapping = Arrays.asList(new DataSegment("a", 8, 0, 100),
                new DataSegment("b", 8, 64, 10));
        TensorPlanBuilder b = builder();
        b.add("blk.0.x.weight", new Counting(0, 1, 4));
        assertThrows(IllegalArgumentException.class, () -> VirtualLayout.of(overlapping, b.build()));
    }

    @Test
    public void aConvertedTensorIsDerivedAlthoughItsSourceIsStored() throws IOException {
        Path dir = SyntheticHf.builder().config("{}")
                .tensor("w", StDType.BF16, new long[]{2, 4}, SyntheticHf.bf16(1, 2, 3, 4, 5, 6, 7, 8))
                .write(tmp.newFolder().toPath());
        try (CheckpointContainer ckpt = CheckpointContainers.open(dir, dir, false)) {
            TensorPlanBuilder f32 = new TensorPlanBuilder(new DtypePolicy(n -> false, OutType.F32),
                    HfForceType.NONE);
            f32.add("blk.0.attn_q.weight", ckpt.source("w"));
            VirtualLayout converted = VirtualLayout.of(ckpt, f32.build());
            assertEquals(1, converted.produced().size());
            assertEquals(converted.producedBase(), converted.tensors().get(0).offset());

            TensorPlanBuilder same = builder();
            same.add("blk.0.attn_q.weight", ckpt.source("w"));
            VirtualLayout stored = VirtualLayout.of(ckpt, same.build());
            assertEquals(0, stored.produced().size());
            assertEquals(ckpt.directVirtualOffset("w"), stored.tensors().get(0).offset());
        }
    }

    @Test
    public void twoRealShardsLayOutAsTheCheckpointDoes() throws IOException {
        Path dir = SyntheticHf.builder().config("{}")
                .tensor("a", StDType.F32, new long[]{3}, SyntheticHf.f32(1, 2, 3))
                .tensor("b", StDType.F32, new long[]{5}, SyntheticHf.f32(1, 2, 3, 4, 5))
                .tensor("c", StDType.F32, new long[]{7}, SyntheticHf.f32(1, 2, 3, 4, 5, 6, 7))
                .shards(2)
                .write(tmp.newFolder().toPath());
        try (CheckpointContainer ckpt = CheckpointContainers.open(dir, dir, false)) {
            List<DataSegment> segs = ckpt.segments();
            assertEquals(2, segs.size());
            assertEquals(0, segs.get(1).virtualBase() % VirtualLayout.ALIGNMENT);
            TensorPlanBuilder b = builder();
            b.add("x.weight", ckpt.source("a"));
            b.add("y.weight", ckpt.source("b"));
            b.add("z.weight", ckpt.source("c"));
            b.add("d.weight", new Counting(9, 2, 3));
            VirtualLayout l = VirtualLayout.of(ckpt, b.build());
            assertEquals(ckpt.directVirtualOffset("a"), offsetOf(l, "x.weight"));
            assertEquals(ckpt.directVirtualOffset("b"), offsetOf(l, "y.weight"));
            assertEquals(ckpt.directVirtualOffset("c"), offsetOf(l, "z.weight"));
            assertNotEquals("b is unaligned in its file", 0, ckpt.directVirtualOffset("b") % 64);
            long p = VirtualLayout.align(segs.get(1).virtualEnd());
            assertEquals(p, l.producedBase());
            assertEquals(p, offsetOf(l, "d.weight"));
            assertEquals(p + 24, l.totalBytes());
        }
    }
}
