package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The tensors a GPU load places on the device: a Nemotron-H model with one NextN block keeps the block off the device
 * and out of the planned bytes unless MTP is loaded, as llama.cpp skips the NextN tensors without {@code load_mtp}
 * (llama-model.cpp:3915-3917) and counts them in its fit only with it (common/fit.cpp:141-143); placing them later
 * uploads exactly the block. A model without NextN blocks places every tensor.
 */
public class WeightPlacementTest {

    private static final String ARCH = "nemotron_h_moe";
    private static final int N_EMBD = 64;
    private static final int N_HEAD = 4;
    private static final int HEAD_DIM = N_EMBD / N_HEAD;
    private static final int N_HEAD_KV = 2;
    private static final int N_VOCAB = 32;
    private static final int N_EXPERT = 4;
    private static final int N_FF_EXP = 32;
    private static final int N_FF_SHEXP = 32;
    private static final int SSM_CONV = 4;
    private static final int SSM_INNER = 64;
    private static final int SSM_STATE = 16;
    private static final int SSM_HEADS = 8;
    private static final int SSM_GROUPS = 2;

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static Map<String, Object> metadata(int blocks, int nextn) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", ARCH);
        m.put(ARCH + ".context_length", 4096);
        m.put(ARCH + ".embedding_length", N_EMBD);
        m.put(ARCH + ".block_count", blocks);
        m.put(ARCH + ".nextn_predict_layers", nextn);
        int[] ff = {0, 0, N_FF_EXP, N_FF_EXP};
        int[] kv = {0, N_HEAD_KV, 0, N_HEAD_KV};
        int[] ffN = new int[blocks];
        int[] kvN = new int[blocks];
        System.arraycopy(ff, 0, ffN, 0, blocks);
        System.arraycopy(kv, 0, kvN, 0, blocks);
        m.put(ARCH + ".feed_forward_length", ffN);
        m.put(ARCH + ".attention.head_count", N_HEAD);
        m.put(ARCH + ".attention.head_count_kv", kvN);
        m.put(ARCH + ".attention.layer_norm_epsilon", 1e-5f);
        m.put(ARCH + ".attention.layer_norm_rms_epsilon", 1e-5f);
        m.put(ARCH + ".ssm.conv_kernel", SSM_CONV);
        m.put(ARCH + ".ssm.inner_size", SSM_INNER);
        m.put(ARCH + ".ssm.state_size", SSM_STATE);
        m.put(ARCH + ".ssm.time_step_rank", SSM_HEADS);
        m.put(ARCH + ".ssm.group_count", SSM_GROUPS);
        m.put(ARCH + ".expert_count", N_EXPERT);
        m.put(ARCH + ".expert_used_count", 2);
        m.put(ARCH + ".expert_feed_forward_length", N_FF_EXP);
        m.put(ARCH + ".expert_shared_feed_forward_length", N_FF_SHEXP);
        m.put(ARCH + ".expert_weights_norm", true);
        return m;
    }

    private static void attention(List<GGUFTensorInfo> ts, int il) {
        String p = "blk." + il + ".";
        ts.add(t(p + "attn_q.weight", N_EMBD, (long) N_HEAD * HEAD_DIM));
        ts.add(t(p + "attn_k.weight", N_EMBD, (long) N_HEAD_KV * HEAD_DIM));
        ts.add(t(p + "attn_v.weight", N_EMBD, (long) N_HEAD_KV * HEAD_DIM));
        ts.add(t(p + "attn_output.weight", (long) N_HEAD * HEAD_DIM, N_EMBD));
    }

    private static void experts(List<GGUFTensorInfo> ts, int il) {
        String p = "blk." + il + ".";
        ts.add(t(p + "ffn_gate_inp.weight", N_EMBD, N_EXPERT));
        ts.add(t(p + "exp_probs_b.bias", N_EXPERT));
        ts.add(t(p + "ffn_up_exps.weight", N_EMBD, N_FF_EXP, N_EXPERT));
        ts.add(t(p + "ffn_down_exps.weight", N_FF_EXP, N_EMBD, N_EXPERT));
        ts.add(t(p + "ffn_up_shexp.weight", N_EMBD, N_FF_SHEXP));
        ts.add(t(p + "ffn_down_shexp.weight", N_FF_SHEXP, N_EMBD));
    }

    /** The three trunk blocks, and the NextN block 3 when {@code withNextn}. */
    private static List<GGUFTensorInfo> tensors(boolean withNextn) {
        long convDim = SSM_INNER + 2L * SSM_GROUPS * SSM_STATE;
        long inProj = SSM_INNER + convDim + SSM_HEADS;
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, N_VOCAB));
        ts.add(t("output_norm.weight", N_EMBD));
        for (int il = 0; il < (withNextn ? 4 : 3); il++) {
            ts.add(t("blk." + il + ".attn_norm.weight", N_EMBD));
        }
        ts.add(t("blk.0.ssm_in.weight", N_EMBD, inProj));
        ts.add(t("blk.0.ssm_conv1d.weight", SSM_CONV, convDim));
        ts.add(t("blk.0.ssm_dt.bias", SSM_HEADS));
        ts.add(t("blk.0.ssm_a", 1, SSM_HEADS));
        ts.add(t("blk.0.ssm_d", 1, SSM_HEADS));
        ts.add(t("blk.0.ssm_norm.weight", SSM_INNER / SSM_GROUPS, SSM_GROUPS));
        ts.add(t("blk.0.ssm_out.weight", SSM_INNER, N_EMBD));
        attention(ts, 1);
        experts(ts, 2);
        if (withNextn) {
            ts.add(t("blk.3.nextn.eh_proj.weight", 2L * N_EMBD, N_EMBD));
            ts.add(t("blk.3.nextn.enorm.weight", N_EMBD));
            ts.add(t("blk.3.nextn.hnorm.weight", N_EMBD));
            ts.add(t("blk.3.nextn.shared_head_norm.weight", N_EMBD));
            attention(ts, 3);
            ts.add(t("blk.3.post_attention_norm.weight", N_EMBD));
            experts(ts, 3);
        }
        return ts;
    }

    private static ModelSpec spec(boolean withNextn) {
        return ModelSpecBuilder.build(GgufKv.ofMap(metadata(withNextn ? 4 : 3, withNextn ? 1 : 0)),
                TensorCatalog.of(ARCH, tensors(withNextn)), MemoryConfig.defaults());
    }

    /** Tensor data of zeros. */
    private static final class ZeroStore implements WeightStore {
        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            Arrays.fill(dst, 0, size, (byte) 0);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            Arrays.fill(dst, dstOffset, dstOffset + size, (byte) 0);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            for (int i = 0; i < size; i++) {
                dst.put((byte) 0);
            }
        }

        @Override
        public void close() {
        }
    }

    private static GpuWeightUploader uploader(FakeComputeBackend backend, List<GGUFTensorInfo> tensors) {
        return new GpuWeightUploader(backend, backend.computeQueue(), tensors, new ZeroStore(), ARCH,
                GpuWeightUploader.config((type, use) -> true).build());
    }

    @Test
    public void nextnBlockIsPlacedOnlyWhenMtpIsLoaded() {
        ModelSpec spec = spec(true);
        WeightPlacement off = WeightPlacement.of(spec, false);
        assertTrue(off.hasNextn());
        assertFalse(off.nextnOnDevice());
        for (GGUFTensorInfo t : tensors(true)) {
            boolean nextn = t.name().startsWith("blk.3.");
            assertEquals(t.name(), nextn, off.isNextn(t));
            assertEquals(t.name(), !nextn, off.onDevice(t));
        }
        WeightPlacement on = WeightPlacement.of(spec, true);
        for (GGUFTensorInfo t : tensors(true)) {
            assertTrue(t.name(), on.onDevice(t));
        }
        assertTrue(off.withNextnOnDevice().onDevice(t("blk.3.nextn.eh_proj.weight", 2L * N_EMBD, N_EMBD)));
        assertSame(on, on.withNextnOnDevice());
    }

    @Test
    public void plannedBytesLeaveTheSkippedBlockOutAndItUploadsOnDemand() {
        ModelSpec spec = spec(true);
        WeightPlacement off = WeightPlacement.of(spec, false);
        FakeComputeBackend backend = FakeComputeBackend.create();
        GpuWeightUploader up = uploader(backend, tensors(true));
        try {
            long all = up.plannedBytes(t -> true);
            long nextn = up.plannedBytes(off::isNextn);
            assertTrue(nextn > 0);
            assertEquals(all - nextn, up.plannedBytes(off::onDevice));
            assertEquals(up.plans(off::isNextn).size(), up.plans(t -> t.name().startsWith("blk.3.")).size());

            up.uploadResident(t -> !off.onDevice(t));
            assertNull(up.find("blk.3.nextn.eh_proj.weight"));
            assertNotNull(up.find("blk.2.ffn_up_exps.weight"));
            assertEquals(all - nextn, up.residentBytes());

            up.uploadResident(t -> !off.isNextn(t));
            assertNotNull(up.find("blk.3.nextn.eh_proj.weight"));
            assertEquals(all, up.residentBytes());
        } finally {
            up.close();
        }
    }

    @Test
    public void aModelWithoutNextnBlocksPlacesEveryTensor() {
        WeightPlacement p = WeightPlacement.of(spec(false), false);
        assertFalse(p.hasNextn());
        for (GGUFTensorInfo t : tensors(false)) {
            assertTrue(t.name(), p.onDevice(t));
            assertFalse(t.name(), p.isNextn(t));
        }
    }
}
