package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link LoraAttachment#cpu}: every pair kind registered on the site its block serves and applied through it, several
 * adapters on one tensor adding up, {@link LoraAttachment#close} removing every delta, and the clear errors (adapter
 * and tensor named, nothing left registered) for an unserved tensor, a refusing block, a kind mismatch and a shape
 * mismatch.
 */
public class LoraAttachmentTest {

    private static final float EPS = 1e-5f;
    private static final int N_EMBD = 6;
    private static final int N_VOCAB = 9;
    private static final int N_FF = 5;
    private static final int N_EXPERT = 3;
    private static final String Q = TensorNames.weight(TensorNames.ATTN_Q);
    private static final String K = TensorNames.weight(TensorNames.ATTN_K);
    private static final String UP_EXPS = TensorNames.weight(TensorNames.FFN_UP_EXPS);
    private static final String OUTPUT = TensorNames.weight(TensorNames.OUTPUT);
    private static final String EMBD = TensorNames.weight(TensorNames.TOKEN_EMBD);

    private static LayerSpec layer(int il) {
        return LayerSpec.builder(il, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .build();
    }

    private static CpuTensorRef f32(long rows, long cols) {
        ByteBuffer buf = ByteBuffer.allocate((int) (rows * cols * Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
        return CpuTensorRef.of(GGMLType.F32, buf, rows, cols);
    }

    /** Deltas of a one-layer model serving attn_q, attn_k, ffn_up_exps, the output head and the token table. */
    private static CpuProjectionDeltas servedModel() {
        CpuProjectionDeltas d = new CpuProjectionDeltas();
        LayerSpec l = layer(0);
        d.serve(l, Q, f32(N_EMBD, N_EMBD), CpuMatVec.PLAIN);
        d.serve(l, K, f32(N_EMBD, N_EMBD), CpuMatVec.PLAIN);
        d.serveExperts(l, UP_EXPS, f32((long) N_EXPERT * N_FF, N_EMBD), N_EXPERT, CpuMatVec.PLAIN);
        d.serveModel(OUTPUT, N_EMBD, N_VOCAB, null);
        d.serveEmbedding(EMBD, N_VOCAB, N_EMBD, null);
        return d;
    }

    private static String blk0(String weight) {
        return layer(0).tensorName(weight);
    }

    @Test
    public void everyPairKindIsAppliedThroughItsSiteAndCloseRemovesIt() {
        CpuProjectionDeltas deltas = servedModel();
        LoraAdapter.Pair q = LoraFixtures.projection(blk0(Q), N_EMBD, N_EMBD, 2, 1);
        LoraAdapter.Pair up = LoraFixtures.experts(blk0(UP_EXPS), N_EMBD, N_FF, 2, N_EXPERT, 3);
        LoraAdapter.Pair out = LoraFixtures.projection(OUTPUT, N_EMBD, N_VOCAB, 3, 5);
        LoraAdapter.Pair embd = LoraFixtures.embedding(EMBD, N_VOCAB, N_EMBD, 2, 7);
        LoraAdapter adapter = LoraFixtures.adapter("all.gguf", 4.0f, q, up, out, embd);
        float user = 0.5f;

        LoraAttachment att = LoraAttachment.cpu(
                Collections.singletonList(new LoraAttachment.Binding(adapter, user)), deltas);
        assertEquals(4, att.deltaCount());

        float[] x = LoraFixtures.random(11, 2 * N_EMBD);
        float[] y = new float[2 * N_EMBD];
        float[] expected = new float[2 * N_EMBD];
        for (int t = 0; t < 2; t++) {
            LoraFixtures.project(q, adapter.scale(q.target(), user), 0, x, t * N_EMBD, expected, t * N_EMBD);
        }
        site(deltas, q.target()).applyRows(2, x, 0, y, 0);
        assertArrayEquals(expected, y, EPS);

        float[] logits = new float[N_VOCAB];
        float[] expectedLogits = new float[N_VOCAB];
        LoraFixtures.project(out, adapter.scale(OUTPUT, user), 0, x, 0, expectedLogits, 0);
        site(deltas, OUTPUT).applyRows(1, x, 0, logits, 0);
        assertArrayEquals(expectedLogits, logits, EPS);

        int[] ids = {2, 0};
        float[] expertOut = new float[2 * N_FF];
        float[] expectedExperts = new float[2 * N_FF];
        for (int p = 0; p < 2; p++) {
            LoraFixtures.project(up, adapter.scale(up.target(), user), ids[p], x, 0, expectedExperts, p * N_FF);
        }
        ((CpuProjectionDeltas.ExpertSite) deltas.served(up.target())).apply(2, 2, ids, x, 0, true, expertOut, 0);
        assertArrayEquals(expectedExperts, expertOut, EPS);

        int[] tokens = {8, 1};
        float[] rows = new float[2 * N_EMBD];
        float[] expectedRows = new float[2 * N_EMBD];
        for (int t = 0; t < 2; t++) {
            LoraFixtures.embed(embd, adapter.scale(EMBD, user), tokens[t], expectedRows, t * N_EMBD);
        }
        ((CpuProjectionDeltas.EmbeddingSite) deltas.served(EMBD)).apply(2, tokens, 0, rows, 0, N_EMBD);
        assertArrayEquals(expectedRows, rows, EPS);

        att.close();
        for (String name : Arrays.asList(q.target(), up.target(), OUTPUT, EMBD)) {
            assertTrue(name + " keeps a delta after close", deltas.served(name).isEmpty());
        }
    }

    @Test
    public void adaptersOnOneTensorAddUpInLoadOrder() {
        CpuProjectionDeltas deltas = servedModel();
        LoraAdapter.Pair q1 = LoraFixtures.projection(blk0(Q), N_EMBD, N_EMBD, 2, 21);
        LoraAdapter.Pair q2 = LoraFixtures.projection(blk0(Q), N_EMBD, N_EMBD, 3, 23);
        LoraAdapter a1 = LoraFixtures.adapter("one.gguf", 0.0f, q1);
        LoraAdapter a2 = LoraFixtures.adapter("two.gguf", 6.0f, q2);
        LoraAttachment att = LoraAttachment.cpu(Arrays.asList(new LoraAttachment.Binding(a1, 1.0f),
                new LoraAttachment.Binding(a2, 0.25f)), deltas);
        assertEquals(2, att.deltaCount());

        float[] x = LoraFixtures.random(25, N_EMBD);
        float[] expected = new float[N_EMBD];
        LoraFixtures.project(q1, a1.scale(q1.target(), 1.0f), 0, x, 0, expected, 0);
        LoraFixtures.project(q2, a2.scale(q2.target(), 0.25f), 0, x, 0, expected, 0);
        float[] y = new float[N_EMBD];
        site(deltas, q1.target()).applyRows(1, x, 0, y, 0);
        assertArrayEquals(expected, y, EPS);
        att.close();
        assertTrue(deltas.served(q1.target()).isEmpty());
    }

    @Test
    public void anUnservedTensorFailsNamingAdapterAndTensorAndRegistersNothing() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        deltas.serve(layer(0), Q, f32(N_EMBD, N_EMBD), CpuMatVec.PLAIN);
        LoraAdapter adapter = LoraFixtures.adapter("partial.gguf", 0.0f,
                LoraFixtures.projection(blk0(Q), N_EMBD, N_EMBD, 2, 31),
                LoraFixtures.projection(blk0(K), N_EMBD, N_EMBD, 2, 33));
        try {
            LoraAttachment.cpu(Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f)), deltas);
            fail("attn_k is served by no block");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("partial.gguf"));
            assertTrue(expected.getMessage(), expected.getMessage().contains(blk0(K)));
        }
        assertTrue("the valid attn_q pair of the failing attachment must not stay registered",
                deltas.served(blk0(Q)).isEmpty());
    }

    @Test
    public void aRefusingBlockFailsWithItsReason() {
        CpuProjectionDeltas deltas = new CpuProjectionDeltas();
        deltas.serve(layer(0), Q, "the input is PRISM-rotated");
        LoraAdapter adapter = LoraFixtures.adapter("prism.gguf", 0.0f,
                LoraFixtures.projection(blk0(Q), N_EMBD, N_EMBD, 2, 41));
        try {
            LoraAttachment.cpu(Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f)), deltas);
            fail("a refusing block must reject the adapter");
        } catch (UnsupportedOperationException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("PRISM-rotated"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("prism.gguf"));
        }
        assertTrue(deltas.served(blk0(Q)).isEmpty());
    }

    @Test
    public void aPairOfAnotherKindThanTheServedProductFails() {
        CpuProjectionDeltas deltas = servedModel();
        LoraAdapter adapter = LoraFixtures.adapter("kind.gguf", 0.0f,
                LoraFixtures.experts(blk0(Q), N_EMBD, N_EMBD, 2, N_EXPERT, 51));
        try {
            LoraAttachment.cpu(Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f)), deltas);
            fail("an expert pair on a dense projection must be rejected");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("EXPERTS"));
        }
        assertTrue(deltas.served(blk0(Q)).isEmpty());
    }

    @Test
    public void aPairOfAnotherShapeThanTheServedProductFails() {
        CpuProjectionDeltas deltas = servedModel();
        LoraAdapter adapter = LoraFixtures.adapter("shape.gguf", 0.0f,
                LoraFixtures.projection(OUTPUT, N_EMBD, N_VOCAB + 1, 2, 61));
        try {
            LoraAttachment.cpu(Collections.singletonList(new LoraAttachment.Binding(adapter, 1.0f)), deltas);
            fail("a pair of " + (N_VOCAB + 1) + " outputs must not attach to a " + N_VOCAB + "-row head");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(OUTPUT));
        }
        assertNotNull(deltas.served(OUTPUT));
        assertTrue(deltas.served(OUTPUT).isEmpty());
    }

    private static CpuProjectionDeltas.Site site(CpuProjectionDeltas deltas, String name) {
        return (CpuProjectionDeltas.Site) deltas.served(name);
    }
}
