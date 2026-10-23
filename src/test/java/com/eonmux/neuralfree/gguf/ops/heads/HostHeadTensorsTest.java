package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The tensors only the host heads read, which a GPU load leaves off the device: the pooling and classifier head
 * tensors (qwen35's embedding output {@code cls.output}, {@code dense_2} / {@code dense_3}, {@code cls}), and with a
 * decision head its {@code decision.*} tensors, {@code token_types} when the embedding stage does not add it, the
 * modern-bert decision blocks outside the forward body and clef's decoder stack. The token table, the final norm and
 * the body's blocks stay on the device.
 */
public class HostHeadTensorsTest {

    private static final int N_EMBD = 8;
    private static final int N_OUT = 4;

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static ModelSpec spec(Arch arch, int nLayer, int nDecision, PoolingKind pooling, TensorCatalog tc,
                                  HeadSpec head) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(nLayer, 0)
                .nCtxTrain(512).nEmbd(N_EMBD).nEmbdOutImpl(N_OUT).normRmsEps(1e-6f).pooling(pooling).causal(false)
                .nLayerDecision(nDecision);
        return ModelSpec.builder(hb.build(), tc)
                .schedule(LayerSchedule.empty())
                .embedding(EmbeddingSpec.builder(tc.get("token_embd.weight"), N_EMBD).build())
                .head(head)
                .build();
    }

    @Test
    public void poolingAndClassifierTensorsAreHostOnly() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 16));
        ts.add(t("output_norm.weight", N_EMBD));
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        ts.add(t(TensorNames.weight(TensorNames.CLS_OUT), N_EMBD, N_OUT));
        ts.add(t(TensorNames.bias(TensorNames.CLS_OUT), N_OUT));
        ts.add(t(TensorNames.weight(TensorNames.DENSE_2_OUT), N_OUT, N_OUT));
        ts.add(t(TensorNames.bias(TensorNames.DENSE_2_OUT), N_OUT));
        ts.add(t(TensorNames.weight(TensorNames.DENSE_3_OUT), N_OUT, N_OUT));
        TensorCatalog tc = TensorCatalog.of(Arch.QWEN35.ggufName(), ts);
        HeadSpec head = HeadSpec.builder().pooling(PoolingKind.MEAN)
                .outputNorm(tc.get("output_norm.weight"), null)
                .output(tc.get("token_embd.weight"), true)
                .dense(tc.get("dense_2.weight"), tc.get("dense_3.weight"))
                .classifier(null, tc.get("cls.output.weight"), null)
                .embeddingOutput(tc.get("cls.output.weight"), tc.get("cls.output.bias"))
                .build();
        HostHeadTensors host = HostHeadTensors.of(spec(Arch.QWEN35, 1, 0, PoolingKind.MEAN, tc, head));

        assertFalse(host.isEmpty());
        for (String n : new String[]{"cls.output.weight", "cls.output.bias", "dense_2.weight", "dense_2.bias",
                "dense_3.weight"}) {
            assertTrue(n, host.test(n));
        }
        for (String n : new String[]{"token_embd.weight", "output_norm.weight", "blk.0.attn_norm.weight"}) {
            assertFalse(n, host.test(n));
        }
    }

    @Test
    public void aModelWithoutHostHeadsHasNone() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 16));
        ts.add(t("output_norm.weight", N_EMBD));
        TensorCatalog tc = TensorCatalog.of(Arch.QWEN35.ggufName(), ts);
        HeadSpec head = HeadSpec.builder().outputNorm(tc.get("output_norm.weight"), null)
                .output(tc.get("token_embd.weight"), true).build();
        HostHeadTensors host = HostHeadTensors.of(spec(Arch.QWEN35, 1, 0, PoolingKind.NONE, tc, head));

        assertTrue(host.isEmpty());
        assertFalse(host.test("token_embd.weight"));
    }

    @Test
    public void modernBertDecisionBlocksStayOnTheHost() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 16));
        ts.add(t("output_norm.weight", N_EMBD));
        ts.add(t(TensorNames.weight(TensorNames.TOKEN_TYPES), N_EMBD, 3));
        ts.add(t(TensorNames.weight(TensorNames.CLS), N_EMBD, N_EMBD));
        ts.add(t(TensorNames.bias(TensorNames.CLS_NORM), N_EMBD));
        ts.add(t(TensorNames.weight(TensorNames.CLS_OUT), N_EMBD, 1));
        for (int il = 0; il < 3; il++) {
            ts.add(t("blk." + il + ".attn_qkv.weight", N_EMBD, 3 * N_EMBD));
        }
        TensorCatalog tc = TensorCatalog.of(Arch.MODERN_BERT.ggufName(), ts);
        HeadSpec head = HeadSpec.builder().outputNorm(tc.get("output_norm.weight"), null).build();
        HostHeadTensors host = HostHeadTensors.of(spec(Arch.MODERN_BERT, 3, 1, PoolingKind.NONE, tc, head));

        assertTrue("the decision block", host.test("blk.2.attn_qkv.weight"));
        assertTrue("the decision block under its Hugging Face name", host.test("model.layers.2.attn_qkv.weight"));
        assertFalse("a body block", host.test("blk.0.attn_qkv.weight"));
        assertFalse("a body block", host.test("blk.1.attn_qkv.weight"));
        assertTrue(host.test("token_types.weight"));
        assertTrue(host.test("cls.weight"));
        assertTrue(host.test("cls.norm.bias"));
        assertTrue(host.test("cls.output.weight"));
        assertFalse(host.test("token_embd.weight"));
    }

    @Test
    public void clefDecisionTensorsAndDecoderStackStayOnTheHost() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, 16));
        ts.add(t("output_norm.weight", N_EMBD));
        ts.add(t("decision.proj_memory.weight", N_EMBD, N_OUT));
        ts.add(t("decision.scales", 3));
        ts.add(t("dec.blk.0.attn_q.weight", N_OUT, N_OUT));
        ts.add(t("blk.0.attn_norm.weight", N_EMBD));
        TensorCatalog tc = TensorCatalog.of(Arch.CLEF.ggufName(), ts);
        HeadSpec head = HeadSpec.builder().outputNorm(tc.get("output_norm.weight"), null)
                .output(tc.get("token_embd.weight"), true).build();
        HostHeadTensors host = HostHeadTensors.of(spec(Arch.CLEF, 1, 0, PoolingKind.NONE, tc, head));

        assertTrue(host.test("decision.proj_memory.weight"));
        assertTrue(host.test("decision.scales"));
        assertTrue(host.test("dec.blk.0.attn_q.weight"));
        assertFalse("clef's lexical term reads the tied table, which the body embeds with",
                host.test("token_embd.weight"));
        assertFalse(host.test("blk.0.attn_norm.weight"));
    }
}
