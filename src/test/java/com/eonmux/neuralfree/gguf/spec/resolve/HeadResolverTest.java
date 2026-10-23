package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsLoader;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Output head resolution: {@code output.bias} only where the graph adds it, chameleon's image-token logit mask,
 * neo-bert's encoder output norm, qwen35's {@code cls.output} embedding output beside its LM head, and the
 * {@link HeadSpec} contract of the embedding projection and output, the logit mask and the grouped output norm.
 */
public class HeadResolverTest {

    private static final int N_EMBD = 64;
    private static final int N_VOCAB = 9000;

    private static Map<String, Object> base(String arch) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", arch);
        m.put(arch + ".context_length", 4096);
        m.put(arch + ".embedding_length", N_EMBD);
        m.put(arch + ".block_count", 1);
        m.put(arch + ".feed_forward_length", 128);
        m.put(arch + ".attention.head_count", 4);
        m.put(arch + ".attention.layer_norm_rms_epsilon", 1e-6f);
        return m;
    }

    private static ModelHparams load(Arch arch) {
        return HparamsLoader.load(GgufKv.ofMap(base(arch.ggufName())), arch);
    }

    private static GGUFTensorInfo t(String name, long... ne) {
        return new GGUFTensorInfo(name, ne, GGMLType.F32, 0L);
    }

    private static List<GGUFTensorInfo> lmHead(boolean bias) {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, N_VOCAB));
        ts.add(t("output_norm.weight", N_EMBD));
        ts.add(t("output.weight", N_EMBD, N_VOCAB));
        if (bias) {
            ts.add(t("output.bias", N_VOCAB));
        }
        return ts;
    }

    private static HeadSpec resolve(Arch arch, List<GGUFTensorInfo> ts) {
        return new HeadResolver(load(arch), TensorCatalog.of(arch.ggufName(), ts)).resolve();
    }

    @Test
    public void outputBiasIsAttachedOnlyForGraphsThatAddIt() {
        assertNotNull(resolve(Arch.QWEN2, lmHead(true)).outputBias());
        assertNull(resolve(Arch.LLAMA, lmHead(true)).outputBias());

        List<GGUFTensorInfo> tied = new ArrayList<>();
        tied.add(t("token_embd.weight", N_EMBD, N_VOCAB));
        tied.add(t("output_norm.weight", N_EMBD));
        tied.add(t("output.bias", N_VOCAB));
        HeadSpec qwen2Tied = resolve(Arch.QWEN2, tied);
        assertTrue(qwen2Tied.tied());
        assertNotNull(qwen2Tied.outputBias());
    }

    @Test
    public void chameleonMasksTheImageTokenLogits() {
        HeadSpec h = resolve(Arch.CHAMELEON, lmHead(false));
        assertTrue(h.masksLogits());
        assertEquals(HeadResolver.CHAMELEON_IMAGE_TOKEN_START, h.maskedLogitsStart());
        assertEquals(HeadResolver.CHAMELEON_IMAGE_TOKEN_END, h.maskedLogitsEnd());
        assertFalse(resolve(Arch.LLAMA, lmHead(false)).masksLogits());
    }

    @Test
    public void neoBertNormalizesWithTheEncoderOutputNorm() {
        List<GGUFTensorInfo> ts = new ArrayList<>();
        ts.add(t("token_embd.weight", N_EMBD, N_VOCAB));
        ts.add(t("enc.output_norm.weight", N_EMBD));
        HeadSpec h = resolve(Arch.NEO_BERT, ts);
        assertNotNull(h.outputNorm());
        assertEquals("enc.output_norm.weight", h.outputNorm().name());
    }

    @Test
    public void embeddingProjectionSetsTheHiddenWidth() {
        TensorRef proj = TensorRef.of(t("output.weight", N_EMBD, 32));
        HeadSpec h = HeadSpec.builder().embeddingProjection(proj).build();
        assertEquals(32, h.hiddenWidth(N_EMBD));
        assertEquals(N_EMBD, HeadSpec.builder().build().hiddenWidth(N_EMBD));
    }

    private static final int N_EMBD_OUT = 32;

    private static HeadSpec resolveQwen35(List<GGUFTensorInfo> ts) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN35)).layerCounts(1, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).nEmbdOutImpl(N_EMBD_OUT).normRmsEps(1e-6f).build();
        return new HeadResolver(hp, TensorCatalog.of(Arch.QWEN35.ggufName(), ts)).resolve();
    }

    @Test
    public void qwen35ClsOutputIsTheEmbeddingOutputBesideTheLmHead() {
        List<GGUFTensorInfo> ts = lmHead(false);
        ts.add(t("cls.output.weight", N_EMBD, N_EMBD_OUT));
        ts.add(t("cls.output.bias", N_EMBD_OUT));
        HeadSpec h = resolveQwen35(ts);
        assertEquals("cls.output.weight", h.embeddingOutput().name());
        assertEquals("cls.output.bias", h.embeddingOutputBias().name());
        assertEquals("output.weight", h.output().name());
        assertNull(h.embeddingProjection());
        assertEquals(N_EMBD, h.hiddenWidth(N_EMBD));
        assertEquals(N_EMBD_OUT, h.embeddingWidth(N_EMBD));
        assertEquals("cls.output.weight", h.clsOutput().name());

        HeadSpec plain = resolveQwen35(lmHead(false));
        assertNull(plain.embeddingOutput());
        assertEquals(N_EMBD, plain.embeddingWidth(N_EMBD));
    }

    @Test
    public void qwen35ClsOutputShapesFollowTheLoader() {
        List<GGUFTensorInfo> wide = lmHead(false);
        wide.add(t("cls.output.weight", N_EMBD, N_EMBD_OUT + 1));
        assertThrows(GgufFormatException.class, () -> resolveQwen35(wide));

        List<GGUFTensorInfo> badBias = lmHead(false);
        badBias.add(t("cls.output.weight", N_EMBD, N_EMBD_OUT));
        badBias.add(t("cls.output.bias", N_EMBD));
        assertThrows(GgufFormatException.class, () -> resolveQwen35(badBias));
    }

    @Test
    public void onlyQwen35FormsEmbeddingsWithClsOutput() {
        List<GGUFTensorInfo> ts = lmHead(false);
        ts.add(t("cls.output.weight", N_EMBD, 2));
        HeadSpec h = resolve(Arch.QWEN3, ts);
        assertNull(h.embeddingOutput());
        assertEquals("cls.output.weight", h.clsOutput().name());
    }

    @Test
    public void embeddingOutputContract() {
        TensorRef w = TensorRef.of(t("cls.output.weight", N_EMBD, N_EMBD_OUT));
        TensorRef b = TensorRef.of(t("cls.output.bias", N_EMBD_OUT));
        TensorRef proj = TensorRef.of(t("output.weight", N_EMBD, N_EMBD_OUT));
        assertEquals(N_EMBD_OUT, HeadSpec.builder().embeddingOutput(w, b).build().embeddingWidth(N_EMBD));
        assertThrows(IllegalArgumentException.class, () -> HeadSpec.builder().embeddingOutput(null, b).build());
        assertThrows(IllegalArgumentException.class,
                () -> HeadSpec.builder().embeddingProjection(proj).embeddingOutput(w, null).build());
        TensorRef shortBias = TensorRef.of(t("cls.output.bias", N_EMBD_OUT - 1));
        assertThrows(IllegalArgumentException.class,
                () -> HeadSpec.builder().embeddingOutput(w, shortBias).build());
    }

    @Test
    public void maskedLogitsMustLieInsideTheHead() {
        TensorRef out = TensorRef.of(t("output.weight", N_EMBD, 16));
        assertThrows(IllegalArgumentException.class,
                () -> HeadSpec.builder().output(out, false).maskedLogits(4, 32).build());
    }

    @Test
    public void groupedOutputNormNeedsADivisibleWeight() {
        TensorRef norm = TensorRef.of(t("output_norm.weight", N_EMBD));
        assertEquals(4, HeadSpec.builder().outputNorm(norm, null).outputNormGroups(4).build().outputNormGroups());
        assertThrows(IllegalArgumentException.class,
                () -> HeadSpec.builder().outputNorm(norm, null).outputNormGroups(5).build());
        assertThrows(IllegalArgumentException.class, () -> HeadSpec.builder().outputNormGroups(2).build());
    }
}
