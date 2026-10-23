package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * {@link MemoryConfig#forModel}: the one ubatch rule of the CPU and GPU orchestrators. Models that evaluate their whole
 * input in one step (an encoder-decoder source, the WavTokenizer codes) get a ubatch of their whole context, the
 * training context when the context is automatic and the user's context otherwise; a model without memory takes at
 * most its context in one step; every other model keeps its settings.
 */
public class MemoryConfigForModelTest {

    private static final int TRAIN = 8192;

    @Test
    public void generativeModelsKeepTheirSettings() {
        MemoryConfig mc = MemoryConfig.defaults();
        ModelSpec gemma3 = MemoryPlannerTest.gemma3();
        assertFalse(MemoryConfig.singleStepInput(gemma3));
        assertSame(mc, mc.forModel(gemma3));
    }

    @Test
    public void encoderDecoderTakesTheTrainingContextWhenAutomatic() {
        ModelSpec t5 = singleLayer(Arch.T5);
        assertEquals(SupportClass.ENC_DEC, t5.supportClass());
        assertTrue(MemoryConfig.singleStepInput(t5));
        assertEquals(TRAIN, MemoryConfig.defaults().forModel(t5).ubatch());
    }

    @Test
    public void wavTokenizerTakesTheUserContext() {
        ModelSpec wav = singleLayer(Arch.WAVTOKENIZER_DEC);
        assertTrue(MemoryConfig.singleStepInput(wav));
        MemoryConfig mc = MemoryConfig.builder().nCtx(2048).build();
        MemoryConfig run = mc.forModel(wav);
        assertEquals(2048, run.ubatch());
        assertEquals(2048, run.nCtx());
    }

    @Test
    public void aLargerUbatchIsKeptWithinTheContextOfAModelWithMemory() {
        ModelSpec t5 = singleLayer(Arch.T5);
        assertTrue(t5.memoryKind() != MemoryKind.NONE);
        MemoryConfig mc = MemoryConfig.builder().nCtx(256).build();
        assertSame(mc, mc.forModel(t5));
        assertEquals(MemoryConfig.DEFAULT_UBATCH, mc.forModel(t5).ubatch());
    }

    @Test
    public void aModelWithoutMemoryTakesAtMostItsContextInOneStep() {
        ModelSpec wav = singleLayer(Arch.WAVTOKENIZER_DEC);
        assertEquals(MemoryKind.NONE, wav.memoryKind());
        MemoryConfig user = MemoryConfig.builder().nCtx(256).build();
        assertEquals(256, user.memorylessContext(wav));
        assertEquals(256, user.forModel(wav).ubatch());
        MemoryConfig auto = MemoryConfig.builder().ubatch(2 * TRAIN).build();
        assertEquals(TRAIN, auto.memorylessContext(wav));
        assertEquals(TRAIN, auto.forModel(wav).ubatch());
    }

    @Test
    public void withoutATrainingContextTheUbatchBoundsAModelWithoutMemory() {
        ModelSpec wav = singleLayer(Arch.WAVTOKENIZER_DEC, 0);
        MemoryConfig mc = MemoryConfig.defaults();
        assertEquals(MemoryConfig.DEFAULT_UBATCH, mc.memorylessContext(wav));
        assertSame(mc, mc.forModel(wav));
    }

    @Test
    public void otherSettingsAreKept() {
        ModelSpec t5 = singleLayer(Arch.T5);
        MemoryConfig mc = MemoryConfig.builder().typeK(KvDType.Q8_0).typeV(KvDType.Q8_0).kvRotation(false)
                .stateSnapshots(true).build();
        MemoryConfig run = mc.forModel(t5);
        assertEquals(KvDType.Q8_0, run.typeK());
        assertEquals(KvDType.Q8_0, run.typeV());
        assertFalse(run.kvRotation());
        assertTrue(run.stateSnapshots());
        assertTrue(run.autoContext());
    }

    private static ModelSpec singleLayer(Arch arch) {
        return singleLayer(arch, TRAIN);
    }

    private static ModelSpec singleLayer(Arch arch, int nCtxTrain) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(1, 0).nCtxTrain(nCtxTrain)
                .headKFull(64).headVFull(64).nHead(0, 4).nHeadKv(0, 4).build();
        AttentionSpec attention = AttentionSpec.builder().nHead(4).nHeadKv(4).headDimK(64).headDimV(64).noRope(true)
                .nRot(0).kqScale(0.125f).kvOwnerLayer(0).build();
        LayerSpec layer = LayerSpec.builder(0, MixerKind.ATTENTION, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-6f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, 256).build())
                .attention(attention)
                .build();
        return ModelSpec.builder(hp, TensorCatalog.of(arch.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, 256).build())
                .head(HeadSpec.builder().build())
                .layer(layer)
                .build();
    }
}
