package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The LM-head shape check over a reduced (FR-Spec) drafter head ({@link ModelSpecValidator#checkMtpDraftShapes}):
 * an untied head carries one row per {@code d2t} entry instead of the file's vocabulary
 * (eagle3.cpp:47-50, qwen4exp.cpp:287-296), and without {@code d2t} the vocabulary rule holds.
 */
public class ModelSpecValidatorD2tTest {

    private static final int N_EMBD = 64;
    private static final int N_VOCAB = 32;

    @Test
    public void aReducedHeadHasOneRowPerD2tEntry() {
        HeadSpec head = head(N_VOCAB, N_VOCAB, true);
        ModelSpecValidator.checkMtpDraftShapes(spec(head), N_VOCAB);
    }

    @Test
    public void headRowsOffTheD2tLengthAreRefused() {
        HeadSpec head = head(8, N_VOCAB, true);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSpecValidator.checkMtpDraftShapes(spec(head), N_VOCAB));
        assertTrue(e.getMessage(), e.getMessage().contains("d2t"));
    }

    @Test
    public void withoutD2tTheHeadKeepsTheVocabularyRows() {
        HeadSpec head = head(8, N_VOCAB, false);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSpecValidator.checkMtpDraftShapes(spec(head), N_VOCAB));
        assertTrue(e.getMessage(), e.getMessage().contains("n_vocab"));

        ModelSpecValidator.checkMtpDraftShapes(spec(head(N_VOCAB, N_VOCAB, false)), N_VOCAB);
    }

    /** A head over {@code rows} rows, with {@code d2t} of the target length when {@code withD2t}. */
    private static HeadSpec head(int rows, int d2tEntries, boolean withD2t) {
        TensorRef norm = TensorRef.of("output_norm.weight", GGMLType.F32, N_EMBD);
        TensorRef out = TensorRef.of("output.weight", GGMLType.F32, N_EMBD, rows);
        HeadSpec.Builder b = HeadSpec.builder().outputNorm(norm, null).output(out, false);
        return withD2t ? b.d2t(TensorRef.of("d2t", GGMLType.I64, d2tEntries)).build() : b.build();
    }

    /** An MTP drafter of one attention block over the target's trunk slot. */
    private static ModelSpec spec(HeadSpec head) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.GLM4_MOE)).layerCounts(2, 1)
                .nCtxTrain(4096).nEmbd(N_EMBD).nVocab(N_VOCAB).headKFull(16).headVFull(16).nHead(0, 4)
                .nHeadKv(0, 2).nHead(1, 4).nHeadKv(1, 2).build();
        LayerSpec block = LayerSpec.builder(0, MixerKind.NONE, FfnKind.NONE)
                .norm(NormSpec.builder(NormKind.RMS, 1e-5f).build())
                .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                .build();
        return ModelSpec.builder(hp, TensorCatalog.of(Arch.GLM4_MOE.ggufName(), Collections.emptyList()))
                .schedule(LayerSchedule.linear(1))
                .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                .head(head)
                .layer(block)
                .nextnLayer(LayerSpec.builder(1, MixerKind.NONE, FfnKind.NONE)
                        .norm(NormSpec.builder(NormKind.RMS, 1e-5f).build())
                        .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                        .build())
                .build();
    }
}
