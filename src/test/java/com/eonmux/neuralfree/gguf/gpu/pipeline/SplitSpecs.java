package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.List;

/**
 * Small llama-shaped {@link ModelSpec} fixtures for the split tests: a dense trunk whose layers own their KV, with
 * per-layer overrides for the shared-KV reuse of gemma3n / gemma4, a looped schedule and an encoder-decoder support
 * class.
 */
public final class SplitSpecs {

    /** Fixture dimensions: two heads of 32 over a 64-wide stream, vocab 32. */
    public static final int N_EMBD = 64;
    public static final int N_FF = 128;
    public static final int VOCAB = 32;
    public static final int N_HEAD = 2;
    public static final int HEAD_DIM = 32;

    private SplitSpecs() {
    }

    /** A dense trunk of {@code nLayer} layers; {@code kind} NONE keeps every fixture memory-free unless asked. */
    public static ModelSpec trunk(int nLayer, MemoryKind kind) {
        return build(hparams(Arch.LLAMA, nLayer).build(), tensors(nLayer), nLayer, kind,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(resolved)
                        .residual(standardResidual())
                        .attention(attention(il, il))
                        .ffn(denseFfn())
                        .build(),
                LayerSchedule.linear(nLayer), nLayer);
    }

    /** A trunk where layer {@code reader} reads the KV cache of layer {@code owner} (gemma3n / gemma4 reuse). */
    public static ModelSpec sharedKv(int nLayer, int reader, int owner) {
        return sharedKv(nLayer, il -> il == reader ? owner : il);
    }

    /** A trunk where every layer from 1 on reads its predecessor: no boundary can separate the chain. */
    public static ModelSpec sharedKvChain(int nLayer) {
        return sharedKv(nLayer, il -> Math.max(0, il - 1));
    }

    private static ModelSpec sharedKv(int nLayer, java.util.function.IntUnaryOperator ownerOf) {
        return build(hparams(Arch.LLAMA, nLayer).build(), tensors(nLayer), nLayer, MemoryKind.KV,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(resolved)
                        .residual(standardResidual())
                        .attention(attention(il, ownerOf.applyAsInt(il)))
                        .ffn(denseFfn())
                        .build(),
                LayerSchedule.linear(nLayer), nLayer);
    }

    /** A looped trunk: {@code nLayer} slots run over {@code nLoops} passes of the same weights. */
    public static ModelSpec looped(int nLayer, int nLoops) {
        int physical = nLayer / nLoops;
        return build(hparams(Arch.LLAMA, physical).build(), tensors(physical), nLayer, MemoryKind.KV,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .weightLayer(il % physical)
                        .norm(resolved)
                        .residual(standardResidual())
                        .attention(attention(il % physical, il % physical))
                        .ffn(denseFfn())
                        .build(),
                LayerSchedule.looped(physical, nLoops, true), physical);
    }

    /** An encoder-decoder model (T5 traits), the shape a split refuses before anything else. */
    public static ModelSpec encDec(int nLayer) {
        ModelHparams hp = new ModelHparamsBuilder(ArchRegistry.traits(Arch.T5)).layerCounts(nLayer, 0)
                .nCtxTrain(512).nEmbd(N_EMBD).nVocab(VOCAB).normRmsEps(1e-6f).normEps(1e-5f)
                .headKFull(HEAD_DIM).headVFull(HEAD_DIM).build();
        return build(hp, tensors(nLayer), nLayer, MemoryKind.NONE,
                (il, resolved) -> LayerSpec.builder(il, MixerKind.ATTENTION, FfnKind.DENSE)
                        .norm(resolved)
                        .residual(standardResidual())
                        .attention(attention(il, il))
                        .ffn(denseFfn())
                        .build(),
                LayerSchedule.linear(nLayer), nLayer);
    }

    private static ModelSpec build(ModelHparams hp, List<GGUFTensorInfo> tensors, int nLayer, MemoryKind kind,
                                   LayerShape shape, LayerSchedule schedule, int normSlots) {
        TensorCatalog tc = TensorCatalog.of(hp.arch().ggufName(), tensors);
        com.eonmux.neuralfree.gguf.spec.resolve.NormResolver norms =
                new com.eonmux.neuralfree.gguf.spec.resolve.NormResolver(hp, tc);
        HeadSpec.Builder head = HeadSpec.builder()
                .outputNorm(tc.get(TensorNames.weight(TensorNames.OUTPUT_NORM)), null)
                .output(tc.get(TensorNames.weight(TensorNames.OUTPUT)), false);
        ModelSpec.Builder b = ModelSpec.builder(hp, tc)
                .schedule(schedule)
                .embedding(EmbeddingSpec.builder(tc.get(TensorNames.weight(TensorNames.TOKEN_EMBD)), N_EMBD)
                        .embdScale(1.0f).build())
                .head(head.build());
        for (int il = 0; il < nLayer; il++) {
            b.layer(shape.layer(il, norms.resolve(il % normSlots)));
        }
        if (kind != null) {
            b.memoryKind(kind);
        }
        return b.build();
    }

    private static ModelHparamsBuilder hparams(Arch arch, int nLayer) {
        ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(arch)).layerCounts(nLayer, 0)
                .nCtxTrain(4096).nEmbd(N_EMBD).nVocab(VOCAB).normRmsEps(1e-6f).normEps(1e-5f)
                .headKFull(HEAD_DIM).headVFull(HEAD_DIM);
        for (int il = 0; il < nLayer; il++) {
            hb.nHead(il, N_HEAD).nHeadKv(il, N_HEAD);
        }
        return hb;
    }

    private static AttentionSpec attention(int il, int kvOwner) {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_DIM).headDimV(HEAD_DIM)
                .noRope(true).nRot(0).kqScale(1.0f / (float) Math.sqrt(HEAD_DIM)).kvOwnerLayer(kvOwner).build();
    }

    private static FfnSpec denseFfn() {
        return FfnSpec.builder(FfnSpec.Role.DENSE).nFf(N_FF).build();
    }

    private static ResidualSpec standardResidual() {
        return ResidualSpec.builder(ResidualKindHolder.KIND, N_EMBD).build();
    }

    /** Token table, output norm and head, and per layer the two norms and a gated FFN. */
    private static List<GGUFTensorInfo> tensors(int nLayer) {
        List<GGUFTensorInfo> t = new ArrayList<>();
        t.add(info(TensorNames.weight(TensorNames.TOKEN_EMBD), N_EMBD, VOCAB));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT_NORM), N_EMBD));
        t.add(info(TensorNames.weight(TensorNames.OUTPUT), N_EMBD, VOCAB));
        for (int il = 0; il < nLayer; il++) {
            t.add(info(blk(il, TensorNames.ATTN_NORM), N_EMBD));
            t.add(info(blk(il, TensorNames.FFN_NORM), N_EMBD));
            t.add(info(blk(il, TensorNames.FFN_GATE), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_UP), N_EMBD, N_FF));
            t.add(info(blk(il, TensorNames.FFN_DOWN), N_FF, N_EMBD));
        }
        return t;
    }

    private static String blk(int il, String base) {
        return TensorNames.layerName(TensorNames.STACK_BLK, il, TensorNames.weight(base));
    }

    private static GGUFTensorInfo info(String name, long... dims) {
        return new GGUFTensorInfo(name, dims, GGMLType.F32, 0);
    }

    /** One layer built from its resolved norm. */
    private interface LayerShape {
        LayerSpec layer(int il, com.eonmux.neuralfree.gguf.spec.NormSpec resolved);
    }

    /** Indirection so the fixture file reads without a second import block for {@link ResidualSpec}. */
    private static final class ResidualKindHolder {
        static final com.eonmux.neuralfree.gguf.arch.ResidualKind KIND =
                com.eonmux.neuralfree.gguf.arch.ResidualKind.STANDARD;
    }
}
