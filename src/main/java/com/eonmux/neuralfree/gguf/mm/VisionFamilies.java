package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The one dispatch over the family graphs both executors run: which encoder graph encodes a tile, which projector
 * graph reads it, and the Granite Vision anyres assembly ({@link CpuVisionEncoder} and {@link GpuVisionEncoder}
 * both delegate here, so the per-family graph description is written once). Package-private executor support.
 */
final class VisionFamilies {

    private VisionFamilies() {
    }

    /**
     * Encodes one image tile to embedding rows, the family's encoder graph and projector graph in sequence, then
     * the separator rows of the image's suffix ({@link ClipSuffix}, the MiniCPM-V 4.6 mergers only).
     */
    static float[][] encode(VisionOps ops, ClipImage image) {
        Objects.requireNonNull(image, "image");
        if (image.suffix() == ClipSuffix.NONE) {
            return encodeTile(ops, image);
        }
        float[][] separators = MiniCpmv46Graph.suffixRows(ops, image.suffix());
        float[][] rows = encodeTile(ops, image);
        float[][] out = Arrays.copyOf(rows, rows.length + separators.length);
        System.arraycopy(separators, 0, out, rows.length, separators.length);
        return out;
    }

    private static float[][] encodeTile(VisionOps ops, ClipImage image) {
        switch (ops.model().encoderGraph()) {
            case ProjectorEntry.ENCODER_CLIP_VIT:
                return project(ops, LlavaGraph.encode(ops, image));
            case ProjectorEntry.ENCODER_SIGLIP_VIT:
                return project(ops, SiglipGraph.encode(ops, image));
            case ProjectorEntry.ENCODER_QWEN2VL_VIT:
                return Qwen2vlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_QWEN3VL_VIT:
                return Qwen3VlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_GRANITE4_VIT:
                return VisionRows.tokenRows(Granite4VisionGraph.encodeTile(ops, image), ops.model().outputDim());
            case ProjectorEntry.ENCODER_MINIMAX_M3_VIT:
                return MinimaxM3Graph.encode(ops, image);
            case ProjectorEntry.ENCODER_LFM3VL_VIT:
                return Lfm3vlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_DEEPSEEK4V_VIT:
                return DeepseekV4Graph.encode(ops, image);
            case ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP:
            case ProjectorEntry.ENCODER_DEEPSEEKOCR2_SAM_QWEN2:
                return DeepseekOcrGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_MOBILENETV5:
                return MobileNetV5Graph.encode(ops, image);
            case ProjectorEntry.ENCODER_YASA2_CONVNEXT:
                return Yasa2Graph.encode(ops, image);
            default:
                return project(ops, encodeFamily(ops, image));
        }
    }

    /**
     * Encodes one anyres image of the Granite Vision 4.1 encoder: the overview tile, then the {@code gridX x
     * gridY} tiles of the resized image in raster order, each an {@code image_size} square; the rows are the
     * overview's tokens, then the tile grid cropped to the original aspect ratio with a newline row after every
     * row ({@code build_anyres_assembly}).
     *
     * @throws IllegalArgumentException when the encoder is not granite4_vision, the tile count is not
     *                                  {@code gridX * gridY}, or a size is not positive (named)
     */
    static float[][] encodeAnyres(VisionOps ops, ClipImage overview, List<ClipImage> tiles, int gridX, int gridY,
                                  int origWidth, int origHeight) {
        Objects.requireNonNull(overview, "overview");
        Objects.requireNonNull(tiles, "tiles");
        if (!ProjectorEntry.ENCODER_GRANITE4_VIT.equals(ops.model().encoderGraph())) {
            throw new IllegalArgumentException("anyres tile assembly is a granite4_vision graph; this encoder is "
                    + ops.hp().projectorType());
        }
        if (gridX <= 0 || gridY <= 0 || origWidth <= 0 || origHeight <= 0) {
            throw new IllegalArgumentException("anyres grid " + gridX + "x" + gridY + " and original size "
                    + origWidth + "x" + origHeight + " must be positive");
        }
        if (tiles.size() != gridX * gridY) {
            throw new IllegalArgumentException("a " + gridX + "x" + gridY + " anyres grid needs " + gridX * gridY
                    + " tiles, got " + tiles.size());
        }
        List<float[]> encoded = new java.util.ArrayList<>(tiles.size());
        for (ClipImage tile : tiles) {
            encoded.add(Granite4VisionGraph.encodeTile(ops, Objects.requireNonNull(tile, "tile")));
        }
        return Granite4VisionGraph.assemble(Granite4VisionGraph.encodeTile(ops, overview), encoded,
                Granite4VisionGraph.outSide(ops.hp(), ops.model().nPatchesPerSide()), ops.model().outputDim(),
                gridX, gridY, origWidth, origHeight, ops.hostVector("v.image_newline"));
    }

    private static Encoded encodeFamily(VisionOps ops, ClipImage image) {
        switch (ops.model().encoderGraph()) {
            case ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT:
                return SiglipResizedGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_MINICPMV_VIT:
                return MiniCpmvGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_INTERNVL_VIT:
                return InternVlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_PIXTRAL_VIT:
                return PixtralGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_LLAMA4_VIT:
                return Llama4Graph.encode(ops, image);
            case ProjectorEntry.ENCODER_PADDLEOCR_VIT:
                return PaddleOcrGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_KIMIVL_VIT:
                return KimiVlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_KIMIK25_VIT:
                return KimiVlGraph.encodeK25(ops, image);
            case ProjectorEntry.ENCODER_DOTS_VIT:
                return DotsOcrGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_HUNYUANVL_VIT:
                return HunyuanVlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_COGVLM_VIT:
                return CogVlmGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_GLM4V_VIT:
                return Glm4vGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_YOUTUVL_VIT:
                return YoutuVlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_NEMOTRON_VIT:
                return NemotronVlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_MUSE_GLIMMER_VIT:
                return MuseGlimmerGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_SENSENOVA_PATCH:
                return SenseNovaGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_STEP3VL_VIT:
                return Step3vlGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_GEMMA4V_VIT:
                return Gemma4vGraph.encode(ops, image);
            case ProjectorEntry.ENCODER_GEMMA4UV:
                return Gemma4vGraph.encodeUnified(ops, image);
            case ProjectorEntry.ENCODER_MINICPMV46_VIT:
                return MiniCpmv46Graph.encode(ops, image);
            case ProjectorEntry.ENCODER_MIMOVL_VIT:
                return MimoVlGraph.encode(ops, image);
            default:
                throw new IllegalStateException("unknown encoder graph " + ops.model().encoderGraph());
        }
    }

    private static float[][] project(VisionOps ops, Encoded enc) {
        switch (ops.model().projectorGraph()) {
            case ProjectorEntry.PROJECTOR_LLAVA_MLP:
                return LlavaGraph.mlp(ops, enc);
            case ProjectorEntry.PROJECTOR_LLAVA_MLP_NORM:
                return LlavaGraph.mlpNorm(ops, enc);
            case ProjectorEntry.PROJECTOR_GEMMA3:
                return SiglipGraph.gemma3(ops, enc);
            case ProjectorEntry.PROJECTOR_IDEFICS3:
                return SiglipProjectors.idefics3(ops, enc);
            case ProjectorEntry.PROJECTOR_PHI4:
                return mlp(ops, enc, "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias", ActOp.GELU_TANH);
            case ProjectorEntry.PROJECTOR_MINICPMV_RESAMPLER:
                return MiniCpmvGraph.resample(ops, enc);
            case ProjectorEntry.PROJECTOR_INTERNVL:
                return InternVlGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_PIXTRAL:
            case ProjectorEntry.PROJECTOR_LIGHTONOCR:
                return PixtralGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_LLAMA4:
                return Llama4Graph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_PADDLEOCR:
                return PaddleOcrGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_KIMIVL:
                return KimiVlGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_KIMIK25:
                return KimiVlGraph.projectK25(ops, enc);
            case ProjectorEntry.PROJECTOR_GEMMA4V:
                return Gemma4vGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_GEMMA4UV:
                return Gemma4vGraph.projectUnified(ops, enc);
            case ProjectorEntry.PROJECTOR_MINICPMV46_MERGER:
                return MiniCpmv46Graph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_DOTS:
                return DotsOcrGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_HUNYUANVL:
                return HunyuanVlGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_JANUS:
                return SiglipProjectors.janus(ops, enc);
            case ProjectorEntry.PROJECTOR_LFM2:
                return SiglipProjectors.lfm2(ops, enc);
            case ProjectorEntry.PROJECTOR_COHERE2V:
                return SiglipProjectors.cohere2v(ops, enc);
            default:
                return projectFamily(ops, enc);
        }
    }

    private static float[][] projectFamily(VisionOps ops, Encoded enc) {
        switch (ops.model().projectorGraph()) {
            case ProjectorEntry.PROJECTOR_LDP:
                return MobileVlmGraph.ldp(ops, enc);
            case ProjectorEntry.PROJECTOR_LDPV2:
                return MobileVlmGraph.ldpv2(ops, enc);
            case ProjectorEntry.PROJECTOR_GLM_EDGE:
                return LlavaGraph.glmEdge(ops, enc);
            case ProjectorEntry.PROJECTOR_COGVLM:
                return CogVlmGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_GLM4V:
                return Glm4vGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_YOUTUVL:
                return YoutuVlGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_NEMOTRON:
                return NemotronVlGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_MUSE_GLIMMER:
                return MuseGlimmerGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_SENSENOVA:
                return SenseNovaGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_STEP3VL:
                return Step3vlGraph.project(ops, enc);
            case ProjectorEntry.PROJECTOR_MIMOVL_MERGER:
                return MimoVlGraph.project(ops, enc);
            default:
                throw new IllegalStateException("projector graph " + ops.model().projectorGraph()
                        + " does not run on the " + ops.model().encoderGraph() + " encoder");
        }
    }

    /** {@code build_ffn} of a projector without gate: {@code down(act(up(x) + upB)) + downB}. */
    private static float[][] mlp(VisionOps ops, Encoded enc, String up, String upB, String down, String downB,
                                 ActOp act) {
        VisionOps.Block h = ops.linear(up, ops.upload(enc.rows, enc.width), upB);
        ops.activate(h, act);
        VisionOps.Block out = ops.linear(down, h, downB);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }
}
