package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkNorm;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkPair;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * Validated geometry of one vision encoder and its projector over an mmproj tensor table: the counterpart of
 * llama.cpp {@code clip_model} (tools/mtmd/clip-model.h) resolved for one family. Every tensor the family's
 * graphs read is resolved by name and shape-checked against the {@link ClipHparams} exactly where the reference
 * loader checks it; a mismatch fails at construction with the tensor name, before any weight is loaded. The
 * encoder blocks are {@link ClipLayer}s, the shared {@code build_vit} block of every clip encoder.
 * Immutable; tensor references view the file, the executor copies values.
 */
public final class ClipVisionModel {

    private final ClipHparams hparams;
    private final String encoderGraph;
    private final String projectorGraph;
    private final int nPatchesX;
    private final int nPatchesY;
    private final int nPatches;
    private final int nPos;
    private final int dHead;
    private final float attnScale;
    private final int patchKernel;

    private final TensorRef patchEmbd;
    private final TensorRef patchEmbd1;
    private final TensorRef patchBias;
    private final TensorRef normEmbdW;
    private final TensorRef normEmbdB;
    private final TensorRef classEmbd;
    private final TensorRef posEmbd;
    private final TensorRef preLnW;
    private final TensorRef preLnB;
    private final TensorRef postLnW;
    private final TensorRef postLnB;
    private final List<ClipLayer> layers;
    private final ClipProjectorTensors projector;

    private ClipVisionModel(ClipHparams hparams, String encoderGraph, String projectorGraph, int nPatchesSide,
                            int nPos, int dHead, Encoder enc, List<ClipLayer> layers, ClipProjectorTensors projector) {
        this.hparams = hparams;
        this.encoderGraph = encoderGraph;
        this.projectorGraph = projectorGraph;
        this.nPatchesX = nPatchesSide;
        this.nPatchesY = nPatchesSide;
        this.nPatches = nPatchesSide * nPatchesSide;
        this.nPos = nPos;
        this.dHead = dHead;
        this.attnScale = dHead > 0 ? (float) (1.0 / Math.sqrt(dHead)) : 0f;
        this.patchKernel = patchKernelSize(hparams, encoderGraph);
        this.patchEmbd = enc.patchEmbd;
        this.patchEmbd1 = enc.patchEmbd1;
        this.patchBias = enc.patchBias;
        this.normEmbdW = enc.normEmbdW;
        this.normEmbdB = enc.normEmbdB;
        this.classEmbd = enc.classEmbd;
        this.posEmbd = enc.posEmbd;
        this.preLnW = enc.preLnW;
        this.preLnB = enc.preLnB;
        this.postLnW = enc.postLnW;
        this.postLnB = enc.postLnB;
        this.layers = layers;
        this.projector = projector;
    }

    /** The encoder-level tensors of one family. */
    private static final class Encoder {
        TensorRef patchEmbd;
        TensorRef patchEmbd1;
        TensorRef patchBias;
        TensorRef normEmbdW;
        TensorRef normEmbdB;
        TensorRef classEmbd;
        TensorRef posEmbd;
        TensorRef preLnW;
        TensorRef preLnB;
        TensorRef postLnW;
        TensorRef postLnB;
    }

    /**
     * Geometry of {@code hp} over the tensors of {@code tc}.
     *
     * @throws IllegalArgumentException for a null argument, an hparams modality other than
     *                                  {@link ProjectorModality#VISION}, a projector type without a runnable
     *                                  registry row, or a tensor shape that differs from the loader's
     * @throws MissingTensorException   when a required tensor is absent
     */
    public static ClipVisionModel from(ClipHparams hp, TensorCatalog tc) {
        Objects.requireNonNull(hp, "hparams");
        Objects.requireNonNull(tc, "tensors");
        if (hp.modality() != ProjectorModality.VISION) {
            throw new IllegalArgumentException("ClipVisionModel resolves the vision encoder, got "
                    + hp.modality());
        }
        ProjectorEntry entry = ProjectorRegistry.get(hp.projectorType().typeName());
        entry.tensors().validate(tc, hp.nLayer());

        String graph = entry.encoderGraph();
        int patchGrid = hp.imageSize() / hp.patchSize();
        Encoder enc = new Encoder();
        boolean mobilenet = ProjectorEntry.ENCODER_MOBILENETV5.equals(graph);
        boolean sam = DeepseekOcrGraph.runsSam(graph);
        if (!mobilenet && !sam) {
            patchEmbedding(tc, hp, graph, enc);
        }
        enc.classEmbd = classEmbd(tc, hp, graph);
        int nPos;
        if (mobilenet) {
            // the MobileNetV5 conv stack runs no position rows; its graph sizes its own rows
            nPos = patchGrid * patchGrid;
        } else if (sam) {
            // the tower reads the SAM rows (a quarter of the patch grid per side), not v.patch_embd
            nPos = DeepseekOcrGraph.towerTokens(graph, patchGrid / DeepseekOcrGraph.SAM_DOWNSAMPLE);
        } else {
            nPos = patchGrid * patchGrid + (enc.classEmbd != null ? (int) enc.classEmbd.ne(1) : 0);
        }
        enc.posEmbd = posEmbd(tc, hp, graph, nPos, patchGrid * patchGrid);
        norms(tc, hp, graph, enc);
        ClipLayer.Spec spec = layerSpec(tc, hp, graph);
        // the MobileNetV5 encoder has no ViT blocks and LFM3-VL's blocks its projector graph resolves; their
        // converters write a placeholder block count
        boolean vitBlocks = !mobilenet && !ProjectorEntry.ENCODER_LFM3VL_VIT.equals(graph);
        int nBlocks = vitBlocks ? hp.nLayer() : 0;
        List<ClipLayer> layers = new ArrayList<>(nBlocks);
        for (int il = 0; il < nBlocks; il++) {
            ClipLayer layer = ClipLayer.resolve(tc, hp, il, spec);
            if (ProjectorEntry.ENCODER_YOUTUVL_VIT.equals(graph) && layer.ffnGateWeight() != null) {
                throw new IllegalArgumentException("architecture '" + tc.archName() + "': the youtuvl encoder"
                        + " builds an ungated FFN, but the file carries " + layer.ffnGateWeight().name());
            }
            if (hp.projectorType() == ProjectorType.EXAONE4_5 && !layer.fusedQkv()) {
                throw new MissingTensorException(tc.archName(), il, "v.blk." + il + ".attn_qkv.weight");
            }
            layers.add(layer);
        }
        if (ProjectorEntry.ENCODER_MIMOVL_VIT.equals(graph)) {
            requireWindowSinks(tc, hp, layers);
        }
        ClipProjectorTensors proj = ClipProjectorTensors.resolve(tc, hp, entry.projectorGraph());
        int dHead = spec.dHead > 0 ? spec.dHead : hp.dHead();
        return new ClipVisionModel(hp, graph, entry.projectorGraph(), patchGrid, nPos, dHead, enc,
                Collections.unmodifiableList(layers), proj);
    }

    /**
     * The geometry of one tower of a CLIP two-tower file ({@link ClipHparams#twoTower}): the {@code family}'s ViT
     * blocks without a projector ({@link ProjectorEntry#PROJECTOR_NONE}), for the executors that run the blocks and
     * norms over the rows the dual encoder assembles. The vision tower reads the patch kernel and bias, the class
     * token of a {@link ClipFamily#CLIP} tower, the position rows and the pre/post norms; the text tower
     * ({@code t.*}) its {@code nCtx} position rows and final norm, and no patch kernel.
     *
     * @throws IllegalArgumentException when {@code hp} is no two-tower tower, or a shape differs from the tower's
     * @throws MissingTensorException   when a required tensor is absent
     */
    public static ClipVisionModel twoTower(ClipHparams hp, TensorCatalog tc, ClipFamily family, int nCtx) {
        Objects.requireNonNull(hp, "hparams");
        Objects.requireNonNull(tc, "tensors");
        Objects.requireNonNull(family, "family");
        if (!hp.twoTower()) {
            throw new IllegalArgumentException("ClipVisionModel.twoTower resolves a two-tower tower, got projector"
                    + " type " + hp.projectorType());
        }
        String graph = family == ClipFamily.SIGLIP ? ProjectorEntry.ENCODER_SIGLIP_VIT
                : ProjectorEntry.ENCODER_CLIP_VIT;
        Encoder enc = new Encoder();
        int side = 0;
        int nPos;
        if (hp.modality() == ProjectorModality.VISION) {
            side = hp.imageSize() / hp.patchSize();
            enc.patchEmbd = tc.require("v.patch_embd.weight");
            shape(tc, enc.patchEmbd, hp.patchSize(), hp.patchSize(), 3, hp.nEmbd());
            enc.patchBias = tc.get("v.patch_embd.bias");
            checkVector(tc, enc.patchBias, hp.nEmbd());
            enc.classEmbd = family == ClipFamily.CLIP ? tc.require("v.class_embd") : null;
            checkClass(tc, enc.classEmbd, hp.nEmbd());
            nPos = side * side + (enc.classEmbd != null ? 1 : 0);
            enc.posEmbd = tc.require("v.position_embd.weight");
            shape(tc, enc.posEmbd, hp.nEmbd(), nPos);
            norms(tc, hp, graph, enc);
        } else if (hp.modality() == ProjectorModality.TEXT) {
            nPos = nCtx;
            enc.posEmbd = tc.require(ClipTextTower.POSITION_EMBD);
            shape(tc, enc.posEmbd, hp.nEmbd(), nCtx);
            enc.postLnW = tc.get(ClipTextTower.POST_LN + ".weight");
            enc.postLnB = tc.get(ClipTextTower.POST_LN + ".bias");
            checkNorm(tc, ClipTextTower.POST_LN, enc.postLnW, enc.postLnB, hp.nEmbd());
        } else {
            throw new IllegalArgumentException("a two-tower file holds text and vision towers, not "
                    + hp.modality());
        }
        List<ClipLayer> layers = new ArrayList<>(hp.nLayer());
        for (int il = 0; il < hp.nLayer(); il++) {
            layers.add(ClipLayer.resolve(tc, hp, il, false));
        }
        return new ClipVisionModel(hp, graph, ProjectorEntry.PROJECTOR_NONE, side, nPos, hp.dHead(), enc,
                Collections.unmodifiableList(layers), ClipProjectorTensors.none(hp.nEmbd()));
    }

    /**
     * What the blocks of {@code graph} read: the build_vit extras, the Gemma 4 clippable-linear bounds, and the
     * MiMo-VL attention sinks with the head width of its fused Q/K/V.
     */
    private static ClipLayer.Spec layerSpec(TensorCatalog tc, ClipHparams hp, String graph) {
        if (ProjectorEntry.ENCODER_MIMOVL_VIT.equals(graph)) {
            return ClipLayer.Spec.PLAIN.withHeadDim(fusedHeadDim(tc, hp)).withSinks();
        }
        if (ProjectorEntry.ENCODER_GEMMA4V_VIT.equals(graph)) {
            return ClipLayer.Spec.EXTRAS.withClamps();
        }
        if (ProjectorEntry.ENCODER_MINICPMV46_VIT.equals(graph)) {
            return ClipLayer.Spec.LAYER_SCALES;
        }
        return MmprojTensorTable.readsVitExtras(graph) ? ClipLayer.Spec.EXTRAS : ClipLayer.Spec.PLAIN;
    }

    /**
     * The MiMo-VL head width: the fused Q/K/V rows of block 0 over {@code n_head + 2 * n_head_kv} heads
     * (models/mimovl.cpp derives it so; {@code n_embd / n_head} need not be it), equal to
     * {@code attention.head_dim} when the file writes that key.
     */
    private static int fusedHeadDim(TensorCatalog tc, ClipHparams hp) {
        TensorRef qkv = tc.require("v.blk.0.attn_qkv.weight");
        long heads = hp.nHead() + 2L * hp.nHeadKv();
        long rows = qkv.ne(1);
        if (rows <= 0 || rows % heads != 0 || rows / heads > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + qkv.name()
                    + " has " + rows + " rows, not a whole head width over " + hp.nHead() + " query and "
                    + hp.nHeadKv() + " key/value heads");
        }
        int dh = (int) (rows / heads);
        if (hp.nEmbdHead() != 0 && hp.nEmbdHead() != dh) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': clip.vision.attention.head_dim"
                    + " is " + hp.nEmbdHead() + ", but " + qkv.name() + " holds heads of " + dh);
        }
        return dh;
    }

    /** Every MiMo-VL block that attends within a window carries its per-head {@code attn_sinks}. */
    private static void requireWindowSinks(TensorCatalog tc, ClipHparams hp, List<ClipLayer> layers) {
        VisionFamilyHparams fam = hp.visionFamily();
        for (int il = 0; il < layers.size(); il++) {
            if (fam.attnMode(il) != VisionFamilyHparams.MODE_FULL && layers.get(il).attnSinks() == null) {
                throw new MissingTensorException(tc.archName(), il, "v.blk." + il + ".attn_sinks");
            }
        }
    }

    /**
     * The side of the pixel square one patch row reads: {@code patch_size}, or {@code patch_size * n_merge} for the
     * Gemma 4 unified embedder, whose patch projection also merges (clip.cpp load_hparams).
     */
    private static int patchKernelSize(ClipHparams hp, String graph) {
        return ProjectorEntry.ENCODER_GEMMA4UV.equals(graph) ? hp.patchSize() * hp.nMerge() : hp.patchSize();
    }

    /**
     * The patch embedding: a linear layer over the unfolded patch for Llama 4, Youtu-VL and the Gemma 4 unified
     * embedder, a conv kernel for the others. The qwen encoders add the optional second frame of their Conv3d
     * kernel, GLM-4V its required second (temporal) conv kernel and the optional embedding norm.
     */
    private static void patchEmbedding(TensorCatalog tc, ClipHparams hp, String graph, Encoder enc) {
        enc.patchEmbd = tc.require("v.patch_embd.weight");
        int ps = hp.patchSize();
        if (ProjectorEntry.ENCODER_GEMMA4UV.equals(graph)) {
            // patch_dense: a linear layer over (c, ky, kx) vectors of the merged patch, with its bias
            int k = patchKernelSize(hp, graph);
            shape(tc, enc.patchEmbd, 3L * k * k, hp.nEmbd());
            enc.patchBias = tc.require("v.patch_embd.bias");
            checkVector(tc, enc.patchBias, hp.nEmbd());
            return;
        }
        if (ProjectorEntry.ENCODER_LLAMA4_VIT.equals(graph)) {
            // Llama4UnfoldConvolution: a linear layer over the unfolded patch
            shape(tc, enc.patchEmbd, 3L * ps * ps, hp.nEmbd());
            return;
        }
        if (ProjectorEntry.ENCODER_YASA2_CONVNEXT.equals(graph)) {
            // the ConvNeXt stem: its width is the first stage's, not n_embd
            long ch = enc.patchEmbd.ne(3);
            shape(tc, enc.patchEmbd, ps, ps, 3, ch);
            enc.patchBias = tc.get("v.patch_embd.bias");
            if (enc.patchBias != null) {
                shape(tc, enc.patchBias, ch);
            }
            return;
        }
        if (ProjectorEntry.ENCODER_MIMOVL_VIT.equals(graph)) {
            // the two frames of the Conv3d patch embedding, both bias-free (models/mimovl.cpp)
            shape(tc, enc.patchEmbd, ps, ps, 3, hp.nEmbd());
            enc.patchEmbd1 = tc.require("v.patch_embd.weight.1");
            shape(tc, enc.patchEmbd1, ps, ps, 3, hp.nEmbd());
            refuse(tc, graph, "v.patch_embd.bias", "adds no patch bias");
            return;
        }
        if (ProjectorEntry.ENCODER_YOUTUVL_VIT.equals(graph)) {
            // Siglip2 NaFlex patch embedding: nn.Linear over (ky, kx, c) patch vectors
            shape(tc, enc.patchEmbd, 3L * ps * ps, hp.nEmbd());
        } else {
            shape(tc, enc.patchEmbd, ps, ps, 3, hp.nEmbd());
        }
        enc.patchBias = tc.get("v.patch_embd.bias");
        checkVector(tc, enc.patchBias, hp.nEmbd());
        if (ProjectorEntry.ENCODER_GEMMA4V_VIT.equals(graph) && enc.patchBias != null) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the gemma4v encoder adds no"
                    + " patch bias, but the file carries " + enc.patchBias.name());
        }
        if (readsSecondFrameKernel(graph)) {
            // the second frame of the Conv3d patch embedding (build_inp_temporal_merge); a still image runs both
            enc.patchEmbd1 = tc.get("v.patch_embd.weight.1");
            if (enc.patchEmbd1 != null) {
                shape(tc, enc.patchEmbd1, ps, ps, 3, hp.nEmbd());
            }
        }
        patchKernelRules(tc, hp, graph, enc);
        if (ProjectorEntry.ENCODER_GLM4V_VIT.equals(graph)) {
            enc.patchEmbd1 = tc.require("v.patch_embd.weight.1");
            shape(tc, enc.patchEmbd1, ps, ps, 3, hp.nEmbd());
            if (enc.patchBias == null) {
                throw new MissingTensorException(tc.archName(), -1, "v.patch_embd.bias");
            }
            enc.normEmbdW = tc.get("v.norm_embd.weight");
            enc.normEmbdB = tc.get("v.norm_embd.bias");
            checkNorm(tc, "v.norm_embd", enc.normEmbdW, enc.normEmbdB, hp.nEmbd());
        }
    }

    /** Refuses a tensor the graph does not apply, by name. */
    private static void refuse(TensorCatalog tc, String graph, String name, String why) {
        if (tc.has(name)) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the " + graph + " encoder "
                    + why + ", but the file carries " + name);
        }
    }

    private static boolean readsSecondFrameKernel(String graph) {
        return ProjectorEntry.ENCODER_QWEN2VL_VIT.equals(graph) || ProjectorEntry.ENCODER_QWEN3VL_VIT.equals(graph)
                || ProjectorEntry.ENCODER_MINIMAX_M3_VIT.equals(graph) || ProjectorEntry.ENCODER_LFM3VL_VIT.equals(graph);
    }

    /**
     * The per-family patch kernel rules: MiniMax-M3 and EXAONE 4.5 run both kernels and assert no patch bias,
     * LFM3-VL's patchifier convolutions add no bias.
     */
    private static void patchKernelRules(TensorCatalog tc, ClipHparams hp, String graph, Encoder enc) {
        boolean bothKernels = ProjectorEntry.ENCODER_MINIMAX_M3_VIT.equals(graph)
                || hp.projectorType() == ProjectorType.EXAONE4_5;
        if (bothKernels && enc.patchBias != null) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the " + hp.projectorType()
                    + " encoder has no patch bias, but the file carries v.patch_embd.bias");
        }
        if (bothKernels && enc.patchEmbd1 == null) {
            throw new MissingTensorException(tc.archName(), -1, "v.patch_embd.weight.1");
        }
        if (ProjectorEntry.ENCODER_LFM3VL_VIT.equals(graph)) {
            enc.patchBias = null;
        }
    }

    private static TensorRef classEmbd(TensorCatalog tc, ClipHparams hp, String graph) {
        TensorRef cls = tc.get("v.class_embd");
        switch (graph) {
            case ProjectorEntry.ENCODER_CLIP_VIT:
            case ProjectorEntry.ENCODER_INTERNVL_VIT:
            case ProjectorEntry.ENCODER_LLAMA4_VIT:
            case ProjectorEntry.ENCODER_COGVLM_VIT:
                checkClass(tc, cls, hp.nEmbd());
                return cls;
            case ProjectorEntry.ENCODER_NEMOTRON_VIT:
                // the register tokens, prepended to the patch rows and dropped after the tower
                if (cls == null) {
                    throw new MissingTensorException(tc.archName(), -1, "v.class_embd");
                }
                if (cls.ne(0) != hp.nEmbd() || cls.ne(1) <= 0 || cls.rank() > 2) {
                    throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + cls.name()
                            + " has shape " + cls + ", the loader expects " + hp.nEmbd() + " columns of registers");
                }
                return cls;
            case ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP:
                if (cls == null) {
                    throw new MissingTensorException(tc.archName(), -1, "v.class_embd");
                }
                checkClass(tc, cls, hp.nEmbd());
                return cls;
            case ProjectorEntry.ENCODER_MINICPMV_VIT:
            case ProjectorEntry.ENCODER_GLM4V_VIT:
            case ProjectorEntry.ENCODER_YOUTUVL_VIT:
            case ProjectorEntry.ENCODER_GRANITE4_VIT:
            case ProjectorEntry.ENCODER_MINIMAX_M3_VIT:
            case ProjectorEntry.ENCODER_LFM3VL_VIT:
            case ProjectorEntry.ENCODER_STEP3VL_VIT:
            case ProjectorEntry.ENCODER_MIMOVL_VIT:
                if (cls != null) {
                    throw new IllegalArgumentException("architecture '" + tc.archName() + "': the " + graph
                            + " encoder has no class token, but the file carries v.class_embd");
                }
                return null;
            default:
                // the siglip, resized-siglip, qwen and pixtral graphs never read a class token
                return null;
        }
    }

    private static void checkClass(TensorCatalog tc, TensorRef cls, int nEmbd) {
        if (cls != null) {
            shape(tc, cls, nEmbd, 1);
        }
    }

    /**
     * The learned position rows, shape-checked per graph; null for the graphs that rotate positions.
     * {@code nPos} counts the {@code image_size} grid's patches plus the class token, {@code nPatches} the
     * patches alone.
     */
    private static TensorRef posEmbd(TensorCatalog tc, ClipHparams hp, String graph, int nPos, int nPatches) {
        if (ProjectorEntry.ENCODER_NEMOTRON_VIT.equals(graph)) {
            // v2-VL adds the rows as stored (the image_size grid), Omni resizes the square grid
            TensorRef pos = requirePos(tc, tc.get("v.position_embd.weight"), hp);
            if (hp.projectorType() == ProjectorType.NEMOTRON_V2_VL) {
                shape(tc, pos, hp.nEmbd(), nPatches);
                return pos;
            }
            return requireSquarePos(tc, pos);
        }
        TensorRef pos = tc.get("v.position_embd.weight");
        switch (graph) {
            case ProjectorEntry.ENCODER_CLIP_VIT:
            case ProjectorEntry.ENCODER_SIGLIP_VIT:
            case ProjectorEntry.ENCODER_INTERNVL_VIT:
            case ProjectorEntry.ENCODER_LLAMA4_VIT:
            case ProjectorEntry.ENCODER_COGVLM_VIT:
            case ProjectorEntry.ENCODER_GRANITE4_VIT:
                if (pos == null) {
                    throw new MissingTensorException(tc.archName(), -1, "v.position_embd.weight");
                }
                shape(tc, pos, hp.nEmbd(), nPos);
                return pos;
            case ProjectorEntry.ENCODER_QWEN2VL_VIT:
                if (pos != null) {
                    shape(tc, pos, hp.nEmbd(), nPos);
                }
                return pos;
            case ProjectorEntry.ENCODER_SIGLIP_RESIZED_VIT:
            case ProjectorEntry.ENCODER_QWEN3VL_VIT:
            case ProjectorEntry.ENCODER_PADDLEOCR_VIT:
            case ProjectorEntry.ENCODER_KIMIVL_VIT:
            case ProjectorEntry.ENCODER_HUNYUANVL_VIT:
            case ProjectorEntry.ENCODER_MUSE_GLIMMER_VIT:
            case ProjectorEntry.ENCODER_STEP3VL_VIT:
                return requireSquarePos(tc, requirePos(tc, pos, hp));
            case ProjectorEntry.ENCODER_KIMIK25_VIT:
                return requireGridPos(tc, pos, hp);
            case ProjectorEntry.ENCODER_GEMMA4V_VIT:
            case ProjectorEntry.ENCODER_GEMMA4UV:
                return requireAxisTables(tc, pos, hp);
            case ProjectorEntry.ENCODER_GLM4V_VIT:
                // GLM-OCR carries no learned position rows
                return pos == null ? null : requireSquarePos(tc, requirePos(tc, pos, hp));
            case ProjectorEntry.ENCODER_MINICPMV_VIT:
            case ProjectorEntry.ENCODER_MINICPMV46_VIT:
                return requirePos(tc, pos, hp);
            case ProjectorEntry.ENCODER_DEEPSEEKOCR_SAM_CLIP:
                // a square grid of rows plus one row the graph moves behind the grid when it resizes
                return requireSquarePlusOnePos(tc, requirePos(tc, pos, hp));
            default:
                // pixtral, dots and youtu-vl rotate their positions; they read no learned rows
                return null;
        }
    }

    /** Position rows stored as a square grid, which the resizing encoders interpolate to the image's grid. */
    private static TensorRef requireSquarePos(TensorCatalog tc, TensorRef p) {
        long side = Math.round(Math.sqrt((double) p.ne(1)));
        if (side * side != p.ne(1)) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor "
                    + p.name() + " holds " + p.ne(1) + " rows, not a square grid to resize");
        }
        return p;
    }

    /** Position rows of a square grid plus one row (the class token's slot of the DeepSeek-OCR CLIP tower). */
    private static TensorRef requireSquarePlusOnePos(TensorCatalog tc, TensorRef p) {
        long side = Math.round(Math.sqrt((double) (p.ne(1) - 1)));
        if (p.ne(1) < 2 || side * side != p.ne(1) - 1) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + p.name()
                    + " holds " + p.ne(1) + " rows, not a square grid plus one");
        }
        return p;
    }

    /** Kimi K2.5 position rows: a {@code [n_embd, W, H]} grid (models/kimik25.cpp reads W and H from the shape). */
    private static TensorRef requireGridPos(TensorCatalog tc, TensorRef pos, ClipHparams hp) {
        if (pos == null) {
            throw new MissingTensorException(tc.archName(), -1, "v.position_embd.weight");
        }
        if (pos.ne(0) != hp.nEmbd() || pos.ne(1) <= 0 || pos.ne(2) <= 0 || pos.rank() > 3) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + pos.name()
                    + " has shape " + pos + ", the loader expects a {" + hp.nEmbd() + ", W, H} position grid");
        }
        return pos;
    }

    /**
     * Gemma 4 position tables: {@code [n_embd, P, 2]}, the column table then the row table, each of {@code P}
     * rows (models/gemma4v.cpp views both halves of the tensor).
     */
    private static TensorRef requireAxisTables(TensorCatalog tc, TensorRef pos, ClipHparams hp) {
        if (pos == null) {
            throw new MissingTensorException(tc.archName(), -1, "v.position_embd.weight");
        }
        if (pos.ne(0) != hp.nEmbd() || pos.ne(1) <= 0 || pos.ne(2) != 2 || pos.rank() != 3) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + pos.name()
                    + " has shape " + pos + ", the loader expects {" + hp.nEmbd() + ", P, 2} column and row tables");
        }
        return pos;
    }

    private static TensorRef requirePos(TensorCatalog tc, TensorRef pos, ClipHparams hp) {
        if (pos == null) {
            throw new MissingTensorException(tc.archName(), -1, "v.position_embd.weight");
        }
        if (pos.ne(0) != hp.nEmbd() || pos.ne(1) <= 0 || pos.rank() > 2) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + pos.name()
                    + " has shape " + pos + ", the loader expects " + hp.nEmbd() + " columns");
        }
        return pos;
    }

    /**
     * The pre/post norms; the CLIP and SigLIP graphs pair weight and bias, the others take a bias-free norm. The
     * cogvlm, granite4 and yasa2 graphs read neither; dots.ocr reads {@code mm.post_norm} as its post-norm.
     */
    private static void norms(TensorCatalog tc, ClipHparams hp, String graph, Encoder enc) {
        if (ProjectorEntry.ENCODER_COGVLM_VIT.equals(graph) || ProjectorEntry.ENCODER_GRANITE4_VIT.equals(graph)
                || ProjectorEntry.ENCODER_YASA2_CONVNEXT.equals(graph)) {
            return;
        }
        if (ProjectorEntry.ENCODER_MIMOVL_VIT.equals(graph)) {
            refuse(tc, graph, "v.pre_ln.weight", "has no pre-norm");
        }
        enc.preLnW = tc.get("v.pre_ln.weight");
        enc.preLnB = tc.get("v.pre_ln.bias");
        // dots.ocr loads its post_trunk_norm, mm.post_norm, into the post-norm slot (clip.cpp load_tensors)
        enc.postLnW = ProjectorEntry.ENCODER_DOTS_VIT.equals(graph) ? tc.require("mm.post_norm.weight")
                : tc.get("v.post_ln.weight");
        enc.postLnB = tc.get("v.post_ln.bias");
        boolean paired = ProjectorEntry.ENCODER_CLIP_VIT.equals(graph)
                || ProjectorEntry.ENCODER_SIGLIP_VIT.equals(graph);
        if (paired) {
            checkPair(tc, "v.pre_ln", enc.preLnW, enc.preLnB, hp.nEmbd());
        } else {
            checkNorm(tc, "v.pre_ln", enc.preLnW, enc.preLnB, hp.nEmbd());
        }
        checkNorm(tc, "v.post_ln", enc.postLnW, enc.postLnB, hp.nEmbd());
    }

    /** The hparams the geometry was resolved from. */
    public ClipHparams hparams() {
        return hparams;
    }

    /** The encoder graph (a {@code ProjectorEntry.ENCODER_*} name). */
    public String encoderGraph() {
        return encoderGraph;
    }

    /** The projector graph (a {@code ProjectorEntry.PROJECTOR_*} name). */
    public String projectorGraph() {
        return projectorGraph;
    }

    /** Patches per row of the square {@code clip.vision.image_size} grid. */
    public int nPatchesPerSide() {
        return nPatchesX;
    }

    /** Patches of the square {@code clip.vision.image_size} grid. */
    public int nPatches() {
        return nPatches;
    }

    /**
     * Rows of the {@code image_size} grid the encoder runs over: the patches plus the class token, or the
     * Nemotron register tokens, if any (for the DeepSeek-OCR encoders, the tower rows of the global view).
     */
    public int nPos() {
        return nPos;
    }

    /** The side of the pixel square one patch row reads ({@link #patchKernelSize(ClipHparams, String)}). */
    public int patchKernelSize() {
        return patchKernel;
    }

    /** Head width. */
    public int dHead() {
        return dHead;
    }

    /** Attention softmax scale {@code 1/sqrt(d_head)}. */
    public float attnScale() {
        return attnScale;
    }

    /** Width of every output row ({@code clip_n_mmproj_embd}), the text model's input embedding width. */
    public int outputDim() {
        return projector.outputDim();
    }

    /** True when the file holds a class embedding (or the Nemotron register rows) the encoder reads. */
    public boolean hasClassEmbd() {
        return classEmbd != null;
    }

    /** True when the file holds a pre-attention norm. */
    public boolean hasPreLn() {
        return preLnW != null;
    }

    /**
     * The patch kernel {@code v.patch_embd.weight}; null for the DeepSeek-OCR encoders, whose towers read SAM
     * rows, and for the convolutional MobileNetV5 encoder.
     */
    public TensorRef patchEmbd() {
        return patchEmbd;
    }

    /**
     * Output channels of the patch embedding: {@code n_embd}, or the first stage's width for the yasa2 ConvNeXt
     * stem.
     */
    public int patchChannels() {
        return (int) patchEmbd.ne(patchEmbd.rank() - 1);
    }

    /**
     * The second patch kernel {@code v.patch_embd.weight.1}: the second Conv3d frame of the qwen encoders, the
     * temporal conv kernel of GLM-4V; null for the other encoders.
     */
    public TensorRef patchEmbd1() {
        return patchEmbd1;
    }

    public TensorRef patchBias() {
        return patchBias;
    }

    /** The GLM-4V embedding norm weight after the patch convolutions, or null. */
    public TensorRef normEmbdWeight() {
        return normEmbdW;
    }

    /** The GLM-4V embedding norm bias, or null. */
    public TensorRef normEmbdBias() {
        return normEmbdB;
    }

    public TensorRef classEmbd() {
        return classEmbd;
    }

    public TensorRef posEmbd() {
        return posEmbd;
    }

    public TensorRef preLnWeight() {
        return preLnW;
    }

    public TensorRef preLnBias() {
        return preLnB;
    }

    /** The post-encoder norm weight, or null when the family's file carries none. */
    public TensorRef postLnWeight() {
        return postLnW;
    }

    public TensorRef postLnBias() {
        return postLnB;
    }

    /** The encoder blocks, in order (unmodifiable). */
    public List<ClipLayer> layers() {
        return layers;
    }

    /** The projector tensors the graph reads by a matmul, by name (unmodifiable). */
    public Map<String, TensorRef> projectorMatrices() {
        return projector.matrices();
    }

    /** The projector tensors the graph reads as values (biases, norms, embedding rows), by name (unmodifiable). */
    public Map<String, TensorRef> projectorVectors() {
        return projector.vectors();
    }

    @Override
    public String toString() {
        return "ClipVisionModel{" + hparams.projectorType() + ", encoder=" + encoderGraph + ", projector="
                + projectorGraph + ", patches=" + nPatchesX + "x" + nPatchesY + ", pos=" + nPos + ", layers="
                + layers.size() + ", type=" + (layers.isEmpty() ? GGMLType.F32 : layers.get(0).oW.type())
                + ", out=" + outputDim() + String.format(Locale.ROOT, ", scale=%.6f}", attnScale);
    }
}
