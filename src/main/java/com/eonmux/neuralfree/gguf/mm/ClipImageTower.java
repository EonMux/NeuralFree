package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.image.ImageOps;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.PadStyle;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.image.ResizeAlgo;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkNorm;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The vision tower of a CLIP two-tower file and its projection, vLLM's {@code CLIPVisionTransformer} plus
 * {@code visual_projection} (vllm/model_executor/models/clip.py) over the HF {@code CLIPImageProcessor} input:
 * the image resizes its shortest edge to {@code image_size} (bicubic), center-crops the square tile and normalizes
 * by the file's mean and std; the patch rows ({@code v.patch_embd}) follow the class token ({@code v.class_embd})
 * at position 0 as HF orders them, plus the learned position rows, then the optional pre-norm ({@code v.pre_ln}),
 * every block of {@link CpuVitTower}, the post-norm ({@code v.post_ln}), the rows the {@link ImageFeatureSelect}
 * keeps, and the bias-free {@code visual_projection}. The {@link ClipFamily#SIGLIP} tower is vLLM's
 * {@code SiglipVisionTransformer} (vllm/model_executor/models/siglip.py) over the HF {@code SiglipImageProcessor}
 * input: the image stretches to the square tile (bicubic, no crop), the patch rows carry no class token, and the
 * attention pooling head ({@link SiglipMapHead}, {@code v.head}) pools the post-normed rows into the one row the
 * select keeps. Immutable weights; not thread-safe across calls. Package-private executor of
 * {@link ClipDualEncoder}.
 */
final class ClipImageTower {

    static final String PROJECTION = "visual_projection.weight";
    static final String CLASS_EMBD = "v.class_embd";
    private static final int[] PAD_BLACK = {0, 0, 0};

    private final ClipHparams hp;
    private final ClipFamily family;
    private final CpuMatVec matVec;
    private final CpuTensorRef patchEmbd;
    private final float[] patchBias;
    private final float[] classEmbd;
    private final float[] posEmbd;
    private final SiglipMapHead mapHead;
    private final CpuTensorRef projection;
    private final CpuVitTower tower;

    private ClipImageTower(ClipHparams hp, ClipFamily family, CpuMatVec matVec, CpuTensorRef patchEmbd,
                           float[] patchBias, float[] classEmbd, float[] posEmbd, SiglipMapHead mapHead,
                           CpuTensorRef projection, CpuVitTower tower) {
        this.hp = hp;
        this.family = family;
        this.matVec = matVec;
        this.patchEmbd = patchEmbd;
        this.patchBias = patchBias;
        this.classEmbd = classEmbd;
        this.posEmbd = posEmbd;
        this.mapHead = mapHead;
        this.projection = projection;
        this.tower = tower;
    }

    /**
     * Resolves and shape-checks the vision tensors of {@code tc} for {@code hp} and the {@code family}'s layout,
     * then binds them through {@code floats} and {@code refs}.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     * @throws IllegalArgumentException when a shape differs from the tower's, or a SigLIP file carries a class token
     */
    static ClipImageTower bind(ClipHparams hp, ClipFamily family, TensorCatalog tc, CpuMatVec matVec,
                               Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(hp, "hparams");
        Objects.requireNonNull(family, "family");
        int nEmbd = hp.nEmbd();
        int ps = hp.patchSize();
        int side = hp.imageSize() / ps;
        boolean siglip = family == ClipFamily.SIGLIP;
        TensorRef patch = tc.require("v.patch_embd.weight");
        shape(tc, patch, ps, ps, 3, nEmbd);
        TensorRef patchB = tc.get("v.patch_embd.bias");
        checkVector(tc, patchB, nEmbd);
        TensorRef cls = siglip ? tc.get(CLASS_EMBD) : tc.require(CLASS_EMBD);
        if (siglip && cls != null) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': a " + family + " vision tower"
                    + " has no class token, but the file carries " + CLASS_EMBD);
        }
        if (cls != null) {
            shape(tc, cls, nEmbd);
        }
        TensorRef pos = tc.require("v.position_embd.weight");
        shape(tc, pos, nEmbd, side * side + (cls == null ? 0 : 1));
        TensorRef preW = tc.get("v.pre_ln.weight");
        TensorRef preB = tc.get("v.pre_ln.bias");
        checkNorm(tc, "v.pre_ln", preW, preB, nEmbd);
        TensorRef postW = tc.get("v.post_ln.weight");
        TensorRef postB = tc.get("v.post_ln.bias");
        checkNorm(tc, "v.post_ln", postW, postB, nEmbd);
        TensorRef proj = ClipTextTower.projection(tc, hp, PROJECTION, nEmbd);
        List<ClipLayer> layers = new ArrayList<>(hp.nLayer());
        for (int il = 0; il < hp.nLayer(); il++) {
            layers.add(ClipLayer.resolve(tc, hp, il, false));
        }
        SiglipMapHead mapHead = siglip && SiglipMapHead.present(tc)
                ? SiglipMapHead.bind(hp, tc, matVec, floats, refs) : null;
        CpuVitTower tower = new CpuVitTower(hp, matVec, layers, preW, preB, postW, postB, floats, refs, null);
        CpuTensorRef kernel = CpuVitTower.matrix(refs.apply(patch.name()), nEmbd, ClipPatches.rowWidth(ps));
        return new ClipImageTower(hp, family, matVec, kernel, CpuVitTower.floatsOrNull(patchB, floats),
                CpuVitTower.floatsOrNull(cls, floats), floats.apply(pos.name()), mapHead,
                proj == null ? null : refs.apply(proj.name()), tower);
    }

    /** The width of the embedding: the projection's, or the tower's without one. */
    int outputDim() {
        return projection == null ? hp.nEmbd() : (int) projection.rows();
    }

    /** The square tile side the tower reads ({@code clip.vision.image_size}). */
    int imageSize() {
        return hp.imageSize();
    }

    /** The tower's pre-norm, blocks and post-norm on the host. */
    TowerBlocks cpuBlocks() {
        return (rows, n) -> tower.forward(rows, n, NormKind.LAYER, CpuVitTower.Options.NONE);
    }

    /** The pooled, projected embedding of one decoded image, the tower's blocks run by {@code blocks}. */
    float[] embed(RasterImage image, ImageFeatureSelect select, TowerBlocks blocks) {
        Objects.requireNonNull(select, "select");
        RasterImage tile = family == ClipFamily.SIGLIP ? squash(image, hp.imageSize())
                : preprocess(image, hp.imageSize());
        ClipImage pixels = ClipImage.of(tile.width(), tile.height(), tile.normalized(hp.imageMean(), hp.imageStd()));
        int nEmbd = hp.nEmbd();
        int side = hp.imageSize() / hp.patchSize();
        int nPatches = side * side;
        int first = classEmbd == null ? 0 : 1;
        int nPos = nPatches + first;
        float[] patches = ClipPatches.raster(pixels, hp.patchSize(), side, side);
        float[] rows = new float[nPos * nEmbd];
        matVec.matmul(patchEmbd, patches, 0, rows, first * nEmbd, nPatches);
        if (patchBias != null) {
            for (int t = first; t < nPos; t++) {
                for (int i = 0; i < nEmbd; i++) {
                    rows[t * nEmbd + i] += patchBias[i];
                }
            }
        }
        if (classEmbd != null) {
            System.arraycopy(classEmbd, 0, rows, 0, nEmbd);
        }
        for (int i = 0; i < nPos * nEmbd; i++) {
            rows[i] += posEmbd[i];
        }
        float[] out = blocks.forward(rows, nPos);
        float[] pooled = mapHead != null ? mapHead.pool(out, nPos) : select.select(out, nPos, nEmbd);
        if (projection == null) {
            return pooled;
        }
        float[] embedding = new float[(int) projection.rows()];
        matVec.matmul(projection, pooled, 0, embedding, 0, 1);
        return embedding;
    }

    /** HF {@code SiglipImageProcessor}'s geometry: the image stretched to the {@code size x size} tile (bicubic). */
    static RasterImage squash(RasterImage image, int size) {
        ImageSize target = new ImageSize(size, size);
        return target.equals(image.size()) ? image
                : ImageOps.resize(image, target, ResizeAlgo.BICUBIC, PadStyle.NONE, PAD_BLACK);
    }

    /**
     * HF {@code CLIPImageProcessor}'s geometry: the shortest edge resized to {@code size} with the longer edge
     * scaled to {@code floor(size * long / short)} (bicubic), then the centered {@code size x size} crop.
     */
    static RasterImage preprocess(RasterImage image, int size) {
        int w = image.width();
        int h = image.height();
        int shortSide = Math.min(w, h);
        int longSide = (int) ((long) size * Math.max(w, h) / shortSide);
        ImageSize target = w <= h ? new ImageSize(size, longSide) : new ImageSize(longSide, size);
        RasterImage resized = target.equals(image.size()) ? image
                : ImageOps.resize(image, target, ResizeAlgo.BICUBIC, PadStyle.NONE, PAD_BLACK);
        int left = (resized.width() - size) / 2;
        int top = (resized.height() - size) / 2;
        if (left == 0 && top == 0 && resized.width() == size && resized.height() == size) {
            return resized;
        }
        return ImageOps.crop(resized, left, top, size, size);
    }
}
