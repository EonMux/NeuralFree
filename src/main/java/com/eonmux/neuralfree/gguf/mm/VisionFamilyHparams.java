package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Arrays;

/**
 * The family-specific vision keys of one mmproj encoder beyond the shared {@link ClipHparams} (clip.cpp
 * {@code load_hparams}, per projector type): the explicit full-attention layers of Youtu-VL
 * ({@code clip.vision.wa_layer_indexes}), the WindowQFormer layout of Granite Vision 4.1
 * ({@code clip.vision.projector.query_side}, {@code window_side}, {@code spatial_offsets}), the SAM backbone of
 * DeepSeek-OCR and DeepSeek-OCR 2 ({@code clip.vision.sam.*}, {@code clip.vision.window_size}), the ViT merger
 * insertion layer of MiniCPM-V 4.6 (the first {@code clip.vision.wa_layer_indexes} entry) and the per-block
 * attention modes and window radius of MiMo-VL ({@code clip.vision.wa_pattern_mode}, {@code clip.vision.window_size}).
 * A family that does not read a key keeps it empty, 0 or null. Immutable.
 */
public final class VisionFamilyHparams {

    static final VisionFamilyHparams NONE = new VisionFamilyHparams(new int[0], new int[0], 0, 0, null, 0,
            new int[0], 0);

    /** MiMo-VL block mode: full attention. */
    static final int MODE_FULL = -1;
    /** MiMo-VL block mode: the banded window over the row-major merge units. */
    static final int MODE_ROW = 0;
    /** MiMo-VL block mode: the banded window over the column-major merge units. */
    static final int MODE_COL = 1;

    private static final String WA_LAYER_INDEXES = "clip.vision.wa_layer_indexes";
    private static final String WA_PATTERN_MODE = "clip.vision.wa_pattern_mode";
    private static final String WINDOW_SIZE = "clip.vision.window_size";
    private static final String HEAD_COUNT_KV = "clip.vision.attention.head_count_kv";

    /** The DeepSeek-OCR SAM backbone keys. Immutable. */
    public static final class Sam {
        private final int nLayer;
        private final int nHead;
        private final int nEmbd;
        private final int window;
        private final ClipFfnOp ffnOp;

        private Sam(int nLayer, int nHead, int nEmbd, int window, ClipFfnOp ffnOp) {
            this.nLayer = nLayer;
            this.nHead = nHead;
            this.nEmbd = nEmbd;
            this.window = window;
            this.ffnOp = ffnOp;
        }

        /** The SAM blocks ({@code clip.vision.sam.block_count}). */
        public int nLayer() {
            return nLayer;
        }

        /** The SAM attention heads ({@code clip.vision.sam.head_count}). */
        public int nHead() {
            return nHead;
        }

        /** The SAM width ({@code clip.vision.sam.embedding_length}). */
        public int nEmbd() {
            return nEmbd;
        }

        /** The side of the local attention windows ({@code clip.vision.window_size}). */
        public int window() {
            return window;
        }

        /** The SAM MLP activation: the {@code clip.use_gelu} / {@code clip.use_silu} choice. */
        public ClipFfnOp ffnOp() {
            return ffnOp;
        }

        @Override
        public String toString() {
            return "Sam{layers=" + nLayer + ", heads=" + nHead + ", embd=" + nEmbd + ", window=" + window + ", ffn="
                    + ffnOp + "}";
        }
    }

    private final int[] fullAttnLayers;
    private final int[] spatialOffsets;
    private final int querySide;
    private final int windowSide;
    private final Sam sam;
    private final int insertLayerId;
    private final int[] attnModes;
    private final int windowRadius;

    private VisionFamilyHparams(int[] fullAttnLayers, int[] spatialOffsets, int querySide, int windowSide, Sam sam,
                                int insertLayerId, int[] attnModes, int windowRadius) {
        this.fullAttnLayers = fullAttnLayers;
        this.spatialOffsets = spatialOffsets;
        this.querySide = querySide;
        this.windowSide = windowSide;
        this.sam = sam;
        this.insertLayerId = insertLayerId;
        this.attnModes = attnModes;
        this.windowRadius = windowRadius;
    }

    /**
     * The keys {@code type} reads.
     *
     * @throws GgufFormatException for a missing required key or an out-of-range value (named)
     */
    static VisionFamilyHparams from(GgufKv kv, ProjectorType type, int nLayer, int nMerge, int[] featureLayers) {
        switch (type) {
            case YOUTUVL:
                return youtuvl(kv, nLayer);
            case GRANITE4_VISION:
                return granite4(kv, nLayer, featureLayers);
            case DEEPSEEKOCR:
            case DEEPSEEKOCR2:
                return deepseekOcr(kv);
            case MINICPMV4_6:
            case MINICPMV4_7:
            case MINICPM_OCR_1_0:
                return minicpmv46(kv, type, nLayer, nMerge);
            case MIMOVL:
                return mimovl(kv, nLayer, nMerge);
            default:
                return NONE;
        }
    }

    private static VisionFamilyHparams youtuvl(GgufKv kv, int nLayer) {
        String key = WA_LAYER_INDEXES;
        int[] layers = kv.i32Array(key);
        if (layers == null) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        for (int il : layers) {
            if (il < 0 || il >= nLayer) {
                throw new GgufFormatException("metadata key '" + key + "' index " + il + " is out of range"
                        + " (block_count " + nLayer + ")");
            }
        }
        if (!kv.has("clip.vision.window_size")) {
            throw new GgufFormatException("required metadata key 'clip.vision.window_size' is missing");
        }
        int[] sorted = layers.clone();
        Arrays.sort(sorted);
        return new VisionFamilyHparams(sorted, new int[0], 0, 0, null, 0, new int[0], 0);
    }

    /**
     * MiniCPM-V 4.6, 4.7 and OCR 1.0: the merge must be 2 (no ViT merger) or 4; the ViT merger follows the block
     * the first {@code clip.vision.wa_layer_indexes} entry names, block 0 when the key is absent or empty.
     */
    private static VisionFamilyHparams minicpmv46(GgufKv kv, ProjectorType type, int nLayer, int nMerge) {
        if (nMerge != 2 && nMerge != 4) {
            throw new GgufFormatException("metadata key 'clip.vision.projector.scale_factor' must be 2 or 4 for "
                    + type + ", got " + nMerge);
        }
        int[] layers = kv.i32Array(WA_LAYER_INDEXES);
        int insert = layers == null || layers.length == 0 ? 0 : layers[0];
        if (insert < 0 || insert >= nLayer) {
            throw new GgufFormatException("metadata key '" + WA_LAYER_INDEXES + "' index " + insert + " is out of"
                    + " range (block_count " + nLayer + ")");
        }
        return new VisionFamilyHparams(new int[0], new int[0], 0, 0, null, insert, new int[0], 0);
    }

    /**
     * MiMo-VL (clip.cpp load_hparams): the 2x2 merge its patch order is written for, the required key/value head
     * count, the positive window radius and one attention mode per block, each -1, 0 or 1.
     */
    private static VisionFamilyHparams mimovl(GgufKv kv, int nLayer, int nMerge) {
        if (nMerge != 2) {
            throw new GgufFormatException("metadata key 'clip.vision.spatial_merge_size' must be 2 for mimovl,"
                    + " whose patch embedding orders 2x2 merge groups; got " + nMerge);
        }
        if (!kv.has(HEAD_COUNT_KV)) {
            throw new GgufFormatException("required metadata key '" + HEAD_COUNT_KV + "' is missing");
        }
        if (!kv.has(WINDOW_SIZE)) {
            throw new GgufFormatException("required metadata key '" + WINDOW_SIZE + "' is missing");
        }
        long window = kv.u32Long(WINDOW_SIZE, 0);
        if (window <= 0 || window > Integer.MAX_VALUE) {
            throw new GgufFormatException("metadata key '" + WINDOW_SIZE + "' must be a positive window radius for"
                    + " mimovl, got " + window);
        }
        int[] modes = kv.i32Array(WA_PATTERN_MODE);
        if (modes == null) {
            throw new GgufFormatException("required metadata key '" + WA_PATTERN_MODE + "' is missing");
        }
        if (modes.length != nLayer) {
            throw new GgufFormatException("metadata key '" + WA_PATTERN_MODE + "' must hold " + nLayer
                    + " values (one per block), got " + modes.length);
        }
        for (int mode : modes) {
            if (mode != MODE_FULL && mode != MODE_ROW && mode != MODE_COL) {
                throw new GgufFormatException("metadata key '" + WA_PATTERN_MODE + "' value " + mode + " is not -1"
                        + " (full), 0 (row window) or 1 (column window)");
            }
        }
        return new VisionFamilyHparams(new int[0], new int[0], 0, 0, null, 0, modes.clone(), (int) window);
    }

    private static VisionFamilyHparams granite4(GgufKv kv, int nLayer, int[] featureLayers) {
        String key = GgufKeys.CLIP_VISION_PROJECTOR_SPATIAL_OFFSETS;
        int[] offsets = kv.i32Array(key);
        if (offsets == null) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        if (featureLayers.length == 0 || featureLayers.length != offsets.length) {
            throw new GgufFormatException("metadata key '" + key + "' holds " + offsets.length + " values, the"
                    + " granite4_vision projectors need one per clip.vision.feature_layer entry ("
                    + featureLayers.length + ", at least one)");
        }
        for (int i = 0; i < featureLayers.length; i++) {
            if (featureLayers[i] < 0 || featureLayers[i] >= nLayer) {
                throw new GgufFormatException("metadata key 'clip.vision.feature_layer' index " + featureLayers[i]
                        + " is out of range for granite4_vision (block_count " + nLayer + ")");
            }
            if (offsets[i] > 3) {
                throw new GgufFormatException("metadata key '" + key + "' value " + offsets[i] + " is not a 2x2"
                        + " sampling offset (0..3) or -1");
            }
        }
        int query = positive(kv, GgufKeys.CLIP_VISION_PROJECTOR_QUERY_SIDE);
        int window = positive(kv, GgufKeys.CLIP_VISION_PROJECTOR_WINDOW_SIDE);
        return new VisionFamilyHparams(new int[0], offsets.clone(), query, window, null, 0, new int[0], 0);
    }

    /** clip.cpp {@code load_hparams}: the SAM block count, heads, width and window are required. */
    private static VisionFamilyHparams deepseekOcr(GgufKv kv) {
        int layers = positive(kv, GgufKeys.CLIP_VISION_SAM_BLOCK_COUNT);
        int heads = positive(kv, GgufKeys.CLIP_VISION_SAM_HEAD_COUNT);
        int embd = positive(kv, GgufKeys.CLIP_VISION_SAM_EMBEDDING_LENGTH);
        int window = positive(kv, "clip.vision.window_size");
        if (embd % heads != 0) {
            throw new GgufFormatException("metadata key '" + GgufKeys.CLIP_VISION_SAM_EMBEDDING_LENGTH + "' ("
                    + embd + ") must be a multiple of '" + GgufKeys.CLIP_VISION_SAM_HEAD_COUNT + "' (" + heads + ")");
        }
        ClipFfnOp ffn = ClipFfnOp.fromUseGeluSilu(kv.bool("clip.use_gelu", false), kv.bool("clip.use_silu", false));
        return new VisionFamilyHparams(new int[0], new int[0], 0, 0, new Sam(layers, heads, embd, window, ffn), 0,
                new int[0], 0);
    }

    private static int positive(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        int v = kv.u32(key, 0);
        if (v <= 0) {
            throw new GgufFormatException("metadata key '" + key + "' must be positive, got " + v);
        }
        return v;
    }

    /** Youtu-VL: the layers that attend over the whole image, ascending; every other layer uses windows. */
    public int[] fullAttnLayers() {
        return fullAttnLayers.clone();
    }

    /** True when layer {@code il} is listed as full attention. */
    boolean isFullAttnLayer(int il) {
        return Arrays.binarySearch(fullAttnLayers, il) >= 0;
    }

    /**
     * Granite Vision 4.1: per projector, the 2x2 sampling offset of the spatial downsampler ({@code (dy << 1) | dx}),
     * or -1 for the average-pool downsampler.
     */
    public int[] spatialOffsets() {
        return spatialOffsets.clone();
    }

    /** Granite Vision 4.1: query tokens per window side of the WindowQFormer. */
    public int querySide() {
        return querySide;
    }

    /** Granite Vision 4.1: patches per window side of the WindowQFormer. */
    public int windowSide() {
        return windowSide;
    }

    /** DeepSeek-OCR and DeepSeek-OCR 2: the SAM backbone keys; null for the other families. */
    public Sam sam() {
        return sam;
    }

    /** MiniCPM-V 4.6: the block after which the ViT merger runs (with a 4x merge). */
    public int insertLayerId() {
        return insertLayerId;
    }

    /**
     * MiMo-VL: the attention mode of every block, {@link #MODE_FULL}, {@link #MODE_ROW} or {@link #MODE_COL}
     * (empty for the other families).
     */
    public int[] attnModes() {
        return attnModes.clone();
    }

    /** MiMo-VL block {@code il}'s attention mode. */
    int attnMode(int il) {
        return attnModes[il];
    }

    /** MiMo-VL: the window radius; a windowed row attends to the keys at most this many rows away. */
    public int windowRadius() {
        return windowRadius;
    }

    @Override
    public String toString() {
        return "VisionFamilyHparams{fullAttn=" + Arrays.toString(fullAttnLayers) + ", spatialOffsets="
                + Arrays.toString(spatialOffsets) + ", query=" + querySide + ", window=" + windowSide
                + (sam != null ? ", sam=" + sam : "") + ", insertLayer=" + insertLayerId
                + ", attnModes=" + Arrays.toString(attnModes) + ", windowRadius=" + windowRadius + "}";
    }
}
