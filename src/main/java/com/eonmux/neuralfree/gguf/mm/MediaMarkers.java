package com.eonmux.neuralfree.gguf.mm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * The text a multimodal prompt wraps around media embeddings, per projector family: llama.cpp's
 * {@code mtmd_context::init_vision} and {@code init_audio} (tools/mtmd/mtmd.cpp). An image is framed by
 * {@link #imageBegin()} / {@link #imageEnd()}; a sliced (llava-uhd) image places its overview between
 * {@link #overviewStart()} / {@link #overviewEnd()} before or after the slices ({@link #overviewFirst()}), and
 * frames the slice grid with the slice, mid and row-end texts of the family's slice template, a slice opening
 * either with a fixed text or with one naming its row and column ({@link #sliceStart(int, int)}).
 * {@link #tiledLayout(int, int)} lays these out for one sliced image; a MiniCPM-V 4.7 image instead takes one
 * canvas chunk ({@link #canvasImages()}, {@link #canvasLayout()}). Audio is framed by {@link #audioBegin()} /
 * {@link #audioEnd()}. Every text is tokenized with special tokens parsed, as the reference tokenizes its markers;
 * an empty text adds nothing. Immutable.
 */
public final class MediaMarkers {

    private static final Map<ProjectorType, String[]> FRAMES = frames();

    private String imgBeg = "";
    private String imgEnd = "";
    private String ovStart = "";
    private String ovEnd = "";
    private String slicesStart = "";
    private String slicesEnd = "";
    private String sliStart = "";
    private String sliStartTmpl = "";
    private String sliEnd = "";
    private String sliMid = "";
    private String rowEnd = "";
    private boolean rowEndTrail;
    private boolean ovFirst;
    private boolean imageId;
    private boolean canvas;
    private String audBeg = "";
    private String audEnd = "";
    private boolean nonCausalImage;
    private boolean zeroBasedImageGrid;
    private int imageLeadPadAlign;
    private boolean overviewViewSeparator;

    private MediaMarkers() {
    }

    /** Markers of a projector file without a vision encoder: nothing around images. */
    static MediaMarkers none() {
        return new MediaMarkers();
    }

    /**
     * The image markers of a vision projector family.
     *
     * @param projector       the vision projector type (the encoder's, resolved for the vision modality)
     * @param minicpmvVersion {@code clip.minicpmv_version}, read for the resampler family
     * @param projectorWidth  the projector's output width ({@code clip_n_mmproj_embd}), which decides whether
     *                        Gemma 4 image rows decode causally
     * @param hasToken        whether the text vocabulary holds a token text, for families whose markers depend on
     *                        the vocabulary (kimik25)
     * @throws IllegalArgumentException for a MiniCPM-V version the reference does not know
     */
    static MediaMarkers vision(ProjectorType projector, int minicpmvVersion, int projectorWidth,
                               Predicate<String> hasToken) {
        Objects.requireNonNull(hasToken, "hasToken");
        MediaMarkers m = new MediaMarkers();
        if (!m.slicedVision(projector, minicpmvVersion)) {
            m.framedVision(projector, hasToken);
        }
        m.nonCausalImage = nonCausal(projector, projectorWidth);
        // mtmd MTMD_POS_TYPE_SENSENOVA_U1: the grid coordinates start at 0 and the image takes one position
        m.zeroBasedImageGrid = projector == ProjectorType.SENSENOVA_U1;
        // mtmd_tokenize: the DeepSeek-V4 text model reads blocks of 4 tokens; lead pads align the image to one
        m.imageLeadPadAlign = projector == ProjectorType.DEEPSEEK4V ? DeepseekV4Layout.ALIGN : 0;
        // mtmd_image_preprocessor_deepseekocr marks its global view add_viewsep
        m.overviewViewSeparator = projector == ProjectorType.DEEPSEEKOCR || projector == ProjectorType.DEEPSEEKOCR2;
        return m;
    }

    /**
     * The families whose sliced images take a slice template (mtmd {@code slice_tmpl}); false for the others.
     */
    private boolean slicedVision(ProjectorType projector, int minicpmvVersion) {
        switch (projector) {
            case MINICPMV:
                minicpmv(minicpmvVersion);
                return true;
            case MINICPMV4_6:
            case MINICPMV4_7:
            case MINICPM_OCR_1_0:
                minicpmv26();
                imageId = true;
                canvas = projector == ProjectorType.MINICPMV4_7;
                return true;
            case IDEFICS3:
                // processing_idefics3.py: <fake_token_around_image><row_r_col_c> per tile, a newline per row,
                // then the global image
                ovStart = "\n\n<fake_token_around_image><global-img>";
                ovEnd = "<fake_token_around_image>";
                rowEnd = "\n";
                sliStartTmpl = "<fake_token_around_image><row_%d_col_%d>";
                return true;
            case LLAMA4:
                // tiles split by <|tile_x_separator|>, every row ended by <|tile_y_separator|>, the overview last
                image("<|image_start|>", "<|image_end|>");
                ovStart = "<|image|>";
                sliMid = "<|tile_x_separator|>";
                rowEnd = "<|tile_y_separator|>";
                rowEndTrail = true;
                return true;
            case STEP3VL:
                ovStart = "<im_start>";
                ovEnd = "<im_end>";
                sliStart = "<patch_start>";
                sliEnd = "<patch_end>";
                rowEnd = "<patch_newline>";
                return true;
            case LFM2:
                // <|img_row_r_col_c|> before every tile, <|img_thumbnail|> before the thumbnail
                image("<|image_start|>", "<|image_end|>");
                sliStartTmpl = "<|img_row_%d_col_%d|>";
                ovStart = "<|img_thumbnail|>";
                return true;
            case COHERE2V:
                image("<|START_OF_IMG|>", "<|END_OF_IMG|>");
                sliEnd = "<|IMG_LINE_BREAK|>";
                ovEnd = sliEnd;
                return true;
            default:
                return false;
        }
    }

    /**
     * The image begin and end texts of the families without a slice template; Kimi K2.5 takes GLM-5.2-V's markers
     * when the vocabulary holds them. The llava MLP / LDP / GLM-Edge / CogVLM / Janus-Pro families, phi4,
     * nemotron_v2_vl and d1-omni add no marker text.
     */
    private void framedVision(ProjectorType projector, Predicate<String> hasToken) {
        if (projector == ProjectorType.KIMIK25) {
            if (hasToken.test("<|begin_of_image|>")) {
                image("<|begin_of_image|>", "<|end_of_image|>");
            } else {
                image("<|media_begin|>", "<|media_end|>");
            }
            return;
        }
        String[] frame = FRAMES.get(projector);
        if (frame != null) {
            image(frame[0], frame[1]);
        }
    }

    /** The image begin and end texts per framed projector type ({@code img_beg} / {@code img_end}). */
    private static Map<ProjectorType, String[]> frames() {
        Map<ProjectorType, String[]> m = new EnumMap<>(ProjectorType.class);
        frame(m, "<|vision_start|>", "<|vision_end|>", ProjectorType.QWEN2VL, ProjectorType.QWEN25VL,
                ProjectorType.QWEN3VL, ProjectorType.LING3VL, ProjectorType.MIMOVL, ProjectorType.LFM3VL,
                ProjectorType.YOUTUVL);
        frame(m, "]<]start of image[>[", "]<]end of image[>[", ProjectorType.MINIMAX_M3);
        frame(m, "<img>", "</img>", ProjectorType.SENSENOVA_U1, ProjectorType.INTERNVL, ProjectorType.NEMOTRON_OMNI);
        frame(m, "<|image_start|>", "<|image_end|>", ProjectorType.MUSE_GLIMMER);
        frame(m, "<image>", "</image>", ProjectorType.YASA2);
        frame(m, "<start_of_image>", "<end_of_image>", ProjectorType.GEMMA3, ProjectorType.GEMMA3NV);
        frame(m, "<|image>", "<image|>", ProjectorType.GEMMA4V, ProjectorType.GEMMA4UV);
        frame(m, "", "[IMG_END]", ProjectorType.PIXTRAL);
        frame(m, "<|im_start|>", "<|im_end|>", ProjectorType.LIGHTONOCR);
        frame(m, "<|media_start|>", "<|media_end|>", ProjectorType.KIMIVL);
        frame(m, "", "\n", ProjectorType.DEEPSEEKOCR, ProjectorType.DEEPSEEKOCR2, ProjectorType.GRANITE4_VISION);
        frame(m, "<|begin_of_image|>", "<|end_of_image|>", ProjectorType.GLM4V, ProjectorType.GLM5V);
        frame(m, "<|img|>", "<|endofimg|>", ProjectorType.DOTS_OCR, ProjectorType.DOTS3NOTE_V);
        frame(m, "<|IMAGE_START|>", "<|IMAGE_END|>", ProjectorType.PADDLEOCR);
        frame(m, "<｜hy_place▁holder▁no▁100｜>", "<｜hy_place▁holder▁no▁101｜>",
                ProjectorType.HUNYUANVL);
        frame(m, "<vision>", "</vision>", ProjectorType.EXAONE4_5);
        return Collections.unmodifiableMap(m);
    }

    private static void frame(Map<ProjectorType, String[]> m, String begin, String end, ProjectorType... types) {
        for (ProjectorType t : types) {
            m.put(t, new String[] {begin, end});
        }
    }

    /**
     * {@code mtmd_decode_use_non_causal}: Gemma 3, Gemma 4 unified, SenseNova-U1 and DeepSeek-V4 image rows attend
     * each other in both directions, Gemma 4 rows too unless the projector is the E2B or E4B one (width 1536 or
     * 2560).
     */
    private static boolean nonCausal(ProjectorType projector, int projectorWidth) {
        switch (projector) {
            case GEMMA4V:
                return projectorWidth != 1536 && projectorWidth != 2560;
            case GEMMA4UV:
            case GEMMA3:
            case SENSENOVA_U1:
            case DEEPSEEK4V:
                return true;
            default:
                return false;
        }
    }

    private void image(String begin, String end) {
        imgBeg = begin;
        imgEnd = end;
    }

    /** MiniCPM-V: the 2.5 template for version 2, the 2.6 template for 3, 4, 5, 6 and 100045. */
    private void minicpmv(int version) {
        if (version == 2) {
            ovStart = "<image>";
            ovEnd = "</image>";
            slicesStart = "<slice>";
            slicesEnd = "</slice>";
            sliStart = ovStart;
            sliEnd = ovEnd;
            rowEnd = "\n";
            ovFirst = true;
        } else if (version == 3 || version == 4 || version == 5 || version == 6 || version == 100045) {
            minicpmv26();
        } else if (version != 0) {
            throw new IllegalArgumentException("unsupported minicpmv version: " + version);
        }
    }

    private void minicpmv26() {
        ovStart = "<image>";
        ovEnd = "</image>";
        sliStart = "<slice>";
        sliEnd = "</slice>";
        rowEnd = "\n";
        ovFirst = true;
    }

    /**
     * Adds the audio markers of an audio projector family.
     *
     * @param type the audio projector type
     */
    MediaMarkers withAudio(ProjectorType type) {
        MediaMarkers m = copy();
        switch (type) {
            case QWEN2A:
            case QWEN25O:
                m.audBeg = "<|audio_bos|>";
                m.audEnd = "<|audio_eos|>";
                break;
            case QWEN3A:
                m.audBeg = "<|audio_start|>";
                m.audEnd = "<|audio_end|>";
                break;
            case VOXTRAL:
                m.audBeg = "[BEGIN_AUDIO]";
                break;
            case MUSIC_FLAMINGO:
                m.audBeg = "<sound>";
                break;
            case GEMMA3NA:
                // Gemma3nProcessor's boa / eoa tokens around the audio soft tokens
                m.audBeg = "<start_of_audio>";
                m.audEnd = "<end_of_audio>";
                break;
            default:
                // ultravox, glma and meralion add no marker text
                break;
        }
        return m;
    }

    private MediaMarkers copy() {
        MediaMarkers m = new MediaMarkers();
        m.imgBeg = imgBeg;
        m.imgEnd = imgEnd;
        m.ovStart = ovStart;
        m.ovEnd = ovEnd;
        m.slicesStart = slicesStart;
        m.slicesEnd = slicesEnd;
        m.sliStart = sliStart;
        m.sliStartTmpl = sliStartTmpl;
        m.sliEnd = sliEnd;
        m.sliMid = sliMid;
        m.rowEnd = rowEnd;
        m.rowEndTrail = rowEndTrail;
        m.ovFirst = ovFirst;
        m.imageId = imageId;
        m.canvas = canvas;
        m.audBeg = audBeg;
        m.audEnd = audEnd;
        m.nonCausalImage = nonCausalImage;
        m.zeroBasedImageGrid = zeroBasedImageGrid;
        m.imageLeadPadAlign = imageLeadPadAlign;
        m.overviewViewSeparator = overviewViewSeparator;
        return m;
    }

    /** The text before an image's embeddings. */
    public String imageBegin() { return imgBeg; }

    /** The text after an image's embeddings. */
    public String imageEnd() { return imgEnd; }

    /** The text before the overview of a sliced image. */
    public String overviewStart() { return ovStart; }

    /** The text after the overview of a sliced image. */
    public String overviewEnd() { return ovEnd; }

    /** The text before the slice grid. */
    public String slicesStart() { return slicesStart; }

    /** The text after the slice grid. */
    public String slicesEnd() { return slicesEnd; }

    /** The fixed text before every slice; empty when the family names the slice's position instead. */
    public String sliceStart() { return sliStart; }

    /**
     * The text before the slice at 1-based {@code row} and {@code col}: the fixed slice start, or the family's
     * template naming the position ({@code sli_img_start_tmpl}, such as Idefics3's
     * {@code <fake_token_around_image><row_1_col_2>}).
     */
    public String sliceStart(int row, int col) {
        if (!sliStart.isEmpty() || sliStartTmpl.isEmpty()) {
            return sliStart;
        }
        return String.format(Locale.ROOT, sliStartTmpl, row, col);
    }

    /** The text after one slice. */
    public String sliceEnd() { return sliEnd; }

    /** The text between two slices of one row. */
    public String sliceMid() { return sliMid; }

    /** The text after a row of slices. */
    public String rowEnd() { return rowEnd; }

    /** Whether the last row of slices also ends with {@link #rowEnd()}. */
    public boolean rowEndTrail() { return rowEndTrail; }

    /** Whether the overview of a sliced image comes before its slices; only the MiniCPM-V templates set it. */
    public boolean overviewFirst() { return ovFirst; }

    /** Whether a sliced image starts with {@code <image_id>N</image_id>} (MiniCPM-V 4.6 / 4.7). */
    public boolean imageId() { return imageId; }

    /**
     * Whether a sliced image (an overview alone included) is one canvas chunk (MiniCPM-V 4.7,
     * {@code add_canvas_chunk}): the overview and the slices encoded with their separator rows
     * ({@link ClipSuffix}) and laid out as {@link #canvasLayout()}.
     */
    public boolean canvasImages() { return canvas; }

    /**
     * The prompt layout of one canvas image ({@code mtmd_tokenizer::add_media} for MiniCPM-V 4.7): the image-id
     * tag when {@link #imageId()}, the overview start text, then the canvas chunk, which carries the closing and
     * slice separators as rows.
     *
     * @throws IllegalStateException when the family has no canvas images
     */
    public List<LayoutPart> canvasLayout() {
        if (!canvas) {
            throw new IllegalStateException("this projector family lays its sliced images out as text and tiles,"
                    + " not as a canvas chunk");
        }
        List<LayoutPart> out = new ArrayList<>();
        if (imageId) {
            out.add(LayoutPart.IMAGE_ID);
        }
        addText(out, ovStart);
        out.add(LayoutPart.CANVAS);
        return Collections.unmodifiableList(out);
    }

    /**
     * The prompt layout of one sliced image with a {@code cols x rows} slice grid (both 0 for an overview alone),
     * the slice-template walk of {@code mtmd_tokenizer::add_media} (tools/mtmd/mtmd.cpp): the image-id tag and the
     * framed overview first when {@link #overviewFirst()}; the slices start text; per row, every slice framed by
     * its start and end texts with the mid text between two slices, and the row end after every row but the last
     * (after the last too with {@link #rowEndTrail()}); the slices end text; the framed overview last otherwise.
     * Empty texts are left out; the image begin and end texts frame the whole image outside this layout.
     *
     * @throws IllegalArgumentException for a negative grid side, or one side 0 and the other not
     */
    public List<LayoutPart> tiledLayout(int cols, int rows) {
        if (cols < 0 || rows < 0 || (cols == 0) != (rows == 0)) {
            throw new IllegalArgumentException("the slice grid " + cols + "x" + rows + " is not a grid");
        }
        List<LayoutPart> out = new ArrayList<>();
        if (ovFirst) {
            if (imageId) {
                out.add(LayoutPart.IMAGE_ID);
            }
            addOverview(out);
        }
        if (cols > 0) {
            addText(out, slicesStart);
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    addText(out, sliceStart(y + 1, x + 1));
                    out.add(LayoutPart.slice(y * cols + x));
                    addText(out, sliEnd);
                    if (x != cols - 1) {
                        addText(out, sliMid);
                    }
                }
                if (y != rows - 1 || rowEndTrail) {
                    addText(out, rowEnd);
                }
            }
            addText(out, slicesEnd);
        }
        if (!ovFirst) {
            addOverview(out);
        }
        return Collections.unmodifiableList(out);
    }

    private void addOverview(List<LayoutPart> out) {
        addText(out, ovStart);
        out.add(LayoutPart.OVERVIEW);
        addText(out, ovEnd);
    }

    private static void addText(List<LayoutPart> out, String text) {
        if (!text.isEmpty()) {
            out.add(LayoutPart.text(text));
        }
    }

    /** The text before an audio clip's embeddings. */
    public String audioBegin() { return audBeg; }

    /** The text after an audio clip's embeddings. */
    public String audioEnd() { return audEnd; }

    /** Whether image rows decode without the causal mask ({@code mtmd_decode_use_non_causal}). */
    public boolean nonCausalImage() { return nonCausalImage; }

    /**
     * Whether an image grid takes the zero-based M-RoPE layout of SenseNova-U1 ({@code MTMD_POS_TYPE_SENSENOVA_U1}):
     * y and x count from 0 rather than from the chunk's base position, and the whole image occupies one position
     * ({@link com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions#image2dZeroBased}).
     */
    public boolean zeroBasedImageGrid() { return zeroBasedImageGrid; }

    /**
     * The token alignment an image's rows start at, or 0 when the family adds no lead pads: DeepSeek-V4 opens its
     * token block with {@code align - 1 - n % align} pad rows after {@code n} prompt tokens
     * ({@code mtmd_tokenize}'s {@code lead_pad}).
     */
    public int imageLeadPadAlign() { return imageLeadPadAlign; }

    /**
     * Whether the overview of a sliced image encodes as the global view whose rows end with the view separator
     * ({@link ClipImage#withViewSeparator}; DeepSeek-OCR and DeepSeek-OCR 2).
     */
    public boolean overviewViewSeparator() { return overviewViewSeparator; }

    /**
     * One part of a sliced image's layout: marker text, the image-id tag, the overview, one slice or the canvas
     * chunk. Immutable.
     */
    public static final class LayoutPart {

        /** What a part holds. */
        public enum Kind {
            /** Marker text, tokenized with special tokens parsed. */
            TEXT,
            /** The {@code <image_id>N</image_id>} tag of the image's index. */
            IMAGE_ID,
            /** The overview's embedding rows. */
            OVERVIEW,
            /** One slice's embedding rows. */
            SLICE,
            /** The canvas chunk: the overview's and every slice's rows, each followed by its separator rows. */
            CANVAS
        }

        static final LayoutPart IMAGE_ID = new LayoutPart(Kind.IMAGE_ID, null, -1);
        static final LayoutPart OVERVIEW = new LayoutPart(Kind.OVERVIEW, null, -1);
        static final LayoutPart CANVAS = new LayoutPart(Kind.CANVAS, null, -1);

        private final Kind kind;
        private final String text;
        private final int slice;

        private LayoutPart(Kind kind, String text, int slice) {
            this.kind = kind;
            this.text = text;
            this.slice = slice;
        }

        static LayoutPart text(String text) {
            return new LayoutPart(Kind.TEXT, text, -1);
        }

        static LayoutPart slice(int index) {
            return new LayoutPart(Kind.SLICE, null, index);
        }

        /** What the part holds. */
        public Kind kind() { return kind; }

        /** The marker text of a {@link Kind#TEXT} part; null otherwise. */
        public String text() { return text; }

        /** The row-major slice index of a {@link Kind#SLICE} part; -1 otherwise. */
        public int slice() { return slice; }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof LayoutPart)) return false;
            LayoutPart that = (LayoutPart) o;
            return kind == that.kind && slice == that.slice && Objects.equals(text, that.text);
        }

        @Override
        public int hashCode() {
            return Objects.hash(kind, text, slice);
        }

        @Override
        public String toString() {
            switch (kind) {
                case TEXT: return "text(" + text.replace("\n", "\\n") + ")";
                case SLICE: return "slice(" + slice + ")";
                default: return kind.name().toLowerCase(Locale.ROOT);
            }
        }
    }
}
