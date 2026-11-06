package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.hooks.HfClipHooks;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.image.ImageDecoder;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;
import java.util.Objects;

/**
 * A CLIP dual-encoder embedding model, vLLM's {@code CLIPEmbeddingModel} (vllm/model_executor/models/clip.py), over
 * the two-tower clip GGUF of llama.cpp's legacy {@code convert_image_encoder_to_gguf.py}: {@code general.architecture}
 * {@code clip}, no projector type, a text tower ({@code clip.has_text_encoder}, {@code t.*} tensors and
 * {@code text_projection.weight}) and/or a vision tower ({@code clip.has_vision_encoder}, {@code v.*} tensors and
 * {@code visual_projection.weight}). Text and images embed into one space of {@link #embeddingLength()} values, so
 * their cosine similarity scores an image against a caption. The sequence pooling is {@code clip.pooling_type}
 * (default LAST, vLLM's: the end-of-text token on the text side, the class token on the image side; CLS and MEAN
 * as {@link ImageFeatureSelect} maps them); the embeddings come unnormalized, as the pooler's input, and
 * {@link #score} turns their cosine similarity into the model's score. A {@link ClipFamily#SIGLIP} file
 * ({@value ClipFamily#KEY}) is vLLM's {@code SiglipEmbeddingModel} (vllm/model_executor/models/siglip.py): the text
 * tower attends bidirectionally over the input padded to the context, pools its last row and applies the text head,
 * the image tower pools through its attention pooling head, and the score is the pair's sigmoid probability. The
 * same layout loads from a Hugging Face {@code CLIPModel} or {@code SiglipModel} checkpoint directory through the
 * Hugging Face loader. Every tensor resolves and shape-checks at {@link #open}; the towers' blocks run on the device
 * the {@link MmprojOptions} choose ({@link #device()}), else on the CPU in the stored weight types, and the input
 * rows, pooling, heads and projections on the host. Thread-safe: the embed calls serialize. Close it to release the
 * device state, the weights and the file.
 */
public final class ClipDualEncoder implements AutoCloseable {

    /** The {@code clip.<suffix>} key of the sequence pooling (llama.cpp {@code %s.pooling_type}). */
    public static final String POOLING_KEY = "clip.pooling_type";
    /** The {@code clip.text.context_length} key of the text positions. */
    public static final String CONTEXT_KEY = "clip.text.context_length";
    /** The SigLIP tensor of the log temperature that scales the cosine similarity. */
    public static final String LOGIT_SCALE = "logit_scale";
    /** The SigLIP tensor of the bias added to the scaled cosine similarity. */
    public static final String LOGIT_BIAS = "logit_bias";

    private final String path;
    private final String name;
    private final ModelSource source;
    private final CpuWeights weights;
    private final ClipTextTower text;
    private final ClipImageTower vision;
    private final GpuDualTowers gpu;
    private final TowerBlocks textBlocks;
    private final TowerBlocks visionBlocks;
    private final GGUFTokenizer tokenizer;
    private final PoolingKind pooling;
    private final ClipFamily family;
    private final Logits logits;
    private final int padId;
    private boolean closed;

    private ClipDualEncoder(String path, String name, ModelSource source, CpuWeights weights, Towers towers,
                            GGUFTokenizer tokenizer, PoolingKind pooling, ClipFamily family, Logits logits,
                            int padId) {
        this.path = path;
        this.name = name;
        this.source = source;
        this.weights = weights;
        this.text = towers.text;
        this.vision = towers.vision;
        this.gpu = towers.gpu;
        this.textBlocks = text == null ? null : gpu != null ? gpu::text : text.cpuBlocks();
        this.visionBlocks = vision == null ? null : gpu != null ? gpu::vision : vision.cpuBlocks();
        this.tokenizer = tokenizer;
        this.pooling = pooling;
        this.family = family;
        this.logits = logits;
        this.padId = padId;
    }

    /** The bound towers and their device executor (null on the CPU). Immutable. */
    private static final class Towers {
        final ClipTextTower text;
        final ClipImageTower vision;
        final GpuDualTowers gpu;

        Towers(ClipTextTower text, ClipImageTower vision, GpuDualTowers gpu) {
            this.text = text;
            this.vision = vision;
            this.gpu = gpu;
        }
    }

    /** SigLIP's {@code logit_scale} (a log temperature) and {@code logit_bias}. Immutable. */
    private static final class Logits {
        final double scale;
        final double bias;

        Logits(double logScale, double bias) {
            this.scale = Math.exp(logScale);
            this.bias = bias;
        }
    }

    /**
     * True when {@code kv} describes a two-tower file: architecture {@code clip}, a text or vision tower, no
     * audio encoder, no {@code clip.projector_type} (common or vision) and no llava projector.
     */
    public static boolean isTwoTower(GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        return "clip".equals(kv.archName())
                && (kv.bool("clip.has_text_encoder", false) || kv.bool("clip.has_vision_encoder", false))
                && !kv.bool("clip.has_audio_encoder", false)
                && !kv.bool("clip.has_llava_projector", false)
                && !kv.has(GgufKeys.CLIP_PROJECTOR_TYPE)
                && !kv.has("clip.vision.projector_type");
    }

    /**
     * True when {@code path} names a GGUF file that {@link #isTwoTower(GgufKv) is a two-tower file}, or a Hugging
     * Face checkpoint (a directory, or a {@code .safetensors} file beside its {@code config.json}) whose
     * architecture is a dual encoder the Hugging Face loader converts ({@link HfClipHooks#DUAL_ENCODERS}); a
     * missing path and any other model are not.
     *
     * @throws IOException when a GGUF file or a checkpoint configuration cannot be parsed
     */
    public static boolean isTwoTower(String path) throws IOException {
        if (path == null || path.isBlank()) {
            return false;
        }
        File file = new File(path);
        if (file.isDirectory()) {
            return isDualCheckpoint(file.toPath());
        }
        if (!file.isFile()) {
            return false;
        }
        if (ModelSources.detect(Paths.get(path)) != ModelFormat.GGUF) {
            return file.getAbsoluteFile().getParentFile() != null
                    && isDualCheckpoint(file.getAbsoluteFile().getParentFile().toPath());
        }
        try (ModelSource probe = ModelSources.parseGguf(path)) {
            return isTwoTower(GgufKv.of(probe));
        }
    }

    private static boolean isDualCheckpoint(Path dir) throws IOException {
        if (!Files.isRegularFile(dir.resolve(HfConfig.CONFIG_JSON))) {
            return false;
        }
        return HfClipHooks.DUAL_ENCODERS.contains(HfConfigLoader.load(dir, Collections.emptyMap()).architecture());
    }

    /** {@link #open(String, ModelSourceOptions)} with the process default options. */
    public static ClipDualEncoder open(String path) throws IOException {
        return open(path, ModelSourceOptions.fromSystemProperties());
    }

    /**
     * Opens the two-tower file at {@code path} under the load {@code options} (metadata overrides, the tokenizer
     * source of {@link ClipDualVocab}), the towers on the CPU.
     *
     * @throws IllegalArgumentException when the file is no two-tower clip file, a tensor shape differs from its
     *                                  tower, or both towers embed to different widths
     * @throws GgufFormatException      for a missing, malformed or unsupported key (the pooling type named)
     * @throws IOException              when the file or the tokenizer source cannot be read
     */
    public static ClipDualEncoder open(String path, ModelSourceOptions options) throws IOException {
        return open(path, options, MmprojOptions.cpu());
    }

    /**
     * As {@link #open(String, ModelSourceOptions)}, the towers' blocks on the device {@code device} chooses as an
     * mmproj's vision encoder does ({@link GpuDualTowers#select}): a GPU that runs both towers gets their blocks and
     * norms, the host keeps the input rows, pooling, heads and projections; otherwise the towers run on the CPU.
     *
     * @throws IllegalArgumentException    when the file is no two-tower clip file, a tensor shape differs from its
     *                                     tower, both towers embed to different widths, or the device selector
     *                                     names no usable device
     * @throws GgufFormatException         for a missing, malformed or unsupported key (the pooling type named)
     * @throws IOException                 when the file or the tokenizer source cannot be read
     * @throws com.eonmux.neuralfree.acceleration.NoBackendAvailableException when an explicit backend mode has no
     *                                     usable device
     */
    public static ClipDualEncoder open(String path, ModelSourceOptions options, MmprojOptions device)
            throws IOException {
        Objects.requireNonNull(path, "path");
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(device, "device");
        ModelSource source = ModelSources.open(path, options);
        CpuWeights weights = null;
        GpuDualTowers gpu = null;
        try {
            GgufKv kv = GgufKv.of(source);
            if (!isTwoTower(kv)) {
                throw new IllegalArgumentException("'" + path + "' is not a CLIP two-tower file: it needs"
                        + " general.architecture 'clip', clip.has_text_encoder or clip.has_vision_encoder, and no"
                        + " projector (clip.projector_type, clip.has_llava_projector)");
            }
            PoolingKind pooling = pooling(kv);
            ClipFamily family = ClipFamily.of(kv);
            TensorCatalog tc = TensorCatalog.of(source);
            InputTransformLoader.Result transforms = InputTransformLoader.load(source, kv, tc,
                    ActivationQuant.WEIGHT_ONLY);
            weights = CpuWeights.open(source, tc, transforms);
            ClipHparams textHp = kv.bool("clip.has_text_encoder", false)
                    ? ClipHparams.twoTower(kv, ProjectorModality.TEXT) : null;
            ClipHparams visionHp = kv.bool("clip.has_vision_encoder", false)
                    ? ClipHparams.twoTower(kv, ProjectorModality.VISION) : null;
            int nCtx = textHp == null ? 0 : contextLength(kv);
            Towers cpu = bindTowers(path, textHp, visionHp, nCtx, family, tc, weights);
            gpu = GpuDualTowers.select(
                    textHp == null ? null : ClipVisionModel.twoTower(textHp, tc, family, nCtx),
                    visionHp == null ? null : ClipVisionModel.twoTower(visionHp, tc, family, 0),
                    family == ClipFamily.CLIP, device, source, transforms);
            ClipTextTower text = cpu.text;
            GGUFTokenizer tokenizer = text == null ? null
                    : ClipDualVocab.resolve(source, kv, options.tokenizerDir(), text.vocabRows());
            String name = kv.str("general.name", new File(path).getName());
            Logits logits = family == ClipFamily.SIGLIP ? logits(tc, weights) : null;
            int padId = family == ClipFamily.SIGLIP && text != null ? padId(kv, text.vocabRows()) : -1;
            return new ClipDualEncoder(path, name, source, weights, new Towers(text, cpu.vision, gpu), tokenizer,
                    pooling, family, logits, padId);
        } catch (IOException | RuntimeException | Error e) {
            closeQuietly(gpu, weights, source, e);
            throw e;
        }
    }

    /**
     * The host towers of the file: the text tower over {@code nCtx} positions (causal for {@link ClipFamily#CLIP})
     * and the vision tower, each bound when its hparams are present.
     *
     * @throws IllegalArgumentException when both towers embed to different widths
     */
    private static Towers bindTowers(String path, ClipHparams textHp, ClipHparams visionHp, int nCtx,
                                     ClipFamily family, TensorCatalog tc, CpuWeights weights) {
        ClipTextTower text = textHp == null ? null : ClipTextTower.bind(textHp, tc, nCtx,
                family == ClipFamily.CLIP, weights.matVec(), weights::floats, weights::get);
        ClipImageTower vision = visionHp == null ? null : ClipImageTower.bind(visionHp, family, tc,
                weights.matVec(), weights::floats, weights::get);
        if (text != null && vision != null && text.outputDim() != vision.outputDim()) {
            throw new IllegalArgumentException("'" + path + "': the text tower embeds to " + text.outputDim()
                    + " values and the vision tower to " + vision.outputDim() + "; a dual encoder shares one"
                    + " embedding space");
        }
        return new Towers(text, vision, null);
    }

    private static PoolingKind pooling(GgufKv kv) {
        if (!kv.has(POOLING_KEY)) {
            return PoolingKind.LAST;
        }
        PoolingKind kind;
        try {
            kind = PoolingKind.fromGgufCode(kv.u32Long(POOLING_KEY, PoolingKind.GGUF_CODE_UNSPECIFIED));
        } catch (IllegalArgumentException e) {
            throw new GgufFormatException("metadata key '" + POOLING_KEY + "': " + e.getMessage(), e);
        }
        if (kind == null) {
            return PoolingKind.LAST;
        }
        if (kind != PoolingKind.CLS && kind != PoolingKind.LAST && kind != PoolingKind.MEAN) {
            throw new GgufFormatException("metadata key '" + POOLING_KEY + "' is "
                    + kind.name().toLowerCase(Locale.ROOT) + "; a CLIP dual encoder pools by cls, last or mean");
        }
        return kind;
    }

    /**
     * SigLIP's {@value #LOGIT_SCALE} and {@value #LOGIT_BIAS}, one value each.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when either is absent
     * @throws IllegalArgumentException when either holds other than one value
     */
    private static Logits logits(TensorCatalog tc, CpuWeights weights) {
        double scale = scalar(tc, weights, LOGIT_SCALE);
        double bias = scalar(tc, weights, LOGIT_BIAS);
        return new Logits(scale, bias);
    }

    private static double scalar(TensorCatalog tc, CpuWeights weights, String name) {
        TensorRef r = tc.require(name);
        if (r.numElements() != 1) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + name + " has shape "
                    + r + ", the loader expects one value");
        }
        return weights.floats(name)[0];
    }

    /**
     * The token the SigLIP text pads with ({@code tokenizer.ggml.padding_token_id}, else the end-of-text token):
     * SigLIP trains on text padded to the context, so its embeddings compare only so.
     *
     * @throws GgufFormatException when the file names neither, or the id is outside the vocabulary
     */
    private static int padId(GgufKv kv, int vocabRows) {
        String key = kv.has(GgufKeys.TOKENIZER_PAD_ID) ? GgufKeys.TOKENIZER_PAD_ID : GgufKeys.TOKENIZER_EOS_ID;
        if (!kv.has(key)) {
            throw new GgufFormatException("a " + ClipFamily.SIGLIP + " text tower pads its input to the context;"
                    + " the file needs " + GgufKeys.TOKENIZER_PAD_ID + " or " + GgufKeys.TOKENIZER_EOS_ID);
        }
        int id = kv.u32(key, -1);
        if (id < 0 || id >= vocabRows) {
            throw new GgufFormatException("metadata key '" + key + "' is " + id + ", outside the vocabulary of "
                    + vocabRows);
        }
        return id;
    }

    private static int contextLength(GgufKv kv) {
        if (!kv.has(CONTEXT_KEY)) {
            throw new GgufFormatException("required metadata key '" + CONTEXT_KEY + "' is missing");
        }
        int n = kv.u32(CONTEXT_KEY, 0);
        if (n <= 0) {
            throw new GgufFormatException("metadata key '" + CONTEXT_KEY + "' must be positive, got " + n);
        }
        return n;
    }

    private static void closeQuietly(GpuDualTowers gpu, CpuWeights weights, ModelSource source,
                                     Throwable failure) {
        try {
            if (gpu != null) {
                gpu.close();
            }
        } catch (RuntimeException e) {
            failure.addSuppressed(e);
        }
        try {
            if (weights != null) {
                weights.close();
            }
        } catch (IOException | RuntimeException e) {
            failure.addSuppressed(e);
        }
        try {
            source.close();
        } catch (IOException | RuntimeException e) {
            failure.addSuppressed(e);
        }
    }

    /** Where the towers' blocks run: {@code "cpu"}, or the GPU device's name. */
    public String device() {
        return gpu == null ? "cpu" : gpu.device();
    }

    /** The opened path. */
    public String path() {
        return path;
    }

    /** The model name ({@code general.name}, else the file name). */
    public String name() {
        return name;
    }

    /** The architecture name, {@code clip}. */
    public String architecture() {
        return "clip";
    }

    /** The model family of the towers ({@value ClipFamily#KEY}). */
    public ClipFamily family() {
        return family;
    }

    /**
     * The model's score of a text-image or same-tower pair from the {@code cosine} similarity of their
     * embeddings: the cosine itself for {@link ClipFamily#CLIP}; for {@link ClipFamily#SIGLIP} the pair's
     * probability {@code sigmoid(cosine * exp(logit_scale) + logit_bias)}, HF {@code SiglipModel}'s
     * {@code logits_per_text} under its sigmoid loss.
     */
    public double score(double cosine) {
        if (logits == null) {
            return cosine;
        }
        return 1.0 / (1.0 + Math.exp(-(cosine * logits.scale + logits.bias)));
    }

    /** True when the file carries the text tower. */
    public boolean hasText() {
        return text != null;
    }

    /** True when the file carries the vision tower. */
    public boolean hasVision() {
        return vision != null;
    }

    /** The sequence pooling of both towers. */
    public PoolingKind pooling() {
        return pooling;
    }

    /** The width of every embedding: the projection width of the towers. */
    public int embeddingLength() {
        return text != null ? text.outputDim() : vision.outputDim();
    }

    /** The positions of the text tower ({@code clip.text.context_length}); 0 without a text tower. */
    public int contextLength() {
        return text == null ? 0 : text.contextLength();
    }

    /** The token-embedding rows of the text tower; 0 without a text tower. */
    public int vocabSize() {
        return text == null ? 0 : text.vocabRows();
    }

    /** The square tile side of the vision tower ({@code clip.vision.image_size}); 0 without a vision tower. */
    public int imageSize() {
        return vision == null ? 0 : vision.imageSize();
    }

    /** True when text tokenizes ({@link ClipDualVocab}). */
    public boolean hasTokenizer() {
        return tokenizer != null;
    }

    /**
     * The tokenizer of the text tower.
     *
     * @throws UnsupportedOperationException when the file has no text tower, or no tokenizer reached it (named)
     */
    public GGUFTokenizer tokenizer() {
        requireText();
        if (tokenizer == null) {
            throw new UnsupportedOperationException(ClipDualVocab.missingReason(path));
        }
        return tokenizer;
    }

    /**
     * The embedding of one tokenized text: the text tower over {@code tokens} (the tokenizer's start and end
     * tokens included, as {@link GGUFTokenizer#encode} adds them), pooled and projected. A SigLIP tower reads the
     * tokens padded to its context and pools the last row (CLS and LAST, vLLM's sequence flip) or the mean.
     *
     * @throws UnsupportedOperationException when the file has no text tower
     * @throws IllegalArgumentException      when the input is empty, longer than the context, or an id is outside
     *                                       the vocabulary
     */
    public synchronized float[] embedText(int[] tokens) {
        ensureOpen();
        Objects.requireNonNull(tokens, "tokens");
        requireText();
        if (family != ClipFamily.SIGLIP) {
            return text.embed(tokens, pooling, textBlocks);
        }
        return text.embed(padded(tokens), pooling == PoolingKind.MEAN ? pooling : PoolingKind.LAST, textBlocks);
    }

    /** {@code tokens} padded with the pad token to the context, as SigLIP's {@code padding="max_length"}. */
    private int[] padded(int[] tokens) {
        int n = text.contextLength();
        if (tokens.length == 0 || tokens.length >= n) {
            return tokens;
        }
        int[] out = Arrays.copyOf(tokens, n);
        Arrays.fill(out, tokens.length, n, padId);
        return out;
    }

    /**
     * The embedding of one encoded image (PNG, JPEG, BMP, GIF, WebP).
     *
     * @throws UnsupportedOperationException when the file has no vision tower
     * @throws com.eonmux.neuralfree.gguf.image.ImageDecodeException when the bytes do not decode
     */
    public float[] embedImage(byte[] encoded) {
        Objects.requireNonNull(encoded, "encoded");
        requireVision();
        return embedImage(ImageDecoder.decode(encoded));
    }

    /**
     * The embedding of one decoded image: CLIP's preprocessing, the vision tower, the rows of the pooling's
     * {@link ImageFeatureSelect}, and the projection.
     *
     * @throws UnsupportedOperationException when the file has no vision tower
     */
    public synchronized float[] embedImage(RasterImage image) {
        ensureOpen();
        Objects.requireNonNull(image, "image");
        requireVision();
        return vision.embed(image, ImageFeatureSelect.of(pooling), visionBlocks);
    }

    private void requireText() {
        if (text == null) {
            throw new UnsupportedOperationException("the CLIP file '" + path + "' has no text tower"
                    + " (clip.has_text_encoder is absent or false); it embeds images only");
        }
    }

    private void requireVision() {
        if (vision == null) {
            throw new UnsupportedOperationException("the CLIP file '" + path + "' has no vision tower"
                    + " (clip.has_vision_encoder is absent or false); it embeds text only");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the CLIP dual encoder '" + path + "' is closed");
        }
    }

    @Override
    public synchronized void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        try {
            if (gpu != null) {
                gpu.close();
            }
        } finally {
            try {
                weights.close();
            } finally {
                source.close();
            }
        }
    }

    @Override
    public String toString() {
        return "ClipDualEncoder{" + name + ", " + family + ", text=" + (text != null) + ", vision=" + (vision != null)
                + ", dim=" + embeddingLength() + ", pooling=" + pooling + ", device=" + device() + "}";
    }
}
