package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.MelFrontend;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkNorm;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.elements;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * Validated geometry of one audio encoder and its projector over an mmproj tensor table: the counterpart of
 * llama.cpp {@code clip_model} (tools/mtmd/clip-model.h) resolved for the audio graphs of tools/mtmd/models.
 * Every tensor the graphs read is resolved by name and shape-checked against the {@link ClipHparams} where the
 * reference loader or graph relies on it; a mismatch fails at construction with the tensor name, before any
 * weight is loaded. The projector output width must equal {@code clip.audio.projection_dim}, the text model's
 * embedding width. The whisper and qwen3a tensors are resolved here; the Gemma 3n conformer (gemma3na, which
 * llama.cpp does not run) resolves through {@link Gemma3nAudioModel}; every other family's tensors are resolved by
 * its tower class ({@code ConformerTower}, {@code Lfm3aTower}, ...), which the executors bind.
 *
 * <p>Immutable; tensor references view the file, the executor ({@link CpuAudioEncoder}) copies values.
 */
public final class ClipAudioModel {

    /** Kernel width of the whisper conv1d blocks and the qwen3a conv2d blocks. */
    static final int CONV_K = 3;
    /** Frames per qwen3a conv batch ({@code n_window * 2}, qwen3a.cpp). */
    public static final int QWEN3A_CHUNK_FRAMES = 100;
    /** Tokens one qwen3a conv batch produces (three stride-2 convs over 100 frames). */
    public static final int QWEN3A_TOKENS_PER_CHUNK = 13;

    private final ClipHparams hparams;
    private final String encoderGraph;
    private final String projectorGraph;
    private final int stackFactor;
    private final int maxPos;
    private final int[] convChannels;
    private final int convHeight;
    private final List<ClipLayer> layers;
    private final Tensors t;
    private final TensorRef melFilters;
    private final TensorRef melWindow;
    private final TensorCatalog catalog;
    private final Gemma3nAudioModel gemma3n;

    /** The resolved tensors; absent optionals and tensors of other graphs are null. */
    static final class Tensors {
        TensorRef conv1W;
        TensorRef conv1B;
        TensorRef conv2W;
        TensorRef conv2B;
        TensorRef conv3W;
        TensorRef conv3B;
        TensorRef convOutW;
        TensorRef convOutB;
        TensorRef posEmbd;
        TensorRef preLnW;
        TensorRef preLnB;
        TensorRef postLnW;
        TensorRef postLnB;
        TensorRef normPreW;
        TensorRef normPreB;
        TensorRef normMidW;
        TensorRef fcW;
        TensorRef fcB;
        TensorRef mm0W;
        TensorRef mm0B;
        TensorRef mm1W;
        TensorRef mm1B;
        TensorRef mm2W;
        TensorRef mm2B;
        TensorRef mm3W;
        TensorRef mm3B;
        TensorRef boi;
        TensorRef eoi;
        TensorRef melFilters;
        TensorRef melWindow;
    }

    private ClipAudioModel(ClipHparams hparams, ProjectorEntry entry, int maxPos, int[] convChannels,
                           int convHeight, List<ClipLayer> layers, Tensors t, TensorCatalog catalog,
                           Gemma3nAudioModel gemma3n) {
        this.hparams = hparams;
        this.encoderGraph = entry.encoderGraph();
        this.projectorGraph = entry.projectorGraph();
        this.stackFactor = Math.max(1, hparams.audio().stackFactor());
        this.maxPos = maxPos;
        this.convChannels = convChannels;
        this.convHeight = convHeight;
        this.layers = Collections.unmodifiableList(layers);
        this.t = t;
        this.melFilters = t.melFilters;
        this.melWindow = t.melWindow;
        this.catalog = catalog;
        this.gemma3n = gemma3n;
    }

    /**
     * Geometry of {@code hp} over the tensors of {@code tc}.
     *
     * @throws IllegalArgumentException for a null argument, an hparams modality other than
     *                                  {@link ProjectorModality#AUDIO}, a projector type without a runnable audio
     *                                  registry row, or a tensor shape that differs from the graph's
     * @throws MissingTensorException   when a required tensor is absent
     */
    public static ClipAudioModel from(ClipHparams hp, TensorCatalog tc) {
        Objects.requireNonNull(hp, "hparams");
        Objects.requireNonNull(tc, "tensors");
        if (hp.modality() != ProjectorModality.AUDIO) {
            throw new IllegalArgumentException("ClipAudioModel resolves the audio encoder, got " + hp.modality());
        }
        ProjectorEntry entry = ProjectorRegistry.get(hp.projectorType().typeName());
        if (entry.modality() != ProjectorModality.AUDIO) {
            throw new IllegalArgumentException("projector type '" + hp.projectorType() + "' is a "
                    + entry.modality() + " projector, the file declares it for its audio encoder");
        }
        entry.tensors().validate(tc, hp.nLayer());
        switch (entry.encoderGraph()) {
            case ProjectorEntry.ENCODER_WHISPER:
                return whisper(hp, entry, tc);
            case ProjectorEntry.ENCODER_QWEN3A:
                return qwen3a(hp, entry, tc);
            case ProjectorEntry.ENCODER_GEMMA3N_USM:
                return new ClipAudioModel(hp, entry, 0, null, 0, List.of(), new Tensors(), tc,
                        Gemma3nAudioModel.resolve(hp, tc));
            default:
                return family(hp, entry, tc);
        }
    }

    /**
     * The geometry of a tower whose tensors its family class resolves and shape-checks; only the parakeet
     * filterbank and window, which the mel frontend reads, are kept here.
     */
    private static ClipAudioModel family(ClipHparams hp, ProjectorEntry entry, TensorCatalog tc) {
        Tensors t = new Tensors();
        switch (entry.encoderGraph()) {
            case ProjectorEntry.ENCODER_CONFORMER:
                ConformerTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_LFM3A:
                Lfm3aTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_GEMMA4A:
                Gemma4aTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_GEMMA4UA:
                Gemma4uaTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_PARAKEET:
                ParakeetTower.validate(hp, tc);
                t.melFilters = tc.require(ParakeetTower.MEL_FILTERS);
                t.melWindow = tc.require(ParakeetTower.WINDOW);
                break;
            case ProjectorEntry.ENCODER_GRANITE_SPEECH:
                GraniteSpeechTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_VOXTRAL_RT:
                VoxtralRtTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_MIMO_AUDIO:
                MimoAudioTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO:
                MuseSparkTower.validate(hp, tc);
                break;
            case ProjectorEntry.ENCODER_DOTS3NOTE_A:
                Dots3NoteATower.validate(hp, tc);
                break;
            default:
                throw new IllegalStateException("unknown audio encoder graph " + entry.encoderGraph());
        }
        return new ClipAudioModel(hp, entry, 0, null, 0, List.of(), t, tc, null);
    }

    /** The whisper-family geometry: position rows, the two conv1d blocks, the ViT blocks and the projector. */
    private static ClipAudioModel whisper(ClipHparams hp, ProjectorEntry entry, TensorCatalog tc) {
        Tensors t = new Tensors();
        t.posEmbd = tc.require("a.position_embd.weight");
        int maxPos = (int) t.posEmbd.ne(1);
        shape(tc, t.posEmbd, hp.nEmbd(), maxPos);
        prePostLns(tc, hp, t);
        whisperFront(tc, hp, t);
        List<ClipLayer> layers = layers(tc, hp, true);
        projector(tc, hp, entry.projectorGraph(), Math.max(1, hp.audio().stackFactor()), t);
        return new ClipAudioModel(hp, entry, maxPos, null, 0, layers, t, tc, null);
    }

    /** The qwen3a geometry: three conv2d blocks, conv_out, one position table, the ViT blocks and the MLP. */
    private static ClipAudioModel qwen3a(ClipHparams hp, ProjectorEntry entry, TensorCatalog tc) {
        Tensors t = new Tensors();
        t.posEmbd = tc.require("a.position_embd.weight");
        int maxPos = (int) t.posEmbd.ne(1);
        shape(tc, t.posEmbd, hp.nEmbd(), maxPos);
        prePostLns(tc, hp, t);
        int[] channels = qwen3aFront(tc, hp, t);
        int convHeight = convOut(convOut(convOut(hp.audio().nMelBins())));
        shape(tc, t.convOutW, (long) convHeight * channels[2], hp.nEmbd());
        if (maxPos < QWEN3A_TOKENS_PER_CHUNK) {
            throw new IllegalArgumentException("tensor a.position_embd.weight holds " + maxPos
                    + " rows, qwen3a needs at least " + QWEN3A_TOKENS_PER_CHUNK);
        }
        List<ClipLayer> layers = layers(tc, hp, false);
        projector(tc, hp, entry.projectorGraph(), Math.max(1, hp.audio().stackFactor()), t);
        return new ClipAudioModel(hp, entry, maxPos, channels, convHeight, layers, t, tc, null);
    }

    private static void prePostLns(TensorCatalog tc, ClipHparams hp, Tensors t) {
        t.preLnW = tc.get("a.pre_ln.weight");
        t.preLnB = tc.get("a.pre_ln.bias");
        checkNorm(tc, "a.pre_ln", t.preLnW, t.preLnB, hp.nEmbd());
        t.postLnW = tc.get("a.post_ln.weight");
        t.postLnB = tc.get("a.post_ln.bias");
        checkNorm(tc, "a.post_ln", t.postLnW, t.postLnB, hp.nEmbd());
    }

    private static List<ClipLayer> layers(TensorCatalog tc, ClipHparams hp, boolean whisper) {
        List<ClipLayer> layers = new ArrayList<>(hp.nLayer());
        for (int il = 0; il < hp.nLayer(); il++) {
            ClipLayer l = ClipLayer.resolve(tc, hp, il, false);
            if (whisper && l.kBias() != null) {
                throw new IllegalArgumentException("tensor " + l.kBias().name() + ": the whisper encoder graph"
                        + " has no key bias");
            }
            layers.add(l);
        }
        return layers;
    }

    /** The two conv1d blocks: kernels {@code [3, in, n_embd]}, biases {@code [1, n_embd]} (whisper-enc.cpp). */
    private static void whisperFront(TensorCatalog tc, ClipHparams hp, Tensors t) {
        int e = hp.nEmbd();
        t.conv1W = tc.require("a.conv1d.1.weight");
        t.conv1B = tc.require("a.conv1d.1.bias");
        t.conv2W = tc.require("a.conv1d.2.weight");
        t.conv2B = tc.require("a.conv1d.2.bias");
        shape(tc, t.conv1W, CONV_K, hp.audio().nMelBins(), e);
        shape(tc, t.conv1B, 1, e);
        shape(tc, t.conv2W, CONV_K, e, e);
        shape(tc, t.conv2B, 1, e);
    }

    /** The three conv2d blocks: kernels {@code [3, 3, in, out]} chained from one mel channel (qwen3a.cpp). */
    private static int[] qwen3aFront(TensorCatalog tc, ClipHparams hp, Tensors t) {
        t.conv1W = tc.require("a.conv2d.1.weight");
        t.conv1B = tc.require("a.conv2d.1.bias");
        t.conv2W = tc.require("a.conv2d.2.weight");
        t.conv2B = tc.require("a.conv2d.2.bias");
        t.conv3W = tc.require("a.conv2d.3.weight");
        t.conv3B = tc.require("a.conv2d.3.bias");
        t.convOutW = tc.require("a.conv_out.weight");
        t.convOutB = tc.get("a.conv_out.bias");
        int c1 = (int) t.conv1W.ne(3);
        int c2 = (int) t.conv2W.ne(3);
        int c3 = (int) t.conv3W.ne(3);
        shape(tc, t.conv1W, CONV_K, CONV_K, 1, c1);
        shape(tc, t.conv2W, CONV_K, CONV_K, c1, c2);
        shape(tc, t.conv3W, CONV_K, CONV_K, c2, c3);
        elements(tc, t.conv1B, c1);
        elements(tc, t.conv2B, c2);
        elements(tc, t.conv3B, c3);
        checkVector(tc, t.convOutB, hp.nEmbd());
        return new int[]{c1, c2, c3};
    }

    /** The output size of a kernel-3, stride-2, padding-1 convolution. */
    static int convOut(int size) {
        return (size + 2 - CONV_K) / 2 + 1;
    }

    private static void projector(TensorCatalog tc, ClipHparams hp, String graph, int stack, Tensors t) {
        int e = hp.nEmbd();
        int d = hp.projectionDim();
        switch (graph) {
            case ProjectorEntry.PROJECTOR_ULTRAVOX: {
                t.normPreW = tc.require("mm.a.norm_pre.weight");
                t.mm1W = tc.require("mm.a.mlp.1.weight");
                t.normMidW = tc.require("mm.a.norm_mid.weight");
                t.mm2W = tc.require("mm.a.mlp.2.weight");
                int ff = (int) t.mm1W.ne(1);
                if (ff % 2 != 0) {
                    throw new IllegalArgumentException("tensor mm.a.mlp.1.weight has an odd output width " + ff
                            + "; the SwiGLU splits it in halves");
                }
                shape(tc, t.normPreW, (long) e * stack);
                shape(tc, t.mm1W, (long) e * stack, ff);
                shape(tc, t.normMidW, ff / 2);
                shape(tc, t.mm2W, ff / 2, d);
                break;
            }
            case ProjectorEntry.PROJECTOR_QWEN2A:
                t.fcW = tc.require("mm.a.fc.weight");
                t.fcB = tc.require("mm.a.fc.bias");
                shape(tc, t.fcW, e, d);
                shape(tc, t.fcB, d);
                break;
            case ProjectorEntry.PROJECTOR_VOXTRAL:
                mlp(tc, t, (long) e * stack, d, false);
                break;
            case ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO:
            case ProjectorEntry.PROJECTOR_QWEN3A:
                mlp(tc, t, e, d, true);
                break;
            case ProjectorEntry.PROJECTOR_MERALION:
                meralion(tc, t, (long) e * stack, d);
                break;
            case ProjectorEntry.PROJECTOR_GLMA:
                t.normPreW = tc.require("mm.a.norm_pre.weight");
                t.normPreB = tc.require("mm.a.norm_pre.bias");
                shape(tc, t.normPreW, e);
                shape(tc, t.normPreB, e);
                mlp(tc, t, (long) e * stack, d, true);
                t.boi = tc.require("v.boi");
                t.eoi = tc.require("v.eoi");
                shape(tc, t.boi, d);
                shape(tc, t.eoi, d);
                break;
            default:
                throw new IllegalStateException("unknown audio projector graph " + graph);
        }
    }

    /** {@code mm.a.mlp.1} {@code [in, ff]} and {@code mm.a.mlp.2} {@code [ff, d]}, with or without biases. */
    private static void mlp(TensorCatalog tc, Tensors t, long in, int d, boolean biased) {
        t.mm1W = tc.require("mm.a.mlp.1.weight");
        t.mm2W = tc.require("mm.a.mlp.2.weight");
        long ff = t.mm1W.ne(1);
        shape(tc, t.mm1W, in, ff);
        shape(tc, t.mm2W, ff, d);
        if (biased) {
            t.mm1B = tc.require("mm.a.mlp.1.bias");
            t.mm2B = tc.require("mm.a.mlp.2.bias");
            shape(tc, t.mm1B, ff);
            shape(tc, t.mm2B, d);
        }
    }

    /** The MERaLiON adaptor: layer norm, {@code mlp.0} + SiLU, {@code mlp.1} gate / {@code mlp.2} pool, {@code mlp.3}. */
    private static void meralion(TensorCatalog tc, Tensors t, long in, int d) {
        t.normPreW = tc.require("mm.a.norm_pre.weight");
        t.normPreB = tc.require("mm.a.norm_pre.bias");
        shape(tc, t.normPreW, in);
        shape(tc, t.normPreB, in);
        t.mm0W = tc.require("mm.a.mlp.0.weight");
        t.mm0B = tc.require("mm.a.mlp.0.bias");
        long h = t.mm0W.ne(1);
        shape(tc, t.mm0W, in, h);
        shape(tc, t.mm0B, h);
        t.mm1W = tc.require("mm.a.mlp.1.weight");
        t.mm1B = tc.require("mm.a.mlp.1.bias");
        t.mm2W = tc.require("mm.a.mlp.2.weight");
        t.mm2B = tc.require("mm.a.mlp.2.bias");
        shape(tc, t.mm1W, h, h);
        shape(tc, t.mm1B, h);
        shape(tc, t.mm2W, h, h);
        shape(tc, t.mm2B, h);
        t.mm3W = tc.require("mm.a.mlp.3.weight");
        t.mm3B = tc.require("mm.a.mlp.3.bias");
        shape(tc, t.mm3W, h, d);
        shape(tc, t.mm3B, d);
    }

    /** The hparams the geometry was resolved from. */
    public ClipHparams hparams() {
        return hparams;
    }

    /** The encoder graph (a {@code ProjectorEntry.ENCODER_*} audio name). */
    public String encoderGraph() {
        return encoderGraph;
    }

    /** The projector graph (a {@code ProjectorEntry.PROJECTOR_*} audio name). */
    public String projectorGraph() {
        return projectorGraph;
    }

    /** True for the whisper-family encoder graph. */
    public boolean isWhisper() {
        return ProjectorEntry.ENCODER_WHISPER.equals(encoderGraph);
    }

    /** The Gemma 3n conformer geometry; null for every other encoder graph. */
    Gemma3nAudioModel gemma3n() {
        return gemma3n;
    }

    /** Frames stacked per token by the projector, at least 1. */
    public int stackFactor() {
        return stackFactor;
    }

    /** True when the tower average-pools its output pairwise (qwen2a, voxtral, musicflamingo). */
    public boolean avgPool() {
        return ProjectorEntry.PROJECTOR_QWEN2A.equals(projectorGraph)
                || ProjectorEntry.PROJECTOR_VOXTRAL.equals(projectorGraph)
                || ProjectorEntry.PROJECTOR_MUSIC_FLAMINGO.equals(projectorGraph);
    }

    /** True when the tower output is stacked before the projector (ultravox, voxtral, meralion). */
    public boolean stackFrames() {
        return ProjectorEntry.PROJECTOR_ULTRAVOX.equals(projectorGraph)
                || ProjectorEntry.PROJECTOR_VOXTRAL.equals(projectorGraph)
                || ProjectorEntry.PROJECTOR_MERALION.equals(projectorGraph);
    }

    /** Rows of the learned position table ({@code a.position_embd.weight}); 0 without one. */
    public int maxPositions() {
        return maxPos;
    }

    /** The qwen3a conv channel counts {@code {c1, c2, c3}}; null for the other graphs. */
    int[] convChannels() {
        return convChannels == null ? null : convChannels.clone();
    }

    /** The mel height the qwen3a conv stack leaves; 0 for the other graphs. */
    int convHeight() {
        return convHeight;
    }

    /** The encoder blocks, in order (unmodifiable); empty for the towers with their own block layout. */
    public List<ClipLayer> layers() {
        return layers;
    }

    /** The tensor catalog the geometry was resolved over; the family towers bind their tensors from it. */
    TensorCatalog catalog() {
        return catalog;
    }

    Tensors tensors() {
        return t;
    }

    /** The text-model embedding width every output row has ({@code clip.audio.projection_dim}). */
    public int outputWidth() {
        return hparams.projectionDim();
    }

    /** The log-mel preprocessor the encoder's checkpoints were trained on (mtmd-audio.cpp, mtmd.cpp). */
    public MelFrontend.Flavor melFlavor() {
        switch (encoderGraph) {
            case ProjectorEntry.ENCODER_WHISPER:
                return MelFrontend.Flavor.WHISPER;
            case ProjectorEntry.ENCODER_QWEN3A:
                return MelFrontend.Flavor.QWEN3A;
            case ProjectorEntry.ENCODER_CONFORMER:
                return ProjectorType.D1OMNI_A.equals(hparams.projectorType()) ? MelFrontend.Flavor.D1OMNI
                        : MelFrontend.Flavor.CONFORMER;
            case ProjectorEntry.ENCODER_LFM3A:
                return MelFrontend.Flavor.LFM3A;
            case ProjectorEntry.ENCODER_GEMMA4A:
                return MelFrontend.Flavor.GEMMA4A;
            case ProjectorEntry.ENCODER_GEMMA4UA:
                return MelFrontend.Flavor.GEMMA4UA;
            case ProjectorEntry.ENCODER_PARAKEET:
                return MelFrontend.Flavor.PARAKEET;
            case ProjectorEntry.ENCODER_GRANITE_SPEECH:
                return MelFrontend.Flavor.GRANITE_SPEECH;
            case ProjectorEntry.ENCODER_VOXTRAL_RT:
                return MelFrontend.Flavor.VOXTRAL_RT;
            case ProjectorEntry.ENCODER_MIMO_AUDIO:
                return MelFrontend.Flavor.MIMO_AUDIO;
            case ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO:
                return MelFrontend.Flavor.MUSE_SPARK_AUDIO;
            case ProjectorEntry.ENCODER_DOTS3NOTE_A:
                return MelFrontend.Flavor.DOTS3NOTE;
            case ProjectorEntry.ENCODER_GEMMA3N_USM:
                return MelFrontend.Flavor.GEMMA3N;
            default:
                throw new IllegalStateException("unknown audio encoder graph " + encoderGraph);
        }
    }

    /**
     * The framing of {@link #melFlavor()}: the mel bins and the projector's fixed rate, FFT, window and hop, and
     * the samples per output row of the flavors that frame whole rows (muse-spark fold, gemma4ua frame, voxtral
     * realtime row of {@code 2 * stack_factor} frames).
     */
    public MelFrontend.Config melConfig() {
        ClipAudioHparams a = hparams.audio();
        switch (melFlavor()) {
            case MUSE_SPARK_AUDIO:
                return MelFrontend.Config.passthrough(1, a.sampleRate(), a.samplesPerToken());
            case GEMMA4UA:
                return MelFrontend.Config.passthrough(a.nMelBins(), a.sampleRate(), a.nMelBins());
            case VOXTRAL_RT:
                return MelFrontend.Config.builder(a.nMelBins(), a.sampleRate(), a.nFft(), a.windowLen(), a.hopLen())
                        .rowSamples(2 * a.stackFactor() * a.hopLen())
                        .build();
            default:
                return MelFrontend.Config.builder(a.nMelBins(), a.sampleRate(), a.nFft(), a.windowLen(), a.hopLen())
                        .chunkSeconds(a.chunkSeconds())
                        .build();
        }
    }

    /** The parakeet filterbank tensor ({@code a.mel_filters}); null without one. */
    public TensorRef melFilters() {
        return melFilters;
    }

    /** The parakeet window tensor ({@code a.window}); null without one. */
    public TensorRef melWindow() {
        return melWindow;
    }

    /**
     * Embedding rows one mel chunk of {@code nFrames} frames produces: for the whisper graph the stride-2 conv
     * halving, the average pool, the zero-padded frame stack and the glma BOI/EOI pair; for qwen3a 13 rows per
     * 100-frame batch; for gemma3na the 188 rows every window is padded to; for the other graphs their own conv
     * and pooling formulas (clip.cpp {@code clip_n_output_tokens}).
     *
     * @throws IllegalArgumentException for a non-positive frame count
     */
    public int nTokens(int nFrames) {
        if (nFrames <= 0) {
            throw new IllegalArgumentException("a mel chunk needs at least one frame, got " + nFrames);
        }
        switch (encoderGraph) {
            case ProjectorEntry.ENCODER_WHISPER:
                return whisperTokens(nFrames);
            case ProjectorEntry.ENCODER_QWEN3A:
                return (nFrames / QWEN3A_CHUNK_FRAMES) * QWEN3A_TOKENS_PER_CHUNK;
            case ProjectorEntry.ENCODER_CONFORMER:
            case ProjectorEntry.ENCODER_LFM3A: {
                int n = (nFrames + 1) / 2;
                n = (n + 1) / 2;
                return (n + 1) / 2;
            }
            case ProjectorEntry.ENCODER_GEMMA4A:
                return gemma4aTokens(nFrames);
            case ProjectorEntry.ENCODER_GEMMA4UA:
                return nFrames;
            case ProjectorEntry.ENCODER_PARAKEET:
                return (nFrames + hparams.audio().subsampling() - 1) / hparams.audio().subsampling();
            case ProjectorEntry.ENCODER_GRANITE_SPEECH:
                return GraniteSpeechTower.tokens(nFrames, hparams.audio().projWindowSize(),
                        hparams.audio().projDownsampleRate());
            case ProjectorEntry.ENCODER_VOXTRAL_RT:
                return VoxtralRtTower.tokens(nFrames, stackFactor);
            case ProjectorEntry.ENCODER_MIMO_AUDIO:
                return MimoAudioTower.tokens(nFrames, hparams.audio().localGroup());
            case ProjectorEntry.ENCODER_MUSE_SPARK_AUDIO: {
                int fold = hparams.audio().samplesPerToken();
                return (nFrames + fold - 1) / fold;
            }
            case ProjectorEntry.ENCODER_DOTS3NOTE_A:
                return (nFrames + 7) / 8;
            case ProjectorEntry.ENCODER_GEMMA3N_USM:
                return Gemma3nAudioModel.TOKENS_PER_WINDOW;
            default:
                throw new IllegalStateException("unknown audio encoder graph " + encoderGraph);
        }
    }

    /** The whisper-family row count: conv halving, average pool, frame stack, glma separators. */
    private int whisperTokens(int nFrames) {
        int n = (nFrames - 1) / 2 + 1;
        if (avgPool()) {
            n = (n - 2) / 2 + 1;
        }
        if (stackFrames() || ProjectorEntry.PROJECTOR_GLMA.equals(projectorGraph)) {
            n = (n + stackFactor - 1) / stackFactor;
        }
        return ProjectorEntry.PROJECTOR_GLMA.equals(projectorGraph) ? n + 2 : n;
    }

    /** Two stride-2 convs of kernel 3 over the mel frames (gemma4a.cpp). */
    private static int gemma4aTokens(int nFrames) {
        int n = nFrames;
        for (int i = 0; i < 2; i++) {
            n = (n - 1) / 2 + 1;
        }
        return n;
    }

    @Override
    public String toString() {
        return "ClipAudioModel{" + hparams.projectorType() + ", encoder=" + encoderGraph + ", projector="
                + projectorGraph + ", n_mel=" + hparams.audio().nMelBins() + ", layers=" + layers.size()
                + ", stack=" + stackFactor + ", out=" + outputWidth() + "}";
    }
}
