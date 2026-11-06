package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Arrays;
import java.util.Locale;

/**
 * The audio part of {@link ClipHparams}: the {@code clip.audio.*} keys an audio encoder reads beyond the shared
 * transformer keys, and the log-mel framing (sample rate, FFT size, window, hop, chunk length) that llama.cpp
 * {@code load_hparams} (tools/mtmd/clip.cpp) hardcodes per projector instead of reading from the file. A framing
 * value of 0 means the projector's preprocessor does not use it; a family key a projector does not read is 0 (or
 * null for an array). Immutable.
 */
public final class ClipAudioHparams {

    private static final String N_MEL_BINS = "clip.audio.num_mel_bins";
    private static final String STACK_FACTOR = "clip.audio.projector.stack_factor";
    private static final String CHUNK_SIZE = "clip.audio.chunk_size";
    private static final String WINDOW_SIZE = "clip.audio.window_size";
    private static final String CONV_KERNEL = "clip.audio.conv_kernel_size";
    private static final String SUBSAMPLING = "clip.audio.subsampling_factor";
    private static final String MAX_POS_EMB = "clip.audio.max_pos_emb";
    private static final String PROJ_WINDOW = "clip.audio.projector.window_size";
    private static final String PROJ_DOWNSAMPLE = "clip.audio.projector.downsample_rate";
    private static final String PROJ_HEADS = "clip.audio.projector.head_count";
    private static final String RVQ_QUANTIZERS = "clip.audio.rvq.num_quantizers";
    private static final String RVQ_CODEBOOK = "clip.audio.rvq.codebook_size";
    private static final String WA_PATTERN = "clip.audio.wa_pattern_mode";
    private static final String LOCAL_BLOCKS = "clip.audio.local_block_count";
    private static final String LOCAL_GROUP = "clip.audio.local_group_size";
    private static final String HIDDEN_SIZES = "clip.audio.hidden_sizes";
    private static final String PATCH_SIZES = "clip.audio.patch_sizes";
    private static final String DEPTHS = "clip.audio.depths";

    /** Mel-bin ceiling of the FFT frontends (clip.cpp {@code load_hparams}). */
    private static final int MAX_FFT_MEL_BINS = 256;
    /** Attention-window ceiling clip.cpp enforces for every tower. */
    private static final int MAX_ATTN_WINDOW = 4096;
    /** Ceiling of the granite-speech attention context (its masks are context squared). */
    private static final int MAX_GRANITE_CHUNK = 8192;
    /** The conv strides of the parakeet subsampling stem fix its frames-per-token. */
    private static final int PARAKEET_SUBSAMPLING = 8;

    private final int nMelBins;
    private final int stackFactor;
    private final int chunkSize;
    private final int chunkSeconds;
    private final int sampleRate;
    private final int nFft;
    private final int windowLen;
    private final int hopLen;
    private final float ropeTheta;
    private final int attnWindow;
    private final int convKernel;
    private final int subsampling;
    private final int maxPosEmb;
    private final int projWindowSize;
    private final int projDownsampleRate;
    private final int projHeadCount;
    private final int[] rvqCodebookSizes;
    private final int[] waPattern;
    private final int localLayers;
    private final int localGroup;
    private final int[] hiddenSizes;
    private final int[] patchSizes;
    private final int[] depths;

    private ClipAudioHparams(GgufKv kv, ProjectorType type, int nLayer) {
        int[] framing = framing(type);
        this.chunkSeconds = framing[0];
        this.sampleRate = framing[1];
        this.nFft = framing[2];
        this.windowLen = framing[3];
        this.hopLen = framing[4];
        this.ropeTheta = ropeTheta(type);
        this.nMelBins = melBins(kv, type, nFft > 0);
        this.stackFactor = stackFactor(kv, type);
        this.attnWindow = attnWindow(kv, type);
        boolean granite = type == ProjectorType.GRANITE_SPEECH;
        // granite reads its chunk size before the conv kernel (clip.cpp load_hparams)
        this.chunkSize = granite ? graniteChunk(kv) : 0;
        this.convKernel = convKernel(kv, type);
        this.subsampling = type == ProjectorType.PARAKEET ? parakeetSubsampling(kv) : 0;
        this.maxPosEmb = granite ? requiredPositive(kv, MAX_POS_EMB) : 0;
        this.projWindowSize = granite ? requiredPositive(kv, PROJ_WINDOW) : 0;
        this.projDownsampleRate = granite ? requiredPositive(kv, PROJ_DOWNSAMPLE) : 0;
        this.projHeadCount = granite ? requiredPositive(kv, PROJ_HEADS) : 0;
        if (granite && projWindowSize % projDownsampleRate != 0) {
            throw new GgufFormatException("metadata key '" + PROJ_DOWNSAMPLE + "' (" + projDownsampleRate
                    + ") must divide '" + PROJ_WINDOW + "' (" + projWindowSize + ")");
        }

        boolean mimo = type == ProjectorType.MIMO_AUDIO;
        this.rvqCodebookSizes = mimo ? rvqCodebooks(kv) : null;
        this.waPattern = mimo ? perLayer(kv, WA_PATTERN, nLayer) : null;
        this.localLayers = mimo ? requiredNonNegative(kv, LOCAL_BLOCKS) : 0;
        this.localGroup = mimo ? requiredPositive(kv, LOCAL_GROUP) : 0;

        boolean muse = type == ProjectorType.MUSE_SPARK_AUDIO;
        this.hiddenSizes = muse ? requiredArray(kv, HIDDEN_SIZES) : null;
        this.patchSizes = muse ? requiredArray(kv, PATCH_SIZES) : null;
        this.depths = muse ? requiredArray(kv, DEPTHS) : null;
        if (muse) {
            validateMuseStages(nLayer);
        }
    }

    /**
     * The audio keys of {@code kv} for the projector {@code type} over {@code nLayer} encoder blocks, checked the
     * way clip.cpp {@code load_hparams} checks them.
     *
     * @throws GgufFormatException for a missing or out-of-range key the projector reads, or a per-block or
     *                             per-stage array whose length does not match
     */
    static ClipAudioHparams from(GgufKv kv, ProjectorType type, int nLayer) {
        return new ClipAudioHparams(kv, type, nLayer);
    }

    private static int melBins(GgufKv kv, ProjectorType type, boolean fftBased) {
        if (type == ProjectorType.GEMMA4UA) {
            return 640;
        }
        int n = requiredPositive(kv, N_MEL_BINS);
        if (fftBased && n > MAX_FFT_MEL_BINS) {
            throw new GgufFormatException("metadata key '" + N_MEL_BINS + "' must be in [1, " + MAX_FFT_MEL_BINS
                    + "], got " + n);
        }
        if (type == ProjectorType.MUSE_SPARK_AUDIO && n != 1) {
            throw new GgufFormatException("metadata key '" + N_MEL_BINS + "' must be 1 for muse_spark_audio (one"
                    + " waveform channel), got " + n);
        }
        return n;
    }

    private static int stackFactor(GgufKv kv, ProjectorType type) {
        if (requiresStack(type)) {
            return requiredPositive(kv, STACK_FACTOR);
        }
        return kv.has(STACK_FACTOR) ? requiredPositive(kv, STACK_FACTOR) : 0;
    }

    /** The projectors whose graphs stack frames and therefore require the stack factor (clip.cpp). */
    private static boolean requiresStack(ProjectorType type) {
        switch (type) {
            case ULTRAVOX:
            case VOXTRAL:
            case MERALION:
            case GLMA:
            case VOXTRAL_RT:
                return true;
            default:
                return false;
        }
    }

    private static int attnWindow(GgufKv kv, ProjectorType type) {
        switch (type) {
            case LFM3A:
            case VOXTRAL_RT:
            case MIMO_AUDIO:
                int window = requiredPositive(kv, WINDOW_SIZE);
                if (window > MAX_ATTN_WINDOW) {
                    throw new GgufFormatException("metadata key '" + WINDOW_SIZE + "' must be at most "
                            + MAX_ATTN_WINDOW + ", got " + window);
                }
                return window;
            default:
                return 0;
        }
    }

    private static int convKernel(GgufKv kv, ProjectorType type) {
        switch (type) {
            case PARAKEET:
            case GRANITE_SPEECH:
                int kernel = requiredPositive(kv, CONV_KERNEL);
                if (kernel % 2 != 1) {
                    throw new GgufFormatException("metadata key '" + CONV_KERNEL + "' must be odd, got " + kernel);
                }
                return kernel;
            case MUSE_SPARK_AUDIO:
                return requiredPositive(kv, CONV_KERNEL);
            default:
                return 0;
        }
    }

    private static int parakeetSubsampling(GgufKv kv) {
        int factor = requiredPositive(kv, SUBSAMPLING);
        if (factor != PARAKEET_SUBSAMPLING) {
            throw new GgufFormatException("metadata key '" + SUBSAMPLING + "' must be " + PARAKEET_SUBSAMPLING
                    + " (the conv strides of the parakeet stem), got " + factor);
        }
        return factor;
    }

    private static int graniteChunk(GgufKv kv) {
        int chunk = requiredPositive(kv, CHUNK_SIZE);
        if (chunk > MAX_GRANITE_CHUNK) {
            throw new GgufFormatException("metadata key '" + CHUNK_SIZE + "' must be at most " + MAX_GRANITE_CHUNK
                    + ", got " + chunk);
        }
        return chunk;
    }

    private static int[] rvqCodebooks(GgufKv kv) {
        int quantizers = requiredPositive(kv, RVQ_QUANTIZERS);
        int[] sizes = requiredArray(kv, RVQ_CODEBOOK);
        if (sizes.length != quantizers) {
            throw new GgufFormatException("metadata key '" + RVQ_CODEBOOK + "' must hold " + quantizers
                    + " values (one per quantizer), got " + sizes.length);
        }
        for (int bins : sizes) {
            if (bins <= 0) {
                throw new GgufFormatException("metadata key '" + RVQ_CODEBOOK + "' values must be positive, got "
                        + bins);
            }
        }
        return sizes;
    }

    private static int[] perLayer(GgufKv kv, String key, int nLayer) {
        int[] values = requiredArray(kv, key);
        if (values.length != nLayer) {
            throw new GgufFormatException("metadata key '" + key + "' must hold " + nLayer
                    + " values (one per block), got " + values.length);
        }
        return values;
    }

    private void validateMuseStages(int nLayer) {
        int stages = patchSizes.length;
        if (stages == 0 || hiddenSizes.length != stages || depths.length != stages) {
            throw new GgufFormatException("metadata keys '" + HIDDEN_SIZES + "', '" + PATCH_SIZES + "' and '"
                    + DEPTHS + "' must be non-empty and of one length, got " + hiddenSizes.length + ", " + stages
                    + " and " + depths.length);
        }
        long blocks = 0;
        long fold = 1;
        for (int i = 0; i < stages; i++) {
            if (hiddenSizes[i] < 1 || patchSizes[i] < 1 || depths[i] < 0) {
                throw new GgufFormatException("muse_spark_audio stage " + i + " needs a positive hidden size and"
                        + " patch size and a non-negative depth, got " + hiddenSizes[i] + ", " + patchSizes[i]
                        + " and " + depths[i]);
            }
            blocks += depths[i];
            fold *= patchSizes[i];
            if (fold > Integer.MAX_VALUE) {
                throw new GgufFormatException("metadata key '" + PATCH_SIZES + "' folds more than "
                        + Integer.MAX_VALUE + " samples per token");
            }
        }
        if (blocks != nLayer) {
            throw new GgufFormatException("the sum of '" + DEPTHS + "' values (" + blocks + ") must equal the audio"
                    + " block count (" + nLayer + ")");
        }
    }

    /** The RoPE base clip.cpp hardcodes for the towers that rotate; 0 when the tower does not rotate. */
    private static float ropeTheta(ProjectorType type) {
        switch (type) {
            case LFM3A:
            case VOXTRAL_RT:
                return 1_000_000f;
            case MIMO_AUDIO:
            case DOTS3NOTE_A:
                return 10_000f;
            default:
                return 0f;
        }
    }

    /**
     * {@code {chunk seconds, sample rate, n_fft, window, hop}} as clip.cpp {@code load_hparams} sets them per
     * projector (for gemma3na, which clip.cpp does not run, as the Hugging Face {@code Gemma3nAudioFeatureExtractor}
     * frames: 32 ms windows every 10 ms, a doubled 1024-point FFT, 30 s per window); zeros for a projector without
     * a log-mel frontend.
     */
    private static int[] framing(ProjectorType type) {
        switch (type) {
            case ULTRAVOX:
            case QWEN2A:
            case QWEN3A:
            case GLMA:
            case VOXTRAL:
            case MERALION:
            case MUSIC_FLAMINGO:
                return new int[]{30, 16000, 400, 400, 160};
            case VOXTRAL_RT:
            case LFM3A:
                return new int[]{0, 16000, 400, 400, 160};
            case DOTS3NOTE_A:
                return new int[]{60, 16000, 400, 400, 160};
            case LFM2A:
            case D1OMNI_A:
                return new int[]{1, 16000, 512, 400, 160};
            case PARAKEET:
            case GRANITE_SPEECH:
                return new int[]{0, 16000, 512, 400, 160};
            case GEMMA4A:
                return new int[]{0, 16000, 512, 320, 160};
            case GEMMA3NA:
                return new int[]{30, 16000, 1024, 512, 160};
            case GEMMA4UA:
                return new int[]{0, 16000, 0, 0, 0};
            case MIMO_AUDIO:
                return new int[]{0, 24000, 960, 960, 240};
            case MUSE_SPARK_AUDIO:
                return new int[]{0, 24000, 0, 0, 0};
            default:
                return new int[]{0, 0, 0, 0, 0};
        }
    }

    /**
     * The FFN activation clip.cpp forces for an audio projector: GELU-erf for the whisper family, qwen3a and
     * mimo_audio (the PyTorch exact GELU), SiLU for voxtral_rt; the other graphs take {@code fromKeys} (the
     * {@code clip.use_gelu} / {@code clip.use_silu} choice) or fix the activation of their blocks in the graph.
     */
    static ClipFfnOp ffnOverride(ProjectorType type, ClipFfnOp fromKeys) {
        switch (type) {
            case ULTRAVOX:
            case QWEN2A:
            case QWEN3A:
            case GLMA:
            case VOXTRAL:
            case MERALION:
            case MUSIC_FLAMINGO:
            case MIMO_AUDIO:
                return ClipFfnOp.GELU_ERF;
            case VOXTRAL_RT:
                return ClipFfnOp.SILU;
            default:
                return fromKeys;
        }
    }

    /**
     * The epsilon clip.cpp hardcodes for the gemma4 audio towers (1e-6), and the {@code rms_norm_eps} of every
     * Gemma 3n audio RMS norm (1e-6; its converter writes 1e-5); {@code fromKeys} otherwise.
     */
    static float epsOverride(ProjectorType type, float fromKeys) {
        return type == ProjectorType.GEMMA4A || type == ProjectorType.GEMMA4UA || type == ProjectorType.GEMMA3NA
                ? 1e-6f : fromKeys;
    }

    private static int requiredPositive(GgufKv kv, String key) {
        int v = requiredNonNegative(kv, key);
        if (v == 0) {
            throw new GgufFormatException("metadata key '" + key + "' must be positive, got 0");
        }
        return v;
    }

    private static int requiredNonNegative(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        int v = kv.u32(key, 0);
        if (v < 0) {
            throw new GgufFormatException("metadata key '" + key + "' exceeds " + Integer.MAX_VALUE + ", got " + v);
        }
        return v;
    }

    private static int[] requiredArray(GgufKv kv, String key) {
        int[] v = kv.i32Array(key);
        if (v == null) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        return v;
    }

    private static int[] copy(int[] values) {
        return values == null ? null : values.clone();
    }

    /** Mel bins of the frontend ({@code clip.audio.num_mel_bins}); the 640-sample frame of gemma4ua. */
    public int nMelBins() {
        return nMelBins;
    }

    /**
     * Frames the projector stacks per token ({@code clip.audio.projector.stack_factor}); 0 when the file has no
     * stack factor and the projector does not need one.
     */
    public int stackFactor() {
        return stackFactor;
    }

    /** Attention context of the granite-speech blocks ({@code clip.audio.chunk_size}); 0 elsewhere. */
    public int chunkSize() {
        return chunkSize;
    }

    /**
     * Attention window of lfm3a, voxtral_rt and mimo_audio ({@code clip.audio.window_size}); each tower defines
     * its mask from it. 0 elsewhere.
     */
    public int attnWindow() {
        return attnWindow;
    }

    /** Kernel width of the parakeet, granite-speech and muse-spark conv modules; 0 elsewhere. */
    public int convKernel() {
        return convKernel;
    }

    /** Mel frames per output token of parakeet ({@code clip.audio.subsampling_factor}); 0 elsewhere. */
    public int subsampling() {
        return subsampling;
    }

    /** RPE distance range of the granite-speech blocks ({@code clip.audio.max_pos_emb}); 0 elsewhere. */
    public int maxPosEmb() {
        return maxPosEmb;
    }

    /** QFormer window of the granite-speech projector ({@code clip.audio.projector.window_size}); 0 elsewhere. */
    public int projWindowSize() {
        return projWindowSize;
    }

    /** QFormer queries per window of granite speech ({@code clip.audio.projector.downsample_rate}). */
    public int projDownsampleRate() {
        return projDownsampleRate;
    }

    /** QFormer attention heads of granite speech ({@code clip.audio.projector.head_count}). */
    public int projHeadCount() {
        return projHeadCount;
    }

    /** RVQ quantizer count of mimo_audio ({@code clip.audio.rvq.num_quantizers}); 0 elsewhere. */
    public int rvqQuantizers() {
        return rvqCodebookSizes == null ? 0 : rvqCodebookSizes.length;
    }

    /** Per-quantizer codebook bins of mimo_audio ({@code clip.audio.rvq.codebook_size}); null elsewhere. */
    public int[] rvqCodebookSizes() {
        return copy(rvqCodebookSizes);
    }

    /**
     * Per-block attention mode of mimo_audio ({@code clip.audio.wa_pattern_mode}): -1 attends causally over all
     * frames, any other value restricts the block to the causal window. Null elsewhere.
     */
    public int[] waPattern() {
        return copy(waPattern);
    }

    /** Blocks of the mimo_audio local transformer ({@code clip.audio.local_block_count}); 0 elsewhere. */
    public int localLayers() {
        return localLayers;
    }

    /** Frames grouped per local-attention block of mimo_audio ({@code clip.audio.local_group_size}). */
    public int localGroup() {
        return localGroup;
    }

    /** Channel width per muse-spark stage ({@code clip.audio.hidden_sizes}); null elsewhere. */
    public int[] hiddenSizes() {
        return copy(hiddenSizes);
    }

    /** Rows folded per token per muse-spark stage ({@code clip.audio.patch_sizes}); null elsewhere. */
    public int[] patchSizes() {
        return copy(patchSizes);
    }

    /** Residual blocks per muse-spark stage ({@code clip.audio.depths}); null elsewhere. */
    public int[] depths() {
        return copy(depths);
    }

    /** Samples folded into one embedding row by the muse-spark stem: the product of the patch sizes; 0 elsewhere. */
    public int samplesPerToken() {
        if (patchSizes == null) {
            return 0;
        }
        int fold = 1;
        for (int p : patchSizes) {
            fold *= p;
        }
        return fold;
    }

    /** RoPE base of the towers that rotate their Q and K rows; 0 when the tower does not rotate. */
    public float ropeTheta() {
        return ropeTheta;
    }

    /** Fixed chunk length of the preprocessor in seconds; 0 = no fixed-length chunking. */
    public int chunkSeconds() {
        return chunkSeconds;
    }

    /** Sample rate the encoder consumes, in Hz; 0 when the projector has no audio frontend. */
    public int sampleRate() {
        return sampleRate;
    }

    /** FFT size of the log-mel frontend; 0 when unused. */
    public int nFft() {
        return nFft;
    }

    /** Window length of the log-mel frontend in samples; 0 when unused. */
    public int windowLen() {
        return windowLen;
    }

    /** Frame step of the log-mel frontend in samples; 0 when unused. */
    public int hopLen() {
        return hopLen;
    }

    /** True when the projector runs a log-mel frontend (FFT, window and hop are set). */
    public boolean hasMelFrontend() {
        return nFft > 0 && windowLen > 0 && hopLen > 0;
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "audio{n_mel=%d, stack=%d, sr=%d, n_fft=%d, window=%d, hop=%d",
                nMelBins, stackFactor, sampleRate, nFft, windowLen, hopLen)
                + (attnWindow > 0 ? ", attn_window=" + attnWindow : "")
                + (convKernel > 0 ? ", conv_kernel=" + convKernel : "")
                + (subsampling > 0 ? ", subsampling=" + subsampling : "")
                + (chunkSize > 0 ? ", chunk=" + chunkSize : "")
                + (rvqCodebookSizes != null ? ", rvq=" + Arrays.toString(rvqCodebookSizes) : "")
                + (localGroup > 0 ? ", local_layers=" + localLayers + ", local_group=" + localGroup : "")
                + (patchSizes != null ? ", patches=" + Arrays.toString(patchSizes) : "")
                + (ropeTheta > 0 ? ", rope_theta=" + ropeTheta : "")
                + "}";
    }
}
