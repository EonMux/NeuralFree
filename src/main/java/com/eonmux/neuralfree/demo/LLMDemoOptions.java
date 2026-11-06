package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.engine.DiffusionConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Command-line options of {@link LLMDemo}: parsing, validation, the usage text and the engine, sampler and diffusion
 * configurations they select. Engine and sampler settings start from the {@code -Dneuralfree.*} system properties
 * ({@link EngineConfig#fromSystemProperties()}, whose sampler defaults are {@link SamplerConfig#fromProperties});
 * the demo's own sampling defaults ({@link #DEFAULT_TEMPERATURE} and the others, and {@link #DEFAULT_MAX_TOKENS})
 * apply only where no property is set, and flags override both. The demo's defaults are fallbacks: a model's
 * sampling defaults ({@code general.sampling.*}, {@code neuralfree.sampling.*}) replace them, never a property or a
 * flag ({@link EngineConfig#modelSamplingDefaults()}). Flag names follow llama.cpp where it has an equivalent
 * ({@code common/arg.cpp}); no short alias is given a meaning that differs from llama.cpp's. The Hugging Face
 * checkpoint flags set the matching {@link ModelSourceOptions} or {@link EngineConfig} fields on the builders, on
 * top of the {@code -Dneuralfree.*} property defaults, {@code --export-gguf} exports the model's GGUF view
 * ({@link GgufWriter}) instead of generating, and {@code -hf} names a Hugging Face Hub model whose GGUF the
 * download resolves into the model path before the demo opens it. The chat-template flags write the template
 * text they contribute into the {@code tokenizer.chat_template} metadata override ({@code ChatTemplateSource}),
 * {@code --no-jinja} renders the detected built-in in place of the model's template, {@code --threads} sets the
 * CPU thread pool property and {@code --grammar}/{@code --json-schema} build the generation request's
 * {@link ConstraintSpec}. The model-load flags (backend, devices and the multi-GPU layer split, placement, threads,
 * flash attention and the KV cache types) parse through {@link EngineLoadFlags}, which the batched tools share.
 * Immutable.
 */
final class LLMDemoOptions {

    /** What the demo runs. */
    enum Mode {
        /**
         * Chosen from the model: classify for a decision head, rerank (with {@code --document}) or classify for RANK
         * pooling, embed for any other pooling type, else from the support class.
         */
        AUTO,
        GENERATE,
        EMBED,
        CLASSIFY,
        RERANK,
        DIFFUSION,
        AUDIO;

        static Mode parse(String value) throws UsageException {
            String s = value.trim().toUpperCase(Locale.ROOT);
            for (Mode m : values()) {
                if (m.name().equals(s)) {
                    return m;
                }
            }
            throw new UsageException("--mode " + value + " is invalid; use generate, embed, classify, rerank,"
                    + " diffusion or audio");
        }

        /** Whether the mode evaluates whole inputs in one non-causal or memoryless pass. */
        boolean pooled() {
            return this == EMBED || this == CLASSIFY || this == RERANK;
        }
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {
        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /**
     * One {@code --image}/{@code --audio}/{@code --video} file of the {@code -p} turn, its kind from its flag
     * (llama.cpp's media flags), so a video named without a video extension still attaches as a video.
     * Immutable.
     */
    static final class MediaFile {

        /** The flag's modality: {@code image}, {@code audio} or {@code video}. */
        final String kind;
        /** The file path as given. */
        final String path;

        MediaFile(String kind, String path) {
            this.kind = kind;
            this.path = path;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof MediaFile && kind.equals(((MediaFile) o).kind)
                    && path.equals(((MediaFile) o).path);
        }

        @Override
        public int hashCode() {
            return kind.hashCode() * 31 + path.hashCode();
        }

        @Override
        public String toString() {
            return kind + ":" + path;
        }
    }

    static final float DEFAULT_TEMPERATURE = 0.6f;
    static final float DEFAULT_TOP_P = 0.95f;
    static final int DEFAULT_TOP_K = 40;
    static final float DEFAULT_MIN_P = 0.05f;
    static final float DEFAULT_REPEAT_PENALTY = 1.0f;
    static final int DEFAULT_REPEAT_WINDOW = 256;
    static final int DEFAULT_MAX_TOKENS = 2560;
    static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
    /**
     * Ubatch of the pooled modes when none is given: llama.cpp {@code llama-embedding} evaluates each input in one
     * batch of {@code max(n_batch = 2048, n_ctx)} tokens (examples/embedding/embedding.cpp:122-131).
     */
    static final int POOLED_UBATCH = 2048;

    /** The model path given on the command line, or null. */
    final String modelPath;
    /** The file the session statistics are appended to ({@code --stats-file}), or null for none. */
    final String statsFile;
    final String prompt;
    final String systemPrompt;
    final boolean raw;
    /**
     * The {@code enable_thinking} the chat prompts render with: true for {@code --thinking} or {@code --reasoning on},
     * false for {@code --reasoning off} and by default, null for {@code --reasoning auto} (the template's own default).
     */
    final Boolean thinking;
    /**
     * The template variables bound into every chat prompt after {@code enable_thinking}, in flag order, a later flag
     * over an earlier one (llama.cpp's {@code default_template_kwargs}): {@link ChatTemplateKwargs#PROPERTY}, then
     * {@code --chat-template-kwargs}, {@code --reasoning on|off} ({@code enable_thinking}), {@code --reasoning-effort}
     * ({@code reasoning_effort}) and {@code --reasoning-preserve} ({@code preserve_reasoning}); unmodifiable, empty
     * when none were given.
     */
    final Map<String, Object> templateKwargs;
    final boolean debugPrompt;
    final boolean nonInteractive;
    final Mode mode;
    final boolean showAllChannels;
    final boolean benchmark;
    final boolean benchStreaming;
    final boolean memoryReportOnly;
    final boolean noMmap;
    /** {@code --no-jinja}: the model's template is replaced by the built-in its text detects. */
    final boolean noJinja;
    final int embedPreview;
    final int[] tokenTypes;
    final String document;
    final int[] decisionOrder;
    final int[] audioCodes;
    final String audioOut;
    /**
     * The generator file of a speech backbone ({@code --mmproj} of the audio mode): the qwen3tts code predictor and
     * code2wav decoder, the pockettts flow network and mimi decoder, or the lfm3a MaskGIT depthformer and
     * liquid-codec decoder. In every other mode the same flag names the projector of a multimodal model
     * (llama.cpp {@code --mmproj}); {@link #mmprojUrl}, {@link #noMmproj}, {@link #mmprojOffload},
     * {@link #mmprojDevice} and the image token budget are its companions there.
     */
    final String mmproj;
    /** The {@code --mmproj-url} projector URL, or null; downloaded once into the hub cache when the model loads. */
    final String mmprojUrl;
    /** {@code --no-mmproj} / {@code --no-mmproj-auto}: no projector, and no automatic sibling search. */
    final boolean noMmproj;
    /** The {@code --mmproj-offload} / {@code --no-mmproj-offload} switch, or null for the default (on). */
    final Boolean mmprojOffload;
    /** The {@code --mmproj-device}/{@code -mmdev} selector of the vision encoder, or null. */
    final String mmprojDevice;
    /** The smallest token count of one image ({@code --image-min-tokens}), 0 for unset. */
    final int imageMinTokens;
    /** The largest token count of one image ({@code --image-max-tokens}), 0 for unset. */
    final int imageMaxTokens;
    /** The language of a text-to-speech synthesis ({@code --audio-lang}); null speaks the default. */
    final String audioLang;
    /** The speaker preset of a text-to-speech synthesis ({@code --voice}); null means none. */
    final String voice;
    /** The reference waveform a text-to-speech synthesis clones the voice from ({@code --audio-speaker}). */
    final String audioSpeaker;
    /** Speech decoding overrides of {@code --audio-fft/--audio-hop/--audio-rate}; 0 keeps the model's value. */
    final int audioFft;
    final int audioHop;
    final int audioRate;
    final boolean help;
    /** The {@code --version} act-and-exit flag: print the version and build info, as llama.cpp's does. */
    final boolean version;
    /** The {@code --list-devices} act-and-exit flag: print the devices every API enumerates. */
    final boolean listDevices;
    /** The {@code -cl}/{@code --cache-list} act-and-exit flag: print the hub cache's models. */
    final boolean cacheList;
    /** The {@code --server-base} URL of a running server to chat with, or null for a local model. */
    final String serverBase;
    /** The {@code --api-key} bearer token of the server request, or null. */
    final String apiKey;
    /** The {@code -o/--output} conversation transcript path, or null for none. */
    final String outFile;
    /** {@code -mli/--multiline-input}: input lines keep joining until {@code /} or {@code \} ends them. */
    final boolean multilineInput;
    /** {@code -i/--interactive}: the completion REPL takes turns after the prompt. */
    final boolean interactive;
    /** {@code -if/--interactive-first}: the completion REPL waits for input before any generation. */
    final boolean interactiveFirst;
    /** The {@code --in-prefix} string of the completion REPL's user inputs, or null. */
    final String inPrefix;
    /** The {@code --in-suffix} string of the completion REPL's user inputs, or null. */
    final String inSuffix;
    /** {@code --in-prefix-bos}: the completion REPL prefixes the BOS token to every user input. */
    final boolean inPrefixBos;
    /** {@code -e/--escape} (default) or {@code --no-escape}: the completion REPL processes escape sequences. */
    final boolean escape;
    /** {@code -st/--single-turn}: the chat ends after its first answer. */
    final boolean singleTurn;
    /** The {@code --image}/{@code --audio}/{@code --video} files attached to the {@code -p} turn, in order. */
    final List<MediaFile> media;
    /** Engine settings before the mode-dependent ubatch default ({@link #engineConfig(Mode)}). */
    final EngineConfig engine;
    final boolean ubatchExplicit;
    /** The request of every autoregressive generation: token budget, sampler, stop strings and ignore-eos. */
    final GenerationRequest generation;
    final DiffusionConfig diffusion;
    /** The {@code --export-gguf} output file, or null for no export. */
    final String exportGguf;
    final GgufWriter.Mode exportMode;
    final boolean exportOverwrite;
    /** The {@code -hf} Hugging Face Hub expression {@code <user>/<model>[:quant][@revision]}, or null. */
    final String hfRepo;
    /** The {@code -hff} repository file overriding the {@code -hf} quant tag, or null. */
    final String hfFile;
    /** The {@code --prompt-cache} session file path ({@code --prompt-cache-all}/{@code --prompt-cache-ro} qualify it), or null for none. */
    final String promptCache;
    /** {@code --prompt-cache-all}: every turn's tokens reach the session file, not the first prompt's alone. */
    final boolean promptCacheAll;
    /** {@code --prompt-cache-ro}: the session file is read but never written. */
    final boolean promptCacheRo;

    private LLMDemoOptions(LLMDemoOptionsParser p) {
        this.modelPath = p.modelPath;
        this.statsFile = p.statsFile;
        this.prompt = p.prompt;
        this.systemPrompt = p.systemPrompt;
        this.raw = p.raw;
        this.thinking = p.thinking;
        this.templateKwargs = Collections.unmodifiableMap(new LinkedHashMap<>(p.templateKwargs));
        this.debugPrompt = p.debugPrompt;
        this.nonInteractive = p.nonInteractive;
        this.mode = p.mode;
        this.showAllChannels = p.showAllChannels;
        this.benchmark = p.benchmark;
        this.benchStreaming = p.benchStreaming;
        this.memoryReportOnly = p.memoryReportOnly;
        this.noMmap = p.noMmap;
        this.noJinja = p.noJinja;
        this.embedPreview = p.embedPreview;
        this.tokenTypes = p.tokenTypes;
        this.document = p.document;
        this.decisionOrder = p.decisionOrder;
        this.audioCodes = p.audioCodes;
        this.audioOut = p.audioOut;
        this.mmproj = p.mmproj;
        this.mmprojUrl = p.mmprojUrl;
        this.noMmproj = p.noMmproj;
        this.mmprojOffload = p.mmprojOffload;
        this.mmprojDevice = p.mmprojDevice;
        this.imageMinTokens = p.imageMinTokens;
        this.imageMaxTokens = p.imageMaxTokens;
        this.audioLang = p.audioLang;
        this.voice = p.voice;
        this.audioSpeaker = p.audioSpeaker;
        this.audioFft = p.audioFft;
        this.audioHop = p.audioHop;
        this.audioRate = p.audioRate;
        this.help = p.help;
        this.version = p.version;
        this.listDevices = p.listDevices;
        this.cacheList = p.cacheList;
        this.serverBase = p.serverBase;
        this.apiKey = p.apiKey;
        this.outFile = p.outFile;
        this.multilineInput = p.multilineInput;
        this.interactive = p.interactive;
        this.interactiveFirst = p.interactiveFirst;
        this.inPrefix = p.inPrefix;
        this.inSuffix = p.inSuffix;
        this.inPrefixBos = p.inPrefixBos;
        this.escape = p.escape;
        this.singleTurn = p.singleTurn;
        this.media = Collections.unmodifiableList(new ArrayList<>(p.media));
        this.engine = p.engine;
        this.ubatchExplicit = p.ubatchExplicit;
        this.generation = p.generation;
        this.diffusion = p.diffusion;
        this.exportGguf = p.exportGguf;
        this.exportMode = p.exportMode;
        this.exportOverwrite = p.exportOverwrite;
        this.hfRepo = p.hfRepo;
        this.hfFile = p.hfFile;
        this.promptCache = p.promptCache;
        this.promptCacheAll = p.promptCacheAll;
        this.promptCacheRo = p.promptCacheRo;
    }

    /**
     * Parses {@code args}; a first argument that is not a flag is the model path.
     *
     * @throws UsageException for an unknown flag, a missing value or a value out of range
     */
    static LLMDemoOptions parse(String[] args) throws UsageException {
        LLMDemoOptionsParser p = new LLMDemoOptionsParser(args);
        p.run();
        return new LLMDemoOptions(p);
    }

    /**
     * The engine configuration for {@code resolved}: pooled modes (embed, classify, rerank) take a ubatch of
     * {@code max(}{@value #POOLED_UBATCH}{@code , ctx)} unless one was given, so long inputs fit one non-causal pass.
     */
    EngineConfig engineConfig(Mode resolved) {
        if (ubatchExplicit || !resolved.pooled()) {
            return engine;
        }
        MemoryConfig m = engine.memory();
        int ubatch = Math.max(POOLED_UBATCH, m.nCtx());
        return engine.toBuilder().memory(m.toBuilder().ubatch(ubatch).build()).build();
    }

    /**
     * Whether the generate path runs the interactive completion REPL: any of {@code -i}, {@code -if},
     * {@code --in-prefix}, {@code --in-suffix} or {@code --in-prefix-bos} given.
     */
    boolean completionInteractive() {
        return interactive || interactiveFirst || inPrefix != null || inSuffix != null || inPrefixBos;
    }

    /**
     * Whether any multimodal input flag besides {@code --mmproj} was given: media files, a projector URL, the
     * projector switches or the image token budget. The audio mode's speech-generator {@code --mmproj} stays
     * out, so the audio run never counts as multimodal input.
     */
    boolean multimodalInputGiven() {
        return !media.isEmpty() || mmprojUrl != null || noMmproj || mmprojOffload != null
                || mmprojDevice != null || imageMinTokens > 0 || imageMaxTokens > 0;
    }

    /** Where the vision encoder of a multimodal run runs ({@link MmprojOptions}). */
    MmprojOptions encoderOptions() {
        MmprojOptions options = mmprojDevice != null ? MmprojOptions.onDevice(mmprojDevice)
                : MmprojOptions.defaults();
        return mmprojOffload != null ? options.withOffload(mmprojOffload) : options;
    }

    /** The per-image token budget of {@code --image-min-tokens} / {@code --image-max-tokens}. */
    ImageTokenBudget imageBudget() {
        return ImageTokenBudget.builder().minTokens(imageMinTokens).maxTokens(imageMaxTokens).build();
    }


    /**
     * Refuses the decoding constraints a mode without an autoregressive token loop cannot apply: the grammar
     * ({@code --grammar}, {@code --json-schema} and their file forms) and the reasoning budget.
     *
     * @param modeName the mode, as the refusal names it
     * @throws IllegalArgumentException naming the flag and the mode
     */
    void requireNoDecodingConstraints(String modeName) {
        if (generation.constraint() != null) {
            throw new IllegalArgumentException("--grammar/--json-schema constrain autoregressive decoding; "
                    + modeName + " has no token loop to constrain");
        }
        if (generation.reasoningBudget() != GenerationRequest.REASONING_BUDGET_DISABLED) {
            throw new IllegalArgumentException("--reasoning-budget counts autoregressive reasoning tokens; "
                    + modeName + " has no token loop to count");
        }
    }

    /** The usage text ({@link LLMDemoUsage}). */
    static List<String> usage() {
        return LLMDemoUsage.lines();
    }
}
