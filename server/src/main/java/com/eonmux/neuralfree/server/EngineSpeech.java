package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.audio.SpeechFormat;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.speech.Lfm3aGenClip;
import com.eonmux.neuralfree.gguf.speech.Lfm3aSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.PocketTtsClip;
import com.eonmux.neuralfree.gguf.speech.PocketTtsSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsClip;
import com.eonmux.neuralfree.gguf.speech.Qwen3TtsSpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechGenerator;
import com.eonmux.neuralfree.gguf.speech.SpeechOptions;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The pool's text-to-speech and audio-decoding surface (llama.cpp server's {@code /v1/audio/*}): the integer-code
 * decoder path over the loaded model, and the generator-file backends (qwen3tts, pockettts, lfm3a) opened lazily on
 * the first speech request and cached until the projector releases. Every synthesis runs on the primary replica's
 * engine thread under the projector's lock discipline, because the pipelines reset and drive the model's forward
 * memory.
 */
final class EngineSpeech {

    /** The service's logger: the generator's release reports on the service it belongs to. */
    private static final Logger LOGGER = Logger.getLogger(EngineService.class.getName());

    private final EngineService owner;
    /** The text-to-speech generator over the primary replica's model; null until the first speech request. */
    private volatile SpeechGenerator speech;

    EngineSpeech(EngineService owner) {
        this.owner = owner;
    }

    /** Closes and forgets the cached generator, if one is open; the service's release and close call it under the
     * projector's lock. A failure is logged; the generator is forgotten either way. */
    void closeGenerator() {
        synchronized (owner.projectorLock) {
            SpeechGenerator generator = speech;
            speech = null;
            if (generator != null) {
                try {
                    generator.close();
                } catch (RuntimeException e) {
                    LOGGER.log(Level.WARNING, "speech generator release failed", e);
                }
            }
        }
    }

    /**
     * The speech decoding geometry the loaded model declares ({@link LLMModel#speechFormat}), or null when it
     * cannot turn its features into a waveform.
     */
    public SpeechFormat speechFormat() {
        if (owner.dual() != null) {
            return null;
        }
        return owner.primary().model().speechFormat();
    }

    /**
     * The waveform of audio codec codes through the loaded model's speech decoder
     * ({@link LLMModel#synthesizeSpeech}).
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    public AudioClip synthesizeSpeech(int[] codes) {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        return owner.primary().model().synthesizeSpeech(codes);
    }

    /**
     * Whether the loaded model turns text into audio, i.e. whether {@link #synthesizeSpeech(String, SpeechOptions)}
     * has a speech-generator backbone to reach. A qwen3tts, pockettts or lfm3a backbone answers true even before
     * its generator file is opened, so the route can prefer text over the integer-code path and let the generator
     * name the missing file.
     */
    public boolean supportsTextSpeech() {
        if (owner.dual() != null || owner.replicas() == null || owner.replicas().isEmpty()) {
            return false;
        }
        Arch arch = owner.primary().model().spec().arch();
        return arch == Arch.QWEN3TTS || arch == Arch.POCKETTTS || arch == Arch.LFM3;
    }

    /**
     * Synthesizes speech from {@code text} through the loaded model's speech generator: a qwen3tts backbone with
     * the code predictor and code2wav decoder of the {@code neuralfree.server.mmproj} generator file, a pockettts
     * backbone with that file's flow network and mimi decoder, or an lfm3a backbone with its MaskGIT depthformer
     * and liquid-codec decoder. Runs on the primary replica's engine thread once its batch engine is idle,
     * because the pipeline resets and drives the model's forward memory.
     *
     * @param text    the text to speak
     * @param options the language, voice, sampling and frame cap
     * @return the mono waveform at the generator file's sample rate
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, no generator file is
     *                                       configured, or the lfm3 omni backbone carries no speech chat
     *                                       template (the message names the missing piece)
     * @throws IllegalArgumentException      when {@code text} or an option is invalid, or the rendered prompt does
     *                                       not open the omni pipeline's fresh turn
     * @throws IOException                   when the generation fails or the engine stays busy past its budget
     */
    public AudioClip synthesizeSpeech(String text, SpeechOptions options) throws IOException {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        SpeechGenerator generator = speechGenerator();
        if (generator instanceof Lfm3aSpeechGenerator omni) {
            // llama.cpp speaks the tts template's render, which must end in the audio start token
            // (server-context.cpp:6477-6478, mtmd-helper-gen.cpp:1337): the render already carries it, so the
            // fresh turn runs over it directly instead of the generator appending one to the raw text
            GGUFTokenizer tokenizer = owner.primary().model().tokenizer();
            if (tokenizer.ttsTemplate() == null) {
                throw new UnsupportedOperationException("lfm3a text-to-speech needs the "
                        + GGUFTokenizer.KEY_TTS_TEMPLATE + " chat template: the omni pipeline speaks the prompt"
                        + " rendered from it, and the render ends in the audio start token");
            }
            String rendered = tokenizer.formatTtsPrompt(text, options.language(), options.voice());
            int[] prompt = tokenizer.encode(rendered, false, true);
            return owner.primary().onPump(() -> omni.generate(prompt, Lfm3aSpeechGenerator.Mode.AUDIO, false, options,
                    null).audio());
        }
        return owner.primary().onPump(() -> generator.synthesize(text, options));
    }

    /**
     * {@link #synthesizeSpeech(String, SpeechOptions)} delivered in decoder chunks: the generator hands each
     * clip to {@code sink} as it decodes it, on the primary replica's engine thread. The clips concatenate to the
     * one-shot result. {@code sink} may throw to stop the generation (a disconnected client); the throw reaches
     * this call.
     *
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, or no generator file
     *                                       is configured (the message names the missing piece)
     * @throws IllegalArgumentException      when {@code text} or an option is invalid
     * @throws IOException                   when the generation fails or the engine stays busy past its budget
     */
    public void synthesizeSpeech(String text, SpeechOptions options, Consumer<AudioClip> sink) throws IOException {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        Objects.requireNonNull(sink, "sink");
        SpeechGenerator generator = speechGenerator();
        owner.primary().onPump(() -> {
            generator.synthesize(text, options, sink);
            return null;
        });
    }

    /**
     * The sample rate of the text-to-speech generator's waveform; opens the generator file on first use, so a
     * missing backbone or file refuses here by name before a streamed response starts.
     *
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, or no generator file
     *                                       is configured
     * @throws IOException                   when the generator file cannot be read
     */
    public int speechSampleRate() throws IOException {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        return speechGenerator().sampleRate();
    }

    /**
     * {@link #synthesizeSpeech(int[])} delivered in chunks through the model's streaming speech decoder.
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    public void synthesizeSpeech(int[] codes, Consumer<AudioClip> sink) {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        Objects.requireNonNull(sink, "sink");
        owner.primary().model().synthesizeSpeech(codes, sink);
    }

    /**
     * The sample rate of the code-to-waveform decoder, refusing like {@link #synthesizeSpeech(int[])} when the
     * model has none so a streamed response starts only for a model that can fill it.
     *
     * @throws UnsupportedOperationException when the model is not a speech decoder or declares no geometry
     */
    public int speechCodeSampleRate() {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        return owner.primary().model().requireSpeechFormat().sampleRate();
    }

    /**
     * One omni chat turn: the conversation renders through the model's conversation template
     * ({@link GGUFTokenizer#formatSpeechChat}, with the request's {@code language} and {@code voice} bound), and
     * the lfm3a omni pipeline answers it. An AUDIO-only turn is fresh and stops at the end-of-audio frame, its
     * prompt ending in the audio start token; a MIXED turn continues the conversation, the pipeline appending the
     * mixed start token, and interleaves text and audio blocks until {@code <|im_end|>} or the step cap
     * (mtmd-helper-gen.cpp:1320-1352, 1392-1450). Runs on the primary replica's engine thread like
     * {@link #synthesizeSpeech(String, SpeechOptions)}, because the pipeline drives the model's forward memory.
     *
     * @param chat       the conversation to answer
     * @param mode       {@link Lfm3aSpeechGenerator.Mode#AUDIO} for speech only, {@link Lfm3aSpeechGenerator.Mode#MIXED}
     *                   for interleaved text and speech
     * @param continuing true for a turn that continues the conversation (the pipeline appends the mode's switch
     *                   token and lets the response continue in text after the audio); false for a fresh AUDIO turn
     * @param options    the text-step sampler, the language and voice variables and the step cap
     * @param sink       receives the turn's text pieces and audio chunks as they happen, or null for none
     * @return the turn's text, whole waveform and prompt length
     * @throws UnsupportedOperationException when the loaded model has no speech-generator backbone, no generator
     *                                       file, or when its generator is not the lfm3a omni pipeline (the message
     *                                       names the missing piece)
     * @throws IllegalArgumentException      when the conversation does not render, or a fresh prompt is not an
     *                                       audio turn
     * @throws IOException                   when the generation fails or the engine stays busy past its budget
     */
    public Lfm3aSpeechGenerator.OmniTurn omniTurn(ChatRequest chat, Lfm3aSpeechGenerator.Mode mode, boolean continuing,
                                                SpeechOptions options, Lfm3aSpeechGenerator.OmniSink sink)
            throws IOException {
        if (owner.dual() != null) {
            throw new UnsupportedOperationException("a CLIP dual encoder does not support audio generation");
        }
        if (owner.replicas() == null || owner.replicas().isEmpty()) {
            throw new UnsupportedOperationException("no model loaded: chat completions audio output needs the lfm3a"
                    + " omni pipeline");
        }
        Arch arch = owner.primary().model().spec().arch();
        if (arch != Arch.LFM3) {
            throw new UnsupportedOperationException("chat completions audio output needs the lfm3a omni pipeline:"
                    + (arch == Arch.QWEN3TTS
                            ? " the loaded qwen3tts backbone speaks through /v1/audio/speech, which does not run"
                              + " the omni chat turn"
                            : " the loaded " + arch.ggufName() + " model has no speech-generator backbone; a"
                              + " qwen3tts or lfm3 backbone with its generator file speaks"));
        }
        SpeechGenerator generator = speechGenerator();
        if (!(generator instanceof Lfm3aSpeechGenerator omni)) {
            throw new UnsupportedOperationException("chat completions audio output needs the lfm3a omni pipeline;"
                    + " the loaded speech generator is " + generator.name() + ", which speaks through"
                    + " /v1/audio/speech");
        }
        GGUFTokenizer tokenizer = owner.primary().model().tokenizer();
        String rendered = tokenizer.formatSpeechChat(chat, options.language(), options.voice());
        int[] prompt = tokenizer.encode(rendered, false, true);
        if (!continuing) {
            // the pipeline decides a fresh turn's mode from the last prompt token, and only an audio turn is valid
            // (mtmd-helper-gen.cpp:1336): the render must end in audio_start
            if (mode != Lfm3aSpeechGenerator.Mode.AUDIO) {
                throw new IllegalArgumentException("a fresh omni turn opens in audio; a " + mode
                        + " turn continues the conversation");
            }
            int audioStart = tokenizer.getTokenId(Lfm3aSpeechGenerator.AUDIO_START_PIECE);
            int[] withStart = Arrays.copyOf(prompt, prompt.length + 1);
            withStart[prompt.length] = audioStart;
            prompt = withStart;
        }
        int[] tokens = prompt;
        return owner.primary().onPump(() -> omni.generate(tokens, mode, continuing, options, sink));
    }

    /**
     * The speech generator of the pool, opened on first use from the generator file and the primary replica's
     * model and cached; models without a speech-generator backbone are refused by name.
     *
     * @throws UnsupportedOperationException when the model has no speech-generator backbone, or no generator file
     *                                       is configured
     * @throws IOException                   when the generator file cannot be read
     */
    private SpeechGenerator speechGenerator() throws IOException {
        synchronized (owner.projectorLock) {
            if (speech != null) {
                return speech;
            }
            if (owner.replicas() == null || owner.replicas().isEmpty()) {
                throw new UnsupportedOperationException("no model loaded: speech generation needs a qwen3tts,"
                        + " pockettts or lfm3a backbone");
            }
            EngineModel model = owner.primary().model();
            Arch arch = model.spec().arch();
            if (arch != Arch.QWEN3TTS && arch != Arch.POCKETTTS && arch != Arch.LFM3) {
                throw new UnsupportedOperationException("this server has no speech-generator backbone: the loaded "
                        + arch.ggufName() + " model cannot turn text into audio; a qwen3tts, pockettts or lfm3a"
                        + " backbone with its generator file does");
            }
            if (owner.mmprojPath() == null) {
                throw new UnsupportedOperationException("speech generation needs the generator file: set "
                        + MediaOptions.MMPROJ_PROPERTY + " (llama.cpp --mmproj) to the " + generatorFileOf(arch));
            }
            speech = openSpeechGenerator(model, arch);
            return speech;
        }
    }

    /** The generator file an arch's speech pipeline reads, for the missing-file message. */
    private static String generatorFileOf(Arch arch) {
        switch (arch) {
            case QWEN3TTS:
                return "qwen3tts code predictor and code2wav GGUF";
            case POCKETTTS:
                return "pockettts flow network and mimi decoder GGUF";
            case LFM3:
                return "lfm3a MaskGIT depthformer and liquid-codec GGUF";
            default:
                return arch.ggufName() + " generator GGUF";
        }
    }

    /** Opens the generator file of {@code arch} and binds it to {@code model}; closes it on a binding failure. */
    private SpeechGenerator openSpeechGenerator(EngineModel model, Arch arch) throws IOException {
        switch (arch) {
            case QWEN3TTS: {
                Qwen3TtsClip clip = Qwen3TtsClip.open(owner.mmprojPath(), owner.config().sourceOptions());
                try {
                    return Qwen3TtsSpeechGenerator.open(model, clip);
                } catch (RuntimeException | Error e) {
                    clip.close();
                    throw e;
                }
            }
            case POCKETTTS: {
                PocketTtsClip clip = PocketTtsClip.open(owner.mmprojPath(), owner.config().sourceOptions());
                try {
                    return PocketTtsSpeechGenerator.open(model, clip);
                } catch (IOException | RuntimeException | Error e) {
                    clip.close();
                    throw e;
                }
            }
            case LFM3: {
                Lfm3aGenClip clip = Lfm3aGenClip.open(owner.mmprojPath(), owner.config().sourceOptions());
                try {
                    return Lfm3aSpeechGenerator.open(model, clip);
                } catch (IOException | RuntimeException | Error e) {
                    clip.close();
                    throw e;
                }
            }
            default:
                throw new UnsupportedOperationException("the " + arch.ggufName()
                        + " backbone has no speech generator");
        }
    }
}
