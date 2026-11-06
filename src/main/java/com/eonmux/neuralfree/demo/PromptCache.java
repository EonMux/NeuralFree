package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The prompt cache of the {@code cli} command (llama.cpp's {@code --prompt-cache} over the completion loop's
 * session file): the model's {@code SessionFile} state loads from the file at start, so the generation loop's
 * prefix reuse skips the tokens a matching prompt shares with the saved sequence, and the state saves back to the
 * file after turns — the first turn's alone, or every turn's prompt and generated tokens with
 * {@code --prompt-cache-all}. {@code --prompt-cache-ro} loads and never saves. An absent or empty file starts a
 * new session (llama.cpp's "will create").
 */
final class PromptCache {

    private final com.eonmux.neuralfree.gguf.engine.EngineModel model;
    private final Path file;
    /** Whether every turn saves, not the first one alone ({@code --prompt-cache-all}). */
    private final boolean all;
    /** Whether the file stays untouched ({@code --prompt-cache-ro}). */
    private final boolean readOnly;
    /** Whether the one-shot save of the plain cache already ran. */
    private boolean saved;

    private PromptCache(com.eonmux.neuralfree.gguf.engine.EngineModel model, Path file, boolean all,
                        boolean readOnly) {
        this.model = model;
        this.file = file;
        this.all = all;
        this.readOnly = readOnly;
    }

    /**
     * Opens the cache over {@code model} when {@code --prompt-cache} was given, loading the saved session into the
     * model's memory; null without the flag. A decoder-only model carries the session file ({@code
     * EngineModel.saveSession}); the others refuse by name.
     *
     * @throws UnsupportedOperationException for a model outside the session file's scope, naming it
     * @throws IOException                   for an unreadable or foreign session file
     */
    static PromptCache open(com.eonmux.neuralfree.gguf.engine.EngineModel model, LLMDemoOptions opts)
            throws IOException {
        if (opts.promptCache == null) {
            return null;
        }
        if (model.isEncoderDecoder()) {
            throw new UnsupportedOperationException("--prompt-cache runs a decoder-only model; an encoder-decoder"
                    + " holds no reusable prompt prefix");
        }
        Path file = Paths.get(opts.promptCache);
        if (Files.exists(file) && !Files.isRegularFile(file)) {
            throw new IOException("--prompt-cache: '" + file + "' is not a regular file");
        }
        if (Files.exists(file) && Files.size(file) > 0) {
            System.out.println("Prompt cache: loading the saved session from '" + file + "'");
            model.loadSession(file);
        } else {
            System.out.println("Prompt cache: the session file does not exist (or is empty), will create.");
        }
        return new PromptCache(model, file, opts.promptCacheAll, opts.promptCacheRo);
    }

    /**
     * The generator that saves the session after turns as the cache's mode says; the given one, when the cache
     * never writes.
     */
    TurnGenerator wrap(TurnGenerator generator) {
        if (readOnly) {
            return generator;
        }
        return new TurnGenerator() {
            @Override
            public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                             String renderedPrompt) throws IOException {
                GenerationResult result = generator.generate(prompt, request, sink, renderedPrompt);
                saveAfterTurn();
                return result;
            }

            @Override
            public GenerationRequest effective(GenerationRequest request) {
                return generator.effective(request);
            }

            @Override
            public void reset() {
                generator.reset();
            }
        };
    }

    /** Saves the model's state to the file: every turn under {@code --prompt-cache-all}, else the first alone. */
    private void saveAfterTurn() throws IOException {
        if (!all && saved) {
            return;
        }
        model.saveSession(file);
        saved = true;
    }
}
