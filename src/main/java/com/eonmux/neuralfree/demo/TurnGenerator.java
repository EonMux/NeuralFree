package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LookaheadDecoder;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;

import java.io.IOException;

/**
 * One way a console session generates a turn: the engine loop, a speculative session, lookahead decoding, or a
 * model's own {@code generate}. The console chat and the completion REPL take one of these instead of a concrete
 * engine object, so a session runs over any of these generation paths.
 */
public interface TurnGenerator {

    /**
     * Generates the turn's response after {@code prompt}, streaming the routed output to {@code sink}.
     *
     * @param renderedPrompt the rendered prompt text, or null to let the engine render it from the tokens
     */
    GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                              String renderedPrompt) throws IOException;

    /** The request {@link #generate} runs for {@code request}, with the model's sampling defaults applied. */
    GenerationRequest effective(GenerationRequest request);

    /** Drops the session's cached state, so the next turn starts fresh. */
    void reset();

    /** The engine {@link GenerationLoop} of {@code model} under the effective config {@code effective}. */
    static TurnGenerator loop(EngineModel model, EngineConfig effective) {
        GenerationLoop loop = GenerationLoop.forModel(model, effective);
        return new TurnGenerator() {
            @Override
            public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                             String rendered) {
                return loop.generate(prompt, request, sink, rendered);
            }

            @Override
            public GenerationRequest effective(GenerationRequest request) {
                return request.withModelDefaults(loop.samplingDefaults());
            }

            @Override
            public void reset() {
                model.resetKVCache();
            }
        };
    }

    /**
     * The {@link LookaheadDecoder} over {@code model}: lookahead (Jacobi) decoding instead of the plain loop. Each
     * generation resets the memory, so {@link #reset()} also empties it.
     */
    static TurnGenerator lookahead(EngineModel model, LookaheadDecoder decoder) {
        return new TurnGenerator() {
            @Override
            public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                             String rendered) {
                return decoder.generate(prompt, request, sink, rendered);
            }

            @Override
            public GenerationRequest effective(GenerationRequest request) {
                return request.withModelDefaults(decoder.samplingDefaults());
            }

            @Override
            public void reset() {
                model.resetKVCache();
            }
        };
    }

    /**
     * A model's own {@link LLMModel#generate(int[], GenerationRequest, com.eonmux.neuralfree.gguf.TokenCallback)},
     * every piece on the content channel.
     */
    static TurnGenerator ofModel(LLMModel model) {
        return new TurnGenerator() {
            @Override
            public GenerationResult generate(int[] prompt, GenerationRequest request, ReasoningFilter.Sink sink,
                                             String rendered) throws IOException {
                return model.generate(prompt, request,
                        (id, text) -> sink.accept(ReasoningFilter.Channel.CONTENT, id, text));
            }

            @Override
            public GenerationRequest effective(GenerationRequest request) {
                return request;
            }

            @Override
            public void reset() {
                model.resetKVCache();
            }
        };
    }
}
