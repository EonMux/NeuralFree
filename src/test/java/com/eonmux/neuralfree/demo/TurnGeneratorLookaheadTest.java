package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.LookaheadDecoder;
import com.eonmux.neuralfree.gguf.engine.LookaheadOptions;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The console's lookahead turn generator over the synthetic llama: it delegates the turn to the
 * {@link LookaheadDecoder}, applies the decoder's sampling defaults, and its reset restarts the next turn from the
 * prompt.
 */
public class TurnGeneratorLookaheadTest {

    private static final ReasoningFilter.Sink DISCARD = (channel, tokenId, text) -> { };

    private static EngineConfig config() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(1024).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(7).kvUnified(true).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory)
                .lookaheadOptions(LookaheadOptions.builder().enabled(true).window(4).ngramSize(3).maxNgrams(2)
                        .build())
                .build();
    }

    @Test
    public void aTurnDecodesThroughTheDecoderAndResetRestartsIt() throws IOException {
        Path file = SyntheticModels.llama().writeTempFile();
        EngineConfig cfg = config();
        try (EngineModel model = LLMModelFactory.open(file.toString(), cfg)) {
            TurnGenerator generator = TurnGenerator.lookahead(model, LookaheadDecoder.forModel(model, cfg));
            int[] prompt = {4, 5, 6, 4, 5, 7};
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(6)
                    .sampler(SamplerConfig.greedyDecoding()).build();
            GenerationResult first = generator.generate(prompt, generator.effective(request), DISCARD, null);
            assertEquals(StopReason.MAX_TOKENS, first.stopReason());
            assertEquals(6, first.tokens().length);
            generator.reset();
            GenerationResult second = generator.generate(prompt, generator.effective(request), DISCARD, null);
            assertArrayEquals("a reset turn restarts from the prompt", first.tokens(), second.tokens());
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
