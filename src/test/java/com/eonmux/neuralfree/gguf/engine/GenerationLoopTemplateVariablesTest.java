package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The chat template variables of a {@link GenerationRequest} select the configuration of a per-request reasoning
 * parser in the core loop, as the server's request does: a generation with {@code enable_thinking} off parses as
 * content, the same generation with it on as reasoning the parser opens at the output's first token.
 */
public class GenerationLoopTemplateVariablesTest {

    /** Thinking on: reasoning opens at the output's first token; thinking off: plain content. */
    private static final ReasoningFilter PER_REQUEST = ReasoningFilter.perRequest(vars ->
            Boolean.FALSE.equals(vars.get("enable_thinking")) ? ReasoningFilter.none()
                    : ReasoningFilter.thinkTags().opensImplicitly());

    private static Set<ReasoningFilter.Channel> channels(GenerationLoop loop, Map<String, ?> variables) {
        GenerationRequest request = GenerationRequest.builder()
                .maxNewTokens(6)
                .sampler(SamplerConfig.greedyDecoding())
                .templateVariables(variables)
                .build();
        Set<ReasoningFilter.Channel> seen = EnumSet.noneOf(ReasoningFilter.Channel.class);
        loop.generate(new int[]{4, 17, 30, 43, 56, 69}, request,
                (channel, tokenId, text) -> {
                    if (!text.isEmpty()) seen.add(channel);
                }, "");
        return seen;
    }

    @Test
    public void theRequestsEnableThinkingSelectsTheParserConfigurationOfTheLoop() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        MemoryConfig mc = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig cfg = EngineConfig.builder().memory(mc).build();
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            GenerationLoop loop = GenerationLoop.builder(one.forwardBackend(), one.tokenizer())
                    .reasoning(PER_REQUEST).memoryConfig(mc).build();
            assertEquals(EnumSet.of(ReasoningFilter.Channel.REASONING),
                    channels(loop, Collections.singletonMap("enable_thinking", true)));
            assertEquals(EnumSet.of(ReasoningFilter.Channel.CONTENT),
                    channels(loop, Collections.singletonMap("enable_thinking", false)));
        }
    }

    @Test
    public void theLoopResolvesTheConfiguredFilterForTheRequestsVariables() {
        GenerationRequest off = GenerationRequest.builder()
                .templateVariables(Collections.singletonMap("enable_thinking", false)).build();
        assertEquals(ReasoningFilter.Mode.NONE, GenerationLoop.forRequest(PER_REQUEST, off).mode());
        ReasoningFilter on = GenerationLoop.forRequest(PER_REQUEST, GenerationRequest.builder().build());
        assertEquals(ReasoningFilter.Mode.THINK_TAGS, on.mode());
        assertTrue(on.startsInReasoning(""));
    }

    @Test
    public void templateVariablesAreCopiedCarriedByToBuilderAndValidated() {
        Map<String, Object> vars = new LinkedHashMap<>();
        vars.put("reasoning_effort", "low");
        GenerationRequest request = GenerationRequest.builder().templateVariables(vars).build();
        vars.put("enable_thinking", true);
        assertEquals(Collections.singletonMap("reasoning_effort", "low"), request.templateVariables());
        assertEquals(request.templateVariables(), request.toBuilder().build().templateVariables());
        assertTrue(GenerationRequest.builder().templateVariables(null).build().templateVariables().isEmpty());
        try {
            GenerationRequest.builder().templateVariables(Collections.singletonMap("", 1));
            throw new AssertionError("expected a refusal");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("non-empty"));
        }
    }
}
