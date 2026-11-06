package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EmbedsPrompt;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSlotKey;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * A completions body made ready for the engine: the served model name, the sampling request, the output constraint
 * and the prompts, each with its encoded tokens, its precomputed embedding rows (vLLM's {@code prompt_embeds}) or
 * its media prompt (llama.cpp's {@code multimodal_data}), and its echo text. The OpenAI completions route and the offline batch runner ({@link BatchRunner}) share it.
 * Immutable.
 *
 * @param model        the served model name the response carries
 * @param top          the {@code logprobs} alternatives count, or -1 without logprobs
 * @param base         the sampling request of choice 0
 * @param constraint   the output constraint, or null
 * @param n            the choices per prompt
 * @param bestOf       the candidates generated per prompt ({@code best_of}): {@code n} unless the request asks
 *                     for more, the best {@code n} of them answering
 * @param prompts        the prompts, one per prompt: token ids ({@code truncate_prompt_tokens} applied) or
 *                       embedding rows, the {@code prompt_embeds} first
 * @param echoes         the echo text per choice (empty when {@code echo} is off), prompt-major like the jobs
 * @param promptTokens   the prompts' positions (tokens and embedding rows), each distinct prompt counted once
 * @param reasoning      the request's reasoning filter ({@code reasoning_format}), or the configured one
 * @param returnTokenIds whether each choice carries {@code token_ids} and {@code prompt_token_ids}
 * @param tokensAsIds    whether logprob tokens render as {@code token_id:{id}} ({@code return_tokens_as_token_ids})
 */
record CompletionPlan(String model, int top, GenerationRequest base, ConstraintSpec constraint, int n, int bestOf,
                      List<Input> prompts, List<String> echoes, long promptTokens,
                      ReasoningFilter reasoning, boolean returnTokenIds, boolean tokensAsIds) {

    /**
     * One prompt for the engine: token ids, or a media prompt (precomputed embedding rows, or the encoded media
     * of a {@code multimodal_data} prompt) — exactly one of the two; {@code cut} when
     * {@code truncate_prompt_tokens} shortened the token ids. {@code placements} names the media items a media
     * prompt spliced in, and {@code offsets} the character span of each token when the request asked for them.
     */
    record Input(int[] tokens, MediaPrompt media, List<ChatMedia.Placement> placements, List<int[]> offsets,
                 boolean cut) {
        Input {
            if ((tokens == null) == (media == null)) {
                throw new IllegalArgumentException("a completion prompt takes token ids or a media prompt, exactly"
                        + " one of the two");
            }
            placements = placements == null ? List.of() : placements;
        }

        /** The KV rows the prompt fills. */
        int rows() {
            return tokens != null ? tokens.length : media.nTokens();
        }
    }

    /** Plans {@code body}, a completions body. */
    static CompletionPlan of(EngineService service, JsonView body) {
        String model = service.loras().model(body);
        int n = OpenAiRequests.choices(body);
        int top = OpenAiRequests.topLogprobs(body, false);
        boolean echo = body.optBool("echo", false);
        GenerationRequest base = service.loras().apply(body, model,
                OpenAiRequests.sampling(body, top, service.vocabSize(), echo ? top : -1,
                        service.options().logprobsMode(), service.logitsProcessors(),
                        service.options().watermark()));
        OpenAiRequests.requireServedPriority(base, service.schedulingPolicy());
        int bestOf = OpenAiRequests.bestOf(body, n, body.optBool("stream", false));
        base = OpenAiRequests.scoredCandidates(base, n, bestOf);
        base = OpenAiRequests.withBeamSearch(base, body, n);
        ConstraintSpec constraint = OpenAiRequests.constraint(body);
        List<OpenAiRequests.Prompt> raw = OpenAiRequests.prompts(body, service.vocabSize(),
                service.embeddingLength());
        OpenAiRequests.requireGenerationBudget(raw.size(), bestOf);
        OpenAiRequests.requireSingleTransfer(base, raw.size(), n, bestOf);
        if (echo && body.has("prompt_embeds")) {
            // vLLM's refusal: an embeddings prompt has no text to echo
            throw ServerJson.badRequest("Echo is unsupported with prompt embeds.");
        }
        List<Input> prompts = new ArrayList<>();
        List<String> echoes = new ArrayList<>();
        long promptTokens = 0;
        boolean offsetsWanted = body.optBool("return_token_offsets", false);
        Supplier<LoraSlotKey> mediaLoras = ChatPlan.mediaLoras(service, base.lora());
        for (OpenAiRequests.Prompt p : raw) {
            Input input = input(service, body, p, offsetsWanted, mediaLoras);
            prompts.add(input);
            promptTokens += input.rows();
            // a truncated prompt echoes its kept tokens, not the text it was cut from
            String echoed = !echo ? "" : input.cut() || p.text() == null ? service.decode(input.tokens()) : p.text();
            for (int i = 0; i < n; i++) {
                echoes.add(echoed);
            }
        }
        // llama.cpp's per-request reasoning_format over the configured parser
        ReasoningFilter reasoning = ReasoningFormats.fromBody(body, service.reasoning(),
                body.optBool("stream", false));
        return new CompletionPlan(model, top, base, constraint, n, bestOf, prompts, echoes, promptTokens, reasoning,
                body.optBool("return_token_ids", false), body.optBool("return_tokens_as_token_ids", false));
    }

    /**
     * The response-body builder of the plan: its model, logprobs count, token rendering, and whether captures
     * hold the requested {@code logprob_token_ids}.
     */
    OpenAiBodies bodies(EngineService service) {
        return new OpenAiBodies(service, top, model, tokensAsIds, base.namesLogprobTokenIds());
    }

    /** The prompt tokens choice {@code choice} answers (choices are prompt-major), or null for an embeddings
     *  prompt, which has no token ids. */
    int[] promptOf(int choice) {
        return prompts.get(choice / n).tokens();
    }

    /** Whether the echoed prompt's logprobs merge into the legacy arrays ({@code echo} with {@code logprobs}). */
    boolean echoesLogprobs() {
        return top >= 0;
    }

    /**
     * The engine input of one parsed prompt: its embedding rows, its text with the media of its
     * {@code multimodal_data}, its token ids, or its text encoded.
     */
    private static Input input(EngineService service, JsonView body, OpenAiRequests.Prompt p,
                               boolean offsetsWanted, Supplier<LoraSlotKey> mediaLoras) {
        if (p.multimodalData() != null) {
            if (body.has("truncate_prompt_tokens")) {
                throw ServerJson.badRequest("truncate_prompt_tokens cannot be combined with multimodal_data");
            }
            ChatMedia.Rendered media = LlamaPrompts.mediaPromptRendered(p.text(), p.multimodalData(), service,
                    p.key(), mediaLoras);
            return new Input(null, media.prompt(), media.placements(), null, false);
        }
        if (p.embeds() != null) {
            if (body.has("truncate_prompt_tokens")) {
                throw ServerJson.badRequest("truncate_prompt_tokens cannot be combined with prompt_embeds");
            }
            try {
                return new Input(null, EmbedsPrompt.ofRows(p.embeds()).toMediaPrompt(), List.of(), null, false);
            } catch (IllegalArgumentException e) {
                throw ServerJson.badRequest("prompt_embeds: " + e.getMessage());
            }
        }
        int[] encoded = p.tokens() != null ? p.tokens() : service.encode(p.text(), true, true);
        int[] tokens = OpenAiRequests.truncate(body, encoded);
        if (tokens.length == 0) {
            throw ServerJson.badRequest("a prompt encodes to no tokens");
        }
        List<int[]> offsets = offsetsWanted && p.tokens() == null
                ? ScaleOutOffsets.slice(ScaleOutOffsets.of(service, p.text(), encoded), tokens, body) : null;
        return new Input(tokens, null, List.of(), offsets, tokens != encoded);
    }

    /** The engine jobs of every prompt times its candidates ({@code bestOf}), prompt-major: prompt {@code p}
     *  fills job slots {@code p * bestOf + i}, each with its own seed offset. */
    List<GenerationBatch.Job> jobs() {
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        for (Input input : prompts) {
            for (int i = 0; i < bestOf; i++) {
                jobs.add(new GenerationBatch.Job(input.tokens(), input.media(), OpenAiRequests.forChoice(base, i),
                        null, constraint, reasoning));
            }
        }
        return jobs;
    }
}
