package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.OutputGates;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.List;

/**
 * A chat-completions body made ready for the engine: the served model name, the sampling request, the output
 * constraint (gated behind the reasoning), the rendered prompt and its tokens, and the tool routing. The OpenAI chat
 * route and the Responses API (whose body converts to a chat body) share it. Immutable.
 *
 * @param model     the served model name the response carries
 * @param top       the {@code top_logprobs} count, or -1 without logprobs
 * @param base      the sampling request of choice 0
 * @param constraint the output constraint, or null
 * @param chat      the chat request
 * @param rendered  the rendered prompt
 * @param routing   the tool routing of the rendered prompt
 * @param tokens    the prompt tokens
 * @param named     whether {@code tool_choice} names one function
 */
record ChatPlan(String model, int top, GenerationRequest base, ConstraintSpec constraint, ChatRequest chat,
                String rendered, ToolRouting routing, int[] tokens, boolean named) {

    /** Plans {@code body}, a chat-completions body. */
    static ChatPlan of(EngineService service, JsonView body) {
        String model = service.served().resolve(body);
        int top = OpenAiRequests.topLogprobs(body, true);
        GenerationRequest base = OpenAiRequests.sampling(body, top);
        ConstraintSpec constraint = OpenAiRequests.constraint(body);
        ChatRequest chat = OpenAiRequests.chat(body);
        String rendered = service.formatChat(chat);
        if (constraint != null) {
            // the constraint holds the answer, not the reasoning before it
            constraint = constraint.withGate(OutputGates.forChat(service.reasoning(), rendered,
                    service.chatTemplate()));
        }
        ToolRouting routing = service.toolRouting(chat, rendered, OpenAiRequests.parallelToolCalls(body));
        if (routing.constraint() != null) {
            if (constraint != null) {
                throw ServerJson.badRequest("tool_choice " + chat.toolChoice() + " constrains the output; it"
                        + " cannot be combined with another output constraint");
            }
            constraint = routing.constraint();
        }
        boolean named = chat.toolChoice() != null && chat.toolChoice().kind() == ChatRequest.ToolChoice.Kind.FUNCTION;
        int[] tokens = service.encode(rendered, false, true);
        return new ChatPlan(model, top, base, constraint, chat, rendered, routing, tokens, named);
    }

    /** The engine jobs of {@code n} choices, each with its own seed offset. */
    List<GenerationBatch.Job> jobs(int n) {
        List<GenerationBatch.Job> jobs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            jobs.add(new GenerationBatch.Job(tokens, OpenAiRequests.forChoice(base, i), rendered, constraint));
        }
        return jobs;
    }
}
