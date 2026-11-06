package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The llama.cpp-native endpoints, with the vLLM spellings of the shared ones:
 *
 * <ul>
 *   <li>{@code GET /health} (also {@code /v1/health}) reports the load state; {@code GET /props} the serving
 *       setup (model path, architecture, context, slots, chat template). The Quarkus health extension serves the
 *       Kubernetes probes separately under {@code /q/health}.</li>
 *   <li>{@code POST /tokenize}: llama.cpp's {@code content} (special tokens off by default,
 *       {@code with_pieces} adds each token's text) or vLLM's {@code prompt} (special tokens on by default) or
 *       chat {@code messages} rendered through the template; the answer carries llama.cpp's {@code tokens} and
 *       vLLM's {@code count} and {@code max_model_len}.</li>
 *   <li>{@code POST /detokenize}: {@code tokens} to text, as llama.cpp's {@code content} and vLLM's
 *       {@code prompt}.</li>
 *   <li>{@code POST /apply-template}: a chat body rendered through the template, as {@code prompt}.</li>
 * </ul>
 */
@Path("/")
public class LlamaCppResource {

    @Inject
    EngineService service;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public LlamaCppResource(EngineService service) {
        this.service = service;
    }

    public LlamaCppResource() {
    }

    @GET
    @Path("/health")
    @Produces(MediaType.APPLICATION_JSON)
    public Response health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", service.loaded() ? "ok" : "error");
        if (!service.loaded()) {
            body.put("error", ServerJson.NO_MODEL);
        }
        return Response.ok(ServerJson.write(body)).build();
    }

    @GET
    @Path("/v1/health")
    @Produces(MediaType.APPLICATION_JSON)
    public Response healthAlias() {
        return health();
    }

    @GET
    @Path("/props")
    @Produces(MediaType.APPLICATION_JSON)
    public Response props() {
        ServerJson.requireLoaded(service);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model_alias", service.served().primary());
        body.put("model_path", service.modelPath());
        body.put("model_arch", service.architecture());
        Map<String, Object> total = new LinkedHashMap<>();
        total.put("n_ctx", service.contextLength());
        total.put("n_slots", service.slots());
        body.put("total_slots", service.slots());
        body.put("default_generation_settings", total);
        body.put("chat_template", service.chatTemplate());
        return Response.ok(ServerJson.write(body)).build();
    }

    @POST
    @Path("/tokenize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response tokenize(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> tokenizeOf(ServerJson.parse(bodyText)));
    }

    @POST
    @Path("/detokenize")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response detokenize(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> {
            JsonView body = ServerJson.parse(bodyText);
            if (!(body.raw("tokens") instanceof List<?> list)) {
                throw ServerJson.badRequest("tokens is required and must be an array of token ids");
            }
            String text = list.isEmpty() ? "" : service.decode(OpenAiRequests.tokens(list, service.vocabSize(),
                    "tokens"));
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("content", text);
            out.put("prompt", text);
            return Response.ok(ServerJson.write(out)).build();
        });
    }

    @POST
    @Path("/apply-template")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response applyTemplate(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> {
            ChatRequest chat = OpenAiRequests.chat(ServerJson.parse(bodyText));
            return Response.ok(ServerJson.write(Map.of("prompt", service.formatChat(chat)))).build();
        });
    }

    private Response tokenizeOf(JsonView body) {
        int[] tokens;
        if (body.has("messages")) {
            String rendered = service.formatChat(OpenAiRequests.chat(body));
            tokens = service.encode(rendered, body.optBool("add_special_tokens", false), true);
        } else if (body.has("content")) {
            tokens = service.encode(body.str("content"), body.optBool("add_special", false),
                    body.optBool("parse_special", true));
        } else if (body.has("prompt")) {
            tokens = service.encode(body.str("prompt"), body.optBool("add_special_tokens", true), true);
        } else {
            throw ServerJson.badRequest("tokenize takes content (llama.cpp), prompt or messages (vLLM)");
        }
        boolean pieces = body.optBool("with_pieces", false);
        List<Object> ids = new ArrayList<>(tokens.length);
        for (int id : tokens) {
            if (pieces) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("id", id);
                entry.put("piece", service.tokenText(id));
                ids.add(entry);
            } else {
                ids.add(id);
            }
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokens", ids);
        out.put("count", tokens.length);
        out.put("max_model_len", service.maxModelLength());
        return Response.ok(ServerJson.write(out)).build();
    }
}
