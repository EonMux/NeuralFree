package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.SlotResult;
import com.eonmux.neuralfree.gguf.engine.batch.SlotStatus;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The llama.cpp slot endpoints over the engine's sequence slots:
 *
 * <ul>
 *   <li>{@code GET /slots} lists the slots (llama.cpp's slot status): id, context, the speculative and
 *       {@code is_processing} flags, and for the slot's running or last request the {@code id_task}, prompt
 *       counters, sampler {@code params}, {@code next_token} and the prompt and generated text.</li>
 *   <li>{@code POST /slots/{id}?action=save|restore} moves a slot file under the {@code --slot-save-path}
 *       directory ({@code neuralfree.server.slot.save.path}), body {@code {"filename": "..."}}; the answer
 *       carries llama.cpp's {@code n_saved} / {@code n_restored} tokens, {@code n_written} / {@code n_read} bytes
 *       and timings.</li>
 *   <li>{@code POST /slots/{id}?action=erase} forgets the slot's state, answering {@code n_erased}.</li>
 * </ul>
 *
 * <p>A slot's state is the prefix-cache chain its last request left in the batch engine, whichever backend holds
 * it ({@code BatchEngine.saveSlot}); a restored slot serves the next request whose prompt starts with its tokens.
 * Without the directory every action answers llama.cpp's not-supported refusal. An engine whose memory cannot move
 * its blocks refuses by name (501), an unusable slot file answers llama.cpp's {@code Unable to restore slot} (400),
 * and a failed write a 500.
 *
 * <p>In router mode the routes follow llama.cpp's router proxy: {@code ?model=} (with {@code ?autoload=} to
 * override the server's) selects the entry whose slots they read or change; a missing name is a 400 and an
 * unknown one a 404.</p>
 */
@Path("/slots")
public class SlotsResource {

    /** llama.cpp's refusal when the server runs without {@code --slot-save-path} (server-context.cpp:5836). */
    static final String DISABLED = "This server does not support slots action. Start it with `--slot-save-path`";

    @Inject
    EngineService service;

    /** The multi-model router; null in manual wiring, where requests serve on {@code service} directly. */
    @Inject
    ModelRouter router;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public SlotsResource(EngineService service) {
        this(service, null);
    }

    /** Test and manual wiring over an explicit service and router. */
    public SlotsResource(EngineService service, ModelRouter router) {
        this.service = service;
        this.router = router;
    }

    public SlotsResource() {
    }

    /** The resource over the entry {@code model} names, answering the query's autoload override. */
    private SlotsResource routed(String model, String autoload) {
        if (router == null || !router.active()) {
            return this;
        }
        return new SlotsResource(ModelRouter.targetQuery(router, service, model, autoload), router);
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public Response slots(@QueryParam("fail_on_no_slot") String failOnNoSlot,
                          @QueryParam("model") String model, @QueryParam("autoload") String autoload) {
        return routed(model, autoload).slotsOf(failOnNoSlot);
    }

    /** Test and manual wiring: the list of the wired service, without a model query. */
    Response slots(String failOnNoSlot) {
        return slotsOf(failOnNoSlot);
    }

    private Response slotsOf(String failOnNoSlot) {
        ServerJson.requireLoaded(service);
        List<SlotStatus> statuses;
        try {
            statuses = service.slotStatuses();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw ServerJson.error(500, "the slot status read was interrupted", "server_error");
        }
        if (failOnNoSlot != null && statuses.stream().allMatch(SlotStatus::processing)) {
            throw ServerJson.error(503, "no slot available", "unavailable_error");
        }
        List<Object> out = new ArrayList<>();
        for (SlotStatus slot : statuses) {
            out.add(view(slot));
        }
        return Response.ok(ServerJson.write(out)).build();
    }

    /** One slot's status as llama.cpp's list reports it: the bare fields, then the task's when one exists. */
    private Map<String, Object> view(SlotStatus slot) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", slot.id());
        out.put("n_ctx", slot.nCtx());
        out.put("speculative", slot.speculative());
        out.put("is_processing", slot.processing());
        if (slot.idTask() < 0) {
            return out;
        }
        out.put("id_task", slot.idTask());
        out.put("n_prompt_tokens", slot.nPromptTokens());
        out.put("n_prompt_tokens_processed", slot.nPromptTokensProcessed());
        out.put("n_prompt_tokens_cache", slot.nPromptTokensCache());
        out.put("params", params(slot));
        Map<String, Object> next = new LinkedHashMap<>();
        next.put("has_next_token", slot.hasNextToken());
        next.put("n_remain", slot.nRemain());
        next.put("n_decoded", slot.nDecoded());
        out.put("next_token", List.of(next));
        out.put("prompt", service.tokenizer().decode(slot.promptTokens()));
        out.put("generated", service.tokenizer().decode(slot.generatedTokens()));
        return out;
    }

    /** The task's sampler parameters under llama.cpp's names ({@code task_params::to_json}). */
    private static Map<String, Object> params(SlotStatus slot) {
        SamplerConfig s = slot.params();
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("seed", s.seed());
        out.put("temperature", (double) s.temperature());
        out.put("dynatemp_range", (double) s.dynatempRange());
        out.put("dynatemp_exponent", (double) s.dynatempExponent());
        out.put("top_k", s.topK());
        out.put("top_p", (double) s.topP());
        out.put("min_p", (double) s.minP());
        out.put("top_n_sigma", (double) s.topNSigma());
        out.put("xtc_probability", (double) s.xtcProbability());
        out.put("xtc_threshold", (double) s.xtcThreshold());
        out.put("typical_p", (double) s.typicalP());
        out.put("repeat_last_n", s.penaltyLastN());
        out.put("repeat_penalty", (double) s.repeatPenalty());
        out.put("presence_penalty", (double) s.presencePenalty());
        out.put("frequency_penalty", (double) s.frequencyPenalty());
        out.put("dry_multiplier", (double) s.dryMultiplier());
        out.put("dry_base", (double) s.dryBase());
        out.put("dry_allowed_length", s.dryAllowedLength());
        out.put("dry_penalty_last_n", s.dryPenaltyLastN());
        out.put("mirostat", s.mirostat());
        out.put("mirostat_tau", (double) s.mirostatTau());
        out.put("mirostat_eta", (double) s.mirostatEta());
        out.put("adaptive_target", (double) s.adaptiveTarget());
        out.put("adaptive_decay", (double) s.adaptiveDecay());
        out.put("max_tokens", slot.maxNewTokens());
        out.put("n_predict", slot.maxNewTokens());
        out.put("min_keep", s.minKeep());
        out.put("samplers", s.order().stream().map(StageKind::label).toList());
        return out;
    }

    @POST
    @Path("/{idSlot}")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response action(@PathParam("idSlot") int idSlot, @QueryParam("action") String action,
                           @QueryParam("model") String model, @QueryParam("autoload") String autoload,
                           String bodyText) {
        return routed(model, autoload).actionOf(idSlot, action, bodyText);
    }

    /** Test and manual wiring: one action on the wired service, without a model query. */
    Response action(int idSlot, String action, String bodyText) {
        return actionOf(idSlot, action, bodyText);
    }

    private Response actionOf(int idSlot, String action, String bodyText) {
        java.nio.file.Path savePath = service.options().slotSavePath();
        if (savePath == null) {
            throw ServerJson.error(501, DISABLED, "not_supported_error");
        }
        ServerJson.requireLoaded(service);
        if (idSlot < 0) {
            throw ServerJson.error(400, "Invalid slot ID", "invalid_request_error");
        }
        return switch (action == null ? "" : action) {
            case "save" -> save(idSlot, savePath, bodyText);
            case "restore" -> restore(idSlot, savePath, bodyText);
            case "erase" -> erase(idSlot);
            default -> throw ServerJson.error(400, "Invalid action", "invalid_request_error");
        };
    }

    /** Writes the slot's file, answering llama.cpp's save result. */
    private Response save(int idSlot, java.nio.file.Path savePath, String bodyText) {
        String name = filename(bodyText);
        long start = System.nanoTime();
        SlotResult saved = run(() -> service.saveSlot(idSlot, savePath.resolve(name)), false);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id_slot", idSlot);
        out.put("filename", name);
        out.put("n_saved", saved.tokens());
        out.put("n_written", saved.bytes());
        out.put("timings", Map.of("save_ms", (System.nanoTime() - start) / 1_000_000.0));
        return Response.ok(ServerJson.write(out)).build();
    }

    /** Reads a slot file into the slot, answering llama.cpp's restore result. */
    private Response restore(int idSlot, java.nio.file.Path savePath, String bodyText) {
        String name = filename(bodyText);
        long start = System.nanoTime();
        SlotResult restored = run(() -> service.restoreSlot(idSlot, savePath.resolve(name)), true);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id_slot", idSlot);
        out.put("filename", name);
        out.put("n_restored", restored.tokens());
        out.put("n_read", restored.bytes());
        out.put("timings", Map.of("restore_ms", (System.nanoTime() - start) / 1_000_000.0));
        return Response.ok(ServerJson.write(out)).build();
    }

    /** Erases the slot's state, answering llama.cpp's erase result. */
    private Response erase(int idSlot) {
        int erased = run(() -> service.eraseSlot(idSlot), false);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id_slot", idSlot);
        out.put("n_erased", erased);
        return Response.ok(ServerJson.write(out)).build();
    }

    /** The request's filename (llama.cpp's {@code fs_validate_filename} rules), refused by name. */
    private static String filename(String bodyText) {
        JsonView body = ServerJson.parse(bodyText);
        Object raw = body.raw("filename");
        if (raw instanceof String name && validFilename(name)) {
            return name;
        }
        throw ServerJson.error(400, "Invalid filename", "invalid_request_error");
    }

    /**
     * Whether {@code name} is a safe file name: at most 255 UTF-8 bytes, no path separators, no control
     * characters or their full-width and mathematical look-alikes, no leading or trailing space, no trailing
     * dot and no {@code ..} (common/common.cpp fs_validate_filename).
     */
    static boolean validFilename(String name) {
        if (name.isEmpty() || name.getBytes(StandardCharsets.UTF_8).length > 255) {
            return false;
        }
        for (int i = 0; i < name.length(); ) {
            int c = name.codePointAt(i);
            if (c <= 0x1F || (c >= 0x7F && c <= 0x9F) || (c >= 0xD800 && c <= 0xDFFF) || c == 0xFF0E
                    || c == 0x2215 || c == 0x2216 || c == 0xFFFD || c == 0xFEFF
                    || c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"'
                    || c == '<' || c == '>' || c == '|') {
                return false;
            }
            i += Character.charCount(c);
        }
        if (name.charAt(0) == ' ' || name.charAt(name.length() - 1) == ' '
                || name.charAt(name.length() - 1) == '.') {
            return false;
        }
        return !name.contains("..");
    }

    /**
     * Runs one slot action as llama.cpp answers it: a slot id outside the slots a 400 {@code Invalid slot ID}, a scope
     * refusal a 501, a failed restore ({@code restore}) a 400 {@code Unable to restore slot}, any other failure a
     * 500.
     */
    private static <T> T run(SlotCall<T> call, boolean restore) {
        try {
            return call.run();
        } catch (EngineService.UnknownSlotException e) {
            throw ServerJson.error(400, "Invalid slot ID", "invalid_request_error");
        } catch (UnsupportedOperationException e) {
            throw ServerJson.error(501, e.getMessage(), "not_supported_error");
        } catch (IOException | UncheckedIOException | IllegalStateException e) {
            if (restore) {
                throw ServerJson.error(400, "Unable to restore slot: " + e.getMessage(), "invalid_request_error");
            }
            throw ServerJson.error(500, e.getMessage(), "server_error");
        }
    }

    /** One slot action over the engine. */
    @FunctionalInterface
    private interface SlotCall<T> {
        T run() throws IOException;
    }
}
