package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The pooling endpoints. {@code POST /v1/embeddings} returns OpenAI's body, one pooled vector per input (text, or
 * token ids) from an embedding or pooling model, shaped by {@link EmbeddingOptions}; a model without pooling
 * refuses, as llama.cpp's does. {@code POST /embeddings} and {@code /embedding} are llama.cpp's native form: an
 * array of {@code {"index", "embedding"}} whose embedding is {@code [vector]} for a pooled model and one unnormalized
 * row per token without pooling, over llama.cpp's prompt forms ({@code input} or {@code content}).
 * {@code POST /v1/rerank} (also {@code /rerank}, {@code /v1/reranking}, {@code /reranking}) scores documents
 * against a query with a rank model. A generative model refuses by name.
 */
@Path("/")
public class PoolingResource {

    @Inject
    EngineService service;

    /** Test and manual wiring: builds the resource over an explicit service. */
    public PoolingResource(EngineService service) {
        this.service = service;
    }

    public PoolingResource() {
    }

    @POST
    @Path("/v1/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response embeddings(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> embeddingsOf(ServerJson.parse(bodyText)));
    }

    /** llama.cpp's native embeddings. */
    @POST
    @Path("/embeddings")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response nativeEmbeddings(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> nativeEmbeddingsOf(ServerJson.parse(bodyText)));
    }

    /** llama.cpp's legacy spelling of the native embeddings. */
    @POST
    @Path("/embedding")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response nativeEmbedding(String bodyText) {
        return nativeEmbeddings(bodyText);
    }

    @POST
    @Path("/v1/rerank")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerank(String bodyText) {
        ServerJson.requireLoaded(service);
        return OpenAiResource.mapped(() -> rerankOf(ServerJson.parse(bodyText)));
    }

    @POST
    @Path("/rerank")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankAlias(String bodyText) {
        return rerank(bodyText);
    }

    @POST
    @Path("/v1/reranking")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankingAlias(String bodyText) {
        return rerank(bodyText);
    }

    @POST
    @Path("/reranking")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response rerankingShortAlias(String bodyText) {
        return rerank(bodyText);
    }

    private Response embeddingsOf(JsonView body) {
        String model = service.served().resolve(body);
        if (service.perTokenEmbeddings()) {
            throw ServerJson.badRequest("Pooling type 'none' is not OAI compatible. Please use a different pooling"
                    + " type, or the native /embeddings endpoint");
        }
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength());
        List<int[]> inputs = embeddingInputs(body, options);
        List<Map<String, Object>> data = new ArrayList<>();
        long tokens = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int[] ids = inputs.get(i);
            tokens += ids.length;
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("object", "embedding");
            entry.put("index", i);
            entry.put("embedding", options.encoded(options.shaped(pooled(ids))));
            data.add(entry);
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", tokens);
        usage.put("total_tokens", tokens);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("object", "list");
        out.put("data", data);
        out.put("model", model);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    /** llama.cpp's native body: per input {@code index} and {@code embedding} as a list of vectors. */
    private Response nativeEmbeddingsOf(JsonView body) {
        Object raw = body.has("input") ? body.raw("input") : body.raw("content");
        if (raw == null) {
            throw ServerJson.badRequest("\"input\" or \"content\" must be provided");
        }
        EmbeddingOptions options = EmbeddingOptions.of(body, service.maxModelLength());
        boolean perToken = service.perTokenEmbeddings();
        List<Object> out = new ArrayList<>();
        List<int[]> prompts = LlamaPrompts.of(raw, service);
        for (int i = 0; i < prompts.size(); i++) {
            int[] ids = options.truncated(prompts.get(i));
            List<Object> vectors = new ArrayList<>();
            if (perToken) {
                for (float[] row : tokenRows(ids)) {
                    vectors.add(EmbeddingOptions.values(row));
                }
            } else {
                vectors.add(EmbeddingOptions.values(options.shaped(pooled(ids))));
            }
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", i);
            entry.put("embedding", vectors);
            out.add(entry);
        }
        return Response.ok(ServerJson.write(out)).build();
    }

    /**
     * {@code input}: a string, strings, token ids or token-id arrays (OpenAI's four forms). Text gets the
     * vocabulary's special tokens unless {@code add_special_tokens} is false, and special-token text parses, as
     * llama.cpp and vLLM tokenize it.
     */
    private List<int[]> embeddingInputs(JsonView body, EmbeddingOptions options) {
        Object raw = body.raw("input");
        List<int[]> out = new ArrayList<>();
        if (raw instanceof String s) {
            out.add(text(s, options));
        } else if (raw instanceof List<?> list && !list.isEmpty()) {
            if (list.get(0) instanceof Number) {
                out.add(options.truncated(OpenAiRequests.tokens(list, service.vocabSize(), "input")));
            } else {
                for (int i = 0; i < list.size(); i++) {
                    Object item = list.get(i);
                    if (item instanceof String s) {
                        out.add(text(s, options));
                    } else if (item instanceof List<?> ids && !ids.isEmpty()) {
                        out.add(options.truncated(OpenAiRequests.tokens(ids, service.vocabSize(),
                                "input[" + i + "]")));
                    } else {
                        throw ServerJson.badRequest("input[" + i + "] must be a string or a non-empty array of"
                                + " token ids");
                    }
                }
            }
        } else {
            throw ServerJson.badRequest("input is required: a string, an array of strings, an array of token ids"
                    + " or an array of token-id arrays");
        }
        for (int[] ids : out) {
            if (ids.length == 0) {
                throw ServerJson.badRequest("Input content cannot be empty");
            }
        }
        return out;
    }

    private int[] text(String s, EmbeddingOptions options) {
        return options.truncated(service.encode(s, options.addSpecial(), true));
    }

    private float[][] tokenRows(int[] ids) {
        try {
            return service.embedTokens(ids);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    private float[] pooled(int[] ids) {
        try {
            return service.embed(ids);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the embedding forward failed: " + e.getMessage(), "server_error");
        }
    }

    private Response rerankOf(JsonView body) {
        String model = service.served().resolve(body);
        String query = body.optStr("query", null);
        if (query == null) {
            throw ServerJson.badRequest("query is required");
        }
        if (!(body.raw("documents") instanceof List<?> list)) {
            throw ServerJson.badRequest("documents is required and must be an array");
        }
        List<String> documents = new ArrayList<>();
        for (Object o : list) {
            if (o instanceof String s) {
                documents.add(s);
            } else if (o instanceof Map<?, ?> map && map.get("text") instanceof String s) {
                documents.add(s);
            } else {
                throw ServerJson.badRequest("every document must be a string or an object with a text field");
            }
        }
        int topN = body.optI32("top_n", documents.size());
        boolean returnText = body.optBool("return_documents", false);
        List<float[]> scored = new ArrayList<>();
        long tokens = 0;
        for (int i = 0; i < documents.size(); i++) {
            int[] input = service.rerankInput(query, documents.get(i));
            tokens += input.length;
            scored.add(new float[]{score(input), i});
        }
        scored.sort((a, b) -> Float.compare(b[0], a[0]));
        List<Map<String, Object>> results = new ArrayList<>();
        for (int r = 0; r < Math.min(Math.max(0, topN), scored.size()); r++) {
            int index = (int) scored.get(r)[1];
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("index", index);
            result.put("relevance_score", (double) scored.get(r)[0]);
            if (returnText) {
                result.put("document", Map.of("text", documents.get(index)));
            }
            results.add(result);
        }
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("total_tokens", tokens);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", "rerank-" + System.nanoTime());
        out.put("model", model);
        out.put("results", results);
        out.put("usage", usage);
        return Response.ok(ServerJson.write(out)).build();
    }

    private float score(int[] input) {
        try {
            return service.rerankScore(input);
        } catch (UnsupportedOperationException e) {
            throw ServerJson.badRequest(e.getMessage());
        } catch (IOException e) {
            throw ServerJson.error(500, "the rerank forward failed: " + e.getMessage(), "server_error");
        }
    }
}
