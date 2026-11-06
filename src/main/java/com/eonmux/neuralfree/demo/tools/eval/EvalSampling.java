package com.eonmux.neuralfree.demo.tools.eval;

/**
 * The sampling parameters a run passes with every request. A null field is not sent, exactly as the
 * reference omits the flag: the server's own default applies.
 */
public final class EvalSampling {

    private final Double temperature;
    private final Integer topK;
    private final Double topP;
    private final Double minP;
    private final String reasoningEffort;
    private final int nPredict;

    /**
     * @param nPredict the maximum tokens per reply, {@code -1} for no limit (the reference's default)
     */
    public EvalSampling(Double temperature, Integer topK, Double topP, Double minP,
                        String reasoningEffort, int nPredict) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.minP = minP;
        this.reasoningEffort = reasoningEffort;
        this.nPredict = nPredict;
    }

    public Double temperature() {
        return temperature;
    }

    public Integer topK() {
        return topK;
    }

    public Double topP() {
        return topP;
    }

    public Double minP() {
        return minP;
    }

    public String reasoningEffort() {
        return reasoningEffort;
    }

    public int nPredict() {
        return nPredict;
    }

    /** The non-null parameters as the state file records them, in the reference's key order. */
    public java.util.Map<String, Object> asMap() {        java.util.Map<String, Object> out = new java.util.LinkedHashMap<>();
        if (temperature != null) {
            out.put("temperature", temperature);
        }
        if (topK != null) {
            out.put("top_k", topK);
        }
        if (topP != null) {
            out.put("top_p", topP);
        }
        if (minP != null) {
            out.put("min_p", minP);
        }
        if (reasoningEffort != null) {
            out.put("reasoning_effort", reasoningEffort);
        }
        return out;
    }

    /**
     * The sampling of a resumed run: the state file's parameters, with the request's token budget
     * taken from the command line because the state does not record it.
     */
    public static EvalSampling fromMap(java.util.Map<String, Object> config, int nPredict) {
        Double temperature = doubleValue(config.get("temperature"));
        Integer topK = intValue(config.get("top_k"));
        Double topP = doubleValue(config.get("top_p"));
        Double minP = doubleValue(config.get("min_p"));
        Object effort = config.get("reasoning_effort");
        return new EvalSampling(temperature, topK, topP, minP,
                effort instanceof String ? (String) effort : null, nPredict);
    }

    private static Double doubleValue(Object value) {
        return value instanceof Number ? ((Number) value).doubleValue() : null;
    }

    private static Integer intValue(Object value) {
        return value instanceof Number ? ((Number) value).intValue() : null;
    }

    /**
     * The sampling fields of a chat request body. The token budget goes out as llama.cpp's
     * {@code n_predict} and, when bounded, also as the OpenAI-compatible
     * {@code max_completion_tokens}, so the same request is understood by llama-server and by a
     * server that reads only the OpenAI field.
     */
    void applyTo(java.util.Map<String, Object> body) {
        body.put("n_predict", nPredict);
        if (nPredict >= 0) {
            body.put("max_completion_tokens", nPredict);
        }
        if (temperature != null) {
            body.put("temperature", temperature);
        }
        if (topK != null) {
            body.put("top_k", topK);
        }
        if (topP != null) {
            body.put("top_p", topP);
        }
        if (minP != null) {
            body.put("min_p", minP);
        }
        if (reasoningEffort != null) {
            body.put("reasoning_effort", reasoningEffort);
        }
    }
}
