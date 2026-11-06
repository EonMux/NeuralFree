package com.eonmux.neuralfree.gguf.chat.autoparser;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The differential autoparser's entry point: {@link #analyze} returns the structure of a chat template, or
 * {@code null} when it does not parse. The analysis is derived once per template source and cached, so callers may
 * ask per request; a failure is cached too, so an unknown template costs one attempt. Immutable; the cache is
 * synchronized and bounded.
 */
public final class AutoParser {

    private static final Logger LOGGER = LogManager.getLogger(AutoParser.class);

    /** Distinct templates held; a serving process sees very few. */
    private static final int CACHE_MAX = 64;

    private static final Map<String, Object> CACHE = Collections.synchronizedMap(
            new LinkedHashMap<String, Object>(16, 0.75f, true) {
                @Override protected boolean removeEldestEntry(Map.Entry<String, Object> eldest) {
                    return size() > CACHE_MAX;
                }
            });

    /** Marks a template whose analysis failed, so it is not retried. */
    private static final Object FAILED = new Object();

    private AutoParser() {
    }

    /**
     * The analysis of {@code template}, or null when it does not parse or its probe renders all fail. The result
     * is the same instance for the same source.
     */
    public static AutoParserAnalysis analyze(String template) {
        if (template == null || template.isEmpty()) {
            return null;
        }
        Object cached = CACHE.get(template);
        if (cached == null) {
            cached = compute(template);
            CACHE.put(template, cached == null ? FAILED : cached);
        }
        return cached == FAILED ? null : (AutoParserAnalysis) cached;
    }

    private static AutoParserAnalysis compute(String template) {
        try {
            AutoParserAnalysis analysis = AutoParserAnalyzer.analyze(template);
            LOGGER.debug("Autoparser analysis of a {}-char template: {}", template.length(), analysis);
            return analysis;
        } catch (RuntimeException e) {
            LOGGER.debug("Autoparser analysis failed: {}", e.getMessage());
            return null;
        }
    }
}
