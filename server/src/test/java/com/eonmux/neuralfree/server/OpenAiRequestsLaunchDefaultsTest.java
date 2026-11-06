package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The launch defaults the request fields fall back to ({@code grp_attn_n}, {@code grp_attn_w},
 * {@code n_cache_reuse}): the {@code -D} properties a launch flag set, the llama.cpp defaults when none did,
 * and a refusal naming the property when a launch set garbage.
 */
class OpenAiRequestsLaunchDefaultsTest {

    @AfterEach
    void clearProperties() {
        System.clearProperty(OpenAiRequests.PROP_GRP_ATTN_N);
        System.clearProperty(OpenAiRequests.PROP_GRP_ATTN_W);
        System.clearProperty(OpenAiRequests.PROP_CACHE_REUSE);
    }

    @Test
    void theDefaultsAreLlamaCppsWhenNoLaunchSetAProperty() {
        assertEquals(1, OpenAiRequests.grpAttnNDefault());
        assertEquals(GenerationRequest.DEFAULT_GRP_ATTN_W, OpenAiRequests.grpAttnWDefault());
        assertEquals(0, OpenAiRequests.cacheReuseDefault());
    }

    @Test
    void aLaunchPropertySetsTheDefault() {
        System.setProperty(OpenAiRequests.PROP_GRP_ATTN_N, "4");
        System.setProperty(OpenAiRequests.PROP_GRP_ATTN_W, "512");
        System.setProperty(OpenAiRequests.PROP_CACHE_REUSE, "256");
        assertEquals(4, OpenAiRequests.grpAttnNDefault());
        assertEquals(512, OpenAiRequests.grpAttnWDefault());
        assertEquals(256, OpenAiRequests.cacheReuseDefault());
    }

    @Test
    void aGarbagePropertyRefusesByName() {
        System.setProperty(OpenAiRequests.PROP_GRP_ATTN_N, "many");
        System.setProperty(OpenAiRequests.PROP_CACHE_REUSE, "-4");
        IllegalArgumentException groupSize = assertThrows(IllegalArgumentException.class,
                OpenAiRequests::grpAttnNDefault);
        assertTrue(groupSize.getMessage().contains(OpenAiRequests.PROP_GRP_ATTN_N), groupSize.getMessage());
        IllegalArgumentException reuse = assertThrows(IllegalArgumentException.class,
                OpenAiRequests::cacheReuseDefault);
        assertTrue(reuse.getMessage().contains(OpenAiRequests.PROP_CACHE_REUSE), reuse.getMessage());
    }
}
