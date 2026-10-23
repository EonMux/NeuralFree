package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * {@link DflashSpec#metaFlag}: llama.cpp {@code llama_model_meta_val_str(key) == "true"}, which formats a value by
 * its stored type (src/llama-impl.cpp {@code gguf_kv_to_str}).
 */
public class DflashSpecMetaFlagTest {

    private static final String KEY = "dflash.flag";

    private static GgufKv kv(Object value) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "dflash");
        if (value != null) {
            m.put(KEY, value);
        }
        return GgufKv.ofMap(m);
    }

    @Test
    public void absentKeyYieldsTheDefault() {
        assertTrue(DflashSpec.metaFlag(kv(null), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv(null), KEY, false));
    }

    @Test
    public void boolPrintsAsTrueOrFalse() {
        assertTrue(DflashSpec.metaFlag(kv(Boolean.TRUE), KEY, false));
        assertFalse(DflashSpec.metaFlag(kv(Boolean.FALSE), KEY, true));
    }

    @Test
    public void onlyTheExactStringTrueIsTrue() {
        assertTrue(DflashSpec.metaFlag(kv("true"), KEY, false));
        assertFalse(DflashSpec.metaFlag(kv("TRUE"), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv("1"), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv("false"), KEY, true));
    }

    @Test
    public void numbersAndArraysNeverPrintAsTrue() {
        assertFalse(DflashSpec.metaFlag(kv(1), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv(1L), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv(1.0f), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv(Arrays.asList(Boolean.TRUE)), KEY, true));
        assertFalse(DflashSpec.metaFlag(kv(Arrays.asList("true")), KEY, true));
    }
}
