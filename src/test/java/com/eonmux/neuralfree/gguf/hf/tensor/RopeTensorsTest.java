package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

import static com.eonmux.neuralfree.gguf.hf.tensor.RowTransformsTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;

/**
 * {@link RopeTensors}: llama3 {@code rope_freqs} against values worked out by hand from llama.py:299-327, the Step35
 * padding, Phi-3 LongRoPE factor validation, Gemma 4 proportional factors and the in-memory constants.
 */
public class RopeTensorsTest {

    private static HfConfig config(String members) {
        return HfConfigLoader.fromJson(Path.of("."), ("{'architectures':['A']," + members + "}").replace('\'', '"'));
    }

    /**
     * head_dim 8, theta 500000, factor 8, low 1, high 4, original context 8192: the frequencies are
     * {@code 500000^(-i/4)}, with wavelengths 6.28, 167.1, 4442.9 and 118142.8 against the bounds 2048 and 8192, so
     * the factors are 1, 1, {@code 1 / ((1 - s) / 8 + s)} with {@code s = (8192 / 4442.88 - 1) / 3}, and 8.
     */
    @Test
    public void llama3FreqsMatchHandValues() throws IOException {
        HfConfig cfg = config("'hidden_size':64,'num_attention_heads':8,'rope_theta':500000.0,"
                + "'rope_scaling':{'rope_type':'llama3','factor':8.0,'low_freq_factor':1.0,'high_freq_factor':4.0,"
                + "'original_max_position_embeddings':8192}");
        Optional<RowSource> f = RopeTensors.llama3Freqs(cfg);
        double s = (8192 / 4442.8833 - 1) / 3;
        float mid = (float) (1 / ((1 - s) / 8 + s));
        assertEquals(2.69453f, mid, 1e-5f);
        assertArrayEquals(new float[]{1f, 1f, mid, 8f}, values(f.get()), 1e-5f);
        assertEquals(ElementFormat.of(GGMLType.F32), f.get().format());
        assertFalse(RopeTensors.llama3Freqs(config("'hidden_size':64,'num_attention_heads':8")).isPresent());
    }

    @Test
    public void step35PadsWithOnes() throws IOException {
        RowSource f = RopeTensors.llama3Freqs(500000.0, 4, 4, 8.0, 1.0, 4.0, 8192, true);
        float[] v = values(f);
        assertEquals(4, v.length);
        assertEquals(1f, v[0], 0f);
        assertEquals(2.69453f, v[1], 1e-5f);
        assertEquals(1f, v[2], 0f);
        assertEquals(1f, v[3], 0f);
        assertThrows(IllegalArgumentException.class, () -> RopeTensors.llama3Freqs(1e4, 3, 0, 8, 1, 4, 8192, false));
    }

    @Test
    public void longropeFactorsAreValidated() throws IOException {
        String base = "'hidden_size':64,'num_attention_heads':4,'max_position_embeddings':131072,";
        String eight = "[1,2,3,4,5,6,7,8]";
        HfConfig ok = config(base + "'rope_scaling':{'type':'longrope','long_factor':" + eight
                + ",'short_factor':[1,1,1,1,1,1,1,1],'original_max_position_embeddings':4096}");
        assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6, 7, 8},
                values(RopeTensors.longropeFactors(ok, RopeTensors.Factors.LONG).get()), 0f);
        assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1, 1, 1},
                values(RopeTensors.longropeFactors(ok, RopeTensors.Factors.SHORT).get()), 0f);
        HfConfig shortList = config(base + "'rope_scaling':{'type':'longrope','long_factor':[1,2],"
                + "'short_factor':[1,2],'original_max_position_embeddings':4096}");
        assertThrows(GgufFormatException.class,
                () -> RopeTensors.longropeFactors(shortList, RopeTensors.Factors.LONG));
        HfConfig linear = config(base + "'rope_scaling':{'type':'linear','factor':2.0,'long_factor':" + eight
                + ",'short_factor':" + eight + "}");
        assertThrows(GgufFormatException.class, () -> RopeTensors.longropeFactors(linear, RopeTensors.Factors.LONG));
        assertFalse(RopeTensors.longropeFactors(config(base.substring(0, base.length() - 1)),
                RopeTensors.Factors.LONG).isPresent());
    }

    @Test
    public void proportionalFreqsMaskTheUnrotatedPairs() throws IOException {
        assertArrayEquals(new float[]{1f, 1e30f, 1e30f, 1e30f}, values(RopeTensors.proportionalFreqs(8, 0.25)), 0f);
        assertArrayEquals(new float[]{1f, 1f}, values(RopeTensors.proportionalFreqs(4, 1.0)), 0f);
    }

    @Test
    public void constantsHoldTheirValues() throws IOException {
        assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f, 0.5f},
                values(RopeTensors.constant(new long[]{2, 2}, 0.5f)), 0f);
        assertArrayEquals(new float[]{1, 2, 3}, values(RopeTensors.fromFloats(new long[]{3}, new float[]{1, 2, 3})),
                0f);
        assertArrayEquals(new float[]{-1, 7}, values(RopeTensors.fromInts(new long[]{2}, new int[]{-1, 7})), 0f);
        assertEquals(ElementFormat.of(GGMLType.I64), RopeTensors.fromLongs(new long[]{1}, new long[]{3}).format());
        assertThrows(IllegalArgumentException.class, () -> RopeTensors.fromFloats(new long[]{2}, new float[3]));
        assertThrows(IllegalArgumentException.class,
                () -> RopeTensors.fromBytes(ElementFormat.of(GGMLType.Q8_0), new long[]{1, 32}, new byte[33]));
    }
}
