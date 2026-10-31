package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** The named transforms of the HK-D1 part E hooks that run without a load: the OpenELM divisible FFN widths and
 *  the GraniteSwitch derived weight-layer block count. The per-hook transforms run through the sources tests. */
public class HfDenseClassicHooksETest {

    @Test
    public void openelmFfnWidthRoundToDivisibleNeverRoundsDownPastNinePercent() {
        assertEquals(4, HfDenseClassicHooksE.makeDivisible(2.0, 4)); // max(divisor, round(v/div)*div) = 4
        assertEquals(4, HfDenseClassicHooksE.makeDivisible(3.7, 4));
        assertEquals(20, HfDenseClassicHooksE.makeDivisible(12.0, 10)); // 10 < 0.9 * 12 -> +divisor
        assertEquals(10, HfDenseClassicHooksE.makeDivisible(10.0, 10));
    }

    @Test
    public void graniteSwitchBlockCountIsTheHighestWeightLayerPlusOne() {
        HfLoadContext ctx = null; // unused on the path under test
        assertEquals(2, HfDenseClassicHooksE.graniteSwitchWeightLayers(Arrays.asList(
                "model.embed_tokens.weight", "model.layers.0.self_attn.qkv_proj.base_layer.weight",
                "model.switch.0.bias", "model.layers.1.self_attn.o_proj.base_layer.weight",
                "control_to_substitute_lut"), ctx));
        assertThrows(GgufFormatException.class, () ->
                HfDenseClassicHooksE.graniteSwitchWeightLayers(Arrays.asList("model.norm.weight"), ctx));
    }
}
