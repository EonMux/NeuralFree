package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfRope}: the inputs ({@code rope_parameters}, per layer type, {@code rope_scaling}), every legacy form
 * vLLM patches, the theta, local theta and partial rotary aliases, and the typed accessors.
 */
public class HfRopeTest {

    private static HfRope rope(String members) {
        String sep = members.isEmpty() ? "" : ",";
        return HfConfigLoader.fromJson(Path.of("."), ("{'architectures':['A']" + sep + members + "}")
                .replace('\'', '"')).rope();
    }

    @Test
    public void noRopeBlockMirrorsTheTopLevelKeys() {
        HfRope r = rope("'rope_theta':500000.0,'partial_rotary_factor':0.5,'original_max_position_embeddings':8192");
        assertEquals(500000.0, r.theta().getAsDouble(), 0);
        assertEquals(0.5, r.global().partialRotaryFactor().getAsDouble(), 0);
        assertEquals(8192, r.global().originalMaxPositionEmbeddings().getAsInt());
        assertFalse(r.global().type().isPresent());
        assertFalse(r.nested());
        assertFalse(rope("").theta().isPresent());
    }

    @Test
    public void thetaAliasesFirstPresentWins() {
        assertEquals(1.0, rope("'global_rope_theta':1.0,'rope_theta':2.0").theta().getAsDouble(), 0);
        assertEquals(3.0, rope("'rope_global_theta':3.0").theta().getAsDouble(), 0);
        assertEquals(4.0, rope("'rope_theta_global':4.0").theta().getAsDouble(), 0);
        assertEquals(5000.0, rope("'rotary_emb_base':5000").theta().getAsDouble(), 0);
        assertEquals(7.0, rope("'rope_parameters':{'rope_theta':7.0},'rope_theta':8.0").theta().getAsDouble(), 0);
    }

    @Test
    public void localThetaAliasesBecomeTheSlidingParameters() {
        for (String alias : HfRope.LOCAL_THETA_ALIASES) {
            HfRope r = rope("'rope_theta':1000000.0,'" + alias + "':10000.0");
            assertEquals(alias, 10000.0, r.localTheta().getAsDouble(), 0);
            assertEquals(1000000.0, r.theta().getAsDouble(), 0);
        }
        assertEquals(5.0, rope("'local_rope_theta':5.0,'swa_rope_theta':6.0").localTheta().getAsDouble(), 0);
        assertFalse(rope("'rope_theta':1.0").localTheta().isPresent());
    }

    @Test
    public void partialRotaryAliases() {
        assertEquals(0.25, rope("'rotary_pct':0.25").global().partialRotaryFactor().getAsDouble(), 0);
        assertEquals(0.5, rope("'rope_pct':0.5").global().partialRotaryFactor().getAsDouble(), 0);
        assertEquals(0.75, rope("'rope_percent':0.75").global().partialRotaryFactor().getAsDouble(), 0);
        assertEquals(0.125, rope("'rotary_emb_fraction':0.125").global().partialRotaryFactor().getAsDouble(), 0);
        assertEquals(1.0, rope("'partial_rotary_factor':1.0,'rotary_pct':0.25").global().partialRotaryFactor()
                .getAsDouble(), 0);
    }

    @Test
    public void legacyTypeBecomesRopeType() {
        HfRope r = rope("'rope_scaling':{'type':'linear','factor':2.0}");
        assertEquals("linear", r.global().type().get());
        assertEquals(2.0, r.global().factor().getAsDouble(), 0);
        assertEquals("dynamic", rope("'rope_scaling':{'type':'dynamic','rope_type':'dynamic'}").global().type().get());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> rope("'rope_scaling':{'type':'linear','rope_type':'yarn'}"));
        assertTrue(e.getMessage(), e.getMessage().contains("conflicting"));
    }

    @Test
    public void suIsLongRope() {
        HfRope r = rope("'rope_scaling':{'type':'su','long_factor':[1.0,2.0],'short_factor':[1,1],"
                + "'original_max_position_embeddings':4096}");
        assertEquals("longrope", r.global().type().get());
        assertEquals(List.of(1.0, 2.0), r.global().longFactor().get());
        assertEquals(List.of(1.0, 1.0), r.global().shortFactor().get());
        assertEquals(4096, r.global().originalMaxPositionEmbeddings().getAsInt());
        assertEquals("longrope", rope("'rope_scaling':{'type':'su','rope_type':'longrope'}").global().type().get());
    }

    @Test
    public void mropeIsDefaultWithSections() {
        HfRope r = rope("'rope_scaling':{'type':'mrope','mrope_section':[16,24,24]}");
        assertEquals("default", r.global().type().get());
        assertEquals(List.of(16, 24, 24), r.mropeSection().get());
        assertEquals("default", rope("'rope_scaling':{'type':'mrope','rope_type':'default','mrope_section':[1]}")
                .global().type().get());
        assertThrows(GgufFormatException.class, () -> rope("'rope_scaling':{'rope_type':'mrope'}"));
        assertEquals(List.of(8, 8), rope("'rope_parameters':{'rope_type':'default','xdrope_section':[8,8]}")
                .mropeSection().get());
        assertThrows(GgufFormatException.class,
                () -> rope("'rope_parameters':{'mrope_section':'16'}").mropeSection());
    }

    @Test
    public void telechat3YarnIsYarnWithItsAttentionFactor() {
        HfRope r = rope("'rope_scaling':{'type':'telechat3-yarn','factor':4.0,'original_max_position_embeddings':8192}");
        assertEquals("yarn", r.global().type().get());
        assertEquals(0.07 * Math.log(4.0) + 1.0, r.global().attentionFactor().getAsDouble(), 1e-12);
        assertFalse(r.global().get("type").isPresent());
        assertThrows(GgufFormatException.class, () -> rope("'rope_scaling':{'rope_type':'telechat3-yarn'}"));
        for (String factor : new String[]{"0.0", "-2.0"}) {
            assertThrows(factor, GgufFormatException.class,
                    () -> rope("'rope_scaling':{'rope_type':'telechat3-yarn','factor':" + factor + "}"));
        }
    }

    @Test
    public void ropeParametersWinOverRopeScaling() {
        HfRope r = rope("'rope_parameters':{'rope_type':'yarn','factor':8.0},'rope_scaling':{'rope_type':'linear'}");
        assertEquals("yarn", r.global().type().get());
    }

    @Test
    public void nestedParametersPerLayerType() {
        HfRope r = rope("'rope_parameters':{'full_attention':{'rope_type':'yarn','rope_theta':1000000.0,"
                + "'factor':8.0},'sliding_attention':{'type':'default','rope_theta':10000.0}},'rope_theta':5.0");
        assertTrue(r.nested());
        assertEquals("yarn", r.global().type().get());
        assertEquals(1000000.0, r.theta().getAsDouble(), 0);
        assertEquals(10000.0, r.localTheta().getAsDouble(), 0);
        assertEquals("default", r.sliding().get().type().get());
        assertFalse(r.layer("chunked_attention").isPresent());
        assertThrows(GgufFormatException.class,
                () -> rope("'rope_parameters':{'full_attention':{'type':'mrope'}}"));
        assertThrows(GgufFormatException.class, () -> rope("'rope_parameters':{'full_attention':3}"));
        assertThrows(GgufFormatException.class, () -> rope("'rope_parameters':[1]"));
    }

    @Test
    public void otherLayerTypeNestingSurvivesTheMirroredAliases() {
        HfRope r = rope("'rope_parameters':{'main':{'rope_type':'yarn','factor':4.0},'compress':{'type':'default'}},"
                + "'rope_theta':5.0");
        assertTrue(r.nested());
        assertEquals("default", r.layer("compress").get().type().get());
        assertEquals("yarn", r.layer("main").get().type().get());
        HfRope dsa = rope("'rope_parameters':{'deepseek_sparse_attention':{'type':'su'}}");
        assertTrue(dsa.nested());
        assertEquals("longrope", dsa.layer("deepseek_sparse_attention").get().type().get());
    }

    @Test
    public void yarnLlama3DeepSeekAndDynamicFields() {
        HfRope.Params yarn = rope("'rope_scaling':{'rope_type':'yarn','factor':40,'beta_fast':32,'beta_slow':1,"
                + "'attn_factor':0.5,'extrapolation_factor':1.0,'truncate':false,'mscale':1.0,"
                + "'mscale_all_dim':0.707,'original_max_position_embeddings':4096}").global();
        assertEquals(40.0, yarn.factor().getAsDouble(), 0);
        assertEquals(32.0, yarn.betaFast().getAsDouble(), 0);
        assertEquals(1.0, yarn.betaSlow().getAsDouble(), 0);
        assertEquals(0.5, yarn.attentionFactor().getAsDouble(), 0);
        assertEquals(1.0, yarn.extrapolationFactor().getAsDouble(), 0);
        assertEquals(Boolean.FALSE, yarn.truncate().get());
        assertEquals(1.0, yarn.mscale().getAsDouble(), 0);
        assertEquals(0.707, yarn.mscaleAllDim().getAsDouble(), 0);
        assertEquals(4096, yarn.originalMaxPositionEmbeddings().getAsInt());

        HfRope.Params llama3 = rope("'rope_scaling':{'rope_type':'llama3','factor':8.0,'low_freq_factor':1.0,"
                + "'high_freq_factor':4.0,'original_max_position_embeddings':8192}").global();
        assertEquals(1.0, llama3.lowFreqFactor().getAsDouble(), 0);
        assertEquals(4.0, llama3.highFreqFactor().getAsDouble(), 0);

        HfRope.Params dynamic = rope("'rope_scaling':{'rope_type':'dynamic','alpha':1000.0}").global();
        assertEquals(1000.0, dynamic.alpha().getAsDouble(), 0);

        assertThrows(GgufFormatException.class,
                () -> rope("'rope_scaling':{'rope_type':'yarn','factor':'x'}").global().factor());
        assertThrows(GgufFormatException.class,
                () -> rope("'rope_scaling':{'rope_type':'yarn','truncate':1}").global().truncate());
    }

    @Test
    public void textConfigRopeWinsThroughFlattening() {
        HfRope r = rope("'rope_theta':1.0,'text_config':{'rope_theta':2.0,'rope_scaling':{'rope_type':'linear',"
                + "'factor':3.0}}");
        assertEquals(2.0, r.theta().getAsDouble(), 0);
        assertEquals("linear", r.global().type().get());
    }
}
