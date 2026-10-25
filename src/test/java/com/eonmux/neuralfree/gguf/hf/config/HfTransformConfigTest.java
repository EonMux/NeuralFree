package com.eonmux.neuralfree.gguf.hf.config;

import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig.Apply;
import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig.Location;
import com.eonmux.neuralfree.gguf.hf.config.HfTransformConfig.Scheme;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.utils.json.Json;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link HfTransformConfig}: compressed-tensors transform schemes at every location, parsed strictly. */
public class HfTransformConfigTest {

    private static HfTransformConfig parse(String json) {
        return HfTransformConfig.parse(Json.parse(HfConfigLoaderTest.json(json), Json.Limits.LENIENT),
                "config.json: quantization_config.transform_config");
    }

    @Test
    public void spinQuantStyleSchemesAtEveryLocation() {
        HfTransformConfig t = parse("{'config_groups':{"
                + "'u':{'type':'hadamard','head_dim':128,'apply':["
                + "{'targets':['re:.*v_proj$'],'location':'weight_output'},"
                + "{'targets':['re:.*o_proj$'],'location':'weight_input','inverse':true}]},"
                + "'r4':{'type':'random-hadamard','randomize':true,'precision':'float64','requires_grad':false,"
                + "'apply':[{'targets':'re:.*down_proj$','location':'input','ignore':['lm_head']},"
                + "{'targets':['re:.*down_proj$'],'location':'weight_input','inverse':true}]},"
                + "'out':{'type':'matrix','apply':[{'targets':['Linear'],'location':'output'}]},"
                + "'kv':{'type':'random-matrix','apply':[{'targets':['LlamaAttention'],'location':'k_cache'},"
                + "{'targets':['LlamaAttention'],'location':'q_attn'}]}}}");
        assertFalse(t.isEmpty());
        List<Scheme> schemes = t.schemes();
        assertEquals(4, schemes.size());

        Scheme u = schemes.get(0);
        assertEquals("u", u.name());
        assertEquals("hadamard", u.type());
        assertEquals(128, u.headDim().getAsInt());
        assertFalse(u.randomize());
        assertEquals(Location.WEIGHT_OUTPUT, u.applies().get(0).location());
        Apply inverse = u.applies().get(1);
        assertEquals(Location.WEIGHT_INPUT, inverse.location());
        assertTrue(inverse.inverse());
        assertFalse(inverse.location().isOnline());

        Scheme r4 = schemes.get(1);
        assertTrue(r4.randomize());
        assertEquals("float64", r4.precision());
        assertFalse(r4.headDim().isPresent());
        Apply input = r4.applies().get(0);
        assertEquals(Location.INPUT, input.location());
        assertEquals(List.of("re:.*down_proj$"), input.targets());
        assertEquals(List.of("lm_head"), input.ignore());
        assertTrue(input.location().isOnline());

        assertEquals(Location.OUTPUT, schemes.get(2).applies().get(0).location());
        assertEquals(Location.K_CACHE, schemes.get(3).applies().get(0).location());
        assertEquals(Location.Q_ATTN, schemes.get(3).applies().get(1).location());
    }

    @Test
    public void emptyConfigGroups() {
        assertTrue(parse("{'config_groups':{}}").isEmpty());
        assertTrue(parse("{'config_groups':{'s':{'type':'hadamard'}}}").schemes().get(0).applies().isEmpty());
    }

    @Test
    public void strictParsing() {
        assertError("[]", "must be an object");
        assertError("{}", "config_groups");
        assertError("{'config_groups':{},'extra':1}", "unknown member 'extra'");
        assertError("{'config_groups':{'s':{'type':'hadamard','scale':2}}}", "unknown member 'scale'");
        assertError("{'config_groups':{'s':{'type':'fwht'}}}", "unsupported transform type 'fwht'");
        assertError("{'config_groups':{'s':{'apply':[]}}}", "unsupported transform type");
        assertError("{'config_groups':{'s':{'type':'hadamard','head_dim':0}}}", "head_dim");
        assertError("{'config_groups':{'s':{'type':'hadamard','apply':{}}}}", "must be an array");
        assertError("{'config_groups':{'s':{'type':'hadamard','apply':[{'location':'input'}]}}}", "'targets'");
        assertError("{'config_groups':{'s':{'type':'hadamard','apply':[{'targets':['Linear']}]}}}", "'location'");
        assertError("{'config_groups':{'s':{'type':'hadamard','apply':[{'targets':['Linear'],"
                + "'location':'everywhere'}]}}}", "unknown transform location 'everywhere'");
        assertError("{'config_groups':{'s':{'type':'hadamard','apply':[{'targets':[1],'location':'input'}]}}}",
                "strings");
        assertError("{'config_groups':{'s':{'type':'hadamard','apply':[{'targets':['Linear'],'location':'input',"
                + "'inverse':'yes'}]}}}", "boolean");
    }

    private static void assertError(String json, String messagePart) {
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> parse(json));
        assertTrue(e.getMessage(), e.getMessage().contains(messagePart));
        assertTrue(e.getMessage(), e.getMessage().startsWith("config.json: quantization_config.transform_config"));
    }
}
