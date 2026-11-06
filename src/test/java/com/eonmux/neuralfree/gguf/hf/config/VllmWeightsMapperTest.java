package com.eonmux.neuralfree.gguf.hf.config;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@link VllmWeightsMapper}: the generated table's mappers rename checkpoint tensors as vLLM's
 * {@code WeightsMapper._map_name_with_shard} does (the expected names are Python's {@code re.sub},
 * {@code str.replace(.., 1)} and {@code rsplit} results), the stacked shards of a fused module, the Python-to-Java
 * regex translation and the table reader's validation.
 */
public class VllmWeightsMapperTest {

    private static final String HEADER =
            "# weights_mappers.tsv - Generated from vLLM abc123 by gen.py; do not edit. Rows: %d\n"
            + "# architecture\tkind\tmatch\treplacement\tshard\n";

    @Test
    public void prefixMappersMoveTheLanguageModelOfVisionCheckpoints() {
        VllmWeightsMapper m = VllmWeightsMapper.of("Qwen2_5_VLForConditionalGeneration");
        assertEquals("Qwen2_5_VLForConditionalGeneration", m.architecture());
        assertFalse(m.unresolved());
        assertEquals("language_model.model.layers.0.mlp.down_proj.weight",
                m.map("model.language_model.layers.0.mlp.down_proj.weight"));
        assertEquals("language_model.model.layers.3.self_attn.q_proj.weight",
                m.map("model.layers.3.self_attn.q_proj.weight"));
        assertEquals("language_model.lm_head.weight", m.map("lm_head.weight"));
        assertEquals("visual.blocks.0.attn.qkv.weight", m.map("model.visual.blocks.0.attn.qkv.weight"));
    }

    @Test
    public void droppedPrefixesMapToNull() {
        VllmWeightsMapper m = VllmWeightsMapper.of("Qwen3_5ForConditionalGeneration");
        assertNull(m.map("mtp.layers.0.mlp.gate_proj.weight"));
        assertEquals("language_model.model.layers.1.linear_attn.in_proj_qkvz.weight",
                m.map("model.language_model.layers.1.linear_attn.in_proj_qkvz.weight"));
    }

    @Test
    public void substrAndSuffixMappersRenameModules() {
        VllmWeightsMapper m = VllmWeightsMapper.of("GptOssForCausalLM");
        assertEquals("model.layers.0.attn.q_proj.weight", m.map("model.layers.0.self_attn.q_proj.weight"));
        assertEquals("model.layers.0.mlp.experts.w13_weight", m.map("model.layers.0.mlp.experts.gate_up_proj"));
        assertEquals("model.layers.0.mlp.experts.w2_bias", m.map("model.layers.0.mlp.experts.down_proj_bias"));
        assertEquals("model.embedding.weight", m.map("model.embed_tokens.weight"));
    }

    @Test
    public void regexMappersRewriteMistralNames() {
        VllmWeightsMapper m = VllmWeightsMapper.of("MistralLarge3ForCausalLM");
        assertEquals("model.layers.4.self_attn.kv_b_proj.weight", m.map("layers.4.attention.wkv_b.weight"));
        assertEquals("model.layers.12.mlp.gate_proj.weight", m.map("layers.12.feed_forward.w1.weight"));
    }

    @Test
    public void stackedMappersNameTheFusedModuleAndItsShards() {
        VllmWeightsMapper llama = VllmWeightsMapper.of("LlamaForCausalLM");
        assertEquals("model.layers.0.self_attn.qkv_proj.weight", llama.map("model.layers.0.self_attn.k_proj.weight"));
        assertEquals("model.layers.0.mlp.gate_up_proj.weight", llama.map("model.layers.0.mlp.up_proj.weight"));
        assertEquals(Arrays.asList("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj"), llama.stackedShards("model.layers.0.self_attn.qkv_proj"));
        assertEquals(Arrays.asList("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"),
                llama.stackedShards("model.layers.0.mlp.gate_up_proj"));
        assertTrue(llama.stackedShards("model.layers.0.mlp.down_proj").isEmpty());
        VllmWeightsMapper internLm = VllmWeightsMapper.of("InternLM2ForCausalLM");
        assertEquals(Arrays.asList("model.layers.2.feed_forward.w1", "model.layers.2.feed_forward.w3"),
                internLm.stackedShards("model.layers.2.feed_forward.gate_up_proj"));
    }

    @Test
    public void architecturesWithoutAMapperAreTheIdentity() {
        assertSame(VllmWeightsMapper.IDENTITY, VllmWeightsMapper.of("NoSuchForCausalLM"));
        assertSame(VllmWeightsMapper.IDENTITY, VllmWeightsMapper.of(null));
        assertEquals("model.layers.0.mlp.down_proj.weight",
                VllmWeightsMapper.IDENTITY.map("model.layers.0.mlp.down_proj.weight"));
        assertTrue(VllmWeightsMapper.IDENTITY.stackedShards("model.layers.0.self_attn.qkv_proj").isEmpty());
    }

    @Test
    public void mappersBuiltByCodeAreUnresolvedAndRefuseToMap() {
        VllmWeightsMapper m = VllmWeightsMapper.of("Cosmos3EdgeForConditionalGeneration");
        assertTrue(m.unresolved());
        assertTrue(m.unresolvedReason(), m.unresolvedReason().contains("range"));
        try {
            m.map("model.layers.0.mlp.down_proj.weight");
            fail("an unresolved mapper must not map");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("Cosmos3EdgeForConditionalGeneration"));
        }
    }

    @Test
    public void kindsApplyInVllmsOrderWhateverTheTableOrder() {
        VllmWeightsMapper m = VllmWeightsMapper.of("T", Arrays.asList(
                new String[]{"suffix", ".a", ".b", ""},
                new String[]{"prefix", "x.", "y.", ""},
                new String[]{"substr", ".m.", ".n.", ""},
                new String[]{"regex", "(\\d+)\\.m", "L\\1.m", ""}));
        assertEquals("y.L1.n.m.a.b", m.map("x.1.m.m.a.a"));
        assertEquals("L2.n.L3.m", m.map("2.m.3.m"));
    }

    @Test
    public void everyKindCanDropTheWeight() {
        for (String kind : new String[]{"regex", "substr", "prefix", "suffix"}) {
            VllmWeightsMapper m = VllmWeightsMapper.of("T", Collections.singletonList(
                    new String[]{kind, "mtp", null, ""}));
            assertNull(kind, m.map("mtp"));
            assertEquals(kind, "model.norm.weight", m.map("model.norm.weight"));
        }
    }

    @Test
    public void unknownKindsAreRefused() {
        try {
            VllmWeightsMapper.of("T", Collections.singletonList(new String[]{"infix", "a", "b", ""}));
            fail("an unknown kind must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("infix"));
        }
    }

    @Test
    public void pythonPatternsTranslateToJava() {
        assertEquals("\\Alayers\\.(\\d+)\\z", VllmWeightsMapper.javaPattern("\\Alayers\\.(\\d+)\\Z"));
        String named = VllmWeightsMapper.javaPattern("(?P<n>\\d+)\\.(?P=n)");
        assertEquals("(?<n>\\d+)\\.\\k<n>", named);
        String replacement = VllmWeightsMapper.javaReplacement("<\\g<n>>");
        assertEquals("a.<7>.b", Pattern.compile(named).matcher("a.7.7.b").replaceAll(replacement));
        assertEquals("\\a$1\\b$2\\c${n}\\$", VllmWeightsMapper.javaReplacement("a\\1b\\g<2>c\\g<n>$"));
        assertEquals("$12\\.", VllmWeightsMapper.javaReplacement("\\12."));
        assertEquals("\\a\\.\\b", VllmWeightsMapper.javaReplacement("a\\.b"));
        Pattern twelve = Pattern.compile("(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)(l)");
        assertEquals("python's \\g<1>0 is group 1 then a literal 0", "a0",
                twelve.matcher("abcdefghijkl").replaceAll(VllmWeightsMapper.javaReplacement("\\g<1>0")));
        assertEquals("l", twelve.matcher("abcdefghijkl").replaceAll(VllmWeightsMapper.javaReplacement("\\12")));
        for (String bad : new String[]{"\\0", "\\g<1"}) {
            try {
                VllmWeightsMapper.javaReplacement(bad);
                fail(bad + " must be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains(bad));
            }
        }
    }

    @Test
    public void theReaderGroupsRowsByArchitecture() throws IOException {
        Map<String, VllmWeightsMapper> t = parse(String.format(HEADER, 3)
                + "A\tprefix\tmodel.\tlm.\t\n"
                + "B\tstacked\t.w1\t.gate_up_proj\t0\n"
                + "A\tsubstr\t.drop.\t<drop>\t\n");
        assertEquals(2, t.size());
        assertEquals("lm.layers.0.x", t.get("A").map("model.layers.0.x"));
        assertNull(t.get("A").map("model.drop.x"));
        assertEquals(Collections.singletonList("m.w1"), t.get("B").stackedShards("m.gate_up_proj"));
    }

    @Test
    public void theReaderRefusesMalformedTables() throws IOException {
        assertRefused("Rows: 2", "# weights_mappers.tsv - Generated from vLLM abc123 by gen.py; do not edit. Rows: 2"
                + "\n# architecture\tkind\tmatch\treplacement\tshard\nA\tprefix\ta\tb\t\n", "header records 2 rows");
        assertRefused("header", "# weights_mappers.tsv\n", ":1: malformed header line");
        assertRefused("columns", String.format(HEADER, 0).replace("shard", "shards"), ":2: the column line");
        assertRefused("4 columns", String.format(HEADER, 1) + "A\tprefix\ta\tb\n", ":3: expected 5 columns");
        assertRefused("no kind", String.format(HEADER, 1) + "A\t\ta\tb\t\n", ":3: expected 5 columns");
        assertRefused("kind", String.format(HEADER, 1) + "A\tinfix\ta\tb\t\n", "the mapper of A");
    }

    @Test
    public void theGeneratedTableLoads() {
        VllmWeightsMapper m = VllmWeightsMapper.of("Gemma3ForConditionalGeneration");
        assertEquals("language_model.model.layers.0.mlp.down_proj.weight",
                m.map("model.language_model.layers.0.mlp.down_proj.weight"));
        assertTrue(m.toString(), m.toString().contains("Gemma3ForConditionalGeneration"));
    }

    private static Map<String, VllmWeightsMapper> parse(String table) throws IOException {
        return VllmWeightsMapper.parse(new BufferedReader(new StringReader(table)));
    }

    private static void assertRefused(String what, String table, String message) throws IOException {
        try {
            parse(table);
            fail(what + " must be refused");
        } catch (IllegalStateException expected) {
            assertTrue(what + ": " + expected.getMessage(), expected.getMessage().contains(message));
        }
    }
}
