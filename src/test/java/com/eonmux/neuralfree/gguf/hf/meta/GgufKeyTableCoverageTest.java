package com.eonmux.neuralfree.gguf.hf.meta;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The key registry and the key table agree: every {@link GgufKeys} constant the Hugging Face path uses has a type,
 * every key of {@code gguf_keys.tsv} has a constant, and every alias row is typed as the key table types its key.
 */
public class GgufKeyTableCoverageTest {

    private static final Path HF_SOURCES = Path.of("src/main/java/com/eonmux/neuralfree/gguf/hf");
    private static final Pattern KEY_USE = Pattern.compile("\\bGgufKeys\\.([A-Z][A-Z0-9_]*)\\b");
    private static final Pattern TSV_PLACEHOLDER = Pattern.compile("\\{(arch|name|id|n|i)}");

    private static Map<String, String> constants() throws IllegalAccessException {
        Map<String, String> out = new HashMap<>();
        for (Field f : GgufKeys.class.getDeclaredFields()) {
            int mod = f.getModifiers();
            if (Modifier.isPublic(mod) && Modifier.isStatic(mod) && f.getType() == String.class) {
                out.put(f.getName(), (String) f.get(null));
            }
        }
        return out;
    }

    private static String javaTemplate(String tsvTemplate) {
        Matcher m = TSV_PLACEHOLDER.matcher(tsvTemplate);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            String p = m.group(1);
            m.appendReplacement(sb, "arch".equals(p) || "name".equals(p) ? "%s" : "%d");
        }
        m.appendTail(sb);
        return sb.toString();
    }

    @Test
    public void everyKeyTheHfPathUsesIsTyped() throws IOException, IllegalAccessException {
        assertTrue("run from the project directory: " + HF_SOURCES.toAbsolutePath(), Files.isDirectory(HF_SOURCES));
        Map<String, String> constants = constants();
        TreeSet<String> used = new TreeSet<>();
        List<Path> files;
        try (Stream<Path> s = Files.walk(HF_SOURCES)) {
            files = s.filter(p -> p.toString().endsWith(".java")).collect(Collectors.toList());
        }
        for (Path p : files) {
            Matcher m = KEY_USE.matcher(new String(Files.readAllBytes(p), StandardCharsets.UTF_8));
            while (m.find()) {
                used.add(m.group(1));
            }
        }
        assertTrue(used.size() > 20);
        List<String> untyped = new ArrayList<>();
        for (String name : used) {
            String value = constants.get(name);
            assertNotNull("GgufKeys." + name + " is not a public String constant", value);
            if (GgufKeyTable.type(value) == null) {
                untyped.add(name + " = " + value);
            }
        }
        assertEquals("GgufKeys constants used under gguf/hf without a GgufKeyTable type", new ArrayList<String>(),
                untyped);
    }

    @Test
    public void everyTableKeyHasAConstant() throws IllegalAccessException {
        Map<String, String> constants = constants();
        Pattern dense = Pattern.compile("%s\\.dense_[0-9]+_feat_(in|out)");
        List<String> missing = new ArrayList<>();
        int rows = 0;
        for (HfNameTables.TsvRow row : HfNameTables.read("gguf_keys.tsv",
                "writer_method", "key_template", "value_type").rows()) {
            rows++;
            String tsv = row.get(1);
            if (tsv.contains("{dense}")) {
                String suffix = tsv.endsWith("_feat_in") ? "in" : "out";
                boolean found = constants.values().stream()
                        .anyMatch(v -> dense.matcher(v).matches() && v.endsWith("_feat_" + suffix));
                if (!found) {
                    missing.add(tsv);
                }
            } else if (!constants.containsValue(javaTemplate(tsv))) {
                missing.add(tsv + " (" + javaTemplate(tsv) + ")");
            }
        }
        assertEquals(389, rows);
        assertEquals("gguf_keys.tsv keys without a GgufKeys constant", new ArrayList<String>(), missing);
    }

    @Test
    public void tableTypesFollowTheWriterMethods() {
        assertEquals(GGUFValueType.UINT32, GgufKeyTable.type(GgufKeys.CONTEXT_LENGTH).type());
        assertTrue(GgufKeyTable.type(GgufKeys.FEED_FORWARD_LENGTH).any());
        assertEquals("ANY", GgufKeyTable.type(GgufKeys.FEED_FORWARD_LENGTH).toString());
        assertEquals("ARRAY:INT32", GgufKeyTable.type(GgufKeys.ROPE_DIMENSION_SECTIONS).toString());
        assertEquals(GGUFValueType.STRING, GgufKeyTable.type(GgufKeys.TOKENIZER_CHAT_TEMPLATE_N).type());
        assertEquals(GGUFValueType.UINT32, GgufKeyTable.type(GgufKeys.TOKENIZER_CLS_ID).type());
        assertEquals(GGUFValueType.BOOL, GgufKeyTable.type(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK).type());
        assertEquals(GGUFValueType.UINT32, GgufKeyTable.type(GgufKeys.DENSE_2_FEAT_OUT).type());
        assertEquals(null, GgufKeyTable.type("%s.no_such_key"));
        for (String template : GgufKeyTable.templates()) {
            assertNotNull(template, GgufKeyTable.type(template));
        }
        Map.Entry<String, GgufKeyTable.KeyType> r = GgufKeyTable.resolve("qwen3.decision.score_bias.1.2", "qwen3");
        assertEquals(GgufKeys.DECISION_SCORE_BIAS, r.getKey());
        assertEquals(GGUFValueType.FLOAT32, r.getValue().type());
        assertEquals(GgufKeys.CONTEXT_LENGTH, GgufKeyTable.resolve("qwen3.context_length", "qwen3").getKey());
        assertEquals(null, GgufKeyTable.resolve("llama.context_length", "qwen3"));
    }

    @Test
    public void aliasRowsAreTypedAsTheirKeys() {
        assertEquals(43, HfHparamAliases.rows().size());
        int writes = 0;
        for (HfHparamAliases.Row r : HfHparamAliases.rows()) {
            if (r.template() != null) {
                writes++;
                GgufKeyTable.KeyType t = GgufKeyTable.type(r.template());
                assertNotNull(r.toString(), t);
                assertEquals(r.toString(), t.toString(), r.type().toString());
            }
        }
        assertEquals(32, writes);
    }
}
