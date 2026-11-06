package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.MiniJinja;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Value-usage statistics follow llama.cpp's {@code is_get_stats} marking (common/jinja/runtime.cpp): lookups,
 * member access, filters, tests and loops mark the values they touch, with the operations recorded.
 */
public class JinjaUsageTest {

    private static Map<String, Object> message(Object content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", new String("user"));
        m.put("content", content);
        return m;
    }

    private static JinjaUsage render(String template, Map<String, Object> context) {
        JinjaUsage usage = new JinjaUsage();
        MiniJinja.compile(template).render(context, usage);
        return usage;
    }

    private static Map<String, Object> messages(Object... items) {
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", new ArrayList<>(Arrays.asList(items)));
        return context;
    }

    private static Object content(JinjaUsage usage, int index) {
        return ((Map<?, ?>) ((List<?>) usage.variable("messages")).get(index)).get("content");
    }

    @Test
    public void anAttributeReadMarksTheValueUsedAndOnlyThatValue() {
        JinjaUsage usage = render("{% for m in messages %}{{ m.role }}{% endfor %}",
                messages(message(new String("a"))));
        Map<?, ?> m = (Map<?, ?>) ((List<?>) usage.variable("messages")).get(0);
        assertTrue(usage.used(m.get("role")));
        assertFalse("content is never read", usage.used(m.get("content")));
        assertTrue(usage.hasOp(m, JinjaUsage.OBJECT_ACCESS));
        assertTrue(usage.hasOp(usage.variable("messages"), JinjaUsage.ARRAY_ACCESS));
    }

    @Test
    public void anIntegerSubscriptIsAnArrayAccess() {
        JinjaUsage usage = render("{{ messages[0]['content'][0] }}", messages(message(new String("xyz"))));
        assertTrue(usage.hasOp(content(usage, 0), JinjaUsage.ARRAY_ACCESS));
        assertTrue(usage.hasOp(usage.variable("messages"), JinjaUsage.ARRAY_ACCESS));
    }

    @Test
    public void testsAndFiltersRecordTheirNames() {
        JinjaUsage usage = render("{% if messages[0].content is string %}{{ messages[0].content | trim }}{% endif %}",
                messages(message(new String(" s "))));
        Object content = content(usage, 0);
        assertTrue(usage.hasOp(content, JinjaUsage.TEST_PREFIX + "string"));
        assertTrue(usage.hasOp(content, "trim"));
        assertFalse(usage.hasOp(content, JinjaUsage.ARRAY_ACCESS));
    }

    @Test
    public void aMethodAccessOnAStringRecordsTheMethod() {
        JinjaUsage usage = render("{{ messages[0].content.strip() }}", messages(message(new String(" s "))));
        assertTrue(usage.hasOp(content(usage, 0), "strip"));
    }

    @Test
    public void selectattrOnTypedContentIsRecorded() {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "text");
        part.put("text", new String("T"));
        List<Object> parts = new ArrayList<>();
        parts.add(part);
        JinjaUsage usage = render("{% for p in messages[0].content | selectattr('type', 'equalto', 'text') %}"
                + "{{ p.text }}{% endfor %}", messages(message(parts)));
        Object content = content(usage, 0);
        assertTrue(usage.hasOp(content, "selectattr"));
        assertTrue(usage.used(((Map<?, ?>) ((List<?>) content).get(0)).get("text")));
    }

    @Test
    public void tojsonMarksTheWholeTreeUsed() {
        Map<String, Object> args = new LinkedHashMap<>();
        Map<String, Object> nested = new LinkedHashMap<>();
        nested.put("deep", new String("v"));
        args.put("arg", nested);
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("args", args);
        JinjaUsage usage = render("{{ args | tojson }}", context);
        Map<?, ?> converted = (Map<?, ?>) usage.variable("args");
        assertTrue(usage.used(((Map<?, ?>) converted.get("arg")).get("deep")));
    }

    @Test
    public void aRaisingRenderKeepsWhatRanBeforeTheError() {
        JinjaUsage usage = new JinjaUsage();
        MiniJinja.Template t = MiniJinja.compile("{{ messages[0].content }}{{ raise_exception('x') }}");
        assertThrows(JinjaRuntime.TemplateException.class,
                () -> t.render(messages(message(new String("c"))), usage));
        assertTrue(usage.used(content(usage, 0)));
    }

    @Test
    public void undefinedIsNeverUsedAndVariablesAreTheConvertedValues() {
        String value = new String("v");
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("x", value);
        JinjaUsage usage = render("{{ missing.attr }}{{ x }}", context);
        assertFalse(usage.used(JinjaValue.UNDEFINED));
        assertSame(value, usage.variable("x"));
        assertTrue(usage.used(value));
        assertEquals(JinjaValue.UNDEFINED, usage.variable("absent"));
    }

    @Test
    public void aRenderWithoutStatisticsRendersTheSame() {
        String template = "{% for m in messages %}{{ m.role }}:{{ m.content }}{% endfor %}";
        Map<String, Object> context = messages(message("hi"));
        assertEquals(MiniJinja.compile(template).render(context),
                MiniJinja.compile(template).render(context, new JinjaUsage()));
    }
}
