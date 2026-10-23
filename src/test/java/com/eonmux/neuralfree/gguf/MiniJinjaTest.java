package com.eonmux.neuralfree.gguf;

import org.junit.Test;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Smoke/regression tests for the {@link MiniJinja} chat-template engine. These exercise the
 * core constructs that GGUF chat templates depend on (variable substitution, conditionals,
 * loops over message lists) so behavior is pinned. Pure logic; no GPU or model file required.
 */
public class MiniJinjaTest {

    private static Map<String, Object> ctx() {
        return new LinkedHashMap<>();
    }

    @Test
    public void variableSubstitution() {
        Map<String, Object> ctx = ctx();
        ctx.put("greeting", "Hello");
        ctx.put("name", "World");
        assertEquals("Hello, World!", MiniJinja.render("{{ greeting }}, {{ name }}!", ctx));
    }

    @Test
    public void forLoopConcatenation() {
        Map<String, Object> ctx = ctx();
        ctx.put("items", List.of("a", "b", "c"));
        assertEquals("abc", MiniJinja.render("{% for x in items %}{{ x }}{% endfor %}", ctx));
    }

    @Test
    public void ifElseBranches() {
        Map<String, Object> on = ctx();
        on.put("flag", true);
        assertEquals("YES", MiniJinja.render("{% if flag %}YES{% else %}NO{% endif %}", on));

        Map<String, Object> off = ctx();
        off.put("flag", false);
        assertEquals("NO", MiniJinja.render("{% if flag %}YES{% else %}NO{% endif %}", off));
    }

    @Test
    public void chatMessageLoopRendersRolesAndContent() {
        // Mirrors the shape of a real chat template iterating over a message list.
        List<Map<String, Object>> messages = new ArrayList<>();
        Map<String, Object> sys = new LinkedHashMap<>();
        sys.put("role", "system");
        sys.put("content", "be helpful");
        messages.add(sys);
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        user.put("content", "hi");
        messages.add(user);

        Map<String, Object> ctx = ctx();
        ctx.put("messages", messages);

        String template = "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}";
        String out = MiniJinja.render(template, ctx);

        assertTrue("missing system message: " + out, out.contains("<system>be helpful</system>"));
        assertTrue("missing user message: " + out, out.contains("<user>hi</user>"));
        assertEquals("<system>be helpful</system><user>hi</user>", out);
    }
}
