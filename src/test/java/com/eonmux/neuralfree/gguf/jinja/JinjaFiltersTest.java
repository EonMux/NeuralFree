package com.eonmux.neuralfree.gguf.jinja;

import com.eonmux.neuralfree.gguf.MiniJinja;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Filters, methods, tests and statements of the chat-template engine, including the constructs
 * llama.cpp's bundled templates (models/templates/*.jinja) depend on. Pure logic; no model needed.
 */
public class JinjaFiltersTest {

    private static String render(String template) {
        return MiniJinja.render(template, new LinkedHashMap<String, Object>());
    }

    private static String render(String template, Map<String, Object> ctx) {
        return MiniJinja.render(template, ctx);
    }

    private static Map<String, Object> msg(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    // ------------------------------------------------------------------ string methods

    @Test
    public void formatMethodSupportsAutoIndexedAndNamedFields() {
        assertEquals("1 + 2 = 3", render("{{ '{} + {} = {}'.format(1, 2, 3) }}"));
        assertEquals("y:x", render("{{ '{name}:{0}'.format('x', name='y') }}"));
        assertEquals("{x}", render("{{ '{{{}}}'.format('x') }}"));
    }

    @Test
    public void findAndSplitlines() {
        assertEquals("2|-1", render("{{ 'hello'.find('l') }}|{{ 'hello'.find('z') }}"));
        assertEquals("3", render("{{ 'a\\nb\\r\\nc'.splitlines() | length }}"));
    }

    @Test
    public void stripSplitAndAffixMethods() {
        assertEquals("hi", render("{{ 'xxhixx'.strip('x') }}"));
        assertEquals("b", render("{{ 'a</think>b'.split('</think>')[-1].lstrip('\\n') }}"));
        assertEquals("True", render("{{ 'x'.startswith(('y', 'x')) }}"));
        assertEquals("['a', 'b c']", render("{{ 'a b c'.split(none, 1) }}"));
    }

    @Test
    public void percentFormattingOperatorAndFilter() {
        assertEquals("a='b'", render("{{ '%s=%r' % ('a', 'b') }}"));
        assertEquals("a-5", render("{{ '%s-%d' | format('a', 5) }}"));
        assertEquals("003.50", render("{{ '%06.2f' % 3.5 }}"));
    }

    @Test
    public void formatSpecAlignsFloatsAndCentresLikePython() {
        assertEquals("   1.5", render("{{ '{:>6}'.format(1.5) }}"));
        assertEquals(" ab  ", render("{{ '{:^5}'.format('ab') }}"));
        assertEquals("  ab ", render("{{ 'ab'.center(5) }}"));
        assertEquals("-0x1f", render("{{ '{:#x}'.format(-31) }}"));
    }

    @Test
    public void integerPowerAndNamespaceTruthiness() {
        assertEquals("1024", render("{{ 2 ** 10 }}"));
        assertEquals("yes", render("{% if namespace() %}yes{% endif %}"));
    }

    // ------------------------------------------------------------------ filters

    @Test
    public void safeUniqueAndReplace() {
        assertEquals("<b>", render("{{ '<b>' | safe }}"));
        assertEquals("a,B", render("{{ ['a', 'B', 'b', 'a'] | unique | join(',') }}"));
        assertEquals("bba", render("{{ 'aaa' | replace('a', 'b', 2) }}"));
    }

    @Test
    public void tojsonMatchesPythonJsonDumps() {
        assertEquals("{\"a\": 1, \"b\": [true, null]}", render("{{ {'a': 1, 'b': [true, none]} | tojson }}"));
        assertEquals("{\n  \"a\": 1\n}", render("{{ {'a': 1} | tojson(indent=2) }}"));
        assertEquals("{\"a\": 2, \"b\": 1}", render("{{ {'b': 1, 'a': 2} | tojson(sort_keys=true) }}"));
        assertEquals("\"\u00e9\"", render("{{ '\u00e9' | tojson }}"));
        assertEquals("\"\\u00e9\"", render("{{ '\u00e9' | tojson(ensure_ascii=true) }}"));
        assertEquals("[1.5, \"x\\ny\"]", render("{{ [1.5, 'x\\ny'] | tojson }}"));
    }

    @Test
    public void fromJsonParsesNestedValues() {
        assertEquals("2.5", render("{{ ('{\"x\": [1, 2.5]}' | from_json).x[1] }}"));
    }

    @Test
    public void dictsortItemsAndAttributePaths() {
        assertEquals("a=1;b=2;", render("{% for k, v in {'b': 2, 'a': 1} | dictsort %}{{ k }}={{ v }};{% endfor %}"));
        assertEquals("a1b2", render("{% for k, v in {'a': 1, 'b': 2}.items() %}{{ k }}{{ v }}{% endfor %}"));
        assertEquals("a", render("{{ {'a': 1, '$ref': 2} | items | rejectattr('0', 'equalto', '$ref')"
                + " | map(attribute='0') | join(',') }}"));
    }

    @Test
    public void mapSelectattrAndMin() {
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("messages", Arrays.asList(msg("system", "s"), msg("user", "u1"), msg("assistant", "a"), msg("user", "u2")));
        assertEquals("2", render("{{ messages | selectattr('role', 'equalto', 'user') | list | length }}", ctx));
        assertEquals("system, user, assistant, user", render("{{ messages | map(attribute='role') | join(', ') }}", ctx));
        assertEquals("AB", render("{{ ['a', 'b'] | map('upper') | join }}"));
        assertEquals("1", render("{{ [3, 1, 2] | min }}"));
    }

    @Test
    public void indentFollowsJinja2() {
        assertEquals("a\n  b", render("{{ 'a\\nb' | indent(2) }}"));
        assertEquals("  a\n  b", render("{{ 'a\\nb' | indent(2, true) }}"));
    }

    @Test
    public void conversionsDefaultsAndTitle() {
        assertEquals("43", render("{{ '42' | int + 1 }}"));
        assertEquals("3.5", render("{{ '3.5' | float }}"));
        assertEquals("d", render("{{ none | default('d') }}"));
        assertEquals("Hello World-Foo", render("{{ 'hello world-foo' | title }}"));
        assertEquals("a-b", render("{{ ('a', 'b') | join('-') }}"));
    }

    // ------------------------------------------------------------------ values, tests, statements

    @Test
    public void pythonValueRendering() {
        assertEquals("[1, 'a', None]", render("{{ [1, 'a', none] }}"));
        assertEquals("1.0", render("{{ 1.0 }}"));
        assertEquals("None", render("{{ none }}"));
        assertEquals("234cba", render("{{ [1, 2, 3, 4][1:] | join }}{{ 'abc'[::-1] }}"));
    }

    @Test
    public void typeTests() {
        assertEquals("TrueTrueFalseTrue", render("{{ none is none }}{{ 1 is integer }}{{ true is integer }}{{ 'a' is string }}"));
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("d", new LinkedHashMap<String, Object>());
        assertEquals("True", render("{{ d.get('zz') is none }}", ctx));
        assertEquals("True", render("{{ 'a' in {'a': 1} }}"));
    }

    @Test
    public void loopVariablesFilterElseBreakContinue() {
        assertEquals("1False2True", render("{% for x in 'ab' %}{{ loop.index }}{{ loop.last }}{% endfor %}"));
        assertEquals("2232", render("{% for x in [1, 2, 3] if x > 1 %}{{ x }}{{ loop.length }}{% else %}none{% endfor %}"));
        assertEquals("none", render("{% for x in [] %}{{ x }}{% else %}none{% endfor %}"));
        assertEquals("02", render("{% for i in range(5) %}{% if i == 3 %}{% break %}{% endif %}"
                + "{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endfor %}"));
    }

    @Test
    public void scopingNamespaceAndSetBlock() {
        assertEquals("1", render("{% set x = 1 %}{% for i in [1] %}{% set x = 2 %}{% endfor %}{{ x }}"));
        assertEquals("3", render("{% set ns = namespace(n=0) %}{% for i in range(3) %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}"));
        assertEquals("[hi 2]", render("{% set x %}hi {{ 1 + 1 }}{% endset %}[{{ x }}]"));
        assertEquals("ab", render("{% set a, b = 'a', 'b' %}{{ a }}{{ b }}"));
    }

    @Test
    public void macrosFilterAndGenerationBlocks() {
        assertEquals("1x12", render("{% macro f(a, b='x') %}{{ a }}{{ b }}{% endmacro %}{{ f(1) }}{{ f(1, b=2) }}"));
        assertEquals("ABC", render("{% filter upper %}abc{% endfilter %}"));
        assertEquals("abc", render("{% generation %}abc{% endgeneration %}"));
    }

    @Test
    public void whitespaceControlMatchesHfEnvironment() {
        assertEquals("yes\n", render("{% if true %}\nyes\n{% endif %}\n"));
        assertEquals("x", render("  {% if true %}x{% endif %}"));
        assertEquals("abc", render("a  {{- 'b' -}}  c"));
        assertEquals("  x", render("  {%+ if true %}x{% endif %}"));
    }

    @Test
    public void strftimeNowRendersYear() {
        assertEquals("4", render("{{ strftime_now('%Y') | length }}"));
    }

    @Test
    public void callerDataIsNeverMutated() {
        List<Object> items = new ArrayList<>();
        items.add("a");
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("items", items);
        assertEquals("2", render("{% set _ = items.append('b') %}{{ items | length }}", ctx));
        assertEquals(1, items.size());
    }

    @Test
    public void compiledTemplateRendersRepeatedly() {
        MiniJinja.Template t = MiniJinja.compile("{{ x * 2 }}");
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("x", 2);
        assertEquals("4", t.render(ctx));
        ctx.put("x", 5);
        assertEquals("10", t.render(ctx));
    }

    // ------------------------------------------------------------------ errors

    @Test
    public void raiseExceptionCarriesMessageAndLine() {
        try {
            render("ok\n{{ raise_exception('boom') }}");
            fail("expected TemplateException");
        } catch (JinjaRuntime.TemplateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Jinja template error: boom"));
            assertEquals(2, e.line());
        }
    }

    @Test(expected = JinjaRuntime.TemplateException.class)
    public void unknownFilterIsRejected() {
        render("{{ x | no_such_filter }}");
    }

    @Test(expected = JinjaRuntime.TemplateException.class)
    public void unsupportedStatementIsRejected() {
        render("{% include 'other.jinja' %}");
    }

    // ------------------------------------------------------------------ template shapes

    @Test
    public void llama3StyleTemplate() {
        String template = String.join("\n",
                "{{- bos_token }}",
                "{%- if messages[0]['role'] == 'system' %}",
                "    {%- set system_message = messages[0]['content']|trim %}",
                "    {%- set messages = messages[1:] %}",
                "{%- else %}",
                "    {%- set system_message = \"\" %}",
                "{%- endif %}",
                "{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}",
                "{{- system_message }}",
                "{{- \"<|eot_id|>\" }}",
                "{%- for message in messages %}",
                "    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}",
                "{%- endfor %}",
                "{%- if add_generation_prompt %}",
                "    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}",
                "{%- endif %}");
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("bos_token", "<|begin_of_text|>");
        ctx.put("messages", Arrays.asList(msg("system", "Be brief. "), msg("user", " Hi ")));
        ctx.put("add_generation_prompt", true);
        assertEquals("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>"
                + "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n\n", render(template, ctx));
    }

    @Test
    public void qwen3StyleLastQueryIndexAndThinkSplit() {
        String template = String.join("\n",
                "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}",
                "{%- for message in messages[::-1] %}",
                "    {%- set index = (messages|length - 1) - loop.index0 %}",
                "    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>')"
                        + " and message.content.endswith('</tool_response>')) %}",
                "        {%- set ns.multi_step_tool = false %}",
                "        {%- set ns.last_query_index = index %}",
                "    {%- endif %}",
                "{%- endfor %}",
                "{{- ns.last_query_index }}");
        Map<String, Object> ctx = new LinkedHashMap<>();
        ctx.put("messages", Arrays.asList(msg("user", "a"), msg("assistant", "b"), msg("user", "c"), msg("assistant", "d")));
        assertEquals("2", render(template, ctx));

        String split = String.join("\n",
                "{%- if '</think>' in content %}",
                "    {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
                "    {%- set content = content.split('</think>')[-1].lstrip('\\n') %}",
                "{%- endif %}",
                "{{- reasoning_content }}|{{ content }}");
        Map<String, Object> c2 = new LinkedHashMap<>();
        c2.put("content", "<think>\nR\n</think>\n\nC");
        assertEquals("R|C", render(split, c2));
    }
}
