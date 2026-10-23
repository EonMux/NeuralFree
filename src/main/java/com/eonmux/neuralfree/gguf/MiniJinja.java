package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.jinja.JinjaAst;
import com.eonmux.neuralfree.gguf.jinja.JinjaParser;
import com.eonmux.neuralfree.gguf.jinja.JinjaRuntime;

import java.util.Collections;
import java.util.Map;

/**
 * Jinja2 renderer for GGUF chat templates ({@code tokenizer.chat_template}).
 *
 * <p>Implements the environment HF transformers and llama.cpp use for chat templates:
 * {@code trim_blocks} and {@code lstrip_blocks} on, {@code -}/{@code +} whitespace control,
 * Python value semantics (str/int/float/bool/None, lists, tuples, dicts, slicing, Python string and
 * container methods), the Jinja2 built-in filters and tests, the HF additions ({@code tojson} with
 * {@code ensure_ascii/indent/separators/sort_keys}, {@code from_json}, {@code raise_exception},
 * {@code strftime_now}) and the statements {@code if}, {@code for} (with {@code loop}, filters,
 * {@code else}, {@code break}/{@code continue}), {@code set} (including {@code namespace} attributes
 * and block form), {@code macro}, {@code filter} and {@code generation} blocks.
 *
 * <p>The engine lives in {@code com.eonmux.neuralfree.gguf.jinja}; this class is its entry point.
 * Errors (syntax, unknown filters or tests, type errors, {@code raise_exception}) surface as
 * {@link JinjaRuntime.TemplateException} carrying the template line.
 */
public final class MiniJinja {

    private MiniJinja() {}

    /**
     * Renders a template once.
     *
     * @param template Jinja source; must not be null
     * @param context  template variables (Java strings, boxed numbers, booleans, null, collections,
     *                 maps and arrays, converted to Python-like values); null means no variables
     * @throws IllegalArgumentException if {@code template} is null
     * @throws JinjaRuntime.TemplateException on syntax or evaluation errors
     */
    public static String render(String template, Map<String, ?> context) {
        return compile(template).render(context);
    }

    /**
     * Parses a template for repeated rendering.
     *
     * @throws IllegalArgumentException if {@code template} is null
     * @throws JinjaRuntime.TemplateException on syntax errors
     */
    public static Template compile(String template) {
        if (template == null) throw new IllegalArgumentException("chat template must not be null");
        return new Template(JinjaParser.parse(template));
    }

    /** A parsed, immutable template; {@link #render} is thread-safe. */
    public static final class Template {
        private final JinjaAst.Program program;

        private Template(JinjaAst.Program program) { this.program = program; }

        /** Renders with the given variables; null means no variables. Caller data is never mutated. */
        public String render(Map<String, ?> context) {
            return program.render(context == null ? Collections.<String, Object>emptyMap() : context);
        }
    }
}
