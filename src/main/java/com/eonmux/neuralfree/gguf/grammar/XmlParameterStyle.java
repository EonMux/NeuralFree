package com.eonmux.neuralfree.gguf.grammar;

import java.util.Arrays;
import java.util.List;

/**
 * The XML parameter styles of xgrammar's structural-tag {@code json_schema} format: how one top-level property is
 * written as a tagged parameter, its name inside the opening tag and its value up to the closing tag.
 *
 * <ul>
 *   <li>{@code qwen_xml}: {@code <parameter=NAME>VALUE</parameter>} (Qwen3-Coder);</li>
 *   <li>{@code minimax_xml}: {@code <parameter name="NAME">VALUE</parameter>} (MiniMax M2);</li>
 *   <li>{@code deepseek_xml}: {@code <｜DSML｜parameter name="NAME" string="true|false">VALUE</｜DSML｜parameter>}
 *       (DeepSeek V3.2 and V4), {@code string="true"} for a string value written raw, {@code "false"} for any
 *       other value written as JSON;</li>
 *   <li>{@code deepseek_v4_1_xml}: the same with {@code <｜DSML｜ parameter} and {@code </｜DSML｜ parameter>};</li>
 *   <li>{@code glm_xml}: {@code <arg_key>NAME</arg_key><arg_value>VALUE</arg_value>} (GLM 4.5 to 4.7).</li>
 * </ul>
 */
enum XmlParameterStyle {
    QWEN("qwen_xml", "<parameter=", ">", null, "</parameter>", false),
    MINIMAX("minimax_xml", "<parameter name=\"", "\">", null, "</parameter>", false),
    DEEPSEEK("deepseek_xml", "<｜DSML｜parameter name=\"", "\"", null, "</｜DSML｜parameter>", true),
    DEEPSEEK_V4_1("deepseek_v4_1_xml", "<｜DSML｜ parameter name=\"", "\"", null,
            "</｜DSML｜ parameter>", true),
    GLM("glm_xml", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", false);

    /** xgrammar's styles whose value models this engine does not compile. */
    static final List<String> UNSUPPORTED = Arrays.asList("minimax_m3_xml", "cohere_xml", "kimi_k3_xml");

    /** The {@code style} value naming this style. */
    final String keyword;
    /** The text before the parameter name. */
    final String keyOpen;
    /** The text right after the parameter name. */
    final String keyClose;
    /** The tag opening the value after whitespace, or null when the value follows the key's tag directly. */
    final String valueOpen;
    /** The closing tag of the value, the text an unconstrained string value never holds. */
    final String valueClose;
    /** Whether the opening tag declares the value's kind ({@code string="true|false"}). */
    final boolean typed;

    XmlParameterStyle(String keyword, String keyOpen, String keyClose, String valueOpen, String valueClose,
            boolean typed) {
        this.keyword = keyword;
        this.keyOpen = keyOpen;
        this.keyClose = keyClose;
        this.valueOpen = valueOpen;
        this.valueClose = valueClose;
        this.typed = typed;
    }

    /** The style named {@code keyword}, or null when no style of this engine has the name. */
    static XmlParameterStyle named(String keyword) {
        for (XmlParameterStyle style : values()) {
            if (style.keyword.equals(keyword)) {
                return style;
            }
        }
        return null;
    }

    /**
     * The GBNF text from the end of the parameter name to the value: the key's closing text, and for GLM the
     * whitespace {@code ws} and the value's opening tag; for a typed style the kind declaration of a string value
     * when {@code stringValue}, of any other value otherwise.
     */
    String keyEnd(String ws, boolean stringValue) {
        if (typed) {
            return JsonSchemaToGbnf.formatLiteral(keyClose + " string=\"" + stringValue + "\">");
        }
        String close = JsonSchemaToGbnf.formatLiteral(keyClose);
        return valueOpen == null ? close : close + " " + ws + " " + JsonSchemaToGbnf.formatLiteral(valueOpen);
    }
}
