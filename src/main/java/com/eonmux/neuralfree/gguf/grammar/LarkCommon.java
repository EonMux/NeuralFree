package com.eonmux.neuralfree.gguf.grammar;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The terminals of Lark's {@code common} module that {@code %import common.NAME} brings in, each as the regular
 * expression of its Lark definition ({@code lark/grammars/common.lark}) — the set and the languages xgrammar's
 * Lark front end provides: numbers ({@code DIGIT}, {@code HEXDIGIT}, {@code INT}, {@code SIGNED_INT},
 * {@code DECIMAL}, {@code _EXP}, {@code FLOAT}, {@code SIGNED_FLOAT}, {@code NUMBER}, {@code SIGNED_NUMBER}), the
 * double-quoted {@code ESCAPED_STRING}, letters and names ({@code LETTER}, {@code LCASE_LETTER},
 * {@code UCASE_LETTER}, {@code WORD}, {@code CNAME}), whitespace and line breaks ({@code WS_INLINE}, {@code WS},
 * {@code CR}, {@code LF}, {@code NEWLINE}) and comments ({@code C_COMMENT}, {@code CPP_COMMENT},
 * {@code SH_COMMENT}, {@code SQL_COMMENT}).
 */
final class LarkCommon {

    private static final String INT = "[0-9]+";
    private static final String EXP = "[eE][+-]?[0-9]+";
    private static final String DECIMAL = "[0-9]+\\.[0-9]*|\\.[0-9]+";
    private static final String FLOAT = "(?:" + DECIMAL + ")(?:" + EXP + ")?|[0-9]+" + EXP;
    private static final String NUMBER = "[0-9]+|(?:" + DECIMAL + ")(?:" + EXP + ")?|[0-9]+" + EXP;

    private static final Map<String, String> PATTERNS = patterns();

    private LarkCommon() {
    }

    /** The regular expression of the common terminal {@code name}, or null when the module has none. */
    static String pattern(String name) {
        return PATTERNS.get(name);
    }

    private static Map<String, String> patterns() {
        Map<String, String> p = new LinkedHashMap<>();
        p.put("DIGIT", "[0-9]");
        p.put("HEXDIGIT", "[a-fA-F0-9]");
        p.put("INT", INT);
        p.put("SIGNED_INT", "[+-]?" + INT);
        p.put("DECIMAL", DECIMAL);
        p.put("_EXP", EXP);
        p.put("FLOAT", FLOAT);
        p.put("SIGNED_FLOAT", "[+-]?(?:" + FLOAT + ")");
        p.put("NUMBER", NUMBER);
        p.put("SIGNED_NUMBER", "[+-]?(?:" + NUMBER + ")");
        p.put("ESCAPED_STRING", "\"(?:[^\"\\\\]|\\\\[^\\n])*\"");
        p.put("LETTER", "[A-Za-z]");
        p.put("LCASE_LETTER", "[a-z]");
        p.put("UCASE_LETTER", "[A-Z]");
        p.put("WORD", "[A-Za-z]+");
        p.put("CNAME", "[_A-Za-z][_A-Za-z0-9]*");
        p.put("WS_INLINE", "[ \\t]+");
        p.put("WS", "[ \\t\\f\\r\\n]+");
        p.put("CR", "\\r");
        p.put("LF", "\\n");
        p.put("NEWLINE", "(?:\\r?\\n)+");
        p.put("C_COMMENT", "/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/");
        p.put("CPP_COMMENT", "//[^\\n]*");
        p.put("SH_COMMENT", "#[^\\n]*");
        p.put("SQL_COMMENT", "--[^\\n]*");
        return Collections.unmodifiableMap(p);
    }
}
