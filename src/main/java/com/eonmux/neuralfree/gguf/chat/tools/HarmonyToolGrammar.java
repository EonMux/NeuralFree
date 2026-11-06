package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.HarmonyDialect;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfExclusion;
import com.eonmux.neuralfree.gguf.grammar.JsonSchemaToGbnf;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The grammars that force a harmony-family model to call a tool ({@code tool_choice: required} or a named
 * function), after llama.cpp's {@code common/parsers/gpt-oss.cpp} and {@code muse-spark.cpp}. The model may first
 * write any number of reasoning and preamble messages; the turn then must end in a call. Special tokens are matched
 * by their text, as llama.cpp's grammar matches token pieces rendered with {@code special = true}.
 *
 * <ul>
 *   <li>gpt-oss: {@code <|channel|>commentary to=functions.NAME <|constrain|>json<|message|>ARGS} (or the
 *       recipient in the role header), ARGS held to the tool's parameter schema; one call ends the turn.</li>
 *   <li>Muse: {@code  to=RECIPIENT<|message|><atem:function_calls><atem:invoke name="NAME">} with ATEM parameters,
 *       a text-valued parameter written raw and any other as JSON of its schema; with parallel calls allowed,
 *       further call messages may follow, each after {@code <|eom|><|start|>assistant}.</li>
 * </ul>
 *
 * The grammar starts where the rendered prompt leaves the model: after the assistant header, or inside a reasoning
 * body the prompt opened.
 */
final class HarmonyToolGrammar {

    private static final String START = "<|start|>assistant";
    private static final String WS = "[ \\t\\n]*";

    private HarmonyToolGrammar() {
    }

    /**
     * The forcing grammar over {@code tools}.
     *
     * @param dialect     the model's harmony dialect
     * @param tools       the callable tools (one for a named function)
     * @param parallel    whether several calls may follow one another (Muse only; a gpt-oss call ends the turn)
     * @param inReasoning whether the rendered prompt leaves the model inside a reasoning body
     */
    static ConstraintSpec forcing(HarmonyDialect dialect, List<ChatTool> tools, boolean parallel, boolean inReasoning) {
        JsonSchemaToGbnf.Composer g = JsonSchemaToGbnf.composer(null);
        String root = dialect == HarmonyDialect.MUSE ? muse(g, tools, parallel, inReasoning) : gptOss(g, tools, inReasoning);
        g.addRule("root", root);
        return ConstraintSpec.gbnf(g.gbnf());
    }

    private static String gptOss(JsonSchemaToGbnf.Composer g, List<ChatTool> tools, boolean inReasoning) {
        String body = GbnfExclusion.excluding(g::addRule, "harmony-body", Collections.singletonList("<|end|>"));
        String next = literal("<|end|>" + START);
        String aside = g.addRule("harmony-aside", "( " + literal("<|channel|>analysis") + " | "
                + literal("<|channel|>commentary") + " ) " + literal("<|message|>") + " " + body + " " + next);
        String channel = g.addRule("harmony-channel", literal("<|channel|>") + " ( " + literal("commentary")
                + " | " + literal("analysis") + " )");
        String constrain = g.addRule("harmony-constrain", "( " + WS + " " + literal("<|constrain|>")
                + "? [A-Za-z0-9_-]+ )?");
        List<String> calls = new ArrayList<>();
        for (int i = 0; i < tools.size(); i++) {
            ChatTool tool = tools.get(i);
            String args = g.addSchema("harmony-tool-" + i + "-args", ToolRouting.write(ToolRouting.parameters(tool)));
            String to = literal(" to=functions." + tool.name());
            String message = literal("<|message|>");
            calls.add(g.addRule("harmony-tool-" + i, channel + " " + to + " " + constrain + " " + message + " "
                    + args + " | " + to + " " + channel + " " + constrain + " " + message + " " + args));
        }
        String call = g.addRule("harmony-call", String.join(" | ", calls));
        String lead = inReasoning ? body + " " + next + " " : "";
        return lead + aside + "* " + call;
    }

    private static String muse(JsonSchemaToGbnf.Composer g, List<ChatTool> tools, boolean parallel,
                               boolean inReasoning) {
        String body = GbnfExclusion.excluding(g::addRule, "muse-body", Collections.singletonList("<|eom|>"));
        String next = literal("<|eom|>" + START);
        String aside = g.addRule("muse-aside", "( " + literal(" to=self") + " | " + literal(" to=commentary")
                + " ) " + literal("<|message|>") + " " + body + " " + next);
        String recipient = GbnfExclusion.excluding(g::addRule, "muse-recipient",
                Collections.singletonList("<|message|>"));
        String raw = GbnfExclusion.excluding(g::addRule, "muse-raw", Collections.singletonList("</atem:parameter>"));
        String ws = g.addRule("muse-ws", WS);
        List<String> invokes = new ArrayList<>();
        ToolSchemas schemas = ToolSchemas.of(tools);
        for (int i = 0; i < tools.size(); i++) {
            ChatTool tool = tools.get(i);
            String params = museParameters(g, tool, i, schemas, raw, ws);
            invokes.add(g.addRule("muse-tool-" + i, literal("<atem:invoke name=\"" + tool.name() + "\">") + " "
                    + ws + " " + params + " " + literal("</atem:invoke>")));
        }
        String call = g.addRule("muse-call", literal(" to=") + " " + recipient + " "
                + literal("<|message|>" + ToolParserRegistry.ATEM_CALLS) + " " + ws + " ( "
                + String.join(" | ", invokes) + " ) " + ws + " " + literal("</atem:function_calls>"));
        String calls = parallel ? call + " ( " + next + " " + call + " )*" : call;
        String lead = inReasoning ? body + " " + next + " " : "";
        return lead + aside + "* " + calls;
    }

    /** {@code (param ws)*} over one tool's parameters, or empty when it declares none. */
    private static String museParameters(JsonSchemaToGbnf.Composer g, ChatTool tool, int index, ToolSchemas schemas,
                                         String raw, String ws) {
        Object props = ToolRouting.parameters(tool).get("properties");
        if (!(props instanceof Map) || ((Map<?, ?>) props).isEmpty()) {
            return "";
        }
        List<String> params = new ArrayList<>();
        int p = 0;
        for (Map.Entry<?, ?> e : ((Map<?, ?>) props).entrySet()) {
            String key = String.valueOf(e.getKey());
            String value;
            if (schemas.textValued(tool.name(), key)) {
                value = raw;
            } else {
                value = g.addSchema("muse-tool-" + index + "-arg-" + p, ToolRouting.write(e.getValue()));
            }
            params.add(g.addRule("muse-tool-" + index + "-param-" + p, literal("<atem:parameter name=\"" + key
                    + "\">") + " " + value + " " + literal("</atem:parameter>")));
            p++;
        }
        return "( ( " + String.join(" | ", params) + " ) " + ws + " )*";
    }

    /** {@code text} as a GBNF string literal. */
    static String literal(String text) {
        return ToolRouting.literal(text);
    }
}
