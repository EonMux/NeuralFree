package com.eonmux.neuralfree.gguf.chat.autoparser;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * llama.cpp's per-template patches of the differential analysis ({@code common/chat-diff-analyzer.cpp}
 * {@code workarounds}): templates whose markers the probes cannot derive, because they render markers the model
 * never writes, split content from reasoning only in their instruction text, or write a call syntax the probes
 * misread. Each patch applies after the probes and rewrites the derived structure, the preserved markers and, where
 * the format needs them, the literal stop strings.
 *
 * <p>The patches are matched on exact source fragments, as llama.cpp matches them, so a template that is already
 * understood by a hand-written registry row never reaches here: the autoparser is the last resort.
 */
final class AutoParserWorkarounds {

    private static final Logger LOGGER = LogManager.getLogger(AutoParserWorkarounds.class);

    private AutoParserWorkarounds() {
    }

    /** Applies every patch whose trigger {@code source} contains. */
    static void apply(String source, AutoParserAnalyzer analysis) {
        if (source == null || source.isEmpty()) {
            return;
        }
        oldQwenThinking(source, analysis);
        granite33(source, analysis);
        cohereCommandR(source, analysis);
        functionary(source, analysis);
        deepseekR1Distill(source, analysis);
        nemotronNanoV2(source, analysis);
        fireworks(source, analysis);
        solarOpen(source, analysis);
        apriel16(source, analysis);
        openaiWrapper(source, analysis);
        laguna(source, analysis);
        bailingV3(source, analysis);
    }

    /** Old Qwen and DeepSeek templates that show reasoning but never render it. */
    private static void oldQwenThinking(String source, AutoParserAnalyzer a) {
        if (source.contains("content.split('</think>')")
                && !source.contains("reasoning_content")
                && !source.contains("<SPECIAL_12>")
                && a.reasoningMode == AutoParserAnalysis.ReasoningMode.NONE) {
            a.reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
            a.reasoningStart = "<think>";
            a.reasoningEnd = "</think>";
            LOGGER.debug("[Patch: old Qwen/Deepseek thinking template]");
        }
    }

    /** Granite 3.3: reasoning in think tags, the answer in response tags. */
    private static void granite33(String source, AutoParserAnalyzer a) {
        if (source.contains("Write your thoughts between <think></think> and write your response between "
                + "<response></response>")) {
            a.reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
            a.reasoningStart = "<think>";
            a.reasoningEnd = "</think>";
            a.contentMode = AutoParserAnalysis.ContentMode.WRAPPED_WITH_REASONING;
            a.contentStart = "<response>";
            a.contentEnd = "</response>";
            LOGGER.debug("[Patch: Granite 3.3]");
        }
    }

    /** Cohere Command R+: the answer is wrapped in the chatbot and end-of-turn tokens. */
    private static void cohereCommandR(String source, AutoParserAnalyzer a) {
        if (source.contains("<|CHATBOT_TOKEN|>") && source.contains("<|END_OF_TURN_TOKEN|>")
                && a.contentStart.isEmpty()) {
            a.contentMode = AutoParserAnalysis.ContentMode.ALWAYS_WRAPPED;
            a.contentStart = "<|CHATBOT_TOKEN|>";
            a.contentEnd = "<|END_OF_TURN_TOKEN|>";
            a.userStart = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>";
            LOGGER.debug("[Patch: Cohere Command R+]");
        }
    }

    /** Functionary: a call is a bare {@code <function=NAME>...</function>} with no section delimiters. */
    private static void functionary(String source, AutoParserAnalyzer a) {
        if (source.contains("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                + "\"code_interpreter\") | list | length > 0")) {
            a.contentMode = AutoParserAnalysis.ContentMode.PLAIN;
            a.contentEnd = "";
            a.functionNamePrefix = "";
            a.sectionStart = "";
            a.sectionEnd = "";
            a.perCallStart = "<function=";
            a.perCallEnd = "</function>";
            a.functionClose = "";
            a.preservedOverride = new ArrayList<>(Arrays.asList("<|eot_id|>", "<|eom_id|>", "<function=", ">",
                    "</function>"));
            LOGGER.debug("[Patch: Functionary 3.1]");
        }
    }

    /** DeepSeek-R1-Distill-Qwen: its call markers differ from the probe-derived ones. */
    private static void deepseekR1Distill(String source, AutoParserAnalyzer a) {
        if (source.contains("{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + "
                + "'<｜tool▁sep｜>'")) {
            a.sectionStart = "<｜tool▁calls▁begin｜>";
            a.sectionEnd = "<｜tool▁calls▁end｜>";
            a.perCallStart = "<｜tool▁call▁begin｜>function";
            a.functionNamePrefix = "<｜tool▁sep｜>";
            a.perCallEnd = "<｜tool▁call▁end｜>";
            a.functionClose = "```";
            LOGGER.debug("[Patch: DeepSeek-R1-Distill-Qwen]");
        }
    }

    /** Nemotron Nano v2: the markers are special tokens the probes cannot read. */
    private static void nemotronNanoV2(String source, AutoParserAnalyzer a) {
        if (source.contains("<SPECIAL_10>") && source.contains("<SPECIAL_11>")
                && source.contains("<SPECIAL_12>") && source.contains("<TOOL_RESPONSE>")) {
            a.toolMode = AutoParserAnalysis.ToolFormat.JSON_NATIVE;
            a.sectionStart = "";
            a.sectionEnd = "";
            a.perCallStart = "<TOOLCALL>";
            a.perCallEnd = "</TOOLCALL>";
            a.toolsArrayWrapped = true;
            a.contentMode = AutoParserAnalysis.ContentMode.PLAIN;
            a.contentStart = "";
            a.contentEnd = "";
            a.reasoningMode = AutoParserAnalysis.ReasoningMode.TAG_BASED;
            a.reasoningStart = "<think>\n";
            a.reasoningEnd = "</think>";
            a.assistantStart = "<SPECIAL_11>Assistant";
            a.userStart = "<SPECIAL_11>User";
            a.preservedOverride = new ArrayList<>(Arrays.asList("<SPECIAL_11>", "</think>", "<TOOLCALL>",
                    "</TOOLCALL>"));
            LOGGER.debug("[Patch: Nemotron Nano v2]");
        }
    }

    /** Fireworks: the header markers the probe reads are replaced by the actual ones. */
    private static void fireworks(String source, AutoParserAnalyzer a) {
        if (source.contains("{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n'"
                + " + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}")) {
            a.assistantStart = "<|start_header_id|>assistant<|end_header_id|>";
            a.userStart = "<|start_header_id|>user<|end_header_id|>";
            LOGGER.debug("[Patch: Fireworks v2]");
        }
    }

    /** Solar Open: the assistant header ends before the think marker it always opens. */
    private static void solarOpen(String source, AutoParserAnalyzer a) {
        if (source.contains("<|begin|>assistant<|think|><|end|>")) {
            a.assistantStart = "<|begin|>assistant";
            LOGGER.debug("[Patch: Solar Open]");
        }
    }

    /** Apriel 1.6: its role markers are not header-like markers. */
    private static void apriel16(String source, AutoParserAnalyzer a) {
        if (source.contains("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text")) {
            a.userStart = "<|begin_user|>";
            a.assistantStart = "<|begin_assistant|>";
            LOGGER.debug("[Patch: Apriel 1.6]");
        }
    }

    /** A JSON {@code {name, parameters}} instruction whose model emits the OpenAI function wrapper. */
    private static void openaiWrapper(String source, AutoParserAnalyzer a) {
        if (source.contains("Respond in the format {\"name\": function name")
                && source.contains("Do not use variables.")) {
            a.openaiWrapperTrigger = true;
            LOGGER.debug("[Patch: JSON name/parameters tool instruction]");
        }
    }

    /**
     * Laguna (poolside): the template formats its markers with whitespace the model does not emit, and the
     * {@code </assistant>} token after a tool call can be spelled out as text.
     */
    private static void laguna(String source, AutoParserAnalyzer a) {
        if (source.contains("laguna_glm_thinking")) {
            a.reasoningStart = a.reasoningStart.trim();
            a.reasoningEnd = a.reasoningEnd.trim();
            a.argValuePrefix = a.argValuePrefix.trim();
            a.argValueSuffix = a.argValueSuffix.trim();
            a.argSeparator = a.argSeparator.trim();
            a.tolerateIntertagWhitespace = true;
            if (!a.additionalStops.contains("</assistant>")) {
                a.additionalStops.add("</assistant>");
            }
            LOGGER.debug("[Patch: Laguna]");
        }
    }

    /** Bailing V3: the value closer carries whitespace the model does not emit. */
    private static void bailingV3(String source, AutoParserAnalyzer a) {
        if (source.contains("Bailing V3 chat template")) {
            a.argValueSuffix = a.argValueSuffix.trim();
            a.tolerateIntertagWhitespace = true;
            LOGGER.debug("[Patch: Bailing V3]");
        }
    }
}
