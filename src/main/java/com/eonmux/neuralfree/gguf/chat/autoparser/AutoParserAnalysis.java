package com.eonmux.neuralfree.gguf.chat.autoparser;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The structure a chat template renders, derived by differential analysis ({@link AutoParserAnalyzer}) the way
 * llama.cpp's autoparser derives it ({@code common/chat-diff-analyzer.cpp}): where a reasoning block, the content
 * and the tool calls sit, and which markers delimit them.
 *
 * <p>The analysis is the whole result of the probe renders: {@link Reasoning} carries the reasoning mode and its
 * start and end tags, {@link Content} the content markers, {@link Tools} the tool-call format and every marker of
 * it, and the user and assistant start markers are read from the same renders. Immutable.
 */
public final class AutoParserAnalysis {

    /** How the template renders reasoning ({@code reasoning_mode}). */
    public enum ReasoningMode {
        /** No reasoning markers were found. */
        NONE,
        /** A leading {@code start ... end} reasoning block; the start may be empty for a delimiter-style format. */
        TAG_BASED,
        /** Reasoning appears only when a tool call follows. */
        TOOLS_ONLY
    }

    /** How the template wraps content ({@code content_mode}). */
    public enum ContentMode {
        /** No content markers. */
        PLAIN,
        /** Content is always wrapped in markers. */
        ALWAYS_WRAPPED,
        /** Content is wrapped only when reasoning is present. */
        WRAPPED_WITH_REASONING
    }

    /** The tool-call syntax the template writes ({@code tool_format}). */
    public enum ToolFormat {
        /** No tool calls were found. */
        NONE,
        /** A JSON object: {@code {"name": ..., "arguments": ...}}. */
        JSON_NATIVE,
        /** A tag around the function name, with a JSON argument object. */
        TAG_WITH_JSON,
        /** A tag around the function name, with tagged argument names and values. */
        TAG_WITH_TAGGED
    }

    /** Where a tool-call id sits relative to the function name and arguments ({@code call_id_position}). */
    public enum CallIdPosition {
        NONE, PRE_FUNC_NAME, BETWEEN_FUNC_AND_ARGS, POST_ARGS
    }

    /** The reasoning structure. Immutable. */
    public static final class Reasoning {
        private final ReasoningMode mode;
        private final String start;
        private final String end;

        Reasoning(ReasoningMode mode, String start, String end) {
            this.mode = Objects.requireNonNull(mode, "mode");
            this.start = Objects.requireNonNull(start, "start");
            this.end = Objects.requireNonNull(end, "end");
        }

        public ReasoningMode mode() { return mode; }
        public String start() { return start; }
        public String end() { return end; }

        boolean present() { return mode != ReasoningMode.NONE; }

        @Override public String toString() {
            return mode + "[" + start + ".." + end + "]";
        }
    }

    /** The content structure. Immutable. */
    public static final class Content {
        private final ContentMode mode;
        private final String start;
        private final String end;

        Content(ContentMode mode, String start, String end) {
            this.mode = Objects.requireNonNull(mode, "mode");
            this.start = Objects.requireNonNull(start, "start");
            this.end = Objects.requireNonNull(end, "end");
        }

        public ContentMode mode() { return mode; }
        public String start() { return start; }
        public String end() { return end; }

        @Override public String toString() {
            return mode + "[" + start + ".." + end + "]";
        }
    }

    /** The tool-call structure ({@code analyze_tools}). Immutable. */
    public static final class Tools {
        private final ToolFormat mode;
        private final String sectionStart;
        private final String sectionEnd;
        private final String perCallStart;
        private final String perCallEnd;
        private final boolean funNameIsKey;
        private final boolean toolsArrayWrapped;
        private final String functionField;
        private final String nameField;
        private final String argsField;
        private final String idField;
        private final String genIdField;
        private final List<String> parameterOrder;
        private final String functionNamePrefix;
        private final String functionNameSuffix;
        private final String functionArgsSeparator;
        private final String functionClose;
        private final String argsStart;
        private final String argsEnd;
        private final String argNamePrefix;
        private final String argNameSuffix;
        private final String argValuePrefix;
        private final String argValueSuffix;
        private final String argSeparator;
        private final CallIdPosition callIdPosition;
        private final String callIdPrefix;
        private final String callIdSuffix;
        private final boolean openaiWrapperTrigger;
        private final boolean tolerateIntertagWhitespace;

        Tools(ToolFormat mode, String sectionStart, String sectionEnd, String perCallStart, String perCallEnd,
              boolean funNameIsKey, boolean toolsArrayWrapped, String functionField,
              String nameField, String argsField, String idField, String genIdField, List<String> parameterOrder,
              String functionNamePrefix, String functionNameSuffix, String functionArgsSeparator, String functionClose,
              String argsStart, String argsEnd, String argNamePrefix, String argNameSuffix, String argValuePrefix,
              String argValueSuffix, String argSeparator, CallIdPosition callIdPosition, String callIdPrefix,
              String callIdSuffix, boolean openaiWrapperTrigger, boolean tolerateIntertagWhitespace) {
            this.mode = Objects.requireNonNull(mode, "mode");
            this.sectionStart = nz(sectionStart);
            this.sectionEnd = nz(sectionEnd);
            this.perCallStart = nz(perCallStart);
            this.perCallEnd = nz(perCallEnd);
            this.funNameIsKey = funNameIsKey;
            this.toolsArrayWrapped = toolsArrayWrapped;
            this.functionField = nz(functionField);
            this.nameField = nz(nameField);
            this.argsField = nz(argsField);
            this.idField = nz(idField);
            this.genIdField = nz(genIdField);
            this.parameterOrder = Collections.unmodifiableList(new ArrayList<>(parameterOrder));
            this.functionNamePrefix = nz(functionNamePrefix);
            this.functionNameSuffix = nz(functionNameSuffix);
            this.functionArgsSeparator = nz(functionArgsSeparator);
            this.functionClose = nz(functionClose);
            this.argsStart = nz(argsStart);
            this.argsEnd = nz(argsEnd);
            this.argNamePrefix = nz(argNamePrefix);
            this.argNameSuffix = nz(argNameSuffix);
            this.argValuePrefix = nz(argValuePrefix);
            this.argValueSuffix = nz(argValueSuffix);
            this.argSeparator = nz(argSeparator);
            this.callIdPosition = Objects.requireNonNull(callIdPosition, "callIdPosition");
            this.callIdPrefix = nz(callIdPrefix);
            this.callIdSuffix = nz(callIdSuffix);
            this.openaiWrapperTrigger = openaiWrapperTrigger;
            this.tolerateIntertagWhitespace = tolerateIntertagWhitespace;
        }

        private static String nz(String s) {
            return s == null ? "" : s;
        }

        public ToolFormat mode() { return mode; }
        public String sectionStart() { return sectionStart; }
        public String sectionEnd() { return sectionEnd; }
        public String perCallStart() { return perCallStart; }
        public String perCallEnd() { return perCallEnd; }
        public boolean funNameIsKey() { return funNameIsKey; }
        public boolean toolsArrayWrapped() { return toolsArrayWrapped; }
        public String functionField() { return functionField; }
        public String nameField() { return nameField; }
        public String argsField() { return argsField; }
        public String idField() { return idField; }
        public String genIdField() { return genIdField; }
        public List<String> parameterOrder() { return parameterOrder; }
        public String functionNamePrefix() { return functionNamePrefix; }
        public String functionNameSuffix() { return functionNameSuffix; }
        public String functionArgsSeparator() { return functionArgsSeparator; }
        public String functionClose() { return functionClose; }
        public String argsStart() { return argsStart; }
        public String argsEnd() { return argsEnd; }
        public String argNamePrefix() { return argNamePrefix; }
        public String argNameSuffix() { return argNameSuffix; }
        public String argValuePrefix() { return argValuePrefix; }
        public String argValueSuffix() { return argValueSuffix; }
        public String argSeparator() { return argSeparator; }
        public CallIdPosition callIdPosition() { return callIdPosition; }
        public String callIdPrefix() { return callIdPrefix; }
        public String callIdSuffix() { return callIdSuffix; }

        /**
         * Whether the template's tool instruction makes the model emit the OpenAI function wrapper, whose
         * leading {@code "type": "function"} field the grammar and the lazy trigger admit.
         */
        public boolean openaiWrapperTrigger() { return openaiWrapperTrigger; }

        /** Whether whitespace is tolerated between a call's tags (Laguna, Bailing V3). */
        public boolean tolerateIntertagWhitespace() { return tolerateIntertagWhitespace; }

        /** Whether any tool-call syntax was found. */
        public boolean present() {
            return mode != ToolFormat.NONE;
        }

        /** The literal a tool call opens with: the section opener, else the per-call opener. */
        public String callOpener() {
            return !sectionStart.isEmpty() ? sectionStart : perCallStart;
        }

        /** The name field path, with the function wrapper applied when the call nests its fields. */
        public String effectiveNameField() {
            return fieldPath(functionField, nameField);
        }

        /** The arguments field path, with the function wrapper applied when the call nests its fields. */
        public String effectiveArgsField() {
            return fieldPath(functionField, argsField);
        }

        private static String fieldPath(String function, String field) {
            if (function.isEmpty() || function.equals("function") || field.isEmpty() || field.contains(".")) {
                return field;
            }
            return function + "." + field;
        }

        @Override public String toString() {
            return mode + "{section=" + sectionStart + ".." + sectionEnd + ", call=" + perCallStart + ".."
                    + perCallEnd + ", name=" + functionNamePrefix + "<name>" + functionNameSuffix
                    + ", args=" + argsStart + ".." + argsEnd + ", tags=" + argNamePrefix + "<key>" + argNameSuffix
                    + "/" + argValuePrefix + "<value>" + argValueSuffix + ", close=" + functionClose + "}";
        }
    }

    private final String userStart;
    private final String assistantStart;
    private final Reasoning reasoning;
    private final Content content;
    private final Tools tools;
    private final List<String> preservedTokens;
    private final List<String> additionalStops;

    AutoParserAnalysis(String userStart, String assistantStart, Reasoning reasoning, Content content, Tools tools,
                       List<String> preservedTokens, List<String> additionalStops) {
        this.userStart = nz(userStart);
        this.assistantStart = nz(assistantStart);
        this.reasoning = Objects.requireNonNull(reasoning, "reasoning");
        this.content = Objects.requireNonNull(content, "content");
        this.tools = Objects.requireNonNull(tools, "tools");
        this.preservedTokens = Collections.unmodifiableList(new ArrayList<>(preservedTokens));
        this.additionalStops = Collections.unmodifiableList(new ArrayList<>(additionalStops));
    }

    private static String nz(String s) {
        return s == null ? "" : s;
    }

    /** The marker a user message starts with, or empty. */
    public String userStart() { return userStart; }
    /** The marker an assistant message starts with, or empty. */
    public String assistantStart() { return assistantStart; }
    /** The reasoning structure. */
    public Reasoning reasoning() { return reasoning; }
    /** The content structure. */
    public Content content() { return content; }
    /** The tool-call structure. */
    public Tools tools() { return tools; }
    /** Every marker the analysis found, for a tokenizer's preserved-token list. */
    public List<String> preservedTokens() { return preservedTokens; }

    /**
     * Literal stop strings the template's format needs in addition to the tokenizer's end-of-generation tokens
     * (Laguna's {@code </assistant>}, which the model can spell out after a tool call).
     */
    public List<String> additionalStops() { return additionalStops; }

    /** Whether the analysis found a tool-call syntax. */
    public boolean hasTools() {
        return tools.present();
    }

    @Override
    public String toString() {
        return "AutoParserAnalysis{user=" + userStart + ", assistant=" + assistantStart + ", reasoning=" + reasoning
                + ", content=" + content + ", tools=" + tools + "}";
    }
}
