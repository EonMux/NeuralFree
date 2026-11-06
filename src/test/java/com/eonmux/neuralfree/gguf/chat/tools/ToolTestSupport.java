package com.eonmux.neuralfree.gguf.chat.tools;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.TriggeredGrammarStage;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/** Shared tool-call test helpers: a listener that records what streams, and a byte-level grammar acceptance check. */
final class ToolTestSupport {

    private ToolTestSupport() {
    }

    /** get_weather(city: string, required) and a parameterless noop. */
    static List<ChatTool> tools() {
        Map<String, Object> city = new LinkedHashMap<>();
        city.put("type", "string");
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("city", city);
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("type", "object");
        params.put("properties", props);
        params.put("required", Collections.singletonList("city"));
        params.put("additionalProperties", Boolean.FALSE);
        return Arrays.asList(ChatTool.builder("get_weather").parameters(params).build(),
                ChatTool.builder("noop").build());
    }

    /** One user message over {@link #tools()} with {@code choice}. */
    static ChatRequest chat(ChatRequest.ToolChoice choice) {
        Map<String, Object> user = new LinkedHashMap<>();
        user.put("role", "user");
        user.put("content", "weather?");
        return ChatRequest.builder().addMessage(user).tools(tools()).toolChoice(choice).build();
    }

    /** Records the content, the call names and the argument text a listener receives. */
    static final class Recorder implements ToolCallStream.Listener {
        final StringBuilder content = new StringBuilder();
        final List<String> names = new ArrayList<>();
        final List<StringBuilder> args = new ArrayList<>();

        @Override public void content(String delta) {
            content.append(delta);
        }

        @Override public void callStarted(int index, String id, String name) {
            assertEquals(names.size(), index);
            names.add(name);
            args.add(new StringBuilder());
        }

        @Override public void arguments(int index, String delta) {
            args.get(index).append(delta);
        }
    }

    /** Byte tokens 1..127, EOG 0. */
    static final class ByteVocab implements SamplerVocab {
        @Override public int vocabSize() { return 128; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) id}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] b = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[b.length];
            for (int i = 0; i < b.length; i++) {
                ids[i] = b[i] & 0xFF;
            }
            return ids;
        }
    }

    /** Whether the constraint lets greedy decoding write the ASCII {@code text} byte by byte and then end. */
    static boolean accepts(CompiledConstraint c, String text) {
        ByteVocab vocab = new ByteVocab();
        SamplerStage stage = c.gated() ? new TriggeredGrammarStage(c.grammar(), c.triggers(), vocab)
                : new GrammarStage(c.grammar(), vocab);
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.singletonList(stage), vocab);
        for (int i = 0; i <= text.length(); i++) {
            int want = i < text.length() ? text.charAt(i) : 0;
            float[] logits = new float[128];
            Arrays.fill(logits, -100f);
            logits[want] = 10f;
            int token;
            try {
                token = sampler.sample(logits);
            } catch (RuntimeException e) {
                return false;
            }
            if (token != want) {
                return false;
            }
            if (want != 0) {
                sampler.accept(token);
            }
        }
        return true;
    }
}
