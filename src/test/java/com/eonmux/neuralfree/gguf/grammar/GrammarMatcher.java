package com.eonmux.neuralfree.gguf.grammar;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.GrammarStage;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;

/**
 * Whether a grammar matches a whole text: greedy decoding over byte tokens must write it and then end. {@link #matches}
 * takes ASCII text over tokens 1..127, {@link #matchesUtf8} any text as UTF-8 bytes (NUL included) over tokens
 * 1..256 (byte value + 1); token 0 is EOG in both.
 */
public final class GrammarMatcher {

    private GrammarMatcher() {
    }

    public static boolean matchesUtf8(Grammar grammar, String text) {
        SamplerVocab vocab = new Utf8Bytes();
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.singletonList(new GrammarStage(grammar, vocab)), vocab);
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (int i = 0; i <= bytes.length; i++) {
            int want = i < bytes.length ? (bytes[i] & 0xFF) + 1 : 0;
            float[] logits = new float[257];
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

    public static boolean matches(Grammar grammar, String text) {
        SamplerVocab vocab = new Bytes();
        Sampler sampler = Sampler.create(SamplerConfig.greedyDecoding(), new int[0],
                Collections.singletonList(new GrammarStage(grammar, vocab)), vocab);
        for (int i = 0; i <= text.length(); i++) {
            int want = i < text.length() ? text.charAt(i) : 0;
            if (want >= 128) {
                throw new IllegalArgumentException("ASCII text only");
            }
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

    /** Byte tokens 1..127, EOG 0. */
    private static final class Bytes implements SamplerVocab {
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

    /** Byte tokens 1..256 for the byte values 0..255, EOG 0. */
    private static final class Utf8Bytes implements SamplerVocab {
        @Override public int vocabSize() { return 257; }
        @Override public boolean isEog(int id) { return id == 0; }
        @Override public int eotToken() { return 0; }
        @Override public int eosToken() { return 0; }
        @Override public byte[] piece(int id) { return id <= 0 ? new byte[0] : new byte[]{(byte) (id - 1)}; }
        @Override public String pieceText(int id) { return new String(piece(id), StandardCharsets.UTF_8); }
        @Override public int[] tokenize(String text) {
            byte[] b = text.getBytes(StandardCharsets.UTF_8);
            int[] ids = new int[b.length];
            for (int i = 0; i < b.length; i++) {
                ids[i] = (b[i] & 0xFF) + 1;
            }
            return ids;
        }
    }
}
