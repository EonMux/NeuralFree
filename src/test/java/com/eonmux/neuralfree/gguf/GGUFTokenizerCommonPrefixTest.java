package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Assume;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link GGUFTokenizer#commonPrefixLength} against the common prefix of the two whole encodings, for BPE, SPM, UGM
 * and WPM vocabularies: chat-shaped texts whose last message differs (the probe of the server's last-message
 * boundary), texts that differ anywhere, equal texts, one text a prefix of the other, and empty texts. With
 * llama.cpp's vocabulary fixtures present, the same over their real vocabularies and inputs.
 */
public class GGUFTokenizerCommonPrefixTest {

    private static final String SP = "▁";
    private static final String[] WORDS = {"hi", "the", "a", "b", "ab", "ba", " ", "  ", "\n", "é", "叫", "<|im_end|>",
            "<|im_start|>", "<s>", "[CLS]", "x"};

    @Test
    public void syntheticVocabulariesAgreeWithTheWholeEncodings() {
        Random rnd = new Random(12);
        for (GGUFTokenizer tokenizer : List.of(new GGUFTokenizer(bpe()), new GGUFTokenizer(spm()),
                new GGUFTokenizer(ugm()), new GGUFTokenizer(wpm()))) {
            for (int t = 0; t < 400; t++) {
                String[] pair = pair(rnd);
                check(tokenizer, pair[0], pair[1]);
            }
            check(tokenizer, "", "");
            check(tokenizer, "", "<|im_start|>hi");
            check(tokenizer, "<|im_start|>hi", "");
            check(tokenizer, "<|im_start|>user\nhi the", "<|im_start|>user\nhi the");
            check(tokenizer, "<|im_start|>user\nhi", "<|im_start|>user\nhi the");
        }
    }

    @Test
    public void llamaCppVocabularyFixturesAgreeWithTheWholeEncodings() throws Exception {
        File models = new File(System.getProperty("user.home"), "Desktop/Projects/llama.cpp/models");
        String[] names = {"qwen2", "llama-spm", "llama-bpe", "phi-3", "bert-bge", "gemma-4", "command-r"};
        int checked = 0;
        for (String name : names) {
            File vocab = new File(models, "ggml-vocab-" + name + ".gguf");
            File inputs = new File(models, "ggml-vocab-" + name + ".gguf.inp");
            if (!vocab.isFile() || !inputs.isFile()) {
                continue;
            }
            GGUFTokenizer tokenizer = new GGUFTokenizer(ModelSources.open(vocab.getPath()));
            String[] texts = new String(java.nio.file.Files.readAllBytes(inputs.toPath()),
                    java.nio.charset.StandardCharsets.UTF_8).split("\n__ggml_vocab_test__\n");
            Random rnd = new Random(name.hashCode());
            for (int t = 0; t < 200; t++) {
                String turns = chat(texts, rnd, 1 + rnd.nextInt(4));
                String last = texts[rnd.nextInt(texts.length)];
                check(tokenizer, turns + "<|im_start|>user\n" + last + "<|im_end|>\n<|im_start|>assistant\n",
                        turns + "<|im_start|>user\n\u0001\u0002boundary<|im_end|>\n<|im_start|>assistant\n");
                checked++;
            }
        }
        Assume.assumeTrue("no llama.cpp vocabulary fixtures here", checked > 0);
    }

    @Test
    public void idsThatAreNotTheFirstTextsEncodingAreRefused() {
        GGUFTokenizer tokenizer = new GGUFTokenizer(bpe());
        int[] other = tokenizer.encode("<|im_start|>the", false, true);
        assertThrows(IllegalArgumentException.class,
                () -> tokenizer.commonPrefixLength("<|im_start|>hi hi", other, "<|im_start|>hi"));
    }

    private static void check(GGUFTokenizer tokenizer, String a, String b) {
        int[] encodedA = tokenizer.encode(a, false, true);
        int[] encodedB = tokenizer.encode(b, false, true);
        int expected = 0;
        while (expected < Math.min(encodedA.length, encodedB.length) && encodedA[expected] == encodedB[expected]) {
            expected++;
        }
        assertEquals(tokenizer.vocab().model() + " '" + a + "' vs '" + b + "'", expected,
                tokenizer.commonPrefixLength(a, encodedA, b));
    }

    /** Two chat-shaped texts sharing their turns but the last message's content, or two texts differing anywhere. */
    private static String[] pair(Random rnd) {
        String turns = chat(WORDS, rnd, rnd.nextInt(4));
        String tail = "<|im_end|>\n<|im_start|>assistant\n";
        if (rnd.nextInt(4) == 0) {
            String a = turns + words(rnd);
            String b = a.substring(0, rnd.nextInt(a.length() + 1)) + words(rnd);
            return new String[]{a, b};
        }
        return new String[]{turns + "<|im_start|>user\n" + words(rnd) + tail,
                turns + "<|im_start|>user\n" + (rnd.nextBoolean() ? "\u0001marker" : words(rnd)) + tail};
    }

    private static String chat(String[] contents, Random rnd, int turns) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < turns; i++) {
            out.append(i % 2 == 0 ? "<|im_start|>user\n" : "<|im_start|>assistant\n")
                    .append(contents[rnd.nextInt(contents.length)]).append("<|im_end|>\n");
        }
        return out.toString();
    }

    private static String words(Random rnd) {
        StringBuilder out = new StringBuilder();
        int n = rnd.nextInt(8);
        for (int i = 0; i < n; i++) {
            out.append(WORDS[rnd.nextInt(WORDS.length)]);
        }
        return out.toString();
    }

    // ------------------------------------------------------------------ synthetic vocabularies

    private static List<String> chatTokens() {
        return List.of("<|im_start|>", "<|im_end|>");
    }

    private static TokenizerVocab bpe() {
        List<String> tokens = new ArrayList<>(chatTokens());
        tokens.addAll(List.of("h", "i", "t", "e", "a", "b", "x", "u", "s", "r", "n", "Ċ", "Ġ", "hi", "th", "the",
                "Ġthe", "ab", "Ġh", "Ġhi", "us", "user"));
        List<String> merges = List.of("h i", "t h", "th e", "Ġ the", "a b", "Ġ h", "Ġh i", "u s", "us e");
        return vocab("gpt2", "gpt-2", tokens, merges, null);
    }

    private static TokenizerVocab spm() {
        List<String> tokens = new ArrayList<>(List.of("<unk>", "<s>", "</s>"));
        tokens.addAll(chatTokens());
        tokens.addAll(List.of(SP, "h", "i", "t", "e", "a", "b", "x", "\n", SP + "h", SP + "hi", "hi", "the",
                SP + "the", "ab", "user", "assistant"));
        Map<String, Object> md = metadata("llama", null, tokens, null, 0);
        md.put("tokenizer.ggml.bos_token_id", 1);
        md.put("tokenizer.ggml.eos_token_id", 2);
        return TokenizerVocab.fromMetadata(md);
    }

    private static TokenizerVocab ugm() {
        List<String> tokens = new ArrayList<>(List.of("<pad>", "</s>", "<unk>"));
        tokens.addAll(chatTokens());
        tokens.addAll(List.of(SP, "h", "i", "t", "e", "a", "b", "x", "\n", SP + "hi", "hi", "the", SP + "the",
                "ab", "user"));
        return vocab("t5", null, tokens, null, 2);
    }

    private static TokenizerVocab wpm() {
        List<String> tokens = new ArrayList<>(List.of("[PAD]", "[UNK]", "[CLS]", "[SEP]"));
        tokens.addAll(chatTokens());
        tokens.addAll(List.of("h", "i", "t", "e", "a", "b", "x", "hi", "the", "ab", "##i", "##e", "##b", "##a",
                "user", "<", ">", "|", "_"));
        return vocab("bert", null, tokens, null, 1);
    }

    private static TokenizerVocab vocab(String model, String pre, List<String> tokens, List<String> merges,
                                        Integer unk) {
        return TokenizerVocab.fromMetadata(metadata(model, pre, tokens, merges, unk));
    }

    /** The metadata of {@code tokens}: the chat markers and bracketed names control, {@code unk} unknown. */
    private static Map<String, Object> metadata(String model, String pre, List<String> tokens, List<String> merges,
                                                Integer unk) {
        List<Object> types = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        for (int i = 0; i < tokens.size(); i++) {
            String t = tokens.get(i);
            boolean control = t.startsWith("<|") || t.equals("<s>") || t.equals("</s>") || t.equals("<pad>")
                    || (t.startsWith("[") && t.length() > 2);
            types.add(unk != null && i == unk || t.equals("<unk>") ? 2 : control ? 3 : 1);
            scores.add(-(float) t.length() * 0.5f - i * 0.01f);
        }
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, model);
        if (pre != null) {
            md.put(TokenizerVocab.KEY_PRE, pre);
        }
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_SCORES, scores);
        if (merges != null) {
            md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>(merges));
        }
        if (unk != null) {
            md.put("tokenizer.ggml.unknown_token_id", unk);
        }
        return md;
    }
}
