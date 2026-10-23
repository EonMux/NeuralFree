package com.eonmux.neuralfree.data.transforms.text;

import com.eonmux.neuralfree.data.transforms.Transform;
import java.util.*;

public class Tokenize implements Transform<Transform.TextData> {
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;
    private final int unkTokenId;
    private final String unkToken;
    private final boolean lowercase;
    private final TokenizerType type;
    
    public enum TokenizerType {
        WHITESPACE,
        CHARACTER,
        BPE
    }
    
    public Tokenize(Map<String, Integer> vocab) {
        this(vocab, "[UNK]", true, TokenizerType.WHITESPACE);
    }
    
    public Tokenize(Map<String, Integer> vocab, String unkToken, boolean lowercase, TokenizerType type) {
        this.vocab = new HashMap<>(vocab);
        this.idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            idToToken.put(entry.getValue(), entry.getKey());
        }
        this.unkToken = unkToken != null ? unkToken : "[UNK]";
        this.unkTokenId = vocab.getOrDefault(this.unkToken, 0);
        this.lowercase = lowercase;
        this.type = type != null ? type : TokenizerType.WHITESPACE;
    }
    
    public static Tokenize whitespace(Map<String, Integer> vocab) {
        return new Tokenize(vocab, "[UNK]", true, TokenizerType.WHITESPACE);
    }
    
    public static Tokenize character(Map<String, Integer> vocab) {
        return new Tokenize(vocab, "[UNK]", true, TokenizerType.CHARACTER);
    }
    
    public static Tokenize fromVocabulary(List<String> vocabulary) {
        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("[UNK]", 0);
        vocab.put("[PAD]", 1);
        vocab.put("[CLS]", 2);
        vocab.put("[SEP]", 3);
        for (int i = 0; i < vocabulary.size(); i++) {
            vocab.put(vocabulary.get(i), i + 4);
        }
        return new Tokenize(vocab);
    }
    
    public static Tokenize simple(List<String> words) {
        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("[UNK]", 0);
        vocab.put("[PAD]", 1);
        for (int i = 0; i < words.size(); i++) {
            vocab.put(words.get(i).toLowerCase(), i + 2);
        }
        return new Tokenize(vocab);
    }
    
    @Override
    public Transform.TextData apply(Transform.TextData input) {
        String text = input.text;
        if (lowercase) {
            text = text.toLowerCase();
        }
        
        int[] tokens;
        switch (type) {
            case WHITESPACE:
                tokens = tokenizeWhitespace(text);
                break;
            case CHARACTER:
                tokens = tokenizeCharacter(text);
                break;
            case BPE:
                tokens = tokenizeBpe(text);
                break;
            default:
                tokens = tokenizeWhitespace(text);
        }
        
        return input.withTokens(tokens);
    }
    
    private int[] tokenizeWhitespace(String text) {
        String[] words = text.split("\\s+");
        int[] tokens = new int[words.length];
        
        for (int i = 0; i < words.length; i++) {
            String word = words[i].trim();
            if (word.isEmpty()) {
                tokens[i] = unkTokenId;
            } else {
                tokens[i] = vocab.getOrDefault(word, unkTokenId);
            }
        }
        
        return tokens;
    }
    
    private int[] tokenizeCharacter(String text) {
        int[] tokens = new int[text.length()];
        
        for (int i = 0; i < text.length(); i++) {
            String ch = String.valueOf(text.charAt(i));
            tokens[i] = vocab.getOrDefault(ch, unkTokenId);
        }
        
        return tokens;
    }
    
    private int[] tokenizeBpe(String text) {
        String[] words = text.split("\\s+");
        List<Integer> tokens = new ArrayList<>();
        
        for (String word : words) {
            if (word.isEmpty()) continue;
            
            List<String> subwords = bpeEncode(word);
            for (String subword : subwords) {
                tokens.add(vocab.getOrDefault(subword, unkTokenId));
            }
        }
        
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
    
    private List<String> bpeEncode(String word) {
        List<String> subwords = new ArrayList<>();
        
        for (int i = 0; i < word.length(); i++) {
            String subword = String.valueOf(word.charAt(i));
            if (i < word.length() - 1) {
                subword += "##";
            }
            
            if (vocab.containsKey(subword)) {
                subwords.add(subword);
            } else {
                subwords.add(String.valueOf(word.charAt(i)));
            }
        }
        
        return subwords;
    }
    
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String word = idToToken.get(token);
            if (word != null && !word.startsWith("[") && !word.endsWith("]")) {
                if (word.endsWith("##")) {
                    sb.append(word.substring(0, word.length() - 2));
                } else {
                    if (sb.length() > 0) {
                        sb.append(" ");
                    }
                    sb.append(word);
                }
            }
        }
        return sb.toString();
    }
    
    public int vocabSize() { return vocab.size(); }
    public int unkTokenId() { return unkTokenId; }
    public String unkToken() { return unkToken; }
    public boolean lowercase() { return lowercase; }
    public TokenizerType type() { return type; }
    public Map<String, Integer> vocab() { return new HashMap<>(vocab); }
}
