package com.eonmux.neuralfree.data;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.Tensor;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class TextFileDataset implements Dataset<TextFileDataset.Sample> {
    private final Path filePath;
    private final ComputeBackend backend;
    private final Tokenizer tokenizer;
    private final int maxSequenceLength;
    private final int paddingTokenId;
    private final boolean paddingEnabled;

    private final List<String> lines;
    private final List<int[]> tokenizedLines;
    private final Map<String, Integer> vocabulary;
    private final Map<Integer, String> inverseVocabulary;
    private final int vocabularySize;

    private static final String UNKNOWN_TOKEN = "<UNK>";
    private static final String PADDING_TOKEN = "<PAD>";
    private static final int UNKNOWN_TOKEN_ID = 0;
    private static final int DEFAULT_PADDING_TOKEN_ID = 1;

    private TextFileDataset(Builder builder) {
        this.filePath = Objects.requireNonNull(builder.filePath, "File path cannot be null");
        this.backend = Objects.requireNonNull(builder.backend, "Backend cannot be null");
        this.tokenizer = builder.tokenizer != null ? builder.tokenizer : new WhitespaceTokenizer();
        this.maxSequenceLength = builder.maxSequenceLength;
        this.paddingTokenId = builder.paddingTokenId;
        this.paddingEnabled = builder.paddingEnabled;

        if (!Files.isRegularFile(filePath)) {
            throw new IllegalArgumentException("File path must be a regular file: " + filePath);
        }

        this.lines = new ArrayList<>();
        this.tokenizedLines = new ArrayList<>();
        this.vocabulary = new HashMap<>();
        this.inverseVocabulary = new HashMap<>();

        vocabulary.put(UNKNOWN_TOKEN, UNKNOWN_TOKEN_ID);
        inverseVocabulary.put(UNKNOWN_TOKEN_ID, UNKNOWN_TOKEN);
        vocabulary.put(PADDING_TOKEN, this.paddingTokenId);
        inverseVocabulary.put(this.paddingTokenId, PADDING_TOKEN);

        loadFile();

        this.vocabularySize = vocabulary.size();
    }

    private void loadFile() {
        try (BufferedReader reader = Files.newBufferedReader(filePath)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue;

                lines.add(line);
                String[] tokens = tokenizer.tokenize(line);
                int[] tokenIds = new int[tokens.length];

                for (int i = 0; i < tokens.length; i++) {
                    tokenIds[i] = getTokenId(tokens[i]);
                }

                tokenizedLines.add(tokenIds);
            }

            if (lines.isEmpty()) {
                throw new IllegalStateException("Text file has no non-empty lines: " + filePath);
            }

        } catch (IOException e) {
            throw new RuntimeException("Failed to read text file: " + filePath, e);
        }
    }

    private int getTokenId(String token) {
        return vocabulary.computeIfAbsent(token, 
            k -> {
                int id = vocabulary.size();
                inverseVocabulary.put(id, token);
                return id;
            });
    }

    @Override
    public int size() {
        return lines.size();
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= lines.size()) {
            throw new IndexOutOfBoundsException(
                "Index " + index + " out of bounds for dataset size " + lines.size());
        }

        String line = lines.get(index);
        int[] tokenIds = tokenizedLines.get(index);

        int sequenceLength = maxSequenceLength > 0 
            ? Math.min(tokenIds.length, maxSequenceLength)
            : tokenIds.length;

        int[] processedIds;
        if (paddingEnabled && maxSequenceLength > 0) {
            processedIds = new int[maxSequenceLength];
            int copyLength = Math.min(tokenIds.length, maxSequenceLength);
            System.arraycopy(tokenIds, 0, processedIds, 0, copyLength);
            for (int i = copyLength; i < maxSequenceLength; i++) {
                processedIds[i] = paddingTokenId;
            }
            sequenceLength = maxSequenceLength;
        } else {
            processedIds = new int[sequenceLength];
            System.arraycopy(tokenIds, 0, processedIds, 0, sequenceLength);
        }

        Tensor tokenTensor = createTokenTensor(processedIds);
        float[] attentionMask = createAttentionMask(sequenceLength, processedIds.length);

        return new Sample(tokenTensor, line, processedIds, attentionMask);
    }

    private Tensor createTokenTensor(int[] tokenIds) {
        float[] floatIds = new float[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            floatIds[i] = tokenIds[i];
        }
        return Tensor.fromArray(backend, floatIds);
    }

    private float[] createAttentionMask(int actualLength, int paddedLength) {
        float[] mask = new float[paddedLength];
        for (int i = 0; i < actualLength && i < paddedLength; i++) {
            mask[i] = 1f;
        }
        return mask;
    }

    public List<String> lines() {
        return Collections.unmodifiableList(lines);
    }

    public int vocabularySize() {
        return vocabularySize;
    }

    public String getToken(int tokenId) {
        return inverseVocabulary.getOrDefault(tokenId, UNKNOWN_TOKEN);
    }

    public int getVocabularyTokenId(String token) {
        return vocabulary.getOrDefault(token, UNKNOWN_TOKEN_ID);
    }

    public Map<String, Integer> vocabulary() {
        return Collections.unmodifiableMap(vocabulary);
    }

    public int maxSequenceLength() {
        return maxSequenceLength;
    }

    @Override
    public String toString() {
        return "TextFileDataset{size=" + lines.size() + 
               ", vocabularySize=" + vocabularySize + 
               ", maxSequenceLength=" + maxSequenceLength + "}";
    }

    public static final class Sample {
        private final Tensor tokens;
        private final String text;
        private final int[] tokenIds;
        private final float[] attentionMask;

        public Sample(Tensor tokens, String text, int[] tokenIds, float[] attentionMask) {
            this.tokens = Objects.requireNonNull(tokens, "Tokens cannot be null");
            this.text = Objects.requireNonNull(text, "Text cannot be null");
            this.tokenIds = tokenIds.clone();
            this.attentionMask = attentionMask.clone();
        }

        public Tensor tokens() {
            return tokens;
        }

        public String text() {
            return text;
        }

        public int[] tokenIds() {
            return tokenIds.clone();
        }

        public float[] attentionMask() {
            return attentionMask.clone();
        }

        public int length() {
            return tokenIds.length;
        }

        @Override
        public String toString() {
            return "Sample{tokens=" + tokens.shape() + ", text=\"" + 
                   (text.length() > 50 ? text.substring(0, 50) + "..." : text) + "\"}";
        }
    }

    public interface Tokenizer {
        String[] tokenize(String text);
    }

    public static final class WhitespaceTokenizer implements Tokenizer {
        @Override
        public String[] tokenize(String text) {
            if (text == null || text.trim().isEmpty()) {
                return new String[0];
            }
            return text.trim().split("\\s+");
        }
    }

    public static final class CharacterTokenizer implements Tokenizer {
        @Override
        public String[] tokenize(String text) {
            if (text == null || text.isEmpty()) {
                return new String[0];
            }
            String[] chars = new String[text.length()];
            for (int i = 0; i < text.length(); i++) {
                chars[i] = String.valueOf(text.charAt(i));
            }
            return chars;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private Path filePath;
        private ComputeBackend backend;
        private Tokenizer tokenizer;
        private int maxSequenceLength = 0;
        private int paddingTokenId = DEFAULT_PADDING_TOKEN_ID;
        private boolean paddingEnabled = false;

        public Builder filePath(Path filePath) {
            this.filePath = filePath;
            return this;
        }

        public Builder backend(ComputeBackend backend) {
            this.backend = backend;
            return this;
        }

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder whitespaceTokenizer() {
            this.tokenizer = new WhitespaceTokenizer();
            return this;
        }

        public Builder characterTokenizer() {
            this.tokenizer = new CharacterTokenizer();
            return this;
        }

        public Builder maxSequenceLength(int maxSequenceLength) {
            this.maxSequenceLength = maxSequenceLength;
            return this;
        }

        public Builder paddingTokenId(int paddingTokenId) {
            this.paddingTokenId = paddingTokenId;
            return this;
        }

        public Builder paddingEnabled(boolean paddingEnabled) {
            this.paddingEnabled = paddingEnabled;
            return this;
        }

        public TextFileDataset build() {
            return new TextFileDataset(this);
        }
    }
}
