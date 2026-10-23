package com.eonmux.neuralfree.data.transforms.text;

import com.eonmux.neuralfree.data.transforms.Transform;

public class PadSequence implements Transform<Transform.TextData> {
    private final int maxLength;
    private final int padTokenId;
    private final PaddingSide paddingSide;
    private final TruncationStrategy truncationStrategy;
    
    public enum PaddingSide {
        LEFT,
        RIGHT
    }
    
    public enum TruncationStrategy {
        NONE,
        RIGHT,
        LEFT
    }
    
    public PadSequence(int maxLength) {
        this(maxLength, 0, PaddingSide.RIGHT, TruncationStrategy.RIGHT);
    }
    
    public PadSequence(int maxLength, int padTokenId) {
        this(maxLength, padTokenId, PaddingSide.RIGHT, TruncationStrategy.RIGHT);
    }
    
    public PadSequence(int maxLength, int padTokenId, PaddingSide paddingSide, TruncationStrategy truncationStrategy) {
        if (maxLength <= 0) {
            throw new IllegalArgumentException("Max length must be positive: " + maxLength);
        }
        this.maxLength = maxLength;
        this.padTokenId = padTokenId;
        this.paddingSide = paddingSide != null ? paddingSide : PaddingSide.RIGHT;
        this.truncationStrategy = truncationStrategy != null ? truncationStrategy : TruncationStrategy.RIGHT;
    }
    
    public static PadSequence toLength(int maxLength) {
        return new PadSequence(maxLength);
    }
    
    public static PadSequence withPadToken(int maxLength, int padTokenId) {
        return new PadSequence(maxLength, padTokenId);
    }
    
    public static PadSequence leftPadding(int maxLength, int padTokenId) {
        return new PadSequence(maxLength, padTokenId, PaddingSide.LEFT, TruncationStrategy.RIGHT);
    }
    
    @Override
    public Transform.TextData apply(Transform.TextData input) {
        if (!input.hasTokens()) {
            return input;
        }
        
        int[] tokens = input.tokens;
        
        if (tokens.length > maxLength && truncationStrategy != TruncationStrategy.NONE) {
            tokens = truncate(tokens);
        }
        
        if (tokens.length < maxLength) {
            tokens = pad(tokens);
        }
        
        return input.withTokens(tokens);
    }
    
    private int[] truncate(int[] tokens) {
        if (truncationStrategy == TruncationStrategy.RIGHT) {
            int[] truncated = new int[maxLength];
            System.arraycopy(tokens, 0, truncated, 0, maxLength);
            return truncated;
        } else {
            int[] truncated = new int[maxLength];
            int start = tokens.length - maxLength;
            System.arraycopy(tokens, start, truncated, 0, maxLength);
            return truncated;
        }
    }
    
    private int[] pad(int[] tokens) {
        int[] padded = new int[maxLength];
        int padLength = maxLength - tokens.length;
        
        if (paddingSide == PaddingSide.RIGHT) {
            System.arraycopy(tokens, 0, padded, 0, tokens.length);
            for (int i = tokens.length; i < maxLength; i++) {
                padded[i] = padTokenId;
            }
        } else {
            for (int i = 0; i < padLength; i++) {
                padded[i] = padTokenId;
            }
            System.arraycopy(tokens, 0, padded, padLength, tokens.length);
        }
        
        return padded;
    }
    
    public int[][] createAttentionMask(Transform.TextData[] inputs) {
        int[][] masks = new int[inputs.length][maxLength];
        
        for (int i = 0; i < inputs.length; i++) {
            Transform.TextData input = inputs[i];
            if (!input.hasTokens()) continue;
            
            int[] tokens = input.tokens;
            int validLength = Math.min(tokens.length, maxLength);
            
            if (paddingSide == PaddingSide.RIGHT) {
                for (int j = 0; j < validLength; j++) {
                    masks[i][j] = tokens[j] != padTokenId ? 1 : 0;
                }
            } else {
                int offset = maxLength - validLength;
                for (int j = 0; j < validLength; j++) {
                    masks[i][offset + j] = tokens[j] != padTokenId ? 1 : 0;
                }
            }
        }
        
        return masks;
    }
    
    public int[] createAttentionMask(Transform.TextData input) {
        int[] mask = new int[maxLength];
        
        if (!input.hasTokens()) {
            return mask;
        }
        
        int[] tokens = input.tokens;
        int validLength = Math.min(tokens.length, maxLength);
        
        if (paddingSide == PaddingSide.RIGHT) {
            for (int j = 0; j < validLength; j++) {
                mask[j] = tokens[j] != padTokenId ? 1 : 0;
            }
        } else {
            int offset = maxLength - validLength;
            for (int j = 0; j < validLength; j++) {
                mask[offset + j] = tokens[j] != padTokenId ? 1 : 0;
            }
        }
        
        return mask;
    }
    
    public int maxLength() { return maxLength; }
    public int padTokenId() { return padTokenId; }
    public PaddingSide paddingSide() { return paddingSide; }
    public TruncationStrategy truncationStrategy() { return truncationStrategy; }
}
