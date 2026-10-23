package com.eonmux.neuralfree.data.transforms;

import com.eonmux.neuralfree.data.transforms.image.*;
import com.eonmux.neuralfree.data.transforms.text.*;
import org.junit.Test;
import static org.junit.Assert.*;

public class TransformTest {
    
    @Test
    public void testTransformInterface() {
        Transform<String> upperCase = String::toUpperCase;
        assertEquals("HELLO", upperCase.apply("hello"));
        assertFalse(upperCase.isGpuAccelerated());
    }
    
    @Test
    public void testImageDataCreation() {
        float[] data = new float[]{0.5f, 0.3f, 0.2f};
        Transform.ImageData image = new Transform.ImageData(data, 1, 1, 3);
        
        assertEquals(1, image.height);
        assertEquals(1, image.width);
        assertEquals(3, image.channels);
        assertEquals(3, image.totalPixels());
        assertArrayEquals(data, image.data, 1e-6f);
    }
    
    @Test
    public void testImageDataWithNewData() {
        float[] data = new float[]{0.5f, 0.3f, 0.2f};
        Transform.ImageData image = new Transform.ImageData(data, 1, 1, 3);
        
        float[] newData = new float[]{0.1f, 0.2f, 0.3f};
        Transform.ImageData newImage = image.withData(newData);
        
        assertEquals(1, newImage.height);
        assertEquals(1, newImage.width);
        assertEquals(3, newImage.channels);
        assertArrayEquals(newData, newImage.data, 1e-6f);
    }
    
    @Test
    public void testTextDataCreation() {
        Transform.TextData text = new Transform.TextData("hello world");
        
        assertEquals("hello world", text.text);
        assertNull(text.tokens);
        assertFalse(text.hasTokens());
    }
    
    @Test
    public void testTextDataWithTokens() {
        Transform.TextData text = new Transform.TextData("hello", new int[]{1, 2, 3});
        
        assertEquals("hello", text.text);
        assertArrayEquals(new int[]{1, 2, 3}, text.tokens);
        assertTrue(text.hasTokens());
    }
    
    @Test
    public void testResizeNearestNeighbor() {
        float[] data = new float[]{
            0.0f, 0.5f,
            1.0f, 1.5f
        };
        Transform.ImageData image = new Transform.ImageData(data, 2, 2, 1);
        
        Resize resize = Resize.nearest(4, 4);
        Transform.ImageData resized = resize.apply(image);
        
        assertEquals(4, resized.height);
        assertEquals(4, resized.width);
        assertEquals(16, resized.data.length);
        
        assertEquals(0.0f, resized.data[0], 1e-6f);
        assertEquals(0.5f, resized.data[3], 1e-6f);
        assertEquals(1.5f, resized.data[15], 1e-6f);
    }
    
    @Test
    public void testResizeBilinear() {
        float[] data = new float[]{
            0.0f, 1.0f,
            1.0f, 2.0f
        };
        Transform.ImageData image = new Transform.ImageData(data, 2, 2, 1);
        
        Resize resize = Resize.bilinear(3, 3);
        Transform.ImageData resized = resize.apply(image);
        
        assertEquals(3, resized.height);
        assertEquals(3, resized.width);
        assertEquals(9, resized.data.length);
        
        assertEquals(0.0f, resized.data[0], 1e-6f);
        assertEquals(2.0f, resized.data[8], 1e-6f);
    }
    
    @Test
    public void testResizeNoChange() {
        float[] data = new float[]{0.5f, 0.3f};
        Transform.ImageData image = new Transform.ImageData(data, 2, 1, 1);
        
        Resize resize = Resize.nearest(2, 1);
        Transform.ImageData resized = resize.apply(image);
        
        assertSame(image, resized);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testResizeInvalidDimensions() {
        Resize resize = new Resize(-1, 10, Resize.InterpolationMode.BILINEAR);
    }
    
    @Test
    public void testRandomCrop() {
        float[] data = new float[]{
            0.0f, 0.1f, 0.2f, 0.3f,
            1.0f, 1.1f, 1.2f, 1.3f,
            2.0f, 2.1f, 2.2f, 2.3f,
            3.0f, 3.1f, 3.2f, 3.3f
        };
        Transform.ImageData image = new Transform.ImageData(data, 4, 4, 1);
        
        RandomCrop crop = RandomCrop.withSeed(2, 2, 42L);
        Transform.ImageData cropped = crop.apply(image);
        
        assertEquals(2, cropped.height);
        assertEquals(2, cropped.width);
        assertEquals(4, cropped.data.length);
    }
    
    @Test
    public void testRandomCropAtPosition() {
        float[] data = new float[]{
            0.0f, 0.1f, 0.2f,
            1.0f, 1.1f, 1.2f,
            2.0f, 2.1f, 2.2f
        };
        Transform.ImageData image = new Transform.ImageData(data, 3, 3, 1);
        
        RandomCrop crop = RandomCrop.of(2, 2);
        Transform.ImageData cropped = crop.applyAt(image, 0, 0);
        
        assertEquals(2, cropped.height);
        assertEquals(2, cropped.width);
        assertEquals(0.0f, cropped.data[0], 1e-6f);
        assertEquals(0.1f, cropped.data[1], 1e-6f);
        assertEquals(1.0f, cropped.data[2], 1e-6f);
        assertEquals(1.1f, cropped.data[3], 1e-6f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testRandomCropExceedsImage() {
        float[] data = new float[]{0.5f, 0.3f};
        Transform.ImageData image = new Transform.ImageData(data, 2, 1, 1);
        
        RandomCrop crop = RandomCrop.of(4, 4);
        crop.apply(image);
    }
    
    @Test
    public void testRandomHorizontalFlip() {
        float[] data = new float[]{
            0.0f, 0.1f, 0.2f,
            1.0f, 1.1f, 1.2f
        };
        Transform.ImageData image = new Transform.ImageData(data, 2, 3, 1);
        
        RandomHorizontalFlip flip = RandomHorizontalFlip.withSeed(42L);
        Transform.ImageData flipped = flip.apply(image);
        
        assertEquals(2, flipped.height);
        assertEquals(3, flipped.width);
    }
    
    @Test
    public void testRandomHorizontalFlipForced() {
        float[] data = new float[]{
            0.0f, 0.1f, 0.2f,
            1.0f, 1.1f, 1.2f
        };
        Transform.ImageData image = new Transform.ImageData(data, 2, 3, 1);
        
        RandomHorizontalFlip flip = new RandomHorizontalFlip(1.0);
        Transform.ImageData flipped = flip.applyForced(image, true);
        
        assertEquals(0.2f, flipped.data[0], 1e-6f);
        assertEquals(0.1f, flipped.data[1], 1e-6f);
        assertEquals(0.0f, flipped.data[2], 1e-6f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testRandomHorizontalFlipInvalidProbability() {
        new RandomHorizontalFlip(1.5);
    }
    
    @Test
    public void testNormalize() {
        float[] data = new float[]{
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f
        };
        Transform.ImageData image = new Transform.ImageData(data, 2, 1, 3);
        
        Normalize normalize = Normalize.of(new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f});
        Transform.ImageData normalized = normalize.apply(image);
        
        for (float v : normalized.data) {
            assertEquals(0.0f, v, 1e-6f);
        }
    }
    
    @Test
    public void testNormalizeDenormalize() {
        float[] data = new float[]{0.5f, 0.75f, 1.0f};
        Transform.ImageData image = new Transform.ImageData(data, 1, 1, 3);
        
        Normalize normalize = Normalize.of(new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.25f, 0.25f, 0.25f});
        Transform.ImageData normalized = normalize.apply(image);
        Transform.ImageData denormalized = normalize.denormalize(normalized);
        
        assertArrayEquals(data, denormalized.data, 1e-6f);
    }
    
    @Test
    public void testNormalizeImageNet() {
        Normalize normalize = Normalize.imagenet();
        
        assertNotNull(normalize);
        assertEquals(3, normalize.mean().length);
        assertEquals(3, normalize.std().length);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeZeroStd() {
        Normalize.of(new float[]{0.5f}, new float[]{0.0f});
    }
    
    @Test
    public void testColorJitterBrightness() {
        float[] data = new float[]{
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f
        };
        Transform.ImageData image = new Transform.ImageData(data, 2, 1, 3);
        
        ColorJitter jitter = ColorJitter.brightnessOnly(0.0f);
        Transform.ImageData result = jitter.apply(image);
        
        assertArrayEquals(data, result.data, 1e-6f);
    }
    
    @Test
    public void testColorJitterGrayscaleImage() {
        float[] data = new float[]{0.5f, 0.5f};
        Transform.ImageData image = new Transform.ImageData(data, 2, 1, 1);
        
        ColorJitter jitter = ColorJitter.defaultTransform();
        Transform.ImageData result = jitter.apply(image);
        
        assertArrayEquals(data, result.data, 1e-6f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testColorJitterInvalidHue() {
        new ColorJitter(0, 0, 0, 0.6f);
    }
    
    @Test
    public void testToTensorScale() {
        float[] data = new float[]{128.0f, 255.0f, 0.0f};
        Transform.ImageData image = new Transform.ImageData(data, 3, 1, 1);
        
        ToTensor toTensor = ToTensor.noScale(null);
        Transform.ImageData result = toTensor.apply(image);
        
        assertEquals(128.0f, result.data[0], 1e-6f);
    }
    
    @Test
    public void testToTensorLayout() {
        assertEquals(ToTensor.Layout.CHW, ToTensor.chw(null).layout());
        assertEquals(ToTensor.Layout.HWC, ToTensor.hwc(null).layout());
    }
    
    @Test
    public void testTokenizeWhitespace() {
        java.util.Map<String, Integer> vocab = new java.util.HashMap<>();
        vocab.put("hello", 1);
        vocab.put("world", 2);
        vocab.put("[UNK]", 0);
        
        Tokenize tokenizer = Tokenize.whitespace(vocab);
        Transform.TextData text = new Transform.TextData("hello world");
        Transform.TextData tokenized = tokenizer.apply(text);
        
        assertArrayEquals(new int[]{1, 2}, tokenized.tokens);
    }
    
    @Test
    public void testTokenizeCharacter() {
        java.util.Map<String, Integer> vocab = new java.util.HashMap<>();
        vocab.put("a", 1);
        vocab.put("b", 2);
        vocab.put("[UNK]", 0);
        
        Tokenize tokenizer = Tokenize.character(vocab);
        Transform.TextData text = new Transform.TextData("ab");
        Transform.TextData tokenized = tokenizer.apply(text);
        
        assertArrayEquals(new int[]{1, 2}, tokenized.tokens);
    }
    
    @Test
    public void testTokenizeUnknownToken() {
        java.util.Map<String, Integer> vocab = new java.util.HashMap<>();
        vocab.put("[UNK]", 0);
        vocab.put("known", 1);
        
        Tokenize tokenizer = Tokenize.whitespace(vocab);
        Transform.TextData text = new Transform.TextData("unknown known");
        Transform.TextData tokenized = tokenizer.apply(text);
        
        assertArrayEquals(new int[]{0, 1}, tokenized.tokens);
    }
    
    @Test
    public void testTokenizeDecode() {
        java.util.Map<String, Integer> vocab = new java.util.HashMap<>();
        vocab.put("hello", 1);
        vocab.put("world", 2);
        vocab.put("[UNK]", 0);
        
        Tokenize tokenizer = Tokenize.whitespace(vocab);
        String decoded = tokenizer.decode(new int[]{1, 2});
        
        assertEquals("hello world", decoded);
    }
    
    @Test
    public void testPadSequenceRight() {
        PadSequence padder = PadSequence.toLength(5);
        
        Transform.TextData text = new Transform.TextData("test", new int[]{1, 2, 3});
        Transform.TextData padded = padder.apply(text);
        
        assertArrayEquals(new int[]{1, 2, 3, 0, 0}, padded.tokens);
    }
    
    @Test
    public void testPadSequenceLeft() {
        PadSequence padder = PadSequence.leftPadding(5, 0);
        
        Transform.TextData text = new Transform.TextData("test", new int[]{1, 2, 3});
        Transform.TextData padded = padder.apply(text);
        
        assertArrayEquals(new int[]{0, 0, 1, 2, 3}, padded.tokens);
    }
    
    @Test
    public void testPadSequenceTruncate() {
        PadSequence padder = PadSequence.toLength(3);
        
        Transform.TextData text = new Transform.TextData("test", new int[]{1, 2, 3, 4, 5});
        Transform.TextData padded = padder.apply(text);
        
        assertArrayEquals(new int[]{1, 2, 3}, padded.tokens);
    }
    
    @Test
    public void testPadSequenceAttentionMask() {
        PadSequence padder = PadSequence.toLength(5);
        
        Transform.TextData text = new Transform.TextData("test", new int[]{1, 2, 3});
        padder.apply(text);
        
        int[] mask = padder.createAttentionMask(text);
        
        assertArrayEquals(new int[]{1, 1, 1, 0, 0}, mask);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testPadSequenceInvalidLength() {
        new PadSequence(0);
    }
}
