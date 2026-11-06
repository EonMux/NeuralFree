package com.eonmux.neuralfree.gguf.image;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The token budget to pixel-limit conversion of llama.cpp's {@code clip_hparams::set_limit_image_tokens}
 * (tools/mtmd/clip-model.h): each limit scales by the merged patch area, and a set limit replaces the model's
 * pixel limit.
 */
public class ImageTokenBudgetTest {

    @Test
    public void emptyBudgetHasNoLimits() {
        ImageTokenBudget budget = ImageTokenBudget.none();
        assertEquals(true, budget.isEmpty());
        assertEquals(0, budget.minPixels(784));
        assertEquals(0, budget.maxPixels(784));
    }

    @Test
    public void limitsScaleByTheMergedPatchArea() {
        // qwen2-vl geometry: patch 14, merge 2 -> one token covers 28x28 = 784 pixels
        ImageTokenBudget budget = ImageTokenBudget.builder().minTokens(8).maxTokens(4096).build();
        assertEquals(8L * 784, budget.minPixels(784));
        assertEquals(4096L * 784, budget.maxPixels(784));
    }

    @Test
    public void oneSidedBudgetKeepsTheOtherSideUnset() {
        ImageTokenBudget budget = ImageTokenBudget.builder().maxTokens(1024).build();
        assertEquals(0, budget.minPixels(256));
        assertEquals(1024L * 256, budget.maxPixels(256));
    }

    @Test
    public void minAboveMaxIsRefused() {
        try {
            ImageTokenBudget.builder().minTokens(1024).maxTokens(256).build();
            fail("expected an inverted budget rejection");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("image-min-tokens"));
        }
    }

    @Test
    public void negativeLimitsAreRefused() {
        try {
            ImageTokenBudget.builder().minTokens(-1).build();
            fail("expected a negative min rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
        try {
            ImageTokenBudget.builder().maxTokens(-1).build();
            fail("expected a negative max rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
    }

    @Test
    public void pixelConversionRefusesNonPositivePatchArea() {
        ImageTokenBudget budget = ImageTokenBudget.builder().maxTokens(16).build();
        try {
            budget.maxPixels(0);
            fail("expected a patch area rejection");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("mergedPatchArea"));
        }
    }
}
