package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.write.QuantizeFtype;
import org.junit.Test;

import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code nftool quantize} command line: flag order, the positional forms, the type parsing by name
 * and id, the derived default output name, and the refusals.
 */
public class QuantizeCommandParseTest {

    @Test
    public void modelThenType() throws Exception {
        QuantizeCommand.Parsed p = QuantizeCommand.parse(new String[] {"model.gguf", "Q8_0"});
        assertEquals(Paths.get("model.gguf"), p.input);
        assertEquals(QuantizeFtype.Q8_0, p.params.ftype);
        // the default output name, derived beside the input as llama.cpp derives it
        assertEquals("ggml-model-q8_0.gguf", p.out.getFileName().toString());
    }

    @Test
    public void modelThenOutputThenType() throws Exception {
        QuantizeCommand.Parsed p = QuantizeCommand.parse(new String[] {"model.gguf", "out.gguf", "q4_k_m"});
        assertEquals(Paths.get("out.gguf"), p.out);
        assertEquals(QuantizeFtype.Q4_K_M, p.params.ftype);
    }

    @Test
    public void numericTypeIdPicksTheFirstRecipeWithTheId() throws Exception {
        QuantizeCommand.Parsed p = QuantizeCommand.parse(new String[] {"model.gguf", "15"});
        assertEquals(QuantizeFtype.Q4_K, p.params.ftype); // Q4_K comes before Q4_K_M in the list
        QuantizeCommand.Parsed zero = QuantizeCommand.parse(new String[] {"model.gguf", "0"});
        assertEquals(QuantizeFtype.F32, zero.params.ftype); // F32 before COPY
    }

    @Test
    public void threadCountAndFlags() throws Exception {
        QuantizeCommand.Parsed p = QuantizeCommand.parse(new String[] {
                "--pure", "--allow-requantize", "--leave-output-tensor", "--keep-split", "--dry-run",
                "--imatrix", "im.gguf", "--include-weights", "attn", "--output-tensor-type", "q8_0",
                "--token-embedding-type", "q5_k", "--tensor-type", "attn_q=q4_0",
                "--prune-layers", "3,1,1", "--override-kv", "general.name=str:x",
                "--max-buffer-size", "64",
                "model.gguf", "q4_k_m", "4"});
        assertEquals(QuantizeFtype.Q4_K_M, p.params.ftype);
        assertTrue(p.params.pure);
        assertTrue(p.params.allowRequantize);
        assertFalse(p.params.quantizeOutputTensor);
        assertTrue(p.params.keepSplit);
        assertTrue(p.params.dryRun);
        assertEquals(Paths.get("im.gguf"), p.params.imatrixFile);
        assertEquals(1, p.params.includeWeights.size());
        assertEquals(GGMLType.Q8_0, p.params.outputTensorType);
        assertEquals(GGMLType.Q5_K, p.params.tokenEmbeddingType);
        assertEquals(1, p.params.tensorTypes.size());
        assertEquals(2, p.params.pruneLayers.size()); // sorted and de-duplicated
        assertEquals(4, p.params.nthread);
    }

    @Test
    public void tensorTypeFileExpands() throws Exception {
        java.nio.file.Path file = java.nio.file.Files.createTempFile("tt", ".txt");
        java.nio.file.Files.writeString(file, "attn_q=q4_0\nffn_down=q6_k");
        QuantizeCommand.Parsed p = QuantizeCommand.parse(new String[] {
                "--tensor-type-file", file.toString(), "model.gguf", "q8_0"});
        assertEquals(2, p.params.tensorTypes.size());
        java.nio.file.Files.delete(file);
    }

    @Test
    public void refusals() {
        refusal(new String[] {"model.gguf"});
        refusal(new String[] {"model.gguf", "no-such-type"});
        refusal(new String[] {"model.gguf", "out.gguf"});
        refusal(new String[] {"--bogus", "model.gguf", "q8_0"});
        refusal(new String[] {"--include-weights", "a", "--exclude-weights", "b", "model.gguf", "q8_0"});
        refusal(new String[] {"--tensor-type", "noequals", "model.gguf", "q8_0"});
        refusal(new String[] {"--tensor-type", "attn_q=nope", "model.gguf", "q8_0"});
        refusal(new String[] {"--imatrix", "model.gguf", "q8_0"});
        refusal(new String[] {"model.gguf", "out.gguf", "q8_0", "2", "extra"});
        refusal(new String[] {"--prune-layers", "x", "model.gguf", "q8_0"});
    }

    private static void refusal(String[] args) {
        try {
            QuantizeCommand.parse(args);
            fail("expected a usage error for " + String.join(" ", args));
        } catch (QuantizeCommand.UsageException e) {
            // the message names what went wrong
            assertTrue(e.getMessage(), !e.getMessage().isEmpty());
        }
    }

    @Test
    public void dryRunNeedsNoOutputPath() throws Exception {
        QuantizeCommand.Parsed p = QuantizeCommand.parse(new String[] {"--dry-run", "model.gguf",
                "q4_k_m"});
        assertNull(p.out);
        assertTrue(p.params.dryRun);
    }
}
