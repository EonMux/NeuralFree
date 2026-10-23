package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link CpuWeights} in both {@link TensorReadMode}s: tensors read into direct buffers ({@code --no-mmap}) decode
 * to the same values as memory-mapped ones, and the mode property parses strictly.
 */
public class CpuWeightsReadModeTest {

    private static final String[] NAMES = {"token_embd.weight", "blk.0.attn_q.weight", "output_norm.weight"};

    private static GgufTestWriter model() {
        return new GgufTestWriter()
                .architecture("llama")
                .tensorRandom(NAMES[0], GGMLType.Q8_0, 11L, -1f, 1f, 64, 5)
                .tensorRandom(NAMES[1], GGMLType.F16, 12L, -1f, 1f, 32, 3)
                .tensorRandom(NAMES[2], GGMLType.F32, 13L, -1f, 1f, 64);
    }

    private static float[] decodeAll(CpuWeights w, String name) {
        CpuTensorRef t = w.get(name);
        float[] out = new float[(int) (t.rows() * t.cols())];
        t.decodeRows(0L, (int) t.rows(), out, 0);
        return out;
    }

    @Test
    public void readModeDecodesLikeMmap() throws IOException {
        Path path = model().writeTempFile();
        GGUFFileParser parser = null;
        try {
            parser = new GGUFFileParser(path.toString());
            TensorCatalog catalog = TensorCatalog.of(parser);
            try (CpuWeights mapped = CpuWeights.open(parser, catalog, null, TensorReadMode.MMAP);
                 CpuWeights read = CpuWeights.open(parser, catalog, null, TensorReadMode.READ)) {
                assertEquals(TensorReadMode.MMAP, mapped.readMode());
                assertEquals(TensorReadMode.READ, read.readMode());
                for (String name : NAMES) {
                    assertArrayEquals(name, decodeAll(mapped, name), decodeAll(read, name), 0f);
                }
            }
        } finally {
            try {
                if (parser != null) {
                    parser.close();
                }
            } finally {
                Files.deleteIfExists(path);
            }
        }
    }

    @Test
    public void modePropertyParsesStrictly() {
        Properties p = new Properties();
        assertEquals(TensorReadMode.MMAP, TensorReadMode.fromProperties(p));
        p.setProperty(TensorReadMode.PROP_DISABLE_MMAP, "true");
        assertEquals(TensorReadMode.READ, TensorReadMode.fromProperties(p));
        p.setProperty(TensorReadMode.PROP_DISABLE_MMAP, " FALSE ");
        assertEquals(TensorReadMode.MMAP, TensorReadMode.fromProperties(p));
        p.setProperty(TensorReadMode.PROP_DISABLE_MMAP, "yes");
        assertThrows(IllegalArgumentException.class, () -> TensorReadMode.fromProperties(p));
    }
}
