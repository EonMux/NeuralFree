package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.IOException;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * {@link CpuWeights} over a split model: tensors of every split are mapped (or read) from their own file at the
 * offsets {@link GGUFFileParser#getShards()} reports, and decode to the same values as the parser's own reads.
 */
public class CpuWeightsSplitTest {

    private static final String[] NAMES = {"token_embd.weight", "output_norm.weight", "blk.0.attn_q.weight"};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static GgufTestWriter split(int no) {
        GgufTestWriter w = new GgufTestWriter()
                .u16(GgufKeys.SPLIT_NO, no)
                .u16(GgufKeys.SPLIT_COUNT, 2)
                .i32(GgufKeys.SPLIT_TENSORS_COUNT, NAMES.length);
        if (no == 0) {
            return w.architecture("llama")
                    .tensorRandom(NAMES[0], GGMLType.Q8_0, 21L, -1f, 1f, 64, 5)
                    .tensorRandom(NAMES[1], GGMLType.F32, 22L, -1f, 1f, 64);
        }
        return w.tensorRandom(NAMES[2], GGMLType.F16, 23L, -1f, 1f, 32, 3);
    }

    private static float[] decodeAll(CpuWeights w, String name) {
        CpuTensorRef t = w.get(name);
        float[] out = new float[(int) (t.rows() * t.cols())];
        t.decodeRows(0L, (int) t.rows(), out, 0);
        return out;
    }

    private void checkMode(TensorReadMode mode) throws IOException {
        File dir = tmp.newFolder();
        for (int no = 0; no < 2; no++) {
            split(no).writeTo(new File(dir, String.format(Locale.ROOT, "model-%05d-of-00002.gguf", no + 1)).toPath());
        }
        GGUFFileParser parser = new GGUFFileParser(new File(dir, "model-00001-of-00002.gguf").getPath());
        try {
            assertEquals(2, parser.getShardCount());
            try (CpuWeights w = CpuWeights.open(parser, TensorCatalog.of(parser), null, mode)) {
                for (String name : NAMES) {
                    assertArrayEquals(name, parser.readTensorDataAsFloat(parser.getTensor(name)),
                            decodeAll(w, name), 0f);
                }
            }
        } finally {
            parser.close();
        }
    }

    @Test
    public void mappedSplitTensorsMatchParserReads() throws IOException {
        checkMode(TensorReadMode.MMAP);
    }

    @Test
    public void readSplitTensorsMatchParserReads() throws IOException {
        checkMode(TensorReadMode.READ);
    }
}
