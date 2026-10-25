package com.eonmux.neuralfree.gguf.cpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.SourceOrigin;
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
import java.nio.file.Paths;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * {@link CpuWeights} over a split model opened through {@link ModelSources}: tensors of every split are mapped (or
 * read) from their own file at the offsets {@link GGUFFileParser#dataSegments()} reports, and decode to the same
 * values as the parser's own reads.
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
        String first = new File(dir, "model-00001-of-00002.gguf").getPath();
        try (ModelSource source = ModelSources.open(first, ModelSourceOptions.defaults())) {
            GGUFFileParser parser = (GGUFFileParser) source;
            assertEquals(2, parser.getShardCount());
            assertEquals(2, parser.dataSegments().size());
            try (CpuWeights w = CpuWeights.open(source, TensorCatalog.of(source), null, mode)) {
                assertEquals(mode, w.readMode());
                assertEquals(new SourceOrigin(Paths.get(first), ModelFormat.GGUF), w.origin());
                for (String name : NAMES) {
                    assertArrayEquals(name, parser.readTensorDataAsFloat(parser.getTensor(name)),
                            decodeAll(w, name), 0f);
                }
            }
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
