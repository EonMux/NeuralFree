package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.packed.PackedExpertEntry;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code nftool pack-experts} command: it packs a routed-expert model into one
 * {@code blk.N.ffn_exps_packed} tensor per layer, refuses a model without routed experts and a
 * malformed command line with the field named, and answers {@code --help} with the usage.
 */
public class PackExpertsCommandTest {

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    /** The llama fixture with routed experts on every layer in place of its dense FFN. */
    static GgufTestWriter llamaMoe() {
        GgufTestWriter w = SyntheticModels.llama();
        w.u32("llama.expert_count", 4).u32("llama.expert_used_count", 2)
                .u32("llama.expert_feed_forward_length", 32);
        for (int il = 0; il < 2; il++) {
            w.removeTensor("blk." + il + ".ffn_gate.weight");
            w.removeTensor("blk." + il + ".ffn_up.weight");
            w.removeTensor("blk." + il + ".ffn_down.weight");
            w.tensorRandom("blk." + il + ".ffn_gate_inp.weight", GGMLType.F32, 1000L + il, -0.1f, 0.1f, 64, 4);
            w.tensorRandom("blk." + il + ".ffn_up_exps.weight", GGMLType.F32, 2000L + il, -0.1f, 0.1f, 64, 32, 4);
            w.tensorRandom("blk." + il + ".ffn_gate_exps.weight", GGMLType.F32, 3000L + il, -0.1f, 0.1f, 64, 32, 4);
            w.tensorRandom("blk." + il + ".ffn_down_exps.weight", GGMLType.F32, 4000L + il, -0.1f, 0.1f, 32, 64, 4);
        }
        return w;
    }

    @Test
    public void packsEveryMoeLayerAndRefusesToReplaceWithoutOverwrite() throws Exception {
        Path model = folder.newFile("model.gguf").toPath();
        llamaMoe().writeTo(model);
        Path out = folder.newFolder("out").toPath().resolve("packed.gguf");
        List<Path> written = PackExpertsCommand.pack(new String[]{model.toString(), out.toString()});
        assertEquals(List.of(out), written);
        try (com.eonmux.neuralfree.gguf.GGUFFileParser parser = ModelSources.parseGguf(out.toString())) {
            for (int il = 0; il < 2; il++) {
                assertNotNull(parser.getTensor("blk." + il + ".ffn_exps_packed"));
                assertNull(parser.getTensor("blk." + il + ".ffn_up_exps.weight"));
                assertEquals(14, parser.getUInt64Array("avalon.experts_packed.blk." + il).length);
            }
        }
        // a second run without --overwrite refuses before anything is replaced, with it the file is
        // replaced
        IOException refused = assertThrows(IOException.class,
                () -> PackExpertsCommand.pack(new String[]{model.toString(), out.toString()}));
        assertTrue(refused.getMessage(), refused.getMessage().contains("already exists"));
        assertEquals(List.of(out),
                PackExpertsCommand.pack(new String[]{model.toString(), out.toString(), "--overwrite"}));
    }

    @Test
    public void refusesAModelWithoutRoutedExpertsByName() throws Exception {
        Path dense = folder.newFile("dense.gguf").toPath();
        SyntheticModels.llama().writeTo(dense);
        Path out = folder.getRoot().toPath().resolve("packed.gguf");
        IOException e = assertThrows(IOException.class,
                () -> PackExpertsCommand.pack(new String[]{dense.toString(), out.toString()}));
        assertTrue(e.getMessage(), e.getMessage().contains("no routed experts"));
    }

    @Test
    public void refusesMalformedCommandLinesByTheField() throws Exception {
        Path model = folder.newFile("model.gguf").toPath();
        llamaMoe().writeTo(model);
        Path out = folder.getRoot().toPath().resolve("p.gguf");
        assertEquals("unknown option --mode (run with --help for the options)",
                usageOf(() -> PackExpertsCommand.pack(new String[]{model.toString(), out.toString(), "--mode", "compat"})));
        assertEquals("too many file arguments; 'x' follows " + model + " and " + out,
                usageOf(() -> PackExpertsCommand.pack(new String[]{model.toString(), out.toString(), "x"})));
        assertEquals("--split-max-size needs a size in bytes, e.g. 2000000 or 2000m, not 'huge'",
                usageOf(() -> PackExpertsCommand.pack(new String[]{model.toString(), out.toString(),
                        "--split-max-size", "huge"})));
        assertNull(PackExpertsCommand.pack(new String[]{"--help"}));
    }

    /** The UsageException message of a pack call that fails on the command line. */
    private static String usageOf(ThrowingCall call) throws IOException {
        try {
            call.call();
        } catch (ExportCommand.UsageException e) {
            return e.getMessage();
        }
        throw new AssertionError("expected a UsageException");
    }

    private interface ThrowingCall {
        void call() throws IOException, ExportCommand.UsageException;
    }
}
