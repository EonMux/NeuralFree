package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@code nftool passkey} over a running engine: the prompt of the reference with the pass key at the given
 * position, the prefill in batches with the context shifted once it fills, the greedy answer loop and the
 * recovered flag, and the command line's refusals. The fixture's training context is 256, which
 * {@code --ctx-size} overrides to force the shift path.
 */
public class PasskeyCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(48).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static String[] runArgs(Path model, String[] extra) {
        String[] args = {model.toString(), "--ctx-size", "48", "--keep", "2", "--batch-size", "8",
                "--junk", "5", "--pos", "2", "--seed", "7"};
        String[] all = new String[args.length + extra.length];
        System.arraycopy(args, 0, all, 0, args.length);
        System.arraycopy(extra, 0, all, args.length, extra.length);
        return all;
    }

    @Test
    public void theRunShiftsThroughALongPromptAndAnswers() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        PrintStream realOut = System.out;
        PrintStream realErr = System.err;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(captured, true, "UTF-8"));
            System.setErr(new PrintStream(new ByteArrayOutputStream(), true, "UTF-8"));
            String answer = PasskeyCommand.dispatch(
                    runArgs(model, new String[] {}), config());
            assertTrue("the answer loop generated text", !answer.isEmpty());
            String stdout = captured.toString("UTF-8");
            assertTrue(stdout, stdout.contains("nftool passkey: passkey = "));
            assertTrue(stdout, stdout.contains("nftool passkey: answer: " + answer));
            assertTrue(stdout, stdout.contains("nftool passkey: passkey recovered: "));
            // the seed fixes both the key and the answer
            String again = PasskeyCommand.dispatch(runArgs(model, new String[] {}), config());
            assertEquals(answer, again);
        } finally {
            System.setOut(realOut);
            System.setErr(realErr);
        }
    }

    @Test
    public void theTrainingContextComesFromTheModelMetadata() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        assertEquals(256, PasskeyCommand.trainingContext(model));
    }

    @Test
    public void theCommandLineRefusesUnsupportedValuesByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        assertThrows(PasskeyCommand.UsageException.class,
                () -> PasskeyCommand.parse(new String[] {model.toString(), "--pos", "5", "--junk", "5"}));
        assertThrows(PasskeyCommand.UsageException.class,
                () -> PasskeyCommand.parse(new String[] {model.toString(), "--junk", "0"}));
        assertThrows(PasskeyCommand.UsageException.class,
                () -> PasskeyCommand.parse(new String[] {model.toString(), "--batch-size", "0"}));
        assertThrows(PasskeyCommand.UsageException.class,
                () -> PasskeyCommand.parse(new String[] {model.toString(), "--no-such-flag"}));
        assertThrows(PasskeyCommand.UsageException.class,
                () -> PasskeyCommand.parse(new String[] {"--junk", "5"}));
    }

    @Test
    public void theRunRefusesBrokenContextsByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        PrintStream realOut = System.out;
        PrintStream realErr = System.err;
        try {
            System.setOut(new PrintStream(new ByteArrayOutputStream(), true, "UTF-8"));
            System.setErr(new PrintStream(new ByteArrayOutputStream(), true, "UTF-8"));
            PasskeyCommand.UsageException zero = assertThrows(PasskeyCommand.UsageException.class,
                    () -> PasskeyCommand.dispatch(
                            runArgs(model, new String[] {"--ctx-size", "0"}), config()));
            assertTrue(zero.getMessage(), zero.getMessage().contains("--ctx-size 0"));
            PasskeyCommand.UsageException keep = assertThrows(PasskeyCommand.UsageException.class,
                    () -> PasskeyCommand.dispatch(
                            runArgs(model, new String[] {"--keep", "200"}), config()));
            assertTrue(keep.getMessage(), keep.getMessage().contains("leaves no room"));
            PasskeyCommand.UsageException batch = assertThrows(PasskeyCommand.UsageException.class,
                    () -> PasskeyCommand.dispatch(
                            runArgs(model, new String[] {"--batch-size", "100"}), config()));
            assertTrue(batch.getMessage(), batch.getMessage().contains("one shift can drop"));
        } finally {
            System.setOut(realOut);
            System.setErr(realErr);
        }
    }
}
