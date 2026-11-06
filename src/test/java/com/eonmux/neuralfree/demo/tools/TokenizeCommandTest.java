package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link TokenizeCommand} against llama.cpp's tokenize tool: the flag parsing (llama.cpp's option names,
 * escapes, the verbatim {@code --file}, {@code --stdin} needing no text, every invalid flag refused by
 * name), the {@code --no-bos} and {@code --no-parse-special} semantics over the synthetic llama
 * vocabulary, the {@code --ids}, piece and {@code --show-count} output shapes, and the run over a model
 * file and stdin.
 */
public class TokenizeCommandTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ the flags

    @Test
    public void parseTakesLlamaCppDefaultsAndTheAliases() throws Exception {
        TokenizeCommand.Options o = TokenizeCommand.parse(new String[]{"m.gguf", "-p", "hi"});
        assertEquals("m.gguf", o.model());
        assertEquals("hi", o.prompt());
        assertEquals("the special tokens parse", true, o.parseSpecial());
        assertEquals("the escapes process", true, o.escape());
        assertEquals(false, o.ids());
        assertEquals(false, o.showCount());
        assertEquals(false, o.noBos());
        assertEquals(false, o.stdin());
        assertEquals("the model flag answers", "m.gguf",
                TokenizeCommand.parse(new String[]{"--model", "m.gguf", "-p", "hi"}).model());
        assertEquals("m.gguf", TokenizeCommand.parse(new String[]{"-m", "m.gguf", "-p", "hi"}).model());
        TokenizeCommand.Options custom = TokenizeCommand.parse(new String[]{"m.gguf", "-p", "hi", "--ids",
                "--show-count", "--no-bos", "--no-parse-special"});
        assertEquals(true, custom.ids());
        assertEquals(true, custom.showCount());
        assertEquals(true, custom.noBos());
        assertEquals(false, custom.parseSpecial());
    }

    /** The prompt's escapes process by default and stay with --no-escape, as llama.cpp's post-parse pass does. */
    @Test
    public void escapesProcessTheTextUnlessTurnedOff() throws Exception {
        assertEquals("a\nb\t\"c\\", TokenizeCommand.parse(
                new String[]{"m.gguf", "-p", "a\\nb\\t\\\"c\\\\"}).prompt());
        assertEquals("a\\nb", TokenizeCommand.parse(new String[]{"m.gguf", "-p", "a\\nb", "--no-escape"}).prompt());
        assertEquals("a\nb", TokenizeCommand.parse(
                new String[]{"m.gguf", "--no-escape", "-e", "-p", "a\\nb"}).prompt());
    }

    /** {@code --file} reads verbatim: one trailing newline stays (it is itself a token), and its escapes process. */
    @Test
    public void theFileFlagReadsTheTextVerbatim() throws Exception {
        Path f = tmp.newFile("prompt.txt").toPath();
        Files.write(f, "first\nsecond\\tpart\n".getBytes(StandardCharsets.UTF_8));
        assertEquals("the escapes process, the trailing newline stays", "first\nsecond\tpart\n",
                TokenizeCommand.parse(new String[]{"m.gguf", "--file", f.toString()}).prompt());
        assertEquals("--no-escape keeps the escapes and the trailing newline", "first\nsecond\\tpart\n",
                TokenizeCommand.parse(new String[]{"m.gguf", "-f", f.toString(), "--no-escape"}).prompt());
        Files.write(f, "plain\n".getBytes(StandardCharsets.UTF_8));
        assertEquals("unlike the other tools, the newline is not dropped", "plain\n",
                TokenizeCommand.parse(new String[]{"m.gguf", "-f", f.toString()}).prompt());
        TokenizeCommand.UsageException e = assertThrows(TokenizeCommand.UsageException.class,
                () -> TokenizeCommand.parse(new String[]{"m.gguf", "-f", tmp.getRoot() + "/missing.txt"}));
        assertTrue(e.getMessage(), e.getMessage().startsWith("-f '"));
    }

    /** An explicit {@code --file} answers any {@code --prompt}, as llama.cpp's prompt_file does, whatever the order. */
    @Test
    public void theFileWinsOverThePromptWhateverTheOrder() throws Exception {
        Path f = tmp.newFile("prompt.txt").toPath();
        Files.write(f, "from file".getBytes(StandardCharsets.UTF_8));
        assertEquals("from file", TokenizeCommand.parse(
                new String[]{"m.gguf", "-p", "inline", "-f", f.toString()}).prompt());
        assertEquals("from file", TokenizeCommand.parse(
                new String[]{"m.gguf", "-f", f.toString(), "-p", "inline"}).prompt());
    }

    /** An explicitly given empty file passes, as llama.cpp's prompt_file check does; a bare -p "" refuses. */
    @Test
    public void anExplicitEmptyFilePasses() throws Exception {
        Path f = tmp.newFile("empty.txt").toPath();
        Files.write(f, new byte[0]);
        assertEquals("", TokenizeCommand.parse(new String[]{"m.gguf", "-f", f.toString()}).prompt());
    }

    /** {@code --stdin} needs no text of its own and takes the run over any -p/-f (execute reads it). */
    @Test
    public void stdinNeedsNoText() throws Exception {
        TokenizeCommand.Options o = TokenizeCommand.parse(new String[]{"m.gguf", "--stdin"});
        assertEquals(true, o.stdin());
        assertNull(o.prompt());
        assertEquals("stdin takes precedence over a given prompt", true, TokenizeCommand.parse(
                new String[]{"m.gguf", "-p", "ignored", "--stdin"}).stdin());
    }

    @Test
    public void parseRefusesEveryInvalidFlagByName() {
        assertRefused("tokenize needs a model path");
        assertRefused("tokenize needs a model path", "-p", "hi");
        assertRefused("tokenize needs one of --stdin, --file or --prompt", "m.gguf");
        assertRefused("tokenize needs one of --stdin, --file or --prompt", "m.gguf", "-p", "");
        assertRefused("--prompt needs a value", "m.gguf", "--prompt");
        assertRefused("-f needs a value", "m.gguf", "-f");
        assertRefused("-m needs a value", "-m");
        assertRefused("too many file arguments", "m.gguf", "-p", "a", "b.gguf");
        assertRefused("unknown option --tokenz", "m.gguf", "-p", "a", "--tokenz");
    }

    /** {@code --help} prints the usage to stdout and parses nothing. */
    @Test
    public void helpPrintsTheUsageAndParsesNothing() throws Exception {
        PrintStream saved = System.out;
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(bytes, true, StandardCharsets.UTF_8));
            assertNull(TokenizeCommand.parse(new String[]{"m.gguf", "-p", "hi", "--help"}));
        } finally {
            System.setOut(saved);
        }
        String usage = bytes.toString(StandardCharsets.UTF_8.name());
        assertTrue(usage, usage.contains("nftool tokenize <model> [options]"));
        for (String flag : new String[]{"--ids", "--show-count", "--no-bos", "--no-parse-special", "--stdin",
                "--file, -f", "--prompt, -p"}) {
            assertTrue(usage, usage.contains(flag));
        }
    }

    private static void assertRefused(String message, String... args) {
        TokenizeCommand.UsageException e = assertThrows(TokenizeCommand.UsageException.class,
                () -> TokenizeCommand.parse(args));
        assertTrue(e.getMessage(), e.getMessage().startsWith(message));
    }

    // ------------------------------------------------------------------ the tokenization

    /** The tokenizer of the synthetic llama model: SPM with byte fallback, BOS added. */
    private static GGUFTokenizer llamaTokenizer(Path model) throws IOException {
        return new GGUFTokenizer(ModelSources.open(model.toString()));
    }

    @Test
    public void theVocabularysBosAppliesUnlessNoBosSuppressesIt() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = llamaTokenizer(model);
        assertEquals(true, tok.vocab().addBos());
        int bos = tok.bosId();
        assertEquals(bos, tok.encode("hello", true, true)[0]);
        int[] plain = TokenizeCommand.tokenize(tok, "hello", TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "hello"}));
        assertEquals("the default adds the BOS", bos, plain[0]);
        assertArrayEquals(tok.encode("hello", true, true), plain);
        int[] noBos = TokenizeCommand.tokenize(tok, "hello", TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "hello", "--no-bos"}));
        assertFalse("the BOS is gone", Arrays.stream(noBos).anyMatch(t -> t == bos));
        assertArrayEquals(tok.encode("hello", false, true), noBos);
        assertArrayEquals("the rest answers the default's tokens",
                Arrays.copyOfRange(plain, 1, plain.length), noBos);
    }

    /** {@code --no-parse-special} leaves the special-token texts as plain text, as llama.cpp's parse_special=false. */
    @Test
    public void noParseSpecialKeepsTheSpecialTextsLiteral() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = llamaTokenizer(model);
        int bos = tok.bosId();
        int[] parsed = TokenizeCommand.tokenize(tok, "a<s>b", TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "a<s>b"}));
        assertEquals("the leading BOS and the parsed literal <s>", 2,
                Arrays.stream(parsed).filter(t -> t == bos).count());
        assertArrayEquals(tok.encode("a<s>b", true, true), parsed);
        int[] literal = TokenizeCommand.tokenize(tok, "a<s>b", TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "a<s>b", "--no-parse-special"}));
        assertArrayEquals(tok.encode("a<s>b", true, false), literal);
        assertEquals("only the vocabulary's leading BOS is left", 1,
                Arrays.stream(literal).filter(t -> t == bos).count());
        assertEquals("the text falls back to its bytes", tok.getTokenId("<0x3C>"), literal[2]);
        assertTrue(Arrays.stream(literal).anyMatch(t -> t == tok.getTokenId("<0x3E>")));
    }

    // ------------------------------------------------------------------ the output

    @Test
    public void idsPrintAsThePythonParseableList() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = llamaTokenizer(model);
        int[] tokens = tok.encode("hello", true, true);
        StringBuilder ids = new StringBuilder("[");
        for (int i = 0; i < tokens.length; i++) {
            ids.append(i > 0 ? ", " : "").append(tokens[i]);
        }
        ids.append(']');
        assertEquals(ids + "\n", TokenizeCommand.render(tokens, tok, TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "hello", "--ids"})));
        assertTrue("the synthetic BOS is id 1", ids.toString().startsWith("[1, "));
    }

    /** The piece lines are llama.cpp's {@code %6d -> 'piece'}: control tokens print their text. */
    @Test
    public void piecesPrintOnePerTokenAsLlamaCppDoes() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = llamaTokenizer(model);
        int[] tokens = tok.encode("hello", true, true);
        String printed = TokenizeCommand.render(tokens, tok, TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "hello"}));
        String[] lines = printed.split("\n", -1);
        assertEquals("one line per token plus the trailing newline", tokens.length + 1, lines.length);
        assertEquals(String.format("%6d -> '%s'", tokens[0], "<s>"), lines[0]);
        for (int i = 0; i < tokens.length; i++) {
            String piece = new String(tok.pieceWithSpecial(tokens[i]), StandardCharsets.UTF_8);
            assertEquals(String.format("%6d -> '%s'", tokens[i], piece), lines[i]);
        }
        assertEquals("", lines[tokens.length]);
    }

    @Test
    public void showCountPrintsTheTotalLast() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = llamaTokenizer(model);
        int[] tokens = tok.encode("hello", true, true);
        String counted = TokenizeCommand.render(tokens, tok, TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "hello", "--show-count"}));
        assertTrue(counted, counted.endsWith("\nTotal number of tokens: " + tokens.length + "\n"));
        String idsAndCount = TokenizeCommand.render(tokens, tok, TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "hello", "--ids", "--show-count"}));
        assertTrue(idsAndCount, idsAndCount.startsWith("["));
        assertTrue(idsAndCount, idsAndCount.endsWith("]\nTotal number of tokens: " + tokens.length + "\n"));
    }

    // ------------------------------------------------------------------ the run

    /** The whole command over a model file: parse, open the vocabulary, tokenize and print the ids. */
    @Test
    public void theRunPrintsTheIdsOverAModelFileEndToEnd() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        StringBuilder out = new StringBuilder();
        TokenizeCommand.execute(TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "Hello world", "--ids"}), stdin(""), out);
        GGUFTokenizer tok = llamaTokenizer(model);
        assertEquals(TokenizeCommand.render(tok.encode("Hello world", true, true), tok,
                TokenizeCommand.parse(new String[]{model.toString(), "-p", "Hello world", "--ids"})),
                out.toString());
        assertTrue(out.toString(), out.toString().startsWith("[1, "));
        assertTrue(out.toString(), out.toString().endsWith("]\n"));
    }

    /** {@code --stdin} reads the whole stream and processes its escapes; it beats a given -p. */
    @Test
    public void stdinIsReadAndEscaped() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        GGUFTokenizer tok = llamaTokenizer(model);
        StringBuilder out = new StringBuilder();
        TokenizeCommand.execute(TokenizeCommand.parse(
                new String[]{model.toString(), "-p", "ignored", "--stdin", "--ids"}), stdin("a\\nb"), out);
        assertArrayEquals(tok.encode("a\nb", true, true), idsOf(out.toString()));
        StringBuilder raw = new StringBuilder();
        TokenizeCommand.execute(TokenizeCommand.parse(
                new String[]{model.toString(), "--stdin", "--ids", "--no-escape"}), stdin("a\\nb"), raw);
        assertArrayEquals(tok.encode("a\\nb", true, true), idsOf(raw.toString()));
    }

    /** The empty stdin tokenizes to its special tokens alone, as llama.cpp's empty prompt does. */
    @Test
    public void emptyStdinTokenizesToTheSpecialsAlone() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        StringBuilder out = new StringBuilder();
        TokenizeCommand.execute(TokenizeCommand.parse(
                new String[]{model.toString(), "--stdin", "--ids"}), stdin(""), out);
        assertEquals("[1]\n", out.toString());
    }

    /** A model that cannot open refuses with its path, as llama.cpp's load failure does. */
    @Test
    public void anUnopenableModelRefuses() throws Exception {
        Path model = tmp.newFile("broken.gguf").toPath();
        Files.write(model, new byte[]{1, 2, 3});
        StringBuilder out = new StringBuilder();
        TokenizeCommand.Options o = TokenizeCommand.parse(new String[]{model.toString(), "-p", "hi"});
        assertThrows(IOException.class, () -> TokenizeCommand.execute(o, stdin(""), out));
        assertEquals("", out.toString());
    }

    /** The tokenize route is on the nftool surface. */
    @Test
    public void theRouteIsImplemented() {
        assertTrue(NfTool.IMPLEMENTED.contains("tokenize"));
        assertTrue(NfTool.COMMANDS.contains("tokenize"));
    }

    private static ByteArrayInputStream stdin(String text) {
        return new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8));
    }

    /** The ids of a {@code --ids} output, for the assertions over the run. */
    private static int[] idsOf(String printed) {
        String list = printed.substring(printed.indexOf('[') + 1, printed.lastIndexOf(']'));
        if (list.trim().isEmpty()) {
            return new int[0];
        }
        return Arrays.stream(list.split(",")).map(String::trim).mapToInt(Integer::parseInt).toArray();
    }
}
