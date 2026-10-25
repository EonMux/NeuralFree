package com.eonmux.neuralfree.gguf.hf.lora;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The validation matrix of {@link PeftAdapterConfig}, read from synthetic {@code adapter_config.json} files of
 * temporary directories (HF.md §LA-1): what {@code convert_lora_to_gguf.py} needs is accepted (plain LoRA,
 * {@code rank_pattern}, an {@code alpha_pattern} whose every value is {@code lora_alpha}, the {@code init_lora_weights}
 * values that seed the two factors only), what an adapter GGUF cannot represent is refused with a named error
 * (the PEFT methods, DoRA, {@code modules_to_save}, {@code layer_replication}, {@code lora_bias}, a non-LoRA init), the
 * rslora adjustment doubles as {@code NeuralFree} alpha ({@code lora_alpha} x sqrt(r)), and the base location and the
 * aLoRA invocation values are read leniently.
 */
public class PeftAdapterConfigTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Path root;

    @Before
    public void setup() throws IOException {
        root = tmp.newFolder().toPath();
    }

    private Path dir(String configText) throws IOException {
        Path d = Files.createDirectory(root.resolve(next()));
        Files.write(d.resolve(PeftAdapterConfig.ADAPTER_CONFIG_JSON),
                SyntheticHf.json(configText).getBytes(StandardCharsets.UTF_8));
        return d;
    }

    private Path dir() throws IOException {
        return Files.createDirectory(root.resolve(next()));
    }

    private int index;

    private String next() {
        return "cfg" + ++index;
    }

    private static String lora(String extras) {
        return "{'peft_type':'lora','r':8,'lora_alpha':16" + (extras.isEmpty() ? "" : "," + extras) + "}";
    }

    // ------------------------------------------------------------------ what is accepted

    @Test
    public void readsAPlainLoraConfiguration() throws IOException {
        Path d = dir("{'peft_type':'lora','r':8,'lora_alpha':16,'base_model_name_or_path':'org/base',"
                + "'target_modules':['self_attn.q_proj','self_attn.k_proj'],'use_rslora':false}");
        PeftAdapterConfig c = PeftAdapterConfig.read(d);
        assertEquals(d, c.dir());
        assertEquals(8, c.rank());
        assertEquals(16.0, c.loraAlpha(), 0.0);
        assertEquals(16.0f, c.alpha(), 0f);
        assertEquals("no rslora, so the alpha is the stored value", false, c.usesRslora());
        assertEquals("org/base", c.baseModelNameOrPath());
        assertNull("neither aLoRA form is set", c.aloraInvocationTokens());
        assertNull(c.invocationString());
        assertEquals("the parsed map is kept unchanged", Arrays.asList("self_attn.q_proj", "self_attn.k_proj"),
                c.values().get("target_modules"));
        assertThrows("the configuration is immutable", UnsupportedOperationException.class,
                () -> c.values().put("x", 1));
        assertTrue(c.toString(), c.toString().contains(PeftAdapterConfig.ADAPTER_CONFIG_JSON));
    }

    @Test
    public void acceptsTheInitOptionsThatSeedLoRAMainly() throws IOException {
        for (String init : new String[]{"true", "false", "'gaussian'", "'loftq'", "'olora'"}) {
            PeftAdapterConfig c = PeftAdapterConfig.read(dir(lora("'init_lora_weights':" + init)));
            assertEquals(8, c.rank());
        }
    }

    @Test
    public void acceptsRankPatternAndAnAlphaPatternOfTheStoreAlpha() throws IOException {
        PeftAdapterConfig c = PeftAdapterConfig.read(dir(lora("'rank_pattern':{'model.layers.1':32},"
                + "'alpha_pattern':{'model.layers.1':16}")));
        assertEquals("ranks come from the factor shapes, not from the patterns", 8, c.rank());
        assertEquals(16.0f, c.alpha(), 0f);
    }

    @Test
    public void acceptsTheLoraPeftTypeCaseInsensitive() throws IOException {
        assertEquals(8, PeftAdapterConfig.read(dir("{'peft_type':'LORA','r':8,'lora_alpha':16}")).rank());
        assertEquals(8, PeftAdapterConfig.read(dir("{'peft_type':'Lora','r':8,'lora_alpha':16}")).rank());
    }

    // ------------------------------------------------------------------ the refusals

    @Test
    public void refusesEveryPeftTypeBesidesLora() throws IOException {
        for (String type : new String[]{"'IA3'", "'LOHA'", "'LOKR'", "'ADALORA'", "'XLORA'", "42", "'lora ' "}) {
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> PeftAdapterConfig.read(dir("{'peft_type':" + type + ",'r':8,'lora_alpha':16}")));
            assertTrue(type, e.getMessage().contains("peft_type"));
            assertTrue(type, e.getMessage().contains("is not supported"));
        }
    }

    @Test
    public void refusesAConfigurationWithoutAPeftType() throws IOException {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir("{'r':8,'lora_alpha':16}")));
        assertTrue(e.getMessage(), e.getMessage().contains("has no peft_type"));
    }

    @Test
    public void refusesTheOptionsAnAdapterGgufCannotRepresent() throws IOException {
        GgufFormatException dora = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'use_dora':true"))));
        assertTrue(dora.getMessage(), dora.getMessage().contains("use_dora"));

        GgufFormatException saves = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'modules_to_save':['embed_tokens']"))));
        assertTrue(saves.getMessage(), saves.getMessage().contains("modules_to_save"));

        GgufFormatException replication = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'layer_replication':[[0,1]]"))));
        assertTrue(replication.getMessage(), replication.getMessage().contains("layer_replication"));

        GgufFormatException bias = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'lora_bias':true"))));
        assertTrue(bias.getMessage(), bias.getMessage().contains("lora_bias"));
    }

    @Test
    public void refusesAnInitThatImplementsParametersBesidesLoraAAndB() throws IOException {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'init_lora_weights':'pissa'"))));
        assertTrue(e.getMessage(), e.getMessage().contains("init_lora_weights"));
        assertTrue(e.getMessage(), e.getMessage().contains("lora_A"));
    }

    @Test
    public void refusesAnAlphaPatternThatDiffersFromTheSingleAlpha() throws IOException {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'alpha_pattern':{'model.layers.1':32}"))));
        assertTrue(e.getMessage(), e.getMessage().contains("alpha_pattern"));
        assertTrue(e.getMessage(), e.getMessage().contains("model.layers.1"));
        assertTrue(e.getMessage(), e.getMessage().contains("adapter.lora.alpha"));
        GgufFormatException shape = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'alpha_pattern':[16]"))));
        assertTrue(shape.getMessage(), shape.getMessage().contains("alpha_pattern must be an object"));
    }

    @Test
    public void validatesTheRankAndTheAlpha() throws IOException {
        GgufFormatException r0 = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir("{'peft_type':'lora','r':0,'lora_alpha':16}")));
        assertTrue(r0.getMessage(), r0.getMessage().contains("not a positive rank"));
        GgufFormatException rNone = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir("{'peft_type':'lora','lora_alpha':16}")));
        assertTrue(rNone.getMessage(), rNone.getMessage().contains("lora_alpha needs a numeric r"));
        GgufFormatException rText = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir("{'peft_type':'lora','r':'eight','lora_alpha':16}")));
        assertTrue(rText.getMessage(), rText.getMessage().contains("lora_alpha needs a numeric r"));
        GgufFormatException alphaNone = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir("{'peft_type':'lora','r':8}")));
        assertTrue(alphaNone.getMessage(), alphaNone.getMessage().contains("lora_alpha needs a finite number"));
    }

    // ------------------------------------------------------------------ the rslora alpha

    @Test
    public void doublesTheAlphaForRankStabilizedAdaptersToKeepAlphaOverR() throws IOException {
        PeftAdapterConfig c = PeftAdapterConfig.read(dir("{'peft_type':'lora','r':64,'lora_alpha':16,"
                + "'use_rslora':true}"));
        assertTrue(c.usesRslora());
        assertEquals("the stored lora_alpha stays", 16.0, c.loraAlpha(), 0.0);
        assertEquals("NF alpha = lora_alpha * sqrt(r) = 16 * 8", 128.0f, c.alpha(), 0f);
        assertEquals("alpha / r = 2 = the rslora scale lora_alpha / sqrt(r)", 2.0f, c.alpha() / c.rank(), 1e-6f);
        assertEquals("2 = 16 / 8", 2.0f, 16f / (float) Math.sqrt(64), 1e-6f);
    }

    @Test
    public void keepsTheStoredAlphaOfNonRsloraAdapters() throws IOException {
        PeftAdapterConfig c = PeftAdapterConfig.read(dir("{'peft_type':'lora','r':64,'lora_alpha':16,"
                + "'use_rslora':false}"));
        assertEquals(16.0f, c.alpha(), 0f);
        assertEquals(16f / 64, c.alpha() / c.rank(), 1e-6f);
    }

    // ------------------------------------------------------------------ the aLoRA values

    @Test
    public void readsBothALoRAInvocationForms() throws IOException {
        PeftAdapterConfig c = PeftAdapterConfig.read(dir(lora("'alora_invocation_tokens':[3,5,4096],"
                + "'invocation_string':'<|answer|>'")));
        assertArrayEquals(new Integer[]{3, 5, 4096}, c.aloraInvocationTokens().toArray(new Integer[0]));
        assertThrows("the tokens list is immutable", UnsupportedOperationException.class,
                () -> c.aloraInvocationTokens().add(7));
        assertEquals("<|answer|>", c.invocationString());
    }

    @Test
    public void refusesALoRAInvocationTokensOutsideTheUnsigned32Range() throws IOException {
        GgufFormatException negative = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'alora_invocation_tokens':[-1]"))));
        assertTrue(negative.getMessage(), negative.getMessage().contains("alora_invocation_tokens"));
        assertTrue(negative.getMessage(), negative.getMessage().contains("unsigned 32-bit"));
        GgufFormatException tooBig = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'alora_invocation_tokens':[4294967296]"))));
        assertTrue(tooBig.getMessage(), tooBig.getMessage().contains("unsigned 32-bit"));
        GgufFormatException shape = assertThrows(GgufFormatException.class,
                () -> PeftAdapterConfig.read(dir(lora("'alora_invocation_tokens':'[3]'"))));
        assertTrue(shape.getMessage(), shape.getMessage().contains("must be an array of token ids"));
    }

    // ------------------------------------------------------------------ the file readings

    @Test
    public void refusesADirectoryWithoutTheAdapterConfiguration() throws IOException {
        IOException e = assertThrows(IOException.class, () -> PeftAdapterConfig.read(dir()));
        assertTrue(e.getMessage(), e.getMessage().contains(PeftAdapterConfig.ADAPTER_CONFIG_JSON + " is missing"));
    }

    @Test
    public void refusesTheFileOnBadJsonOrWrongShape() throws IOException {
        Path bad = dir();
        Files.write(bad.resolve(PeftAdapterConfig.ADAPTER_CONFIG_JSON), "{'peft_type':".getBytes(StandardCharsets.UTF_8));
        assertThrows("unparsable JSON", IOException.class, () -> PeftAdapterConfig.read(bad));

        Path array = dir();
        Files.write(array.resolve(PeftAdapterConfig.ADAPTER_CONFIG_JSON), "[1]".getBytes(StandardCharsets.UTF_8));
        IOException e = assertThrows(IOException.class, () -> PeftAdapterConfig.read(array));
        assertTrue(e.getMessage(), e.getMessage().contains("expected a JSON object"));
    }
}
