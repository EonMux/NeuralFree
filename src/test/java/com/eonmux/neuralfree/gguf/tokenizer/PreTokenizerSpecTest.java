package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Dialect;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Kind;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.PrependScheme;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Step;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link PreTokenizerSpec} construction, validation, canonical JSON and its registry use. */
public class PreTokenizerSpecTest {

    private static PreTokenizerSpec everyKind() {
        List<Step> steps = Arrays.asList(
                Step.regex("\\p{Han}+", Dialect.ONIGURUMA, Behaviour.MERGED_WITH_NEXT, true),
                Step.literal("-", Behaviour.CONTIGUOUS, false),
                Step.whitespace(),
                Step.whitespaceSplit(),
                Step.punctuation(Behaviour.REMOVED),
                Step.charDelimiter('|'),
                Step.digits(true),
                Step.metaspace(0x2581, PrependScheme.FIRST, false),
                Step.unicodeScripts(),
                Step.bert(),
                Step.byteLevel());
        return new PreTokenizerSpec(steps, true, true, true, false);
    }

    @Test
    public void canonicalJsonRoundTripsEveryKind() {
        PreTokenizerSpec spec = everyKind();
        String json = spec.toCanonicalJson();
        PreTokenizerSpec back = PreTokenizerSpec.fromCanonicalJson(json);
        assertEquals(spec, back);
        assertEquals(spec.hashCode(), back.hashCode());
        assertEquals(json, back.toCanonicalJson());
    }

    @Test
    public void canonicalJsonRoundTripsEveryRegistryName() {
        for (String name : PreTokenizerRegistry.names()) {
            PreTokenizerSpec spec = PreTokenizerRegistry.specFor(name);
            assertEquals(name, spec, PreTokenizerSpec.fromCanonicalJson(spec.toCanonicalJson()));
        }
    }

    @Test
    public void canonicalJsonIsSortedAndOmitsUnusedFields() {
        PreTokenizerSpec spec = new PreTokenizerSpec(Collections.singletonList(Step.digits(false)), false, false, false,
                false);
        assertEquals("{\"add_prefix_space\":false,\"byte_encode\":false,"
                + "\"steps\":[{\"individual_digits\":false,\"kind\":\"DIGITS\"}],"
                + "\"trim_offsets\":false,\"use_regex\":false}", spec.toCanonicalJson());
    }

    @Test
    public void isolatedStepsNeverInvert() {
        Step inverted = Step.regex("a", Dialect.ONIGURUMA, Behaviour.ISOLATED, true);
        assertFalse(inverted.invert());
        assertEquals(Step.regex("a", Dialect.ONIGURUMA, Behaviour.ISOLATED, false), inverted);
    }

    @Test
    public void literalStepsAreEscapedOnigurumaRegexes() {
        Step step = Step.literal("a.b", Behaviour.ISOLATED, false);
        assertEquals(Kind.REGEX_SPLIT, step.kind());
        assertEquals(Dialect.ONIGURUMA, step.dialect());
        assertEquals("a\\.b", step.pattern());
    }

    @Test
    public void registryNamesAreLlamaCppIsolatedSpecs() {
        PreTokenizerSpec spec = PreTokenizerRegistry.specFor("llama-bpe");
        assertTrue(spec.byteEncode());
        assertEquals(PreTokenizerType.LLAMA3.regexes().size(), spec.steps().size());
        for (int i = 0; i < spec.steps().size(); i++) {
            Step s = spec.steps().get(i);
            assertEquals(Kind.REGEX_SPLIT, s.kind());
            assertEquals(Dialect.LLAMA_CPP, s.dialect());
            assertEquals(Behaviour.ISOLATED, s.behaviour());
            assertEquals(PreTokenizerType.LLAMA3.regexes().get(i), s.pattern());
        }
        assertSame(PreTokenizerRegistry.forName("llama-bpe"), PreTokenizerRegistry.forName("llama-bpe"));
        assertEquals(spec, PreTokenizerRegistry.forName("llama-bpe").spec());
        assertThrows(IllegalArgumentException.class, () -> PreTokenizerRegistry.specFor("no-such-pre"));
    }

    @Test
    public void fromSpecUsesDefaultFlags() {
        PreTokenizerRegistry.PreTokenizer p = PreTokenizerRegistry.fromSpec(everyKind());
        assertEquals(PreTokenizerRegistry.CUSTOM_NAME, p.name());
        assertNull(p.type());
        assertTrue(p.cleanSpaces());
        assertFalse(p.ignoreMerges());
        assertFalse(p.escapeWhitespaces());
        assertNull(p.addBos());
        assertFalse(p.forceAddBos());
        assertFalse(p.suppressBos());
        assertFalse(p.wholeNewlineWords());
        assertFalse(p.metaspaceWords());
        assertTrue(p.byteEncode());
        assertEquals(Arrays.asList("\\p{Han}+", "\\-"), p.regexes());
    }

    @Test
    public void wordsByteEncodeAfterTheSplit() {
        PreTokenizerRegistry.PreTokenizer p = PreTokenizerRegistry.forName("llama-bpe");
        assertEquals(Arrays.asList("Hello", " world"), p.split("Hello world"));
        assertEquals(Arrays.asList("Hello", "Ġworld"), p.words("Hello world"));
    }

    @Test
    public void constructorRejectsInconsistentSpecs() {
        List<Step> twoByteLevels = Arrays.asList(Step.byteLevel(), Step.byteLevel());
        assertThrows(IllegalArgumentException.class, () -> new PreTokenizerSpec(twoByteLevels, true, false, false, true));
        List<Step> oneByteLevel = Collections.singletonList(Step.byteLevel());
        assertThrows(IllegalArgumentException.class, () -> new PreTokenizerSpec(oneByteLevel, false, false, false, true));
        List<Step> none = Collections.singletonList(Step.whitespace());
        assertThrows(IllegalArgumentException.class, () -> new PreTokenizerSpec(none, true, true, false, false));
        assertThrows(IllegalArgumentException.class,
                () -> new PreTokenizerSpec(Collections.singletonList(null), false, false, false, false));
        assertThrows(IllegalArgumentException.class, () -> Step.charDelimiter(0xD800));
    }

    @Test
    public void invalidRegexFailsWhenCompiled() {
        PreTokenizerSpec spec = new PreTokenizerSpec(
                Collections.singletonList(Step.regex("(", Dialect.ONIGURUMA, Behaviour.ISOLATED, false)),
                false, false, false, false);
        assertThrows(IllegalArgumentException.class, () -> PreTokenizerRegistry.fromSpec(spec));
    }

    @Test
    public void fromCanonicalJsonRejectsMalformedInput() {
        assertThrows(GgufFormatException.class, () -> PreTokenizerSpec.fromCanonicalJson("{"));
        assertThrows(GgufFormatException.class, () -> PreTokenizerSpec.fromCanonicalJson("[]"));
        String valid = new PreTokenizerSpec(Collections.singletonList(Step.whitespace()), false, false, false, false)
                .toCanonicalJson();
        assertThrows(GgufFormatException.class,
                () -> PreTokenizerSpec.fromCanonicalJson(valid.replace("\"use_regex\":false", "\"use_regex\":false,\"x\":1")));
        assertThrows(GgufFormatException.class,
                () -> PreTokenizerSpec.fromCanonicalJson(valid.replace("WHITESPACE", "NO_SUCH_KIND")));
        assertThrows(GgufFormatException.class,
                () -> PreTokenizerSpec.fromCanonicalJson(valid.replace(",\"trim_offsets\":false", "")));
    }
}
