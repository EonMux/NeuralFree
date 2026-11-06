package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;
import com.eonmux.neuralfree.gguf.engine.draft.VerifyMode;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link SpeculativeOptions}, {@link DraftTypes} and the speculative keys of {@link EngineConfig}: every drafter and
 * verifier setting is reachable from {@code neuralfree.*} properties under llama.cpp's and vLLM's meanings, the type
 * list keeps llama.cpp's ask order, and settings nothing would read are refused by name.
 */
public class SpeculativeOptionsTest {

    private static Properties props(String... kv) {
        Properties p = new Properties();
        for (int i = 0; i < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return p;
    }

    private static IllegalArgumentException refused(String... kv) {
        return assertThrows(IllegalArgumentException.class, () -> EngineConfig.fromProperties(props(kv)));
    }

    @Test
    public void theDefaultsAreTheDrafterDefaultsAndMatchVerification() {
        SpeculativeOptions d = SpeculativeOptions.defaults();
        assertTrue(d.isDefault());
        assertEquals(VerifyMode.MATCH, d.verifyMode());
        // llama.cpp's defaults (common/common.h): ngram_map 12 / 48, ngram_mod 24 / 64 / 48, the cache's 8
        NgramDrafter.Options simple = d.ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(Arrays.asList(12, 12, 48, 0),
                Arrays.asList(simple.sizeN(), simple.sizeNMin(), simple.sizeM(), simple.draftMin()));
        for (NgramDrafter.Kind map : new NgramDrafter.Kind[]{NgramDrafter.Kind.MAP_K, NgramDrafter.Kind.MAP_K4V}) {
            NgramDrafter.Options o = d.ngramOptions(map);
            assertEquals(Arrays.asList(12, 48, 1), Arrays.asList(o.sizeN(), o.sizeM(), o.minHits()));
        }
        NgramDrafter.Options mod = d.ngramOptions(NgramDrafter.Kind.MOD);
        assertEquals(Arrays.asList(24, 64, 48), Arrays.asList(mod.sizeN(), mod.sizeM(), mod.draftMin()));
        assertEquals(4 * 1024 * 1024, mod.modTableSize());
        assertEquals(8, d.ngramOptions(NgramDrafter.Kind.CACHE).sizeM());
        assertEquals(10000, d.suffixOptions().maxCachedRequests());
        assertNull(d.lookupCacheStatic());
        assertNull(d.synthLength());
        assertTrue(SpeculativeOptions.fromProperties(new Properties()).isDefault());
        assertTrue(EngineConfig.defaults().speculativeOptions().isDefault());
    }

    /** {@code neuralfree.draft.verify.device} (llama.cpp {@code -bs}): on by default, off by name, a typo refused. */
    @Test
    public void deviceVerificationIsOnByDefaultAndSwitchedByItsProperty() {
        assertTrue(SpeculativeOptions.defaults().verifyOnDevice());
        SpeculativeOptions off = SpeculativeOptions.fromProperties(props("neuralfree.draft.verify.device", "false"));
        assertFalse(off.verifyOnDevice());
        assertFalse(off.isDefault());
        assertTrue(off.toString().contains("verifyOnDevice=false"));
        assertFalse(off.toBuilder().build().verifyOnDevice());
        assertTrue(EngineConfig.defaults().speculativeOptions().verifyOnDevice());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.verify.device", "yes")));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.verify.device"));
    }

    @Test
    public void deviceDflashDraftingIsOnByDefaultAndSwitchedByItsProperty() {
        assertTrue(SpeculativeOptions.defaults().dflashOnDevice());
        SpeculativeOptions off = SpeculativeOptions.fromProperties(props("neuralfree.draft.dflash.device", "false"));
        assertFalse(off.dflashOnDevice());
        assertTrue(off.verifyOnDevice());
        assertFalse(off.isDefault());
        assertTrue(off.toString().contains("dflashOnDevice=false"));
        assertFalse(off.toBuilder().build().dflashOnDevice());
        assertTrue(EngineConfig.defaults().speculativeOptions().dflashOnDevice());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.dflash.device", "on")));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.dflash.device"));
    }

    /** {@code neuralfree.weights.w12} streams the lossless target weights opt-in; the head needs it. The old
     * {@code neuralfree.draft.w12[.head]} spellings are accepted as aliases. */
    @Test
    public void w12StreamingIsOptInAndTheHeadNeedsIt() {
        assertFalse(SpeculativeOptions.defaults().w12());
        assertFalse(SpeculativeOptions.defaults().w12Head());
        SpeculativeOptions on = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_W12, "true"));
        assertTrue(on.w12());
        assertFalse(on.w12Head());
        assertFalse(on.isDefault());
        assertTrue(on.toString().contains("w12=true"));
        assertFalse(on.toBuilder().build().w12Head());
        SpeculativeOptions head = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_W12, "true", SpeculativeOptions.PROP_W12_HEAD, "true"));
        assertTrue(head.w12());
        assertTrue(head.w12Head());
        // the head alone is refused by name, from properties and from the builder alike
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_W12_HEAD, "true")));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_W12_HEAD));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_W12));
        IllegalArgumentException b = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.defaults().toBuilder().w12Head(true).build());
        assertTrue(b.getMessage(), b.getMessage().contains(SpeculativeOptions.PROP_W12_HEAD));
        assertTrue(b.getMessage(), b.getMessage().contains(SpeculativeOptions.PROP_W12));
        // a value the boolean property reader does not take is refused by name
        IllegalArgumentException v = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_W12, "maybe")));
        assertTrue(v.getMessage(), v.getMessage().contains(SpeculativeOptions.PROP_W12));
    }

    /** The old {@code neuralfree.draft.w12[.head]} spellings still set the options, and the new key wins
     * when both are set. */
    @Test
    public void theDraftW12SpellingsRemainAliases() {
        SpeculativeOptions on = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_W12_ALIAS, "true"));
        assertTrue(on.w12());
        assertFalse(on.w12Head());
        SpeculativeOptions head = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_W12_ALIAS, "true", SpeculativeOptions.PROP_W12_HEAD_ALIAS, "true"));
        assertTrue(head.w12());
        assertTrue(head.w12Head());
        // the primary key wins when both are set
        SpeculativeOptions both = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_W12, "true", SpeculativeOptions.PROP_W12_ALIAS, "false"));
        assertTrue(both.w12());
        SpeculativeOptions aliasOnly = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_W12_ALIAS, "true", SpeculativeOptions.PROP_W12, "false"));
        assertFalse(aliasOnly.w12());
        // an unusable value names the alias it came from
        IllegalArgumentException v = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_W12_ALIAS, "maybe")));
        assertTrue(v.getMessage(), v.getMessage().contains(SpeculativeOptions.PROP_W12_ALIAS));
    }

    /** {@code neuralfree.weights.w12.f16} holds an F16 tensor to the bfloat16 grid: refuse (default) keeps the
     * W12 streaming lossless, round rounds lossily; an unusable value is refused naming the property. */
    @Test
    public void theW12F16GridModeIsReadFromItsProperty() {
        assertEquals(W12F16Mode.REFUSE, SpeculativeOptions.defaults().w12F16());
        SpeculativeOptions round = SpeculativeOptions.fromProperties(
                props(SpeculativeOptions.PROP_W12_F16, "round"));
        assertEquals(W12F16Mode.ROUND, round.w12F16());
        assertTrue(round.toString().contains("w12F16=round"));
        assertEquals(W12F16Mode.ROUND, round.toBuilder().build().w12F16());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_W12_F16, "lossy")));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_W12_F16));
    }

    /** {@code neuralfree.draft.tree.nodes} fixes the verify width; it survives a restatement and refuses ranges. */
    @Test
    public void theTreeVerifyWidthIsAFixedNodeCount() {
        assertEquals(SpeculativeOptions.DEFAULT_TREE_NODES, SpeculativeOptions.defaults().treeNodes());
        SpeculativeOptions o = SpeculativeOptions.fromProperties(props(
                "neuralfree.draft.tree", "true", "neuralfree.draft.tree.nodes", "8"));
        assertTrue(o.treeVerify());
        assertEquals(8, o.treeNodes());
        assertFalse(o.isDefault());
        assertTrue(o.toString(), o.toString().contains("treeNodes=8"));
        assertEquals("the width survives a restatement", o.toString(), o.toBuilder().build().toString());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.tree.nodes", "64")));
        assertTrue(e.getMessage(), e.getMessage().contains("neuralfree.draft.tree.nodes"));
        assertTrue(e.getMessage(), e.getMessage().contains("63"));
    }

    @Test
    public void everyNgramSizeIsReadFromItsProperty() {
        SpeculativeOptions o = SpeculativeOptions.fromProperties(props(
                "neuralfree.draft.ngram.simple.size.n", "4", "neuralfree.draft.ngram.simple.size.m", "12",
                "neuralfree.draft.ngram.map.k.size.n", "5", "neuralfree.draft.ngram.map.k.size.m", "9",
                "neuralfree.draft.ngram.map.k.min.hits", "2",
                "neuralfree.draft.ngram.map.k4v.size.n", "6", "neuralfree.draft.ngram.map.k4v.size.m", "10",
                "neuralfree.draft.ngram.map.k4v.min.hits", "3",
                "neuralfree.draft.ngram.mod.n.match", "5", "neuralfree.draft.ngram.mod.n.max", "16",
                "neuralfree.draft.ngram.mod.n.min", "4"));
        NgramDrafter.Options simple = o.ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(4, simple.sizeN());
        assertEquals("one lookup size: llama.cpp's search", 4, simple.sizeNMin());
        assertFalse(simple.lookupRange());
        assertEquals(12, simple.sizeM());
        NgramDrafter.Options mapK = o.ngramOptions(NgramDrafter.Kind.MAP_K);
        assertEquals(Arrays.asList(5, 9, 2), Arrays.asList(mapK.sizeN(), mapK.sizeM(), mapK.minHits()));
        NgramDrafter.Options k4v = o.ngramOptions(NgramDrafter.Kind.MAP_K4V);
        assertEquals(Arrays.asList(6, 10, 3), Arrays.asList(k4v.sizeN(), k4v.sizeM(), k4v.minHits()));
        NgramDrafter.Options mod = o.ngramOptions(NgramDrafter.Kind.MOD);
        assertEquals(Arrays.asList(5, 16, 4), Arrays.asList(mod.sizeN(), mod.sizeM(), mod.draftMin()));
        assertFalse(o.isDefault());
        assertTrue(o.toString(), o.toString().contains("mod=[n=5, m=16, minHits=1, min=4]"));
    }

    @Test
    public void thePromptLookupRangeIsReadFromItsProperty() {
        SpeculativeOptions o = SpeculativeOptions.fromProperties(props(
                "neuralfree.draft.ngram.simple.size.n", "5", "neuralfree.draft.ngram.simple.size.n.min", "2"));
        NgramDrafter.Options simple = o.ngramOptions(NgramDrafter.Kind.SIMPLE);
        assertEquals(5, simple.sizeN());
        assertEquals(2, simple.sizeNMin());
        assertTrue(simple.lookupRange());
        assertTrue(o.toString(), o.toString().contains("simple=[n=2..5,"));
        assertEquals("the range survives a restatement", o.toString(), o.toBuilder().build().toString());
        assertEquals("a single size follows a later sizeN", 7,
                SpeculativeOptions.defaults().ngramOptions(NgramDrafter.Kind.SIMPLE).toBuilder().sizeN(7).build()
                        .sizeNMin());
        assertEquals("vLLM's size-1 lookup is a range", 1, SpeculativeOptions.fromProperties(props(
                "neuralfree.draft.ngram.simple.size.n", "3", "neuralfree.draft.ngram.simple.size.n.min", "1"))
                .ngramOptions(NgramDrafter.Kind.SIMPLE).sizeNMin());

        IllegalArgumentException inverted = refused("neuralfree.draft.ngram.simple.size.n", "3",
                "neuralfree.draft.ngram.simple.size.n.min", "4");
        assertTrue(inverted.getMessage(), inverted.getMessage().startsWith("neuralfree.draft.ngram.simple: "));
        assertTrue(inverted.getMessage(), inverted.getMessage().contains("sizeNMin 4 exceeds sizeN 3"));
        IllegalArgumentException zero = refused("neuralfree.draft.ngram.simple.size.n.min", "0");
        assertTrue(zero.getMessage(), zero.getMessage().startsWith("neuralfree.draft.ngram.simple.size.n.min=0"));
        assertThrows("only ngram-simple searches a range", IllegalArgumentException.class,
                () -> NgramDrafter.Options.builder(NgramDrafter.Kind.MAP_K).sizeNMin(2).build());
    }

    @Test
    public void outOfRangeSizesAreRefusedNamingTheProperty() {
        IllegalArgumentException big = refused("neuralfree.draft.ngram.simple.size.m", "1000");
        assertTrue(big.getMessage(), big.getMessage().startsWith("neuralfree.draft.ngram.simple.size.m=1000"));
        IllegalArgumentException one = refused("neuralfree.draft.ngram.simple.size.n", "1");
        assertTrue(one.getMessage(), one.getMessage().startsWith("neuralfree.draft.ngram.simple"));
        IllegalArgumentException text = refused("neuralfree.draft.ngram.map.k.min.hits", "many");
        assertTrue(text.getMessage(), text.getMessage().contains("neuralfree.draft.ngram.map.k.min.hits"));
        IllegalArgumentException mod = refused("neuralfree.draft.ngram.mod.n.min", "-1");
        assertTrue(mod.getMessage(), mod.getMessage().contains("neuralfree.draft.ngram.mod.n.min=-1"));
        assertEquals("llama.cpp keeps a full n_max chain below n_min, so a floor above the cap is valid", 9,
                SpeculativeOptions.fromProperties(props("neuralfree.draft.ngram.mod.n.max", "8",
                        "neuralfree.draft.ngram.mod.n.min", "9")).ngramOptions(NgramDrafter.Kind.MOD).draftMin());
        IllegalArgumentException cached = refused("neuralfree.draft.suffix.max.cached.requests", "-1");
        assertTrue(cached.getMessage(), cached.getMessage().contains("neuralfree.draft.suffix.max.cached.requests"));
        IllegalArgumentException prob = refused("neuralfree.draft.suffix.min.token.prob", "1.5");
        assertTrue(prob.getMessage(), prob.getMessage().contains("neuralfree.draft.suffix.min.token.prob"));
    }

    @Test
    public void suffixLimitsAndLookupCacheFilesAreRead() {
        SpeculativeOptions o = SpeculativeOptions.fromProperties(props(
                "neuralfree.draft.suffix.max.tree.depth", "24", "neuralfree.draft.suffix.max.spec.factor", "2.5",
                "neuralfree.draft.suffix.max.spec.offset", "1", "neuralfree.draft.suffix.min.token.prob", "0.2",
                "neuralfree.draft.suffix.max.cached.requests", "3",
                "neuralfree.draft.lookup.cache.static", "/tmp/static.bin",
                "neuralfree.draft.lookup.cache.dynamic", "/tmp/dynamic.bin"));
        assertEquals(24, o.suffixOptions().maxTreeDepth());
        assertEquals(2.5, o.suffixOptions().maxSpecFactor(), 0.0);
        assertEquals(1.0, o.suffixOptions().maxSpecOffset(), 0.0);
        assertEquals(0.2, o.suffixOptions().minTokenProb(), 0.0);
        assertEquals(3, o.suffixOptions().maxCachedRequests());
        assertTrue(o.toString(), o.toString().contains("cachedRequests=3"));
        assertEquals(0, SpeculativeOptions.fromProperties(props("neuralfree.draft.suffix.max.cached.requests", "0"))
                .suffixOptions().maxCachedRequests());
        assertEquals("/tmp/static.bin", o.lookupCacheStatic());
        assertEquals("/tmp/dynamic.bin", o.lookupCacheDynamic());
        assertEquals(o.toString(), o.toBuilder().build().toString());
    }

    @Test
    public void theVerificationModeAndItsSettingsAreRead() {
        SpeculativeOptions fly = SpeculativeOptions.fromProperties(props("neuralfree.draft.verify", "FLY",
                "neuralfree.draft.fly.window", "4", "neuralfree.draft.fly.entropy.threshold", "0.5",
                "neuralfree.draft.fly.entropy.top.k", "5"));
        assertEquals(VerifyMode.FLY, fly.verifyMode());
        assertEquals(4, fly.flyWindow());
        assertEquals(0.5, fly.flyEntropyThreshold(), 0.0);
        assertEquals(5, fly.flyEntropyTopK());
        assertEquals(VerifyMode.REJECTION, SpeculativeOptions.fromProperties(
                props("neuralfree.draft.verify", "rejection")).verifyMode());
        IllegalArgumentException unknown = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.verify", "typical")));
        assertTrue(unknown.getMessage(), unknown.getMessage().startsWith("neuralfree.draft.verify=typical"));
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.fly.entropy.top.k", "0")));
    }

    @Test
    public void theDraftSplitProbabilityIsReadAndBounded() {
        assertEquals("llama.cpp's --spec-draft-p-split default", SpeculativeOptions.DEFAULT_P_SPLIT,
                SpeculativeOptions.defaults().pSplit(), 0.0);
        SpeculativeOptions split = SpeculativeOptions.fromProperties(
                props(SpeculativeOptions.PROP_P_SPLIT, "0.25"));
        assertEquals(0.25, split.pSplit(), 0.0);
        assertEquals(split.pSplit(), split.toBuilder().build().pSplit(), 0.0);
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_P_SPLIT, "1.5")));
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_P_SPLIT, "high")));
        assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.defaults().toBuilder().pSplit(-0.1));
    }

    @Test
    public void syntheticAcceptanceSelectsSyntheticVerificationAsLlamaCppDoes() {
        SpeculativeOptions len = SpeculativeOptions.fromProperties(props("neuralfree.draft.synth.len", "2.5"));
        assertEquals(VerifyMode.SYNTHETIC, len.verifyMode());
        assertEquals(2.5, len.synthLength(), 0.0);
        SpeculativeOptions rates = SpeculativeOptions.fromProperties(
                props("neuralfree.draft.synth.rates", "0.9, 0.6,0.3"));
        assertArrayEquals(new double[]{0.9, 0.6, 0.3}, rates.synthRates(), 0.0);

        assertThrows(IllegalArgumentException.class, () -> SpeculativeOptions.fromProperties(
                props("neuralfree.draft.synth.len", "2", "neuralfree.draft.synth.rates", "0.5")));
        IllegalArgumentException other = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.synth.len", "2",
                        "neuralfree.draft.verify", "match")));
        assertTrue(other.getMessage(), other.getMessage().contains("neuralfree.draft.synth.len"));
        IllegalArgumentException bare = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.verify", "synthetic")));
        assertTrue(bare.getMessage(), bare.getMessage().contains("neuralfree.draft.synth"));
        IllegalArgumentException increasing = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.fromProperties(props("neuralfree.draft.synth.rates", "0.3,0.6")));
        assertTrue(increasing.getMessage(), increasing.getMessage().startsWith("neuralfree.draft.synth.rates"));
    }

    /** {@code neuralfree.draft.head.*} and {@code neuralfree.draft.prec}: the derived candidate head settings. */
    @Test
    public void theDerivedDraftHeadSettingsAreReadFromTheirProperties() {
        assertNull("off unless a head property is set",
                SpeculativeOptions.fromProperties(props("neuralfree.draft.head.vocab", "0")).draftHead());
        SpeculativeOptions o = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB, "98304",
                SpeculativeOptions.PROP_DRAFT_HEAD_FREQ, "/tmp/ranks.json",
                SpeculativeOptions.PROP_DRAFT_HEAD_KEEP, " 128, 42,7 ",
                SpeculativeOptions.PROP_DRAFT_HEAD_PREC, "fp8",
                SpeculativeOptions.PROP_DRAFT_PREC, "q8_0"));
        DraftHeadOptions head = o.draftHead();
        assertEquals(98304, head.vocabSize());
        assertEquals("/tmp/ranks.json", head.frequencyFile());
        assertArrayEquals(new int[] {128, 42, 7}, head.keepIds());
        assertEquals(DraftHeadOptions.Prec.FP8, head.headPrec());
        assertEquals(DraftHeadOptions.Prec.Q8_0, head.matricesPrec());
        SpeculativeOptions vocabOnly = SpeculativeOptions.fromProperties(
                props(SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB, "256"));
        assertEquals(256, vocabOnly.draftHead().vocabSize());
        assertNull(vocabOnly.draftHead().headPrec());
        assertEquals(DraftHeadOptions.Prec.NVFP4, SpeculativeOptions.fromProperties(
                props(SpeculativeOptions.PROP_DRAFT_HEAD_PREC, "nvfp4")).draftHead().headPrec());
        assertEquals(DraftHeadOptions.Prec.NVFP4, SpeculativeOptions.fromProperties(
                props(SpeculativeOptions.PROP_DRAFT_PREC, "NVFP4")).draftHead().matricesPrec());
        assertEquals("the round trip keeps the settings", head.describe(),
                SpeculativeOptions.fromProperties(props(
                        SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB, "98304",
                        SpeculativeOptions.PROP_DRAFT_HEAD_FREQ, "/tmp/ranks.json",
                        SpeculativeOptions.PROP_DRAFT_HEAD_KEEP, "128,42,7",
                        SpeculativeOptions.PROP_DRAFT_HEAD_PREC, "fp8",
                        SpeculativeOptions.PROP_DRAFT_PREC, "q8_0")).draftHead().describe());
        assertTrue("the settings appear in the option summary",
                o.toString().contains("draftHead{vocab=98304"));
    }

    /** The derived head settings refuse what they cannot serve, naming the property and the value. */
    @Test
    public void theDerivedDraftHeadSettingsRefuseByName() {
        assertTrue(refused(SpeculativeOptions.PROP_DRAFT_HEAD_PREC, "int4").getMessage()
                .contains("neuralfree.draft.head.prec"));
        assertTrue(refused(SpeculativeOptions.PROP_DRAFT_HEAD_PREC, "int4").getMessage().contains("int4"));
        assertTrue(refused(SpeculativeOptions.PROP_DRAFT_PREC, "int4").getMessage()
                .contains("neuralfree.draft.prec"));
        String keep = refused(SpeculativeOptions.PROP_DRAFT_HEAD_KEEP, "4,x,9").getMessage();
        assertTrue(keep, keep.contains("neuralfree.draft.head.keep"));
        assertTrue(keep, keep.contains("x"));
        String empty = refused(SpeculativeOptions.PROP_DRAFT_HEAD_KEEP, "4,,9").getMessage();
        assertTrue(empty, empty.contains("neuralfree.draft.head.keep"));
        assertTrue(refused(SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB, "big").getMessage()
                .contains(SpeculativeOptions.PROP_DRAFT_HEAD_VOCAB));
    }

    /** {@link SpeculativeOptions#parseIds}: comma-separated ids, whitespace kept out, entries refused by index. */
    @Test
    public void commaSeparatedIdsParseAndRefuseByEntry() {
        assertArrayEquals(new int[] {4, 8, 15}, SpeculativeOptions.parseIds("k", "4, 8,15"));
        assertArrayEquals(new int[] {16}, SpeculativeOptions.parseIds("k", " 16 "));
        IllegalArgumentException empty = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.parseIds("k", "4,,8"));
        assertTrue(empty.getMessage(), empty.getMessage().contains("k"));
        IllegalArgumentException word = assertThrows(IllegalArgumentException.class,
                () -> SpeculativeOptions.parseIds("k", "4,eight"));
        assertTrue(word.getMessage(), word.getMessage().contains("eight"));
    }

    @Test
    public void restatedOptionsKeepADerivedModeDerived() {
        SpeculativeOptions len = SpeculativeOptions.defaults().toBuilder().synthLength(2.0).build();
        assertEquals("a synthetic length over the defaults selects synthetic", VerifyMode.SYNTHETIC,
                len.verifyMode());
        SpeculativeOptions back = len.toBuilder().synthLength(null).build();
        assertEquals("dropping it returns to match", VerifyMode.MATCH, back.verifyMode());
        SpeculativeOptions block = SpeculativeOptions.builder().verifyMode(VerifyMode.BLOCK).build();
        assertThrows("an explicit mode stays explicit", IllegalArgumentException.class,
                () -> block.toBuilder().synthLength(2.0).build());
        NgramDrafter.Options cache = SpeculativeOptions.defaults().ngramOptions(NgramDrafter.Kind.CACHE);
        assertEquals(cache.toString(), cache.toBuilder().build().toString());
    }

    @Test
    public void theTypeListKeepsLlamaCppAskOrder() {
        List<DraftType> types = DraftTypes.parse("suffix, draft-simple,ngram-cache,ngram-simple,suffix,mod");
        assertEquals(Arrays.asList(DraftType.NGRAM, DraftType.NGRAM_MOD, DraftType.NGRAM_CACHE, DraftType.SUFFIX,
                DraftType.MODEL), types);
        assertEquals("ngram-simple,ngram-mod,ngram-cache,suffix,draft-simple", DraftTypes.names(types));
        assertEquals(DraftType.MODEL, DraftTypes.fileType(types));
        assertEquals(DraftType.AUTO, DraftTypes.fileType(DraftTypes.parse("ngram-map-k4v")));
        assertEquals(Arrays.asList(DraftType.NGRAM_MAP_K, DraftType.NGRAM_MAP_K4V),
                DraftTypes.lookups(DraftTypes.parse("map-k4v,ngram-map")));
        IllegalArgumentException twoFiles = assertThrows(IllegalArgumentException.class,
                () -> DraftTypes.parse("draft-mtp,draft-simple"));
        assertTrue(twoFiles.getMessage(), twoFiles.getMessage().contains("draft-mtp"));
        assertThrows(IllegalArgumentException.class, () -> DraftTypes.parse(" , "));
        assertThrows(IllegalArgumentException.class, () -> DraftTypes.parse("ngram-simple,lookahead"));
    }

    @Test
    public void theEngineConfigurationCarriesTheListAndTheOptions() {
        EngineConfig c = EngineConfig.fromProperties(props("neuralfree.draft.type", "suffix,ngram-simple,draft-mtp",
                "neuralfree.draft.verify", "block"));
        assertEquals(Arrays.asList(DraftType.NGRAM, DraftType.SUFFIX, DraftType.MTP), c.draftTypes());
        assertEquals(Arrays.asList(DraftType.NGRAM, DraftType.SUFFIX), c.lookupTypes());
        assertEquals(DraftType.MTP, c.draftType());
        assertTrue("draft-mtp without a draft file drafts with the model's own head", c.mtp());
        assertTrue(c.speculative());
        assertEquals(VerifyMode.BLOCK, c.speculativeOptions().verifyMode());
        EngineConfig copy = c.toBuilder().build();
        assertEquals(c.draftTypes(), copy.draftTypes());
        assertEquals(c.speculativeOptions().toString(), copy.speculativeOptions().toString());
        assertTrue(c.toString(), c.toString().contains("draftType=ngram-simple,suffix,draft-mtp"));

        EngineConfig lookupOnly = EngineConfig.fromProperties(props("neuralfree.draft.type", "ngram-cache"));
        assertEquals(DraftType.AUTO, lookupOnly.draftType());
        assertFalse(lookupOnly.mtp());
        assertTrue(lookupOnly.speculative());
    }

    @Test
    public void settingsNothingWouldReadAreRefusedByName() {
        IllegalArgumentException file = refused("neuralfree.draft", "draft.gguf", "neuralfree.draft.type",
                "ngram-simple");
        assertTrue(file.getMessage(), file.getMessage().startsWith("neuralfree.draft is set"));
        // a draft-file type beside the lookups runs the file after them
        assertEquals(DraftType.MODEL, EngineConfig.fromProperties(props("neuralfree.draft", "draft.gguf",
                "neuralfree.draft.type", "ngram-simple,draft-simple")).draftType());

        IllegalArgumentException cache = refused("neuralfree.draft.lookup.cache.static", "/tmp/s.bin",
                "neuralfree.draft.type", "ngram-simple");
        assertTrue(cache.getMessage(), cache.getMessage().contains("ngram-cache"));
        IllegalArgumentException types = refused("neuralfree.draft.type", "draft-simple,draft-mtp");
        assertTrue(types.getMessage(), types.getMessage().startsWith("neuralfree.draft.type"));
    }

    @Test
    public void theDraftSampleMethodIsReadAsVllmsDraftSampleMethod() {
        assertFalse(SpeculativeOptions.defaults().probabilisticDraft());
        SpeculativeOptions probabilistic = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD, "Probabilistic"));
        assertTrue(probabilistic.probabilisticDraft());
        assertFalse(probabilistic.isDefault());
        assertTrue(probabilistic.toString(), probabilistic.toString().contains("probabilistic"));
        assertTrue(probabilistic.toBuilder().build().probabilisticDraft());
        assertTrue(SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD, "greedy"))
                .isDefault());
        IllegalArgumentException bad = refused(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD, "beam");
        assertTrue(bad.getMessage(), bad.getMessage().startsWith(SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=beam"));
    }

    @Test
    public void heterogeneousVocabDraftingIsReadAsVllmsDraftModelOption() {
        assertFalse(SpeculativeOptions.defaults().heterogeneousVocab());
        SpeculativeOptions on = SpeculativeOptions.fromProperties(props(
                SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB, "true"));
        assertTrue(on.heterogeneousVocab());
        assertFalse(on.isDefault());
        assertTrue(on.toString(), on.toString().contains("heterogeneousVocab=true"));
        assertTrue(on.toBuilder().build().heterogeneousVocab());
        assertTrue(SpeculativeOptions.fromProperties(props(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB, "false"))
                .isDefault());
    }

    @Test
    public void heterogeneousVocabDraftingNeedsADraftModelAndGreedyDrafting() {
        IllegalArgumentException noModel = refused(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB, "true");
        assertTrue(noModel.getMessage(), noModel.getMessage().startsWith(
                SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB));
        assertTrue(noModel.getMessage(), noModel.getMessage().contains(EngineConfig.PROP_DRAFT));

        IllegalArgumentException probabilistic = refused(EngineConfig.PROP_DRAFT, "draft.gguf",
                SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB, "true",
                SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD, "probabilistic");
        assertTrue(probabilistic.getMessage(), probabilistic.getMessage().contains(
                SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD));
    }
}
