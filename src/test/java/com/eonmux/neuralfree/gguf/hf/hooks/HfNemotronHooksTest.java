package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * {@code NemotronModel} (LC/conversion/nemotron.py:366-423) as {@link HfDenseClassicHooksD} converts it: the
 * partial-rotary RoPE width over {@code head_dim} for a non-standard grouped-query layout, the linear scaling factor,
 * and the SentencePiece-style BPE vocabulary of a checkpoint without {@code tokenizer.model}.
 */
public class HfNemotronHooksTest {

    private static final String BASE = "'architectures': ['NemotronForCausalLM'], 'hidden_size': 64,"
            + " 'intermediate_size': 128, 'num_attention_heads': 2, 'num_key_value_heads': 1, 'num_hidden_layers': 2,"
            + " 'max_position_embeddings': 64, 'norm_eps': 1e-05, 'vocab_size': 8, 'rope_theta': 10000.0,"
            + " 'partial_rotary_factor': 0.5";

    /** A Metaspace BPE tokenizer.json with byte fallback and the {@code <0x0A>} byte token. */
    private static final String SPM_STYLE_BPE = "{'added_tokens':[{'id':0,'content':'<unk>','special':true},"
            + "{'id':1,'content':'<s>','special':true},{'id':2,'content':'</s>','special':true}],"
            + "'normalizer':null,"
            + "'pre_tokenizer':{'type':'Metaspace','replacement':'▁','prepend_scheme':'first','split':false},"
            + "'post_processor':null,"
            + "'decoder':{'type':'Sequence','decoders':[{'type':'Replace','pattern':{'String':'▁'},'content':' '},"
            + "{'type':'ByteFallback'},{'type':'Fuse'},{'type':'Strip','content':' ','start':1,'stop':0}]},"
            + "'model':{'type':'BPE','unk_token':'<unk>','byte_fallback':true,'ignore_merges':false,"
            + "'vocab':{'<unk>':0,'<s>':1,'</s>':2,'<0x0A>':3,'<0x0a>':4,'▁':5,'a':6,'▁a':7},"
            + "'merges':['▁ a']}}";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theRopeWidthFollowsTheHeadDimOfANonStandardLayout() {
        HfMetadata gqa = meta("{" + BASE + ", 'head_dim': 48}");
        assertEquals(24L, num(gqa, GgufKeys.ROPE_DIMENSION_COUNT));
        HfMetadata standard = meta("{" + BASE + "}");
        assertEquals("0.5 of hidden_size / num_attention_heads", 16L, num(standard, GgufKeys.ROPE_DIMENSION_COUNT));
        assertEquals("none", standard.get(key(GgufKeys.ROPE_SCALING_TYPE)));
        assertFalse(standard.has(key(GgufKeys.ROPE_SCALING_FACTOR)));
    }

    @Test
    public void aLinearFactorWritesItsScalingFactor() {
        HfMetadata m = meta("{" + BASE + ", 'factor': 2.0}");
        assertEquals("linear", m.get(key(GgufKeys.ROPE_SCALING_TYPE)));
        assertEquals(2.0, ((Number) m.get(key(GgufKeys.ROPE_SCALING_FACTOR))).doubleValue(), 0.0);
    }

    @Test
    public void withoutTokenizerModelTheBpeVocabularyTypesItsByteTokens() throws IOException {
        Path dir = tmp.newFolder().toPath();
        String config = "{" + BASE + "}";
        write(dir, "config.json", config);
        write(dir, "tokenizer.json", SPM_STYLE_BPE);
        write(dir, "tokenizer_config.json", "{'bos_token':'<s>','eos_token':'</s>','add_bos_token':true}");
        HfConfig cfg = config(dir, config);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry, Arch.NEMOTRON)
                .tokenizerFiles(HfTokenizerFiles.locate(dir))
                .build();
        Map<String, Object> md = HfDenseClassicHooksD.nemotronVocab(ctx, (tj, encode) -> "domyn-small");

        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals("domyn-small", md.get(GgufKeys.TOKENIZER_PRE));
        List<?> types = (List<?>) md.get(GgufKeys.TOKENIZER_TOKEN_TYPE);
        assertEquals("<0x0A> is a byte token", 6, ((Number) types.get(3)).intValue());
        assertEquals("<0x0a> is not upper-case hex", 1, ((Number) types.get(4)).intValue());
        assertEquals(1, ((Number) types.get(7)).intValue());
        assertEquals(List.of("▁ a"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals("first", md.get(GgufKeys.TOKENIZER_METASPACE_PREPEND_SCHEME));
        assertFalse("no SentencePiece sentinel ids", md.containsKey(GgufKeys.TOKENIZER_PAD_ID));

        TokenizerVocab vocab = TokenizerVocab.fromMetadata(md);
        assertEquals(0x0A, vocab.byteValue(3));
        assertEquals(true, vocab.isByte(3));
    }

    private HfMetadata meta(String config) {
        HfConfig cfg = config(Path.of("."), config);
        HfArchEntry entry = HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT);
        HfLoadContext ctx = HfLoadContext.builder(cfg, entry, Arch.NEMOTRON).build();
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.NEMOTRON.ggufName());
        HfDenseClassicHooksD.nemotronMeta(ctx, kv, null);
        return kv.build();
    }

    private static HfConfig config(Path dir, String singleQuotedJson) {
        return HfConfigLoader.fromJson(dir, singleQuotedJson.replace('\'', '"'));
    }

    private static void write(Path dir, String name, String singleQuotedJson) throws IOException {
        Files.writeString(dir.resolve(name), singleQuotedJson.replace('\'', '"'), StandardCharsets.UTF_8);
    }

    private static String key(String template) {
        return String.format(template, Arch.NEMOTRON.ggufName());
    }

    private static long num(HfMetadata m, String template) {
        return ((Number) m.get(key(template))).longValue();
    }
}
