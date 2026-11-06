package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * {@link HfEngramTokenMap} against the converter's normalizer chain (deepseek41.py:55-78): tokens whose decoded text
 * normalizes alike collapse to one compressed id, whitespace collapses and trims, a lone space survives through the
 * private-use sentinel, and the token count and compressed vocabulary the configuration declares are checked.
 */
public class HfEngramTokenMapTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void tokensThatNormalizeAlikeShareACompressedId() throws IOException {
        HfEngramTokenMap.Result result = map();
        assertArrayEquals(new int[]{0, 1, 2, 3, 4, 5, 6, 2, 2, 7, 7}, result.lookup());
        assertEquals(11, result.tokens());
        assertEquals(8, result.compressedVocab());
    }

    private HfEngramTokenMap.Result map() throws IOException {
        Path dir = tmp.newFolder("tok").toPath();
        java.nio.file.Files.writeString(dir.resolve("config.json"),
                SyntheticHf.json("{'architectures': ['DeepseekV41ForCausalLM'], 'vocab_size': 11}"));
        java.nio.file.Files.writeString(dir.resolve("tokenizer.json"), tokenizer());
        HfConfig cfg = HfConfigLoader.load(dir, com.eonmux.neuralfree.gguf.ModelSourceOptions.builder().build());
        HfTokenizerFiles files = HfTokenizerFiles.locate(dir);
        return HfEngramTokenMap.build(files, cfg.requireInt("vocab_size"));
    }

    private static String tokenizer() {
        return SyntheticHf.json("{'added_tokens':["
                + "{'id':0,'content':'<s>','special':true,'normalized':false},"
                + "{'id':1,'content':'</s>','special':true,'normalized':false},"
                + "{'id':5,'content':'<|begin|>','special':true,'normalized':false},"
                + "{'id':6,'content':'Ġhi','special':false,'normalized':false,'lstrip':true},"
                + "{'id':7,'content':'A','special':false,'normalized':false},"
                + "{'id':8,'content':' A ','special':false,'normalized':false},"
                + "{'id':9,'content':' ','special':false,'normalized':false},"
                + "{'id':10,'content':'  ','special':false,'normalized':false}],"
                + "'normalizer':null,"
                + "'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,'trim_offsets':true,'use_regex':true},"
                + "'post_processor':null,"
                + "'decoder':{'type':'ByteLevel','add_prefix_space':true,'trim_offsets':true,'use_regex':true},"
                + "'model':{'type':'BPE','unk_token':null,'byte_fallback':false,'ignore_merges':false,"
                + "'vocab':{'<s>':0,'</s>':1,'a':2,'b':3,'ab':4},'merges':['a b']}}");
    }
}
