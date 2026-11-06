package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link HfBertHooks#xlmRobertaArch}: the converter's {@code lora_adaptations} truthiness switch (bert.py:478-481). */
public class HfBertHooksTest {

    private static Arch arch(String loraAdaptations) {
        String member = loraAdaptations == null ? "" : ", 'lora_adaptations': " + loraAdaptations;
        return HfBertHooks.xlmRobertaArch(config("{'architectures': ['XLMRobertaModel']" + member + "}"), null);
    }

    @Test
    public void taskAdapterNamesSelectJinaBertV3() {
        assertEquals(Arch.JINA_BERT_V3, arch("['retrieval.query', 'retrieval.passage', 'text-matching']"));
    }

    @Test
    public void absentOrFalsyAdaptationsKeepBert() {
        assertEquals(Arch.BERT, arch(null));
        assertEquals(Arch.BERT, arch("null"));
        assertEquals(Arch.BERT, arch("[]"));
        assertEquals(Arch.BERT, arch("false"));
        assertEquals(Arch.BERT, arch("0"));
        assertEquals(Arch.BERT, arch("''"));
        assertEquals(Arch.BERT, arch("{}"));
    }

    @Test
    public void aTruthyValueThatIsNotAListIsRefusedByName() {
        HfConfig cfg = config("{'architectures': ['XLMRobertaModel'], 'lora_adaptations': {'tasks': ['a']}}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfBertHooks.xlmRobertaArch(cfg, null));
        assertTrue(e.getMessage(), e.getMessage().contains("lora_adaptations"));
    }
}
