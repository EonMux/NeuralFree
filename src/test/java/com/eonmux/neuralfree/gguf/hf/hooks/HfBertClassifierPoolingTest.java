package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's classifier checkpoint metadata over the BERT family: a token classifier scores every token without the
 * {@code classifier.token_direct} flag, a sequence classifier pools first — with {@code classifier.token_direct}
 * only for ModernBert (vLLM's {@code for_seq_cls}) — and the pooler activation resolves {@code problem_type} or the
 * sentence-transformers activation function (bert.py:341-380, modernbert.py:381-384).
 */
public class HfBertClassifierPoolingTest {

    private static HfMetadataBuilder meta(String architecture, String extra) {
        HfMetadataBuilder kv = new HfMetadataBuilder("bert");
        HfBertHooks.applyClassifierPooling(config("{'architectures': ['" + architecture + "']" + extra + "}"), kv);
        return kv;
    }

    @Test
    public void aTokenClassifierScoresEveryTokenWithoutTheDirectFlag() {
        HfMetadataBuilder kv = meta("BertForTokenClassification", ", 'num_labels': 3");
        assertEquals(PoolingKind.NONE.ggufCode(), kv.get(GgufKeys.POOLING_TYPE));
        assertEquals("token", kv.get(GgufKeys.CLASSIFIER_SCOPE));
        assertFalse(kv.has(GgufKeys.CLASSIFIER_TOKEN_DIRECT));
    }

    @Test
    public void modernBertSequenceClassificationRunsTheFinalProjectionPerToken() {
        HfMetadataBuilder kv = meta("ModernBertForSequenceClassification", ", 'num_labels': 2");
        assertEquals(PoolingKind.CLS.ggufCode(), kv.get(GgufKeys.POOLING_TYPE));
        assertEquals("sequence", kv.get(GgufKeys.CLASSIFIER_SCOPE));
        assertEquals(Boolean.TRUE, kv.get(GgufKeys.CLASSIFIER_TOKEN_DIRECT));
        assertEquals("gelu", kv.get(GgufKeys.CLASSIFIER_ACTIVATION));
        assertEquals("softmax", kv.get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
    }

    @Test
    public void theOtherSequenceClassifiersKeepTheWholeChain() {
        HfMetadataBuilder kv = meta("BertForSequenceClassification", ", 'num_labels': 2");
        assertNull("the full chain reads the hidden projection", kv.get(GgufKeys.CLASSIFIER_TOKEN_DIRECT));
        assertNull(kv.get(GgufKeys.CLASSIFIER_ACTIVATION));
        assertEquals("softmax", kv.get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
    }

    @Test
    public void theConfiguredMeanPoolingReplacesTheClsDefault() {
        assertEquals(PoolingKind.MEAN.ggufCode(),
                meta("BertForSequenceClassification",
                        ", 'num_labels': 2, 'classifier_pooling': 'mean'").get(GgufKeys.POOLING_TYPE));
    }

    @Test
    public void theProblemTypeSetsThePoolerActivation() {
        assertEquals("identity", meta("BertForSequenceClassification",
                ", 'num_labels': 2, 'problem_type': 'regression'").get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
        assertEquals("sigmoid", meta("BertForSequenceClassification",
                ", 'num_labels': 2, 'problem_type': 'multi_label_classification'")
                .get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
        // single_label softmaxes, except its one-output regression shape
        assertEquals("softmax", meta("BertForSequenceClassification",
                ", 'num_labels': 3, 'problem_type': 'single_label_classification'")
                .get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
        assertEquals("sigmoid", meta("BertForSequenceClassification",
                ", 'num_labels': 1").get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
    }

    @Test
    public void theProblemTypeWinsOverTheSentenceTransformersActivation() {
        // vLLM reads problem_type first and only falls back to the sentence-transformers function
        assertEquals("softmax", meta("BertForSequenceClassification",
                ", 'num_labels': 2, 'problem_type': 'single_label_classification',"
                        + " 'sentence_transformers': {'activation_fn': 'torch.nn.modules.activation.Identity'}")
                .get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
    }

    @Test
    public void theSentenceTransformersActivationAppliesWithoutAProblemType() {
        assertEquals("identity", meta("BertForSequenceClassification",
                ", 'num_labels': 2, 'sentence_transformers':"
                        + " {'activation_fn': 'torch.nn.modules.activation.Identity'}")
                .get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
        assertEquals("tanh", meta("BertForSequenceClassification",
                ", 'num_labels': 1, 'sbert_ce_default_activation_function':"
                        + " 'torch.nn.modules.activation.Tanh'")
                .get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
    }

    @Test
    public void anActivationOutsideTheTorchModulesRefusesByName() {
        GgufFormatException outside = assertThrows(GgufFormatException.class,
                () -> meta("BertForSequenceClassification",
                        ", 'num_labels': 2, 'sbert_ce_default_activation_function': 'torch.nn.ReLU'"));
        assertTrue(outside.getMessage(), outside.getMessage().contains("torch.nn.modules"));

        GgufFormatException unsupported = assertThrows(GgufFormatException.class,
                () -> meta("BertForSequenceClassification",
                        ", 'num_labels': 2, 'sentence_transformers':"
                                + " {'activation_fn': 'torch.nn.modules.activation.ReLU'}"));
        assertTrue(unsupported.getMessage(), unsupported.getMessage().contains("ReLU"));
    }

    @Test
    public void otherArchitecturesKeepTheirPooling() {
        HfMetadataBuilder kv = meta("BertModel", "");
        assertFalse(kv.has(GgufKeys.CLASSIFIER_SCOPE));
        assertFalse(kv.has(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));
    }
}
