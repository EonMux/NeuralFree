package com.eonmux.neuralfree.gguf.tokenizer.hf;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link ChkhshTable}: the converter's fingerprint formula over golden id lists, the bundled table and text, and the
 * table format checks.
 *
 * <p>Each id list is the Hugging Face encoding of {@code chktxt} with special tokens added, as
 * {@code get_vocab_base_pre} computes it: the last line of llama.cpp's
 * {@code models/ggml-vocab-<name>.gguf.out} (written by convert_hf_to_gguf_update.py with
 * {@code add_special_tokens=False}) plus the ids the reference tokenizer's post-processor adds (Llama-3 and
 * DeepSeek-LLM put BOS first, BGE wraps the text in {@code [CLS]}/{@code [SEP]}).</p>
 */
public class ChkhshGoldenTest {

    /** {@code chktxt} as LC/conversion/base.py:1809 writes it. */
    static final String CHKTXT =
            "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍\uD83C"
            + "\uDF2B️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 "
            + "333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិស"
            + "េសអាច😁 ?我想在apple工作1314151天"
            + "～ ------======= нещо на Българ"
            + "ски ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you s"
            + "ure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL";

    /** meta-llama/Meta-Llama-3-8B (llama-bpe): BOS 128000 + 167 ids. */
    static final int[] LLAMA_BPE_IDS = {
            128000, 198, 4815, 15073, 66597, 8004, 1602, 2355, 79772, 11187, 9468, 248, 222, 320, 8416, 8,
            27623, 114, 102470, 9468, 234, 104, 31643, 320, 36773, 100166, 98634, 8, 26602, 227, 11410, 99,
            247, 9468, 99, 247, 220, 18, 220, 1644, 220, 8765, 220, 8765, 18, 220, 8765, 1644,
            220, 8765, 8765, 220, 8765, 8765, 18, 220, 8765, 8765, 1644, 220, 18, 13, 18, 220,
            18, 497, 18, 220, 18, 1131, 18, 220, 21549, 222, 98629, 241, 45358, 233, 21549, 237,
            45358, 224, 21549, 244, 21549, 115, 21549, 253, 45358, 223, 21549, 253, 21549, 95, 98629, 227,
            76460, 223, 949, 37046, 101067, 19000, 23182, 102301, 9263, 18136, 16, 36827, 21909, 56560, 54337, 19175,
            102118, 13373, 64571, 34694, 3114, 112203, 80112, 3436, 106451, 14196, 14196, 74694, 3089, 3089, 29249, 17523,
            3001, 27708, 7801, 358, 3077, 1027, 364, 83, 820, 568, 596, 1070, 11, 364, 793, 499,
            2771, 30, 364, 44, 539, 2771, 358, 3358, 1304, 433, 11, 364, 35, 499, 1093, 1063,
            15600, 30, 1226, 6, 43712, 264, 64966, 43
    };

    /** Qwen/Qwen1.5-7B (qwen2): 181 ids, no special tokens added. */
    static final int[] QWEN2_IDS = {
            198, 4710, 14731, 65497, 7847, 1572, 2303, 78672, 10947, 145836, 320, 8252, 8, 26525, 114, 378,
            235, 149921, 30543, 320, 35673, 99066, 97534, 8, 25521, 227, 11162, 99, 247, 149955, 220, 18,
            220, 18, 18, 220, 18, 18, 18, 220, 18, 18, 18, 18, 220, 18, 18, 18,
            18, 18, 220, 18, 18, 18, 18, 18, 18, 220, 18, 18, 18, 18, 18, 18,
            18, 220, 18, 18, 18, 18, 18, 18, 18, 18, 220, 18, 13, 18, 220, 18,
            496, 18, 220, 18, 1112, 18, 220, 146394, 97529, 241, 44258, 233, 146568, 44258, 224, 147603,
            20879, 115, 146280, 44258, 223, 146280, 147272, 97529, 227, 144534, 937, 104100, 18493, 22377, 99257, 16,
            18, 16, 19, 16, 20, 16, 35727, 21216, 55460, 53237, 18658, 14144, 1456, 13073, 63471, 33594,
            3038, 133178, 79012, 3355, 4605, 4605, 13874, 13874, 73594, 3014, 3014, 28149, 17085, 2928, 26610, 7646,
            358, 3003, 1012, 364, 83, 813, 566, 594, 1052, 11, 364, 787, 498, 2704, 30, 364,
            44, 537, 2704, 358, 3278, 1281, 432, 11, 364, 35, 498, 1075, 1045, 15243, 30, 1205,
            6, 42612, 264, 63866, 43
    };

    /** openai-community/gpt2 (gpt-2): 222 ids, no special tokens added. */
    static final int[] GPT2_IDS = {
            198, 220, 628, 220, 628, 198, 220, 197, 220, 197, 197, 220, 197, 198, 220, 220,
            198, 220, 220, 220, 198, 220, 220, 220, 220, 198, 220, 220, 220, 220, 220, 198,
            8582, 248, 222, 357, 11265, 8, 30325, 114, 447, 235, 8582, 234, 104, 37929, 357, 48101,
            795, 13210, 271, 1673, 36686, 515, 8, 14519, 227, 12520, 99, 247, 8582, 99, 247, 513,
            4747, 23460, 513, 20370, 23460, 2091, 23460, 20370, 23460, 24840, 23460, 2091, 20370, 513, 13, 18,
            513, 492, 18, 513, 986, 18, 28053, 252, 222, 157, 252, 114, 157, 252, 241, 157,
            253, 233, 157, 252, 237, 157, 253, 224, 157, 252, 244, 157, 252, 115, 157, 252,
            253, 157, 253, 223, 157, 252, 253, 157, 252, 95, 157, 252, 114, 157, 252, 227,
            47249, 223, 5633, 22755, 239, 46349, 111, 28839, 101, 18040, 32432, 98, 43291, 1485, 1415, 24309,
            25465, 171, 121, 252, 40103, 1421, 18604, 12466, 121, 16843, 141, 231, 15166, 12466, 121, 16142,
            12466, 239, 141, 232, 30143, 140, 111, 16142, 21169, 21727, 31583, 18849, 705, 39115, 6, 33153,
            15506, 63, 15931, 15931, 16317, 13896, 3228, 9805, 3548, 314, 1053, 587, 705, 44040, 339, 338,
            612, 11, 705, 2200, 345, 1654, 30, 705, 44, 407, 1654, 314, 1183, 787, 340, 11,
            705, 35, 345, 588, 617, 8887, 30, 775, 6, 26979, 257, 6, 75, 43
    };

    /** deepseek-ai/deepseek-llm-7b-base (deepseek-llm): BOS 100000 + 212 ids. */
    static final int[] DEEPSEEK_LLM_IDS = {
            100000, 185, 207, 185, 185, 207, 185, 185, 185, 207, 11969, 486, 22504, 185, 243, 185,
            300, 185, 251, 185, 663, 185, 10044, 95300, 334, 8754, 8, 33701, 114, 350, 222, 10044,
            221, 104, 46713, 334, 34732, 996, 24250, 262, 80923, 8, 207, 37103, 214, 12356, 99, 234,
            10044, 99, 234, 207, 18, 207, 18, 18, 207, 18, 18, 18, 207, 18, 18, 18,
            18, 207, 18, 18, 18, 18, 18, 207, 18, 18, 18, 18, 18, 18, 207, 18,
            18, 18, 18, 18, 18, 18, 207, 18, 18, 18, 18, 18, 18, 18, 18, 207,
            18, 13, 18, 207, 18, 526, 18, 207, 18, 1204, 18, 207, 71374, 209, 71374, 114,
            71374, 228, 155, 240, 220, 71374, 224, 155, 240, 211, 71374, 231, 71374, 115, 71374, 240,
            155, 240, 210, 71374, 240, 71374, 95, 71374, 114, 71374, 214, 71899, 210, 3025, 19017, 612,
            9407, 2681, 16, 18, 16, 19, 16, 20, 16, 1398, 68940, 239, 78827, 55170, 76659, 620,
            91754, 31116, 36804, 4885, 4885, 10897, 4390, 4390, 41047, 15278, 3033, 14986, 5675, 304, 6, 313,
            803, 655, 33326, 362, 6, 82, 745, 11, 655, 1374, 340, 2049, 30, 655, 44, 441,
            2049, 304, 6, 647, 1099, 359, 11, 655, 35, 340, 837, 742, 10842, 30, 1003, 6,
            10699, 245, 6, 75, 43
    };

    /** bigcode/starcoder2-3b (starcoder): 198 ids, no special tokens added. */
    static final int[] STARCODER_IDS = {
            353, 736, 8886, 221, 10883, 4238, 16101, 28540, 222, 3822, 272, 246, 327, 4434, 46, 18445,
            152, 46030, 45022, 142, 13878, 327, 12585, 19884, 33773, 40920, 751, 46, 41839, 5954, 137, 271,
            3822, 137, 271, 244, 56, 244, 56, 56, 244, 56, 56, 56, 244, 56, 56, 56,
            56, 244, 56, 56, 56, 56, 56, 244, 56, 56, 56, 56, 56, 56, 244, 56,
            56, 56, 56, 56, 56, 56, 244, 56, 56, 56, 56, 56, 56, 56, 56, 244,
            56, 51, 56, 244, 56, 516, 56, 244, 56, 1198, 56, 244, 14566, 246, 14566, 152,
            14566, 265, 30428, 257, 14566, 261, 30428, 248, 14566, 268, 14566, 153, 14566, 277, 30428, 247,
            14566, 277, 14566, 133, 14566, 152, 14566, 251, 36570, 247, 1037, 4995, 13379, 2924, 9515, 17823,
            54, 56, 54, 57, 54, 58, 54, 11904, 47892, 20895, 16625, 13047, 8389, 1059, 9504, 40216,
            13858, 2073, 8983, 12571, 1539, 10721, 5918, 9643, 13298, 932, 31723, 31330, 9221, 3226, 35426, 10400,
            457, 4783, 2602, 349, 121, 1477, 957, 1200, 2038, 49, 349, 632, 863, 3673, 68, 349,
            82, 666, 3673, 457, 4650, 1949, 580, 49, 349, 73, 863, 2144, 1649, 35941, 68, 2726,
            44, 7728, 331, 44, 113, 81
    };

    /** mosaicml/mpt-7b (mpt; allenai/OLMo-1.7-7B-hf shares the fingerprint): 179 ids. */
    static final int[] MPT_IDS = {
            586, 1744, 33525, 186, 209, 623, 28910, 187, 50276, 187, 50275, 187, 50274, 187, 50273, 187,
            14931, 237, 211, 313, 6320, 10, 49042, 116, 325, 224, 14931, 223, 106, 171, 118, 226,
            313, 34263, 802, 13511, 261, 32147, 456, 10, 3384, 239, 216, 22692, 101, 236, 14931, 101,
            236, 495, 5922, 30057, 495, 20084, 495, 26409, 30057, 20084, 495, 26409, 1610, 495, 26409, 20084,
            495, 15, 20, 495, 537, 20, 495, 1051, 20, 209, 18081, 211, 18081, 116, 18081, 230,
            39936, 222, 18081, 226, 39936, 213, 18081, 233, 18081, 117, 18081, 242, 39936, 212, 18081, 242,
            18081, 97, 18081, 116, 18081, 216, 14931, 235, 212, 3736, 15367, 41197, 13610, 19934, 41869, 21275,
            1012, 1047, 18795, 40120, 20422, 241, 16081, 6877, 12880, 11514, 1068, 8713, 38177, 13396, 3415, 9925,
            12559, 10453, 1389, 42011, 35033, 34842, 11202, 9739, 9739, 33021, 18963, 4672, 25561, 8220, 309, 1849,
            644, 686, 42618, 344, 434, 627, 13, 686, 1848, 368, 2119, 32, 686, 46, 417, 2119,
            309, 1833, 1056, 352, 13, 686, 37, 368, 751, 690, 10331, 32, 844, 8, 31516, 247,
            8, 77, 45
    };

    /**
     * BAAI/bge-small-en-v1.5 (bert-bge; jinaai/jina-embeddings-v2-base-en shares the fingerprint):
     * [CLS] 101 + 160 ids + [SEP] 102.
     */
    static final int[] BERT_BGE_IDS = {
            101, 100, 1006, 3671, 1007, 100, 1006, 3674, 7861, 29147, 2483, 9530, 16280, 23854, 1007, 100,
            100, 1017, 3943, 21211, 21211, 2509, 21211, 22394, 21211, 22394, 2509, 21211, 22394, 22394, 21211, 22394,
            22394, 2509, 1017, 1012, 1017, 1017, 1012, 1012, 1017, 1017, 1012, 1012, 1012, 1017, 100, 1029,
            1855, 100, 100, 6207, 100, 100, 14677, 23632, 22203, 1811, 1995, 1011, 1011, 1011, 1011, 1011,
            1011, 1027, 1027, 1027, 1027, 1027, 1027, 1027, 1192, 15290, 29754, 14150, 1192, 10260, 1181, 29755,
            29436, 29741, 10260, 16856, 29747, 23925, 10325, 1005, 1005, 1005, 1005, 1005, 1005, 1036, 1036, 1036,
            1036, 1036, 1036, 1036, 1000, 1000, 1000, 1000, 1012, 1012, 1012, 1012, 1012, 1012, 999, 999,
            999, 999, 999, 999, 1029, 1029, 1029, 1029, 1029, 1029, 1045, 1005, 2310, 2042, 1005, 2409,
            2002, 1005, 1055, 2045, 1010, 1005, 2128, 2017, 2469, 1029, 1005, 1049, 2025, 2469, 1045, 1005,
            2222, 2191, 2009, 1010, 1005, 1040, 2017, 2066, 2070, 5572, 1029, 2057, 1005, 2310, 1037, 1005,
            2222, 102
    };

    private static final String LLAMA_BPE_HASH = "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5";
    private static final String QWEN2_HASH = "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea";
    private static final String GPT2_HASH = "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454";
    private static final String DEEPSEEK_LLM_HASH = "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754";
    private static final String STARCODER_HASH = "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34";
    private static final String MPT_HASH = "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166";
    private static final String BERT_BGE_HASH = "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f";

    private static final String HEADER = "# chkhsh.tsv - Generated from llama.cpp a37cc4610 by the updateHfTables"
            + " Gradle task; do not edit. Rows: ";
    private static final String COLUMNS = "# sha256\tpre_name\trepo\n";

    private static void assertGolden(int[] ids, int length, String sha256, String name) {
        assertEquals(length, ids.length);
        assertEquals(sha256, ChkhshTable.hash(ids));
        assertTrue(ChkhshTable.builtIn().names(sha256).contains(name));
    }

    @Test
    public void llamaBpeIdsHashToItsRow() {
        assertGolden(LLAMA_BPE_IDS, 168, LLAMA_BPE_HASH, "llama-bpe");
        assertEquals("llama-bpe", ChkhshTable.builtIn().name(LLAMA_BPE_HASH));
    }

    @Test
    public void qwen2IdsHashToItsRow() {
        assertGolden(QWEN2_IDS, 181, QWEN2_HASH, "qwen2");
        assertEquals("qwen2", ChkhshTable.builtIn().name(QWEN2_HASH));
    }

    @Test
    public void gpt2IdsHashToItsRow() {
        assertGolden(GPT2_IDS, 222, GPT2_HASH, "gpt-2");
        assertEquals("gpt-2", ChkhshTable.builtIn().name(GPT2_HASH));
    }

    @Test
    public void deepseekLlmIdsHashToItsRow() {
        assertGolden(DEEPSEEK_LLM_IDS, 213, DEEPSEEK_LLM_HASH, "deepseek-llm");
        assertEquals("deepseek-llm", ChkhshTable.builtIn().name(DEEPSEEK_LLM_HASH));
    }

    @Test
    public void starcoderIdsHashToItsRow() {
        assertGolden(STARCODER_IDS, 198, STARCODER_HASH, "starcoder");
        assertEquals("starcoder", ChkhshTable.builtIn().name(STARCODER_HASH));
    }

    @Test
    public void sharedFingerprintResolvesToTheLastRowAsTheConverterDoes() {
        assertGolden(MPT_IDS, 179, MPT_HASH, "mpt");
        assertEquals(Arrays.asList("mpt", "olmo"), ChkhshTable.builtIn().names(MPT_HASH));
        assertEquals("olmo", ChkhshTable.builtIn().name(MPT_HASH));

        assertGolden(BERT_BGE_IDS, 162, BERT_BGE_HASH, "bert-bge");
        assertEquals(Arrays.asList("bert-bge", "jina-v2-en"), ChkhshTable.builtIn().names(BERT_BGE_HASH));
        assertEquals("jina-v2-en", ChkhshTable.builtIn().name(BERT_BGE_HASH));
    }

    @Test
    public void idsAreFormattedAsAPythonList() {
        assertEquals("[]", ChkhshTable.pythonListText(new int[0]));
        assertEquals("[7]", ChkhshTable.pythonListText(new int[]{7}));
        assertEquals("[1, 22, 333]", ChkhshTable.pythonListText(new int[]{1, 22, 333}));
        assertEquals("[-1, 0]", ChkhshTable.pythonListText(new int[]{-1, 0}));
        assertEquals("4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                ChkhshTable.hash(new int[0]));
        assertEquals("276d8737d9d5a80805e0731e8d5ae18972a8e3ba721f4988f67151c6e9159068",
                ChkhshTable.hash(new int[]{1, 22, 333}));
    }

    @Test
    public void bundledTableHoldsTheConverterRowsAndText() {
        ChkhshTable t = ChkhshTable.builtIn();
        assertEquals(103, t.size());
        assertEquals(CHKTXT, t.checkText());
        assertEquals("chatglm-bpe", t.rows().get(0).preName());
        assertEquals("THUDM/glm-4-9b-chat", t.rows().get(0).repo());
        assertNull(t.name("0000000000000000000000000000000000000000000000000000000000000000"));
        assertEquals(Collections.emptyList(), t.names("not a hash"));
    }

    @Test
    public void parseReadsTheGeneratedFormat() {
        String sha = LLAMA_BPE_HASH;
        ChkhshTable t = ChkhshTable.parse("t.tsv", HEADER + "2\n" + COLUMNS + sha + "\tllama-bpe\tmeta/a\n"
                + sha + "\tfalcon3\ttiiuae/b\n", "x");
        assertEquals(2, t.size());
        assertEquals("falcon3", t.name(sha));
        assertEquals(Arrays.asList("llama-bpe", "falcon3"), t.names(sha));
        assertEquals("x", t.checkText());
    }

    @Test
    public void parseRejectsMalformedTables() {
        String row = LLAMA_BPE_HASH + "\tllama-bpe\tmeta/a\n";
        assertRejected("t.tsv:1:", "# chkhsh.tsv rows\n" + COLUMNS + row);
        assertRejected("t.tsv:2:", HEADER + "1\n# sha\tname\n" + row);
        assertRejected("header records 2 rows", HEADER + "2\n" + COLUMNS + row);
        assertRejected("t.tsv:3: expected 3", HEADER + "1\n" + COLUMNS + LLAMA_BPE_HASH + "\tllama-bpe\n");
        assertRejected("t.tsv:3: 'ABC'", HEADER + "1\n" + COLUMNS + "ABC\tllama-bpe\tmeta/a\n");
        assertRejected("t.tsv:3: pre-tokenizer name", HEADER + "1\n" + COLUMNS + LLAMA_BPE_HASH + "\tllama bpe\tm\n");
        assertRejected("t.tsv:4: expected 3", HEADER + "2\n" + COLUMNS + row + "\n");
        assertRejected("fingerprint text is empty", HEADER + "1\n" + COLUMNS + row, "");
    }

    private static void assertRejected(String expected, String table) {
        assertRejected(expected, table, "x");
    }

    private static void assertRejected(String expected, String table, String text) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ChkhshTable.parse("t.tsv", table, text));
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }
}
