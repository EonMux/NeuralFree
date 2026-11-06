package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The {@code collect-env} report over a constructed snapshot: the version, the Java and OS facts,
 * the device listing, the hub cache line, the server jar line and the environment section are all
 * present, and the secret variables report only that they are set — never their value.
 */
public class CollectEnvTest {

    private CollectEnv.Snapshot snapshot(Map<String, String> env) {
        Map<String, String> java = new LinkedHashMap<>();
        java.put("version", "21.0.5");
        java.put("vendor", "TestVendor");
        java.put("home", "/usr/lib/jvm/test");
        java.put("vm", "Test VM");
        Map<String, String> os = new LinkedHashMap<>();
        os.put("name", "Linux");
        os.put("version", "6.11");
        os.put("arch", "amd64");
        return new CollectEnv.Snapshot("9.9.9-test", java, os,
                Arrays.asList(new DeviceListing.Entry("OpenCL", 0, "Test GPU", 1024L * 1024L)),
                "/tmp/hub", 2, "/opt/neuralfree/server/server.jar", env);
    }

    private static Map<String, String> env(String... pairs) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < pairs.length; i += 2) {
            m.put(pairs[i], pairs[i + 1]);
        }
        return m;
    }

    @Test
    public void theReportNamesEverySection() {
        String report = CollectEnv.report(snapshot(env()));
        assertTrue(report, report.contains("NeuralFree version: 9.9.9-test\n"));
        assertTrue(report, report.contains("Java version: 21.0.5\n"));
        assertTrue(report, report.contains("Java vendor: TestVendor\n"));
        assertTrue(report, report.contains("OS name: Linux\n"));
        assertTrue(report, report.contains("OS arch: amd64\n"));
        assertTrue(report, report.contains("Available devices:\n"));
        assertTrue(report, report.contains("  OpenCL 0: Test GPU (1 MiB)\n"));
        assertTrue(report, report.contains("Hub cache: /tmp/hub (2 models)\n"));
        assertTrue(report, report.contains("Server jar: /opt/neuralfree/server/server.jar\n"));
        assertTrue(report, report.contains("Environment:\n"));
    }

    @Test
    public void aMissingServerJarSaysSoInsteadOfPrintingNull() {
        CollectEnv.Snapshot none = new CollectEnv.Snapshot("1.0", new LinkedHashMap<>(),
                new LinkedHashMap<>(), Collections.emptyList(), "/tmp/hub", 0, null, env());
        assertTrue(CollectEnv.report(none), CollectEnv.report(none).contains("Server jar: <not found>"));
    }

    @Test
    public void anEmptyDeviceListPrintsTheNoneLine() {
        CollectEnv.Snapshot none = new CollectEnv.Snapshot("1.0", new LinkedHashMap<>(),
                new LinkedHashMap<>(), Collections.emptyList(), "/tmp/hub", 0, null, env());
        assertTrue(CollectEnv.report(none), CollectEnv.report(none).contains("  (none)\n"));
    }

    @Test
    public void aSecretVariableReportsOnlyThatItIsSet() {
        assertEquals("<set>", CollectEnv.envValue("HF_TOKEN", "hf_secret_value"));
        assertEquals("<set>", CollectEnv.envValue("NEURALFREE_API_KEY", "sk-secret"));
        assertEquals("<set>", CollectEnv.envValue("NEURALFREE_WATERMARK_KEY", "3141592653589793238"));
        assertEquals("<set>", CollectEnv.envValue("NEURALFREE_RPC_KEY", "rpc-secret"));
        assertEquals("<set>", CollectEnv.envValue(KvTransferConfig.ENV_TOKEN, "kv-transfer-secret"));
    }

    @Test
    public void aPlainVariablePrintsItsValue() {
        assertEquals("/models", CollectEnv.envValue("NEURALFREE_MODEL", "/models"));
        assertEquals("<not set>", CollectEnv.envValue("NEURALFREE_MODEL", null));
    }

    @Test
    public void noSecretValueReachesTheReport() {
        String report = CollectEnv.report(snapshot(env(
                "HF_TOKEN", "hf_secret_value", "NEURALFREE_API_KEY", "sk-secret",
                "NEURALFREE_WATERMARK_KEY", "3141592653589793238",
                "NEURALFREE_WATERMARK_KEY_FILE", "/etc/neuralfree/watermark.key",
                "NEURALFREE_RPC_KEY", "rpc-secret",
                KvTransferConfig.ENV_TOKEN, "kv-transfer-secret",
                KvTransferConfig.ENV_TOKEN_FILE, "/etc/neuralfree/kv-transfer.token",
                "NEURALFREE_MODEL", "/models", "HF_HUB_OFFLINE", "1")));
        assertFalse(report, report.contains("hf_secret_value"));
        assertFalse(report, report.contains("sk-secret"));
        assertFalse(report, report.contains("3141592653589793238"));
        assertFalse(report, report.contains("rpc-secret"));
        assertFalse(report, report.contains("kv-transfer-secret"));
        assertTrue(report, report.contains("  HF_TOKEN = <set>\n"));
        assertTrue(report, report.contains("  NEURALFREE_API_KEY = <set>\n"));
        assertTrue(report, report.contains("  NEURALFREE_WATERMARK_KEY = <set>\n"));
        assertTrue(report, report.contains("  NEURALFREE_RPC_KEY = <set>\n"));
        assertTrue(report, report.contains("  " + KvTransferConfig.ENV_TOKEN + " = <set>\n"));
        assertTrue(report, report.contains(
                "  " + KvTransferConfig.ENV_TOKEN_FILE + " = /etc/neuralfree/kv-transfer.token\n"));
        assertTrue(report, report.contains(
                "  NEURALFREE_WATERMARK_KEY_FILE = /etc/neuralfree/watermark.key\n"));
        assertTrue(report, report.contains("  NEURALFREE_MODEL = /models\n"));
        assertTrue(report, report.contains("  HF_HUB_OFFLINE = 1\n"));
    }
    @Test
    public void theVariableTableIsTheEnginesOwn() {
        assertTrue(CollectEnv.ENV_VARS.contains("NEURALFREE_MODEL"));
        assertTrue(CollectEnv.ENV_VARS.contains("HF_TOKEN"));
        assertTrue(CollectEnv.ENV_VARS.contains("NEURALFREE_WATERMARK_KEY"));
        assertTrue(CollectEnv.ENV_VARS.contains("NEURALFREE_WATERMARK_KEY_FILE"));
        assertTrue(CollectEnv.ENV_VARS.contains("NEURALFREE_RPC_KEY"));
        assertTrue(CollectEnv.ENV_VARS.contains(KvTransferConfig.ENV_TOKEN));
        assertTrue(CollectEnv.ENV_VARS.contains(KvTransferConfig.ENV_TOKEN_FILE));
        assertTrue(CollectEnv.ENV_VARS.contains("LLAMA_CACHE"));
        assertTrue(CollectEnv.ENV_VARS.contains("HF_HUB_CACHE"));
    }

    @Test
    public void theSnapshotExposesItsParts() {
        CollectEnv.Snapshot s = snapshot(env("NEURALFREE_MODEL", "/models"));
        assertEquals("9.9.9-test", s.version());
        assertEquals("21.0.5", s.java().get("version"));
        assertEquals("Linux", s.os().get("name"));
        assertEquals(1, s.devices().size());
        assertEquals("/tmp/hub", s.cacheRoot());
        assertEquals(2, s.cachedModels());
        assertEquals("/opt/neuralfree/server/server.jar", s.serverJar());
        assertEquals("/models", s.env().get("NEURALFREE_MODEL"));
        assertNull(s.java().get("missing"));
    }
}
