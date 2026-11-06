package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.batch.PrefixHashAlgo;
import com.eonmux.neuralfree.gguf.engine.batch.SchedulingPolicy;
import com.eonmux.neuralfree.gguf.engine.kvevents.KvEventsConfig;
import com.eonmux.neuralfree.gguf.engine.kvevents.ZmqEventPublisherConfig;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvConnectors;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvRole;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferConfig;
import com.eonmux.neuralfree.gguf.image.ImageLoadOptions;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.EncoderCacheConfig;
import com.eonmux.neuralfree.gguf.mm.MmEncoderTpMode;
import com.eonmux.neuralfree.gguf.mm.MmprojOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Launch flags map to the properties a {@code -D} launch would set: llama.cpp's and vLLM's spellings, repeatable
 * keys and adapters, vLLM's {@code --kv-events-config} JSON onto the KV cache event publisher, vLLM's
 * {@code --scheduling-policy} onto the batch engine's scheduling policy, the media flags,
 * the pooling override of {@code --pooling} and vLLM's pooler config JSON, and refusals naming
 * the flag (unknown flags, missing values, ports outside {@code [1, 65535]}, missing files, unusable ZMQ
 * transport fields).
 */
class ServerLaunchArgsTest {

    @TempDir
    Path dir;

    private static String refusalOf(Runnable call) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call::run);
        return e.getMessage();
    }

    @Test
    void rpcAddsRemoteDevicesToTheLayerSplit() {
        Map<String, String> p = ServerLaunchArgs.parse("--device", "0", "--rpc", "box:50052@k").properties();
        assertEquals("layer", p.get(EngineConfig.PROP_SPLIT_MODE));
        assertEquals("box:50052@k", p.get(EngineConfig.PROP_RPC));
        assertEquals("0", p.get(EngineConfig.PROP_GPU_DEVICES));

        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--device", "none", "--rpc", "box:50052@k"))
                .contains("--rpc"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--rpc", "box:50052@k", "--device", "none"))
                .contains("--rpc"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--rpc", "box:50052@k,")).contains("empty"));
    }

    @Test
    void rpcRefusalsNeverPrintAnInlineKey() {
        String empty = refusalOf(() -> ServerLaunchArgs.parse("--rpc", "box:50052@s3cret,"));
        assertFalse(empty.contains("s3cret"), empty);
        assertTrue(empty.contains("box:50052@..."), empty);

        String malformed = refusalOf(() -> ServerLaunchArgs.parse("--rpc", "box:abc@s3cret"));
        assertFalse(malformed.contains("s3cret"), malformed);
        assertTrue(malformed.contains("box:abc@..."), malformed);
    }

    @Test
    void aPresetRpcPropertyStillPairsWithTheDeviceFlag() {
        String saved = System.getProperty(EngineConfig.PROP_RPC);
        try {
            // the serve relaunch puts the servers in a preset property, not in --rpc
            System.setProperty(EngineConfig.PROP_RPC, "box:50052@k");
            assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--device", "none")).contains("--rpc"));
            assertEquals("layer", ServerLaunchArgs.parse("--device", "0").properties()
                    .get(EngineConfig.PROP_SPLIT_MODE));
        } finally {
            restore(EngineConfig.PROP_RPC, saved);
        }
    }

    @Test
    void backendSamplingFlagsAreLaunchLocalAndNeedNoValue() {
        String sampling = "neuralfree.sampling.device";
        String verify = "neuralfree.draft.verify.device";
        String draft = "neuralfree.draft.dflash.device";
        String before = System.getProperty(sampling);
        for (String flag : new String[]{"-bs", "--backend-sampling"}) {
            Map<String, String> p = ServerLaunchArgs.parse(flag, "--port", "8081").properties();
            assertEquals("true", p.get(sampling));
            assertEquals("true", p.get(verify));
            assertEquals("true", p.get(draft));
            assertEquals("8081", p.get("quarkus.http.port"));
        }
        assertEquals(Map.of(sampling, "false", verify, "false", draft, "false"),
                ServerLaunchArgs.parse("--no-backend-sampling").properties());
        assertEquals(Map.of(draft, "false"), ServerLaunchArgs.parse("--no-spec-draft-backend-sampling").properties());
        assertEquals(Map.of(draft, "true"), ServerLaunchArgs.parse("--spec-draft-backend-sampling").properties());
        Map<String, String> p = ServerLaunchArgs.parse("--no-backend-sampling", "--spec-draft-backend-sampling")
                .properties();
        assertEquals("false", p.get(sampling));
        assertEquals("false", p.get(verify));
        assertEquals("true", p.get(draft));
        assertEquals("false", ServerLaunchArgs.parse("-bs", "--no-spec-draft-backend-sampling").properties().get(draft));
        assertEquals("true", ServerLaunchArgs.parse("--no-backend-sampling", "-bs").properties().get(sampling));
        assertEquals(before, System.getProperty(sampling));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--backend-sampling", "false")).contains("false"));
    }

    @Test
    void numaAndMemoryLockFlagsMapToTheirProperties() {
        String priorMode = System.getProperty(TensorReadMode.PROP_READ_MODE);
        String priorDisabled = System.getProperty(TensorReadMode.PROP_DISABLE_MMAP);
        try {
            System.clearProperty(TensorReadMode.PROP_READ_MODE);
            System.clearProperty(TensorReadMode.PROP_DISABLE_MMAP);

            Map<String, String> p = ServerLaunchArgs.parse("--numa", "isolate", "--mlock").properties();
            assertEquals("isolate", p.get(NumaStrategy.PROPERTY));
            assertEquals("mmap+mlock", p.get(TensorReadMode.PROP_READ_MODE));
            assertEquals("disabled", ServerLaunchArgs.parse("--numa", "off").properties()
                    .get(NumaStrategy.PROPERTY));

            assertEquals("mlock", ServerLaunchArgs.parse("--no-mmap", "--mlock").properties()
                    .get(TensorReadMode.PROP_READ_MODE));
            assertEquals("mlock", ServerLaunchArgs.parse("--mlock", "--no-mmap").properties()
                    .get(TensorReadMode.PROP_READ_MODE));
            assertEquals("read", ServerLaunchArgs.parse("--no-mmap").properties()
                    .get(TensorReadMode.PROP_READ_MODE));
            assertEquals("direct", ServerLaunchArgs.parse("-lm", "dio").properties()
                    .get(TensorReadMode.PROP_READ_MODE));
            assertEquals("mmap+mlock", ServerLaunchArgs.parse("--load-mode", "mmap+mlock").properties()
                    .get(TensorReadMode.PROP_READ_MODE));
            assertFalse(ServerLaunchArgs.parse("-lm", "auto").properties()
                    .containsKey(TensorReadMode.PROP_READ_MODE));

            assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--numa", "mirror")).contains("--numa"));
            assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-lm", "mmap+mlock+dio")).contains("-lm"));
            System.setProperty(TensorReadMode.PROP_READ_MODE, "lazy");
            assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mlock")).contains("lazily read tensors"));
        } finally {
            restore(TensorReadMode.PROP_READ_MODE, priorMode);
            restore(TensorReadMode.PROP_DISABLE_MMAP, priorDisabled);
        }
    }

    private static void restore(String key, String prior) {
        if (prior == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, prior);
        }
    }

    @Test
    void listDevicesActsAndStartsNothing() {
        ServerLaunchArgs parsed = ServerLaunchArgs.parse("--list-devices");
        assertTrue(parsed.listDevices());
        assertFalse(parsed.help());
        assertTrue(parsed.properties().isEmpty(), "the listing sets no property");
        assertTrue(ServerLaunchArgs.parse("-m", "m.gguf", "--list-devices").listDevices(),
                "the flag is taken wherever it appears");
        assertFalse(ServerLaunchArgs.parse("-m", "m.gguf").listDevices());
    }

    @Test
    void theDeviceListMapsToTheLoadProperties() {
        Map<String, String> name = ServerLaunchArgs.parse("-dev", "RTX").properties();
        assertEquals("RTX", name.get(EngineConfig.PROP_GPU_DEVICES));
        assertEquals("RTX", name.get(EngineConfig.PROP_DEVICE));
        assertFalse(name.containsKey(EngineConfig.PROP_SPLIT_MODE));

        Map<String, String> split = ServerLaunchArgs.parse("--device", "0,1").properties();
        assertEquals("0,1", split.get(EngineConfig.PROP_GPU_DEVICES));
        assertEquals("layer", split.get(EngineConfig.PROP_SPLIT_MODE));
        assertFalse(split.containsKey(EngineConfig.PROP_DEVICE), "an index needs no name filter");

        Map<String, String> one = ServerLaunchArgs.parse("-device", "0").properties();
        assertEquals("0", one.get(EngineConfig.PROP_GPU_DEVICES));
        assertFalse(one.containsKey(EngineConfig.PROP_DEVICE));

        Map<String, String> none = ServerLaunchArgs.parse("-dev", "none").properties();
        assertEquals("cpu", none.get(EngineConfig.PROP_BACKEND));
        assertFalse(none.containsKey(EngineConfig.PROP_GPU_DEVICES));

        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-dev", "none,0")).contains("-dev"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--device", "0,,1")).contains("--device"));

        Map<String, String> later = ServerLaunchArgs.parse("-dev", "none", "-dev", "0,1").properties();
        assertEquals("0,1", later.get(EngineConfig.PROP_GPU_DEVICES));
        assertEquals("layer", later.get(EngineConfig.PROP_SPLIT_MODE));
        assertFalse(later.containsKey(EngineConfig.PROP_BACKEND), "a later device list undoes the earlier none");
    }

    @Test
    void theBatchInvariantSwitchMapsToItsProperty() {
        assertEquals("true", ServerLaunchArgs.parse("--batch-invariant").properties()
                .get(BatchInvariance.PROPERTY));
        assertEquals("true", ServerLaunchArgs.parse("--host", "0.0.0.0", "--batch-invariant").properties()
                .get(BatchInvariance.PROPERTY));
        assertFalse(ServerLaunchArgs.parse("--host", "0.0.0.0").properties()
                .containsKey(BatchInvariance.PROPERTY), "the mode stays off unless the flag asks for it");
    }

    @Test
    void theKvOffloadFlagsMapToTheProperty() {
        assertEquals("false", ServerLaunchArgs.parse("-nkvo").properties().get(EngineConfig.PROP_KV_OFFLOAD));
        assertEquals("false",
                ServerLaunchArgs.parse("--no-kv-offload").properties().get(EngineConfig.PROP_KV_OFFLOAD));
        assertEquals("true", ServerLaunchArgs.parse("-kvo").properties().get(EngineConfig.PROP_KV_OFFLOAD));
        assertEquals("true", ServerLaunchArgs.parse("--kv-offload").properties().get(EngineConfig.PROP_KV_OFFLOAD));
        assertEquals("true", ServerLaunchArgs.parse("-nkvo", "-kvo").properties().get(EngineConfig.PROP_KV_OFFLOAD),
                "the later flag wins");
    }

    @Test
    void theRouterFlagsMapToTheModelsProperties() throws IOException {
        Path preset = Files.write(dir.resolve("models.ini"), new byte[0]);
        Map<String, String> properties = ServerLaunchArgs.parse(
                "--models-dir", dir.toString(), "--models-preset", preset.toString(),
                "--models-max", "8", "--models-memory-margin", "512",
                "--models-autoload").properties();
        assertEquals(dir.toString(), properties.get(ModelRouter.DIR_PROPERTY));
        assertEquals(preset.toString(), properties.get(ModelRouter.PRESET_PROPERTY));
        assertEquals("8", properties.get(ModelRouter.MAX_PROPERTY));
        assertEquals("512", properties.get(ModelRouter.MARGIN_PROPERTY));
        assertEquals("true", properties.get(ModelRouter.AUTOLOAD_PROPERTY));
        assertEquals("false", ServerLaunchArgs.parse("--no-models-autoload").properties()
                .get(ModelRouter.AUTOLOAD_PROPERTY));
        assertEquals("true", ServerLaunchArgs.parse("--no-models-autoload", "--models-autoload")
                .properties().get(ModelRouter.AUTOLOAD_PROPERTY), "the last flag wins");
        assertEquals("0", ServerLaunchArgs.parse("--models-max", "0").properties()
                .get(ModelRouter.MAX_PROPERTY), "0 means no limit");
    }

    @Test
    void theRouterFlagsRefuseUnusableValuesByName() throws IOException {
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--models-dir",
                dir.resolve("missing").toString())).contains("--models-dir"));
        Path notAFile = Files.createDirectories(dir.resolve("preset-dir"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--models-preset", notAFile.toString()))
                .contains("--models-preset"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--models-max", "-1")).contains("--models-max"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--models-max", "many")).contains("--models-max"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--models-memory-margin", "-1"))
                .contains("--models-memory-margin"));
    }

    @Test
    void llamaFlagsMapToTheirProperties() throws IOException {
        Path keys = Files.write(dir.resolve("keys.txt"), new byte[0]);
        Map<String, String> properties = ServerLaunchArgs.parse(
                "-m", "model.gguf", "--host", "0.0.0.0", "--port", "8081", "-a", "qwen3", "-c", "32768",
                "-np", "4", "--chat-template", "chatml", "--api-key", "k1",
                "--api-key-file", keys.toString(), "--slot-save-path", dir.toString()).properties();
        Map<String, String> expected = Map.of(
                "neuralfree.server.model", "model.gguf",
                "quarkus.http.host", "0.0.0.0",
                "quarkus.http.port", "8081",
                "neuralfree.server.model.name", "qwen3",
                "neuralfree.ctx", "32768",
                "neuralfree.parallel", "4",
                "neuralfree.chat.template", "chatml",
                "neuralfree.server.api.key", "k1",
                "neuralfree.server.api.key.file", keys.toString(),
                ServerOptions.SLOT_SAVE_PATH_PROPERTY, dir.toString());
        assertEquals(expected, properties);
    }

    @Test
    void allowedHostsMapsToItsPropertyAndRefusesUnsupportedEntries() {
        String hosts = "proxy.example,[::1],192.0.2.4";
        assertEquals(hosts, ServerLaunchArgs.parse("--allowed-hosts", hosts).properties()
                .get("neuralfree.server.allowed.hosts"));
        assertEquals("*", ServerLaunchArgs.parse("--allowed-hosts", "*").properties()
                .get("neuralfree.server.allowed.hosts"));
        for (String invalid : new String[]{"https://proxy.example", "proxy.example:8080", "*.example", "ok,"}) {
            assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--allowed-hosts", invalid))
                    .contains("--allowed-hosts"));
        }
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--allowed-hosts")).contains("--allowed-hosts"));
    }

    @Test
    void theHostRamCacheSizeMapsToItsProperty() {
        assertEquals("8192", ServerLaunchArgs.parse("--cache-ram", "8192").properties()
                .get(MemoryConfig.PROP_CACHE_RAM));
        assertEquals("-1", ServerLaunchArgs.parse("-cram", "-1").properties().get(MemoryConfig.PROP_CACHE_RAM));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cache-ram", "-2")).contains("--cache-ram"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-cram", "lots")).contains("-cram"));
    }

    @Test
    void theDiskCacheSizeAndDirectoryMapToTheirProperties() {
        assertEquals("-1", ServerLaunchArgs.parse("--cache-disk", "-1").properties()
                .get(MemoryConfig.PROP_CACHE_DISK_MIB));
        assertEquals("512", ServerLaunchArgs.parse("-cdisk", "512").properties()
                .get(MemoryConfig.PROP_CACHE_DISK_MIB));
        assertEquals(dir.toString(), ServerLaunchArgs.parse("--cache-disk-dir", dir.toString()).properties()
                .get(MemoryConfig.PROP_CACHE_DISK_DIR));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cache-disk", "-2")).contains("--cache-disk"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cache-disk-dir",
                dir.resolve("missing").toString())).contains("--cache-disk-dir"));
    }

    @Test
    void vllmSpellingsMapToTheSameProperties() throws IOException {
        Path cert = Files.write(dir.resolve("cert.pem"), new byte[0]);
        Path key = Files.write(dir.resolve("key.pem"), new byte[0]);
        Map<String, String> properties = ServerLaunchArgs.parse(
                "model.gguf", "--served-model-name", "qwen3", "--max-model-len", "4096",
                "--ssl-certfile", cert.toString(), "--ssl-keyfile", key.toString(),
                "--trust-request-chat-template").properties();
        assertEquals("model.gguf", properties.get("neuralfree.server.model"));
        assertEquals("true", properties.get(ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY));
        assertEquals("qwen3", properties.get(ServedModels.PROPERTY));
        assertEquals("4096", properties.get("neuralfree.ctx"));
        assertEquals(cert.toString(), properties.get("quarkus.http.ssl.certificate.files"));
        assertEquals(key.toString(), properties.get("quarkus.http.ssl.certificate.key-files"));
    }

    @Test
    void reasoningPreserveFlagsSetTheDefaultOfPreserveReasoning() {
        assertEquals("true", ServerLaunchArgs.parse("--reasoning-preserve").properties()
                .get(ChatFormatter.PRESERVE_REASONING_PROPERTY));
        assertEquals("false", ServerLaunchArgs.parse("--no-reasoning-preserve").properties()
                .get(ChatFormatter.PRESERVE_REASONING_PROPERTY));
        assertEquals("true", ServerLaunchArgs.parse("--no-reasoning-preserve", "--reasoning-preserve")
                .properties().get(ChatFormatter.PRESERVE_REASONING_PROPERTY), "the last flag wins");
        assertFalse(ServerLaunchArgs.parse("model.gguf").properties()
                .containsKey(ChatFormatter.PRESERVE_REASONING_PROPERTY));
    }

    @Test
    void theContentFormatFlagTakesVllmsValues() {
        assertEquals("openai", ServerLaunchArgs.parse("--chat-template-content-format", "openai").properties()
                .get(ChatFormatter.CONTENT_FORMAT_PROPERTY));
        assertEquals("string", ServerLaunchArgs.parse("--chat-template-content-format", "string").properties()
                .get(ChatFormatter.CONTENT_FORMAT_PROPERTY));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse("--chat-template-content-format", "typed"));
        assertTrue(e.getMessage().contains("--chat-template-content-format"), e.getMessage());
    }

    @Test
    void tlsServesOnTheLaunchPortAndPlainHttpGoesOff() throws IOException {
        Path cert = Files.write(dir.resolve("cert.pem"), new byte[0]);
        Path key = Files.write(dir.resolve("key.pem"), new byte[0]);
        Map<String, String> tls = ServerLaunchArgs.parse(
                "--port", "18435", "--ssl-cert-file", cert.toString(), "--ssl-key-file", key.toString())
                .properties();
        assertEquals("18435", tls.get("quarkus.http.ssl-port"));
        assertEquals("disabled", tls.get("quarkus.http.insecure-requests"));
        Map<String, String> defaultPort = ServerLaunchArgs.parse(
                "--ssl-cert-file", cert.toString(), "--ssl-keyfile", key.toString()).properties();
        assertEquals("8080", defaultPort.get("quarkus.http.ssl-port"));
    }

    @Test
    void repeatedKeysAndAdaptersJoinTheirProperty() {
        Map<String, String> properties = ServerLaunchArgs.parse(
                "--api-key", "k1", "--api-key", "k2", "--lora", "a.gguf", "--lora-scaled", "b.gguf", "0.5")
                .properties();
        assertEquals("k1,k2", properties.get(ApiKeys.PROPERTY));
        assertEquals("a.gguf,b.gguf:0.5", properties.get("neuralfree.lora"));
    }

    @Test
    void acceptedTemplateFlagsAndLastValueWins() {
        Map<String, String> properties = ServerLaunchArgs.parse(
                "--jinja", "--metrics", "--port", "1", "--port", "8090").properties();
        assertEquals(Map.of("quarkus.http.port", "8090", ChatTemplateSource.JINJA_PROPERTY, "true"), properties);
    }

    @Test
    void theJinjaSwitchFlagsMapToTheirPropertyAndNoJinjaVerifiesItsOverride() throws IOException {
        String key = ChatTemplateSource.JINJA_PROPERTY;
        assertEquals("true", ServerLaunchArgs.parse("--jinja").properties().get(key));
        assertEquals("false", ServerLaunchArgs.parse("--no-jinja").properties().get(key));
        assertEquals("true", ServerLaunchArgs.parse("--no-jinja", "--jinja").properties().get(key),
                "a later flag wins, so the pair stays order-independent");
        assertEquals("false", ServerLaunchArgs.parse("--jinja", "--no-jinja").properties().get(key));

        Map<String, String> p = ServerLaunchArgs.parse("--no-jinja", "--chat-template", "chatml").properties();
        assertEquals("false", p.get(key));
        assertEquals("chatml", p.get(ChatTemplateSource.OVERRIDE_PROPERTY));
        Path file = Files.writeString(dir.resolve("zephyr.jinja"),
                "<|user|>\n{{ m['content'] }}<|endoftext|>\n");
        p = ServerLaunchArgs.parse("--no-jinja", "--chat-template-file", file.toString()).properties();
        assertEquals("false", p.get(key));
        assertEquals(file.toString(), p.get(ChatTemplateSource.OVERRIDE_FILE_PROPERTY));
        assertTrue(ServerLaunchArgs.parse("--no-jinja").properties().containsKey(key),
                "without an override the model's own template is detected at load");
    }

    @Test
    void noJinjaRefusesAnOverrideNothingIsDetectedFrom() throws IOException {
        String refusal = refusalOf(() -> ServerLaunchArgs.parse("--no-jinja", "--chat-template",
                "[{{ m['role'] }}]{{ m['content'] }}"));
        assertTrue(refusal.startsWith("--no-jinja: no built-in chat template was detected"), refusal);
        Path file = Files.writeString(dir.resolve("plain.jinja"), "[{{ m['role'] }}]{{ m['content'] }}");
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--no-jinja", "--chat-template-file", file.toString()))
                .startsWith("--no-jinja: "));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--no-jinja", "--chat-template", "dots1"))
                        .contains("'dots1'"), "the detection-only name is no --chat-template value");
    }

    @Test
    void helpShortCircuitsBeforeAnyOtherFlagIsRead() {
        ServerLaunchArgs help = ServerLaunchArgs.parse("--help", "--port", "not-a-number");
        assertTrue(help.help());
        assertTrue(help.properties().isEmpty());
        assertFalse(ServerLaunchArgs.parse("-m", "m.gguf").help());
    }

    @Test
    void refusalsNameTheirFlag() throws IOException {
        Path cert = Files.write(dir.resolve("c.pem"), new byte[0]);
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--port")).contains("--port"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--port", "0")).contains("--port"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--port", "abc")).contains("--port"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-np", "0")).contains("--parallel"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-c", "-1")).contains("--ctx-size"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--api-key", " ")).contains("--api-key"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--slot-save-path", "absent-dir"))
                .contains("--slot-save-path"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--chat-template-file", "absent.jinja"))
                .contains("--chat-template-file"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--chat-template", "absent.jinja"))
                .startsWith("--chat-template: "));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--lora", "a:b")).contains("--lora-scaled"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--lora-scaled", "a.gguf", "big")).contains("--lora-scaled"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--lora-modules", "x=y")).contains("--lora-modules"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--ssl-cert-file", cert.toString()))
                .contains("--ssl-key-file"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--ssl-key-file", cert.toString()))
                .contains("--ssl-cert-file"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--sql-magic")).contains("--sql-magic"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-m", "a.gguf", "b.gguf")).contains("twice"));
    }

    @Test
    void theSchedulingPolicyMapsByNameAndRefusesOthersNamingTheFlag() {
        String key = MemoryConfig.PROP_SCHEDULING_POLICY;
        assertEquals("priority", ServerLaunchArgs.parse("--scheduling-policy", "priority").properties().get(key));
        assertEquals("priority", ServerLaunchArgs.parse("--scheduling-policy", "PRIORITY").properties().get(key),
                "stored canonical");
        assertEquals("fcfs", ServerLaunchArgs.parse("--scheduling-policy", "fcfs").properties().get(key));
        Properties p = new Properties();
        p.putAll(ServerLaunchArgs.parse("--scheduling-policy", "priority").properties());
        assertEquals(SchedulingPolicy.PRIORITY, MemoryConfig.fromProperties(p).schedulingPolicy(),
                "the engine reads what the flag sets");
        String refusal = refusalOf(() -> ServerLaunchArgs.parse("--scheduling-policy", "lifo"));
        assertTrue(refusal.startsWith("--scheduling-policy=lifo"), refusal);
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--scheduling-policy")).contains("needs a value"));
    }

    /** llama-server's checkpoint flags land on the same properties the CLI documents. */
    @Test
    void theCheckpointFlagsSetTheMemoryProperties() {
        Properties p = new Properties();
        p.putAll(ServerLaunchArgs.parse("--ctx-checkpoints", "8", "--checkpoint-min-step", "2048").properties());
        assertEquals(8, MemoryConfig.fromProperties(p).ctxCheckpoints());
        assertEquals(2048, MemoryConfig.fromProperties(p).checkpointMinStep());
        assertEquals("8", ServerLaunchArgs.parse("--swa-checkpoints", "8").properties()
                .get(MemoryConfig.PROP_CTX_CHECKPOINTS));
        assertEquals("8", ServerLaunchArgs.parse("-ctxcp", "8").properties()
                .get(MemoryConfig.PROP_CTX_CHECKPOINTS));
        assertEquals("0", ServerLaunchArgs.parse("-cms", "0").properties()
                .get(MemoryConfig.PROP_CHECKPOINT_MIN_STEP));
        Properties zero = new Properties();
        zero.putAll(ServerLaunchArgs.parse("--ctx-checkpoints", "0").properties());
        assertEquals(0, MemoryConfig.fromProperties(zero).ctxCheckpoints());
        String refusal = refusalOf(() -> ServerLaunchArgs.parse("--ctx-checkpoints", "-1"));
        assertTrue(refusal.contains("--ctx-checkpoints"), refusal);
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-cms", "late")).contains("-cms"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--checkpoint-min-step")).contains("needs a value"));
    }

    @Test
    void thePrefixCachingHashAlgoMapsByNameAndRefusesOthersNamingTheFlag() {
        String key = MemoryConfig.PROP_PREFIX_HASH_ALGO;
        assertEquals("sha256", ServerLaunchArgs.parse("--prefix-caching-hash-algo", "sha256").properties().get(key));
        assertEquals("sha256_cbor", ServerLaunchArgs.parse("--prefix-caching-hash-algo", "SHA256_CBOR")
                .properties().get(key), "stored canonical");
        assertEquals("xxhash", ServerLaunchArgs.parse("--prefix-caching-hash-algo", "xxhash").properties().get(key));
        assertEquals("xxhash_cbor", ServerLaunchArgs.parse("--prefix-caching-hash-algo", "xxhash_cbor")
                .properties().get(key));
        Properties p = new Properties();
        p.putAll(ServerLaunchArgs.parse("--prefix-caching-hash-algo", "sha256_cbor").properties());
        assertEquals(PrefixHashAlgo.SHA256_CBOR, MemoryConfig.fromProperties(p).prefixHashAlgo(),
                "the engine hashes with what the flag selects");
        String refusal = refusalOf(() -> ServerLaunchArgs.parse("--prefix-caching-hash-algo", "murmur"));
        assertTrue(refusal.startsWith("--prefix-caching-hash-algo=murmur"), refusal);
        assertTrue(refusal.contains("sha256_cbor"), refusal);
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--prefix-caching-hash-algo")).contains("needs a value"));
    }

    @Test
    void theKvEventsConfigPicksThePublisher() {
        String key = MemoryConfig.PROP_KV_EVENTS;
        assertEquals("zmq", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": true}").properties().get(key),
                "enabled without a publisher publishes on vLLM's ZMQ transport");
        assertEquals("zmq", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": true, \"publisher\": null}").properties().get(key));
        assertEquals("zmq", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\"}").properties().get(key));
        assertEquals("log", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": true, \"publisher\": \"log\"}").properties().get(key));
        assertEquals("none", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": true, \"publisher\": \"null\"}").properties().get(key));
        assertEquals("none", ServerLaunchArgs.parse("--kv-events-config", "{}").properties().get(key),
                "vLLM's default leaves events off");
        assertEquals("none", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": false, \"endpoint\": \"tcp://*:6000\"}").properties().get(key));
        assertEquals("com.example.Sink", ServerLaunchArgs.parse("--kv-events-config",
                "{\"enable_kv_cache_events\": true, \"publisher\": \"com.example.Sink\"}").properties().get(key));
    }

    @Test
    void theKvTransferConfigMapsAndRefusesUnsupportedConnectors() {
        String key = ServerOptions.KV_TRANSFER_CONFIG_PROPERTY;
        Map<String, String> properties = ServerLaunchArgs.parse("--kv-transfer-config",
                "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_producer\","
                        + " \"engine_id\": \"prefill-1\"}").properties();
        KvTransferConfig config = KvTransferConfig.parse(properties.get(key), "--kv-transfer-config");
        assertEquals("TcpKvConnector", config.kvConnector());
        assertEquals(KvRole.PRODUCER, config.role());
        assertEquals("prefill-1", config.engineId());

        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse("--kv-transfer-config", "{\"kv_role\": \"kv_both\"}"));
        assertTrue(missing.getMessage().contains("kv_connector"), missing.getMessage());
        KvTransferConfig nixlConfig = KvTransferConfig.parse(ServerLaunchArgs.parse("--kv-transfer-config",
                "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_both\"}").properties().get(key),
                "--kv-transfer-config");
        UnsupportedOperationException nixl = assertThrows(UnsupportedOperationException.class,
                () -> KvConnectors.create(nixlConfig));
        assertTrue(nixl.getMessage().contains("NixlConnector"), nixl.getMessage());
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--kv-transfer-config")).contains("needs a value"));
    }

    @Test
    void theKvTransferConfigValidatesOverTheLaunchEnvironment() {
        String key = ServerOptions.KV_TRANSFER_CONFIG_PROPERTY;
        String json = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_consumer\","
                + " \"kv_connector_extra_config\": {\"x\": 1}}";
        assertEquals(json, ServerLaunchArgs.parse(Map.of(KvTransferConfig.ENV_TOKEN, "from-env"),
                "--kv-transfer-config", json).properties().get(key));
        assertEquals(json, ServerLaunchArgs.parse("--kv-transfer-config", json).properties().get(key));

        String inJson = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_consumer\","
                + " \"kv_connector_extra_config\": {\"auth_token\": \"in-json\"}}";
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse(Map.of(KvTransferConfig.ENV_TOKEN, "from-env"),
                        "--kv-transfer-config", inJson));
        assertTrue(refused.getMessage().contains(KvTransferConfig.ENV_TOKEN), refused.getMessage());
        assertTrue(refused.getMessage().contains("auth_token"), refused.getMessage());
    }

    @Test
    void theKvEventsConfigCarriesTheZmqTransportFields() {
        String json = "{\"enable_kv_cache_events\": true, \"publisher\": \"zmq\", \"endpoint\": \"tcp://*:6000\","
                + " \"replay_endpoint\": \"tcp://*:6001\", \"buffer_steps\": 64, \"hwm\": 500,"
                + " \"max_queue_size\": 32, \"topic\": \"kv\"}";
        String parsed = ServerLaunchArgs.parse("--kv-events-config", json).properties()
                .get(MemoryConfig.PROP_KV_EVENTS);
        assertEquals(json, parsed, "the property carries the config itself");
        MemoryConfig memory = MemoryConfig.builder().kvEvents(parsed).build();
        assertEquals(new ZmqEventPublisherConfig("tcp://*:6000", "tcp://*:6001", 64, 500, 32, "kv", 0),
                KvEventsConfig.parse(memory.kvEvents(), MemoryConfig.PROP_KV_EVENTS).zmq(),
                "the engine's memory config reads every transport field");
    }

    @Test
    void theKvEventsConfigRefusesWhatItCannotCarry() {
        String flag = "--kv-events-config";
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag)).contains(flag));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag, "not json")).contains(flag));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag, "[1]")).contains("JSON object"));
        String[][] bad = {{"endpoint", "\"ipc:///tmp/kv\""}, {"replay_endpoint", "\"inproc://kv\""},
                {"buffer_steps", "-1"}, {"hwm", "\"x\""}, {"max_queue_size", "1.5"}, {"topic", "3"}};
        for (String[] field : bad) {
            String json = "{\"enable_kv_cache_events\": true, \"" + field[0] + "\": " + field[1] + "}";
            String refusal = refusalOf(() -> ServerLaunchArgs.parse(flag, json));
            assertTrue(refusal.contains(flag) && refusal.contains(field[0]), refusal);
        }
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag,
                "{\"enable_kv_cache_events\": true, \"publisher\": \"log\", \"endpoint\": \"tcp://*:6000\"}"))
                .contains("endpoint"), "a transport field beside another publisher is read by nothing");
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag, "{\"publish\": \"log\"}")).contains("publish"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag, "{\"enable_kv_cache_events\": \"yes\"}"))
                .contains("enable_kv_cache_events"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(flag,
                "{\"enable_kv_cache_events\": true, \"publisher\": 3}")).contains("publisher"));
    }

    @Test
    void emptyArgumentsBootUnloadedWithoutHelp() {
        ServerLaunchArgs empty = ServerLaunchArgs.parse();
        assertFalse(empty.help());
        assertTrue(empty.properties().isEmpty());
    }

    @Test
    void mediaFlagsMapToTheMediaProperties() throws IOException {
        Path mmproj = Files.write(dir.resolve("mmproj.gguf"), new byte[0]);
        Map<String, String> properties = ServerLaunchArgs.parse(
                "-m", "model.gguf", "--mmproj", mmproj.toString(), "--allowed-local-media-path", dir.toString(),
                "--allowed-media-domains", "upload.wikimedia.org,example.com",
                "--limit-mm-per-prompt", "{\"image\": 2, \"audio\": 0}",
                "--image-min-tokens", "64", "--image-max-tokens", "1024",
                "--mm-processor-cache-gb", "0.5",
                "--media-io-kwargs", "{\"image\": {\"image_mode\": \"L\"}}").properties();
        assertEquals(mmproj.toString(), properties.get(MediaOptions.MMPROJ_PROPERTY));
        assertEquals(dir.toString(), properties.get(MediaOptions.LOCAL_PATH_PROPERTY));
        assertEquals("upload.wikimedia.org,example.com", properties.get(MediaOptions.DOMAINS_PROPERTY));
        assertEquals("{\"image\": 2, \"audio\": 0}", properties.get(MediaOptions.LIMITS_PROPERTY));
        assertEquals("64", properties.get(MediaOptions.IMAGE_MIN_TOKENS_PROPERTY));
        assertEquals("1024", properties.get(MediaOptions.IMAGE_MAX_TOKENS_PROPERTY));
        assertEquals("0.5", properties.get(MediaOptions.CACHE_GB_PROPERTY));
        assertEquals(1L << 29, MediaOptions.cacheBytes(properties.get(MediaOptions.CACHE_GB_PROPERTY)));
        assertEquals(0L, MediaOptions.cacheBytes("0"));
        assertEquals(MediaOptions.defaultCacheBytes(), MediaOptions.cacheBytes(null));
        assertEquals("{\"image\": {\"image_mode\": \"L\"}}", properties.get(MediaOptions.IO_KWARGS_PROPERTY));
        assertEquals(ImageLoadOptions.Mode.L,
                MediaIoOptions.parse(properties.get(MediaOptions.IO_KWARGS_PROPERTY)).image().mode());
        assertEquals(dir.toString(), ServerLaunchArgs.parse("--media-path", dir.toString()).properties()
                .get(MediaOptions.LOCAL_PATH_PROPERTY));
        assertEquals(mmproj.toString(), ServerLaunchArgs.parse("-mm", mmproj.toString()).properties()
                .get(MediaOptions.MMPROJ_PROPERTY));
    }

    @Test
    void videoFlagsMapToTheVideoProperties() throws IOException {
        Map<String, String> properties = ServerLaunchArgs.parse(
                "--video-fps", "2.5", "--video-timestamp-interval", "1000",
                "--video-ffmpeg-dir", dir.toString(), "--video-pruning-rate", "0.25",
                "--video-pruning-method", "VIDCOM2").properties();
        assertEquals("2.5", properties.get(MediaOptions.VIDEO_FPS_PROPERTY));
        assertEquals("1000", properties.get(MediaOptions.VIDEO_TIMESTAMP_PROPERTY));
        assertEquals(dir.toString(), properties.get(MediaOptions.VIDEO_FFMPEG_DIR_PROPERTY));
        assertEquals("0.25", properties.get(MediaOptions.VIDEO_PRUNING_RATE_PROPERTY));
        assertEquals("vidcom2", properties.get(MediaOptions.VIDEO_PRUNING_METHOD_PROPERTY));
        // 0 is the source rate and interval; malformed values are refused naming the flag
        assertEquals("0", ServerLaunchArgs.parse("--video-fps", "0", "--video-timestamp-interval", "0")
                .properties().get(MediaOptions.VIDEO_FPS_PROPERTY));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--video-fps", "-1")).startsWith("--video-fps"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--video-fps", "fast")).startsWith("--video-fps"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--video-pruning-rate", "1"))
                .startsWith("--video-pruning-rate"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--video-pruning-method", "attention"))
                .startsWith("--video-pruning-method"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--video-ffmpeg-dir", dir.resolve("nope").toString()))
                .startsWith("--video-ffmpeg-dir is not a directory"));
    }

    @Test
    void noMmprojTurnsMediaInputOffWhereverItStands() throws IOException {
        Path mmproj = Files.write(dir.resolve("mmproj.gguf"), new byte[0]);
        assertEquals("", ServerLaunchArgs.parse("--no-mmproj", "--mmproj", mmproj.toString()).properties()
                .get(MediaOptions.MMPROJ_PROPERTY));
        assertEquals("", ServerLaunchArgs.parse("--mmproj", mmproj.toString(), "--no-mmproj").properties()
                .get(MediaOptions.MMPROJ_PROPERTY));
    }

    @Test
    void theProjectorUrlFlagDownloadsThroughItsProperty() {
        assertEquals("https://h/mmproj-Q8_0.gguf", ServerLaunchArgs.parse("-mmu", "https://h/mmproj-Q8_0.gguf")
                .properties().get(MediaOptions.MMPROJ_URL_PROPERTY));
        assertEquals("http://h/p.gguf", ServerLaunchArgs.parse("--mmproj-url", "http://h/p.gguf")
                .properties().get(MediaOptions.MMPROJ_URL_PROPERTY));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mmproj-url", "ftp://h/p.gguf"))
                .startsWith("--mmproj-url must be an http or https URL"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-mmu", "/local/p.gguf"))
                .startsWith("-mmu must be an http or https URL"));
    }

    @Test
    void theProjectorSwitchFamilyFollowsItsLastFlag() throws IOException {
        Path mmproj = Files.write(dir.resolve("mmproj.gguf"), new byte[0]);
        Map<String, String> off = ServerLaunchArgs.parse("--mmproj", mmproj.toString(), "--no-mmproj").properties();
        assertEquals("", off.get(MediaOptions.MMPROJ_PROPERTY));
        assertEquals("", off.get(MediaOptions.MMPROJ_URL_PROPERTY));
        assertEquals("false", off.get(MediaOptions.MMPROJ_AUTO_PROPERTY));
        assertEquals("true", ServerLaunchArgs.parse("--mmproj-auto").properties()
                .get(MediaOptions.MMPROJ_AUTO_PROPERTY));
        assertEquals("false", ServerLaunchArgs.parse("--no-mmproj-auto").properties()
                .get(MediaOptions.MMPROJ_AUTO_PROPERTY));
        Map<String, String> backOn = ServerLaunchArgs.parse("--no-mmproj", "--mmproj-auto").properties();
        assertEquals("true", backOn.get(MediaOptions.MMPROJ_AUTO_PROPERTY));
        assertFalse(backOn.containsKey(MediaOptions.MMPROJ_PROPERTY));
        assertFalse(backOn.containsKey(MediaOptions.MMPROJ_URL_PROPERTY));
    }

    @Test
    void mmprojDeviceFlagsChooseWhereTheVisionEncoderRuns() {
        assertEquals(MmprojOptions.cpu(), encoderOf("--no-mmproj-offload"));
        assertEquals(MmprojOptions.defaults(), encoderOf("--mmproj-offload"));
        assertEquals(MmprojOptions.onDevice("RTX"), encoderOf("-mmdev", "RTX"));
        assertEquals(MmprojOptions.cpu(), encoderOf("--mmproj-device", "none"));
        // the last flag wins, as in llama.cpp: a device turns the offload back on, none or --no-mmproj-offload off
        assertEquals(MmprojOptions.onDevice("1"), encoderOf("--no-mmproj-offload", "--mmproj-device", "1"));
        assertEquals(MmprojOptions.cpu(), encoderOf("--mmproj-device", "1", "--no-mmproj-offload"));
        assertEquals(MmprojOptions.defaults(), encoderOf("-mmdev", "none", "--mmproj-offload"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-mmdev", "0,1")).startsWith("-mmdev: only one device"));
    }

    /** The encoder options the launch properties of {@code args} give. */
    private static MmprojOptions encoderOf(String... args) {
        Properties p = new Properties();
        p.putAll(ServerLaunchArgs.parse(args).properties());
        return MmprojOptions.fromProperties(p);
    }

    @Test
    void encoderScaleOutFlagsChooseTheParallelismAndTheRankGroup() {
        assertEquals(MmEncoderTpMode.WEIGHTS, encoderOf("--mm-encoder-tp-mode", "weights").tpMode());
        assertEquals(MmEncoderTpMode.DATA, encoderOf("--mm-encoder-tp-mode", "data").tpMode());
        MmprojOptions group = encoderOf("--mm-encoder-tp-mode", "data", "--mmproj-devices", "0, RTX");
        assertEquals(MmEncoderTpMode.DATA, group.tpMode());
        assertEquals(java.util.Arrays.asList("0", "RTX"), group.devices());
        assertEquals("0,RTX", ServerLaunchArgs.parse("--mm-encoder-tp-mode", "data", "--mmproj-devices",
                " 0 , RTX ").properties().get(MmprojOptions.PROP_DEVICES));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mmproj-devices", "0,1"))
                .contains("--mm-encoder-tp-mode"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mmproj-devices", "0,,1"))
                .startsWith("--mmproj-devices:"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mm-encoder-tp-mode", "tensor"))
                .startsWith("--mm-encoder-tp-mode:"));
    }

    @Test
    void ecTransferConfigNamesTheSharedEncoderCache() throws IOException {
        Path shared = Files.createDirectories(dir.resolve("shared"));
        Map<String, String> p = ServerLaunchArgs.parse("--ec-transfer-config",
                "{\"ec_connector\": \"ECExampleConnector\", \"ec_role\": \"ec_both\","
                        + " \"ec_connector_extra_config\": {\"shared_storage_path\": \"" + shared + "\"}}")
                .properties();
        assertEquals(shared.toString(), p.get(EncoderCacheConfig.PROP_DIR));
        assertEquals("ec_both", p.get(EncoderCacheConfig.PROP_ROLE));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--ec-transfer-config",
                "{\"ec_connector\": \"ECMooncakeConnector\", \"ec_role\": \"ec_producer\"}"))
                .contains("ECMooncakeConnector"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--ec-transfer-config", "nope"))
                .startsWith("--ec-transfer-config is not valid JSON"));
    }

    @Test
    void unusableMediaFlagsAreRefusedByName() {
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mmproj", dir.resolve("absent.gguf").toString()))
                .startsWith("--mmproj is not a regular file"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--media-path", dir.resolve("absent").toString()))
                .startsWith("--media-path is not a directory"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--limit-mm-per-prompt", "{\"pictures\": 1}"))
                .contains("'pictures'"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--image-max-tokens", "-1"))
                .startsWith("--image-max-tokens"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mm-processor-cache-gb", "-1"))
                .startsWith("--mm-processor-cache-gb must be a non-negative number of GiB"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mm-processor-cache-gb", "lots"))
                .contains("'lots'"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--media-io-kwargs", "{\"video\": {\"backend\": \"opencv\"}}"))
                .startsWith("--media-io-kwargs video option 'backend' is not supported"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--media-io-kwargs", "{\"image\": {\"crop\": 1}}"))
                .contains("image option 'crop' is not supported"));
    }

    /** The pooling override of llama.cpp's {@code --pooling} and vLLM's pooler config JSON. */
    @Test
    void poolingFlagsMapToThePoolingProperty() {
        assertEquals("mean", ServerLaunchArgs.parse("--pooling", "mean").properties().get("neuralfree.pooling"));
        assertEquals("cls", ServerLaunchArgs.parse("--pooling", "CLS").properties().get("neuralfree.pooling"));
        assertEquals("last", ServerLaunchArgs.parse("--override-pooler-config", "{\"pooling_type\": \"LAST\"}")
                .properties().get("neuralfree.pooling"));
        assertEquals("rank", ServerLaunchArgs.parse("--pooler-config", "{\"pooling_type\":\"RANK\"}")
                .properties().get("neuralfree.pooling"));
        assertEquals("none", ServerLaunchArgs.parse("--override-pooler-config", "{\"pooling_type\":\"ALL\"}")
                .properties().get("neuralfree.pooling"));
        assertEquals("mean", ServerLaunchArgs.parse("--pooler-config", "{\"seq_pooling_type\": \"mean\"}")
                .properties().get("neuralfree.pooling"));
        assertEquals("none", ServerLaunchArgs.parse("--pooler-config", "{\"tok_pooling_type\": \"ALL\"}")
                .properties().get("neuralfree.pooling"));
        // vLLM's empty object means the defaults: nothing to override.
        assertFalse(ServerLaunchArgs.parse("--override-pooler-config", "{}").properties()
                .containsKey("neuralfree.pooling"));
        // The last pooling flag wins.
        assertEquals("rank", ServerLaunchArgs.parse("--pooling", "mean", "--pooling", "rank")
                .properties().get("neuralfree.pooling"));
    }

    @Test
    void poolingRefusalsNameTheirFlagAndKey() {
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--pooling", "avg")).contains("--pooling"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--pooling")).contains("--pooling"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--override-pooler-config", "mean"))
                .contains("--override-pooler-config"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--override-pooler-config", "{not json}"))
                .contains("--override-pooler-config"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--override-pooler-config", "[]"))
                .contains("--override-pooler-config"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--override-pooler-config", "{\"pooling_type\":1}"))
                .contains("--override-pooler-config"));
        String unknownKey = refusalOf(() -> ServerLaunchArgs.parse("--override-pooler-config",
                "{\"normalize\": true}"));
        assertTrue(unknownKey.contains("normalize"), unknownKey);
        assertTrue(unknownKey.contains("pooling_type"), unknownKey);
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--pooler-config", "{\"pooling_type\":\"STEP\"}"))
                .contains("STEP"));
        // vLLM refuses pooling_type beside the split types; one engine pooling refuses the two split types together.
        String both = refusalOf(() -> ServerLaunchArgs.parse("--pooler-config",
                "{\"pooling_type\": \"MEAN\", \"seq_pooling_type\": \"CLS\"}"));
        assertTrue(both.contains("pooling_type") && both.contains("seq_pooling_type"), both);
        String split = refusalOf(() -> ServerLaunchArgs.parse("--pooler-config",
                "{\"seq_pooling_type\": \"CLS\", \"tok_pooling_type\": \"ALL\"}"));
        assertTrue(split.contains("tok_pooling_type"), split);
        String badSeq = refusalOf(() -> ServerLaunchArgs.parse("--pooler-config", "{\"seq_pooling_type\": \"ALL\"}"));
        assertTrue(badSeq.contains("seq_pooling_type") && badSeq.contains("ALL"), badSeq);
    }

    @Test
    void theToolFlagsMapToTheirProperties() {
        assertEquals("all", ServerLaunchArgs.parse("--tools", "all").properties()
                .get(ServerOptions.TOOLS_PROPERTY));
        assertEquals("read_file,get_info", ServerLaunchArgs.parse("--tools", "read_file,get_info").properties()
                .get(ServerOptions.TOOLS_PROPERTY));
        assertEquals("all", ServerLaunchArgs.parse("--agent").properties().get(ServerOptions.TOOLS_PROPERTY));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--tools", "nope")).
                contains("nope"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--tools", "nope")).contains("read_file"),
                "the refusal lists the available tools");
    }

    @Test
    void theMcpFlagsMapToTheirProperties(@TempDir Path dir) throws IOException {
        Path config = Files.writeString(dir.resolve("mcp.json"),
                "{\"mcpServers\": {\"one\": {\"command\": \"c\"}}}");
        assertEquals(config.toString(), ServerLaunchArgs.parse("--mcp-servers-config", config.toString())
                .properties().get(ServerOptions.MCP_CONFIG_PROPERTY));
        String inline = "{\"mcpServers\": {\"one\": {\"command\": \"c\"}}}";
        assertEquals(inline, ServerLaunchArgs.parse("--mcp-servers-json", inline).properties()
                .get(ServerOptions.MCP_JSON_PROPERTY));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mcp-servers-config",
                dir.resolve("absent.json").toString())).contains("--mcp-servers-config"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--mcp-servers-json", "{not json"))
                .contains("--mcp-servers-json"));
    }

    @Test
    void theToolsRuntimeIsRefusedByThisBuild() {
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--tools-runtime", "docker:alpine"))
                .contains("--tools-runtime"));
    }

    @Test
    void theUiSwitchFlagsMapToTheWebuiProperty() {
        assertEquals("true", ServerLaunchArgs.parse("--ui").properties().get(UiResource.WEBUI_PROPERTY));
        assertEquals("true", ServerLaunchArgs.parse("--webui").properties().get(UiResource.WEBUI_PROPERTY));
        assertEquals("false", ServerLaunchArgs.parse("--no-ui").properties().get(UiResource.WEBUI_PROPERTY));
        assertEquals("false", ServerLaunchArgs.parse("--no-webui").properties().get(UiResource.WEBUI_PROPERTY));
        assertFalse(ServerLaunchArgs.parse("-m", "model.gguf").properties().containsKey(UiResource.WEBUI_PROPERTY),
                "without a flag the UI stays on by default");
    }

    @Test
    void theStaticPathFlagMapsToItsPropertyAndRefusesUnusableDirectories() {
        assertEquals(dir.toString(), ServerLaunchArgs.parse("--path", dir.toString()).properties()
                .get(UiResource.PATH_PROPERTY));
        String missing = refusalOf(() -> ServerLaunchArgs.parse("--path", dir.resolve("absent").toString()));
        assertTrue(missing.contains("--path") && missing.contains("absent"), missing);
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--path")).contains("--path"));
    }

    @Test
    void theUiConfigFlagsMapToTheConfigPropertyAndValidateTheirJson() throws IOException {
        String json = "{\"temperature\": 0.5} ";
        assertEquals("{\"temperature\": 0.5}", ServerLaunchArgs.parse("--ui-config", json).properties()
                .get(UiResource.CONFIG_PROPERTY), "the value is trimmed, kept as JSON text");
        assertEquals("true", ServerLaunchArgs.parse("--webui-config", "true").properties()
                .get(UiResource.CONFIG_PROPERTY), "llama.cpp's ui_config takes any JSON value");
        Path file = dir.resolve("ui-config.json");
        Files.writeString(file, "{\"theme\": \"light\"}\n");
        assertEquals("{\"theme\": \"light\"}", ServerLaunchArgs.parse("--ui-config-file", file.toString())
                .properties().get(UiResource.CONFIG_PROPERTY));
        assertEquals("{\"theme\": \"light\"}", ServerLaunchArgs.parse("--webui-config-file", file.toString())
                .properties().get(UiResource.CONFIG_PROPERTY), "the alias reads the same file");
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--ui-config", "{not json}")).contains("--ui-config"));
        String missingFile = refusalOf(() -> ServerLaunchArgs.parse("--ui-config-file",
                dir.resolve("absent.json").toString()));
        assertTrue(missingFile.contains("--ui-config-file"), missingFile);
        Path bad = dir.resolve("bad.json");
        Files.writeString(bad, "{oops}");
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--ui-config-file", bad.toString()))
                .contains("--ui-config-file"));
    }

    @Test
    void theApiPrefixFlagCanonicalizesAndRefusesUnusablePrefixes() {
        assertEquals("/api", ServerLaunchArgs.parse("--api-prefix", "api").properties()
                .get(ApiPrefixRouter.PROPERTY));
        assertEquals("/api/v1", ServerLaunchArgs.parse("--api-prefix", "/api/v1/").properties()
                .get(ApiPrefixRouter.PROPERTY));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--api-prefix", "..")).contains("--api-prefix"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--api-prefix", "a b")).contains("--api-prefix"));
    }

    @Test
    void theReusePortFlagMapsToTheQuarkusSocketOption() {
        assertEquals("true", ServerLaunchArgs.parse("--reuse-port").properties()
                .get("quarkus.http.so-reuse-port"));
    }

    @Test
    void theLogitsProcessorsFlagTakesOneOrMoreClassNames() {
        assertEquals("a.b.One,c.d.Two", ServerLaunchArgs.parse("--logits-processors",
                "a.b.One", "c.d.Two").properties().get(EngineConfig.PROP_LOGITS_PROCESSORS));
        assertEquals("a.b.One,c.d.Two", ServerLaunchArgs.parse("--logits_processors",
                "a.b.One,c.d.Two").properties().get(EngineConfig.PROP_LOGITS_PROCESSORS),
                "vLLM's underscore spelling and comma-separated values are accepted");
        assertEquals("a.b.One", ServerLaunchArgs.parse("--logits-processors", "a.b.One", "--port", "8081")
                .properties().get(EngineConfig.PROP_LOGITS_PROCESSORS), "the next flag ends the list");
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--logits-processors")).contains("--logits-processors"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--logits-processors", "a.b.One,"))
                .contains("--logits-processors"));
    }

    @Test
    void theParserFlagsSelectTheRegisteredParsersAndRefuseUnknownNames() {
        assertEquals("hermes", ServerLaunchArgs.parse("--tool-call-parser", "hermes").properties()
                .get("neuralfree.tool.parser"));
        assertEquals("autoparser", ServerLaunchArgs.parse("--tool-call-parser", "autoparser").properties()
                .get("neuralfree.tool.parser"));
        assertEquals("auto", ServerLaunchArgs.parse("--tool-call-parser", "auto").properties()
                .get("neuralfree.tool.parser"));
        assertEquals("none", ServerLaunchArgs.parse("--tool-call-parser", "none").properties()
                .get("neuralfree.tool.parser"));
        assertEquals("qwen3", ServerLaunchArgs.parse("--reasoning-parser", "qwen3").properties()
                .get("neuralfree.reasoning.parser"));
        assertEquals("auto", ServerLaunchArgs.parse("--reasoning-parser", "auto").properties()
                .get("neuralfree.reasoning.parser"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--tool-call-parser", "no_such_parser"))
                .contains("--tool-call-parser: unknown tool-call parser 'no_such_parser'"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--reasoning-parser", "no_such_parser"))
                .contains("--reasoning-parser: unknown reasoning parser 'no_such_parser'"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--tool-call-parser")).contains("--tool-call-parser"));
    }

    @Test
    void enableAutoToolChoiceNeedsTheToolCallParser() {
        Map<String, String> properties = ServerLaunchArgs.parse("--enable-auto-tool-choice",
                "--tool-call-parser", "hermes").properties();
        assertEquals("hermes", properties.get("neuralfree.tool.parser"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--enable-auto-tool-choice"))
                .contains("--enable-auto-tool-choice requires --tool-call-parser"));
    }

    @Test
    void theCorsFlagsMapToTheQuarkusCorsPropertiesAndRefuseUnusableValues() {
        Map<String, String> properties = ServerLaunchArgs.parse(
                "--cors-origins", "http://a.example", "--cors-methods", "GET,PUT",
                "--cors-headers", "X-Custom", "--cors-credentials").properties();
        assertEquals("true", properties.get(CorsLaunchFlags.ENABLED));
        assertEquals("http://a.example", properties.get(CorsLaunchFlags.ORIGINS));
        assertEquals("GET,PUT", properties.get(CorsLaunchFlags.METHODS));
        assertEquals("X-Custom", properties.get(CorsLaunchFlags.HEADERS));
        assertEquals("true", properties.get(CorsLaunchFlags.CREDENTIALS));
        assertEquals("true", properties.get(CorsLaunchFlags.EXACT_ORIGINS));
        assertEquals("false", ServerLaunchArgs.parse("--no-cors-credentials").properties()
                .get(CorsLaunchFlags.CREDENTIALS));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cors-origins", "a,,b")).contains("--cors-origins"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cors-methods", "GET PUT")).contains("--cors-methods"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cors-headers")).contains("--cors-headers"));
    }

    @Test
    void vllmsCorsSpellingsMapToTheSamePropertiesAndRefuseMalformedLists() {
        Map<String, String> properties = ServerLaunchArgs.parse(
                "--allowed-origins", "[\"http://a.example\"]",
                "--allowed-methods", "[\"GET\", \"POST\"]",
                "--allowed-headers", "[\"Authorization\"]",
                "--allow-credentials").properties();
        assertEquals("http://a.example", properties.get(CorsLaunchFlags.ORIGINS));
        assertEquals("GET,POST", properties.get(CorsLaunchFlags.METHODS));
        assertEquals("Authorization", properties.get(CorsLaunchFlags.HEADERS));
        assertEquals("true", properties.get(CorsLaunchFlags.CREDENTIALS));
        assertEquals("true", properties.get(CorsLaunchFlags.ENABLED));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--allowed-origins", "http://a.example"))
                .contains("--allowed-origins"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--allowed-methods", "GET,POST"))
                .contains("--allowed-methods"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--allowed-headers", "[]"))
                .contains("--allowed-headers"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--allowed-origins"))
                .contains("--allowed-origins"));
    }

    @Test
    void theTracesEndpointFlagTurnsTheSdkOnAndPointsTheExporter() {
        Map<String, String> p = ServerLaunchArgs.parse(
                ServerOptions.OTLP_TRACES_ENDPOINT_FLAG, "collector.example:4317").properties();
        assertEquals("collector.example:4317", p.get(ServerOptions.OTLP_TRACES_ENDPOINT_PROPERTY));
        assertEquals("false", p.get("quarkus.otel.sdk.disabled"),
                "the endpoint turns the OpenTelemetry SDK on");
        assertEquals("collector.example:4317", p.get("quarkus.otel.exporter.otlp.traces.endpoint"));

        assertTrue(ServerLaunchArgs.parse("-m", "m.gguf").properties()
                .containsKey("neuralfree.server.model"), "a launch without the flag stays as it is");
        assertFalse(ServerLaunchArgs.parse("-m", "m.gguf").properties()
                .containsKey("quarkus.otel.sdk.disabled"));

        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(ServerOptions.OTLP_TRACES_ENDPOINT_FLAG))
                .contains(ServerOptions.OTLP_TRACES_ENDPOINT_FLAG));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse(ServerOptions.OTLP_TRACES_ENDPOINT_FLAG, " "))
                .contains(ServerOptions.OTLP_TRACES_ENDPOINT_FLAG));
    }

    /** llama.cpp's self-extend and chunk-reuse flags land on the request fields' launch-default properties. */
    @Test
    void theSelfExtendAndCacheReuseFlagsMapToTheirProperties() {
        Map<String, String> p = ServerLaunchArgs.parse(
                "--cache-reuse", "256", "-gan", "2", "-gaw", "64").properties();
        assertEquals("256", p.get(OpenAiRequests.PROP_CACHE_REUSE));
        assertEquals("2", p.get(OpenAiRequests.PROP_GRP_ATTN_N));
        assertEquals("64", p.get(OpenAiRequests.PROP_GRP_ATTN_W));

        Map<String, String> longForms = ServerLaunchArgs.parse(
                "--grp-attn-n", "4", "--grp-attn-w", "512").properties();
        assertEquals("4", longForms.get(OpenAiRequests.PROP_GRP_ATTN_N));
        assertEquals("512", longForms.get(OpenAiRequests.PROP_GRP_ATTN_W));

        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cache-reuse", "-1"))
                .contains("--cache-reuse"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cache-reuse", "x"))
                .contains("--cache-reuse"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--cache-reuse"))
                .contains("--cache-reuse"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("-gan", "0")).contains("-gan"));
        assertTrue(refusalOf(() -> ServerLaunchArgs.parse("--grp-attn-w", "0")).contains("--grp-attn-w"));
    }
}
