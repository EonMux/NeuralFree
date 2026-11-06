package com.eonmux.neuralfree.gguf.rpc;

import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The TLS configuration of the transport: a server context builds from a PEM certificate chain and an unencrypted
 * PKCS#8 key, the property and environment names configure both sides, and the refused shapes are named with the
 * command that fixes them.
 */
public class RpcTlsTest {

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();

    @After
    public void clearTheProperties() {
        System.clearProperty(RpcTls.PROP_CERT);
        System.clearProperty(RpcTls.PROP_KEY);
        System.clearProperty(RpcTls.PROP_TRUSTSTORE);
        System.clearProperty(RpcTls.PROP_TRUSTSTORE_PASSWORD);
    }

    @Test
    public void fromTheEnvironmentConfiguresNothingWhenUnset() {
        assertNull(RpcTls.fromEnvironment());
    }

    @Test
    public void fromTheEnvironmentRefusesHalfAConfiguration() {
        System.setProperty(RpcTls.PROP_CERT, "/etc/rpc/cert.pem");
        try {
            RpcTls.fromEnvironment();
            fail("built TLS from a certificate without a key");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(RpcTls.PROP_KEY));
            assertTrue(expected.getMessage().contains(RpcTls.PROP_CERT));
        }
    }

    @Test
    public void buildsAServerContextFromPemFiles() throws Exception {
        TlsFiles files = new TlsFiles(folder.getRoot().toPath());
        RpcTls.Server server = RpcTls.server(files.cert, files.key);
        assertEquals(files.cert, server.cert());
        assertEquals(files.key, server.key());
        assertNotNull(server.context());
        assertNotNull(RpcTls.serverSocket(server.context()));
    }

    @Test
    public void aPkcs1KeyIsRefusedWithTheConversionCommand() throws Exception {
        TlsFiles files = new TlsFiles(folder.getRoot().toPath());
        Path pkcs1 = folder.newFile("pkcs1.pem").toPath();
        // a PKCS#1 RSA key: the header JDK KeyFactory refuses
        String pem = new String(Files.readAllBytes(files.key), java.nio.charset.StandardCharsets.UTF_8);
        String der = pem.replace("-----BEGIN PRIVATE KEY-----", "").replace("-----END PRIVATE KEY-----", "")
                .replaceAll("\\s", "");
        byte[] pkcs8 = java.util.Base64.getDecoder().decode(der);
        Files.write(pkcs1, ("-----BEGIN RSA PRIVATE KEY-----\n"
                + java.util.Base64.getMimeEncoder(64, "\n".getBytes()).encodeToString(pkcs8)
                + "\n-----END RSA PRIVATE KEY-----\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
        try {
            RpcTls.server(files.cert, pkcs1);
            fail("accepted a PKCS#1 key");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("openssl pkcs8"));
        }
    }

    @Test
    public void aMissingFileIsRefusedByName() throws Exception {
        TlsFiles files = new TlsFiles(folder.getRoot().toPath());
        try {
            RpcTls.server(folder.getRoot().toPath().resolve("nope.pem"), files.key);
            fail("built TLS from a missing certificate");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("nope.pem"));
        }
        try {
            RpcTls.server(files.cert, folder.getRoot().toPath().resolve("nope.key"));
            fail("built TLS from a missing key");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("nope.key"));
        }
    }

    /** A self-signed certificate and its unencrypted PKCS#8 key as PEM files, over JDK {@code keytool}. */
    private static final class TlsFiles {
        final Path cert;
        final Path key;

        TlsFiles(Path dir) throws Exception {
            Path keystore = dir.resolve("identity.p12");
            cert = dir.resolve("cert.pem");
            key = dir.resolve("key.pem");
            ProcessBuilder keytool = new ProcessBuilder("keytool", "-genkeypair", "-alias", "rpc",
                    "-keyalg", "EC", "-keysize", "256", "-dname", "CN=rpc-test", "-storetype", "PKCS12",
                    "-keystore", keystore.toString(), "-storepass", "changeit", "-noprompt");
            keytool.redirectErrorStream(true);
            Process process = keytool.start();
            if (process.waitFor() != 0) {
                throw new IllegalStateException("keytool failed: "
                        + new String(process.getInputStream().readAllBytes()));
            }
            java.security.KeyStore store = java.security.KeyStore.getInstance("PKCS12");
            try (java.io.InputStream in = Files.newInputStream(keystore)) {
                store.load(in, "changeit".toCharArray());
            }
            byte[] encoded = store.getCertificate("rpc").getEncoded();
            Files.write(cert, pem("CERTIFICATE", encoded));
            Files.write(key, pem("PRIVATE KEY", store.getKey("rpc", "changeit".toCharArray()).getEncoded()));
        }

        private static byte[] pem(String label, byte[] der) {
            String base64 = java.util.Base64.getMimeEncoder(64, "\n".getBytes()).encodeToString(der);
            return ("-----BEGIN " + label + "-----\n" + base64 + "\n-----END " + label + "-----\n")
                    .getBytes(java.nio.charset.StandardCharsets.UTF_8);
        }
    }
}
