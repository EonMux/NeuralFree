package com.eonmux.neuralfree.gguf.rpc;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Objects;
import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLServerSocket;
import javax.net.ssl.SSLServerSocketFactory;
import javax.net.ssl.SSLSocket;
import javax.net.ssl.TrustManagerFactory;

/**
 * The optional TLS of the remote-compute transport ({@link RpcServerSpec} {@code tls://} endpoints): an
 * {@link SSLContext} for the server built from a PEM certificate chain and a PKCS#8 private key, and one for the
 * client built from a trust store, both loaded from files named by properties or environment variables. The HMAC
 * handshake of the protocol runs on top of either transport, so a TLS endpoint still authenticates with its key.
 *
 * <p>Server side: {@value #PROP_CERT} or {@value #ENV_CERT} names the certificate chain and {@value #PROP_KEY} or
 * {@value #ENV_KEY} the private key, both PEM files. The key must be an unencrypted PKCS#8 {@code BEGIN PRIVATE
 * KEY} block (RSA or EC); the OpenSSL PKCS#1 shapes are refused with the command that converts them.</p>
 *
 * <p>Client side: {@value #PROP_TRUSTSTORE} names a PKCS12 or JKS trust store holding the server certificate (or
 * its CA), opened with {@value #PROP_TRUSTSTORE_PASSWORD} or {@value #ENV_TRUSTSTORE_PASSWORD}; without it the
 * client trusts the JVM's default authorities. Client sockets verify the endpoint's host name against the
 * certificate's subject alternative names, so a self-signed server certificate carries the host or address it is
 * reached at.</p>
 */
public final class RpcTls {

    /** Property naming the server's PEM certificate chain file. */
    public static final String PROP_CERT = "neuralfree.rpc.tls.cert";
    /** Property naming the server's PEM PKCS#8 private key file. */
    public static final String PROP_KEY = "neuralfree.rpc.tls.key";
    /** Environment variable naming the server's PEM certificate chain file. */
    public static final String ENV_CERT = "NEURALFREE_RPC_TLS_CERT";
    /** Environment variable naming the server's PEM PKCS#8 private key file. */
    public static final String ENV_KEY = "NEURALFREE_RPC_TLS_KEY";
    /** Property naming the client's trust store file (PKCS12 or JKS). */
    public static final String PROP_TRUSTSTORE = "neuralfree.rpc.tls.truststore";
    /** Property naming the client trust store's password. */
    public static final String PROP_TRUSTSTORE_PASSWORD = "neuralfree.rpc.tls.truststore.password";
    /** Environment variable naming the client trust store's password. */
    public static final String ENV_TRUSTSTORE_PASSWORD = "NEURALFREE_RPC_TLS_TRUSTSTORE_PASSWORD";

    private static final String BEGIN_CERT = "-----BEGIN CERTIFICATE-----";
    private static final String END_CERT = "-----END CERTIFICATE-----";
    private static final String BEGIN_KEY = "-----BEGIN PRIVATE KEY-----";
    private static final String END_KEY = "-----END PRIVATE KEY-----";

    private RpcTls() {
        throw new AssertionError("no instances");
    }

    /**
     * The server side of a TLS listener: its context and the files it was built from. Build with {@link
     * #server(Path, Path)}; {@link #fromEnvironment()} reads the property and environment names.
     */
    public static final class Server {

        private final SSLContext context;
        private final Path cert;
        private final Path key;

        Server(SSLContext context, Path cert, Path key) {
            this.context = Objects.requireNonNull(context, "context");
            this.cert = Objects.requireNonNull(cert, "cert");
            this.key = Objects.requireNonNull(key, "key");
        }

        /** The context the listener's sockets are created from. */
        public SSLContext context() {
            return context;
        }

        /** The certificate chain file. */
        public Path cert() {
            return cert;
        }

        /** The private key file. */
        public Path key() {
            return key;
        }

        @Override
        public String toString() {
            return "RpcTls.Server{" + cert + ", " + key + "}";
        }
    }

    /**
     * The server TLS named by the properties and environment variables, or null when none is configured.
     *
     * @throws IllegalArgumentException naming the setting when exactly one of the cert and key is configured, a
     *                                 file is missing or cannot be parsed
     */
    public static Server fromEnvironment() {
        String cert = first(System.getProperty(PROP_CERT), System.getenv(ENV_CERT));
        String key = first(System.getProperty(PROP_KEY), System.getenv(ENV_KEY));
        if (cert == null && key == null) {
            return null;
        }
        if (cert == null || key == null) {
            throw new IllegalArgumentException("rpc-server TLS needs both " + PROP_CERT + " (or " + ENV_CERT
                    + ") and " + PROP_KEY + " (or " + ENV_KEY + "); only "
                    + (cert == null ? "the key" : "the certificate") + " is configured");
        }
        return server(cert, key);
    }

    /** {@link #server(Path, Path)} from path text. */
    public static Server server(String certFile, String keyFile) {
        return server(Paths.get(certFile), Paths.get(keyFile));
    }

    /**
     * Builds the server TLS from a PEM certificate chain and an unencrypted PKCS#8 private key.
     *
     * @throws IllegalArgumentException naming the file when either is missing or cannot be parsed
     */
    public static Server server(Path certFile, Path keyFile) {
        Objects.requireNonNull(certFile, "certFile");
        Objects.requireNonNull(keyFile, "keyFile");
        List<Certificate> chain = readCertificateChain(certFile);
        PrivateKey privateKey = readPrivateKey(keyFile);
        try {
            KeyStore store = KeyStore.getInstance("PKCS12");
            store.load(null, null);
            store.setKeyEntry("rpc-server", privateKey, new char[0], chain.toArray(new Certificate[0]));
            KeyManagerFactory kmf = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
            kmf.init(store, new char[0]);
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(kmf.getKeyManagers(), null, null);
            return new Server(context, certFile, keyFile);
        } catch (GeneralSecurityException | IOException e) {
            throw new IllegalArgumentException("rpc-server TLS could not be built from " + certFile + " and "
                    + keyFile + ": " + e.getMessage(), e);
        }
    }

    /**
     * The client SSLContext: from {@value #PROP_TRUSTSTORE} (opened with {@value #PROP_TRUSTSTORE_PASSWORD} or
     * {@value #ENV_TRUSTSTORE_PASSWORD}) when the property is set, else the JVM's default authorities.
     *
     * @throws IllegalArgumentException naming the property when the trust store is missing or cannot be opened
     */
    public static SSLContext clientContext() {
        String truststore = System.getProperty(PROP_TRUSTSTORE);
        if (truststore == null || truststore.trim().isEmpty()) {
            try {
                return SSLContext.getDefault();
            } catch (GeneralSecurityException e) {
                throw new IllegalStateException("the JVM's default TLS context is unavailable: " + e.getMessage(), e);
            }
        }
        Path file = Paths.get(truststore.trim());
        String password = first(System.getProperty(PROP_TRUSTSTORE_PASSWORD), System.getenv(ENV_TRUSTSTORE_PASSWORD));
        char[] pass = password == null ? null : password.toCharArray();
        try (InputStream in = Files.newInputStream(file)) {
            KeyStore store = KeyStore.getInstance("pkcs12");
            try {
                store.load(in, pass);
            } catch (IOException e) {
                // a JKS store refuses the PKCS12 stream read before any password check
                store = KeyStore.getInstance("jks");
                try (InputStream again = Files.newInputStream(file)) {
                    store.load(again, pass);
                }
            }
            TrustManagerFactory tmf = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
            tmf.init(store);
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, tmf.getTrustManagers(), null);
            return context;
        } catch (GeneralSecurityException | IOException e) {
            throw new IllegalArgumentException(PROP_TRUSTSTORE + " '" + truststore + "' could not be opened:"
                    + " " + e.getMessage(), e);
        }
    }

    /**
     * A client TLS socket to {@code host}:{@code port} over {@code context}, with host-name verification against
     * the server certificate's subject alternative names.
     *
     * @throws IOException when the TLS handshake or the verification fails
     */
    public static SSLSocket socket(SSLContext context, String host, int port) throws IOException {
        SSLSocket socket = (SSLSocket) context.getSocketFactory().createSocket(host, port);
        javax.net.ssl.SSLParameters parameters = socket.getSSLParameters();
        parameters.setEndpointIdentificationAlgorithm("HTTPS");
        socket.setSSLParameters(parameters);
        return socket;
    }

    /** An unbound TLS listening socket over {@code context}; the caller binds it. */
    public static SSLServerSocket serverSocket(SSLContext context) throws IOException {
        SSLServerSocketFactory factory = context.getServerSocketFactory();
        return (SSLServerSocket) factory.createServerSocket();
    }

    private static String first(String a, String b) {
        if (a != null && !a.trim().isEmpty()) {
            return a.trim();
        }
        return b == null || b.trim().isEmpty() ? null : b.trim();
    }

    /** Every certificate of the PEM chain {@code file}, in file order. */
    private static List<Certificate> readCertificateChain(Path file) {
        String pem;
        try {
            pem = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalArgumentException("rpc-server TLS certificate '" + file + "' could not be read:"
                    + " " + e.getMessage(), e);
        }
        List<Certificate> out = new ArrayList<>();
        int from = 0;
        while (true) {
            int begin = pem.indexOf(BEGIN_CERT, from);
            if (begin < 0) {
                break;
            }
            int end = pem.indexOf(END_CERT, begin);
            if (end < 0) {
                throw new IllegalArgumentException("rpc-server TLS certificate '" + file
                        + "' has an unterminated " + BEGIN_CERT + " block");
            }
            byte[] der = Base64.getMimeDecoder().decode(pem.substring(begin + BEGIN_CERT.length(), end));
            try {
                CertificateFactory factory = CertificateFactory.getInstance("X.509");
                out.addAll(factory.generateCertificates(new java.io.ByteArrayInputStream(der)));
            } catch (GeneralSecurityException e) {
                throw new IllegalArgumentException("rpc-server TLS certificate '" + file
                        + "' holds no X.509 certificate: " + e.getMessage(), e);
            }
            from = end + END_CERT.length();
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException("rpc-server TLS certificate '" + file + "' holds no " + BEGIN_CERT
                    + " block");
        }
        return out;
    }

    /**
     * The unencrypted PKCS#8 private key of the PEM {@code file}, RSA or EC.
     *
     * @throws IllegalArgumentException naming the file and the conversion command for a PKCS#1 or SEC1 key
     */
    private static PrivateKey readPrivateKey(Path file) {
        String pem;
        try {
            pem = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalArgumentException("rpc-server TLS key '" + file + "' could not be read: "
                    + e.getMessage(), e);
        }
        int begin = pem.indexOf(BEGIN_KEY);
        if (begin < 0) {
            if (pem.contains("-----BEGIN RSA PRIVATE KEY-----") || pem.contains("-----BEGIN EC PRIVATE KEY-----")) {
                throw new IllegalArgumentException("rpc-server TLS key '" + file + "' is a PKCS#1 or SEC1 key;"
                        + " convert it with: openssl pkcs8 -topk8 -nocrypt -in <key> -out <key>.pkcs8");
            }
            throw new IllegalArgumentException("rpc-server TLS key '" + file + "' holds no " + BEGIN_KEY
                    + " block; encrypted keys are not accepted, decrypt it first with: openssl pkcs8 -topk8"
                    + " -nocrypt -in <key> -out <key>.pkcs8");
        }
        int end = pem.indexOf(END_KEY, begin);
        if (end < 0) {
            throw new IllegalArgumentException("rpc-server TLS key '" + file + "' has an unterminated " + BEGIN_KEY
                    + " block");
        }
        byte[] der = Base64.getMimeDecoder().decode(pem.substring(begin + BEGIN_KEY.length(), end));
        String[] algorithms = {"RSA", "EC"};
        for (String algorithm : algorithms) {
            try {
                KeyFactory factory = KeyFactory.getInstance(algorithm);
                return factory.generatePrivate(new PKCS8EncodedKeySpec(der));
            } catch (GeneralSecurityException ignored) {
                // the key is of the other algorithm
            }
        }
        throw new IllegalArgumentException("rpc-server TLS key '" + file + "' is neither an RSA nor an EC key;"
                + " the transport serves those");
    }
}
