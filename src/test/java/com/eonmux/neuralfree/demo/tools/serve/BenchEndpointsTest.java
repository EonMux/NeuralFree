package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The server discovery the benchmarks run before a workload: the first model card, the tokenizer's
 * vocabulary and special ids, and the prompt lengths through {@code POST /tokenize}.
 */
public class BenchEndpointsTest {

    @Test
    public void theFirstModelCarriesItsIdAndRoot() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            server.modelId = "served-name";
            server.modelRoot = "/models/base.gguf";
            BenchEndpoints.ServedModel model = BenchEndpoints.firstModel(server.baseUrl(),
                    new LinkedHashMap<>(), false);
            assertEquals("served-name", model.id);
            assertEquals("/models/base.gguf", model.root);
        }
    }

    @Test
    public void aServerWithoutAModelIsNamed() throws Exception {
        com.sun.net.httpserver.HttpServer empty = com.sun.net.httpserver.HttpServer.create(
                new java.net.InetSocketAddress("127.0.0.1", 0), 0);
            empty.createContext("/v1/models", exchange -> {
                byte[] body = "{\"data\":[]}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, body.length);
                exchange.getResponseBody().write(body);
                exchange.close();
            });
            empty.start();
            try {
                BenchEndpoints.firstModel("http://127.0.0.1:" + empty.getAddress().getPort(),
                        new LinkedHashMap<>(), false);
                throw new AssertionError("an empty model list must be refused");
            } catch (java.io.IOException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("no models"));
            } finally {
                empty.stop(0);
            }
    }

    @Test
    public void theTokenizerInfoCarriesTheVocabularyAndTheSpecialIds() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            server.vocabSize = 1234;
            BenchEndpoints.TokenizerInfo info = BenchEndpoints.tokenizerInfo(server.baseUrl(),
                    new LinkedHashMap<>(), false);
            assertEquals(1234, info.vocabSize);
            assertArrayEquals(new int[]{1, 2, 0, 3}, info.skipIds);
        }
    }

    @Test
    public void theTokenizerCountsThroughTheServer() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            BenchTokenizer tokenizer = BenchEndpoints.tokenizer(server.baseUrl(), new LinkedHashMap<>(), false);
            assertEquals(5, tokenizer.tokenize("hello").length);
            assertEquals(0, tokenizer.tokenize("").length);
        }
    }

    @Test
    public void anUnreachableServerNamesItsUrl() {
        try {
            BenchEndpoints.firstModel("http://127.0.0.1:1", new LinkedHashMap<>(), false);
            throw new AssertionError("an unreachable server must be refused");
        } catch (java.io.IOException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("http://127.0.0.1:1/v1/models"));
        }
    }

    @Test
    public void theTokenizerOfTheServerCountsThePrompts() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            int[] ids = BenchEndpoints.tokenizer(server.baseUrl(), new LinkedHashMap<>(), false).tokenize("abc");
            assertEquals(3, ids.length);
            assertEquals(100, ids[0]);
        }
    }

    @Test
    public void aModelCardWithoutAnIdIsRefused() throws Exception {
        com.sun.net.httpserver.HttpServer server = com.sun.net.httpserver.HttpServer.create(
                new java.net.InetSocketAddress("127.0.0.1", 0), 0);
            server.createContext("/v1/models", exchange -> {
                byte[] body = "{\"data\":[{\"root\":\"/x\"}]}".getBytes(
                        java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, body.length);
                exchange.getResponseBody().write(body);
                exchange.close();
            });
            server.start();
            try {
                BenchEndpoints.firstModel("http://127.0.0.1:" + server.getAddress().getPort(),
                        new LinkedHashMap<>(), false);
                throw new AssertionError("a card without an id must be refused");
            } catch (java.io.IOException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("'id'"));
            } finally {
                server.stop(0);
            }
    }

    @Test
    public void everyTokenOfTheTokenizerIsAnInteger() {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            List<String> texts = java.util.Arrays.asList("", "a", "abcdef");
            BenchTokenizer tokenizer = BenchEndpoints.tokenizer(server.baseUrl(), new LinkedHashMap<>(), false);
            for (String text : texts) {
                assertEquals(text.length(), tokenizer.tokenize(text).length);
            }
        } catch (java.io.IOException e) {
            throw new AssertionError(e);
        }
    }
}
