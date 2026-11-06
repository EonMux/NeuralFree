package com.eonmux.neuralfree.gguf.remote;

import org.junit.After;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

/** {@link ObjectStores}: each scheme is built from its system properties, the environment otherwise. */
public class ObjectStoresTest {

    private FakeObjectStoreServer server;

    @After
    public void stop() {
        System.clearProperty(ObjectStores.PROP_GCS_ENDPOINT);
        System.clearProperty(ObjectStores.PROP_AZURE_ENDPOINT);
        System.clearProperty(ObjectStores.PROP_AZURE_ACCOUNT);
        System.clearProperty(ObjectStores.PROP_AZURE_SAS);
        if (server != null) {
            server.close();
        }
    }

    @Test
    public void buildsAGcsStoreFromItsProperty() throws IOException {
        server = new FakeObjectStoreServer(FakeObjectStoreServer.Kind.GCS);
        server.put("models/one.bin", new byte[]{1});
        System.setProperty(ObjectStores.PROP_GCS_ENDPOINT, server.base());
        try (ObjectStore store = ObjectStores.open(ObjectStoreUri.parse("gs://bucket/models"))) {
            assertEquals("GCS bucket 'bucket' at " + server.base(), store.description());
            assertEquals(1, store.list("models/").size());
        }
    }

    @Test
    public void buildsAnAzureStoreFromItsProperties() throws IOException {
        server = new FakeObjectStoreServer(FakeObjectStoreServer.Kind.AZURE);
        server.put("models/one.bin", new byte[]{1});
        System.setProperty(ObjectStores.PROP_AZURE_ENDPOINT, server.base());
        System.setProperty(ObjectStores.PROP_AZURE_ACCOUNT, "account");
        System.setProperty(ObjectStores.PROP_AZURE_SAS, "sig=abc");
        try (ObjectStore store = ObjectStores.open(ObjectStoreUri.parse("az://container/models"))) {
            assertEquals("Azure container 'container' at " + server.base(), store.description());
            assertEquals(1, store.list("models/").size());
        }
    }
}
