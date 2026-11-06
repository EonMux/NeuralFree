package com.eonmux.neuralfree.gguf.remote;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link ObjectStoreUri} parsing: the three schemes, the Azure alias, prefix trimming and the refusals. */
public class ObjectStoreUriTest {

    @Test
    public void parsesTheThreeSchemes() {
        ObjectStoreUri s3 = ObjectStoreUri.parse("s3://my-bucket/models/llama");
        assertEquals(ObjectStoreUri.Scheme.S3, s3.scheme());
        assertEquals("my-bucket", s3.bucket());
        assertEquals("models/llama", s3.prefix());
        assertEquals("s3://my-bucket/models/llama", s3.toString());

        ObjectStoreUri gcs = ObjectStoreUri.parse("gs://gcs-bucket/a/b");
        assertEquals(ObjectStoreUri.Scheme.GCS, gcs.scheme());
        assertEquals("gcs-bucket", gcs.bucket());
        assertEquals("a/b", gcs.prefix());

        ObjectStoreUri azure = ObjectStoreUri.parse("az://container/prefix");
        assertEquals(ObjectStoreUri.Scheme.AZURE, azure.scheme());
        assertEquals("container", azure.bucket());
        assertEquals("prefix", azure.prefix());
    }

    @Test
    public void azureAliasAndRootPrefixesAreNormalized() {
        ObjectStoreUri alias = ObjectStoreUri.parse("AZURE://container/path");
        assertEquals(ObjectStoreUri.Scheme.AZURE, alias.scheme());
        assertEquals("path", alias.prefix());
        assertEquals("az://container/path", alias.toString());

        ObjectStoreUri root = ObjectStoreUri.parse("s3://bucket");
        assertEquals("", root.prefix());
        assertEquals("s3://bucket", root.toString());

        ObjectStoreUri slashes = ObjectStoreUri.parse("s3://bucket//a/b//");
        assertEquals("a/b", slashes.prefix());
    }

    @Test
    public void recognizesObjectStorePaths() {
        assertTrue(ObjectStoreUri.isObjectStorePath("s3://bucket/x"));
        assertTrue(ObjectStoreUri.isObjectStorePath("GS://bucket"));
        assertTrue(ObjectStoreUri.isObjectStorePath("az://container"));
        assertTrue(ObjectStoreUri.isObjectStorePath("azure://container"));
        assertFalse(ObjectStoreUri.isObjectStorePath("/tmp/model"));
        assertFalse(ObjectStoreUri.isObjectStorePath("hf://bucket"));
        assertFalse(ObjectStoreUri.isObjectStorePath(null));
    }

    @Test
    public void refusesMalformedUris() {
        assertThrows(IllegalArgumentException.class, () -> ObjectStoreUri.parse("s3://"));
        assertThrows(IllegalArgumentException.class, () -> ObjectStoreUri.parse("s3:///prefix"));
        assertThrows(IllegalArgumentException.class, () -> ObjectStoreUri.parse("ftp://bucket/key"));
        assertThrows(IllegalArgumentException.class, () -> ObjectStoreUri.parse("s3://bucket/key?x=1"));
        assertThrows(IllegalArgumentException.class, () -> ObjectStoreUri.parse("/local/path"));
        assertThrows(IllegalArgumentException.class, () -> ObjectStoreUri.parse("s3://bu cket/key"));
    }

    @Test
    public void equalityUsesTheCanonicalForm() {
        assertEquals(ObjectStoreUri.parse("az://bucket/x"), ObjectStoreUri.parse("azure://bucket/x"));
        assertEquals(ObjectStoreUri.parse("s3://bucket/x/").hashCode(),
                ObjectStoreUri.parse("s3://bucket/x").hashCode());
    }
}
