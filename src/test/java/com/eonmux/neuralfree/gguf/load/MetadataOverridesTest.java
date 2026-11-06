package com.eonmux.neuralfree.gguf.load;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import org.junit.Test;

import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** {@link MetadataOverrides}: the {@code --override-kv} grammar, its types and values, the list form, and how
 *  {@code ModelSourceOptions} carries it. */
public class MetadataOverridesTest {

    // ---------------------------------------------------------------- parse: every type and value form

    @Test
    public void parsesEveryType() {
        MetadataOverrides.Override a = MetadataOverrides.parse("a=str:x y:z", "test");
        assertEquals("a", a.key());
        assertEquals(GGUFValueType.STRING, a.type());
        assertEquals("x y:z", a.value());

        MetadataOverrides.Override b = MetadataOverrides.parse("b=bool:true", "test");
        assertEquals(GGUFValueType.BOOL, b.type());
        assertEquals(Boolean.TRUE, b.value());
        assertEquals(Boolean.FALSE, MetadataOverrides.parse("b=bool:False", "test").value());

        MetadataOverrides.Override c = MetadataOverrides.parse("c=int:-5", "test");
        assertEquals(GGUFValueType.INT32, c.type());
        assertEquals(-5, c.value());

        MetadataOverrides.Override d = MetadataOverrides.parse("d=float:1.75", "test");
        assertEquals(GGUFValueType.FLOAT32, d.type());
        assertEquals(1.75f, d.value());

        // a str value keeps ':' and '=' and is not trimmed
        MetadataOverrides.Override e = MetadataOverrides.parse("e=str:key=str:key", "test");
        assertEquals("key=str:key", e.value());
        MetadataOverrides.Override f = MetadataOverrides.parse(" f=str: kept ", "test");
        assertEquals("f", f.key());
        assertEquals(" kept", f.value());

        assertEquals(MetadataOverrides.parse("a=str:x", "test"), MetadataOverrides.parse("a=str:x", "test"));
        assertEquals(MetadataOverrides.parse("a=str:x", "test").hashCode(),
                MetadataOverrides.parse("a=str:x", "test").hashCode());
    }

    /** The parsed form of one entry round-trips. */
    @Test
    public void toStringIsTheParsedForm() {
        MetadataOverrides.Override f = MetadataOverrides.parse("d=float:1.75", "test");
        assertEquals("d=float:1.75", f.toString());
        assertEquals(f, MetadataOverrides.parse(f.toString(), "test"));
        MetadataOverrides.Override s = MetadataOverrides.parse("a=str:hello world", "test");
        assertEquals("a=str:hello world", s.toString());
    }

    // ---------------------------------------------------------------- parse: rejected entries

    @Test
    public void rejectsUntypedValues() {
        assertRejected("just-a-key", "expected type:value after the key");
        assertRejected("a=str", "expected type:value after the key");
        assertRejected("a=strt:x", "the type 'strt' is unknown");
        assertRejected("a=int:x", "is not an integer");
        assertRejected("a=int:2147483648", "does not fit INT32");
        assertRejected("a=int:-2147483649", "does not fit INT32");
        assertRejected("a=float:-inf", "is not a number");
        assertRejected("a=float:2e80", "is not finite");
        assertRejected("a=bool:yes", "is not true or false");
        assertRejected("=str:x", "the key is empty");
    }

    private static void assertRejected(String entry, String detail) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.parse(entry, "test"));
        assertTrue("entry name missing: '" + e.getMessage() + "'",
                e.getMessage().contains("test") && e.getMessage().contains("'" + entry + "'"));
        assertTrue("detail missing: '" + e.getMessage() + "'", e.getMessage().contains(detail));
    }

    // ---------------------------------------------------------------- the ; list form

    @Test
    public void parsesListForm() {
        List<MetadataOverrides.Override> list = MetadataOverrides.parseList(
                "a=str:one; b=bool:true ; c=int:7", "test");
        assertEquals(3, list.size());
        assertEquals(MetadataOverrides.parse("a=str:one", "test"), list.get(0));
        assertEquals(MetadataOverrides.parse("b=bool:true", "test"), list.get(1));
        assertEquals(MetadataOverrides.parse("c=int:7", "test"), list.get(2));
    }

    @Test
    public void rejectsBrokenList() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.parseList("a=str:x;;b=str:y", "test"));
        assertTrue(e.getMessage().contains("empty entry"));
        e = assertThrows(IllegalArgumentException.class, () -> MetadataOverrides.parseList(" ", "test"));
        assertTrue(e.getMessage().contains("test") && e.getMessage().contains("is empty"));
        e = assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.parseList("a=str:x; b=strt:y", "test"));
        assertTrue(e.getMessage().contains("b=strt:y"));
    }

    @Test
    public void roundtripsThroughStringForm() {
        List<MetadataOverrides.Override> list = MetadataOverrides.parseList("a=str:x; b=int:3", "test");
        assertEquals(list, MetadataOverrides.parseList(MetadataOverrides.toString(list), "test"));
        assertEquals("a=str:x;b=int:3", MetadataOverrides.toString(list));
    }

    // ---------------------------------------------------------------- the scalar and array factories

    @Test
    public void scalarAndArrayOverridesCarryTheTypesTheGrammarCannot() {
        MetadataOverrides.Override count = MetadataOverrides.Override.scalar("c", GGUFValueType.UINT32, 2);
        assertEquals(GGUFValueType.UINT32, count.type());
        assertEquals("c=uint32:2", count.toString());
        assertEquals("w=int64:5", MetadataOverrides.Override.scalar("w", GGUFValueType.INT64, 5L).toString());
        assertEquals("d=float64:1.0E300",
                MetadataOverrides.Override.scalar("d", GGUFValueType.FLOAT64, 1.0e300).toString());

        MetadataOverrides.Override tags = MetadataOverrides.Override.array("t", GGUFValueType.STRING,
                List.of("a", "b"));
        assertEquals(GGUFValueType.ARRAY, tags.type());
        assertEquals(GGUFValueType.STRING, tags.elementType());
        assertEquals(List.of("a", "b"), tags.value());
        assertEquals("t=array(str):[a, b]", tags.toString());
        assertEquals(null, count.elementType());
    }

    @Test
    public void factoriesRefuseValuesOutsideTheirRepresentation() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.Override.scalar("k", GGUFValueType.INT32, 2L));
        assertTrue(e.getMessage(), e.getMessage().contains("'k'") && e.getMessage().contains("Integer"));
        assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.Override.scalar("k", GGUFValueType.ARRAY, List.of(1)));
        assertEquals("k=uint8:1",
                MetadataOverrides.Override.scalar("k", GGUFValueType.UINT8, 1).toString());
        assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.Override.scalar("k", GGUFValueType.UINT8, 1L));
        assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.Override.array("k", GGUFValueType.ARRAY, List.of(1)));
        assertThrows(IllegalArgumentException.class,
                () -> MetadataOverrides.Override.array("k", GGUFValueType.FLOAT32, List.of(1.0)));
        assertThrows(NullPointerException.class,
                () -> MetadataOverrides.Override.array("k", GGUFValueType.STRING, java.util.Arrays.asList("a", null)));
    }

    // ---------------------------------------------------------------- ModelSourceOptions carries it

    @Test
    public void flowsThroughModelSourceOptions() {
        ModelSourceOptions o = ModelSourceOptions.builder()
                .metadataOverride("a=str:x")
                .metadataOverride("b=int:2")
                .build();
        assertEquals(2, o.metadataOverrides().size());
        assertEquals(MetadataOverrides.parse("a=str:x", PROP_OVERRIDE_KV), o.metadataOverrides().get(0));
        assertEquals(MetadataOverrides.parse("b=int:2", PROP_OVERRIDE_KV), o.metadataOverrides().get(1));

        // repeated metadataOverride calls accumulate while metadataOverrideList replaces
        ModelSourceOptions replaced = ModelSourceOptions.builder().metadataOverride("a=str:x").build()
                .toBuilder().metadataOverrideList("c=str:z").build();
        assertEquals(1, replaced.metadataOverrides().size());
        assertEquals("c", replaced.metadataOverrides().get(0).key());

        assertEquals(ModelSourceOptions.builder().build().metadataOverrides().size(), 0);
    }

    @Test
    public void flowsThroughProperties() {
        Properties p = new Properties();
        p.setProperty(ModelSourceOptions.PROP_OVERRIDE_KV, "a=str:x; b=bool:true");
        ModelSourceOptions from = ModelSourceOptions.fromProperties(p);
        assertEquals(2, from.metadataOverrides().size());
        assertEquals("a", from.metadataOverrides().get(0).key());
        assertEquals(Boolean.TRUE, from.metadataOverrides().get(1).value());

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ModelSourceOptions.builder().metadataOverride("nope").build());
        assertTrue(e.getMessage().contains(ModelSourceOptions.PROP_OVERRIDE_KV));
        e = assertThrows(IllegalArgumentException.class, () -> ModelSourceOptions.fromProperties(p)
                .toBuilder().metadataOverrideList("a=int:99999999999").build());
        assertTrue(e.getMessage().contains(ModelSourceOptions.PROP_OVERRIDE_KV));
    }

    /** The options name every entry in errors with the flag's property. */
    @Test
    public void blankListPropertyClearsOverrides() {
        Properties p = new Properties();
        p.setProperty(ModelSourceOptions.PROP_OVERRIDE_KV, " ");
        assertEquals(0, ModelSourceOptions.fromProperties(p).metadataOverrides().size());
        ModelSourceOptions cleared = ModelSourceOptions.builder().metadataOverride("a=str:x").build()
                .toBuilder().metadataOverrideList(null).build();
        assertEquals(0, cleared.metadataOverrides().size());
    }

    private static final String PROP_OVERRIDE_KV = ModelSourceOptions.PROP_OVERRIDE_KV;
}
