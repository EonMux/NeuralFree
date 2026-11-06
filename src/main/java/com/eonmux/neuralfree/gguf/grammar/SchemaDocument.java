package com.eonmux.neuralfree.gguf.grammar;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The {@code $ref} resolution of one JSON schema document: the schema resources its {@code $id}s declare (the root
 * included), the anchors ({@code $anchor}, {@code $dynamicAnchor} and the draft 6/7 {@code "$id": "#name"} form)
 * and the base URI in effect at every subschema. A reference resolves against the base URI of the subschema holding
 * it (RFC 3986 section 5) to a resource of this document, then its fragment selects a JSON pointer (RFC 6901, with
 * percent-decoding) or an anchor inside that resource.
 *
 * <p>Schemas are never fetched: a reference to a resource the document does not embed is refused by name, as are
 * dangling pointers and anchors. Every resolved target carries its canonical key, {@code "#"} plus its JSON pointer
 * from the document root, so references that reach one subschema along different spellings share one rule.</p>
 */
final class SchemaDocument {

    /** The base URI of a document whose root declares no {@code $id}. */
    static final URI DEFAULT_BASE = URI.create("nf-schema:/root.json");

    /** Keywords whose values are JSON data, never subschemas. */
    private static final Set<String> VALUE_KEYWORDS =
            new HashSet<>(Arrays.asList("enum", "const", "default", "examples"));

    /** Keywords whose values map names (not keywords) to subschemas. */
    private static final Set<String> NAMED_SCHEMAS = new HashSet<>(Arrays.asList(
            "properties", "patternProperties", "$defs", "definitions", "dependentSchemas"));

    /** One resolved reference: the target value and its canonical key. */
    static final class Target {
        final Object node;
        final String key;

        Target(Object node, String key) {
            this.node = node;
            this.key = key;
        }
    }

    private final Map<Object, URI> bases = new IdentityHashMap<>();
    private final Map<Object, String> pointers = new IdentityHashMap<>();
    private final Map<String, Object> resources = new HashMap<>();
    private final Map<String, Object> anchors = new HashMap<>();

    /**
     * Indexes {@code root}.
     *
     * @throws JsonSchemaToGbnf.SchemaErrorException when an {@code $id} is not a URI or names two subschemas
     */
    SchemaDocument(Object root) {
        resources.put(DEFAULT_BASE.toString(), root);
        index(root, "", DEFAULT_BASE, false);
    }

    /** The base URI in effect inside {@code node}, null when the node is not an indexed subschema. */
    URI baseOf(Object node) {
        return bases.get(node);
    }

    /**
     * Resolves {@code ref} written in a subschema whose base URI is {@code base}.
     *
     * @param path the location of the reference, named in failures
     * @throws JsonSchemaToGbnf.SchemaErrorException when the reference is malformed, leaves the document, or
     *                                              names a pointer or anchor the document lacks
     */
    Target resolve(String ref, URI base, String path) {
        String absolute;
        if (ref.startsWith("#")) {
            absolute = withoutFragment(base.toString()) + ref;
        } else {
            try {
                absolute = base.resolve(new URI(ref)).toString();
            } catch (URISyntaxException | IllegalArgumentException e) {
                throw new JsonSchemaToGbnf.SchemaErrorException(path, "$ref " + ref + " is not a valid URI reference");
            }
        }
        int hash = absolute.indexOf('#');
        String resource = hash < 0 ? absolute : absolute.substring(0, hash);
        String fragment = hash < 0 ? "" : percentDecode(absolute.substring(hash + 1));
        Object node = resources.get(resource);
        if (node == null) {
            throw new JsonSchemaToGbnf.SchemaErrorException(path, "unsupported $ref " + ref
                    + ", only references into the same document are supported (" + resource
                    + " is not a schema resource of this document; remote schemas are not fetched)");
        }
        if (fragment.isEmpty()) {
            return new Target(node, "#" + pointers.get(node));
        }
        if (fragment.startsWith("/")) {
            return navigate(node, fragment.substring(1), ref, path);
        }
        Object anchored = anchors.get(resource + "#" + fragment);
        if (anchored == null) {
            throw new JsonSchemaToGbnf.SchemaErrorException(path, "cannot resolve $ref " + ref + ", anchor "
                    + fragment + " not found");
        }
        return new Target(anchored, "#" + pointers.get(anchored));
    }

    /** RFC 6901 navigation from {@code node} (llama.cpp {@code resolve_ref}, with {@code ~0}/{@code ~1}). */
    private Target navigate(Object node, String pointer, String ref, String path) {
        StringBuilder key = new StringBuilder("#").append(pointers.get(node));
        Object target = node;
        int start = 0;
        while (start <= pointer.length()) {
            int slash = pointer.indexOf('/', start);
            int end = slash < 0 ? pointer.length() : slash;
            String raw = pointer.substring(start, end);
            String token = raw.replace("~1", "/").replace("~0", "~");
            if (target instanceof Map && ((Map<?, ?>) target).containsKey(token)) {
                target = ((Map<?, ?>) target).get(token);
            } else if (target instanceof List) {
                List<?> list = (List<?>) target;
                int index = arrayIndex(token, list.size());
                if (index < 0) {
                    throw new JsonSchemaToGbnf.SchemaErrorException(path, "cannot resolve $ref " + ref + ", "
                            + raw + " is out of range");
                }
                target = list.get(index);
            } else {
                throw new JsonSchemaToGbnf.SchemaErrorException(path, "cannot resolve $ref " + ref + ", " + raw
                        + " not found");
            }
            key.append('/').append(escape(token));
            start = end + 1;
        }
        return new Target(target, key.toString());
    }

    /** The array index a pointer token names, -1 when it names none of {@code size} elements. */
    private static int arrayIndex(String token, int size) {
        if (token.isEmpty() || token.length() > 9) {
            return -1;
        }
        for (int i = 0; i < token.length(); i++) {
            if (token.charAt(i) < '0' || token.charAt(i) > '9') {
                return -1;
            }
        }
        int index = Integer.parseInt(token);
        return index < size ? index : -1;
    }

    private void index(Object node, String pointer, URI base, boolean namedSchemas) {
        if (node instanceof List) {
            List<?> list = (List<?>) node;
            for (int i = 0; i < list.size(); i++) {
                index(list.get(i), pointer + "/" + i, base, false);
            }
            return;
        }
        if (!(node instanceof Map)) {
            return;
        }
        Map<?, ?> map = (Map<?, ?>) node;
        if (namedSchemas) {
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                index(entry.getValue(), pointer + "/" + escape(String.valueOf(entry.getKey())), base, false);
            }
            return;
        }
        URI own = map.get("$id") instanceof String ? declare((String) map.get("$id"), base, node, pointer) : base;
        for (String anchorKey : new String[] {"$anchor", "$dynamicAnchor"}) {
            if (map.get(anchorKey) instanceof String) {
                anchors.put(withoutFragment(own.toString()) + "#" + map.get(anchorKey), node);
            }
        }
        bases.put(node, own);
        pointers.put(node, pointer);
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            String key = String.valueOf(entry.getKey());
            if (!VALUE_KEYWORDS.contains(key)) {
                index(entry.getValue(), pointer + "/" + escape(key), own, NAMED_SCHEMAS.contains(key));
            }
        }
    }

    /** Registers the resource or anchor an {@code $id} declares; returns the base URI inside the subschema. */
    private URI declare(String id, URI base, Object node, String pointer) {
        URI resolved;
        try {
            resolved = id.startsWith("#") ? URI.create(withoutFragment(base.toString()) + id)
                    : base.resolve(new URI(id));
        } catch (URISyntaxException | IllegalArgumentException e) {
            throw new JsonSchemaToGbnf.SchemaErrorException("#" + pointer, "$id " + id + " is not a valid URI");
        }
        String text = resolved.toString();
        int hash = text.indexOf('#');
        String resource = hash < 0 ? text : text.substring(0, hash);
        String fragment = hash < 0 ? "" : percentDecode(text.substring(hash + 1));
        if (!fragment.isEmpty()) {
            anchors.put(resource + "#" + fragment, node);
        }
        if (id.startsWith("#")) {
            return base; // a draft 6/7 plain-name identifier names an anchor, not a resource
        }
        Object previous = resources.put(resource, node);
        if (previous != null && previous != node) {
            throw new JsonSchemaToGbnf.SchemaErrorException("#" + pointer, "$id " + id
                    + " names a second schema resource " + resource);
        }
        return URI.create(resource);
    }

    private static String withoutFragment(String uri) {
        int hash = uri.indexOf('#');
        return hash < 0 ? uri : uri.substring(0, hash);
    }

    /** An RFC 6901 reference token. */
    private static String escape(String token) {
        return token.replace("~", "~0").replace("/", "~1");
    }

    /** Decodes the {@code %XX} escapes of a URI fragment as UTF-8; other characters are kept. */
    private static String percentDecode(String text) {
        if (text.indexOf('%') < 0) {
            return text;
        }
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        int i = 0;
        while (i < text.length()) {
            if (text.charAt(i) == '%' && i + 2 < text.length() && hex(text.charAt(i + 1)) >= 0
                    && hex(text.charAt(i + 2)) >= 0) {
                bytes.write(hex(text.charAt(i + 1)) * 16 + hex(text.charAt(i + 2)));
                i += 3;
            } else {
                int codePoint = text.codePointAt(i);
                byte[] encoded = new String(Character.toChars(codePoint)).getBytes(StandardCharsets.UTF_8);
                bytes.write(encoded, 0, encoded.length);
                i += Character.charCount(codePoint);
            }
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    private static int hex(char c) {
        return Character.digit(c, 16);
    }
}
