package com.eonmux.neuralfree.gguf.hf.torch;

import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link RestrictedUnpickler}: the allow-list accepts what torch pickles build and refuses what it must — the
 * malicious globals, the out-of-list opcodes, the legacy and out-of-range protocols, the oversized inputs — under
 * its limits, and the persistent ids go to the loader.
 */
public class RestrictedUnpicklerTest {

    // ------------------------------------------------------------------ builder

    /** A protocol-2 pickle writer: enough of pickle.py for torch's {@code data.pkl} shapes. */
    static final class Pb {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();

        Pb proto(int version) {
            out.write(0x80);
            out.write(version);
            return this;
        }

        Pb global(String module, String name) {
            out.write(0x63);
            line(module);
            line(name);
            return this;
        }

        Pb str(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            out.write(0x58);
            int32(b.length);
            write(b);
            return this;
        }

        Pb raw(int opcode) {
            out.write(opcode);
            return this;
        }

        Pb int32(int v) {
            out.write(v & 0xFF);
            out.write((v >>> 8) & 0xFF);
            out.write((v >>> 16) & 0xFF);
            out.write((v >>> 24) & 0xFF);
            return this;
        }

        Pb b1(long v) {
            out.write(0x4b);
            out.write((int) (v & 0xFF));
            return this;
        }

        Pb binput(int slot) {
            out.write(0x71);
            out.write(slot);
            return this;
        }

        Pb binget(int slot) {
            out.write(0x68);
            out.write(slot);
            return this;
        }

        Pb mark() {
            out.write(0x28);
            return this;
        }

        Pb tuple() {
            out.write(0x74);
            return this;
        }

        Pb storageRef(String typeName, String key, String location) {
            mark();
            str("storage");
            global("torch", typeName);
            str(key);
            str(location);
            tuple();
            raw(0x51);
            return this;
        }

        /** The six values of one tensor view, inside the caller's marked argument tuple. */
        Pb viewBytes(String typeName, String key, String at, long offset, int outer, int inner) {
            storageRef(typeName, key, at);
            b1(offset);
            mark();
            b1(outer);
            b1(inner);
            tuple();
            mark();
            b1(inner);
            b1(1);
            tuple();
            raw(0x88);
            newOd();
            return this;
        }

        Pb tensorValue(String typeName, String key, long offset, int outer, int inner) {
            global("torch._utils", "_rebuild_tensor_v2");
            mark();
            viewBytes(typeName, key, "cpu", offset, outer, inner);
            tuple();
            reduce();
            return this;
        }

        /** A tensor view with explicit shape and strides (the non-contiguous and shared-storage shapes). */
        Pb tensorOf(String typeName, String key, long offset, long[] size, long[] stride) {
            global("torch._utils", "_rebuild_tensor_v2");
            mark();
            storageRef(typeName, key, "cpu");
            b1(offset);
            ints(size);
            ints(stride);
            raw(0x88);
            newOd();
            tuple();
            reduce();
            return this;
        }

        private Pb ints(long[] vs) {
            mark();
            for (long v : vs) {
                b1(v);
            }
            tuple();
            return this;
        }

        Pb newOd() {
            return global("collections", "OrderedDict").emptyTuple().reduce();
        }

        Pb emptyTuple() {
            out.write(0x29);
            return this;
        }

        Pb emptyList() {
            out.write(0x5d);
            return this;
        }

        Pb appends() {
            out.write(0x65);
            return this;
        }

        Pb tuple1() {
            out.write(0x85);
            return this;
        }

        Pb emptyDict() {
            out.write(0x7d);
            return this;
        }

        Pb reduce() {
            out.write(0x52);
            return this;
        }

        Pb stop() {
            out.write(0x2e);
            return this;
        }

        private Pb line(String s) {
            write(s.getBytes(StandardCharsets.UTF_8));
            out.write('\n');
            return this;
        }

        private Pb write(byte[] bytes) {
            out.write(bytes, 0, bytes.length);
            return this;
        }

        Pb rawBytes(byte[] bytes) {
            write(bytes);
            return this;
        }

        byte[] build() {
            return out.toByteArray();
        }
    }

    private static final RestrictedUnpickler.PersistentLoader LOCAL_STORAGE =
            id -> "storage:" + id.get(2);

    private static Object unpickle(byte[] pickle) throws IOException {
        return RestrictedUnpickler.unpickle(pickle, LOCAL_STORAGE);
    }

    // ------------------------------------------------------------------ accepted pickles

    @Test
    public void stateDictParsesIntoTensorViewsAndPlainValues() throws IOException {
        Pb b = new Pb().proto(2).emptyDict().binput(1);
        b.str("model.w");
        b.tensorValue("FloatStorage", "12345", 2, 3, 4);
        b.raw(0x73); // SETITEM
        b.str("meta");
        b.str("value");
        b.raw(0x73);
        b.stop();
        Map<?, ?> d = (Map<?, ?>) unpickle(b.build());
        assertEquals(2, d.size());
        RestrictedUnpickler.TensorView v = (RestrictedUnpickler.TensorView) d.get("model.w");
        assertEquals("storage:12345", v.storage().toString());
        assertEquals(2, v.storageOffset());
        assertEquals(Arrays.toString(new long[]{3, 4}), Arrays.toString(v.size()));
        assertEquals(Arrays.toString(new long[]{4, 1}), Arrays.toString(v.stride()));
        assertEquals(Boolean.TRUE, v.requiresGrad());
        assertEquals("value", d.get("meta"));
    }

    @Test
    public void protocol4StackGlobalMemoizeAndFrameParse() throws IOException {
        // A protocol 4 pickle: FRAME heads the payload, the globals stack and MEMOIZE keeps the helpers.
        ByteArrayOutputStream o = new ByteArrayOutputStream();
        o.write(0x80);
        o.write(4);
        o.write(0x95); // FRAME
        for (int i = 0; i < 8; i++) {
            o.write(0);
        }
        o.write(0x8c); // SHORT_BINUNICODE module
        o.write(12);
        o.write("torch._utils".getBytes(StandardCharsets.UTF_8));
        o.write(0x8c);
        o.write(18);
        o.write("_rebuild_tensor_v2".getBytes(StandardCharsets.UTF_8));
        o.write(0x93); // STACK_GLOBAL
        o.write(0x94); // MEMOIZE (slot 0)
        o.write(0x2e); // STOP
        assertEquals("torch._utils._rebuild_tensor_v2", unpickle(o.toByteArray()));
    }

    @Test
    public void nestedMapsOneDeepJoinTheViews() throws IOException {
        Pb b = new Pb().proto(2).emptyDict().binput(0);
        b.str("outer");
        b.emptyDict();
        b.str("inner");
        b.tensorValue("FloatStorage", "5", 0, 1, 2);
        b.raw(0x73);
        b.raw(0x73);
        b.stop();
        Map<?, ?> top = (Map<?, ?>) unpickle(b.build());
        Map<?, ?> inner = (Map<?, ?>) top.get("outer");
        assertTrue(inner.get("inner") instanceof RestrictedUnpickler.TensorView);
    }

    @Test
    public void rebuildParameterReducesToTheSavedTensor() throws IOException {
        Pb b = new Pb().proto(2).emptyDict();
        b.str("p");
        b.global("torch._utils", "_rebuild_parameter");
        b.mark();
        b.viewBytes("HalfStorage", "9", "cpu", 0, 2, 3);
        b.raw(0x88); // the parameter's requires_grad
        b.newOd();   // the parameter's hooks
        b.tuple();
        b.raw(0x73); // SETITEM
        b.stop();
        Map<?, ?> d = (Map<?, ?>) unpickle(b.build());
        RestrictedUnpickler.TensorView v = (RestrictedUnpickler.TensorView) d.get("p");
        assertEquals(2, v.size().length);
        assertEquals("storage:9", v.storage().toString());
    }

    @Test
    public void memoRoundTripsThroughPutAndGet() throws IOException {
        Pb b = new Pb().proto(2).str("kept").binput(7).raw(0x30).binget(7).stop(); // POP then GET
        assertEquals("kept", unpickle(b.build()));
    }

    @Test
    public void orderedDictReceivesItsItems() throws IOException {
        Pb b = new Pb().proto(2);
        b.global("collections", "OrderedDict");
        b.emptyList();
        b.mark();
        b.mark();
        b.str("k1");
        b.str("v1");
        b.tuple();
        b.mark();
        b.str("k2");
        b.str("v2");
        b.tuple();
        b.appends();
        b.tuple1();
        b.reduce();
        b.stop();
        Map<?, ?> od = (Map<?, ?>) unpickle(b.build());
        assertEquals(2, od.size());
        assertEquals("v1", od.get("k1"));
        assertEquals("v2", od.get("k2"));
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void maliciousGlobalIsRefused() {
        Pb b = new Pb().proto(2);
        b.global("os", "system");
        b.str("/bin/sh -c rm -rf /");
        b.reduce();
        b.stop();
        IOException e = assertThrows(IOException.class, () -> unpickle(b.build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the global 'os.system' is not on the allowed list"
                + " (torch._utils._rebuild_tensor_v2, torch._utils._rebuild_parameter, torch's storage classes,"
                + " torch.storage.UntypedStorage, collections.OrderedDict)", e.getMessage());
    }

    @Test
    public void unknownStorageStorageGlobalIsRefused() {
        Pb b = new Pb().proto(2);
        b.global("torch", "ExecutableStorage");
        IOException e = assertThrows(IOException.class, () -> unpickle(b.build()));
        assertTrue(e.getMessage().startsWith("Invalid PyTorch checkpoint pickle: the global 'torch.ExecutableStorage'"));
    }

    @Test
    public void disallowedOpcodeIsRefusedByName() {
        Pb b = new Pb().proto(2).emptyDict().raw(0x64).stop(); // DICT, an opcode torch never writes here
        IOException e = assertThrows(IOException.class, () -> unpickle(b.build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the opcode 0x64 (at pickle byte 3) is not on the allowed"
                + " opcode list", e.getMessage());
    }

    @Test
    public void legacyProtocolWithoutProtoHeadIsRefused() {
        Pb b = new Pb().global("torch", "FloatStorage");
        IOException e = assertThrows(IOException.class, () -> unpickle(b.build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the pickle's first opcode is not PROTO (a protocol-0/1"
                + " legacy pickle)", e.getMessage());
    }

    @Test
    public void protocolOutOfRangeIsRefused() {
        IOException e = assertThrows(IOException.class,
                () -> unpickle(new Pb().proto(1).stop().build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the pickle is protocol 1; torch checkpoints are"
                + " protocols 2-5", e.getMessage());
        e = assertThrows(IOException.class, () -> unpickle(new Pb().proto(6).stop().build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the pickle is protocol 6; torch checkpoints are"
                + " protocols 2-5", e.getMessage());
    }

    @Test
    public void rebuildOfUnknownCallableIsRefused() {
        Pb b = new Pb().proto(2).global("torch", "FloatStorage").emptyTuple().reduce().stop();
        IOException e = assertThrows(IOException.class, () -> unpickle(b.build()));
        assertEquals("Invalid PyTorch checkpoint pickle: REDUCE builds from 'torch.FloatStorage', which is not an"
                + " allowed global", e.getMessage());
    }

    @Test
    public void tensorWithNonTupleShapeIsRefused() {
        Pb b = new Pb().proto(2);
        b.global("torch._utils", "_rebuild_tensor_v2");
        b.mark();
        b.storageRef("FloatStorage", "1", "cpu");
        b.b1(0);
        b.str("shape"); // not a tuple
        b.mark();
        b.b1(1);
        b.tuple();
        b.raw(0x88);
        b.newOd();
        b.tuple();
        b.reduce();
        b.stop();
        IOException e = assertThrows(IOException.class, () -> unpickle(b.build()));
        assertEquals("Invalid PyTorch checkpoint pickle: _rebuild_tensor_v2 takes shape of integers (a tuple),"
                + " the pickle passes string 'shape'", e.getMessage());
    }

    @Test
    public void storageOnAnotherDeviceIsRefusedByTheLoader() {
        Pb b = new Pb().proto(2);
        b.global("torch._utils", "_rebuild_tensor_v2");
        b.mark();
        b.storageRef("FloatStorage", "1", "cuda:0");
        b.b1(0);
        b.mark();
        b.b1(1);
        b.b1(1);
        b.tuple();
        b.mark();
        b.b1(1);
        b.b1(1);
        b.tuple();
        b.raw(0x88);
        b.newOd();
        b.tuple();
        b.reduce();
        b.stop();
        IOException e = assertThrows(IOException.class, () -> RestrictedUnpickler.unpickle(b.build(),
                id -> null));
        assertEquals("Invalid PyTorch checkpoint pickle: the persistent id resolves to nothing", e.getMessage());
    }

    @Test
    public void truncatedAndOversizedInputsAreRefused() {
        IOException e = assertThrows(IOException.class, () -> unpickle(new Pb().proto(2).raw(0x58).build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the pickle ends inside a 4 byte integer at 3",
                e.getMessage());
        e = assertThrows(IOException.class, () -> unpickle(new Pb().proto(2).stop().build()));
        assertEquals("Invalid PyTorch checkpoint pickle: the pickle ends with no value", e.getMessage());
        e = assertThrows(IOException.class, () -> RestrictedUnpickler.unpickle(new byte[0], LOCAL_STORAGE));
        assertEquals("Invalid PyTorch checkpoint pickle: the pickle is empty", e.getMessage());
    }

    @Test
    public void long1CarriesOffsetsPastTheIntMax() throws IOException {
        Pb b = new Pb().proto(2);
        b.global("torch._utils", "_rebuild_tensor_v2");
        b.mark();
        b.storageRef("FloatStorage", "big", "cpu");
        b.raw(0x8a); // LONG1: a 4 byte little-endian integer
        b.raw(4);
        b.raw(0x00);
        b.raw(0x00);
        b.raw(0x00);
        b.raw(0x40);
        b.mark();
        b.b1(2);
        b.tuple();
        b.mark();
        b.b1(1);
        b.b1(1);
        b.tuple();
        b.raw(0x88);
        b.newOd();
        b.tuple();
        b.reduce();
        b.stop();
        Object view = unpickle(b.build());
        assertEquals(0x40000000L, ((RestrictedUnpickler.TensorView) view).storageOffset());
    }
}
