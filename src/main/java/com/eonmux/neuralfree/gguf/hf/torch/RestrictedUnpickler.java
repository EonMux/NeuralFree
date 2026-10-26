package com.eonmux.neuralfree.gguf.hf.torch;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The restricted unpickler of PyTorch pickles (FX-1, §3.6): a stack machine over torch's {@code data.pkl}
 * bytecode that accepts opcodes and globals by allow-list and refuses everything else with a named error. It
 * plays the role the Python {@code Unpickler} plays for {@code torch.load}, but:
 * <ul>
 *   <li>{@code ObjectInputStream} and reflection-based deserialization are never used, so nothing is executed;
 *       values are built from the allow-listed opcodes only;</li>
 *   <li>{@code GLOBAL} is restricted to {@code torch._utils._rebuild_tensor_v2}, {@code
 *       torch._utils._rebuild_parameter}, the plain {@code torch.*Storage} classes (the closed set {@link
 *       TorchStorage} maps), {@code torch.storage.UntypedStorage} and {@code collections.OrderedDict}; any other
 *       global is refused;</li>
 *   <li>the opcode allow-list is the plan's: {@code PROTO} (protocols 2-5, the first opcode), {@code FRAME},
 *       containers and their fills (dicts, lists, tuples, {@code SETITEM(S)}), the integer and unicode pushes,
 *       {@code BINPERSID} (the persistent ids), {@code REDUCE}/{@code BUILD}/{@code NEWOBJ} of the allowed
 *       globals, {@code STOP} and the memo ops, plus the plain stack ops ({@code MARK}, {@code POP},
 *       {@code POP_MARK}, {@code DUP}). The constants pushes and stack plumbing a real torch pickle needs — the
 *       boolean constants, {@code EMPTY_LIST}/{@code APPEND(S)}, {@code LONG1}/{@code LONG4} (storage offsets of
 *       tensors with more than 2^31 elements), {@code NONE} — build nothing on their own, so allowing them keeps
 *       the restriction whole;</li>
 *   <li>every input is capped, the explicit limits §3.6 asks for PyTorch pickles: pickle size, opcode count,
 *       stack depth, memo slots and string length.</li>
 * </ul>
 *
 * <p>The built value graph holds {@link String}, {@link Long}, {@link Boolean}, {@code Object[]} (tuples),
 * {@code List<Object>} (lists), {@code LinkedHashMap} (dicts and {@code collections.OrderedDict}), the {@link
 * TensorView} {@code torch._utils._rebuild_tensor_v2} reduced to (its storage through the persistent loader) and
 * plain {@code None} values. Nothing is mapped, nothing is executed, nothing is trusted beyond this graph.</p>
 */
public final class RestrictedUnpickler {

    /** Largest pickle evaluated. */
    public static final long MAX_PICKLE_BYTES = 512L << 20;
    /** Most opcodes of one pickle (a state dict spends some 40 per tensor). */
    public static final int MAX_OPS = 32_000_000;
    /** Most values on the stack at once. */
    public static final int MAX_STACK = 1_000_000;
    /** Most memo slots ({@code PUT}). */
    public static final int MAX_MEMO_SLOTS = 16_777_216;
    /** Longest string of one pickle (a tensor name). */
    public static final int MAX_STRING_BYTES = 1 << 20;

    private static final int PROTO = 0x80;
    private static final int NEWOBJ = 0x81;
    private static final int NEWOBJ_EX = 0x92;
    private static final int FRAME = 0x95;
    private static final int MEMOIZE = 0x94;
    private static final int STACK_GLOBAL = 0x93;
    private static final int TUPLE1 = 0x85;
    private static final int TUPLE2 = 0x86;
    private static final int TUPLE3 = 0x87;
    private static final int EMPTY_TUPLE = 0x29;
    private static final int TUPLE = 0x74;
    private static final int EMPTY_LIST = 0x5d;
    private static final int EMPTY_DICT = 0x7d;
    private static final int APPEND = 0x61;
    private static final int APPENDS = 0x65;
    private static final int SETITEM = 0x73;
    private static final int SETITEMS = 0x75;
    private static final int MARK = 0x28;
    private static final int STOP = 0x2e;
    private static final int BINPERSID = 0x51;
    private static final int REDUCE = 0x52;
    private static final int BUILD = 0x62;
    private static final int GLOBAL = 0x63;
    private static final int BINUNICODE = 0x58;
    private static final int SHORT_BINUNICODE = 0x8c;
    private static final int BINUNICODE8 = 0x8d;
    private static final int BININT = 0x4a;
    private static final int BININT1 = 0x4b;
    private static final int BININT2 = 0x4d;
    private static final int LONG1 = 0x8a;
    private static final int LONG4 = 0x8b;
    private static final int PUT = 0x70;
    private static final int BINPUT = 0x71;
    private static final int LONG_BINPUT = 0x72;
    private static final int GET = 0x67;
    private static final int BINGET = 0x68;
    private static final int LONG_BINGET = 0x6a;
    private static final int NONE = 0x4e;
    private static final int NEWTRUE = 0x88;
    private static final int NEWFALSE = 0x89;
    private static final int POP = 0x30;
    private static final int POP_MARK = 0x31;
    private static final int DUP = 0x32;

    /** The persistent-id resolver ({@code BINPERSID}), asked by the checkpoint that runs the unpickler. */
    public interface PersistentLoader {

        /**
         * The value of the persistent id {@code id}. A torch checkpoint's ids are the tuple
         * {@code ('storage', <storage type>, <key>, <location>)} the loader resolves to a {@link TorchStorage}.
         *
         * @throws IOException when the id names something the checkpoint refuses
         */
        Object load(List<Object> id) throws IOException;
    }

    /** What {@code torch._utils._rebuild_tensor_v2} reduces to: a view over the value the persistent loader
     *  resolved (a torch checkpoint requires a {@link TorchStorage}, refused by the checkpoint build). */
    public static final class TensorView {

        private final Object storage;
        private final long storageOffset;
        private final long[] size;
        private final long[] stride;
        private final boolean requiresGrad;

        TensorView(Object storage, long storageOffset, long[] size, long[] stride, boolean requiresGrad) {
            this.storage = storage;
            this.storageOffset = storageOffset;
            this.size = size.clone();
            this.stride = stride.clone();
            this.requiresGrad = requiresGrad;
        }

        /** The loader's resolved value (a torch checkpoint's is a {@link TorchStorage}). */
        public Object storage() {
            return storage;
        }

        /** The element offset the tensor starts at in the storage. */
        public long storageOffset() {
            return storageOffset;
        }

        /** The shape, outermost dimension first (a copy). */
        public long[] size() {
            return size.clone();
        }

        /** The element strides per dimension (a copy). */
        public long[] stride() {
            return stride.clone();
        }

        /** The pickled {@code requires_grad} flag (never used when reading). */
        public boolean requiresGrad() {
            return requiresGrad;
        }
    }

    private RestrictedUnpickler() {
    }

    /**
     * Evaluates {@code bytes}, a pickle of protocol 2-5, under the allow-list, resolving persistent ids through
     * {@code loader}, and returns the built object graph.
     *
     * @throws IOException when a limit is exceeded, an unsupported opcode or global appears, a persistent id is
     *                     refused, or the pickle does not end in {@code STOP}
     */
    public static Object unpickle(byte[] bytes, PersistentLoader loader) throws IOException {
        java.util.Objects.requireNonNull(bytes, "bytes");
        java.util.Objects.requireNonNull(loader, "loader");
        if (bytes.length > MAX_PICKLE_BYTES) {
            throw pickle("the pickle is " + bytes.length + " bytes, more than the limit of " + MAX_PICKLE_BYTES);
        }
        return new Machine(bytes, loader).run();
    }

    private static IOException pickle(String reason) {
        return new IOException("Invalid PyTorch checkpoint pickle: " + reason);
    }

    private static final class Machine {
        private final byte[] data;
        private final PersistentLoader loader;
        private final List<Object> stack = new ArrayList<>();
        private final Map<Integer, Object> memo = new HashMap<>();
        private final Marker mark = new Marker();
        private int position;
        private int ops;
        private boolean protocolSeen;
        private boolean stopped;
        private Object result;

        Machine(byte[] data, PersistentLoader loader) {
            this.data = data;
            this.loader = loader;
        }

        private static final class Marker {
        }

        Object run() throws IOException {
            if (data.length == 0) {
                throw pickle("the pickle is empty");
            }
            while (position < data.length) {
                opcode();
            }
            if (!stopped) {
                throw pickle("the pickle ends before STOP");
            }
            return result;
        }

        private void opcode() throws IOException {
            if (++ops > MAX_OPS) {
                throw pickle("the pickle holds more than the " + MAX_OPS + " opcodes of the limit");
            }
            if (stack.size() > MAX_STACK) {
                throw pickle("the pickle holds more than the " + MAX_STACK + " stack values of the limit");
            }
            int code = data[position] & 0xFF;
            position++;
            switch (code) {
                case PROTO: {
                    if (protocolSeen || ops != 1) {
                        throw pickle("the pickle names a protocol after its head");
                    }
                    int version = byte1() & 0xFF;
                    if (version < 2 || version > 5) {
                        throw pickle("the pickle is protocol " + version + "; torch checkpoints are protocols"
                                + " 2-5");
                    }
                    protocolSeen = true;
                    return;
                }
                case FRAME:
                    demandsProtocol();
                    position += 8;
                    if (position > data.length) {
                        throw pickle("the pickle ends inside a FRAME header");
                    }
                    return;
                case MARK:
                    demandsProtocol();
                    stack.add(mark);
                    return;
                case STOP: {
                    demandsProtocol();
                    if (stack.isEmpty()) {
                        throw pickle("the pickle ends with no value");
                    }
                    Object value = takeOne();
                    if (!stack.isEmpty()) {
                        throw pickle("the pickle ends with " + stack.size()
                                + (stack.get(stack.size() - 1) == mark ? " values under a mark" : " values"));
                    }
                    // Bytes after STOP stay unread: Python's load stops at the opcode and so does this one.
                    result = value;
                    stopped = true;
                    position = data.length;
                    return;
                }
                case POP_MARK: {
                    demandsProtocol();
                    popToMark();
                    return;
                }
                case POP: {
                    demandsProtocol();
                    takeOne();
                    return;
                }
                case DUP: {
                    demandsProtocol();
                    if (stack.isEmpty()) {
                        throw pickle("DUP duplicates from an empty stack");
                    }
                    Object top = stack.get(stack.size() - 1);
                    stack.add(top);
                    return;
                }
                case BININT: {
                    demandsProtocol();
                    int v = littleEndianInt(data, position);
                    position += 4;
                    stack.add((long) v);
                    return;
                }
                case BININT1: {
                    demandsProtocol();
                    stack.add((long) (byte1() & 0xFF));
                    return;
                }
                case BININT2: {
                    demandsProtocol();
                    int v = (byte1() & 0xFF) | ((byte1() & 0xFF) << 8);
                    stack.add((long) v);
                    return;
                }
                case LONG1:
                case LONG4: {
                    demandsProtocol();
                    int n = code == LONG1 ? byte1() & 0xFF : littleEndianInt(data, position);
                    if (code == LONG4) {
                        position += 4;
                    }
                    if (n < 0 || n > 8) {
                        throw pickle("a " + n + " byte integer exceeds the 8 bytes of a long");
                    }
                    long v = signedLittleEndian(data, position, n);
                    position += n;
                    stack.add(v);
                    return;
                }
                case BINUNICODE: {
                    demandsProtocol();
                    int len = littleEndianInt(data, position);
                    position += 4;
                    stack.add(unicode("BINUNICODE", len));
                    return;
                }
                case SHORT_BINUNICODE: {
                    demandsProtocol();
                    stack.add(unicode("SHORT_BINUNICODE", byte1() & 0xFF));
                    return;
                }
                case BINUNICODE8: {
                    demandsProtocol();
                    long len = littleEndianLong(data, position);
                    position += 8;
                    if (len > MAX_STRING_BYTES) {
                        throw pickle("a " + len + " byte string exceeds the limit of " + MAX_STRING_BYTES);
                    }
                    stack.add(unicode("BINUNICODE8", (int) len));
                    return;
                }
                case NONE:
                    demandsProtocol();
                    stack.add(null);
                    return;
                case NEWTRUE:
                case NEWFALSE:
                    demandsProtocol();
                    stack.add(code == NEWTRUE);
                    return;
                case EMPTY_TUPLE:
                    demandsProtocol();
                    stack.add(new Object[0]);
                    return;
                case EMPTY_LIST:
                    demandsProtocol();
                    stack.add(new ArrayList<Object>());
                    return;
                case EMPTY_DICT:
                    demandsProtocol();
                    stack.add(new LinkedHashMap<Object, Object>());
                    return;
                case TUPLE1: {
                    demandsProtocol();
                    Object v1 = takeOne();
                    stack.add(new Object[]{v1});
                    return;
                }
                case TUPLE2: {
                    demandsProtocol();
                    Object v2 = takeOne();
                    Object v1 = takeOne();
                    stack.add(new Object[]{v1, v2});
                    return;
                }
                case TUPLE3: {
                    demandsProtocol();
                    Object v3 = takeOne();
                    Object v2 = takeOne();
                    Object v1 = takeOne();
                    stack.add(new Object[]{v1, v2, v3});
                    return;
                }
                case TUPLE: {
                    demandsProtocol();
                    int mark = markIndex();
                    Object[] t = new Object[stack.size() - mark - 1];
                    for (int i = 0; i < t.length; i++) {
                        t[i] = stack.get(mark + 1 + i);
                    }
                    stack.subList(mark, stack.size()).clear();
                    stack.add(t);
                    return;
                }
                case APPEND: {
                    demandsProtocol();
                    Object value = takeOne();
                    listTarget("APPEND").add(value);
                    return;
                }
                case APPENDS: {
                    demandsProtocol();
                    int m = markIndex();
                    List<Object> values = new ArrayList<>(stack.subList(m + 1, stack.size()));
                    stack.subList(m, stack.size()).clear();
                    listTarget("APPENDS").addAll(values);
                    return;
                }
                case SETITEM: {
                    demandsProtocol();
                    Object value = takeOne();
                    Object key = takeOne();
                    mapTarget("SETITEM").put(key, value);
                    return;
                }
                case SETITEMS: {
                    demandsProtocol();
                    int m = markIndex();
                    int items = stack.size() - m - 1;
                    if (items % 2 != 0) {
                        stack.subList(m, stack.size()).clear();
                        throw pickle("SETITEMS ends on a key without a value");
                    }
                    List<Object> keys = new ArrayList<>(items / 2);
                    List<Object> values = new ArrayList<>(items / 2);
                    for (int i = m + 1; i < stack.size(); i += 2) {
                        keys.add(stack.get(i));
                        values.add(stack.get(i + 1));
                    }
                    stack.subList(m, stack.size()).clear();
                    Map<Object, Object> target = mapTarget("SETITEMS");
                    for (int i = 0; i < keys.size(); i++) {
                        target.put(keys.get(i), values.get(i));
                    }
                    return;
                }
                case PUT:
                    demandsProtocol();
                    store(decimal("PUT"));
                    return;
                case BINPUT:
                    demandsProtocol();
                    store(byte1() & 0xFF);
                    return;
                case LONG_BINPUT: {
                    demandsProtocol();
                    int slot = littleEndianInt(data, position);
                    position += 4;
                    if (slot < 0) {
                        throw pickle("LONG_BINPUT names the memo slot " + slot);
                    }
                    store(slot);
                    return;
                }
                case GET:
                    demandsProtocol();
                    recall(decimal("GET"));
                    return;
                case BINGET:
                    demandsProtocol();
                    recall(byte1() & 0xFF);
                    return;
                case LONG_BINGET: {
                    demandsProtocol();
                    int slot = littleEndianInt(data, position);
                    position += 4;
                    if (slot < 0) {
                        throw pickle("LONG_BINGET names the memo slot " + slot);
                    }
                    recall(slot);
                    return;
                }
                case MEMOIZE: {
                    demandsProtocol();
                    if (stack.isEmpty()) {
                        throw pickle("MEMOIZE stores from an empty stack");
                    }
                    store(slot());
                    return;
                }
                case GLOBAL: {
                    demandsProtocol();
                    String module = line();
                    String name = line();
                    global(module, name);
                    return;
                }
                case STACK_GLOBAL: {
                    demandsProtocol();
                    Object name = takeOne();
                    Object module = takeOne();
                    if (!(name instanceof String) || !(module instanceof String)) {
                        throw pickle("STACK_GLOBAL pushes " + classify(module) + " and " + classify(name)
                                + " instead of a module and name");
                    }
                    global((String) module, (String) name);
                    return;
                }
                case REDUCE: {
                    demandsProtocol();
                    Object args = takeOne();
                    Object callable = takeOne();
                    if (!(callable instanceof String)) {
                        throw pickle("REDUCE builds from " + classify(callable) + " instead of an allowed global");
                    }
                    Object[] t = argTuple(callable.toString(), args);
                    stack.add(reduce(callable.toString(), t));
                    return;
                }
                case NEWOBJ: {
                    demandsProtocol();
                    Object args = takeOne();
                    Object callable = takeOne();
                    stack.add(newObj(callable, args, false));
                    return;
                }
                case NEWOBJ_EX: {
                    demandsProtocol();
                    Object kwargs = takeOne();
                    Object args = takeOne();
                    Object callable = takeOne();
                    if (!(kwargs instanceof Map) || !((Map<?, ?>) kwargs).isEmpty()) {
                        throw pickle("NEWOBJ_EX builds " + describe(callable) + " with non-empty keyword"
                                + " arguments (the " + classify(kwargs) + ")");
                    }
                    stack.add(newObj(callable, args, true));
                    return;
                }
                case BUILD: {
                    demandsProtocol();
                    Object state = takeOne();
                    Object target = takeOne();
                    if (!(target instanceof Map)) {
                        throw pickle("BUILD updates the " + classify(target) + " instead of a dict or an"
                                + " OrderedDict");
                    }
                    @SuppressWarnings("unchecked")
                    Map<Object, Object> out = (Map<Object, Object>) target;
                    if (state instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<Object, Object> stateMap = (Map<Object, Object>) state;
                        out.putAll(stateMap);
                        stack.add(target);
                        return;
                    }
                    Object[] pairs = state instanceof Object[] ? (Object[]) state
                            : state instanceof List ? ((List<?>) state).toArray()
                            : null;
                    if (pairs == null) {
                        throw pickle("BUILD updates a dict with the " + classify(state) + " state");
                    }
                    for (Object p : pairs) {
                        if (!(p instanceof Object[]) || ((Object[]) p).length != 2) {
                            throw pickle("BUILD's state carries " + classify(p) + " as an item");
                        }
                        Object[] pair = (Object[]) p;
                        out.put(pair[0], pair[1]);
                    }
                    stack.add(target);
                    return;
                }
                case BINPERSID: {
                    demandsProtocol();
                    Object id = takeOne();
                    List<Object> l = id instanceof Object[] ? new ArrayList<>(Arrays.asList((Object[]) id))
                            : id instanceof List ? (List<Object>) id
                            : null;
                    if (l == null) {
                        throw pickle("BINPERSID carries " + classify(id) + " as its persistent id");
                    }
                    Object resolved = loader.load(l);
                    if (resolved == null) {
                        throw pickle("the persistent id resolves to nothing");
                    }
                    stack.add(resolved);
                    return;
                }
                default:
                    throw pickle("the opcode 0x" + Integer.toHexString(code) + " (at pickle byte "
                            + (position - 1) + ") is not on the allowed opcode list");
            }
        }

        /** The functions the reducers of the allowed globals build. */
        private Object reduce(String callable, Object[] t) throws IOException {
            if ("collections.OrderedDict".equals(callable)) {
                LinkedHashMap<Object, Object> out = new LinkedHashMap<>();
                fillDict(out, t, callable);
                return out;
            }
            if ("torch._utils._rebuild_tensor_v2".equals(callable)) {
                if (t.length != 6) {
                    throw pickle("_rebuild_tensor_v2 takes 6 arguments, the pickle passes " + t.length);
                }
                if (t[0] == null) {
                    throw pickle("_rebuild_tensor_v2 rebuilds from nothing: the persistent id resolved to"
                            + " nothing");
                }
                if (!(t[1] instanceof Long)) {
                    throw pickle("_rebuild_tensor_v2 takes an integer storage offset, the pickle passes "
                            + classify(t[1]));
                }
                if ((Long) t[1] < 0) {
                    throw pickle("_rebuild_tensor_v2 takes a storage offset of " + t[1]
                            + ", which is negative");
                }
                long[] size = ints(t, 2, "_rebuild_tensor_v2", "shape");
                long[] stride = ints(t, 3, "_rebuild_tensor_v2", "strides");
                if (size.length != stride.length) {
                    throw pickle("_rebuild_tensor_v2 takes a shape of length " + size.length + " and strides of"
                            + " length " + stride.length);
                }
                for (long d : size) {
                    if (d < 0) {
                        throw pickle("_rebuild_tensor_v2 takes shape " + Arrays.toString(size) + ", which has a"
                                + " negative dimension");
                    }
                }
                if (!(t[4] instanceof Boolean)) {
                    throw pickle("_rebuild_tensor_v2 takes a requires_grad flag, the pickle passes "
                            + classify(t[4]));
                }
                if (!(t[5] instanceof Map) && !(t[5] instanceof List) && !(t[5] instanceof Object[])) {
                    throw pickle("_rebuild_tensor_v2 takes backward hooks, the pickle passes " + classify(t[5]));
                }
                return new TensorView(t[0], (Long) t[1], size, stride, (Boolean) t[4]);
            }
            if ("torch._utils._rebuild_parameter".equals(callable)) {
                if (t.length < 3 || t.length > 5) {
                    throw pickle("_rebuild_parameter takes 3 to 5 arguments, the pickle passes " + t.length);
                }
                if (!(t[0] instanceof TensorView)) {
                    throw pickle("_rebuild_parameter rebuilds from " + classify(t[0]) + ", not a saved tensor");
                }
                if (!(t[1] instanceof Boolean)) {
                    throw pickle("_rebuild_parameter takes a requires_grad flag, the pickle passes "
                            + classify(t[1]));
                }
                return t[0];
            }
            throw pickle("REDUCE builds from " + describe(callable) + ", which is not an allowed global");
        }

        private Object newObj(Object callable, Object args, boolean fromEx) throws IOException {
            if (!"collections.OrderedDict".equals(callable)) {
                throw pickle((fromEx ? "NEWOBJ_EX" : "NEWOBJ") + " builds from " + describe(callable)
                        + " instead of collections.OrderedDict");
            }
            LinkedHashMap<Object, Object> out = new LinkedHashMap<>();
            fillDict(out, argTuple("collections.OrderedDict", args), "collections.OrderedDict");
            return out;
        }

        private void fillDict(LinkedHashMap<Object, Object> out, Object[] t, String callable) throws IOException {
            if (t.length == 0) {
                return;
            }
            if (t.length != 1) {
                throw pickle(callable + " takes 0 or 1 argument, the pickle passes " + t.length);
            }
            List<Object> list = t[0] instanceof Object[] ? Arrays.asList((Object[]) t[0])
                    : t[0] instanceof List ? (List<Object>) t[0]
                    : null;
            if (list == null) {
                throw pickle(callable + " receives " + classify(t[0]) + " instead of its items");
            }
            for (Object p : list) {
                if (!(p instanceof Object[]) || ((Object[]) p).length != 2) {
                    throw pickle(callable + " carries " + classify(p) + " as an item");
                }
                Object[] pair = (Object[]) p;
                out.put(pair[0], pair[1]);
            }
        }

        /** The GLOBAL allow-list (see the class javadoc). */
        private void global(String module, String name) throws IOException {
            if ("torch._utils".equals(module) && "_rebuild_tensor_v2".equals(name)) {
                stack.add("torch._utils._rebuild_tensor_v2");
                return;
            }
            if ("torch._utils".equals(module) && "_rebuild_parameter".equals(name)) {
                stack.add("torch._utils._rebuild_parameter");
                return;
            }
            if ("collections".equals(module) && "OrderedDict".equals(name)) {
                stack.add("collections.OrderedDict");
                return;
            }
            if ("torch".equals(module) && name.endsWith("Storage")
                    && TorchStorage.TYPES.containsKey("torch." + name)) {
                stack.add("torch." + name);
                return;
            }
            if ("torch.storage".equals(module) && "UntypedStorage".equals(name)) {
                stack.add("torch.storage.UntypedStorage");
                return;
            }
            throw pickle("the global " + describe(module + "." + name) + " is not on the allowed list (torch._utils"
                    + "._rebuild_tensor_v2, torch._utils._rebuild_parameter, torch's storage classes,"
                    + " torch.storage.UntypedStorage, collections.OrderedDict)");
        }

        // ------------------------------------------------------------------ stack plumbing

        private void demandsProtocol() throws IOException {
            if (!protocolSeen) {
                throw pickle("the pickle's first opcode is not PROTO (a protocol-0/1 legacy pickle)");
            }
        }

        private Object takeOne() throws IOException {
            if (stack.isEmpty()) {
                throw pickle("the stack is empty");
            }
            return stack.remove(stack.size() - 1);
        }

        private void popToMark() throws IOException {
            int m = markIndex();
            stack.subList(m, stack.size()).clear();
        }

        private int markIndex() throws IOException {
            int m = stack.lastIndexOf(mark);
            if (m < 0) {
                throw pickle("the stack has no mark for the op that takes one");
            }
            return m;
        }

        private Map<Object, Object> mapTarget(String opName) throws IOException {
            Object target = stack.isEmpty() ? null : stack.get(stack.size() - 1);
            if (!(target instanceof Map)) {
                throw pickle(opName + " fills " + classify(target));
            }
            @SuppressWarnings("unchecked")
            Map<Object, Object> map = (Map<Object, Object>) target;
            return map;
        }

        private List<Object> listTarget(String opName) throws IOException {
            Object target = stack.isEmpty() ? null : stack.get(stack.size() - 1);
            if (!(target instanceof List)) {
                throw pickle(opName + " fills " + classify(target));
            }
            @SuppressWarnings("unchecked")
            List<Object> list = (List<Object>) target;
            return list;
        }

        private int slot() throws IOException {
            if (memo.size() >= MAX_MEMO_SLOTS) {
                throw pickle("the pickle uses more than " + MAX_MEMO_SLOTS + " memo slots");
            }
            return memo.size();
        }

        private void store(int slot) throws IOException {
            if (stack.isEmpty()) {
                throw pickle("a memo put stores from an empty stack");
            }
            if (slot > MAX_MEMO_SLOTS) {
                throw pickle("a memo put names slot " + slot + ", more than the limit of " + MAX_MEMO_SLOTS);
            }
            memo.put(slot, stack.get(stack.size() - 1));
        }

        private void recall(int slot) throws IOException {
            if (slot > MAX_MEMO_SLOTS) {
                throw pickle("a memo get names slot " + slot + ", more than the limit of " + MAX_MEMO_SLOTS);
            }
            Object value = memo.get(slot);
            if (value == null) {
                throw pickle("a memo get reads slot " + slot + ", which the pickle never stored");
            }
            stack.add(value);
        }

        private Object[] argTuple(String callable, Object args) throws IOException {
            if (!(args instanceof Object[])) {
                throw pickle(callable + " receives " + classify(args) + " instead of its arguments");
            }
            return (Object[]) args;
        }

        private long[] ints(Object[] args, int index, String callable, String what) throws IOException {
            if (!(args[index] instanceof Object[])) {
                throw pickle(callable + " takes " + what + " of integers (a tuple), the pickle passes "
                        + classify(args[index]));
            }
            Object[] t = (Object[]) args[index];
            long[] out = new long[t.length];
            for (int i = 0; i < t.length; i++) {
                if (!(t[i] instanceof Long)) {
                    throw pickle(callable + " takes " + what + " of integers, the pickle passes " + classify(t[i]));
                }
                out[i] = (Long) t[i];
            }
            return out;
        }

        private String describe(Object value) {
            if (value == null) {
                return "None";
            }
            if (value == mark) {
                return "the stack mark";
            }
            if (value instanceof String) {
                return "'" + value + "'";
            }
            return "the " + classify(value);
        }

        private String classify(Object value) {
            if (value == null) {
                return "None";
            }
            if (value == mark) {
                return "the stack mark";
            }
            if (value instanceof String) {
                return "string '" + value + "'";
            }
            if (value instanceof Long) {
                return "integer " + value;
            }
            if (value instanceof Boolean) {
                return "flag " + value;
            }
            if (value instanceof Object[]) {
                return "tuple of " + ((Object[]) value).length;
            }
            if (value instanceof List) {
                return "list of " + ((List<?>) value).size();
            }
            if (value instanceof Map) {
                return ((Map<?, ?>) value).isEmpty() ? "empty dict" : "dict of "
                        + ((Map<?, ?>) value).size();
            }
            if (value instanceof TorchStorage) {
                return "storage '" + ((TorchStorage) value).key() + "'";
            }
            if (value instanceof TensorView) {
                Object s = ((TensorView) value).storage();
                return "saved tensor of '" + (s instanceof TorchStorage ? ((TorchStorage) s).key() : "?") + "'";
            }
            return "value the restricted unpickler cannot name";
        }

        // ------------------------------------------------------------------ bytes

        private byte byte1() throws IOException {
            if (position >= data.length) {
                throw pickle("the pickle ends inside the opcode at " + (position - 1));
            }
            return data[position++];
        }

        /** A decimal ending in a newline, as PUT and GET carry their memo slot. */
        private int decimal(String opName) throws IOException {
            int start = position;
            while (position < data.length && data[position] >= '0' && data[position] <= '9') {
                position++;
            }
            if (position >= data.length || start == position || data[position] != '\n') {
                throw pickle(opName + " does not carry a memo slot");
            }
            long v = 0;
            for (int i = start; i < position; i++) {
                v = v * 10 + (data[i] - '0');
                if (v > MAX_MEMO_SLOTS) {
                    throw pickle(opName + " names a memo slot over " + MAX_MEMO_SLOTS);
                }
            }
            position++;
            return (int) v;
        }

        /** A plain printable line, as GLOBAL carries its module and name. */
        private String line() throws IOException {
            int start = position;
            while (position < data.length && (data[position] >= 0x20 && data[position] < 0x7f)) {
                position++;
            }
            if (position >= data.length || data[position] != '\n') {
                throw pickle("GLOBAL does not carry a printable module and name line");
            }
            String out = new String(data, start, position - start, StandardCharsets.ISO_8859_1);
            position++;
            return out;
        }

        private String unicode(String opName, int len) throws IOException {
            if (len > MAX_STRING_BYTES) {
                throw pickle(opName + " carries a " + len + " byte string, more than the limit of "
                        + MAX_STRING_BYTES);
            }
            if (position + len > data.length) {
                throw pickle(opName + " carries a " + len + " byte string, more than the pickle's "
                        + (data.length - position) + " remaining bytes");
            }
            String out = new String(data, position, len, StandardCharsets.UTF_8);
            position += len;
            return out;
        }

        private int littleEndianInt(byte[] b, int at) throws IOException {
            if (at + 4 > b.length) {
                throw pickle("the pickle ends inside a 4 byte integer at " + at);
            }
            return (b[at] & 0xFF) | ((b[at + 1] & 0xFF) << 8) | ((b[at + 2] & 0xFF) << 16)
                    | ((b[at + 3] & 0xFF) << 24);
        }

        private long littleEndianLong(byte[] b, int at) throws IOException {
            if (at + 8 > b.length) {
                throw pickle("the pickle ends inside an 8 byte integer at " + at);
            }
            return (littleEndianInt(b, at) & 0xFFFFFFFFL) | ((littleEndianInt(b, at + 4) & 0xFFFFFFFFL) << 32);
        }

        private long signedLittleEndian(byte[] b, int at, int n) throws IOException {
            if (at + n > b.length) {
                throw pickle("the pickle ends inside a " + n + " byte integer at " + at);
            }
            long v = 0;
            for (int i = 0; i < n; i++) {
                v |= (b[at + i] & 0xFFL) << (8 * i);
            }
            if (n < 8 && (b[at + n - 1] & 0x80) != 0) {
                v |= -1L << (8 * n);
            }
            return v;
        }
    }
}
