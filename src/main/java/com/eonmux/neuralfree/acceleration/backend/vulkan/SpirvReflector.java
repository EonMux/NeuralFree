package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Pure-Java SPIR-V reflection for compute modules: walks the instruction stream
 * and extracts bindings (grouped by set and binding), push-constant members,
 * local size, SpecIds and capabilities.
 *
 * <p>Only the module preamble is read (capabilities, entry points, execution modes, annotations,
 * types, constants and global variables); the walk stops at the first function. Storage buffers
 * are {@code StorageBuffer} variables, or {@code Uniform} variables whose block is decorated
 * {@code BufferBlock} (SPIR-V 1.0 modules). Every storage-buffer variable sharing a
 * (set, binding) pair is one binding whose access is the union of its views (a binding is
 * read-only only when every view is {@code NonWritable}). A view is non-writable or non-readable
 * when the variable carries the decoration or every member of its block does, which covers both
 * forms glslang emits. The local size comes from the constant decorated
 * {@code BuiltIn WorkgroupSize}, else from {@code LocalSizeId}, else from {@code LocalSize}.</p>
 *
 * <p>Uniform buffers, images, samplers, descriptor arrays and descriptor sets other than 0 are
 * rejected with {@link UnsupportedKernelException}, because the backend binds storage buffers of
 * set 0 only.</p>
 */
public final class SpirvReflector {

    private static final int MAGIC = 0x07230203;
    private static final int HEADER_WORDS = 5;
    private static final int MAX_ID_BOUND = 1 << 22;

    private static final int OP_ENTRY_POINT = 15;
    private static final int OP_EXECUTION_MODE = 16;
    private static final int OP_CAPABILITY = 17;
    private static final int OP_TYPE_BOOL = 20;
    private static final int OP_TYPE_INT = 21;
    private static final int OP_TYPE_FLOAT = 22;
    private static final int OP_TYPE_VECTOR = 23;
    private static final int OP_TYPE_MATRIX = 24;
    private static final int OP_TYPE_ARRAY = 28;
    private static final int OP_TYPE_RUNTIME_ARRAY = 29;
    private static final int OP_TYPE_STRUCT = 30;
    private static final int OP_TYPE_POINTER = 32;
    private static final int OP_CONSTANT_TRUE = 41;
    private static final int OP_CONSTANT_FALSE = 42;
    private static final int OP_CONSTANT = 43;
    private static final int OP_CONSTANT_COMPOSITE = 44;
    private static final int OP_SPEC_CONSTANT_TRUE = 48;
    private static final int OP_SPEC_CONSTANT_FALSE = 49;
    private static final int OP_SPEC_CONSTANT = 50;
    private static final int OP_SPEC_CONSTANT_COMPOSITE = 51;
    private static final int OP_FUNCTION = 54;
    private static final int OP_VARIABLE = 59;
    private static final int OP_DECORATE = 71;
    private static final int OP_MEMBER_DECORATE = 72;
    private static final int OP_EXECUTION_MODE_ID = 331;

    private static final int EXEC_MODEL_GL_COMPUTE = 5;
    private static final int EXEC_MODE_LOCAL_SIZE = 17;
    private static final int EXEC_MODE_LOCAL_SIZE_ID = 38;

    private static final int DECO_SPEC_ID = 1;
    private static final int DECO_BLOCK = 2;
    private static final int DECO_BUFFER_BLOCK = 3;
    private static final int DECO_ARRAY_STRIDE = 6;
    private static final int DECO_MATRIX_STRIDE = 7;
    private static final int DECO_BUILT_IN = 11;
    private static final int DECO_NON_WRITABLE = 24;
    private static final int DECO_NON_READABLE = 25;
    private static final int DECO_BINDING = 33;
    private static final int DECO_DESCRIPTOR_SET = 34;
    private static final int DECO_OFFSET = 35;

    private static final int BUILT_IN_WORKGROUP_SIZE = 25;

    private static final int SC_UNIFORM_CONSTANT = 0;
    private static final int SC_UNIFORM = 2;
    private static final int SC_PUSH_CONSTANT = 9;
    private static final int SC_STORAGE_BUFFER = 12;

    private SpirvReflector() {
    }

    /**
     * Reflects a module; the buffer's position and limit are not changed.
     *
     * @throws IllegalArgumentException   when the bytes are not a valid SPIR-V compute module
     * @throws UnsupportedKernelException when the module uses a resource the backend cannot bind
     */
    public static SpirvModuleInfo reflect(ByteBuffer spirv) {
        if (spirv == null) {
            throw new IllegalArgumentException("SPIR-V buffer is null");
        }
        ByteBuffer view = spirv.duplicate();
        int bytes = view.remaining();
        if (bytes % 4 != 0 || bytes < HEADER_WORDS * 4) {
            throw new IllegalArgumentException("SPIR-V module of " + bytes
                    + " bytes is not a whole number of words or is shorter than its header");
        }
        int base = view.position();
        view.order(ByteOrder.LITTLE_ENDIAN);
        if (view.getInt(base) != MAGIC) {
            view.order(ByteOrder.BIG_ENDIAN);
            if (view.getInt(base) != MAGIC) {
                throw new IllegalArgumentException(String.format(
                        "not a SPIR-V module: magic word 0x%08x", view.order(ByteOrder.LITTLE_ENDIAN).getInt(base)));
            }
        }
        int[] words = new int[bytes / 4];
        for (int i = 0; i < words.length; i++) {
            words[i] = view.getInt(base + 4 * i);
        }
        return new Parser(words).parse();
    }

    /** Same as {@link #reflect(ByteBuffer)} for a byte array. */
    public static SpirvModuleInfo reflect(byte[] spirv) {
        if (spirv == null) {
            throw new IllegalArgumentException("SPIR-V bytes are null");
        }
        return reflect(ByteBuffer.wrap(spirv));
    }

    /** One type, constant or variable declaration: its opcode and the operands after its result id. */
    private static final class Decl {
        final int op;
        final int[] operands;

        Decl(int op, int[] operands) {
            this.op = op;
            this.operands = operands;
        }
    }

    /** Single-use parser of one module's preamble. */
    private static final class Parser {
        private final int[] words;
        private final Set<Integer> capabilities = new HashSet<>();
        private final Map<Integer, Decl> types = new HashMap<>();
        private final Map<Integer, Decl> constants = new HashMap<>();
        /** Result type, storage class and result id of each global variable. */
        private final List<int[]> variables = new ArrayList<>();
        private final Map<Integer, Map<Integer, int[]>> decorations = new HashMap<>();
        private final Map<Integer, Map<Integer, Map<Integer, int[]>>> memberDecorations = new HashMap<>();
        private final Map<Integer, int[]> localSizeModes = new HashMap<>();
        private final Map<Integer, int[]> localSizeIdModes = new HashMap<>();
        private int entryId = -1;
        private String entryName;

        Parser(int[] words) {
            this.words = words;
        }

        SpirvModuleInfo parse() {
            int bound = words[3];
            if (bound <= 0 || bound > MAX_ID_BOUND) {
                throw new IllegalArgumentException("SPIR-V id bound " + bound + " is out of range");
            }
            walk();
            if (entryId < 0) {
                throw new IllegalArgumentException("SPIR-V module has no GLCompute entry point");
            }
            String where = "SPIR-V entry point '" + entryName + "'";
            Map<Integer, boolean[]> access = new TreeMap<>();
            int pushVariable = -1;
            for (int[] v : variables) {
                int storageClass = v[1];
                if (storageClass == SC_PUSH_CONSTANT) {
                    if (pushVariable >= 0) {
                        throw new IllegalArgumentException(where + " declares more than one push-constant block");
                    }
                    pushVariable = v[0];
                } else if (storageClass == SC_STORAGE_BUFFER || storageClass == SC_UNIFORM
                        || storageClass == SC_UNIFORM_CONSTANT) {
                    collectBinding(v, access, where);
                }
            }
            List<SpirvModuleInfo.BindingInfo> bindings = new ArrayList<>(access.size());
            for (Map.Entry<Integer, boolean[]> e : access.entrySet()) {
                bindings.add(new SpirvModuleInfo.BindingInfo(e.getKey(), e.getValue()[0], e.getValue()[1]));
            }
            List<SpirvModuleInfo.PushMember> members = new ArrayList<>();
            int pushSize = pushVariable >= 0 ? collectPushMembers(pushVariable, members, where) : 0;
            int[] localSize = new int[3];
            int[] localSpecIds = new int[3];
            resolveLocalSize(localSize, localSpecIds, where);
            Map<Integer, Integer> specDefaults = new HashMap<>();
            Map<Integer, Integer> specSizes = new HashMap<>();
            collectSpecConstants(specDefaults, specSizes, where);
            return new SpirvModuleInfo(entryName, bindings, members, pushSize, localSize, localSpecIds,
                    specDefaults, specSizes, capabilities);
        }

        private void walk() {
            int pos = HEADER_WORDS;
            while (pos < words.length) {
                int first = words[pos];
                int count = first >>> 16;
                int op = first & 0xFFFF;
                if (count == 0 || pos + count > words.length) {
                    throw new IllegalArgumentException("truncated or malformed SPIR-V instruction (opcode " + op
                            + ", " + count + " words) at word " + pos);
                }
                if (op == OP_FUNCTION) {
                    return;
                }
                record(op, pos, count);
                pos += count;
            }
        }

        private void record(int op, int pos, int count) {
            switch (op) {
                case OP_CAPABILITY:
                    capabilities.add(word(pos, 1, count));
                    break;
                case OP_ENTRY_POINT:
                    if (word(pos, 1, count) == EXEC_MODEL_GL_COMPUTE && entryId < 0) {
                        entryId = word(pos, 2, count);
                        entryName = literalString(pos + 3, pos + count);
                    }
                    break;
                case OP_EXECUTION_MODE:
                    if (word(pos, 2, count) == EXEC_MODE_LOCAL_SIZE) {
                        localSizeModes.put(word(pos, 1, count),
                                new int[]{word(pos, 3, count), word(pos, 4, count), word(pos, 5, count)});
                    }
                    break;
                case OP_EXECUTION_MODE_ID:
                    if (word(pos, 2, count) == EXEC_MODE_LOCAL_SIZE_ID) {
                        localSizeIdModes.put(word(pos, 1, count),
                                new int[]{word(pos, 3, count), word(pos, 4, count), word(pos, 5, count)});
                    }
                    break;
                case OP_DECORATE:
                    decorations.computeIfAbsent(word(pos, 1, count), k -> new HashMap<>())
                            .put(word(pos, 2, count), tail(pos, 3, count));
                    break;
                case OP_MEMBER_DECORATE:
                    memberDecorations.computeIfAbsent(word(pos, 1, count), k -> new HashMap<>())
                            .computeIfAbsent(word(pos, 2, count), k -> new HashMap<>())
                            .put(word(pos, 3, count), tail(pos, 4, count));
                    break;
                case OP_TYPE_BOOL:
                case OP_TYPE_INT:
                case OP_TYPE_FLOAT:
                case OP_TYPE_VECTOR:
                case OP_TYPE_MATRIX:
                case OP_TYPE_ARRAY:
                case OP_TYPE_RUNTIME_ARRAY:
                case OP_TYPE_STRUCT:
                case OP_TYPE_POINTER:
                    types.put(word(pos, 1, count), new Decl(op, tail(pos, 2, count)));
                    break;
                case OP_CONSTANT_TRUE:
                case OP_CONSTANT_FALSE:
                case OP_CONSTANT:
                case OP_CONSTANT_COMPOSITE:
                case OP_SPEC_CONSTANT_TRUE:
                case OP_SPEC_CONSTANT_FALSE:
                case OP_SPEC_CONSTANT:
                case OP_SPEC_CONSTANT_COMPOSITE:
                    constants.put(word(pos, 2, count), new Decl(op, withType(word(pos, 1, count), tail(pos, 3, count))));
                    break;
                case OP_VARIABLE:
                    variables.add(new int[]{word(pos, 2, count), word(pos, 3, count), word(pos, 1, count)});
                    break;
                default:
                    break;
            }
        }

        /** Merges one descriptor variable into {@code access} (binding to {readable, writable}). */
        private void collectBinding(int[] variable, Map<Integer, boolean[]> access, String where) {
            int id = variable[0];
            int storageClass = variable[1];
            Decl pointer = types.get(variable[2]);
            if (pointer == null || pointer.op != OP_TYPE_POINTER) {
                throw new IllegalArgumentException(where + ": variable %" + id + " does not have a pointer type");
            }
            Integer binding = decorationValue(id, DECO_BINDING);
            Integer set = decorationValue(id, DECO_DESCRIPTOR_SET);
            String name = "binding " + (binding == null ? "?" : binding) + " (variable %" + id + ")";
            if (storageClass == SC_UNIFORM_CONSTANT) {
                throw new UnsupportedKernelException(where + ": " + name
                        + " is an image, sampler or other opaque resource; only storage buffers are supported");
            }
            Decl pointee = types.get(pointer.operands[1]);
            if (pointee == null) {
                throw new IllegalArgumentException(where + ": " + name + " points to an undeclared type");
            }
            if (pointee.op == OP_TYPE_ARRAY || pointee.op == OP_TYPE_RUNTIME_ARRAY) {
                throw new UnsupportedKernelException(where + ": " + name
                        + " is a descriptor array; only single storage buffers are supported");
            }
            if (pointee.op != OP_TYPE_STRUCT) {
                throw new IllegalArgumentException(where + ": " + name + " is not a block");
            }
            int structId = pointer.operands[1];
            if (storageClass == SC_UNIFORM && !hasDecoration(structId, DECO_BUFFER_BLOCK)) {
                throw new UnsupportedKernelException(where + ": " + name
                        + " is a uniform buffer; only storage buffers are supported");
            }
            if (storageClass == SC_STORAGE_BUFFER && !hasDecoration(structId, DECO_BLOCK)) {
                throw new IllegalArgumentException(where + ": storage buffer " + name + " is not decorated Block");
            }
            if (binding == null || set == null) {
                throw new IllegalArgumentException(where + ": storage buffer variable %" + id
                        + " lacks a Binding or DescriptorSet decoration");
            }
            if (set != 0) {
                throw new UnsupportedKernelException(where + ": storage buffer " + name + " is in descriptor set "
                        + set + "; only set 0 is supported");
            }
            boolean readable = !(hasDecoration(id, DECO_NON_READABLE)
                    || allMembersDecorated(structId, pointee.operands.length, DECO_NON_READABLE));
            boolean writable = !(hasDecoration(id, DECO_NON_WRITABLE)
                    || allMembersDecorated(structId, pointee.operands.length, DECO_NON_WRITABLE));
            boolean[] merged = access.get(binding);
            if (merged == null) {
                access.put(binding, new boolean[]{readable, writable});
            } else {
                merged[0] |= readable;
                merged[1] |= writable;
            }
        }

        /** Fills {@code members} and returns the block size rounded up to 4 bytes. */
        private int collectPushMembers(int variable, List<SpirvModuleInfo.PushMember> members, String where) {
            int typeId = -1;
            for (int[] v : variables) {
                if (v[0] == variable) {
                    typeId = v[2];
                }
            }
            Decl pointer = types.get(typeId);
            Decl struct = pointer == null || pointer.op != OP_TYPE_POINTER ? null : types.get(pointer.operands[1]);
            if (struct == null || struct.op != OP_TYPE_STRUCT) {
                throw new IllegalArgumentException(where + ": the push-constant variable is not a block");
            }
            int structId = pointer.operands[1];
            int end = 0;
            for (int m = 0; m < struct.operands.length; m++) {
                Integer offset = memberDecorationValue(structId, m, DECO_OFFSET);
                if (offset == null) {
                    throw new IllegalArgumentException(where + ": push-constant member " + m + " has no Offset");
                }
                int memberType = struct.operands[m];
                Decl t = types.get(memberType);
                if (t == null) {
                    throw new IllegalArgumentException(where + ": push-constant member " + m
                            + " has an undeclared type");
                }
                SpirvModuleInfo.PushMember.Kind kind = scalarKind(t);
                int size = kind != null ? (kind == SpirvModuleInfo.PushMember.Kind.I64
                        || kind == SpirvModuleInfo.PushMember.Kind.F64 ? 8 : 4)
                        : sizeOf(memberType, memberDecorationValue(structId, m, DECO_MATRIX_STRIDE), where);
                members.add(new SpirvModuleInfo.PushMember(offset, size,
                        kind != null ? kind : SpirvModuleInfo.PushMember.Kind.BYTES));
                end = Math.max(end, offset + size);
            }
            return (end + 3) & ~3;
        }

        private SpirvModuleInfo.PushMember.Kind scalarKind(Decl t) {
            if (t.op == OP_TYPE_INT && t.operands[0] == 32) {
                return t.operands[1] != 0 ? SpirvModuleInfo.PushMember.Kind.I32 : SpirvModuleInfo.PushMember.Kind.U32;
            }
            if (t.op == OP_TYPE_INT && t.operands[0] == 64) {
                return SpirvModuleInfo.PushMember.Kind.I64;
            }
            if (t.op == OP_TYPE_FLOAT && t.operands[0] == 32) {
                return SpirvModuleInfo.PushMember.Kind.F32;
            }
            if (t.op == OP_TYPE_FLOAT && t.operands[0] == 64) {
                return SpirvModuleInfo.PushMember.Kind.F64;
            }
            return null;
        }

        /** Explicit-layout size of a push-constant member type. */
        private int sizeOf(int typeId, Integer matrixStride, String where) {
            Decl t = types.get(typeId);
            if (t == null) {
                throw new IllegalArgumentException(where + ": undeclared type %" + typeId + " in the push constants");
            }
            switch (t.op) {
                case OP_TYPE_BOOL:
                    return 4;
                case OP_TYPE_INT:
                case OP_TYPE_FLOAT:
                    return t.operands[0] / 8;
                case OP_TYPE_VECTOR:
                    return t.operands[1] * sizeOf(t.operands[0], null, where);
                case OP_TYPE_MATRIX: {
                    int column = matrixStride != null ? matrixStride : sizeOf(t.operands[0], null, where);
                    return t.operands[1] * column;
                }
                case OP_TYPE_ARRAY: {
                    Decl length = constants.get(t.operands[1]);
                    if (length == null || length.op != OP_CONSTANT) {
                        throw new UnsupportedKernelException(where
                                + ": a push-constant array length is not a plain constant");
                    }
                    Integer stride = decorationValue(typeId, DECO_ARRAY_STRIDE);
                    int element = stride != null ? stride : sizeOf(t.operands[0], matrixStride, where);
                    return length.operands[1] * element;
                }
                case OP_TYPE_STRUCT: {
                    int end = 0;
                    for (int m = 0; m < t.operands.length; m++) {
                        Integer offset = memberDecorationValue(typeId, m, DECO_OFFSET);
                        if (offset == null) {
                            throw new IllegalArgumentException(where + ": nested push-constant member " + m
                                    + " has no Offset");
                        }
                        end = Math.max(end, offset + sizeOf(t.operands[m],
                                memberDecorationValue(typeId, m, DECO_MATRIX_STRIDE), where));
                    }
                    return end;
                }
                default:
                    throw new UnsupportedKernelException(where + ": push-constant type with opcode " + t.op
                            + " is not supported");
            }
        }

        private void resolveLocalSize(int[] size, int[] specIds, String where) {
            int[] components = null;
            for (Map.Entry<Integer, Map<Integer, int[]>> e : decorations.entrySet()) {
                int[] builtIn = e.getValue().get(DECO_BUILT_IN);
                if (builtIn != null && builtIn.length > 0 && builtIn[0] == BUILT_IN_WORKGROUP_SIZE) {
                    Decl composite = constants.get(e.getKey());
                    if (composite == null || (composite.op != OP_CONSTANT_COMPOSITE
                            && composite.op != OP_SPEC_CONSTANT_COMPOSITE) || composite.operands.length != 4) {
                        throw new IllegalArgumentException(where
                                + ": the WorkgroupSize built-in is not a three-component constant");
                    }
                    components = new int[]{composite.operands[1], composite.operands[2], composite.operands[3]};
                }
            }
            if (components == null) {
                components = localSizeIdModes.get(entryId);
            }
            if (components != null) {
                for (int d = 0; d < 3; d++) {
                    Decl c = constants.get(components[d]);
                    if (c != null && c.op == OP_CONSTANT) {
                        size[d] = c.operands[1];
                        specIds[d] = -1;
                    } else if (c != null && c.op == OP_SPEC_CONSTANT) {
                        Integer specId = decorationValue(components[d], DECO_SPEC_ID);
                        if (specId == null) {
                            throw new IllegalArgumentException(where + ": local-size dimension " + d
                                    + " is a specialisation constant without a SpecId");
                        }
                        size[d] = c.operands[1];
                        specIds[d] = specId;
                    } else {
                        throw new UnsupportedKernelException(where + ": local-size dimension " + d
                                + " is neither a constant nor a plain specialisation constant");
                    }
                }
                return;
            }
            int[] literal = localSizeModes.get(entryId);
            if (literal == null) {
                throw new IllegalArgumentException(where + " declares no local size");
            }
            for (int d = 0; d < 3; d++) {
                size[d] = literal[d];
                specIds[d] = -1;
            }
        }

        private void collectSpecConstants(Map<Integer, Integer> defaults, Map<Integer, Integer> sizes, String where) {
            for (Map.Entry<Integer, Map<Integer, int[]>> e : decorations.entrySet()) {
                int[] specIdOperands = e.getValue().get(DECO_SPEC_ID);
                if (specIdOperands == null || specIdOperands.length == 0) {
                    continue;
                }
                int specId = specIdOperands[0];
                Decl c = constants.get(e.getKey());
                if (c == null) {
                    throw new IllegalArgumentException(where + ": SpecId " + specId
                            + " decorates something other than a specialisation constant");
                }
                int value;
                int size = SpirvModuleInfo.SPEC_SIZE_32;
                if (c.op == OP_SPEC_CONSTANT_TRUE) {
                    value = 1;
                } else if (c.op == OP_SPEC_CONSTANT_FALSE) {
                    value = 0;
                } else if (c.op == OP_SPEC_CONSTANT) {
                    value = c.operands[1];
                    Decl type = types.get(c.operands[0]);
                    if (type != null && (type.op == OP_TYPE_INT || type.op == OP_TYPE_FLOAT)
                            && type.operands[0] == 64) {
                        size = SpirvModuleInfo.SPEC_SIZE_64;
                    }
                } else {
                    throw new IllegalArgumentException(where + ": SpecId " + specId
                            + " decorates a constant with opcode " + c.op);
                }
                if (!defaults.containsKey(specId)) {
                    defaults.put(specId, value);
                    sizes.put(specId, size);
                }
            }
        }

        private boolean hasDecoration(int id, int decoration) {
            Map<Integer, int[]> d = decorations.get(id);
            return d != null && d.containsKey(decoration);
        }

        private Integer decorationValue(int id, int decoration) {
            Map<Integer, int[]> d = decorations.get(id);
            int[] operands = d == null ? null : d.get(decoration);
            return operands == null || operands.length == 0 ? null : operands[0];
        }

        private Integer memberDecorationValue(int structId, int member, int decoration) {
            Map<Integer, Map<Integer, int[]>> byMember = memberDecorations.get(structId);
            Map<Integer, int[]> d = byMember == null ? null : byMember.get(member);
            int[] operands = d == null ? null : d.get(decoration);
            return operands == null || operands.length == 0 ? null : operands[0];
        }

        private boolean allMembersDecorated(int structId, int memberCount, int decoration) {
            Map<Integer, Map<Integer, int[]>> byMember = memberDecorations.get(structId);
            if (byMember == null || memberCount == 0) {
                return false;
            }
            for (int m = 0; m < memberCount; m++) {
                Map<Integer, int[]> d = byMember.get(m);
                if (d == null || !d.containsKey(decoration)) {
                    return false;
                }
            }
            return true;
        }

        private int word(int pos, int index, int count) {
            if (index >= count) {
                throw new IllegalArgumentException("SPIR-V instruction at word " + pos + " has " + count
                        + " words; operand " + index + " is missing");
            }
            return words[pos + index];
        }

        private int[] tail(int pos, int from, int count) {
            if (from > count) {
                throw new IllegalArgumentException("SPIR-V instruction at word " + pos + " has " + count
                        + " words; expected at least " + from);
            }
            int[] out = new int[count - from];
            System.arraycopy(words, pos + from, out, 0, out.length);
            return out;
        }

        private static int[] withType(int type, int[] rest) {
            int[] out = new int[rest.length + 1];
            out[0] = type;
            System.arraycopy(rest, 0, out, 1, rest.length);
            return out;
        }

        /** Decodes a nul-terminated literal string packed little-end-first into words {@code [from, to)}. */
        private String literalString(int from, int to) {
            byte[] chars = new byte[(to - from) * 4];
            int n = 0;
            for (int w = from; w < to; w++) {
                int word = words[w];
                for (int b = 0; b < 4; b++) {
                    byte c = (byte) (word >>> (8 * b));
                    if (c == 0) {
                        return new String(chars, 0, n, StandardCharsets.UTF_8);
                    }
                    chars[n++] = c;
                }
            }
            throw new IllegalArgumentException("unterminated SPIR-V literal string at word " + from);
        }
    }
}
