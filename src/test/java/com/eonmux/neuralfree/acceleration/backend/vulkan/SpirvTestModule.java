package com.eonmux.neuralfree.acceleration.backend.vulkan;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Assembles small SPIR-V compute modules for reflection tests: the module preamble in the order the
 * specification requires (capabilities, memory model, entry point, execution modes, annotations,
 * types, constants and global variables), followed by one {@code OpFunction}, where reflection stops.
 * Instruction words are {@code (wordCount << 16) | opcode}, then the operands.
 */
final class SpirvTestModule {

    static final int MAGIC = 0x07230203;
    static final int VERSION_1_5 = 0x00010500;

    static final int CAP_SHADER = 1;

    static final int SC_UNIFORM = 2;
    static final int SC_PUSH_CONSTANT = 9;
    static final int SC_STORAGE_BUFFER = 12;

    static final int DECO_SPEC_ID = 1;
    static final int DECO_BLOCK = 2;
    static final int DECO_ARRAY_STRIDE = 6;
    static final int DECO_BUILT_IN = 11;
    static final int DECO_NON_WRITABLE = 24;
    static final int DECO_NON_READABLE = 25;
    static final int DECO_BINDING = 33;
    static final int DECO_DESCRIPTOR_SET = 34;
    static final int DECO_OFFSET = 35;

    static final int BUILT_IN_WORKGROUP_SIZE = 25;

    private static final int OP_MEMORY_MODEL = 14;
    private static final int OP_ENTRY_POINT = 15;
    private static final int OP_EXECUTION_MODE = 16;
    private static final int OP_CAPABILITY = 17;
    private static final int OP_TYPE_VOID = 19;
    private static final int OP_TYPE_INT = 21;
    private static final int OP_TYPE_FLOAT = 22;
    private static final int OP_TYPE_VECTOR = 23;
    private static final int OP_TYPE_RUNTIME_ARRAY = 29;
    private static final int OP_TYPE_STRUCT = 30;
    private static final int OP_TYPE_POINTER = 32;
    private static final int OP_TYPE_FUNCTION = 33;
    private static final int OP_CONSTANT = 43;
    private static final int OP_SPEC_CONSTANT = 50;
    private static final int OP_SPEC_CONSTANT_COMPOSITE = 51;
    private static final int OP_FUNCTION = 54;
    private static final int OP_VARIABLE = 59;
    private static final int OP_DECORATE = 71;
    private static final int OP_MEMBER_DECORATE = 72;
    private static final int OP_EXECUTION_MODE_ID = 331;

    private static final int ADDRESSING_LOGICAL = 0;
    private static final int MEMORY_MODEL_GLSL450 = 1;
    private static final int EXEC_MODEL_GL_COMPUTE = 5;
    private static final int EXEC_MODE_LOCAL_SIZE = 17;
    private static final int EXEC_MODE_LOCAL_SIZE_ID = 38;

    private final List<int[]> capabilities = new ArrayList<>();
    private final List<int[]> modes = new ArrayList<>();
    private final List<int[]> annotations = new ArrayList<>();
    private final List<int[]> globals = new ArrayList<>();
    private final String entryName;
    private final int main;
    private final int voidType;
    private final int functionType;
    private int nextId = 1;

    SpirvTestModule(String entryName) {
        this.entryName = entryName;
        this.main = id();
        this.voidType = id();
        this.functionType = id();
        capability(CAP_SHADER);
        globals.add(instr(OP_TYPE_VOID, voidType));
        globals.add(instr(OP_TYPE_FUNCTION, functionType, voidType));
    }

    int id() {
        return nextId++;
    }

    SpirvTestModule capability(int capability) {
        capabilities.add(instr(OP_CAPABILITY, capability));
        return this;
    }

    /** {@code OpExecutionMode LocalSize x y z}. */
    SpirvTestModule localSize(int x, int y, int z) {
        modes.add(instr(OP_EXECUTION_MODE, main, EXEC_MODE_LOCAL_SIZE, x, y, z));
        return this;
    }

    /** {@code OpExecutionModeId LocalSizeId x y z} with constant ids. */
    SpirvTestModule localSizeId(int x, int y, int z) {
        modes.add(instr(OP_EXECUTION_MODE_ID, main, EXEC_MODE_LOCAL_SIZE_ID, x, y, z));
        return this;
    }

    SpirvTestModule decorate(int target, int decoration, int... values) {
        int[] operands = new int[2 + values.length];
        operands[0] = target;
        operands[1] = decoration;
        System.arraycopy(values, 0, operands, 2, values.length);
        annotations.add(instr(OP_DECORATE, operands));
        return this;
    }

    SpirvTestModule memberDecorate(int struct, int member, int decoration, int... values) {
        int[] operands = new int[3 + values.length];
        operands[0] = struct;
        operands[1] = member;
        operands[2] = decoration;
        System.arraycopy(values, 0, operands, 3, values.length);
        annotations.add(instr(OP_MEMBER_DECORATE, operands));
        return this;
    }

    int typeInt(int width, boolean signed) {
        int id = id();
        globals.add(instr(OP_TYPE_INT, id, width, signed ? 1 : 0));
        return id;
    }

    int typeFloat(int width) {
        int id = id();
        globals.add(instr(OP_TYPE_FLOAT, id, width));
        return id;
    }

    int typeVector(int component, int count) {
        int id = id();
        globals.add(instr(OP_TYPE_VECTOR, id, component, count));
        return id;
    }

    int typeRuntimeArray(int element, int stride) {
        int id = id();
        globals.add(instr(OP_TYPE_RUNTIME_ARRAY, id, element));
        decorate(id, DECO_ARRAY_STRIDE, stride);
        return id;
    }

    int typeStruct(int... members) {
        int id = id();
        int[] operands = new int[1 + members.length];
        operands[0] = id;
        System.arraycopy(members, 0, operands, 1, members.length);
        globals.add(instr(OP_TYPE_STRUCT, operands));
        return id;
    }

    int typePointer(int storageClass, int type) {
        int id = id();
        globals.add(instr(OP_TYPE_POINTER, id, storageClass, type));
        return id;
    }

    int constant(int type, int value) {
        int id = id();
        globals.add(instr(OP_CONSTANT, type, id, value));
        return id;
    }

    /** {@code OpSpecConstant} decorated with {@code SpecId specId}. */
    int specConstant(int type, int value, int specId) {
        int id = id();
        globals.add(instr(OP_SPEC_CONSTANT, type, id, value));
        decorate(id, DECO_SPEC_ID, specId);
        return id;
    }

    /** {@code OpSpecConstantComposite} decorated {@code BuiltIn WorkgroupSize}. */
    int workgroupSize(int vectorType, int x, int y, int z) {
        int id = id();
        globals.add(instr(OP_SPEC_CONSTANT_COMPOSITE, vectorType, id, x, y, z));
        decorate(id, DECO_BUILT_IN, BUILT_IN_WORKGROUP_SIZE);
        return id;
    }

    int variable(int pointerType, int storageClass) {
        int id = id();
        globals.add(instr(OP_VARIABLE, pointerType, id, storageClass));
        return id;
    }

    /**
     * One storage-buffer view {@code buffer { T data[]; }} at {@code (set, binding)}.
     *
     * @param varDecorations decorations put on the variable ({@code NonWritable}, {@code NonReadable})
     * @param memberDecorations decorations put on every member of the block
     * @return the variable id
     */
    int storageBuffer(int set, int binding, int element, int stride, int[] varDecorations,
                      int[] memberDecorations) {
        int array = typeRuntimeArray(element, stride);
        int block = typeStruct(array);
        decorate(block, DECO_BLOCK);
        memberDecorate(block, 0, DECO_OFFSET, 0);
        for (int d : memberDecorations) {
            memberDecorate(block, 0, d);
        }
        int pointer = typePointer(SC_STORAGE_BUFFER, block);
        int var = variable(pointer, SC_STORAGE_BUFFER);
        decorate(var, DECO_DESCRIPTOR_SET, set);
        decorate(var, DECO_BINDING, binding);
        for (int d : varDecorations) {
            decorate(var, d);
        }
        return var;
    }

    /** The push-constant block with these member types at offsets {@code 4 * i}. */
    int pushBlock(int... memberTypes) {
        int block = typeStruct(memberTypes);
        decorate(block, DECO_BLOCK);
        for (int m = 0; m < memberTypes.length; m++) {
            memberDecorate(block, m, DECO_OFFSET, 4 * m);
        }
        return variable(typePointer(SC_PUSH_CONSTANT, block), SC_PUSH_CONSTANT);
    }

    /** The module as words. */
    int[] words() {
        List<int[]> all = new ArrayList<>(capabilities);
        all.add(instr(OP_MEMORY_MODEL, ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450));
        all.add(entryPoint());
        all.addAll(modes);
        all.addAll(annotations);
        all.addAll(globals);
        all.add(instr(OP_FUNCTION, voidType, main, 0, functionType));
        int total = 5;
        for (int[] i : all) {
            total += i.length;
        }
        int[] out = new int[total];
        out[0] = MAGIC;
        out[1] = VERSION_1_5;
        out[2] = 0;
        out[3] = nextId;
        out[4] = 0;
        int pos = 5;
        for (int[] i : all) {
            System.arraycopy(i, 0, out, pos, i.length);
            pos += i.length;
        }
        return out;
    }

    /** The module as bytes in the given order. */
    byte[] bytes(ByteOrder order) {
        int[] w = words();
        ByteBuffer b = ByteBuffer.allocate(w.length * 4).order(order);
        for (int word : w) {
            b.putInt(word);
        }
        return b.array();
    }

    byte[] bytes() {
        return bytes(ByteOrder.LITTLE_ENDIAN);
    }

    private int[] entryPoint() {
        byte[] name = entryName.getBytes(StandardCharsets.UTF_8);
        int nameWords = name.length / 4 + 1;
        int[] operands = new int[2 + nameWords];
        operands[0] = EXEC_MODEL_GL_COMPUTE;
        operands[1] = main;
        for (int i = 0; i < name.length; i++) {
            operands[2 + i / 4] |= (name[i] & 0xFF) << (8 * (i % 4));
        }
        return instr(OP_ENTRY_POINT, operands);
    }

    private static int[] instr(int opcode, int... operands) {
        int[] w = new int[operands.length + 1];
        w[0] = ((operands.length + 1) << 16) | opcode;
        System.arraycopy(operands, 0, w, 1, operands.length);
        return w;
    }
}
