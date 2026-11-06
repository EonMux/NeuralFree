package neuralfree.gpu.metal;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.spvc.Spvc;
import org.lwjgl.util.spvc.SpvcMslResourceBinding;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

/**
 * Generates Metal Shading Language from one Vulkan SPIR-V compute module with SPIRV-Cross
 * (LWJGL's {@code spvc} binding).
 *
 * <p>The resource-binding remap is the contract of the Metal backend: a storage buffer with
 * {@code Binding b} in descriptor set 0 becomes MSL buffer index {@code b}, and the push-constant
 * block becomes MSL buffer index {@code bufferCount}. The entry point {@code main} is renamed to
 * the kernel name, so one module becomes one kernel function. Specialisation constants stay
 * function constants at their {@code SpecId}, which the backend sets when it builds a pipeline.</p>
 */
final class SpirvCrossMsl {

    /** {@code spv::ExecutionModel::GLCompute}. */
    private static final int EXECUTION_MODEL_GL_COMPUTE = 5;
    /** {@code SpvVersion} of MSL 2.0 (macOS 10.13 and later). */
    private static final int MSL_VERSION_2_0 = 0x00020000;

    /**
     * Compiles one SPIR-V module to MSL.
     *
     * @param spirv        the SPIR-V words, in native byte order
     * @param kernelName   entry point of the module and name of the generated kernel function
     * @param bufferCount  number of storage-buffer bindings, {@code 0..bufferCount-1}
     * @param pushConstant whether the module has a push-constant block (bound at {@code bufferCount})
     * @return the complete MSL translation unit
     * @throws IllegalStateException with SPIRV-Cross's message when the module cannot be translated
     */
    String compile(byte[] spirv, String kernelName, int bufferCount, boolean pushConstant) {
        if (spirv == null || spirv.length % 4 != 0) {
            throw new IllegalArgumentException("SPIR-V of " + kernelName + " is not a whole number of words");
        }
        IntBuffer words = null;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer contextOut = stack.mallocPointer(1);
            check(Spvc.spvc_context_create(contextOut), "create the SPIRV-Cross context", -1);
            long context = contextOut.get(0);
            try {
                words = MemoryUtil.memAllocInt(spirv.length / 4);
                words.put(ByteBuffer.wrap(spirv).order(ByteOrder.nativeOrder()).asIntBuffer());
                words.flip();
                PointerBuffer moduleOut = stack.mallocPointer(1);
                check(Spvc.spvc_context_parse_spirv(context, words, words.remaining(), moduleOut),
                        "parse the SPIR-V module", context);
                PointerBuffer compilerOut = stack.mallocPointer(1);
                check(Spvc.spvc_context_create_compiler(context, Spvc.SPVC_BACKEND_MSL, moduleOut.get(0),
                        Spvc.SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, compilerOut), "create the MSL compiler", context);
                long compiler = compilerOut.get(0);
                configure(compiler, context, stack);
                remap(compiler, context, stack, bufferCount, pushConstant);
                check(Spvc.spvc_compiler_rename_entry_point(compiler, "main", kernelName,
                        EXECUTION_MODEL_GL_COMPUTE), "rename the entry point to " + kernelName, context);
                PointerBuffer sourceOut = stack.mallocPointer(1);
                check(Spvc.spvc_compiler_compile(compiler, sourceOut), "translate the module to MSL", context);
                return sourceOut.getStringUTF8(0);
            } finally {
                Spvc.spvc_context_destroy(context);
            }
        } finally {
            if (words != null) {
                MemoryUtil.memFree(words);
            }
        }
    }

    private void configure(long compiler, long context, MemoryStack stack) {
        PointerBuffer optionsOut = stack.mallocPointer(1);
        check(Spvc.spvc_compiler_create_compiler_options(compiler, optionsOut),
                "create the compiler options", context);
        long options = optionsOut.get(0);
        Spvc.spvc_compiler_options_set_uint(options, Spvc.SPVC_COMPILER_OPTION_MSL_VERSION, MSL_VERSION_2_0);
        Spvc.spvc_compiler_options_set_bool(options, Spvc.SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, false);
        Spvc.spvc_compiler_options_set_uint(options, Spvc.SPVC_COMPILER_OPTION_MSL_PLATFORM,
                Spvc.SPVC_MSL_PLATFORM_MACOS);
        check(Spvc.spvc_compiler_install_compiler_options(compiler, options),
                "install the compiler options", context);
    }

    private void remap(long compiler, long context, MemoryStack stack, int bufferCount, boolean pushConstant) {
        try (MemoryStack bindings = MemoryStack.stackPush()) {
            for (int binding = 0; binding < bufferCount; binding++) {
                SpvcMslResourceBinding resource = SpvcMslResourceBinding.malloc(bindings);
                resource.set(EXECUTION_MODEL_GL_COMPUTE, 0, binding, binding, 0, 0);
                check(Spvc.spvc_compiler_msl_add_resource_binding(compiler, resource),
                        "bind storage buffer " + binding + " to MSL buffer " + binding, context);
            }
            if (pushConstant) {
                SpvcMslResourceBinding resource = SpvcMslResourceBinding.malloc(bindings);
                resource.set(EXECUTION_MODEL_GL_COMPUTE, Spvc.SPVC_MSL_PUSH_CONSTANT_DESC_SET,
                        Spvc.SPVC_MSL_PUSH_CONSTANT_BINDING, bufferCount, 0, 0);
                check(Spvc.spvc_compiler_msl_add_resource_binding(compiler, resource),
                        "bind the push constants to MSL buffer " + bufferCount, context);
            }
        }
    }

    private static void check(int result, String what, long context) {
        if (result == Spvc.SPVC_SUCCESS) {
            return;
        }
        String message = context < 0 ? null : Spvc.spvc_context_get_last_error_string(context);
        throw new IllegalStateException("SPIRV-Cross could not " + what + " (error " + result + ")"
                + (message == null || message.isEmpty() ? "" : ": " + message));
    }
}
