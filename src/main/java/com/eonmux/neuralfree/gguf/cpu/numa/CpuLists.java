package com.eonmux.neuralfree.gguf.cpu.numa;

import java.util.ArrayList;
import java.util.List;

/**
 * Parses Linux cpulists such as {@code /sys/devices/system/node/node0/cpulist} or
 * {@code /proc/self/status}'s {@code Cpus_allowed_list}: comma-separated CPU numbers and inclusive ranges
 * ({@code "0-23,48-71"}).
 */
final class CpuLists {

    private CpuLists() {
    }

    /**
     * The CPUs of {@code text}, in increasing order; empty for null, blank or the special value {@code -1} a
     * cpulist with no CPU uses.
     *
     * @throws IllegalArgumentException when a list entry is malformed
     */
    static int[] parse(String text) {
        if (text == null) {
            return new int[0];
        }
        String s = text.trim();
        if (s.isEmpty() || s.equals("-1")) {
            return new int[0];
        }
        List<Integer> cpus = new ArrayList<>();
        for (String part : s.split(",")) {
            String entry = part.trim();
            if (entry.isEmpty()) {
                throw new IllegalArgumentException("empty CPU list entry in '" + text + "'");
            }
            int dash = entry.indexOf('-');
            if (dash < 0) {
                cpus.add(Integer.valueOf(cpu(entry, text)));
            } else {
                int first = cpu(entry.substring(0, dash), text);
                int last = cpu(entry.substring(dash + 1), text);
                if (last < first) {
                    throw new IllegalArgumentException("CPU range '" + entry + "' of '" + text + "' runs backwards");
                }
                for (int cpu = first; cpu <= last; cpu++) {
                    cpus.add(Integer.valueOf(cpu));
                }
            }
        }
        cpus.sort(null);
        int[] out = new int[cpus.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = cpus.get(i).intValue();
        }
        return out;
    }

    /** {@code cpus} (in increasing order) as a cpulist, ranges joined: {@code "0-7,16"}; {@code "none"} when empty. */
    static String format(int[] cpus) {
        if (cpus.length == 0) {
            return "none";
        }
        StringBuilder sb = new StringBuilder();
        int first = cpus[0];
        int last = first;
        for (int i = 1; i <= cpus.length; i++) {
            if (i < cpus.length && cpus[i] == last + 1) {
                last = cpus[i];
                continue;
            }
            if (sb.length() > 0) {
                sb.append(',');
            }
            sb.append(first);
            if (last != first) {
                sb.append('-').append(last);
            }
            if (i < cpus.length) {
                first = cpus[i];
                last = first;
            }
        }
        return sb.toString();
    }

    private static int cpu(String text, String whole) {
        try {
            int cpu = Integer.parseInt(text.trim());
            if (cpu < 0) {
                throw new IllegalArgumentException("CPU " + cpu + " in '" + whole + "' is negative");
            }
            return cpu;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("'" + text + "' in '" + whole + "' is not a CPU number", e);
        }
    }
}
