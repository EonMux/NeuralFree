package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.batch.SlotResult;
import com.eonmux.neuralfree.gguf.engine.batch.SlotStatus;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The slot operations of one {@link EngineService}: the prefix-cache chain of a pool slot saved to, restored from
 * or erased in a file, and the statuses of every slot of the pool (llama.cpp's slot save, restore, erase and
 * {@code GET /slots}). Slot ids run over the pool replica by replica; each action runs on the owning replica's pump
 * thread once its engine is idle. The service owns the replicas and resolves a pool slot to its replica and
 * sequence.
 */
final class EngineSlots {

    private final EngineService owner;

    EngineSlots(EngineService owner) {
        this.owner = Objects.requireNonNull(owner, "owner");
    }

    /**
     * Writes slot {@code slot}'s state to {@code file}: the prefix-cache chain the slot's last sequence left in its
     * replica's batch engine.
     *
     * @return the tokens saved and the bytes written
     * @throws EngineService.UnknownSlotException for a slot id outside the pool's slots
     * @throws IOException                        when the write fails or the wait outlives the request budget
     * @throws UnsupportedOperationException      when the engine runs no prefix cache or its memory's blocks do not
     *                                            hold a sequence's whole state
     */
    SlotResult saveSlot(int slot, Path file) throws IOException {
        SlotTarget target = slotTarget(slot);
        return target.replica().onIdleEngine(() -> target.replica().batchEngine().saveSlot(target.seq(), file));
    }

    /**
     * Reads the slot file at {@code file} into slot {@code slot}: its blocks join the replica's prefix cache, so the
     * next request whose prompt starts with the restored tokens adopts their rows.
     *
     * @return the tokens restored and the bytes read
     * @throws EngineService.UnknownSlotException for a slot id outside the pool's slots
     * @throws IOException                        for an unreadable or foreign file, or a wait past the budget
     * @throws IllegalStateException              when no KV page can be freed for the restored blocks
     * @throws UnsupportedOperationException      as {@link #saveSlot}
     */
    SlotResult restoreSlot(int slot, Path file) throws IOException {
        SlotTarget target = slotTarget(slot);
        return target.replica().onIdleEngine(() -> target.replica().batchEngine().restoreSlot(target.seq(), file));
    }

    /**
     * Erases slot {@code slot}'s state: forgets its history and evicts the cached blocks no other slot walks.
     *
     * @return the tokens the slot held
     * @throws EngineService.UnknownSlotException for a slot id outside the pool's slots
     * @throws IOException                        when the wait outlives the request budget
     * @throws UnsupportedOperationException      as {@link #saveSlot}
     */
    int eraseSlot(int slot) throws IOException {
        SlotTarget target = slotTarget(slot);
        return target.replica().onIdleEngine(() -> target.replica().batchEngine().eraseSlot(target.seq()));
    }

    /**
     * The status of every slot of the pool, replica by replica in pool order; a replica whose engine is not built
     * holds none.
     *
     * @throws IllegalStateException when a pump did not answer within the request budget
     */
    List<SlotStatus> slotStatuses() throws InterruptedException {
        List<SlotStatus> out = new ArrayList<>();
        for (EngineReplica replica : owner.replicas()) {
            out.addAll(replica.slotStatusesOp());
        }
        return out;
    }

    /** The replica and its own sequence id a pool slot id names. */
    private record SlotTarget(EngineReplica replica, int seq) { }

    /**
     * The replica holding pool slot {@code slot} and the slot's sequence id there, counting each replica's slots in
     * pool order; an engine no request built yet builds on its pump thread.
     *
     * @throws EngineService.UnknownSlotException for a slot id outside the pool's slots
     */
    private SlotTarget slotTarget(int slot) throws IOException {
        owner.primary();
        int base = 0;
        for (EngineReplica replica : owner.replicas()) {
            int n = replica.onIdleEngine(() -> replica.batchEngine().maxSequences());
            if (slot >= base && slot < base + n) {
                return new SlotTarget(replica, slot - base);
            }
            base += n;
        }
        throw new EngineService.UnknownSlotException(slot, base);
    }
}
