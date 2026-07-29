"""Shared study runtime: checkpoints and the model resume lifecycle.

The event/observer modules that lived here fed the live monitor. The monitor was
removed in the 2026-07-29 cleanup and its plumbing went with it; what remains is
the part the replication depends on -- checkpoint storage and the fixed-JM
resume contract.
"""
