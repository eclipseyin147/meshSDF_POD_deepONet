#!/usr/bin/env python3
"""Runtime compatibility shim for the vendored PhysicsNeMo 2.3.0a0 tree on
torch 2.5 (spec
docs/superpowers/specs/2026-09-15-deepsdf-physicsnemo-deeponet-mvp-design.md
section 3). Import this module before any ``import physicsnemo...``.

Effects: (1) inserts ``third-party/physicsnemo`` into ``sys.path``;
(2) aliases ``torch.Tag.cudagraph_unsafe`` (absent before torch 2.10) to
``nondeterministic_bitwise`` so module-level attribute reads succeed;
(3) wraps ``torch.library.custom_op`` to drop the ``tags`` kwarg that
torch 2.5 does not accept. The vendored source stays unmodified.
"""

import inspect
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PHYSICSNEMO = os.path.join(_REPO_ROOT, "third-party", "physicsnemo")
if os.path.isdir(_PHYSICSNEMO) and _PHYSICSNEMO not in sys.path:
    sys.path.insert(0, _PHYSICSNEMO)

if not hasattr(torch.Tag, "cudagraph_unsafe"):
    torch.Tag.cudagraph_unsafe = torch.Tag.nondeterministic_bitwise

if "tags" not in inspect.signature(torch.library.custom_op).parameters:
    _orig_custom_op = torch.library.custom_op

    def _custom_op_drop_tags(*args, **kwargs):
        kwargs.pop("tags", None)
        return _orig_custom_op(*args, **kwargs)

    torch.library.custom_op = _custom_op_drop_tags
