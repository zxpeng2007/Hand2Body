"""Hand2Body: generate whole-body SMPL motion from sparse wrist tracking.

A general causal *sparse-tracking -> full-body* model: drive it from one wrist (12D) or both
wrists (24D) and it lifts the wrist signal(s) to the whole body in real time. Two example
applications:
  1-wrist (table tennis):  paddle-hand generator -> Hand2Body -> GMR retarget -> HoloMotion (Unitree G1).
  2-wrist (manipulation):  ARCTIC-style bimanual tracking -> Hand2Body -> whole-body SMPL.
See docs/CONTRACT.md for the (table-tennis) inter-stage data contract.
"""

__version__ = "0.2.0"
