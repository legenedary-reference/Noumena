"""Noumena: a universal lens for machine perception.

    ring    zero-copy shared memory ring buffer
    forms   the a priori forms: time, space, substance, cause
    lens    modality adapters and the packet the model receives
    eye     the runtime: daemons, sync, triggers, supervision, dispatch
    sensors simulated sources, so everything runs without hardware
"""

from .eye import Eye, OnEvent, OnStream, Percept, Periodic, SensorSpec, SummaryModel, Trigger
from .forms import BodyGrid, Event, Forms, MonotonicTime, Pose, RefractoryCause, RunningBaseline
from .lens import Lens, PerceptionPacket, RawSample
from .ring import RingReader, SharedRing, SlotRef

__all__ = [
    "BodyGrid", "Event", "Eye", "Forms", "Lens", "MonotonicTime", "OnEvent", "OnStream",
    "Percept", "PerceptionPacket", "Periodic", "Pose", "RawSample", "RefractoryCause",
    "RingReader", "RunningBaseline", "SensorSpec", "SharedRing", "SlotRef", "SummaryModel", "Trigger",
]
