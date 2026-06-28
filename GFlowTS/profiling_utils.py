import contextlib
import datetime as dt
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from typing import Dict, Iterator, Optional, Tuple


_ACTIVE_COLLECTOR = None


class ProfileCollector:
    def __init__(self, script_name: str, label: str = ""):
        self.script_name = str(script_name)
        self.label = str(label or "")
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self._wall_start = time.perf_counter()
        self._records: "OrderedDict[Tuple[str, str, str], Dict[str, float]]" = OrderedDict()
        self._counters: "OrderedDict[str, int]" = OrderedDict()
        self.metadata: Dict[str, object] = {}

    def set_metadata(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.metadata[key] = value

    @contextlib.contextmanager
    def scope(self, phase: str, major_module: str, minor_module: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.add_time(phase, major_module, minor_module, elapsed)

    def add_time(
        self,
        phase: str,
        major_module: str,
        minor_module: str,
        elapsed_s: float,
        calls: int = 1,
    ) -> None:
        key = (str(phase), str(major_module), str(minor_module))
        rec = self._records.get(key)
        if rec is None:
            rec = {
                "phase": str(phase),
                "major_module": str(major_module),
                "minor_module": str(minor_module),
                "total_s": 0.0,
                "calls": 0,
            }
            self._records[key] = rec
        rec["total_s"] += float(max(0.0, elapsed_s))
        rec["calls"] += int(max(0, calls))

    def wall_time_s(self) -> float:
        return float(time.perf_counter() - self._wall_start)

    def increment_counter(self, name: str, value: int = 1) -> None:
        key = str(name)
        self._counters[key] = int(self._counters.get(key, 0)) + int(value)

    def to_dict(self) -> Dict[str, object]:
        timers = []
        for rec in self._records.values():
            total_s = float(rec["total_s"])
            calls = int(rec["calls"])
            timers.append(
                {
                    "phase": rec["phase"],
                    "major_module": rec["major_module"],
                    "minor_module": rec["minor_module"],
                    "total_s": total_s,
                    "calls": calls,
                    "avg_s": float(total_s / max(1, calls)),
                }
            )
        return {
            "script_name": self.script_name,
            "label": self.label,
            "started_at": self.started_at,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wall_s": self.wall_time_s(),
            "metadata": self.metadata,
            "timers": timers,
            "counters": dict(self._counters),
        }

    def save(self, path: str) -> str:
        out_path = os.path.abspath(path)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return out_path


def set_active_collector(collector: Optional[ProfileCollector]) -> None:
    global _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector


def get_active_collector() -> Optional[ProfileCollector]:
    return _ACTIVE_COLLECTOR


@contextlib.contextmanager
def profile_scope(phase: str, major_module: str, minor_module: str) -> Iterator[None]:
    collector = get_active_collector()
    if collector is None:
        yield
        return
    with collector.scope(phase=phase, major_module=major_module, minor_module=minor_module):
        yield


def init_profile_collector(
    profile_json: str,
    script_name: str,
    profile_label: str = "",
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Optional[ProfileCollector]:
    if not profile_json:
        set_active_collector(None)
        return None
    collector = ProfileCollector(script_name=script_name, label=profile_label)
    collector.set_metadata(
        command_line=list(sys.argv),
        cwd=os.getcwd(),
        python_executable=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
    )
    if extra_metadata:
        collector.set_metadata(**extra_metadata)
    set_active_collector(collector)
    return collector


def increment_profile_counter(name: str, value: int = 1) -> None:
    collector = get_active_collector()
    if collector is None:
        return
    collector.increment_counter(name=name, value=value)


def finalize_profile(profile_json: str) -> Optional[str]:
    collector = get_active_collector()
    if collector is None or (not profile_json):
        set_active_collector(None)
        return None
    try:
        return collector.save(profile_json)
    finally:
        set_active_collector(None)
