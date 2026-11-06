"""Run the app modules without a GPU stack: ctranslate2 is stubbed so tests
only need fastapi/pydantic/numpy; torch and pyannote are imported lazily by
the code under test and never reached here."""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

os.environ.setdefault("WARMUP_ON_START", "false")

if "ctranslate2" not in sys.modules:
    stub = types.ModuleType("ctranslate2")
    stub.get_cuda_device_count = lambda: 0
    sys.modules["ctranslate2"] = stub
