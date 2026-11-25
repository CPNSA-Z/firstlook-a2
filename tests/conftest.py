import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/fake_backend.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the app package
