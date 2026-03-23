"""
python -m timecapsule

Enables:
    python -m timecapsule recording.tc
    python -m timecapsule recording.tc --callgraph
    python -m timecapsule recording.tc --for-claude
    python -m timecapsule --speedtest
"""
from timecapsule.recorder import _cli

if __name__ == "__main__":
    _cli()
