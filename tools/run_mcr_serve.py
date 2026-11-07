#!/usr/bin/env python3
"""One-command launcher for the Botzone MCR evaluation TCP server."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tools.mcr_serve import serve

if __name__ == "__main__":
    ckpt = ROOT / "baselines" / "botzone_mcr" / "botzone_model_bc.pt"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18766
    serve(port, str(ckpt))
