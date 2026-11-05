import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
