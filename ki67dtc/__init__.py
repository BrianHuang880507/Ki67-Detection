import os

# Windows workaround for OpenMP runtime collision (libomp vs libiomp5md).
# Keep this at package import time so all `ki67dtc.*` entry paths inherit it.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
