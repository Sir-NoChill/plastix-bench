# Marks baselines/ as a package so `from baselines.common import ...` resolves
# from the run_baselines.py driver. The per-framework stubs under baselines/<fw>/
# import `common` directly (they put baselines/ on sys.path themselves).
