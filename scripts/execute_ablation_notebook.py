import sys
import nbformat
from nbclient import NotebookClient

path = "frb_ablation_study.ipynb"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="frb-venv", resources={"metadata": {"path": "."}})

try:
    client.execute()
finally:
    nbformat.write(nb, path)
    print("notebook saved (partial or complete)")
