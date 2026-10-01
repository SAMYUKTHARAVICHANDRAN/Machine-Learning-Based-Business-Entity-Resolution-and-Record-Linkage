"""Execute TRIDENT_pipeline.ipynb in a fresh kernel, saving after every cell and echoing
cell outputs to stdout (so long runs can be followed and resumed)."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NB = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "TRIDENT_pipeline.ipynb")
nb = nbformat.read(NB, as_version=4)
t0 = time.time()


def on_done(cell, cell_index, **kw):
    for o in cell.get("outputs", []):
        txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) or ""
        if o.get("output_type") == "error":
            txt = "\n".join(o.get("traceback", []))
        if txt:
            print(txt[-4000:], flush=True)
    print(f"=== cell {cell_index} done ({time.time() - t0:.0f}s) ===", flush=True)
    nbformat.write(nb, NB)


client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}},
                        on_cell_executed=on_done)
try:
    client.execute()
    print("NOTEBOOK COMPLETED", flush=True)
except Exception as e:
    print("NOTEBOOK FAILED:", type(e).__name__, str(e)[-3000:], flush=True)
finally:
    nbformat.write(nb, NB)
