"""Execute real notebook cells without opening kernel sockets.

Useful in restricted environments. IPython executes every code cell in order;
rich display data, stdout, and errors are captured in standard notebook format.
This does not synthesize outputs or skip model phases: the notebook's explicit
FORCE_REBUILD setting controls whether those phases reuse saved artifacts.
"""
from pathlib import Path
import os
import sys
import traceback

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
path = ROOT / "ReadmitRisk_Notebook.ipynb"
notebook = nbformat.read(path, as_version=4)
shell = InteractiveShell.instance()
count = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    cell.outputs = []
    with capture_output(stdout=True, stderr=True, display=True) as captured:
        execution = shell.run_cell(cell.source, store_history=True, silent=False)
    cell.execution_count = count
    for name in ["stdout", "stderr"]:
        content = getattr(captured, name)
        if content:
            cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=content))
    for output in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    error = execution.error_before_exec or execution.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__,
            evalue=str(error), traceback=traceback.format_exception(error)))
        nbformat.write(notebook, path)
        raise RuntimeError(f"Notebook cell {count} failed") from error
    print(f"Executed notebook code cell {count}.", flush=True)
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"All {count} code cells executed; outputs saved to {path.name}.")
