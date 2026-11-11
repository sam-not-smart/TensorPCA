"""Run every sales walkthrough cell in a fresh kernel without changing the file."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(__file__).resolve().parents[1] / "applications/sales/synthetic_sales_walkthrough.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
NotebookClient(
    notebook, timeout=300, kernel_name="python3",
    resources={"metadata": {"path": str(path.parent)}},
).execute()
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
print(f"Passed: all {len(code_cells)} notebook code cells executed in a fresh Python kernel.")
