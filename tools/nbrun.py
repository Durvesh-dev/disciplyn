"""Minimal notebook builder+executor (no jupyter needed): runs cells in one namespace and
stores stdout, DataFrame HTML and matplotlib figures as real .ipynb outputs."""
import ast
import base64
import contextlib
import io
import json
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run(cells, out_path, workdir):
    import os
    os.chdir(workdir)
    sys.path.insert(0, workdir)
    ns = {"__name__": "__main__"}
    nb_cells, count = [], 0
    for kind, src in cells:
        src = src.strip("\n")
        if kind == "md":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
            continue
        count += 1
        outputs, buf = [], io.StringIO()
        t0 = time.time()
        print(f"[cell {count}] running...", file=sys.__stdout__, flush=True)
        try:
            tree = ast.parse(src)
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
            with contextlib.redirect_stdout(buf):
                exec(compile(body, f"<cell{count}>", "exec"), ns)
                val = eval(compile(ast.Expression(last.value), f"<cell{count}>", "eval"), ns) if last else None
        except Exception:
            val = None
            outputs.append({"output_type": "error", "ename": "Error", "evalue": "",
                            "traceback": traceback.format_exc().splitlines()})
            print(traceback.format_exc(), file=sys.__stdout__)
        text = buf.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text.splitlines(True)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        if val is not None:
            data = {"text/plain": repr(val).splitlines(True)}
            if hasattr(val, "_repr_html_"):
                try:
                    data["text/html"] = val._repr_html_().splitlines(True)
                except Exception:
                    pass
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
        print(f"[cell {count}] done in {time.time()-t0:.0f}s", file=sys.__stdout__, flush=True)
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {},
                         "outputs": outputs, "source": src.splitlines(True)})
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}}}
    json.dump(nb, open(out_path, "w"), indent=1)
    return ns
