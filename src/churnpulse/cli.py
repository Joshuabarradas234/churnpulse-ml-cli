from __future__ import annotations

import typer
from pathlib import Path
from churnpulse.train import run_training

app = typer.Typer()

@app.command()
def train(csv: str = typer.Option("", "--csv")):
    path = Path(csv) if csv else None
    run_training(path)

def main():
    app()

if __name__ == "__main__":
    main()
