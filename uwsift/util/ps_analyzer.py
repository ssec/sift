#!/usr/bin/env python
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def format_byte_count(byte_count: int):
    if byte_count == 0:
        return "0 B"

    prefix = ""
    if byte_count < 0:
        prefix = "-"
        byte_count = abs(byte_count)

    symbols = OrderedDict()
    for idx, symbol in enumerate(["B", "KiB", "MiB", "GiB", "TiB", "PiB"]):
        symbols[symbol] = 1 << idx * 10
    for symbol, min_byte_count in reversed(symbols.items()):
        if byte_count >= min_byte_count:
            value = byte_count / min_byte_count
            return f"{prefix}{value:.2f} {symbol}"
    raise RuntimeError(f"can't display the byte count: {byte_count}")


def create_plot(df: pd.DataFrame):
    del df["lib"], df["vms"], df["shared"], df["rss"]

    combined_df = pd.DataFrame(columns=["time", "uss", "pss"])
    for time_step in df["time"]:
        rows = df.loc[df["time"] == time_step]
        entry = {
            "time": time_step,
            "uss": rows["uss"].sum(),
            "pss": rows["pss"].sum(),
        }
        combined_df = combined_df.append(entry, ignore_index=True)

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format_byte_count(int(x))))
    combined_df.plot(x="time", ax=ax)

    plt.xlabel("Time in seconds")
    plt.ylabel("RAM Usage")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("PsAnalyzer")
    parser.add_argument("input")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    create_plot(df)
