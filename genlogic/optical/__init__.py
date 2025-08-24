#!/usr/bin/env python3

# vim: autoindent noexpandtab tabstop=4 shiftwidth=4

import matplotlib.pyplot as plt  # type: ignore[import-untyped]

from . import caching, classes, io, plotting, utils

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "grid.linewidth": 0.25,
    "figure.max_open_warning": 0,
})
