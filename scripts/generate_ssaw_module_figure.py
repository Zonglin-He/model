from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


def add_box(ax, x, y, w, h, title, lines, facecolor):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#3a3a3a",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.045, title, fontsize=12.5, fontweight="bold", va="top", color="#111111")
    ax.text(x + 0.018, y + h - 0.10, "\n".join(lines), fontsize=10.0, va="top", color="#222222", linespacing=1.35)


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.7,
            color="#666666",
        )
    )


def draw_wave(ax, x0, y0, w, h, color, mode="raw", linewidth=1.9):
    xs = np.linspace(0, 1, 300)
    base = 0.52 * np.sin(2 * np.pi * 2.2 * xs) + 0.16 * np.sin(2 * np.pi * 5.5 * xs + 0.4)
    if mode == "smooth":
        env = 1.0 + 0.22 * np.sin(2 * np.pi * 1.0 * xs - 0.7)
        ys = base * env
    elif mode == "noisy":
        ys = base + 0.18 * np.sign(np.sin(2 * np.pi * 18 * xs))
    else:
        ys = base
    ax.plot(x0 + xs * w, y0 + 0.5 * h + ys * 0.38 * h, color=color, linewidth=linewidth, clip_on=False)


def draw_spline(ax, x0, y0, w, h):
    ctrl_x = np.linspace(0.08, 0.92, 6)
    ctrl_y = np.array([0.45, 0.62, 0.79, 0.68, 0.49, 0.56])
    xs = np.linspace(0.08, 0.92, 200)
    ys = np.interp(xs, ctrl_x, ctrl_y)
    ax.plot(x0 + xs * w, y0 + ys * h, color="#d62728", linewidth=2.0)
    ax.scatter(x0 + ctrl_x * w, y0 + ctrl_y * h, s=26, color="#111111", zorder=3)
    ax.text(x0 + 0.01 * w, y0 + 0.83 * h, r"$\mathbf{k}^{(i)} \in [1-\sigma, 1+\sigma]^M$", fontsize=10, color="#222222")
    ax.text(x0 + 0.01 * w, y0 - 0.06 * h, r"cubic-spline scaling curve $w^{(i)}(\tau)$", fontsize=9.6, color="#444444")


def draw_candidates(ax, x0, y0, w, h):
    rows = [0.74, 0.50, 0.26]
    colors = ["#9e9e9e", "#b8b8b8", "#d62728"]
    labels = [r"$x_t^{(1)}$", r"$x_t^{(2)}$", r"$x_t^{(N_{cand})}$"]
    for yy, color, label in zip(rows, colors, labels):
        draw_wave(ax, x0 + 0.18 * w, y0 + yy * h, 0.66 * w, 0.13 * h, color=color, mode="smooth", linewidth=1.8)
        ax.text(x0 + 0.02 * w, y0 + yy * h + 0.01, label, fontsize=9.8, color="#222222")
    ax.text(x0 + 0.02 * w, y0 + 0.06 * h, r"$x_t^{(i)} = x_t \odot w^{(i)}(\tau)$", fontsize=10.2, color="#222222")


def main():
    out_pdf = ROOT / "ssaw_module_diagram.pdf"
    out_png = ROOT / "ssaw_module_diagram.png"

    fig = plt.figure(figsize=(14.2, 6.0))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.035, 0.945, "Stochastic Smooth Adversarial Warping (SSAW)", fontsize=19, fontweight="bold", ha="left")
    fig.text(
        0.035,
        0.900,
        "A local augmentation module that searches bounded smooth drifts and returns the worst-case physically plausible view.",
        fontsize=11.5,
        ha="left",
        color="#333333",
    )

    y = 0.47
    h = 0.34
    w = 0.205
    xs = [0.035, 0.285, 0.535, 0.785]

    add_box(
        ax,
        xs[0],
        y,
        w,
        h,
        "1. Raw window and perturbation goal",
        [
            r"Input raw segment $x_t$.",
            r"Target: smooth sensor-like drift,",
            r"not PGD-style high-frequency noise.",
        ],
        facecolor="#eef4fb",
    )
    draw_wave(ax, xs[0] + 0.03, y + 0.06, 0.14, 0.08, color="#1f77b4", mode="raw")
    draw_wave(ax, xs[0] + 0.03, y + 0.20, 0.14, 0.08, color="#8a8a8a", mode="noisy")
    ax.text(xs[0] + 0.03, y + 0.02, "raw input", fontsize=9.3, color="#1f77b4")
    ax.text(xs[0] + 0.03, y + 0.30, "avoid unrealistic sawtooth artifacts", fontsize=8.7, color="#555555")

    add_box(
        ax,
        xs[1],
        y,
        w,
        h,
        "2. Bounded smooth warp search",
        [
            r"Sample $N_{cand}$ control-point vectors.",
            r"Each vector lives in a safety corridor",
            r"and is upsampled with cubic splines.",
        ],
        facecolor="#fdf3e7",
    )
    draw_spline(ax, xs[1] + 0.03, y + 0.06, 0.15, 0.14)

    add_box(
        ax,
        xs[2],
        y,
        w,
        h,
        "3. Parallel candidate view bank",
        [
            r"Apply all smooth scaling curves to $x_t$.",
            r"Build warped candidates in one shot",
            r"with GPU-friendly broadcasting.",
        ],
        facecolor="#f7f3fb",
    )
    draw_candidates(ax, xs[2] + 0.02, y + 0.05, 0.16, 0.17)

    add_box(
        ax,
        xs[3],
        y,
        w,
        h,
        "4. One-shot entropy selection",
        [
            r"Evaluate all candidates with the current model.",
            r"Choose $i^* = \arg\max_i H(p_t^{(i)})$.",
            r"Return the smooth worst-case view $x_t^{pert}$.",
        ],
        facecolor="#eef8ef",
    )
    ax.text(xs[3] + 0.03, y + 0.12, r"$H(p_t^{(1)}) \quad H(p_t^{(2)}) \quad \cdots \quad H(p_t^{(N)})$", fontsize=10.4, color="#222222")
    ax.text(xs[3] + 0.03, y + 0.07, "pick highest-entropy smooth candidate", fontsize=9.7, color="#d62728")
    draw_wave(ax, xs[3] + 0.03, y + 0.01, 0.15, 0.08, color="#d62728", mode="smooth")

    add_arrow(ax, (xs[0] + w, y + 0.17), (xs[1], y + 0.17))
    add_arrow(ax, (xs[1] + w, y + 0.17), (xs[2], y + 0.17))
    add_arrow(ax, (xs[2] + w, y + 0.17), (xs[3], y + 0.17))

    footer_y = 0.12
    ax.text(0.04, footer_y + 0.11, "What SSAW is designed to generate", fontsize=12.5, fontweight="bold", color="#111111")
    ax.text(
        0.04,
        footer_y + 0.04,
        "Bounded low-frequency amplitude drifts that are adversarial enough to stress the model,\n"
        "but still smooth and physically plausible enough to serve as a consistency view in streaming TTA.",
        fontsize=10.5,
        color="#222222",
        linespacing=1.45,
    )
    ax.text(
        0.61,
        footer_y + 0.07,
        r"$x_t \;\rightarrow\; \{w^{(i)}(\tau)\}_{i=1}^{N_{cand}} \;\rightarrow\; \{x_t^{(i)}\} \;\rightarrow\; x_t^{pert}$",
        fontsize=14,
        color="#111111",
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out_pdf)


if __name__ == "__main__":
    main()
