import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

from deprl.vendor.tonic.plot import plot

parser = argparse.ArgumentParser()
parser.add_argument("--paths", nargs="+", default=[])
parser.add_argument("--x_axis", default="train/steps")
parser.add_argument("--y_axis", default="test/episode_score")
parser.add_argument("--y_axis_norm", default="")
parser.add_argument("--x_label")
parser.add_argument("--y_label")
parser.add_argument("--interval", default="std")
parser.add_argument("--window", type=int, default=10)
parser.add_argument("--show_seeds", type=bool, default=False)
parser.add_argument("--columns", type=int)
parser.add_argument("--x_min", type=int)
parser.add_argument("--x_max", type=int)
parser.add_argument("--y_min", type=float)
parser.add_argument("--y_max", type=float)
parser.add_argument("--baselines", nargs="+")
parser.add_argument("--baselines_source", default="tensorflow")
parser.add_argument("--name")
parser.add_argument("--save_formats", nargs="*", default=["png"])
# parser.add_argument("--seconds", type=int, default=0)
parser.add_argument("--cmap")
parser.add_argument("--legend_columns", type=int)
parser.add_argument("--font_size", type=int, default=10)
parser.add_argument("--font_family", default="serif")
parser.add_argument("--legend_font_size", type=int)
parser.add_argument("--legend_marker_size", type=int, default=10)
parser.add_argument("--backend", default="agg")
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--title")
args = parser.parse_args()


# Backend selection, e.g. agg for non-GUI.
if args.backend:
    mpl.use(args.backend)
del args.backend

# Fonts.
plt.rc("font", family=args.font_family, size=args.font_size)
if args.legend_font_size:
    plt.rc("legend", fontsize=args.legend_font_size)
del args.font_family, args.font_size, args.legend_font_size


graphs = [
    ("test/action", None, "std"),
    ("test/effort", None, "std"),
    ("test/episode_length", None, "std"),
    ("test/episode_score", None, "std"),
    ("test/rwd_metrics/gaussian_vel", None, "std"),
    ("test/rwd_metrics/forward_lean", None, "std"),
    ("test/rwd_metrics/sideways_lean", None, "std"),
    ("test/rwd_metrics/forward_direction", None, "std"),
    ("test/rwd_metrics/grf", None, "std"),
    ("test/rwd_metrics/done", None, "std"),
    ("test/rwd_metrics/smooth_exc", None, "std"),
    ("test/rwd_metrics/number_muscles", None, "std"),
    ("test/rwd_metrics/joint_limit", None, "std"),
    ("test/terminated", None, "std"),
    ("train/action", None, "std"),
    ("train/episode_length", None, "std"),
    ("train/episode_score", None, "std"),
    ("train/action_cost_coeff", None, None),
    ("train/energy_buffer/prev_cdt", None, None),
    ("train/energy_buffer/self.score_avg", None, None),
    ("train/energy_buffer/lr", None, None),
    ("train/energy_buffer/action_cost_intern", None, None),
    ("train/energy_buffer/avg_relabel_action_cost", None, None),
    ("train/episodes", None, None),
    ("train/epoch_seconds", None, None),
    ("train/epoch_steps", None, None),
    ("train/epochs", None, None),
    ("train/seconds", None, None),
    ("train/steps", None, None),
    ("train/steps_per_second", None, None),
    ("train/worker_steps", None, None),
    ("test/action", "test/episode_length/mean", None),
    ("test/effort", "test/episode_length/mean", None),
    ("test/episode_score", "test/episode_length/mean", None),
    ("test/rwd_metrics/gaussian_vel", "test/episode_length/mean", None),
    ("test/rwd_metrics/forward_lean", "test/episode_length/mean", None),
    ("test/rwd_metrics/sideways_lean", "test/episode_length/mean", None),
    ("test/rwd_metrics/forward_direction", "test/episode_length/mean", None),
    ("test/rwd_metrics/grf", "test/episode_length/mean", None),
    ("test/rwd_metrics/done", "test/episode_length/mean", None),
    ("test/rwd_metrics/smooth_exc", "test/episode_length/mean", None),
    ("test/rwd_metrics/number_muscles", "test/episode_length/mean", None),
    ("test/rwd_metrics/joint_limit", "test/episode_length/mean", None),
    ("test/terminated", "test/episode_length/mean", None),
    ("train/action", "train/episode_length/mean", None),
    ("train/episode_score", "train/episode_length/mean", None),
]

plot_path = f"{args.paths[0]}/plots"

# Plot.
for y_axis, y_axis_norm, interval in graphs:
    args.y_axis = y_axis
    label = y_axis.replace("/", "_").replace(".", "_")
    args.y_label = label

    if y_axis_norm is not None:
        args.y_axis_norm = y_axis_norm

    sub_path = ""
    if y_axis.startswith("test/"):
        sub_path = "test"
    elif y_axis.startswith("train/"):
        sub_path = "train"
    if y_axis_norm is not None:
        sub_path += "/normalised"
    name = f"{plot_path}/{sub_path}/{label}"
    if y_axis_norm is not None:
        name += "_norm"
    args.name = name

    args.interval = interval

    print(f"Plotting: {label}")
    plot(**vars(args), fig=None)
