"""
Report Agent — writes a transparent LaTeX PDF documenting every decision made
by the QA/QC and correction agents, with figures and error statistics.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
from motus.agent import ReActAgent
from motus.tools import tool

from utils.client import make_client
from scipy.spatial import cKDTree

from utils.accumulation import day_accumulation
from utils.geo import latlon_to_xyz

mpl.use("Agg")
logger = logging.getLogger(__name__)

PRECIP_LEVELS = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100]
PRECIP_CMAP   = plt.get_cmap("YlGnBu", len(PRECIP_LEVELS) - 1)
PRECIP_NORM   = BoundaryNorm(PRECIP_LEVELS, PRECIP_CMAP.N)
DIFF_LEVELS   = [-30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30]
DIFF_CMAP     = plt.get_cmap("RdBu_r", len(DIFF_LEVELS) - 1)
DIFF_NORM     = BoundaryNorm(DIFF_LEVELS, DIFF_CMAP.N)

_SYSTEM_PROMPT = """
You are a scientific report writer specialising in meteorological data products.

You will receive:
- A structured log of every QA/QC decision made
- A structured log of every bias correction decision made, with metrics

Your job:
1. Write clear, honest narrative for each section.
2. Cite SPECIFIC NUMBERS (bias, RMSE, correlation, station counts, thresholds).
   Never write "the bias was large" -- write "bias = +3.2 mm/day".
3. For every parameter choice, explain WHY that value was chosen over alternatives.
4. Include ALL correction attempts -- parameters tried, metrics, accept/reject reason.
5. If no QA/QC changes were made, explain why the data was already clean.
6. If the correction was ultimately rejected, explain that too.

FORMATTING RULES -- MANDATORY. Violations break the PDF compilation:
- Write PLAIN PROSE ONLY. No markdown whatsoever.
- No **, no *, no #, no - bullets, no numbered lists, no horizontal rules.
- No Unicode: no em-dash, no arrow, no Unicode minus, no checkmarks, no emoji.
  Use ASCII only: -- for dashes, -> for arrows, - for minus.
- No curly/smart quotes. Straight ASCII quotes only.
- Separate paragraphs with one blank line. That is all the formatting you need.

Call narrate() before each write_section to share what you will write.
Call write_section for each section, then finish_report.
Available sections: executive_summary, qaqc_narrative, correction_narrative, conclusions
""".strip()


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _tex(s: str) -> str:
    """Escape LaTeX special characters in a plain-text string."""
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                     ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")]:
        s = s.replace(old, new)
    return s


def _inline_md(s: str) -> str:
    """Convert inline markdown to LaTeX (call AFTER _tex so * is untouched)."""
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', s)
    s = re.sub(r'`(.+?)`', r'\\texttt{\1}', s)
    # Unicode chars LLMs emit despite instructions
    s = s.replace('−', '-').replace('→', '->').replace('—', '--')
    s = s.replace('–', '--').replace('✅', '').replace('✔', '')
    s = s.replace('❌', '').replace('✓', '').replace('×', 'x')
    s = s.replace('≥', r'$\geq$').replace('≤', r'$\leq$')
    s = s.replace('≈', r'$\approx$').replace('±', r'$\pm$')
    return s


def _md_to_tex(s: str) -> str:
    """Convert a markdown/mixed narrative block to valid LaTeX prose."""
    lines = s.split('\n')
    out = []
    list_env: str | None = None

    def close_list():
        nonlocal list_env
        if list_env:
            out.append(f'\\end{{{list_env}}}')
            list_env = None

    for line in lines:
        m = re.match(r'^(#{1,3})\s+(.+)', line)
        if m:
            close_list()
            out.append(f'\\medskip\\noindent\\textbf{{{_inline_md(_tex(m.group(2)))}}}\\par')
            continue

        m = re.match(r'^[-*+]\s+(.+)', line)
        if m:
            if list_env == 'enumerate':
                close_list()
            if list_env is None:
                out.append(r'\begin{itemize}')
                list_env = 'itemize'
            out.append(f'  \\item {_inline_md(_tex(m.group(1)))}')
            continue

        m = re.match(r'^\d+[.)]\s+(.+)', line)
        if m:
            if list_env == 'itemize':
                close_list()
            if list_env is None:
                out.append(r'\begin{enumerate}')
                list_env = 'enumerate'
            out.append(f'  \\item {_inline_md(_tex(m.group(1)))}')
            continue

        if re.match(r'^[-*_]{3,}\s*$', line):
            close_list()
            out.append(r'\medskip\hrule\medskip')
            continue

        if not line.strip():
            close_list()
            out.append('')
            continue

        out.append(_inline_md(_tex(line)))

    close_list()
    return '\n'.join(out)


# ── QA/QC log renderer ─────────────────────────────────────────────────────────

_QAQC_LABELS = {
    "clip_negative_grid":   "Clip negatives (model)",
    "flag_extreme_grid":    "Flag extremes (model)",
    "zero_trace_grid":      "Zero trace (model)",
    "no_change_grid":       "No change (model)",
    "clip_negative_obs":    "Clip negatives (obs)",
    "flag_extreme_obs":     "Flag extremes (obs)",
    "drop_sparse_stations": "Drop sparse stations",
    "no_change_obs":        "No change (obs)",
    "finish":               "Summary",
}


def _clean(s: str) -> str:
    """Strip emoji/unicode and collapse whitespace for LaTeX prose."""
    s = re.sub(r'[✅✔❌✓✗✖♥♦♠♣]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return _inline_md(_tex(s))


def _render_qaqc_log(log: list[dict]) -> str:
    """Render QA/QC decision log as a readable LaTeX table."""
    if not log:
        return r"\textit{No QA/QC actions recorded.}"

    rows: list[tuple[str, str]] = []
    for entry in log:
        action = entry.get("action", "")
        label  = _QAQC_LABELS.get(action, _tex(action))

        if action == "finish":
            detail = _clean(entry.get("summary", ""))

        elif action == "clip_negative_grid":
            n   = entry.get("n_clipped", 0)
            var = _tex(entry.get("variable", ""))
            detail = f"Clipped {n:,} negative \\texttt{{{var}}} grid values to zero."

        elif action == "flag_extreme_grid":
            n   = entry.get("n_flagged", 0)
            thr = entry.get("threshold", "?")
            var = _tex(entry.get("variable", ""))
            detail = (f"Flagged {n:,} \\texttt{{{var}}} values above "
                      f"{thr}\\,mm as NaN.")

        elif action == "zero_trace_grid":
            n   = entry.get("n_zeroed", 0)
            thr = entry.get("threshold", "?")
            var = _tex(entry.get("variable", ""))
            detail = (f"Set {n:,} \\texttt{{{var}}} trace values "
                      f"($0 < x < {thr}$\\,mm) to exactly 0.")

        elif action in ("no_change_grid", "no_change_obs"):
            detail = _clean(entry.get("reason", ""))

        elif action == "clip_negative_obs":
            n = entry.get("n_clipped", 0)
            detail = f"Clipped {n:,} negative observation values to zero."

        elif action == "flag_extreme_obs":
            n   = entry.get("n_flagged", 0)
            thr = entry.get("threshold", "?")
            detail = f"Flagged {n:,} obs values above {thr}\\,mm as NaN."

        elif action == "drop_sparse_stations":
            removed  = entry.get("removed", 0)
            min_days = entry.get("min_days", "?")
            detail = (f"Removed {removed:,} stations with fewer than "
                      f"{min_days} valid reporting days.")

        else:
            rest   = {k: v for k, v in entry.items() if k != "action"}
            detail = _tex(json.dumps(rest, default=str))

        rows.append((label, detail))

    lines = [
        r"\begin{tabular}{@{}p{0.30\textwidth}p{0.64\textwidth}@{}}\toprule",
        r"\textbf{Action} & \textbf{Detail} \\\midrule",
    ]
    for label, detail in rows:
        # escape label (already clean ASCII) and wrap in bold
        lines.append(f"\\textbf{{{label}}} & {detail} \\\\[3pt]")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines)


# ── Fixed methods block ────────────────────────────────────────────────────────

def _methods_latex(obs_utc_cutoff: int = 0) -> str:
    return r"""
\subsection*{Correction Methods}

The pipeline selects a correction method based on the diagnosed precipitation
regime. For stratiform events (spatially coherent bias), Inverse Distance
Weighting (IDW) is applied. For convective events (high spatial variability,
low obs--model spatial correlation), Quantile Mapping (QM) is preferred because
it corrects the distribution shape without requiring spatial coherence.

\subsubsection*{IDW Correction}

For each reported observation date, a multiplicative ratio field $R(\mathbf{x})$ and an
additive offset field $A(\mathbf{x})$ are estimated from station observations
and then applied to every model grid point.

\subsubsection*{Observation accumulation window}

""" + (
    rf"""CoCoRaHS and ACIS observers typically read their gauges at approximately
7\,am local time.  For the Central Daylight Time zone this corresponds to
{obs_utc_cutoff:02d}:00\,UTC.  An observation labelled date $D$ therefore
represents the 24-hour accumulation ending at {obs_utc_cutoff:02d}:00\,UTC on day $D$,
i.e.\ the window $[\text{{D-1}}\;{obs_utc_cutoff:02d}:00\;\text{{UTC}},\;
\text{{D}}\;{obs_utc_cutoff:02d}:00\;\text{{UTC}})$.
Model hourly fields are summed over that same window before computing
bias ratios and validation metrics.
Using the calendar day (00:00--23:00\,UTC) instead would introduce a
systematic 12-hour phase offset that artificially degrades spatial
correlation for convective precipitation events."""
    if obs_utc_cutoff > 0 else
    r"""The model is accumulated over the calendar day (00:00--23:00\,UTC)
to match observation date labels.
Note that CoCoRaHS/ACIS daily totals typically represent a 24-hour window
ending at approximately 12:00\,UTC (7\,am local time for Central Daylight Time),
introducing a $\sim$12-hour phase offset relative to the calendar-day sum.
Set \texttt{obs\_utc\_cutoff = 12} in the configuration to use the rigorous window."""
) + r"""

\subsubsection*{IDW weight and ratio field}

For a grid point $\mathbf{x}$, let $S(\mathbf{x})$ be the set of stations within
the search radius $r$ that have valid observations. The IDW weight for station $i$ is:

\begin{equation}
  w_i(\mathbf{x}) = d_i(\mathbf{x})^{-p}
\end{equation}

where $d_i(\mathbf{x})$ is the great-circle distance and $p$ is the power parameter.
The multiplicative ratio field is the distance-weighted mean of station bias ratios:

\begin{equation}
  R(\mathbf{x}) =
    \frac{\displaystyle\sum_{i \in S(\mathbf{x})} w_i\, r_i}
         {\displaystyle\sum_{i \in S(\mathbf{x})} w_i},
  \qquad
  r_i = \frac{O_i}{M_i^{\mathrm{stn}}}
\end{equation}

where $O_i$ is the observed daily accumulation and $M_i^{\mathrm{stn}}$ is the model
value interpolated to station $i$ by nearest-neighbour lookup.
The additive field $A(\mathbf{x})$ is computed analogously for stations where the
model predicts zero but observations record precipitation ($O_i > \varepsilon$,
$M_i^{\mathrm{stn}} \leq \varepsilon$).

\subsubsection*{Application to the model grid}

Let $M(\mathbf{x},t)$ be the raw model value at grid point $\mathbf{x}$ and hour $t$.
The corrected value is:

\begin{equation}
  C(\mathbf{x},t) =
  \begin{cases}
    \max\!\bigl(0,\; M(\mathbf{x},t)\cdot R(\mathbf{x})\bigr) & M(\mathbf{x},t) > \varepsilon \\[4pt]
    \max\!\bigl(0,\; M(\mathbf{x},t) + A(\mathbf{x})\bigr)    & M(\mathbf{x},t) \leq \varepsilon
  \end{cases}
\end{equation}

where $\varepsilon = 0.1$\,mm is the trace-precipitation threshold.

\subsubsection*{Quantile Mapping Correction}

Quantile mapping (QM) corrects the model distribution globally by mapping model
quantiles to observed quantiles. All station--day pairs are pooled to estimate
the empirical cumulative distribution functions (CDFs) of model output
$\hat{F}_M$ and observations $\hat{F}_O$.

The corrected value at any grid point is:

\begin{equation}
  C(x) = \hat{F}_O^{-1}\!\bigl(\hat{F}_M(x)\bigr)
\end{equation}

where $\hat{F}_O^{-1}$ is the empirical quantile function of the observations.
In practice, a piecewise-linear transfer function is built from 100 equally
spaced quantiles and applied pointwise to every model grid value.
QM does not alter the spatial pattern of model precipitation -- it only adjusts
the distribution -- making it robust to convective phase errors.

\subsection*{Regime Diagnosis}

Before correction, the pipeline computes two spatial statistics per day:

\begin{itemize}
  \item Coefficient of variation: $\mathrm{CV} = \sigma_M / \mu_M$, where
        $\sigma_M$ and $\mu_M$ are the standard deviation and mean of daily
        model precipitation over the domain.
  \item Obs--model spatial correlation: Pearson $r$ between observed station
        values and nearest-neighbour model values at station locations.
\end{itemize}

A regime is classified as \textit{convective} if $\mathrm{CV} > 2$ or $r < 0.3$,
and as \textit{stratiform} otherwise. QM is the preferred method for convective
regimes; IDW is preferred for stratiform.

\subsection*{Validation Metrics}

Performance is evaluated at station locations using nearest-neighbour grid
interpolation. Let $M_i$ and $O_i$ be model and observed daily accumulations at
station location $i$, and $N$ the total number of station-day pairs.

\begin{align}
  \mathrm{Bias} &= \frac{1}{N}\sum_{i=1}^{N}(M_i - O_i) \\[6pt]
  \mathrm{RMSE} &= \sqrt{\frac{1}{N}\sum_{i=1}^{N}(M_i - O_i)^2} \\[6pt]
  r             &= \frac{\displaystyle\sum_{i=1}^{N}(O_i-\bar{O})(M_i-\bar{M})}
                        {\sqrt{\displaystyle\sum_{i=1}^{N}(O_i-\bar{O})^2
                               \sum_{i=1}^{N}(M_i-\bar{M})^2}}
\end{align}

RMSE improvement is expressed as a percentage relative to the pre-correction RMSE:

\begin{equation}
  \Delta\mathrm{RMSE}\% =
    \frac{|\mathrm{RMSE}_{\mathrm{pre}}| - |\mathrm{RMSE}_{\mathrm{post}}|}
         {|\mathrm{RMSE}_{\mathrm{pre}}| + 10^{-9}} \times 100
\end{equation}
""".strip()


# ── Figure generation ──────────────────────────────────────────────────────────

def _save_fig(fig, path: Path, dpi: int = 110) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_comparison_figures(ds_before: xr.Dataset, ds_after: xr.Dataset,
                              df_obs: pd.DataFrame, variable: str,
                              fig_dir: Path,
                              obs_utc_cutoff: int = 0) -> list[Path]:
    grid_lat = ds_before["lat"].values
    grid_lon = ds_before["lon"].values
    df_obs   = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date
    paths = []

    days = sorted({pd.Timestamp(t).date() for t in ds_before.time.values})
    for day in days:
        pre  = day_accumulation(ds_before, variable, day, obs_utc_cutoff)
        post = day_accumulation(ds_after,  variable, day, obs_utc_cutoff)
        diff = post - pre
        obs_day = df_obs[df_obs["date"] == day]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, data, title, cmap, norm in [
            (axes[0], pre,  "Before correction", PRECIP_CMAP, PRECIP_NORM),
            (axes[1], post, "After correction",  PRECIP_CMAP, PRECIP_NORM),
            (axes[2], diff, "Difference",         DIFF_CMAP,  DIFF_NORM),
        ]:
            pcm = ax.pcolormesh(grid_lon, grid_lat, data, cmap=cmap, norm=norm,
                                shading="auto")
            if ax is axes[0] and not obs_day.empty:
                ax.scatter(obs_day["lon"], obs_day["lat"], s=4, c="red",
                           alpha=0.6, zorder=5)
            plt.colorbar(pcm, ax=ax, label="mm/day", pad=0.02, shrink=0.85)
            ax.set_xlim(float(grid_lon.min()), float(grid_lon.max()))
            ax.set_ylim(float(grid_lat.min()), float(grid_lat.max()))
            ax.set_title(f"{title}\n{day}", fontsize=10)
            ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        fig.suptitle(f"wxCal Bias Correction -- {variable} -- {day}", fontsize=12)
        fig.tight_layout()
        paths.append(_save_fig(fig, fig_dir / f"compare_{day}.png"))

    return paths


def _make_scatter_plots(ds_before: xr.Dataset, ds_after: xr.Dataset,
                        df_obs: pd.DataFrame, variable: str,
                        fig_dir: Path,
                        obs_utc_cutoff: int = 0) -> list[Path]:
    """Obs vs model scatter plots (before and after) with Bias/RMSE/r annotations."""
    grid_lat = ds_before["lat"].values
    grid_lon = ds_before["lon"].values
    df_obs   = df_obs.copy()
    df_obs["date"] = pd.to_datetime(df_obs["date"]).dt.date

    tree = cKDTree(latlon_to_xyz(grid_lat.ravel(), grid_lon.ravel()))
    all_obs, all_pre, all_post = [], [], []

    for day in sorted({pd.Timestamp(t).date() for t in ds_before.time.values}):
        pre  = day_accumulation(ds_before, variable, day, obs_utc_cutoff)
        post = day_accumulation(ds_after,  variable, day, obs_utc_cutoff)
        obs_day = df_obs[df_obs["date"] == day].dropna(subset=["value"])
        if obs_day.empty:
            continue
        _, idx = tree.query(
            latlon_to_xyz(obs_day["lat"].values, obs_day["lon"].values),
            k=1, workers=-1)
        all_obs.append(obs_day["value"].values)
        all_pre.append(pre.ravel()[idx].astype(float))
        all_post.append(post.ravel()[idx].astype(float))

    if not all_obs:
        return []

    obs  = np.concatenate(all_obs)
    pre  = np.concatenate(all_pre)
    post = np.concatenate(all_post)

    mk_pre  = np.isfinite(obs) & np.isfinite(pre)
    mk_post = np.isfinite(obs) & np.isfinite(post)
    lim = max(obs[mk_pre].max()  if mk_pre.any()  else 1,
              pre[mk_pre].max()  if mk_pre.any()  else 1,
              post[mk_post].max() if mk_post.any() else 1) * 1.05

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, mask, title in [
        (axes[0], pre,  mk_pre,  "Before Correction"),
        (axes[1], post, mk_post, "After Correction"),
    ]:
        o, m = obs[mask], model[mask]
        ax.scatter(o, m, s=3, alpha=0.25, color="steelblue", rasterized=True)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)

        if len(o) >= 2:
            bias = float(np.mean(m - o))
            rmse = float(np.sqrt(np.mean((m - o) ** 2)))
            corr = (float(np.corrcoef(o, m)[0, 1])
                    if o.std() > 0 and m.std() > 0 else float("nan"))
        else:
            bias = rmse = corr = float("nan")

        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed (mm/day)", fontsize=9)
        ax.set_ylabel("Model (mm/day)", fontsize=9)
        ax.set_title(f"{title}  (n = {len(o):,})", fontsize=10)
        ax.text(0.05, 0.95,
                f"Bias = {bias:+.2f} mm\nRMSE = {rmse:.2f} mm\nr = {corr:.3f}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.6))

    fig.suptitle(f"Model vs. Observations -- {variable} (all days combined)", fontsize=11)
    fig.tight_layout()
    return [_save_fig(fig, fig_dir / "scatter_model_vs_obs.png")]


# ── LaTeX compilation ──────────────────────────────────────────────────────────

def _compile_latex(tex_path: Path) -> Path:
    cmd = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        r = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"xelatex failed:\n{r.stdout[-3000:]}")
    return tex_path.with_suffix(".pdf")


# ── Agent class ────────────────────────────────────────────────────────────────

class ReportAgent:

    def __init__(self, config):
        self.config = config
        self.sections: dict[str, str] = {}
        self._done = False
        self._agent = self._build_agent()

    def _build_agent(self) -> ReActAgent:

        @tool
        def write_section(section_name: str, content: str) -> str:
            """
            Write a report narrative section. section_name must be one of:
            executive_summary, qaqc_narrative, correction_narrative, conclusions.
            Write plain prose -- no markdown, no Unicode special chars.
            """
            self.sections[section_name] = content
            return f"Section '{section_name}' saved ({len(content)} chars)."

        @tool
        def finish_report() -> str:
            """Call when all four sections have been written."""
            self._done = True
            return "Report narrative complete."

        @tool
        def narrate(message: str) -> str:
            """Share what you are about to write."""
            logger.info(f"  [report] {message}")
            return "ok"

        client, model_name = make_client(self.config.llm_model)
        return ReActAgent(
            client=client,
            model_name=model_name,
            system_prompt=_SYSTEM_PROMPT,
            tools=[narrate, write_section, finish_report],
            max_steps=15,
        )

    def run(self, ds_before: xr.Dataset, ds_after: xr.Dataset,
            df_obs: pd.DataFrame, variable: str,
            qaqc_log: list[dict], correction_log: list[dict],
            output_paths: list[Path]) -> Path:

        cfg = self.config
        pdf_path = cfg.output_dir / f"wxcal_report_{cfg.date_tag}.pdf"

        # Save figures to a persistent directory alongside the PDF.
        # Must be absolute — xelatex compiles from a temp dir and can't resolve relative paths.
        fig_dir = (cfg.output_dir / "figures").resolve()
        fig_dir.mkdir(parents=True, exist_ok=True)

        cutoff = getattr(cfg, "obs_utc_cutoff", 0)
        logger.info("Generating comparison maps...")
        fig_paths     = _make_comparison_figures(ds_before, ds_after, df_obs,
                                                  variable, fig_dir, cutoff)
        logger.info("Generating scatter plots...")
        scatter_paths = _make_scatter_plots(ds_before, ds_after, df_obs,
                                             variable, fig_dir, cutoff)
        logger.info(f"  {len(fig_paths)} map(s), {len(scatter_paths)} scatter plot(s) "
                    f"saved to {fig_dir}")

        prompt = (
            f"Write the wxCal report for this run.\n\n"
            f"Variable: {variable}\n"
            f"Model source: {cfg.model_source}\n"
            f"Observation source: {cfg.obs_source}\n"
            f"Period: {cfg.date_start} to {cfg.date_end}\n\n"
            f"QA/QC decision log:\n{json.dumps(qaqc_log, indent=2, default=str)}\n\n"
            f"Bias correction log:\n"
            f"{json.dumps(correction_log, indent=2, default=str)}\n\n"
            f"Output files: {[str(p) for p in output_paths]}\n\n"
            f"Write all four sections then call finish_report.\n"
            f"IMPORTANT: plain prose only. No markdown. No Unicode."
        )

        import asyncio
        asyncio.get_event_loop().run_until_complete(self._agent(prompt))

        with tempfile.TemporaryDirectory(prefix="wxcal_report_") as tmp:
            tmp_dir  = Path(tmp)
            tex      = self._build_latex(fig_paths, scatter_paths, variable,
                                          qaqc_log, correction_log, output_paths)
            tex_path = tmp_dir / "report.tex"
            tex_path.write_text(tex, encoding="utf-8")
            compiled = _compile_latex(tex_path)
            shutil.copy(compiled, pdf_path)

        logger.info(f"Report saved -> {pdf_path}")
        return pdf_path

    def _build_latex(self, fig_paths: list[Path], scatter_paths: list[Path],
                      variable: str, qaqc_log: list[dict],
                      correction_log: list[dict], output_paths: list[Path]) -> str:
        cfg = self.config
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

        def sec(name: str, default: str = "") -> str:
            return self.sections.get(name, default)

        def fig_block(path: Path, caption: str = "",
                      width: str = r"\textwidth") -> str:
            return (r"\begin{figure}[htbp]" "\n"
                    r"  \centering" "\n"
                    rf"  \includegraphics[width={width}]{{{str(path)}}}" "\n"
                    rf"  \caption{{{_tex(caption)}}}" "\n"
                    r"\end{figure}" "\n"
                    r"\clearpage" "\n")

        lines = [
            r"\documentclass[11pt,letterpaper]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{fontspec}",
            r"\usepackage{graphicx,amsmath,amssymb,booktabs,float,parskip,hyperref}",
            r"\usepackage{listings,xcolor}",
            r"\lstset{basicstyle=\scriptsize\ttfamily,breaklines=true,"
            r"breakatwhitespace=false,columns=flexible,keepspaces=true,"
            r"frame=single,backgroundcolor=\color{gray!8}}",
            r"\title{\textbf{wxCal --- Weather Calibration Report}\\[6pt]"
            rf"\large {_tex(variable)} $\cdot$ "
            rf"{_tex(str(cfg.date_start))} -- {_tex(str(cfg.date_end))}" + r"}",
            r"\author{Dimitrios K.\ Fytanidis \and wxCal Agent (Claude)}",
            rf"\date{{{_tex(now)}}}",
            r"\begin{document}",
            r"\maketitle\tableofcontents\newpage",

            r"\section{Executive Summary}",
            _md_to_tex(sec("executive_summary", "No summary written.")),
            r"\newpage",

            r"\section{Pipeline Parameters}",
            r"\begin{tabular}{@{}ll@{}}\toprule",
            r"\textbf{Parameter} & \textbf{Value} \\\midrule",
            rf"Model source & \texttt{{{_tex(str(cfg.model_source))}}} \\",
            rf"Variable & {_tex(variable)} \\",
            rf"Observation source & \texttt{{{_tex(str(cfg.obs_source))}}} \\",
            rf"Period & {_tex(str(cfg.date_start))} -- {_tex(str(cfg.date_end))} \\",
            rf"LLM model & \texttt{{{_tex(cfg.llm_model)}}} \\",
            rf"Max correction retries & {cfg.max_correction_retries} \\",
            r"\bottomrule\end{tabular}",

            r"\section{QA/QC}",
            _md_to_tex(sec("qaqc_narrative", "QA/QC narrative not written.")),
            r"\subsection*{QA/QC Decision Log}",
            _render_qaqc_log(qaqc_log),

            r"\section{Bias Correction}",
            _md_to_tex(sec("correction_narrative", "Correction narrative not written.")),
            _methods_latex(getattr(cfg, "obs_utc_cutoff", 0)),

            r"\subsection*{Correction Attempt Results}",
        ]

        for entry in correction_log:
            if entry.get("action") == "finish":
                accepted = "Accepted" if entry.get("accepted") else "Rejected"
                reasoning = _clean(entry.get("reasoning", ""))
                lines += [
                    rf"\medskip\textbf{{Final decision: {_tex(accepted)}}} --- {reasoning}",
                    "",
                ]
            elif "attempt" in entry:
                pre = entry.get("metrics", {}).get("pre_correction",  {})
                post = entry.get("metrics", {}).get("post_correction", {})
                imp  = entry.get("metrics", {}).get("improvement",     {})

                def _f(d: dict, k: str) -> str:
                    v = d.get(k, float("nan"))
                    return f"{v:.3f}" if isinstance(v, float) else str(v)

                method = entry.get("method", "idw")
                if method == "quantile_mapping":
                    param_str = "Quantile Mapping"
                else:
                    param_str = (rf"IDW $p = {entry['idw_power']}$, "
                                 rf"radius $= {entry['radius_km']}$\,km")

                lines += [
                    rf"\paragraph{{Attempt {entry['attempt']}: {param_str}}}",
                    r"\begin{tabular}{@{}lrr@{}}\toprule",
                    r"Metric & Pre-correction & Post-correction \\\midrule",
                    rf"Bias (mm/day) & {_f(pre,'bias')} & {_f(post,'bias')} \\",
                    rf"RMSE (mm/day) & {_f(pre,'rmse')} & {_f(post,'rmse')} \\",
                    rf"Correlation   & {_f(pre,'corr')} & {_f(post,'corr')} \\",
                    rf"RMSE change   & \multicolumn{{2}}{{r}}"
                    rf"{{{imp.get('rmse_improvement_pct', float('nan')):.1f}\%}} \\",
                    r"\bottomrule\end{tabular}",
                    "",
                ]

        lines += [r"\subsection*{Station vs.\ Model Scatter}", r"\clearpage"]
        for p in scatter_paths:
            lines.append(fig_block(
                p,
                f"Observed vs. model {variable} at station locations (all days combined). "
                f"Left: before correction. Right: after correction. "
                f"Bias, RMSE, and Pearson r annotated in each panel.",
            ))

        lines += [r"\subsection*{Before / After Spatial Maps}", r"\clearpage"]
        for p in fig_paths:
            day_str = p.stem.replace("compare_", "")
            lines.append(fig_block(
                p,
                f"Daily {variable} for {day_str}. "
                f"Left: raw model. Centre: bias-corrected. "
                f"Right: difference (corrected minus raw). "
                f"Red dots: station locations (before-correction panel only).",
            ))

        lines += [
            r"\section{Conclusions}",
            _md_to_tex(sec("conclusions", "No conclusions written.")),

            r"\section*{Output Files}",
            r"\begin{itemize}",
        ]
        for p in output_paths:
            lines.append(rf"  \item \texttt{{{_tex(str(p))}}}")
        lines += [r"\end{itemize}", r"\end{document}"]

        return "\n".join(lines)
