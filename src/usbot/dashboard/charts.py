"""Plotly chart builders. Each returns an HTML <div> string (no full page) or a
graceful placeholder when data is insufficient. Dark theme throughout.

Plotly JS is loaded once from a CDN (see builder); charts use include_plotlyjs=False.
"""
from __future__ import annotations


_DARK = dict(
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#c9d1d9", family="-apple-system, Segoe UI, Roboto, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=30),
)
_RDYLGN = [[0.0, "#c5221f"], [0.5, "#30363d"], [1.0, "#137333"]]


def _placeholder(msg: str) -> str:
    return (f'<div class="placeholder">{msg}</div>')


def _to_div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def treemap(rows: list[dict], color_label: str = "Daily return") -> str:
    """Finviz-style sector→ticker heatmap. rows: {symbol, sector, size, color, label}."""
    rows = [r for r in rows if r.get("size", 0) and r.get("sector")]
    if not rows:
        return _placeholder("Market heatmap unavailable — no price/sector data yet.")
    try:
        import plotly.graph_objects as go

        labels, parents, values, colors, text = [], [], [], [], []
        sectors = sorted({r["sector"] for r in rows})
        for sec in sectors:
            labels.append(sec); parents.append(""); values.append(0); colors.append(0.0)
            text.append("")
        for r in rows:
            labels.append(r["symbol"]); parents.append(r["sector"])
            values.append(float(r["size"])); colors.append(float(r.get("color", 0.0)))
            text.append(r.get("label", ""))
        fig = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values, branchvalues="total",
            marker=dict(colors=colors, colorscale=_RDYLGN, cmid=0.0,
                        colorbar=dict(title=color_label)),
            text=text, textinfo="label+text",
            hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
        ))
        fig.update_layout(height=520, title=f"Market Heatmap — colored by {color_label}", **_DARK)
        return _to_div(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"Heatmap render failed: {exc}")


def perf_lines(series: list, title: str = "Cumulative Return (=100)") -> str:
    """Cumulative return lines; series: list of PerfSeries."""
    valid = [s for s in series if s.equity is not None and len(s.equity) >= 2]
    if not valid:
        return _placeholder("Not enough history yet for a performance curve "
                            "(builds up as the bot runs daily).")
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for s in valid:
            fig.add_trace(go.Scatter(
                x=s.equity.index, y=s.equity.values, mode="lines", name=s.name,
                line=dict(width=2.4 if s.is_portfolio else 1.4,
                          dash=None if s.is_portfolio else "dot")))
        fig.update_layout(height=420, title=title, hovermode="x unified", **_DARK)
        fig.update_yaxes(gridcolor="#21262d"); fig.update_xaxes(gridcolor="#21262d")
        return _to_div(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"Performance chart failed: {exc}")


def drawdown_lines(series: list) -> str:
    from .perf import drawdown

    valid = [s for s in series if s.equity is not None and len(s.equity) >= 2]
    if not valid:
        return _placeholder("Drawdown chart will appear once enough history accrues.")
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for s in valid:
            dd = drawdown(s.equity) * 100.0
            fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=s.name,
                          fill="tozeroy" if s.is_portfolio else None,
                          line=dict(width=2.0 if s.is_portfolio else 1.0)))
        fig.update_layout(height=360, title="Drawdown (%)", hovermode="x unified", **_DARK)
        fig.update_yaxes(gridcolor="#21262d"); fig.update_xaxes(gridcolor="#21262d")
        return _to_div(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"Drawdown chart failed: {exc}")


def sector_bar(rows: list) -> str:
    """Sector ranking by relative strength vs SPY. rows: list[SectorRow]."""
    if not rows:
        return _placeholder("Sector ranking unavailable — sector ETF data missing.")
    try:
        import plotly.graph_objects as go

        rows = sorted(rows, key=lambda r: r.rs_vs_spy)
        colors = ["#137333" if r.rs_vs_spy >= 0 else "#c5221f" for r in rows]
        fig = go.Figure(go.Bar(
            x=[r.rs_vs_spy for r in rows], y=[f"{r.name} ({r.etf})" for r in rows],
            orientation="h", marker_color=colors,
            text=[f"{r.rs_vs_spy:+.1f}%" for r in rows], textposition="auto"))
        fig.update_layout(height=460, title="Sector Relative Strength vs SPY (1M)", **_DARK)
        fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#484f58")
        return _to_div(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"Sector bar failed: {exc}")


def rrg_scatter(rows: list) -> str:
    """StockCharts-style Relative Rotation Graph quadrant scatter."""
    pts = [r for r in rows if r.rrg_x and r.rrg_y]
    if not pts:
        return _placeholder("RRG quadrant chart will populate once ETF history is available.")
    try:
        import plotly.graph_objects as go

        qcolor = {"Leading": "#137333", "Weakening": "#b06000",
                  "Lagging": "#c5221f", "Improving": "#1f6feb"}
        fig = go.Figure()
        for r in pts:
            fig.add_trace(go.Scatter(
                x=[r.rrg_x], y=[r.rrg_y], mode="markers+text", text=[r.etf],
                textposition="top center", name=r.etf, showlegend=False,
                marker=dict(size=14, color=qcolor.get(r.quadrant, "#8b949e"),
                            line=dict(width=1, color="#0e1117")),
                hovertemplate=(f"<b>{r.name} ({r.etf})</b><br>Quadrant: {r.quadrant}"
                               f"<br>RS: %{{x:.1f}}<br>Momentum: %{{y:.1f}}"
                               f"<br>1D {r.ret.get('1D', float('nan'))*100:+.1f}%  "
                               f"1W {r.ret.get('1W', float('nan'))*100:+.1f}%  "
                               f"1M {r.ret.get('1M', float('nan'))*100:+.1f}%<extra></extra>")))
        xs = [r.rrg_x for r in pts]; ys = [r.rrg_y for r in pts]
        x0, x1 = min(xs + [100]) - 1, max(xs + [100]) + 1
        y0, y1 = min(ys + [100]) - 1, max(ys + [100]) + 1
        fig.add_hline(y=100, line_color="#484f58"); fig.add_vline(x=100, line_color="#484f58")
        fig.add_annotation(x=x1, y=y1, text="Leading", showarrow=False, font=dict(color="#137333"))
        fig.add_annotation(x=x1, y=y0, text="Weakening", showarrow=False, font=dict(color="#b06000"))
        fig.add_annotation(x=x0, y=y0, text="Lagging", showarrow=False, font=dict(color="#c5221f"))
        fig.add_annotation(x=x0, y=y1, text="Improving", showarrow=False, font=dict(color="#1f6feb"))
        fig.update_layout(height=480, title="Relative Rotation Graph (vs SPY)",
                          xaxis_title="Relative Strength", yaxis_title="Momentum", **_DARK)
        fig.update_yaxes(gridcolor="#21262d"); fig.update_xaxes(gridcolor="#21262d")
        return _to_div(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"RRG chart failed: {exc}")


def sankey(pairs: list, title: str = "Estimated Smart Money Rotation Proxy") -> str:
    """Sankey of outflow→inflow sector pairs. pairs: [(src, dst, value)]."""
    if not pairs:
        return _placeholder("No meaningful rotation flows above threshold right now.")
    try:
        import plotly.graph_objects as go

        nodes = list(dict.fromkeys([p[0] for p in pairs] + [p[1] for p in pairs]))
        idx = {n: i for i, n in enumerate(nodes)}
        ncolors = ["#c5221f" if "▼" in n else "#137333" for n in nodes]
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, pad=16, thickness=16, color=ncolors,
                      line=dict(color="#0e1117", width=0.5)),
            link=dict(source=[idx[s] for s, _, _ in pairs],
                      target=[idx[d] for _, d, _ in pairs],
                      value=[v for _, _, v in pairs],
                      color="rgba(139,148,158,0.35)")))
        fig.update_layout(
            height=420, **_DARK,
            title=dict(text=(f"{title}<br><sup>Proxy from relative strength, volume, "
                             "MFI/OBV & momentum — NOT actual dollar flow.</sup>")))
        return _to_div(fig)
    except Exception as exc:  # noqa: BLE001
        return _placeholder(f"Sankey chart failed: {exc}")
