from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numbers
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def shade(ax, segs, rgba):
    """
    Add vertical shading to a time-series plot.
    
    :param ax: axis object
    :param segs: list of segment starts and ends
    :param rgba: rgba shading value
    """
    for seg in segs:
        ax.axvspan(*seg, color=rgba)
    
    return ax


def unity_line(ax, x_min, x_max, **kwargs):
    """
    Plot a unity line on an axis object.
    """
    xs = [x_min, x_max]
    ax.plot(xs, xs, **kwargs)
    
    return ax


def get_line(x, y):
    nnan_mask = (~np.isnan(x)) & (~np.isnan(y))
    slp, icpt, r, p, stderr = stats.linregress(x[nnan_mask], y[nnan_mask])
    
    x_ln = np.array([np.nanmin(x), np.nanmax(x)])
    y_ln = slp*x_ln + icpt
    
    return x_ln, y_ln, (slp, icpt, r, p, stderr)

        
def set_font_size(ax, font_size, legend_font_size=None):
    """Set fontsize of all axis text objects to specified value."""

    texts = ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())

    for text in texts:

        text.set_fontsize(font_size)

    if legend_font_size is None:

        legend_font_size = font_size

    legend = ax.get_legend()

    if legend:

        for text in legend.get_texts():

            text.set_fontsize(legend_font_size)


def get_n_colors(n, colormap='rainbow'):
    """
    Return a list of colors equally spaced over a color map.
    :param n: number of colors
    :param colormap: colormap to use
    :return: list of colors that can be passed directly to color argument of plotting
    """

    return getattr(cm, colormap)(np.linspace(0, 1, n))


def fast_fig(n_ax, ax_size, fig_w=15):
    """Quickly make figure and axes objects from number of axes and ax size (h, w)."""
    n_col = int(round(fig_w/ax_size[1]))
    n_row = int(np.ceil(n_ax/n_col))
    
    fig_h = n_row*ax_size[0]
    
    fig, axs = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h), tight_layout=True, squeeze=False)
    return fig, axs.flatten()


def conditional_dependency_diagram(ax, nodes, labels, cd_info, lw_scale, text_font_size):
    """
    Make a diagram of conditional dependencies among different random variables
    :param ax: axis object
    :param nodes: names of random variables as referred to in cd_info dict
    :param labels: labels to print on plotted nodes
    :param cd_info: information about conditional dependencies between random variables
    :param lw_scale: factor to scale linewidths by
    :param text_font_size: font size for textual annotations
    """

    scale = 10
    scatter_size = 1000

    shrink = 0.8
    offset = 0.05
    text_offset = 0.07
    head_width = .1

    node_locations = dict(zip(nodes, [[1, 0], [0, 1], [-1, 0], [0, -1]]))

    for node, label in zip(nodes, labels):
        node_location = node_locations[node]

        ax.scatter(
            scale * node_location[0], scale * node_location[1],
            s=scatter_size, c='k', lw=0, zorder=1)

        ax.text(scale * node_location[0], scale * node_location[1], label,
            ha='center', va='center', color='yellow', weight='bold',
            size='x-large', zorder=2)

    # draw connections

    for src in nodes:

        src_loc = scale * np.array(node_locations[src])

        for targ in nodes:

            if targ == src:

                continue

            # skip if p-value is too high

            if cd_info[targ][src]['color'] is None:

                continue

            # get linewidth

            lw = cd_info[targ][src]['lw'] * lw_scale

            # get color

            color = cd_info[targ][src]['color']

            targ_loc = scale * np.array(node_locations[targ])

            # get displacement and direction from src to targ

            arrow_disp = targ_loc - src_loc
            arrow_dir = arrow_disp / np.linalg.norm(arrow_disp)

            # get 90 rotation for direction of offsetting arrow

            offset_dir = np.array([arrow_dir[1], -arrow_dir[0]])

            # get arrow start and end points

            midpoint = 0.5 * (targ_loc + src_loc) + scale * offset * offset_dir

            half_disp = arrow_disp / 2

            start = midpoint - shrink * half_disp
            end = midpoint + shrink * half_disp

            ax.arrow(
                start[0], start[1], end[0] - start[0], end[1] - start[1],
                head_width=scale * head_width, fc=color, ec=color,
                length_includes_head=True, lw=lw, zorder=0)

            text_center = midpoint + scale * text_offset * offset_dir

            text_rotation = np.arctan(arrow_dir[1] / arrow_dir[0])
            text_rotation *= (180 / np.pi)

            coefs = '{0:0.3}    ,    {1:0.3}'.format(
                cd_info[targ][src]['coef_paused'], cd_info[targ][src]['coef_walking'],
            )

            ax.text(text_center[0], text_center[1], coefs, rotation=text_rotation,
                ha='center', va='center', fontsize=text_font_size)

    ax.set_aspect('equal')
    ax.set_axis_off()


def stacked_overlay(
        ax, xs, data, colors=None, labels=None, refs=None, z_orders=None,
        scales='std', spacing=6, x_lim=None, align='mean', masks=None,
        colormap='hsv', **kwargs):
    """
    Plot multiple sets of overlayed time-series on the same axis object,
    with the sets of time-series stacked on top of one another so that
    they can be easily visualized.
    :param ax: axis object
    :param xs: list of x-coordinates
    :param data: list of dicts of variables to plot
    :param colors: dict of colors
    :param labels: dict of labels
    :param refs: dict of reference lines
    :param scales: how to scale variables (can also be dict)
    :param spacing: how far to separate stacked traces
    :param x_lim: x limits
    :param align: how to align variables
    :param colormap: what color map to use to for colors

    :return: list of handles, list of y-ticks
    """
    if isinstance(xs[0], numbers.Number):
        xs = [xs] * len(data)

    ## format some settings
    variables = sum([d.keys() for d in data], [])

    # make sure "colors" is dictionary with one key per data variable
    colors = {} if colors is None else colors
    vs_no_color = [v not in colors for v in variables]
    missing_colors = get_n_colors(len(vs_no_color), colormap=colormap)
    missing_colors = {v: c for v, c in zip(vs_no_color, missing_colors)}

    for v, c in missing_colors.items():
        colors[v] = c

    # make sure "labels" is dictionary with one key per data variable
    labels = {} if labels is None else labels
    for v in variables:
        if v not in labels:
            labels[v] = v

    # make sure "refs" is dictionary with one key per data variable
    refs = {} if refs is None else refs
    for v in variables:
        if v not in refs:
            refs[v] = []

    # make sure "z_orders" is a dictionary with one key per data variable
    z_orders = {} if z_orders is None else z_orders
    for v in variables:
        if v not in z_orders:
            z_orders[v] = 0

    # make sure "scales" is dictionary with one key per data variable
    if isinstance(scales, basestring) or type(scales) in [float, int, tuple]:
        scales = {variable: scales for variable in variables}
    for v in variables:
        if v not in scales:
            scales[v] = 'std'

    # make sure "align" is dictionary with one key per data variable
    if isinstance(align, basestring) or type(align) in [float, int]:
        align = {variable: align for variable in variables}
    for v in variables:
        if v not in align:
            align[v] = 'mean'

    # turn masks into empty dict if None
    masks = {} if masks is None else masks

    plotted_vars = []
    handles = []
    y_ticks = []

    offset = 0

    y_min = np.inf
    y_max = -np.inf

    for xs_, d_dict in zip(xs, data):
        offset -= spacing

        if not isinstance(xs_, dict):
            xs_ = {k: xs_ for k in d_dict}

        ss_pinned = {}
        ms_pinned = {}

        for variable, d in d_dict.items():
            x = xs_[variable]
            c = colors[variable]
            l = labels[variable]
            rs = refs[variable]
            s = scales[variable]
            a = align[variable]
            z_order = z_orders[variable]

            h = None

            if variable in masks:
                diffs = np.diff(np.concatenate([[0], d, [0]]))
                x_ = np.concatenate([x, [x[-1] + x[1] - x[0]]])
                starts = x_[diffs == 1]
                stops = x_[diffs == -1] - (x_[1] - x_[0])

                for start, stop in zip(starts, stops):
                    h = ax.fill_between(
                        [start, stop],
                        [offset - 2, offset - 2],
                        [offset + 2, offset + 2],
                        label=l,
                        **masks[variable])
            else:
                # scale and align data
                if isinstance(s, tuple):
                    idx = s[1]
                else:
                    idx = None

                if idx is not None:
                    s_ = ss_pinned[idx] if idx in ss_pinned else s[0]
                else:
                    s_ = s

                # scale d
                s_ = 1/np.nanstd(d) if s_ == 'std' else s_
                d = d * s_

                if idx is not None:
                    ss_pinned[idx] = s_

                # align d
                if idx is not None:
                    m = ms_pinned[idx] if idx in ms_pinned else None

                if idx is None or m is None:
                    if a == 'mean':
                        m = np.nanmean(d)
                    elif a == 'median':
                        m = np.nanmedian(d)
                    elif a == 'midrange':
                        m = 0.5 * (np.nanmax(d) - np.nanmin(d))
                    elif not isinstance(a, (int, float)):
                        m = 0
                    else:
                        m = a*s_

                if idx is not None:
                    ms_pinned[idx] = m

                d = d - m

                # plot d
                h = ax.plot(
                    x, d + offset, color=c, label=l, zorder=z_order, **kwargs)[0]

                # plot reference lines
                r_min = np.inf
                r_max = -np.inf

                for r in rs:
                    if r is None: continue

                    r_ = r*s_ - m
                    ax.axhline(r_ + offset, color=c, ls='--')

                    r_min = min(r_min, r_ + offset)
                    r_max = max(r_max, r_ + offset)

                y_min = min([y_min, np.nanmin(d) + offset, r_min])
                y_max = max([y_max, np.nanmax(d) + offset, r_max])

            if variable not in plotted_vars and h is not None:
                handles.append(h)
                plotted_vars.append(variable)

        y_ticks.append(offset)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([])

    ax.set_xlim(x_lim)

    y_range = y_max - y_min
    ax.set_ylim(y_min - .1*y_range, y_max + .1*y_range)

    return handles, y_ticks
