# Visualization: plotting and fitting functions

# ============================================================================
# Paper-quality theme
# ============================================================================

const PAPER_FIGSIZE = (246, 170)         # PRL/PRX single column (3.375 in × 2.36 in)
const PAPER_FIGSIZE_WIDE = (510, 200)    # PRX/Nature double column (7.08 in × 2.78 in)
const PAPER_FONT = "Helvetica"
const PAPER_FONTSIZE = 10
const PAPER_AXIS_LABELSIZE = 10
const PAPER_TICKLABELSIZE = 9
const PAPER_TITLESIZE = 11
const PAPER_LEGEND_LABELSIZE = 8

"""
    paper_theme()

Science/PRX-style Makie theme: Helvetica sans-serif, compact margins, framed
legends, light grid. Apply with `set_theme!(paper_theme())` or
`with_theme(paper_theme()) do ... end`.
"""
function paper_theme()
    Theme(
        fontsize = PAPER_FONTSIZE,
        font = PAPER_FONT,
        figure_padding = 6,
        palette = (color = [:steelblue, :firebrick, :seagreen, :darkorange,
                            :purple, :saddlebrown, :hotpink, :teal, :gray],),
        Axis = (
            xlabelsize = PAPER_AXIS_LABELSIZE, ylabelsize = PAPER_AXIS_LABELSIZE,
            xticklabelsize = PAPER_TICKLABELSIZE, yticklabelsize = PAPER_TICKLABELSIZE,
            titlesize = PAPER_TITLESIZE,
            xgridvisible = true, ygridvisible = true,
            xgridcolor = (:gray, 0.25), ygridcolor = (:gray, 0.25),
            xgridwidth = 0.5, ygridwidth = 0.5,
            spinewidth = 0.8,
            xtickwidth = 0.8, ytickwidth = 0.8,
        ),
        Legend = (
            framevisible = true, framewidth = 0.5,
            labelsize = PAPER_LEGEND_LABELSIZE, padding = (3, 3, 3, 3),
            rowgap = 1,
        ),
        Lines = (linewidth = 1.0, cycle = [:color]),
        Scatter = (markersize = 6, strokewidth = 0.5, cycle = [:color]),
        ScatterLines = (linewidth = 1.0, markersize = 6, cycle = [:color]),
        Errorbars = (linewidth = 0.8, whiskerwidth = 4),
    )
end

function compact_reference_label(kind::Symbol, value::Real)
    rounded_value = round(value, digits=4)
    if kind === :pepskit
        return "PEPSKit ($rounded_value)"
    elseif kind === :dmrg
        return "DMRG ($rounded_value)"
    else
        throw(ArgumentError("unknown reference label kind: $kind"))
    end
end

function m2_phase_annotations(ymax::Real)
    [
        (x=0.20, y=0.05, label="Neel order", align=(:center, :center)),
        (x=0.57, y=0.05, label="VBS", align=(:center, :center)),
        (x=0.80, y=0.05, label="Stripe order", align=(:center, :center)),
    ]
end

function add_paper_legend!(ax::Axis; position=:rt, nbanks::Int=1)
    axislegend(ax;
               position=position,
               nbanks=nbanks,
               labelsize=PAPER_LEGEND_LABELSIZE,
               padding=(1, 1, 1, 1),
               margin=(1, 1, 1, 1),
               framevisible=false)
end

