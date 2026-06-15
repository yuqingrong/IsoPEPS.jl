# Visualization: plotting and fitting functions

# ============================================================================
# Paper-quality theme
# ============================================================================

const PAPER_FIGSIZE = (246, 170)         # PRL/PRX single column (3.375 in × 2.36 in)
const PAPER_FIGSIZE_WIDE = (510, 200)    # PRX/Nature double column (7.08 in × 2.78 in)
const PAPER_FONT = :regular
const PAPER_FONTSIZE = 10
const PAPER_AXIS_LABELSIZE = 10
const PAPER_TICKLABELSIZE = 9
const PAPER_TITLESIZE = 11
const PAPER_LEGEND_LABELSIZE = 8

math_label(text::AbstractString) = Makie.LaTeXString(String(text))

const ENERGY_PER_SITE_LABEL =
    math_label(raw"\mathit{E}/\mathit{N}_{\mathrm{site}}")
const FIELD_LABEL = math_label(raw"\mathit{g}")
const J2_OVER_J1_LABEL =
    math_label(raw"\mathit{J}_2/\mathit{J}_1")
const CORRELATION_LENGTH_LABEL = math_label(raw"\xi")
const M2_LABEL = math_label(raw"\mathit{M}^2(\mathbf{q})")
const QX_LABEL = math_label(raw"q_x")
const QY_LABEL = math_label(raw"q_y")
const X_EXPECTATION_LABEL = math_label(raw"\langle \mathit{X}\rangle")
const Z_EXPECTATION_LABEL = math_label(raw"\langle \mathit{Z}\rangle")
const MAGNETISATION_LABEL = "Magnetisation"

"""
    paper_theme()

Quantum-journal-style Makie theme: Computer Modern serif text, compact margins,
and light grids. Mathematical labels should use `math_label` so physical
quantities are italic while explicit `\\mathrm{...}` descriptions and numeric
tick labels remain upright. Apply with `set_theme!(paper_theme())` or
`with_theme(paper_theme()) do ... end`.
"""
function paper_theme()
    Theme(
        fonts = Makie.theme_latexfonts().fonts,
        fontsize = PAPER_FONTSIZE,
        font = PAPER_FONT,
        figure_padding = 6,
        palette = (color = [:steelblue, :firebrick, :seagreen, :darkorange,
                            :purple, :saddlebrown, :hotpink, :teal, :gray],),
        Axis = (
            xlabelsize = PAPER_AXIS_LABELSIZE, ylabelsize = PAPER_AXIS_LABELSIZE,
            xticklabelsize = PAPER_TICKLABELSIZE, yticklabelsize = PAPER_TICKLABELSIZE,
            titlesize = PAPER_TITLESIZE, titlefont = :regular,
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

function m2_phase_annotations(ymax::Real;
                              ranges=((0.0, 0.4), (0.4, 0.6), (0.6, 1.0)))
    centers = ((ranges[1][1] + ranges[1][2]) / 2,
               (ranges[2][1] + ranges[2][2]) / 2,
               (ranges[3][1] + ranges[3][2]) / 2)
    [
        (x=centers[1], y=0.72, range=ranges[1], bar_y=0.28,
         tick_low=0.12, tick_high=0.42, label="Neel order", align=(:center, :center)),
        (x=centers[2], y=0.72, range=ranges[2], bar_y=0.28,
         tick_low=0.12, tick_high=0.42, label="VBS", align=(:center, :center)),
        (x=centers[3], y=0.72, range=ranges[3], bar_y=0.28,
         tick_low=0.12, tick_high=0.42, label="Stripe order", align=(:center, :center)),
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
