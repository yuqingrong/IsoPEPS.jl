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
const PAPER_LARGE_AXIS_LABELSIZE = 12
const PAPER_LARGE_TICKLABELSIZE = 11
const PAPER_LARGE_LEGEND_LABELSIZE = 10
const PAPER_LEGEND_ROWGAP = 2
const PAPER_LEGEND_COLGAP = 4
const PAPER_LEGEND_PATCHSIZE = (12, 10)
const PAPER_LEGEND_PATCHLABELGAP = 2

math_label(text::AbstractString) = Makie.LaTeXString(String(text))

const ENERGY_PER_SITE_LABEL =
    math_label(raw"\mathit{E}/\mathit{N}_{\mathrm{site}}")
const FIELD_LABEL = math_label(raw"\mathit{g}")
const J2_OVER_J1_LABEL =
    math_label(raw"\mathit{J}_2/\mathit{J}_1")
const CORRELATION_LENGTH_LABEL = math_label(raw"\xi")
const M2_LABEL = math_label(raw"\mathit{M}^2(\mathbf{q})")
const QX_LABEL = rich("q", subscript("x"); font=:italic)
const QY_LABEL = rich("q", subscript("y"); font=:italic)
const X_EXPECTATION_LABEL = math_label(raw"\langle \mathit{X}\rangle")
const Z_EXPECTATION_LABEL = math_label(raw"\langle \mathit{Z}\rangle")
const MAGNETISATION_LABEL = "Magnetisation"

"""
    paper_theme()

Quantum-journal-style Makie theme: Computer Modern serif text and compact
margins. Mathematical labels should use `math_label` so physical quantities
are italic while explicit `\\mathrm{...}` descriptions and numeric tick
labels remain upright. Apply with `set_theme!(paper_theme())` or
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
            xgridvisible = false, ygridvisible = false,
            spinewidth = 0.8,
            xtickwidth = 0.8, ytickwidth = 0.8,
        ),
        Legend = (
            framevisible = true, framewidth = 0.5,
            labelsize = PAPER_LEGEND_LABELSIZE, padding = (3, 3, 3, 3),
            rowgap = PAPER_LEGEND_ROWGAP,
            colgap = PAPER_LEGEND_COLGAP,
            patchsize = PAPER_LEGEND_PATCHSIZE,
            patchlabelgap = PAPER_LEGEND_PATCHLABELGAP,
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

function m2_phase_annotations(::Real)
    representative_points = (0.2, 0.55, 0.8)
    [
        (x=representative_points[1], label_x=representative_points[1],
         y=0.62, marker_y=0.20,
         label="Néel", color=:black, align=(:center, :center)),
        (x=representative_points[2], label_x=representative_points[2],
         y=0.62, marker_y=0.20,
         label="VBS", color=:blue, align=(:center, :center)),
        (x=representative_points[3], label_x=representative_points[3],
         y=0.62, marker_y=0.20,
         label="Stripe", color=:black, align=(:center, :center)),
    ]
end

function add_paper_legend!(ax::Axis; position=:rt, nbanks::Int=1,
                           labelsize::Real=PAPER_LEGEND_LABELSIZE)
    axislegend(ax;
               position=position,
               nbanks=nbanks,
               labelsize=labelsize,
               rowgap=PAPER_LEGEND_ROWGAP,
               colgap=PAPER_LEGEND_COLGAP,
               patchsize=PAPER_LEGEND_PATCHSIZE,
               patchlabelgap=PAPER_LEGEND_PATCHLABELGAP,
               padding=(1, 1, 1, 1),
               margin=(1, 1, 1, 1),
               framevisible=false)
end
