#import "@preview/cetz:0.3.2"

#set document(title: "IsoPEPS.jl - Lattice and Ansatz Topology")
#set page(margin: 1.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[Lattice and Ansatz Topology]
  #v(0.3cm)
  #text(size: 12pt)[IsoPEPS.jl Implementation Details]
]

#v(0.5cm)

= Hamiltonian Lattice: Infinite Cylinder

The Transverse Field Ising Model is defined on a 2D square lattice wrapped into an *infinite cylinder*:
- Circumference: `row` sites (periodic boundary conditions)
- Length: infinite (translation invariant)

#figure(
  cetz.canvas({
    import cetz.draw: *

    let rows = 4
    let cols = 6
    let spacing = 1.0

    // Draw lattice sites and bonds
    for i in range(rows) {
      for j in range(cols) {
        let x = j * spacing
        let y = i * spacing

        // Horizontal bonds (ZZ interaction)
        if j < cols - 1 {
          line((x, y), (x + spacing, y), stroke: blue + 1.5pt)
        }

        // Vertical bonds (ZZ interaction) - with periodic BC indication
        if i < rows - 1 {
          line((x, y), (x, y + spacing), stroke: blue + 1.5pt)
        }
      }
    }

    // Periodic boundary - dashed lines connecting top and bottom
    for j in range(cols) {
      let x = j * spacing
      line((x, (rows - 1) * spacing), (x, (rows - 1) * spacing + 0.4), stroke: (paint: blue, dash: "dashed", thickness: 1.5pt))
      line((x, -0.4), (x, 0), stroke: (paint: blue, dash: "dashed", thickness: 1.5pt))
    }

    // Draw sites on top
    for i in range(rows) {
      for j in range(cols) {
        let x = j * spacing
        let y = i * spacing
        circle((x, y), radius: 0.15, fill: red.lighten(30%), stroke: red)
      }
    }

    // Infinite extension arrows
    line((cols * spacing - 0.3, 1.5), (cols * spacing + 0.5, 1.5), mark: (end: ">"), stroke: 1pt)
    line((-0.7, 1.5), (-0.3, 1.5), mark: (start: ">"), stroke: 1pt)
    content((cols * spacing + 0.3, 1.8), text(size: 8pt)[$infinity$])
    content((-0.5, 1.8), text(size: 8pt)[$infinity$])

    // Labels
    content((-1.2, 1.5), text(size: 9pt)[row])
    content((2.5, -1.0), text(size: 9pt)[columns (infinite)])

    // Periodic BC annotation
    bezier((-0.5, -0.6), (-0.5, 3.6), (-1.5, 1.5), stroke: (paint: gray, dash: "dotted"))
    content((-1.8, 1.5), text(size: 8pt, fill: gray)[PBC])
  }),
  caption: [2D square lattice on infinite cylinder. Blue bonds represent $Z_i Z_j$ interactions. Red sites carry transverse field $g X_i$. Dashed lines indicate periodic boundary conditions (PBC) in vertical direction.]
) <fig:lattice>

#v(0.3cm)

*Hamiltonian:* $ H = -g sum_i X_i - J sum_(⟨i,j⟩) Z_i Z_j $

The sum $⟨i,j⟩$ runs over nearest-neighbor pairs:
- *Vertical bonds*: sites $(i, j)$ and $(i+1 mod "row", j)$
- *Horizontal bonds*: sites $(i, j)$ and $(i, j+1)$

#pagebreak()

= PEPS Tensor Network Structure

Each site carries a rank-5 tensor $A^p_(u d l r)$ with indices:
- $p$: physical index (dimension 2 for spin-1/2)
- $u, d, l, r$: virtual indices connecting to neighbors

#figure(
  cetz.canvas({
    import cetz.draw: *

    let cx = 0
    let cy = 0
    let len = 0.8

    // Central tensor
    rect((cx - 0.3, cy - 0.3), (cx + 0.3, cy + 0.3), fill: orange.lighten(50%), stroke: orange)
    content((cx, cy), text(weight: "bold")[$A$])

    // Virtual indices
    line((cx, cy + 0.3), (cx, cy + len + 0.3), stroke: 1.5pt, mark: (end: ">"))
    content((cx + 0.25, cy + len + 0.1), text(size: 9pt)[$u$])

    line((cx, cy - 0.3), (cx, cy - len - 0.3), stroke: 1.5pt, mark: (end: ">"))
    content((cx + 0.25, cy - len - 0.1), text(size: 9pt)[$d$])

    line((cx - 0.3, cy), (cx - len - 0.3, cy), stroke: 1.5pt, mark: (end: ">"))
    content((cx - len - 0.1, cy + 0.25), text(size: 9pt)[$l$])

    line((cx + 0.3, cy), (cx + len + 0.3, cy), stroke: 1.5pt, mark: (end: ">"))
    content((cx + len + 0.1, cy + 0.25), text(size: 9pt)[$r$])

    // Physical index (out of plane - shown as diagonal)
    line((cx + 0.2, cy + 0.2), (cx + 0.7, cy + 0.7), stroke: (paint: red, thickness: 1.5pt), mark: (end: ">"))
    content((cx + 0.9, cy + 0.7), text(size: 9pt, fill: red)[$p$])

    // Legend
    content((3, 0.5), text(size: 9pt)[Virtual bond (dimension $D$)])
    line((2.2, 0.5), (2.8, 0.5), stroke: 1.5pt)
    content((3, 0), text(size: 9pt, fill: red)[Physical index (dimension 2)])
    line((2.2, 0), (2.8, 0), stroke: (paint: red, thickness: 1.5pt))
  }),
  caption: [Single PEPS tensor with 5 indices: physical ($p$) and four virtual ($u$, $d$, $l$, $r$).]
) <fig:tensor>

#v(0.5cm)

= IsoPEPS: Isometric Tensor Construction

In IsoPEPS, tensors are *isometries* derived from unitary gates. A unitary $U$ acting on `nqubits` is reshaped:

#figure(
  cetz.canvas({
    import cetz.draw: *

    // Unitary gate
    rect((-0.5, -1), (0.5, 1), fill: green.lighten(70%), stroke: green)
    content((0, 0), text(weight: "bold")[$U$])

    // Input qubits (left side)
    for i in range(5) {
      let y = -0.8 + i * 0.4
      line((-1.2, y), (-0.5, y), stroke: 1pt)
      content((-1.5, y), text(size: 8pt)[$i_#(i+1)$])
    }

    // Output qubits (right side)
    for i in range(5) {
      let y = -0.8 + i * 0.4
      line((0.5, y), (1.2, y), stroke: 1pt)
      content((1.5, y), text(size: 8pt)[$o_#(i+1)$])
    }

    // Arrow
    line((2.2, 0), (3.3, 0), mark: (end: ">"), stroke: 1.5pt)
    content((2.75, 0.3), text(size: 9pt)[reshape])

    // Reshaped tensor
    let tx = 5
    rect((tx - 0.4, -0.4), (tx + 0.4, 0.4), fill: orange.lighten(50%), stroke: orange)
    content((tx, 0), text(weight: "bold")[$A$])

    // Physical (measured qubit o1)
    line((tx, 0.4), (tx, 1.0), stroke: (paint: red, thickness: 1.5pt), mark: (end: ">"))
    content((tx + 0.25, 0.85), text(size: 8pt, fill: red)[$o_1$])

    // Up/Down (remaining outputs grouped)
    line((tx, -0.4), (tx, -1.0), stroke: 1.5pt, mark: (end: ">"))
    content((tx - 0.5, -0.85), text(size: 8pt)[$o_(2..k)$])

    // Left (inputs grouped)
    line((tx - 0.4, 0), (tx - 1.1, 0), stroke: 1.5pt, mark: (end: ">"))
    content((tx - 1.0, 0.25), text(size: 8pt)[$i_(1..k)$])

    // Right (remaining inputs)
    line((tx + 0.4, 0), (tx + 1.1, 0), stroke: 1.5pt, mark: (end: ">"))
    content((tx + 1.0, 0.25), text(size: 8pt)[$i_(k..n)$])
  }),
  caption: [Unitary gate $U$ (5 qubits shown) reshaped into PEPS tensor $A$. Output qubit $o_1$ becomes the physical index (to be measured).]
) <fig:reshape>

#v(0.3cm)

*Isometry property*: Since $U^dagger U = I$, the tensor $A$ satisfies:
$ sum_p A^p_(u d l r) (A^p_(u' d' l' r'))^* = delta_(u u') delta_(d d') delta_(l l') delta_(r r') $

This ensures the PEPS contraction is well-conditioned.

#pagebreak()

= Quantum Channel: Iterative State Construction

The key insight of IsoPEPS is building the 2D state *column by column* via a quantum channel:

#figure(
  cetz.canvas({
    import cetz.draw: *

    let col_spacing = 2.5
    let row_spacing = 0.8
    let rows = 3

    // Column 1 (previous)
    content((0, rows * row_spacing + 0.5), text(size: 9pt, weight: "bold")[Col $n-1$])
    for i in range(rows) {
      circle((0, i * row_spacing), radius: 0.2, fill: blue.lighten(60%), stroke: blue)
    }
    // Environment label
    content((0, -0.8), text(size: 8pt, fill: blue)[Environment $rho$])

    // Arrow to channel
    line((0.5, rows * row_spacing / 2), (1.3, rows * row_spacing / 2), mark: (end: ">"), stroke: 1pt)

    // Quantum channel box
    rect((1.5, -0.5), (4.0, rows * row_spacing + 0.5), fill: green.lighten(80%), stroke: green)
    content((2.75, rows * row_spacing + 0.2), text(size: 9pt, weight: "bold")[Quantum Channel])

    // Inside channel: gates and measurements
    for i in range(rows) {
      let y = i * row_spacing
      // Input from environment
      line((1.5, y), (2.0, y), stroke: 1pt)
      // Fresh ancilla
      circle((2.2, y + 0.35), radius: 0.12, fill: yellow, stroke: orange)
      content((2.2, y + 0.65), text(size: 7pt)[$|0⟩$])
      line((2.2, y + 0.23), (2.2, y + 0.1), stroke: 0.5pt)
      // Gate
      rect((2.4, y - 0.25), (3.0, y + 0.25), fill: orange.lighten(50%), stroke: orange)
      content((2.7, y), text(size: 8pt)[$U_#(i+1)$])
      // Output to next column
      line((3.0, y), (3.4, y), stroke: 1pt)
      // Measurement
      rect((3.4, y - 0.15), (3.7, y + 0.15), fill: red.lighten(60%), stroke: red)
      content((3.55, y), text(size: 7pt)[M])
      // Measured output (classical)
      line((3.7, y), (4.0, y), stroke: (dash: "dashed"))
    }

    // Arrow to next column
    line((4.2, rows * row_spacing / 2), (5.0, rows * row_spacing / 2), mark: (end: ">"), stroke: 1pt)

    // Column 2 (next)
    content((5.5, rows * row_spacing + 0.5), text(size: 9pt, weight: "bold")[Col $n$])
    for i in range(rows) {
      circle((5.5, i * row_spacing), radius: 0.2, fill: blue.lighten(60%), stroke: blue)
    }
    content((5.5, -0.8), text(size: 8pt, fill: blue)[New $rho'$])

    // Samples output
    line((4.5, -0.3), (4.5, -1.0), mark: (end: ">"), stroke: (dash: "dashed"))
    content((4.5, -1.3), text(size: 8pt)[Samples $Z_i$ or $X_i$])
  }),
  caption: [Quantum channel iteration. For each column: (1) add fresh ancilla $|0⟩$ for each row, (2) apply unitary gates $U_i$, (3) measure and remove one qubit per row. The unmeasured qubits form the new environment.]
) <fig:channel>

#v(0.3cm)

*Algorithm* (`sample_quantum_channel`):
1. Initialize environment state $rho$ with $(("nqubits"-1)/2) dot ("row"+1)$ qubits
2. For each column iteration:
   - For each row $j = 1, ..., "row"$:
     - Add ancilla qubit $|0⟩$
     - Apply gate $U_j$ to selected qubits
     - Measure qubit 1 in $Z$ (or $X$) basis, record outcome
3. After convergence, collect measurement statistics

#pagebreak()

= Transfer Matrix Structure

The *transfer matrix* $T$ propagates correlations between columns:

#figure(
  cetz.canvas({
    import cetz.draw: *

    let rows = 3
    let spacing = 1.2

    // Left boundary
    for i in range(rows) {
      line((-0.8, i * spacing), (-0.3, i * spacing), stroke: 1.5pt)
      content((-1.1, i * spacing), text(size: 8pt)[$l_#(i+1)$])
    }
    // Periodic connection (left side)
    bezier((-0.5, -0.3), (-0.5, rows * spacing - spacing + 0.3), (-1.2, (rows - 1) * spacing / 2), stroke: (paint: gray, dash: "dotted"))

    // Ket tensors (top row)
    for i in range(rows) {
      let y = i * spacing
      rect((-0.3, y - 0.3), (0.3, y + 0.3), fill: orange.lighten(50%), stroke: orange)
      content((0, y), text(size: 8pt)[$A_#(i+1)$])

      // Physical index going up
      line((0, y + 0.3), (0, y + 0.6), stroke: (paint: red, thickness: 1.5pt))

      // Vertical connections between tensors
      if i < rows - 1 {
        line((0, y + 0.6), (0, y + spacing - 0.6), stroke: 1pt)
      }
    }

    // Contract physical indices
    for i in range(rows) {
      let y = i * spacing
      line((0, y + 0.6), (0.8, y + 0.6), stroke: (paint: red, thickness: 1pt))
      line((0.8, y + 0.6), (0.8, y + 0.9), stroke: (paint: red, thickness: 1pt))
      line((0.8, y + 0.9), (1.6, y + 0.9), stroke: (paint: red, thickness: 1pt))
      line((1.6, y + 0.9), (1.6, y + 0.6), stroke: (paint: red, thickness: 1pt))
      line((1.6, y + 0.6), (2.4, y + 0.6), stroke: (paint: red, thickness: 1pt))
    }

    // Bra tensors (bottom row, conjugate)
    for i in range(rows) {
      let y = i * spacing
      rect((2.1, y - 0.3), (2.7, y + 0.3), fill: purple.lighten(60%), stroke: purple)
      content((2.4, y), text(size: 8pt)[$A_#(i+1)^*$])

      // Physical index
      line((2.4, y + 0.3), (2.4, y + 0.6), stroke: (paint: red, thickness: 1.5pt))

      // Vertical connections
      if i < rows - 1 {
        line((2.4, y + 0.6), (2.4, y + spacing - 0.6), stroke: 1pt)
      }
    }

    // Right boundary
    for i in range(rows) {
      line((2.7, i * spacing), (3.2, i * spacing), stroke: 1.5pt)
      content((3.5, i * spacing), text(size: 8pt)[$r_#(i+1)$])
    }
    // Periodic connection (right side)
    bezier((2.9, -0.3), (2.9, rows * spacing - spacing + 0.3), (3.6, (rows - 1) * spacing / 2), stroke: (paint: gray, dash: "dotted"))

    // Labels
    content((1.2, -0.8), text(size: 10pt, weight: "bold")[Transfer Matrix $T$])
    content((1.2, rows * spacing + 0.8), text(size: 8pt, fill: red)[Physical indices contracted])
  }),
  caption: [Transfer matrix $T$ for `row=3`. Ket tensors $A$ (orange) and bra tensors $A^*$ (purple) are contracted along physical indices (red). Left/right virtual indices remain open. Dotted lines indicate periodic boundary conditions.]
) <fig:transfer>

#v(0.3cm)

*Spectral properties*:
- Largest eigenvalue: $lambda_1 = 1$ (normalization)
- Spectral gap: $Delta = -log|lambda_2|$ (correlation decay rate)
- Fixed point: $rho = $ leading eigenvector (infinite-width limit)

#pagebreak()

= Complete PEPS Network on Cylinder

#figure(
  cetz.canvas({
    import cetz.draw: *

    let rows = 4
    let cols = 5
    let xspace = 1.4
    let yspace = 1.0

    // Draw all tensors
    for i in range(rows) {
      for j in range(cols) {
        let x = j * xspace
        let y = i * yspace

        // Tensor box
        rect((x - 0.25, y - 0.25), (x + 0.25, y + 0.25), fill: orange.lighten(50%), stroke: orange)

        // Physical index (small line going out of page)
        circle((x + 0.15, y + 0.15), radius: 0.08, fill: red.lighten(50%), stroke: red)

        // Horizontal bond (right)
        if j < cols - 1 {
          line((x + 0.25, y), (x + xspace - 0.25, y), stroke: blue + 1pt)
        }

        // Vertical bond (up)
        if i < rows - 1 {
          line((x, y + 0.25), (x, y + yspace - 0.25), stroke: blue + 1pt)
        }
      }
    }

    // Periodic BC in vertical direction (dashed)
    for j in range(cols) {
      let x = j * xspace
      line((x, (rows - 1) * yspace + 0.25), (x, (rows - 1) * yspace + 0.5), stroke: (paint: blue, dash: "dashed"))
      line((x, -0.5), (x, -0.25), stroke: (paint: blue, dash: "dashed"))
    }

    // Infinite extension
    line((cols * xspace - 0.5, (rows - 1) * yspace / 2), (cols * xspace + 0.3, (rows - 1) * yspace / 2), mark: (end: ">"))
    content((cols * xspace + 0.1, (rows - 1) * yspace / 2 + 0.4), text(size: 8pt)[$dots.c$])

    // Column labels
    for j in range(cols) {
      content((j * xspace, -1.0), text(size: 8pt)[col #(j+1)])
    }

    // Row labels
    for i in range(rows) {
      content((-0.8, i * yspace), text(size: 8pt)[row #(i+1)])
    }

    // Annotations
    content((2.8, rows * yspace + 0.8), text(size: 9pt)[#text(fill: orange)[Orange]: PEPS tensors $A^p_(u d l r)$])
    content((2.8, rows * yspace + 0.4), text(size: 9pt)[#text(fill: blue)[Blue]: Virtual bonds (dimension $D$)])
    content((2.8, rows * yspace + 0.0), text(size: 9pt)[#text(fill: red)[Red]: Physical indices (dimension 2)])
  }),
  caption: [Full PEPS tensor network on infinite cylinder with `row=4`. Each tensor connects to 4 neighbors via virtual bonds. Physical indices (red dots) carry the quantum spin degrees of freedom.]
) <fig:full-peps>

#v(0.5cm)

= Summary of Dimensions

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: (left, center, left),
  [*Parameter*], [*Symbol*], [*Description*],
  [`row`], [$r$], [Cylinder circumference (number of rows)],
  [`nqubits`], [$n$], [Qubits per unitary gate],
  [Physical dim], [$d = 2$], [Spin-1/2 Hilbert space],
  [Bond dim], [$D = 2^((n-1)/2)$], [Virtual index dimension],
  [Environment], [$(n-1)/2 dot (r+1)$], [Qubits in quantum channel state],
)

The IsoPEPS ansatz achieves an effective bond dimension $D = 2^((n-1)/2)$ determined by the gate size, while maintaining efficient simulation through the quantum channel formulation.
