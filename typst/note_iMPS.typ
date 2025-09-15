#import "@preview/touying:0.6.1": *
#import "@preview/touying-simpl-hkustgz:0.1.2": *
#import "@preview/cetz:0.3.1": canvas, draw

// Specify `lang` and `font` for the theme if needed.
#show: hkustgz-theme.with(
  // lang: "zh",
  // font: (
  //   (
  //     name: "Linux Libertine",
  //     covers: "latin-in-cjk",
  //   ),
  //   "Source Han Sans SC",
  //   "Source Han Sans",
  // ),
  config-info(
    title: [Variational iMPS on Quantum Circuits],
    subtitle: [],
    author: [Yuqing Rong],
    date: datetime.today(),
    institution: [HKUST(GZ)],
  ),
)

#let mpscell(pos, name, text: none) = {
  import draw: *
  let a = 1.0
  let (x, y) = pos
  rect((x - a/2, y - a/2), (x + a/2, y + a/2), radius: 20%, fill: white, name: name)
  if text != none{
    content(name, text)
  }
}

#let environment(pos, name, text: none) = {
  import draw: *
  let a = 0.5
  let (x, y) = pos
  circle((x, y), radius: a, fill: white, name: name)
  if text != none{
    content(name, text)
  }
}

#title-slide()

#outline-slide()

= Background

== Quantum many-body challenges
generic quantum state with $N$ sites, $"spin-"d$  has $d^N$ dimensional Hilbert space

$|psi angle.r = sum_(s_1,s_2,...,s_N) psi_(s_1,s_2,...,s_N) |s_1 angle.r |s_2 angle.r...|s_N angle.r, s_n = 1,...,d$

In exact diagonalization, the number of quantum state parameters increases exponentially with $N$.
#grid(
  columns: (70%, 1fr),
  column-gutter: 1em,
  image("images/many-body.jpg", width: 100%),
 text(14pt,[Figure is from Ran Shiju's lecture note.])
)
 
== Entanglement
Decompose a pure state into a superposition of product states through Schmidt decomposition:

$|psi angle.r = sum_(alpha) Lambda_(alpha) |alpha angle.r_A times.circle |alpha angle.r_B$ 

with $angle.l alpha|alpha^' angle.r = delta_(alpha alpha^')$ and $sum_alpha Lambda_(alpha)^2 = 1$

Entanglement entropy: $S = - sum_alpha Lambda_(alpha)^2 log(Lambda_(alpha)^2)$

Area law: for ground states of gapped Hamiltonians in 1D system

$ S(L) = "const" $
#grid(
  columns: (30%, 1fr),
  column-gutter: 1em,
  image("images/entanglement.png", width: 100%),
  text(size: 12pt)[Eisert, J., Cramer, M., Plenio, M.B., 2010. Area laws for the entanglement entropy - a review. Rev. Mod. Phys. 82, 277-306. https://doi.org/10.1103/RevModPhys.82.277]
)
== MPS

#figure(
  canvas(length: 0.8cm, {
  import draw: *
  content((0,0), [$|psi angle.r = $])
  content((2,0.5),[$A^[1]$])
  content((3.5,0.5),[$A^[2]$])
  content((5,0.5),[$A^[3]$])
  content((8,0.5),[$A^[N]$])
  circle((2,0), radius: 0.1, fill: black, name: "A")
  circle((3.5,0), radius: 0.1, fill: black, name: "B")
  circle((5,0), radius: 0.1, fill: black, name: "C")
  content((6.5,0),[...])
  circle((8,0), radius: 0.1, fill: black, name: "D")
  line("A", "B")
  line("B", "C")
  line("C",(6,0))
  line("D",(7,0))
  line("A",(2,-1))
  line("B",(3.5,-1))
  line("C",(5,-1))
  line("D",(8,-1))

  content((12,0),[$A_(alpha,beta)^i = $])
  content((15.5,0.5),[$A$])
  content((14,0),[$alpha$],name: "alpha")
  content((17,0),[$beta$],name: "beta")
  content((15.5,-1.5),[$i$],name:"i")
  circle((15.5,0), radius: 0.1, fill: black, name: "A")
  line("alpha.east", "A")
  line("beta.west", "A")
  line("A", "i.north")

  content((14,-2),$alpha, beta = 1, ..., D$)
  content((14,-3),$i = 1, ..., d$)
  }))

 

Canonical form: Use the gauge degree of freedom to find a convenient representation 
#figure(
  canvas(length: 0.8cm, {
  import draw: *
  content((2,0.5),[$A^[1]$])
  content((3.5,0.5),[$A^[2]$])
  content((5,0.5),[$Lambda^[3]$])
  content((8,0.5),[$B^[N]$])
  circle((2,0), radius: 0.1, fill: black, name: "A")
  circle((3.5,0), radius: 0.1, fill: black, name: "B")
  circle((5,0), radius: 0.1, fill: black, name: "C")
  content((6.5,0),[...])
  circle((8,0), radius: 0.1, fill: black, name: "D")
  line("A", "B")
  line("B", "C")
  line("C",(6,0))
  line("D",(7,0))
  line("A",(2,-1))
  line("B",(3.5,-1))
  line("C",(5,-1))
  line("D",(8,-1))

  circle((12,0), radius: 0.1, fill: black, name: "A")
  circle((12,-1.5), radius: 0.1, fill: black, name: "A^dagger")
  content((12,-2),[$A^dagger$])
  line("A.south", "A^dagger.north")
  line("A.east", (13,0),mark: (end: ">"))
  line("A^dagger.east", (13,-1.5),mark: (end: ">"))
  line((11,0),"A.west",mark: (end: ">"))
  line((11,-1.5),"A^dagger.west",mark: (end: ">"))
  line((11,0),(11,-1.5))
  content((11,0.5),[$A$])
  content((14,-0.7),[$= I$])
  
  circle((17,0), radius: 0.1, fill: black, name: "B")
  circle((17,-1.5), radius: 0.1, fill: black, name: "B^dagger")
  content((17,-2),[$B^dagger$])
  line("B.south", "B^dagger.north")
  line((18,0),"B.east",mark: (end: ">"))
  line((18,-1.5),"B^dagger.east",mark: (end: ">"))
  line("B.west",(16,0),mark: (end: ">"))
  line("B^dagger.west",(16,-1.5),mark: (end: ">"))
  line((18,0),(18,-1.5))
  content((17,0.5),[$B$])
  content((19,-0.7),[$= I$])

  content((24, -0.7),[isometric condition])
  }))

(1) Reduction of the number of variables: $d^N -> N d D^2$ ;

(2) Efficient to get expectation values

#grid(
  columns: (40%, 1fr),
  column-gutter: 1em,
  text([$=>$ suited variational ansatz] ),

  text(11pt,[Cirac, I., Perez-Garcia, D., Schuch, N., Verstraete, F., 2021. Matrix Product States and Projected Entangled Pair States: Concepts, Symmetries, and Theorems.])
)

== infinite uniform MPS

// 绘制无限 MPS (iMPS) 宏
#let draw_imps(
  N: 4,            // 中间显示的张量个数
  D: $D$,          // 虚键维度标签
  d: [$sigma$],   // 物理腿标签
  dx: 2.2,         // 张量间距
  boxw: 1.2,       // 张量宽
  boxh: 1.2,       // 张量高
  leg: 1.2,        // 物理腿长度
  radius: 20%,     // 圆角
  lw: 1pt          // 线宽
) = {
  canvas({
    import draw: *

    set-style(stroke: lw, fill: white)

    // ---- 中间张量 ----
    for i in range(0, N) {
      let x = i * dx
      rect((x - boxw/2, -boxh/2), (rel: (boxw, boxh)),
           radius: radius, name: "A" + str(i))
      content(("A" + str(i) + ".center"), [$A$], anchor: "center")
    }

    // ---- 横向 bond + D 标签 ----
    for i in range(0, N - 1) {
      let start = "A" + str(i) + ".east"
      let end = "A" + str(i + 1) + ".west"
      line(start, end)
      let sx = i * dx + boxw/2
      let ex = (i + 1) * dx - boxw/2
      let mx = (sx + ex) / 2
      content((mx, 0), D, anchor: "north")
    }

    // ---- 左省略号 ----
    let leftDotsX = -dx
    content((leftDotsX, 0), [$dots$], anchor: "center")
    line((leftDotsX + 0.6, 0), ("A0.west"))
    content((leftDotsX + dx/2 - 0.3, 0), D, anchor: "north")

    // ---- 右省略号 ----
    let rightDotsX = N * dx
    line(("A" + str(N - 1) + ".east"), (rightDotsX - 0.6, 0), dashed: true)
    content((rightDotsX, 0), [$dots$], anchor: "center")
    content((rightDotsX - dx/2 + 0.3, 0), D, anchor: "north")

    // ---- 物理腿 ----
    for i in range(0, N) {
      let x = i * dx
      line(("A" + str(i) + ".north"), (x, boxh/2 + leg))
      content((x, boxh/2 + leg), d + [$$], anchor: "south")
    }
  })
}

// ====== 调用示例 ======
#align(center)[
  #draw_imps(
    N: 4,
    D: $D$,
    d: [$d$],
    dx: 2.5,    
    boxw: 1.0,  
    boxh: 1.0,  
    leg: 1.0    
  )
]

- Properties: 
  - Translational invariance: The same local tensor A is repeated across all sites. $=>$ Efficient parameterization: Describes an infinite system with a finite number of parameters.
  - The whole state only depend on tensor $A$, which reflect the bulk properties of the system.
  - Allow to work directly in the thermodynamic limit.

#text(12pt,[Vanderstraeten, L., Haegeman, J., Verstraete, F., 2019. Tangent-space methods for uniform matrix product states. SciPost Phys. Lect. Notes 7. https://doi.org/10.21468/SciPostPhysLectNotes.7])

== infinite uniform MPS

- Transfer matrix:

  #figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Draw the mathematical equation
    content((0.3, 0.4), [$E = sum_(s=1)^d A^s times.circle (A^s)^dagger =$])
    
    // Draw the tensor network diagram
    // Top tensor A
    rect((5.8, 1.2), (7, 2.4),radius: 20%, stroke: 1pt, name: "A")
    content((6.35, 1.8), [$A$])
    
    // Bottom tensor A-bar  
    rect((5.8, -1.2), (7, 0.0), radius: 20%, stroke: 1pt, name: "Abar")
    content((6.45, -0.6), [$A^dagger$])
    
    // Left bonds
    line((4.8, 1.8), (5.8, 1.8), stroke: 1pt)
    line((4.8, -0.6), (5.8, -0.6), stroke: 1pt)
    
    // Right bonds  
    line((7, 1.8), (8, 1.8), stroke: 1pt)
    line((7, -0.6), (8, -0.6), stroke: 1pt)
    
    // Vertical bond connecting A and A-bar (physical index)
    line((6.35, 0.0), (6.35, 1.2), stroke: 1pt)
    
    // Label the physical index
  
  }),
 
)


  In generic case, leading eigenvalue is non-degenerate.(Perron-Frobenius theorem: If degenerate, there exist phase transition, not a equilibria state.)

- Spectral decomposition:

    $E = lambda_0 |r angle.r angle.l l| + sum_i lambda_i |r_i angle.r angle.l l_i|$

    $lambda_0$ is the leading eigenvalue, which should be scaled to 1 by normalizing the MPS tensor $A -> A/sqrt(lambda_0)$. Thus the rest eigenvalues $lambda_i <1$


#figure(
  canvas(length: 1cm, {
    import draw: *
    let s(it) = text(16pt)[#it]
    content((0, 1), s[$lim_(N->infinity) E^N =lambda_0^N |r angle.r angle.l l| + sum_i lambda_i^N |r_i angle.r angle.l l_i| = $])
    environment((6,1), "r", text:s[$r$])
    bezier("r.north", (5,3), (rel: (0, 1.5), to: "r"))
    bezier("r.south", (5,-1), (rel: (0, -1.5), to: "r"))
    environment((8,1), "l", text:s[$l$])
    bezier("l.north", (9,3), (rel: (0, 1.5), to: "l"))
    bezier("l.south", (9,-1), (rel: (0, -1.5), to: "l"))
  }),
 
)


- fixed point equation:

    $E^N |l angle.r = |r angle.r angle.l l|l angle.r = |r angle.r => E (E^N |l angle.r) = E |r angle.r = |r angle.r$
    
    $angle.l r| E^N = angle.l r|r angle.r angle.l l|= angle.l l| => (angle.l l| E^N) E = angle.l l| E = angle.l l|$

== infinite uniform MPS
represent by tensor network:

#figure(canvas({
  import draw: *
  let s(it) = text(14pt)[#it]
  mpscell((0, 3), "A", text:s[$A$])
  mpscell((0, 0), "A^dagger", text:s[$A^dagger$])
  environment((-2,1.5), "l", text:s[$l$])
  line("A.south", "A^dagger.north")
  line("A.east", (1.5,3))
  line("A^dagger.east", (1.5,0))
  bezier("A.west", "l.north", (rel: (0, 1.5), to: "l"))
  bezier("A^dagger.west", "l.south", (rel: (0, -1.5), to: "l"))
  content((2, 1.5), s[$=$])
  environment((3.5,1.5), "l", text:s[$l$])
  bezier("l.north", (5,3), (rel: (0, 1.5), to: "l"))
  bezier("l.south", (5,0), (rel: (0, -1.5), to: "l"))

  content((6.5, 1.5), s[and])
  mpscell((10, 3), "A", text:s[$A$])
  mpscell((10, 0), "A^dagger", text:s[$A^dagger$])
  environment((12,1.5), "r", text:s[$r$])
  line("A.south", "A^dagger.north")
  line("A.west", (8.5,3))
  line("A^dagger.west", (8.5,0))
  bezier("A.east", "r.north", (rel: (0, 1.5), to: "r"))
  bezier("A^dagger.east", "r.south", (rel: (0, -1.5), to: "r"))
  content((13, 1.5), s[$=$])
  environment((15,1.5), "r", text:s[$r$])
  bezier("r.north", (13.5,3), (rel: (0, 1.5), to: "r"))
  bezier("r.south", (13.5,0), (rel: (0, -1.5), to: "r"))
}))
  $l$ is left environment, $r$ is right environment.

== 
For simplicity, let us suppose $D=d=2$, and the MPS is normalized and in right canonical form. Thus we have

#figure(canvas({
  import draw: *
  let s(it) = text(14pt)[#it]
  
  environment((0,0), "l", text:s[$l$])
  environment((2,0), "r", text:s[$r$])
  bezier("l.south", "r.south", (rel: (0, -1.5), to: "l"), (rel: (0, -1.5), to: "r"))
  bezier("l.north", "r.north", (rel: (0, 1.5), to: "l"), (rel: (0, 1.5), to: "r"))
  content((3.5, 0), s[$= 1$])
}))

and

#figure(canvas(
  {
  import draw: *
  let s(it) = text(14pt)[#it]
  environment((15,1.5), "r", text:s[$r$])
  bezier("r.north", (13.5,3), (rel: (0, 1.5), to: "r"))
  bezier("r.south", (13.5,0), (rel: (0, -1.5), to: "r"))


  content((17, 1.5), s[$ = alpha$])
  bezier((18,0), (18,3), (rel: (1.5, 0.5), to: (18,0)), (rel: (1.5, -0.5), to: (18,3)))
  }
))
#text(12pt,[Vanderstraeten, L., Haegeman, J., Verstraete, F., 2019. Tangent-space methods for uniform matrix product states. SciPost Phys. Lect. Notes 7. https://doi.org/10.21468/SciPostPhysLectNotes.7])

== Mapping to circuits
$A$ plays the role of Kraus operator. The transfer matrix plays the role of quantum channel. 
#figure(
  canvas(length: 1cm, {
    import draw: *
    
    // Draw the mathematical equation
    content((0.3, 0.4), [$E = sum_(s=1)^d A^s times.circle (A^s)^dagger =$])
    
    // Draw the tensor network diagram
    // Top tensor A
    rect((5.8, 1.2), (7, 2.4),radius: 20%, stroke: 1pt, name: "A")
    content((6.35, 1.8), [$A$])
    
    // Bottom tensor A-bar  
    rect((5.8, -1.2), (7, 0.0), radius: 20%, stroke: 1pt, name: "Abar")
    content((6.45, -0.6), [$A^dagger$])
    
    // Left bonds
    line((4.8, 1.8), (5.8, 1.8), stroke: 1pt)
    line((4.8, -0.6), (5.8, -0.6), stroke: 1pt)
    
    // Right bonds  
    line((7, 1.8), (8, 1.8), stroke: 1pt)
    line((7, -0.6), (8, -0.6), stroke: 1pt)
    
    // Vertical bond connecting A and A-bar (physical index)
    line((6.35, 0.0), (6.35, 1.2), stroke: 1pt)
    
    // Label the physical index
  }),
)


Since $A$ is right canonical, we can always define a unitary matrix $tilde(A)$ such that:

#figure(canvas({
  import draw: *
  rect((0,0), (1.5,3), radius: 10%, stroke: 1pt, name: "A")
  content((0.75, 1.5), [$tilde(A)$])
  line((-1,0.4),(0,0.4),mark: (end: ">"))
  line((-1,2.6),(0,2.6),mark: (end: ">"))
  line((1.5,2.6),(2.5,2.6),mark:(end:">"))
  line((1.5,0.4),(2.5,0.4),mark:(end:">"))
  content((-1.5,2.5),[$|0 angle.r$])
  content((-1.5,0.5),[$i$])
  content((3,2.6),[$j$])
  content((3,0.5),[$k$])
  content((4,1.5),[$=$])

  mpscell((7,1.5), "A", text: [$A$])
  line((5.5,1.5),"A.west",mark: (end: ">"))
  line("A.east",(8.5,1.5),mark: (end: ">"))
  line("A.north",(7,3),mark: (end: ">"))
  content((5,1.5),[$i$])
  content((9,1.5),[$j$])
  content((7,3.5),[$k$])
}))

#figure(canvas({
  import draw: *
  let gap = 1.2
  let cell_size = 1.0
  let DY = 1.5
  let s(it) = text(14pt)[#it] 
  let n = 4
  
  set-origin((0, 0))
  for i in range(n){
    mpscell((i * (cell_size + gap), 0), "A_"+str(i), text: s[$tilde(A)^dagger$])
    mpscell((i * (cell_size + gap), 2*DY), "B_"+str(i), text: s[$tilde(A)$])
    line("A_"+str(i), "B_"+str(i))
    line( (rel: (-1, -1), to: "B_"+str(i)), (rel: (-1, 1), to: "B_"+str(i)),(rel: (0, 1), to: "B_"+str(i)),"B_"+str(i)+".north", mark: (end: ">"))
    line( (rel: (-1, 1), to: "A_"+str(i)), (rel: (-1, -1), to: "A_"+str(i)),(rel: (0, -1), to: "A_"+str(i)),"A_"+str(i)+".south", mark: (end: ">"))
    content((rel:(-1,-1.5), to: "B_"+str(i)), s[$|0 angle.r angle.l 0|$])
  }

  for i in range(n - 1){
    line("A_"+str(i), "A_"+str(i+1),mark: (end: ">"))
    line("B_"+str(i), "B_"+str(i+1),mark: (end: ">"))
  }
  
  environment((-3,1.5), "rho", text:s[$rho$])

  bezier(("rho.south"),("A_0.west"),(rel: (-2.5,0),to:"A_0"), mark: (end: ">"))
  bezier(("rho.north"),("B_0.west"),(rel: (-2.5,0),to:"B_0"), mark: (end: ">"))

  line(("A_3.east"), (rel: (1.5,0),to:"A_3"),mark: (end: ">"))
  line(("B_3.east"), (rel: (1.5,0),to:"B_3"),mark: (end: ">"))

  line((-2,4),(-2,-1),stroke: (dash: "dashed", paint: red))
  
  content((8.5,1.5), s[$dots = $])

  environment((10,1.5), "rho", text:s[$rho$])
  environment((12,1.5), "r", text:s[$r$])
  environment((14,1.5), "l", text:s[$l$])
  bezier(("rho.south"),("r.south"),(rel: (0, -1.5), to: "rho"),(rel: (0, -1.5), to: "r"))
  bezier(("rho.north"),("r.north"),(rel: (0, 1.5), to: "rho"),(rel: (0, 1.5), to: "r"))
  bezier(("l.south"),(rel:(1,-1.2),to:"l"),(rel: (0, -1.2), to: "l"))
  bezier(("l.north"),(rel:(1,1.2),to:"l"),(rel: (0, 1.2), to: "l"))

  content((16,1.5), s[$= alpha$])
  environment((17.5,1.5), "rho", text:s[$rho$])
  bezier(("rho.north"),("rho.south"),(rel: (2, 2.5), to: "rho"),(rel: (2, -2.5), to: "rho"))
  environment((20.5,1.5), "l", text:s[$l$])
  bezier(("l.south"),(rel:(1,-1.2),to:"l"),(rel: (0, -1.2), to: "l"))
  bezier(("l.north"),(rel:(1,1.2),to:"l"),(rel: (0, 1.2), to: "l"))

  content((23,1.5), s[$=alpha$])
  environment((24.5,1.5), "l", text:s[$l$])
  bezier(("l.south"),(rel:(1,-1.2),to:"l"),(rel: (0, -1.2), to: "l"))
  bezier(("l.north"),(rel:(1,1.2),to:"l"),(rel: (0, 1.2), to: "l"))
}))

$"Tr"(rho) = 1, r = alpha II$


==
Fixed point equation of channel: $epsilon(rho_*)=rho_*$. 

#figure(canvas({
  import draw: *
  let gap = 1.2
  let cell_size = 1.0
  let height = 2
  let s(it) = text(14pt)[#it]
  environment((0,0), "l", text:s[$rho$])
  let pr2 = (cell_size, -height * cell_size + cell_size/2)
  let pr1 = (cell_size, height * cell_size - cell_size/2)
  let dx = 1
  mpscell((dx + cell_size/2, (height - 0.5) * cell_size), "c1", text: s[$A$])
  mpscell((dx + cell_size/2, -(height - 0.5) * cell_size), "c2", text: s[$A^dagger$])
  bezier("c1.west", "l.north", (rel: (0, 1.5), to: "l"))
  bezier("c2.west", "l.south", (rel: (0, -1.5), to: "l"))
  line("c1.south", (rel: (0, -1), to: "c1"))
  line("c2.north", (rel: (0, 1), to: "c2"))
  
  line("c1", (rel: (1.5, 0), to: "c1"))
  line("c2", (rel: (1.5, 0), to: "c2"))
  content((dx + cell_size/2, 0), [$|0angle.r angle.l 0|$])
  line("c1", (rel: (0, 1), to: "c1"), (rel: (1, 1), to: "c1"), (rel: (1, -1), to: "c2"), (rel: (0, -1), to: "c2"), "c2")

  content((4, 0), [$=$])

  let DX = 6
  environment((DX,0), "l", text:s[$rho$])
  bezier("l.north",(DX + cell_size/2, (height - 0.5) * cell_size), (rel: (0, 1.5), to: "l"))
  bezier( "l.south",(DX + cell_size/2, -(height - 0.5) * cell_size), (rel: (0, -1.5), to: "l"))
}))

 Thus $rho_* = alpha l$.

== Past method
Cholesky decomposition of $rho$: $rho = V V^dagger$, $tr(rho) = 1$

#image("images/circuit.png", width: 30%)

$V$ has also been parameterized. 

Judge whether $V V^dagger$ is left environment by swap test.

#text(size: 10pt)[Barratt, F., Dborin, J., Bal, M., Stojevic, V., Pollmann, F., Green, A.G., 2021. Parallel Quantum Simulation of Large Systems on Small Quantum Computers. npj Quantum Inf 7, 79. https://doi.org/10.1038/s41534-021-00420-3]

#image("images/swap_test.png", width: 70%)

$tr((rho-sigma)^dagger (rho-sigma))$ would be minized at $rho = sigma$.

#text(12pt)[Garcia-Escartin, J.C., Chamorro-Posada, P., 2013. The SWAP test and the Hong-Ou-Mandel effect are equivalent. Phys. Rev. A 87, 052330. https://doi.org/10.1103/PhysRevA.87.052330]

==
#grid(
  columns: (30%, 1fr),
  column-gutter: 1em,
  text(size: 20pt)[TFIM ground state:],
  image("images/result.png", width: 80%)
)

#grid(
  columns: (30%, 1fr),
  column-gutter: 1em,
  text(size: 20pt)[Time evolution:],
  image("images/echo.png", width: 80%)
  
)


Disadvantages: need so many circuits, more variables, high complexity.

= Our method
== Goal
Hamiltonian of transverse field Ising model on an infinite chain:

$ H = - sum_i Z_i Z_(i+1) - g sum_i X_i $

where $Z_i$ is the Pauli-Z operator on the $i$-th site, $X_i$ is the Pauli-X operator on the $i$-th site, and $g$ is the transverse field.
The goal is to find the ground state using infinite uniform MPS as variational ansatz.

== 
Energy per site of 1D TFIM can be evaluated as:

#figure(canvas({
  import draw: *
  let n = 3
  let gap = 1.2
  let cell_size = 0.5
  let DY = 1.5
  let s(it) = text(14pt)[#it]
  content((-1.6, 1.5), s[$-h$])
  for i in range(n){
    mpscell((i * (cell_size + gap), 0), "A_"+str(i), text: s[$A$])
    mpscell((i * (cell_size + gap), 2*DY), "B_"+str(i), text: s[$A^dagger$])
  }
  mpscell((gap + cell_size, DY), "X", text: s[$X$])
  line("A_0", "B_0")
  line("A_2", "B_2")
  line("A_1", "X")
  line("B_1", "X")
  for i in range(n - 1){
    line("A_"+str(i), "A_"+str(i+1))
    line("B_"+str(i), "B_"+str(i+1))
  }
  line("A_0", (rel: (-0.7, 0)))
  content((rel: (-1, 0.1), to: "A_0"), s[$dots$])
  line("B_0", (rel: (-0.7, 0)))
  content((rel: (-1, 0.1), to: "B_0"), s[$dots$])
  line("A_"+str(n - 1), (rel: (0.7, 0)))
  content((rel: (1, 0.1), to: "A_"+str(n - 1)), s[$dots$])
  line("B_"+str(n - 1), (rel: (0.7, 0)))
  content((rel: (1, 0.1), to: "B_"+str(n - 1)), s[$dots$])

  content((5.3, 1.5), s[$-$])
  let n = 4
  set-origin((7, 0))
  for i in range(n){
    mpscell((i * (cell_size + gap), 0), "A_"+str(i), text: s[$A$])
    mpscell((i * (cell_size + gap), 2*DY), "B_"+str(i), text: s[$A^dagger$])
  }
  mpscell((gap + cell_size, DY), "Z1", text: s[$Z$])
  mpscell((2*(gap + cell_size), DY), "Z2", text: s[$Z$])
  line("A_0", "B_0")
  line("A_3", "B_3")
  line("A_1", "Z1")
  line("B_1", "Z1")
  line("A_2", "Z2")
  line("B_2", "Z2")
  for i in range(n - 1){
    line("A_"+str(i), "A_"+str(i+1))
    line("B_"+str(i), "B_"+str(i+1))
  }
  line("A_0", (rel: (-0.7, 0)))
  content((rel: (-1, 0.1), to: "A_0"), s[$dots$])
  line("B_0", (rel: (-0.7, 0)))
  content((rel: (-1, 0.1), to: "B_0"), s[$dots$])
  line("A_"+str(n - 1), (rel: (0.7, 0)))
  content((rel: (1, 0.1), to: "A_"+str(n - 1)), s[$dots$])
  line("B_"+str(n - 1), (rel: (0.7, 0)))
  content((rel: (1, 0.1), to: "B_"+str(n - 1)), s[$dots$])


}))

#figure(canvas(
  {
  import draw: *
  let s(it) = text(14pt)[#it]
  content((0.0, 1.5), s[$= alpha [-h$])
  mpscell((3.5, 3), "A", text:s[$A$])
  mpscell((3.5, 0), "A^dagger", text:s[$A^dagger$])
  mpscell((3.5, 1.5), "X", text:s[$X$])
  environment((2.0,1.5), "l", text:s[$l$])
  line("A.south", "X.north")
  line("A^dagger.north", "X.south")
  bezier("A.west", "l.north", (rel: (0, 1.5), to: "l"))
  bezier("A^dagger.west", "l.south", (rel: (0, -1.5), to: "l"))
  bezier("A.east", "A^dagger.east", (rel: (1.5, -0.5), to: "A"), (rel: (1.5, 0.5), to: "A^dagger"))
  content((5.5, 1.5), s[$-$])

  mpscell((8, 3), "A1", text:s[$A$])
  mpscell((8, 0), "A1^dagger", text:s[$A^dagger$])
  mpscell((10, 3), "A2", text:s[$A$])
  mpscell((10, 0), "A2^dagger", text:s[$A^dagger$])
  mpscell((8, 1.5), "Z1", text:s[$Z$])
  mpscell((10, 1.5), "Z2", text:s[$Z$])
  environment((6.5,1.5), "l", text:s[$l$])
  line("A1.south", "Z1.north")
  line("A2.south", "Z2.north")
  line("A1^dagger.north", "Z1.south")
  line("A2^dagger.north", "Z2.south")
  line("A1^dagger.east", "A2^dagger.west")
  line("A1.east", "A2.west")
  bezier("A1.west", "l.north", (rel: (0, 1.5), to: "l"))
  bezier("A1^dagger.west", "l.south", (rel: (0, -1.5), to: "l"))
  bezier("A2.east", "A2^dagger.east", (rel: (1.5, -0.5), to: "A2"), (rel: (1.5, 0.5), to: "A2^dagger"))
  content((11.5, 1.5), s[$]$])
  }))
The contraction complexity is: $ O(D^3)$


== 
As we said before, we can get left environment by iterating channel, after it satisfy fixed point equation, we can get observables by measuring the physical qubits and trace the ancilla qubits.

#grid(
  columns: (50%, 1fr),
  column-gutter: 1em,
  image("images/channel.png", width: 80%),
  image("images/expect.png", width: 50%)
)

The complexity is $O(N_"measure")$

== Compilation 

// 绘制电路分解图
#let draw_circuit_decomposition() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1.5pt, fill: white)
    
    // Left side - Unitary U
    rect((-2.5, -0.8), (rel: (1.2, 1.6)), 
         radius: 10%, fill: rgb(240, 240, 240), name: "U")
    content(("U.center"), text(size: 16pt, [$U$]), anchor: "center")
    
    // Input and output lines for U
    line((-3.5, 0.4), ((-2.5,0.4)))
    line((-3.5, -0.4), (-2.5, -0.4))
    line((-1.3,0.4), (-0.3, 0.4))
    line((-1.3,-0.4), (-0.3, -0.4))
    
    // Equals sign
    content((0.4, 0), text(size: 18pt, [$=$]), anchor: "center")
    
    // Right side - Circuit decomposition
    // Top wire
    line((1.0, 0.4), (6.5, 0.4))
    // Bottom wire
    line((1.0, -0.4), (6.5, -0.4))
    
    // R_z(α) gate
    rect((1.5, 0.1), (rel: (1.0, 0.6)), 
         radius: 5%, fill: white, name: "Rz_alpha")
    content(("Rz_alpha.center"), text(size: 10pt, [$R_z (alpha)$]), anchor: "center")
    
    // R_x(β) gate
    rect((3.0, 0.1), (rel: (1.0, 0.6)), 
         radius: 5%, fill: white, name: "Rx_beta")
    content(("Rx_beta.center"), text(size: 10pt, [$R_x (beta)$]), anchor: "center")
    
    // R_z(γ) gate on bottom
    rect((1.5, -0.7), (rel: (1.0, 0.6)), 
         radius: 5%, fill: white, name: "Rz_gamma")
    content(("Rz_gamma.center"), text(size: 10pt, [$R_z (gamma)$]), anchor: "center")
    
    // R_x(δ) gate on bottom
    rect((3.0, -0.7), (rel: (1.0, 0.6)), 
         radius: 5%, fill: white, name: "Rx_delta")
    content(("Rx_delta.center"), text(size: 10pt, [$R_x (delta)$]), anchor: "center")
    
    // CNOT gate
    // Control (filled circle)
    circle((5.0, 0.4), radius: 0.1, fill: black, name: "control")
    // Target (circle with plus)
    circle((5.0, -0.4), radius: 0.15, fill: white, stroke: 2pt, name: "target")
    line((4.85, -0.4), (5.15, -0.4), stroke: 2pt)
    line((5.0, -0.55), (5.0, -0.25), stroke: 2pt)
    
    // Vertical line connecting control and target
    line((5.0, 0.4), (5.0, -0.4), stroke: 1.5pt)
  })
}

#align(center)[
  #draw_circuit_decomposition()
]

or

// 绘制电路分解图
#let draw_Ry_decomposition() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1.5pt, fill: white)
    
    // Left side - Unitary U
    rect((-2.5, -0.8), (rel: (1.2, 1.6)), 
         radius: 10%, fill: rgb(240, 240, 240), name: "U")
    content(("U.center"), text(size: 16pt, [$U$]), anchor: "center")
    
    // Input and output lines for U
    line((-3.5, 0.4), ((-2.5,0.4)))
    line((-3.5, -0.4), (-2.5, -0.4))
    line((-1.3,0.4), (-0.3, 0.4))
    line((-1.3,-0.4), (-0.3, -0.4))
    
    // Equals sign
    content((0.4, 0), text(size: 18pt, [$=$]), anchor: "center")
    
    // Right side - Circuit decomposition
    // Top wire
    line((1.0, 0.4), (5.0, 0.4))
    // Bottom wire
    line((1.0, -0.4), (5.0, -0.4))
    
    // R_y(α) gate
    rect((1.5, 0.1), (rel: (1.0, 0.6)), 
         radius: 5%, fill: white, name: "Rz_alpha")
    content(("Rz_alpha.center"), text(size: 10pt, [$R_y (alpha)$]), anchor: "center")
    
    
    // R_y(β) gate on bottom
    rect((1.5, -0.7), (rel: (1.0, 0.6)), 
         radius: 5%, fill: white, name: "Rz_gamma")
    content(("Rz_gamma.center"), text(size: 10pt, [$R_y (beta)$]), anchor: "center")
    
   
    // CNOT gate
    // Control (filled circle)
    circle((4.0, 0.4), radius: 0.1, fill: black, name: "control")
    // Target (circle with plus)
    circle((4.0, -0.4), radius: 0.15, fill: white, stroke: 2pt, name: "target")
    line((3.85, -0.4), (4.15, -0.4), stroke: 2pt)
    line((4.0, -0.55), (4.0, -0.25), stroke: 2pt)
    
    // Vertical line connecting control and target
    line((4.0, 0.4), (4.0, -0.4), stroke: 1.5pt)
  })
}

#align(center)[
  #draw_Ry_decomposition()
]

#import "@preview/quill:0.7.2": *
#import "@preview/quill:0.7.2" as quill: tequila as tq

(1) $angle.l X angle.r$:

#quill.quantum-circuit(
    lstick($|0〉$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),$H$,meter(),midstick($|0 angle.r$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),$H$,meter(),midstick($|0 angle.r$),$R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),$H$,meter(),rstick("..."),[\ ],
    lstick($rho$), $R_z (theta_3)$, $R_x (theta_4)$, targ(), 3,$R_z (theta_3)$, $R_x (theta_4)$, targ(),3,$R_z (theta_3)$, $R_x (theta_4)$, targ(),2,
  rstick("..."),
  quill.gategroup(x: 1,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 7,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 13,y:0,2,3,label: (content:$times p$,pos:top)),
 
)

  #text(15pt,[$p$ is repeat times])

(2) $angle.l Z Z angle.r$:

#quill.quantum-circuit(
    lstick($|0〉$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),meter(),midstick($|0 angle.r$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),meter(),midstick($|0 angle.r$),$R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),meter(),1,rstick("..."),[\ ],
    lstick($rho$), $R_z (theta_3)$, $R_x (theta_4)$, targ(), 2,$R_z (theta_3)$, $R_x (theta_4)$, targ(),2,$R_z (theta_3)$, $R_x (theta_4)$, targ(),2,
  rstick("..."),
  quill.gategroup(x: 1,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 6,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 11,y:0,2,3,label: (content:$times p$,pos:top)),
 
)

  Converged condition: $angle.l X angle.r_n approx angle.l X angle.r_(n-1)$,  $angle.l Z_i Z_(i+1) angle.r_n approx angle.l Z_i Z_(i+1) angle.r_(n-1)$, $n$ means the channel applying iteration. 
  
 

(3) update $theta$: 

    $angle.l H(theta) angle.r = -g angle.l X(theta) angle.r - J angle.l Z_i (theta) Z_(i+1) (theta) angle.r $
    
    $partial/(partial theta_i) angle.l H(theta) angle.r => theta->theta+delta theta$ 

Iterate until it reach converged condition:

 $angle.l H(theta_n) angle.r  - angle.l H(theta_(n-1)) angle.r  <=10^(-8) $, $n$ means the n-th update iteration. 

== Algorithm Box


#import "@preview/algorithmic:1.0.3"
#import algorithmic: style-algorithm, algorithm-figure
#show: style-algorithm

#algorithm-figure(
  text(size: 18pt, "Variational iMPS Ground State Optimization"),
  vstroke: .5pt + luma(200),
  caption-style: text.with(size: 16pt),

  {
    import algorithmic: *
    Procedure(
      "Variational-iMPS",
      ($theta_0$, $g$, $J$, "maxiter"),
      {
        
        Comment[Initialize parameters and convergence criteria]
        Assign[$theta$][$theta_0, p$]
        Assign[iter][$0$]
        LineBreak
        While(
          "iter < maxiter && f_tol > 1e-8",
          {
             Comment[Construct parameterized quantum circuit]
             Assign[$U(theta)$][$"ConstructCircuit"(theta)$]
         
             LineBreak
             Comment[Iterate quantum channel to fixed point]
             Assign[$rho$][$"IterateChannel"(U(theta))$]
             LineBreak
             Comment[Evaluate energy expectation]
             Assign[$angle.l X angle.r$][$"Expectation"(rho_L, X)$]
             Assign[$angle.l Z Z angle.r$][$"Expectation"(rho_L, Z ⊗ Z)$]
             Assign[$E$][$-g · angle.l X angle.r - J · angle.l Z Z angle.r$]
             LineBreak
             Comment[Update parameters]
             Assign[$theta$][$"NelderMead"(theta, E)$]
            Assign[iter][iter + 1]
          },
        )
        Return[$theta$, $E$]
      },
    )
  }
) 

== Construct parameterized quantum circuit



= Results

== Exact contraction 

#align(center)[
  #image("images/energy_errorvs_g-1.png", width: 60%)
]

#align(center)[
  #text(size: 15pt)[Fig. 1 Energy error v.s. transverse field strength for different circuit depths. iMPS: analytical results of $d=2, D=2$ infinite MPS from MPSKit.jl.]
]

== Measurement

#align(center)[
  #image("images/energy_errorvs_g-2.png", width: 60%)
]

#align(center)[
  #text(size: 15pt)[Fig. 2 Energy error v.s. transverse field strength for p=3 through measurement]
]

==

The phase diagram of the 1D TFIM has a critical point at $g = 1$, with a corresponding infinite correlation length, where a finite depth circuit of nearest-neighbor gates will be unable to exactly prepare this state. It explains why at $g=1$ the simulation result is the worst.

Besides, the correlation length $epsilon=-1/log(lambda_1)$, where $lambda_1$ is the second largest magnitude eigenvalue of the ground state transfer matrix. If $epsilon$ diverges, $lambda_1 = 1 = lambda_0$, the eigenvalue is degenerate.


= Summary
==

1. Simple circuit; 

2. Less variables;

3. Easy to implement on current hardware.

Future work: 
1. Larger virtual dimension; 

2. How to get gradient of infinite circuit; 

3. Variational compilation.

