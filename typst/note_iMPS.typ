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
    title: [Note],
    subtitle: [iMPS to quantum circuit],
    author: [Yuqing Rong],
    date: datetime.today(),
    institution: [HKUST(GZ)],
  ),
)

#title-slide()

#outline-slide()

= Introduction


== Summary
We translate infinite, translationally invariant matrix product states (iMPS) into finite-depth quantum circuits. 

The ground state of the transverse field Ising model is obtained through variational optimization of circuit parameters. 

== iMPS

- an infinite, translationally  invariant quantum spin chain (1-site unit cell): 

// 绘制无限 MPS (iMPS) 宏
#let draw_imps(
  N: 4,            // 中间显示的张量个数
  D: $D$,          // 虚键维度标签
  d: [$sigma$],   // 物理腿标签
  dx: 2.2,         // 张量间距
  boxw: 1.2,       // 张量宽
  boxh: 0.9,       // 张量高
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
    dx: 2.5,    // 减小张量间距
    boxw: 1.0,  // 减小张量宽度
    boxh: 0.7,  // 减小张量高度
    leg: 1.0    // 减小物理腿长度
  )
]

- left and right environment representation of overlap (in mixed canonical form):

// 绘制三层张量网络 (AL, AC, AR with operators)
#let draw_three_layer_with_operators(
  boxw: 0.8,       // 张量宽
  boxh: 0.5,       // 张量高
  layer_sep: 1.2,  // 层间距
  dx: 1.8,         // 张量间距
  radius: 15%,     // 圆角
  lw: 1pt          // 线宽
) = {
  canvas({
    import draw: *
    
    set-style(stroke: lw, fill: white)
    
    // 定义张量标签和位置
    let labels_top = ("A", "A", "A", "A", "A")
    let labels_op = ("O", "O", "O", "O", "O")
    let labels_bottom = ("A", "A", "A", "A", "A")
    let colors = (white, white, white, white, white)
    
    let N = 5
    
    // ---- 上层张量 (A) ----
    let labels = ("L", "L", "C", "R", "R")
    for i in range(0, N) {
      let x = (i - 2) * dx  // 居中
      rect((x - boxw/2, layer_sep/2), (rel: (boxw, boxh)),
           radius: radius, name: "top" + str(i), fill: colors.at(i))
      content(("top" + str(i) + ".center"), text(size: 8pt, [$A_#labels.at(i)$]), anchor: "center")
    }
    
    // ---- 下层张量 (A†) ----
    for i in range(0, N) {
      let x = (i - 2) * dx
      rect((x - boxw/2, -layer_sep/2 - boxh), (rel: (boxw, boxh)),
           radius: radius, name: "bottom" + str(i), fill: colors.at(i))
      content(("bottom" + str(i) + ".center"), text(size: 8pt, [$A_#labels.at(i)^dagger$]), anchor: "center")
    }
    
    // ---- 水平连接线 (上层) ----
    for i in range(0, N - 1) {
      line(("top" + str(i) + ".east"), ("top" + str(i + 1) + ".west"))
    }
    
    // ---- 水平连接线 (下层) ----
    for i in range(0, N - 1) {
      line(("bottom" + str(i) + ".east"), ("bottom" + str(i + 1) + ".west"))
    }
    
    // ---- 垂直连接线 (物理腿) ----
    for i in range(0, N) {
      line(("top" + str(i) + ".south"), ("bottom" + str(i) + ".north"))
    }
    
    // ---- 左省略号 ----
    let leftX = -3 * dx
    content((leftX, layer_sep/2 + boxh/2), [$dots$], anchor: "center")
    content((leftX, -layer_sep/2 - boxh/2), [$dots$], anchor: "center")
    
    // 连接到省略号
    line((leftX + 0.4, layer_sep/2 + boxh/2), ("top0.west"))
    line((leftX + 0.4, -layer_sep/2 - boxh/2), ("bottom0.west"))
    
    // ---- 右省略号 ----
    let rightX = 3 * dx
    content((rightX, layer_sep/2 + boxh/2), [$dots$], anchor: "center")
    content((rightX, -layer_sep/2 - boxh/2), [$dots$], anchor: "center")
    
    // 连接到省略号
    line(("top4.east"), (rightX - 0.4, layer_sep/2 + boxh/2))
    line(("bottom4.east"), (rightX - 0.4, -layer_sep/2 - boxh/2))
  })
}

// 绘制简单的两层表示 (function definition)
#let draw_simple_two_layers(
  N: 3,            // 张量个数
  boxw: 0.7,       // 张量宽
  boxh: 0.45,      // 张量高
  layer_sep: 1.4,  // 层间距
  dx: 1.6,         // 张量间距
  radius: 15%,     // 圆角
  lw: 1pt          // 线宽
) = {
  canvas({
    import draw: *
    
    set-style(stroke: lw, fill: white)
    
    // ---- 上层张量 (A) ----
    let labels = ("L", "L", "C", "R", "R")
    for i in range(0, N) {
      let x = (i - 1) * dx  // 居中
      rect((x - boxw/2, layer_sep/2), (rel: (boxw, boxh)),
           radius: radius, name: "top" + str(i), fill: white)
      content(("top" + str(i) + ".center"), text(size: 8pt, [$A_#labels.at(i)$]), anchor: "center")
    }
    
    // ---- 下层张量 (A†) ----
    for i in range(0, N) {
      let x = (i - 1) * dx
      rect((x - boxw/2, -layer_sep/2 - boxh), (rel: (boxw, boxh)),
           radius: radius, name: "bottom" + str(i), fill: white)
      content(("bottom" + str(i) + ".center"), text(size: 8pt, [$A_#labels.at(i)^dagger$]), anchor: "center")
    }
    
    // ---- 水平连接线 (上层) ----
    for i in range(0, N - 1) {
      line(("top" + str(i) + ".east"), ("top" + str(i + 1) + ".west"))
    }
    
    // ---- 水平连接线 (下层) ----
    for i in range(0, N - 1) {
      line(("bottom" + str(i) + ".east"), ("bottom" + str(i + 1) + ".west"))
    }
    
    // ---- 垂直连接线 (物理腿) ----
    for i in range(0, N) {
      line(("top" + str(i) + ".south"), ("bottom" + str(i) + ".north"))
    }
    
    // ---- 左侧环境连接 ----
    let env_x = (0 - 1) * dx - boxw/2 - 1.2
    let top_y = layer_sep/2 + boxh/2
    let bottom_y = -layer_sep/2 - boxh/2
    let env_height = top_y - bottom_y + 0.4
    
    // 绘制高矩形环境
    rect((env_x - 0.4, bottom_y - 0.3), (rel: (0.8, env_height + 0.2)), 
         radius: radius, fill: rgb(200, 230, 255), name: "left_env")
    content(("left_env.center"), text(size: 15pt, [$F_L$]), anchor: "center")
    
    // 水平直线连接到张量
    line((env_x + 0.4, layer_sep/2 + boxh/2), ("top0.west"))
    line((env_x + 0.4, -layer_sep/2 - boxh/2), ("bottom0.west"))
    
    // ---- 右侧环境连接 ----
    let right_env_x = (N - 1 - 1) * dx + boxw/2 + 1.2
    
    // 绘制右侧高矩形环境
    rect((right_env_x - 0.4, bottom_y - 0.3), (rel: (0.8, env_height + 0.2)), 
         radius: radius, fill: rgb(255, 230, 200), name: "right_env")
    content(("right_env.center"), text(size: 15pt, [$F_R$]), anchor: "center")
    
    // 水平直线连接到最右张量
    line(("top" + str(N - 1) + ".east"), (right_env_x - 0.4, layer_sep/2 + boxh/2))
    line(("bottom" + str(N - 1) + ".east"), (right_env_x - 0.4, -layer_sep/2 - boxh/2))
  })
}

#align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 0.5em,
    align: center + horizon,
    
    // Left diagram
    draw_three_layer_with_operators(
      boxw: 0.6,
      boxh: 0.4,
      layer_sep: 1.0,
      dx: 1.2
    ),
    
    // Equals sign
    text(size: 18pt)[$=$],
    
    // Right diagram
    draw_simple_two_layers(
      N: 5,
      boxw: 0.6,
      boxh: 0.4,
      layer_sep: 1.0,
      dx: 1.2
    )
  )
]


- $F_L$ and $F_R$ satisfy fixed point equation:
// 绘制简单的特征值方程
#let draw_FL_equation() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1pt, fill: white)
    
    // F_L rectangle
    rect((-1.0, -0.8), (rel: (0.6, 1.6)), 
         radius: 10%, fill: rgb(200, 230, 255), name: "FL")
    content(("FL.center"), text(size: 12pt, [$F_L$]), anchor: "center")
    
    // Top A_L
    rect((0.2, 0.3), (rel: (0.6, 0.4)),
         radius: 10%, name: "AL_top", fill: white)
    content(("AL_top.center"), text(size: 10pt, [$A_L$]), anchor: "center")
    
    // Bottom A_L†  
    rect((0.2, -0.7), (rel: (0.6, 0.4)),
         radius: 10%, name: "AL_bottom", fill: white)
    content(("AL_bottom.center"), text(size: 10pt, [$A_L^dagger$]), anchor: "center")
    
    // Lines from northeast and southeast of F_L to A_L tensors
    line((-0.4, 0.5), ("AL_top.west"))
    line((-0.4, -0.5), ("AL_bottom.west"))
    
    // Vertical connection between A_L and A_L†
    line(("AL_top.south"), ("AL_bottom.north"))
    
    // Extension lines to the right
    line(("AL_top.east"), (1.5, 0.5))
    line(("AL_bottom.east"), (1.5, -0.5))
  })
}

#align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - our F_L equation
    draw_FL_equation(),
    
    // Equals sign
    text(size: 16pt)[$=$],
    
    // Right side - simple F_L with extension lines
    canvas({
      import draw: *
      
      set-style(stroke: 1pt, fill: white)
      
      // Tall F_L rectangle
      rect((-0.3, -0.8), (rel: (0.6, 1.6)), 
           radius: 10%, fill: rgb(200, 230, 255), name: "FL_simple")
      content(("FL_simple.center"), text(size: 12pt, [$F_L$]), anchor: "center")
      
      // Two horizontal extension lines (no middle layer)
      line((0.3, 0.4), (1.0, 0.4))
      line((0.3, -0.4), (1.0, -0.4))
    })
  )
]

// 绘制右环境方程
#let draw_FR_equation() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1pt, fill: white)
    
    // Top A_R
    rect((-0.8, 0.3), (rel: (0.6, 0.4)),
         radius: 10%, name: "AR_top", fill: white)
    content(("AR_top.center"), text(size: 10pt, [$A_R$]), anchor: "center")
    
    // Bottom A_R†  
    rect((-0.8, -0.7), (rel: (0.6, 0.4)),
         radius: 10%, name: "AR_bottom", fill: white)
    content(("AR_bottom.center"), text(size: 10pt, [$A_R^dagger$]), anchor: "center")
    
    // F_R rectangle on the right
    rect((0.4, -0.8), (rel: (0.6, 1.6)), 
         radius: 10%, fill: rgb(255, 230, 200), name: "FR")
    content(("FR.center"), text(size: 12pt, [$F_R$]), anchor: "center")
    
    // Lines from A_R tensors to northeast and southeast of F_R
    line(("AR_top.east"), (0.4, 0.5))
    line(("AR_bottom.east"), (0.4, -0.5))
    
    // Vertical connection between A_R and A_R†
    line(("AR_top.south"), ("AR_bottom.north"))
    
    // Extension lines to the left
    line((-1.5, 0.5), ("AR_top.west"))
    line((-1.5, -0.5), ("AR_bottom.west"))
  })
}

#align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - A_R tensors with F_R
    draw_FR_equation(),
    
    // Equals sign
    text(size: 16pt)[$=$],
    
    // Right side - simple F_R
    canvas({
      import draw: *
      
      set-style(stroke: 1pt, fill: white)
      
      // Tall F_R rectangle
      rect((-0.3, -0.8), (rel: (0.6, 1.6)), 
           radius: 10%, fill: rgb(255, 230, 200), name: "FR_simple")
      content(("FR_simple.center"), text(size: 12pt, [$F_R$]), anchor: "center")
      
      // Two horizontal extension lines (no middle layer)
      line((-1.0, 0.4), (-0.3, 0.4))
      line((-1.0, -0.4), (-0.3, -0.4))
    })
  )
]


- Power method: in each iteration, contract the environment and  transfer matrix until they follow the fixed point equation.

== Quantum Channel
- Quantum channel representation:
$ Phi(rho_*) = rho_* $
// 绘制量子通道表示 (大版本)
#let draw_quantum_channel_large() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1.5pt, fill: white)
    
    // Left ρ rectangle (bigger)
    rect((-4.5, -1.5), (rel: (1.8, 3.0)), 
         radius: 10%, fill: rgb(200, 230, 255), name: "rho_left")
    content(("rho_left.center"), text(size: 18pt, [$rho_L$]), anchor: "center")
    
    // Middle section with A_L and A_L† (bigger)
    // Top A_L
    rect((-0.8, 0.7), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_top", fill: white)
    content(("AL_mid_top.center"), text(size: 14pt, [$A_L$]), anchor: "center")
    
    // Bottom A_L†
    rect((-0.8, -1.6), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_bottom", fill: white)
    content(("AL_mid_bottom.center"), text(size: 14pt, [$A_L^dagger$]), anchor: "center")
    

    
    // |0⟩⟨0| label in the middle between A_L and A_L† (bigger)
    content((-0.2, -0.0), text(size: 16pt, [$|0 angle.r angle.l 0|$]), anchor: "center")
    
    // Vertical connection between A_L and A_L† (adjusted for bigger size)
    line(("AL_mid_top.south"), (-0.2, 0.2))
    line(("AL_mid_bottom.north"), (-0.2, -0.2))
    
    // Horizontal connections (adjusted for bigger size)
    line((-2.7, 1.1), ("AL_mid_top.west"))
    line((-2.7, -1.2), ("AL_mid_bottom.west"))
   
    line(("AL_mid_top.north"),(-0.2,2.0))
    line(("AL_mid_bottom.south"),(-0.2,-2.0))
    line((-0.2,2.0),(1.0,2.0))
    line((-0.2,-2.0),(1.0,-2.0))
    line((1.0,2.0),(1.0,-2.0))
    line("AL_mid_top.east",(1.5,1.1))
    line("AL_mid_bottom.east",(1.5,-1.2))

  })
}

#align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - quantum channel (large)
    draw_quantum_channel_large(),
    
    // Equals sign
    text(size: 16pt)[$=$],
    
    // Right side - simple ρ (larger)
    canvas({
      import draw: *
      
      set-style(stroke: 1.5pt, fill: white)
      
      // Simple ρ rectangle (bigger)
      rect((-0.9, -1.5), (rel: (1.8, 3.0)), 
           radius: 10%, fill: rgb(200, 230, 255), name: "rho_simple")
      content(("rho_simple.center"), text(size: 18pt, [$rho_L$]), anchor: "center")
      
      // Extension lines (bigger)
      line((0.9,1.1), (1.8, 1.1))
      line((0.9, -1.1), (1.8, -1.1))
    })
  )
]

- Suppose: 
  - $d=D=2$
  - the iMPS are in right canonical form, $=>$ $rho_R = II$.
  
- Compile to circuit:

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

- methods to get local expectation value:
  - contract directly (exact result of the ansatz):
    - $ angle.l X_i angle.r$
    // 绘制量子通道表示 (大版本)
    #let draw_quantum_channel_large() = {
      canvas({
      import draw: *
    
      set-style(stroke: 1.5pt, fill: white)
    
      // Left ρ rectangle (bigger)
      rect((-4.5, -1.5), (rel: (1.8, 3.0)), 
         radius: 10%, fill: rgb(200, 230, 255), name: "rho_left")
    content(("rho_left.center"), text(size: 18pt, [$rho_L$]), anchor: "center")
    
    // Middle section with A_L and A_L† (bigger)
    // Top A_L
    rect((-0.8, 0.7), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_top", fill: white)
    content(("AL_mid_top.center"), text(size: 14pt, [$A_L$]), anchor: "center")
    
    // Bottom A_L†
    rect((-0.8, -1.6), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_bottom", fill: white)
    content(("AL_mid_bottom.center"), text(size: 14pt, [$A_L^dagger$]), anchor: "center")
    

    
    // |0⟩⟨0| label in the middle between A_L and A_L† (bigger)
    content((-0.2, -0.0), text(size: 16pt, [$|0 angle.r angle.l 0|$]), anchor: "center")
    
    // Vertical connection between A_L and A_L† (adjusted for bigger size)
    line(("AL_mid_top.south"), (-0.2, 0.2))
    line(("AL_mid_bottom.north"), (-0.2, -0.2))
    
    // Horizontal connections (adjusted for bigger size)
    line((-2.7, 1.1), ("AL_mid_top.west"))
    line((-2.7, -1.2), ("AL_mid_bottom.west"))
   
    // add X
    rect((0.6, -0.5), (rel: (0.8, 0.8)),
         radius: 10%, name: "X", fill: white)
    content(("X.center"), text(size: 14pt, [$X$]), anchor: "center")

    line(("AL_mid_top.north"),(-0.2,2.0))
    line(("AL_mid_bottom.south"),(-0.2,-2.0))
    line("AL_mid_top.east",(1.7,1.1))
    line("AL_mid_bottom.east",(1.7,-1.2))
    line((1.7,1.1),(1.7,-1.2))
    line((-0.2,2.0),(1.0,2.0))
    line((-0.2,-2.0),(1.0,-2.0))
    line((1.0,2.0),"X.north")
    line("X.south",(1.0,-2.0))
  })
}

    #align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - quantum channel (large)
    draw_quantum_channel_large(),
   
  )
]

      - $ angle.l Z_i Z_(i+1) angle.r$
      // 绘制量子通道表示 (大版本)
#let draw_quantum_channel_large() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1.5pt, fill: white)
    
    // Left ρ rectangle (bigger)
    rect((-4.5, -1.5), (rel: (1.8, 3.0)), 
         radius: 10%, fill: rgb(200, 230, 255), name: "rho_left")
    content(("rho_left.center"), text(size: 18pt, [$rho_L$]), anchor: "center")
    
    // Middle section with A_L and A_L† (bigger)
    // Top A_L1
    rect((-0.8, 0.7), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_top", fill: white)
    content(("AL_mid_top.center"), text(size: 14pt, [$A_L$]), anchor: "center")
    
    // Bottom A_L†1
    rect((-0.8, -1.6), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_bottom", fill: white)
    content(("AL_mid_bottom.center"), text(size: 14pt, [$A_L^dagger$]), anchor: "center")
    
     // Top A_L2
    rect((2.0, 0.7), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_top2", fill: white)
    content(("AL_mid_top2.center"), text(size: 14pt, [$A_L$]), anchor: "center")
    
    // Bottom A_L†2
    rect((2.0, -1.6), (rel: (1.2, 0.8)),
         radius: 10%, name: "AL_mid_bottom2", fill: white)
    content(("AL_mid_bottom2.center"), text(size: 14pt, [$A_L^dagger$]), anchor: "center")

    
    // |0⟩⟨0| label in the middle between A_L and A_L† (bigger)
    content((-0.2, -0.0), text(size: 16pt, [$|0 angle.r angle.l 0|$]), anchor: "center")
    
    content((2.6, -0.0), text(size: 16pt, [$|0 angle.r angle.l 0|$]), anchor: "center")

    // Vertical connection between A_L and A_L† (adjusted for bigger size)
    line(("AL_mid_top.south"), (-0.2, 0.2))
    line(("AL_mid_bottom.north"), (-0.2, -0.2))

    line(("AL_mid_top2.south"), (2.6, 0.2))
    line(("AL_mid_bottom2.north"), (2.6, -0.2))
    
    // Horizontal connections (adjusted for bigger size)
    line((-2.7, 1.1), ("AL_mid_top.west"))
    line((-2.7, -1.2), ("AL_mid_bottom.west"))

    line(("AL_mid_top.east"), ("AL_mid_top2.west"))
    line(("AL_mid_bottom.east"), ("AL_mid_bottom2.west"))
   
    // add ZZ
    rect((0.6, -0.5), (rel: (0.8, 0.8)),
         radius: 10%, name: "Z1", fill: white)
    content(("Z1.center"), text(size: 14pt, [$Z$]), anchor: "center")

    rect((3.4, -0.5), (rel: (0.8, 0.8)),
         radius: 10%, name: "Z2", fill: white)
    content(("Z2.center"), text(size: 14pt, [$Z$]), anchor: "center")

    line(("AL_mid_top.north"),(-0.2,2.0))
    line(("AL_mid_bottom.south"),(-0.2,-2.0))
    line((-0.2,2.0),(1.0,2.0))
    line((-0.2,-2.0),(1.0,-2.0))
    line((1.0,2.0),"Z1.north")
    line("Z1.south",(1.0,-2.0))

    line(("AL_mid_top2.north"),(2.6,2.0))
    line(("AL_mid_bottom2.south"),(2.6,-2.0))
    line((2.6,2.0),(3.8,2.0))
    line((2.6,-2.0),(3.8,-2.0))
    line((3.8,2.0),"Z2.north")
    line("Z2.south",(3.8,-2.0))
    line("AL_mid_top2.east",(4.5,1.1))
    line("AL_mid_bottom2.east",(4.5,-1.2))
    line((4.5,1.1),(4.5,-1.2))

  })
}

    #align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - quantum channel (large)
    draw_quantum_channel_large(),
   
  )
]

  - Measure: 
    - $ angle.l X_i angle.r$
      #let draw_quantum_channel_large() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1.5pt, fill: white)
    
    // Left ρ rectangle (bigger)
    rect((-4.5, -1.5), (rel: (1.8, 3.0)), 
         radius: 10%, fill: rgb(200, 230, 255), name: "rho_left")
    content(("rho_left.center"), text(size: 18pt, [$rho_L$]), anchor: "center")
    
    // M
    rect((-1.8, -0.7), (rel: (0.8, 1.6)),
         radius: 10%, name: "M", fill: white)
    content(("M.center"), text(size: 14pt, [$M$]), anchor: "center")
    
    line((-2.7,1.2),(-0.5,1.2))
    line((-2.7,0.5),(-1.8,0.5))
    line((-2.7,-0.3),(-1.8,-0.3))
    line((-2.7,-1.2),(-0.5,-1.2))
   

  })
}

  #align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - quantum channel (large)
    draw_quantum_channel_large(),
   
  )
]
    - $ angle.l Z_i Z_(i+1) angle.r$

     #let draw_quantum_channel_large() = {
  canvas({
    import draw: *
    
    set-style(stroke: 1.5pt, fill: white)
    
    // Left ρ rectangle (bigger)
    rect((-4.5, -1.5), (rel: (1.8, 3.0)), 
         radius: 10%, fill: rgb(200, 230, 255), name: "rho_left")
    content(("rho_left.center"), text(size: 18pt, [$rho_L$]), anchor: "center")
    
    // M
    rect((-1.8, -0.7), (rel: (0.6, 1.4)),
         radius: 10%, name: "M1", fill: white)
    content(("M1.center"), text(size: 14pt, [$M$]), anchor: "center")
    
    rect((-0.7, -0.9), (rel: (0.6, 1.8)),
         radius: 10%, name: "M2", fill: white)
    content(("M2.center"), text(size: 14pt, [$M$]), anchor: "center")

   
    line((-2.7,0.5),(-1.8,0.5))
    line((-2.7,-0.5),(-1.8,-0.5))
   
    line((-2.7,0.8),(-0.7,0.8))
    line((-2.7,-0.8),(-0.7,-0.8))

    line((-2.7,1.2),(0.5,1.2))
    line((-2.7,-1.2),(0.5,-1.2))
   
  
  })
}

    #align(center)[
  #grid(
    columns: (auto, auto, auto),
    column-gutter: 1em,
    align: center + horizon,
    
    // Left side - quantum channel (large)
    draw_quantum_channel_large(),
   
  )
]


= Algorithm Box

== 

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
        Assign[$theta$][$theta_0$]
        Assign[iter][$0$]
        LineBreak
        While(
          "iter < maxiter && g_tol > 1e-10",
          {
             Comment[Construct parameterized quantum circuit]
             Assign[$U(theta)$][$"ConstructCircuit"(theta)$]
         
             LineBreak
             Comment[Iterate quantum channel to fixed point]
             Assign[$rho_L$][$"IterateChannel"(U(theta))$]
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
#import "@preview/quill:0.7.2": *
#import "@preview/quill:0.7.2" as quill: tequila as tq

(1) get $angle.l X angle.r$ by:
#quill.quantum-circuit(
    lstick($|0〉$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),$H$, meter(),midstick($|0 angle.r$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),3,$R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),$H$,meter(),1,rstick("..."),[\ ],
    lstick($|0〉$), $R_z (theta_3)$, $R_x (theta_4)$, targ(), 3,$R_z (theta_3)$, $R_x (theta_4)$, targ(),$H$,meter(),midstick($|0 angle.r$),$R_z (theta_3)$, $R_x (theta_4)$, targ(),3,
  rstick("..."),
  quill.gategroup(x: 1,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 7,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 13,y:0,2,3,label: (content:$times p$,pos:top)),
 
)

(2) get $angle.l Z Z angle.r$ by:

#quill.quantum-circuit(
    lstick($|0〉$), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1), 3,meter(),midstick($|0 angle.r$), 3,$R_z (theta_1)$, $R_x (theta_2)$,ctrl(1),1, rstick("..."),[\ ],
    lstick($|0〉$), $R_z (theta_3)$, $R_x (theta_4)$, targ(), $R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),meter(),midstick($|0 angle.r$),$R_z (theta_1)$, $R_x (theta_2)$, ctrl(1),$R_z (theta_3)$, $R_x (theta_4)$, targ(),meter(),rstick("..."),[\ ],
    lstick($|0〉$), 3,$R_z (theta_3)$, $R_x (theta_4)$, targ(),2,$R_z (theta_3)$, $R_x (theta_4),$, targ(),3,meter(),rstick("..."),

  quill.gategroup(x: 1,y:0,2,3,label: (content:$times p$,pos:top)),
  quill.gategroup(x: 4,y:1,2,3,label: (content:$times p$,pos:bottom)),
  quill.gategroup(x: 9,y:1,2,3,label: (content:$times p$,pos:bottom)),
  quill.gategroup(x: 12,y:0,2,3,label: (content:$times p$,pos:top)),
 
)

Iterate until $angle.l X(theta) angle.r _n - angle.l X(theta) angle.r _(n-1) <=10^(-5) $ and  $angle.l Z_i (theta) Z_(i+1) (theta) angle.r _n - angle.l Z_i (theta) Z_(i+1) (theta) angle.r _(n-1) <=10^(-5) $, $n$ means the n-th iteration.

#align(center)[
  #image("images/arrow.png", width: 5%)
  repeat (1) (2)
]
(3) $angle.l H(theta) angle.r = -g angle.l X(theta) angle.r - J angle.l Z_i (theta) Z_(i+1) (theta) angle.r $
    
    $partial/(partial theta_i) angle.l H(theta) angle.r => theta->theta+delta theta$ 


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

