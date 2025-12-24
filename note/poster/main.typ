#import "@preview/peace-of-posters:0.5.0" as pop
#import "@preview/cetz:0.2.2": canvas, draw, tree, plot
#import "@preview/pinit:0.1.3": *


#show link: set text(blue)
#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 1.2em
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#pop.title-box(
  "Designing isoPEPS Ansatz to Solve the Barren Plateau Problem",
  authors: [Yuqing RONG, Longli ZHENG, Guoyi ZHU, Jinguo LIU],
  institutes: text(36pt)[
  $""$Advanced Materials Thrust, Function Hub, Hong Kong University of Science and Technology (Guangzhou)\
  ],
  image: image("amat-dark.png", width: 150%),
  title-size: 1.5em,
)

#columns(2,[

  #pop.column-box(heading: "Abstract")[
The Variational Quantum Algorithm (VQA) is a powerful tool for solving quantum many-body problems, but it faces a critical challenge: the barren plateau problem, where the optimization landscape becomes flat and difficult to navigate, slowing convergence. To address this, we propose using Isometric Projected Entangled Pair States (isoPEPS) as an initialization strategy. IsoPEPS with lower entanglement entropy offer a smoother optimization landscape, reducing susceptibility to barren plateaus and improving convergence.
We map the isoPEPS ansatz onto quantum circuits and benchmark it on Rydberg atom arrays, which offer key advantages for large-scale quantum computations. These arrays enable high parallelization, allowing simultaneous operations that enhance efficiency, and their scalability supports testing quantum algorithms across various problem sizes. Additionally, long-range interactions between Rydberg atoms facilitate complex quantum operations, which are essential for optimizing isoPEPS-based circuits. By leveraging these features, we aim to demonstrate computational advantages over classical methods, particularly in efficiently computing long-range correlation functions, showcasing the potential of quantum methods for tackling challenges of many-body problems.


  ]


  #pop.column-box(heading: "Barren Plateau problem")[
Vanishing gradients
#grid(
    columns: 2,
    gutter: 20pt,
image("circuits.png", width: 70%),
box(inset: 1pt)[
  #v(5cm)
  $C(theta)<->theta'$
])

#pinit-highlight(1, 2)
#pinit-point-from(1, pin-dy: 15pt, pin-dx: 10pt, body-dx: -350pt)[4 independent sets with 2 vertices]

- cost function for the optimization problem: $C(theta)=⟨0|U^+(theta)O U(theta)|0⟩$

- "Deep" random parameterized circuits $d~O(p o l y(n))$
- Random initialization of parameters causes

 1.The circuits exhibits the characteristics of a *2-design*

 2.Gradients vanish *exponentially* with the number of qubits
 
 #h(7cm)$⟨partial_k C⟩ =0$, $V a r[partial_k C] approx 2^(-n)$
  ]

  // These properties will be given to the function which is responsible for creating the heading
  #let hba = pop.uni-fr.heading-box-args
  #hba.insert("stroke", (paint: gradient.linear(green, red, blue), thickness: 10pt))

  // and these are for the body.
  #let bba = pop.uni-fr.body-box-args
  #bba.insert("inset", 30pt)
  #bba.insert("stroke", (paint: gradient.linear(green, red, blue), thickness: 10pt))

  #pop.column-box(heading: "Why isoPEPS?", stretch-to-next: true)[
    PEPS ( Projected Entangled Pair States) is a tensor network method for representing 2D quantum many-body states. 
For a rank-$d D^4$ tensor,
$
 |𝜓⟩ = sum_({s}=1)^(d)(sum_({alpha beta gamma delta}=1)^D ... A_(alpha_i beta_i,gamma_i,delta_i)^(s_i) ...)|...s_i ...⟩
$
For isoPEPS, each site $A^(s_i)$ satisfies the isometry condition 
$A^(s_i) (A^(s_i))^+ =II $, ensuring the state is orthogonal.  @Zaletel_2020
 
#grid(
    columns: 2,
    gutter: 20pt,
    
  
  image("isoPEPS.png", width: 80%),  box(inset: 1pt)[
- *Efficient representation*: IsoPEPS alleviate the entanglement of quantum states without storing the full wavefunction.

- *Scalability*: IsoPEPS handles the exponentially large Hilbert space in the thermodynamic limit.
- *Efficient optimization*: IsoPEPS addresses the barren plateau problem in Variational Quantum Algorithms (VQAs), facilitating effective optimization of quantum states.
  ])


  ]

#colbreak()

  #pop.column-box(
    heading: "Mapping to quantum circuits!",
  )[
First we introduce how one column of PEPS (MPS) map to quantum circuits. 
The idea is to construct unitary matrix product operators, dubbed as matrix product disentanglers (MPD) that disentangle the targeted MPS.@Ran_2020 @Liu_2019
  #grid(
    columns: 2,
    gutter: 20pt,
    
  image("mpscircuit.png", width: 90%),  box(inset: 5pt)[
#import "@preview/physica:0.9.1": *
MPS: $A^([i])->$ MPD: $G^([i])->$ quantum circuit

They satisfy:

(a)i=0, $G_(0 j k l)^([n])=A_(j k l)^([n])$

(b)i = 1, · · · , d-1,$G_(i j k l)$ are obtained in the kernel of $A_(j k l)$. $sum_(k l) G_(i^' j^' k l)^n G_(i j k l)^n=I_(i^' i)I_(j^' j)$ 

  ])

  #highlight([For PEPS, methods are similar]) 

   1. *Encoding Tensors as Gates:* Each tensor $A^[i]$ is mapped to a unitary operator $U^[i]$ on the quantum circuit.

   2. *Quantum Circuit Construction:* The isoPEPS network is mapped to a quantum circuit by representing the tensors as a series of quantum gates. The quantum circuit structure mirrors the connectivity of the isoPEPS network, ensuring that the entanglement and correlations are preserved across the qubits.

   3. *Variational Parameterization and Optimization:* The quantum gates $U^[i]$ are parameterized by variational parameters $theta$,which are optimized to minimize the energy of the quantum state. The energy expectation value is computed as: $E(theta)=⟨psi(theta)|H|psi(theta)⟩$.Optimization techniques, such as gradient descent, are employed to update $theta$, leading to an optimized quantum state.

  ]

  #pop.column-box(heading: "Why Rydberg Atom Arrays?")[
  
  #image("Rydeberg.jpg", width: 70%) 

-  *Parallelization*: Rydberg atom arrays allow for high parallelism by enabling the simultaneous execution of quantum operations on multiple qubits, significantly enhancing the speed and efficiency of the optimization process.

- *Long-range connectivity*: The strong van der Waals interactions between Rydberg atoms, $V_(v d W)~C_6/r^6$ ,  is essential for implementing complex quantum operations and optimizing quantum circuits.@Ebadi_2022

  ]


  #pop.column-box(heading: "References", stretch-to-next: true)[ 
    #bibliography("bibliography.bib", title: none)
  ]
])

#pop.bottom-box()[
  #align(right, [#align(horizon, grid(columns:5, column-gutter: 30pt, image("github-dark.png", width: 70pt), "yuqingrong", h(50pt), image("email.png", width: 70pt), "yrong265@connect.hkust-gz.edu.cn"))])
]

