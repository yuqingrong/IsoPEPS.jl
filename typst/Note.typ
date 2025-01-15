#import "@preview/unequivocal-ams:0.1.1": ams-article, theorem, proof

#show: ams-article.with(
  title: [note],
  authors: (
    (
      name: "Yuqing RONG",
      department: [],
      organization: [HUSTGS],
      location: [Tennessee, TN 59341],
      email: "yrong265@connect.hkust-gz.edu.cn",
      url: "math.ue.edu/~jdoe"
    ),
  ),

  
)



= Isometric tensor network states in two dimensions
== MPS 
#import "@preview/cetz:0.1.2": canvas, draw
#canvas(length: 0.5cm, {
  import draw: *
  
    circle((0, 0))
    circle((4,0))
    circle((8,0))
    circle((12,0))
    line((1,0), (3,0))
    line((5,0), (7,0))
    line((9,0), (11,0))
    line((0,1),(0,2))
    line((4,1),(4,2))
    line((8,1),(8,2))
    line((12,1),(12,2))
  }
)
$psi$=$T^1T^2...T^N$

not take dimension contraction into account, $kai_n=min(d_1 times...times d_n, d_(n+1) times...times d_N) $

isometry condition: $A A^*=I$, $B B^*=I$, then

#import "@preview/cetz:0.1.2": canvas, draw
#canvas(length: 0.5cm, {
  import draw: *
  
    circle((0, 0),)
    circle((4,0)) 
    circle((12,0))
    line((1,0), (3,0),mark:(end:">"))
    line((5,0), (7,0),mark:(end:">"))
    line((9,0), (11,0),mark:(start:">"))
    line((0,1),(0,2))
    line((4,1),(4,2))
    line((8,1),(8,2))
    line((12,1),(12,2))
    line((7,0),(8,1))
    line((8,1),(9,0))
    line((7,0),(8,-1))
    line((8,-1),(9,0)) 
  }
)
$psi$=$A^1 A^2...A^(l-1) Lambda^l B^(l+1)...B^N$, $Lambda$ is the orthogonal center (also the entanglement spectrum) 

$<psi|O^l|psi> = <Lambda|O^l|Lambda>$ (A...B...are isometry)

== 2D
#figure(
  image("images/isoTNS.png", width: 80%),
  caption: [isoTNS],
) 
the red tensor = orthogonality center (OC) 

red and the blue tensors with light red background = the orthogonality hypersurface

== Moses Move

very close to the optimal variational result; fast.
#figure(
  image("images/MM.jpg", width: 100%),
  caption: [Moses Move detail],
) 
Repeat the steps above, the orthogonal center moves upward,so we get FIGURE 3:
#figure(
  image("images/MM.png", width: 80%),
  caption: [Moses Move],
) 
Then $Lambda$ and $B^(l+1)$ are tied together to form new $Lambda$,
$Lambda^(l+1)=Lambda B^(l+1)$

== $T E B D^2$ algorithm

(1)Suzuki-Trotter decomposition:
#import "@preview/physica:0.9.1": *
$vu(U)$(dt)=$product_(r,i) e^(-i d t H_i^r) product_(c,j) e^(-i d t H_j^c) $

(2)Bond truncation local updated at the orthogonality center, just like time evolution operator act on it.

(3)Utilize the SVD and MM to move around the orthogonality center and orthogonality hypersurface.

#figure(
  image("images/TEBD.png", width: 80%),
  caption: [TEBD],
) 
(i)After 1 round, the isometries rotate $90 degree$ counterclockwise, apply 
#import "@preview/physica:0.9.1": *
$vu(U)^(c o l)$(dt)=$product_(c,j) e^(-i d t H_j^c) $ to the states.

(ii)After 2 round,  the isometries rotate $180 degree$ counterclockwise, apply 
#import "@preview/physica:0.9.1": *
$vu(U)^(r o w)$(dt)=$product_(r,i) e^(-i d t H_i^r) $ to the states.

(iii)After 4 round, the orientation of the isometries back to initialization. Errors are canceled out via symmetrization. 

== DMRG algorithm
=== 1D

update each tensor $A^((n))$, such that

#import "@preview/physica:0.9.1": *
$E_g=min_(<psi|psi> =1) <psi|vu(H)|psi>$ = $min_(<psi|psi> =1) sum_(i,j)<psi|vu(H)_(i j)|psi>$

eg: update the 2nd tensor(other tensor can be seen as fixed tensors), the effective Hamiltonian $H_(e f f)$:
#figure(
  image("images/EH.jpg", width: 100%),
  caption: [effective Hamiltonian],
) 
Because all tensors are isometric and canonical transformation, we can move the orthogonality center to the 2nd tensor. Then the problem become

#import "@preview/physica:0.9.1": *
$min_(<A^((2))|A^(2)^*> =1) <A^((2))|vu(h_2)|A^(2)^*>$

#figure(
  image("images/FP.jpg", width: 50%),
  caption: [contract other tensors],
) 

summary:

(1)Initialize all the tensors randomly

(2)SWEEP: Update the tensors from the 1st to Nth in order, then from Nth to 1st as:

(i)Move the orthogonality center to the nth tensor

(ii)Calculate the corresponding effective Hamiltonian $h^n$ 

(iii)Update $A^((n))$ as the min eigenstate 

(3)If MPS converge, done; else if, return to (2)

=== 2D

The operations are similar to 1D, except with the addition of a dimension:
#figure(
  image("images/2D_DMRG.png", width: 80%),
  caption: [2D DMRG],
) 

== Area Law

(1)a pure state's density operator $rho=|psi><psi|$,at zero temperature has vanishing von-Neumann entropy:

S($rho$)=-tr[$rho log_2 rho$]= - $sum_x lambda_x log_2 lambda_x$

We can use partial trace to find the reduced density matrix:
$rho_A=Tr_B (rho)$

(2)area law of entanglement entropy: 

divide a D dimension lattice quantum states into two parts, their entropy satisfy:

$S prop O(l^(D-1))$, $l$ represents the length scale

eg: MPS, D=1, $S prop O(1)$; PEPS, D=2, $S prop O(l^1)$

== truncation error


= Synergy Between Quantum Circuits and Tensor Networks: Short-cutting the Race to Practical Quantum Advantage 

== Born rule

if a quantum system is described by a wave function 
$|psi>$, and an observable (e.g., position, momentum) has eigenstates $|𝜙_𝑛⟩$ with corresponding eigenvalues. the Born rule states that the probability P of measuring the system to be in the state $|𝜙_𝑛⟩$ is given by:

$ P=|⟨ϕ_n|ψ⟩|^2$

== Kullback-Leibler (KL) divergence

Given two probability distributions P(true distribution) and 
Q(approximate distribution), the KL divergence from Q to P is defined as:

(1)discrete case:$D_(K L) (P||Q) =sum_x P(x)log P(x)/Q(x)$ 

(2)continuous case: $D_(K L) (P||Q) = integral P(x) log P(X)/Q(x) d x $

Properties: 

(a)$D_(K L) (P||Q)>=0 $ ;(b)$D_(K L) (P||Q) eq.not D_(K L) (Q||P) $

== some gates

(1)*U(2)* gate: can represent any rotation on a qubit and can represent any single qubit gates.

$ U=mat(e^(i alpha)cos theta, -e^(i beta)sin theta; e^(i beta)sin theta, e^(-i alpha)cos theta)$  or  $ U(theta,Phi,lambda)=mat(cos theta/2, -e^(i lambda) sin theta/2 ;e^(i Phi) sin theta/2, e^(i (Phi+lambda)cos theta/2) )$

(2)*XX,YY,ZZ* gate: represent a rotation around XX, YY, ZZ axis in the space of two qubits; generate entanglement between 2 qubits.

$ X X(theta)= exp(-i theta/2 X_1 X_2)= mat(cos theta/2, 0, 0, -i sin theta/2; 0, cos theta/2, -i sin theta/2, 0; 0, -i sin theta/2, cos theta/2, 0; -i sin theta/2, 0, 0, cos theta/2)$

$X_1$ and $X_2$ are the Pauli-X operators acting on the first and second qubits respectively; $theta$ is a real parameter determines the rotation angle around the XX axis.

(3)*SU(4)* gate: two-qubits quantum gates; satisfy $U^+U=I,$ determinant=1.

$S U(4)_(i j) (theta)=U(2)(theta)_i(theta_(1:3)) times U(2)(theta)_j(theta_(4:6)) times X X_(i j) (theta_7) times Y Y_(i j) (theta_8) times Z Z_(i j) (theta_9) times U(2)(theta)_i(theta_(10:12)) times U(2)(theta)_j(theta_(13:15))$

(4)*U(4)* gate: 

$U(4)=S U(4)(theta) times e^(-i Phi)$, $e^(-i Phi)$ is a global phase.

= Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates 

== Encoding matrix product state into single-layer quantum circuit

#figure(
  image("images/qc.jpg", width: 90%),
  caption: [MPS $->$ MPD $->$ QC],
) 

MPS(N sites):

$|psi> = sum_(a_1...a_(N-1))sum_(s_1...s_N) A_(s_1,a_1)^[1]A_(s_1,a_1,a_2)^[2]...A_(s_N,a_(n-1))^[N] product_(n=1)^N |S_N> $

(1)They satisfy the normalization and left orthogonal conditions.

$sum_(s_1,a_1)A_(s_1,a_1)^[1]A_(s_1,a_1)^([1]*)=1\ sum_(s_n,a_n)A_(s_n,a_(n-1),a_n)^[n] A_(s_n,a_(n-1)^',a_n)^([n]*)=I_(a_(n-1),a^'_((n-1)))\
sum_(s_N)A_(s_N,a_(N-1))^[N]A_(s_N,a^'_(N-1))^([N]*)=I_(a_(N-1),a^'_((N-1)))$

(2)$kai=d=2$

MPS: $|psi>$; 
#import "@preview/physica:0.9.1": *
MPD: $vu(U) $ 

$|psi> =vu(U)^+|0>$

$A_(j k l)->G_(i j k l)$ (a)i=0, $G_(0 j k l)^([n])=A_(j k l)^([n])$
(b)i = 1, · · · , d − 1,$G_(i j k l)$ are obtained in the kernel of $A_(j k l)$. $sum_(k l) G_(i^' j^' k l)^n G_(i j k l)^n=I_(i^' i)I_(j^' j)$ 

dim(i)=dim(k); dim(j)=dim(l)

(3)negative logarithmic fidelities (NLF) per site:

#import "@preview/physica:0.9.1": *
$F_0 = -(ln|<psi|psi_(kai=1)^~ >|)/N$ 

$F_1 = -(ln|<psi|vu(U^+)|0>|)/N$

if $|psi_(kai=1)^~ > = |psi>$, $F_0=0$; The gap between $|psi_(kai=1)^~ >$ and $|psi> $ greater, $F_0$ greater.

Our goal is minimizing $F_1$. 
#show: set text(red)
cost function?

#show: set text(black)
== Deep quantum circuit 

#figure(
  image("images/dqc.png", width: 80%),
  caption: [D=2 layers quantum circuit],
) 

D layers: $|psi^~ > = vu(U)_D^+...vu(U)_2^+ vu(U)_1^+|0>$
$quad F_D =-(ln|<psi|vu(U)_D^+...vu(U)_2^+ vu(U)_1^+|0>|)/N$

= Simulating Large PEPs Tensor Networks on Small Quantum Devices 

== PEPS

#figure(
  image("images/PEPS.png", width: 60%),
  caption: [PEPS],
) 

$|psi> =sum_(s=1)^N (sum_({alpha,beta,gamma,delta}=1)^D ...A_(alpha_i,beta_i,gamma_i,delta_i)^(s_i)...) |S_n>$

== Relationship of the qubits

#import "@preview/i-figured:0.2.3"
1. virtual qubits: $N_B = ceil(log kai)$

$kai$ represents the rank of entanglement,that is, the number of possible states that can be shared across the bond. So, $kai=2^(N_B)$

2. $N times M$ lattices, total qubits: $N_Q=(N+1) times N_B+1$
$N+1$ means open boundary condition, $+1$ means auxiliary qubit uesd to reset or measure.

$N_Q$ is the total number of virtual qubits. Physical qubits are not included because they are not the main resource cost.

3. qubit-efficient:

$N_Q=(N+1) times N_B+1 <N times M$, that is

$M>floor((1+1/N)N_B+1/N) $ or $N_B<floor((N times M -1)/(N+1))$

== more output than input, add additional input 

#figure(
  image("images/indices.png", width: 60%),
  caption: [PEPS],
) 

tensor 1 has 3 output and 0 input, so we add 3 input and fix them |0>

== Example of $4 times 4$

1. zig-zag pattern
#figure(
  image("images/pattern.jpg", width: 60%),
  caption: [zig-zag pattern of $4 times 5$],
) 

2. map to quantum circuit

#figure(
  image("images/circuit.jpg", width: 120%),
  caption: [6-qubits circuit of $4 times 4$],
)
the qubits needed only rely on N

= MPO representation of Hamiltonian

Given an Hamiltonian H, the MPO expresses it as a tensor network:

$H=sum_(i j k...)M^[1]M[2]...M[N]$

M is a tensor encodes local operators.
1. Ising model

$H_(I s i n g)=-J sum_i S_i^z S_(i+1)^z-h sum_i S_i^x $

$M=mat(I, S^z, S^x; 0, 0, S^z; 0, 0, I)$

The left boundary: ML=$vec(1,0,0) $ ; The right boundary: MR=$vec(0,0,1) $. Therefore, only the 1st row 3rd column works.

bond dimension: D=3
 
2. Heisenberg model 

$H_(H e i s e n b e r g)=J sum_i S_i^x S_(i+1)^x + sum_i S_i^y S_(i+1)^y + sum_i S_i^z S_(i+1)^z $

M=$mat(I, S^x,  S^y,  S^z, 0; 0, 0, 0, 0,  S^x; 0, 0, 0, 0,  S^y; 0, 0, 0, 0,  S^z; 0, 0, 0, 0, I)$

D=5

= Lanczos method

1. Courant-Fischer Minimax Theorem: If $A∈RR^(n times n)$ is symmetric, then

$lambda_k (A)=max_(dim(S)=k) min_(0 !=y∈S) (y^T A y)/(y^T y)$, for k=1:n.

make sense: $y=sum_(i=1)^k c_i q_i$, $A q_i=lambda_i q_i =>q_i^T A q_i=lambda_i=>R(A,y)=(y^T A y)/(y^T y)=((sum_(i=1)^k c_i q_i)^T A (sum_(i=1)^k c_i q_i))/((sum_(i=1)^k c_i q_i)^T (sum_(i=1)^k c_i q_i))=(sum_(i=1)^k lambda_i c_i^2)/(sum_(i=1)^k c_i^2)$

$lambda_1>lambda_2>...>lambda_k>...>lambda_n$

set of all possible solutions of $min_(0 !=y∈S) (y^T A y)/(y^T y)$ is ${lambda_k,...,lambda_n}$, then, the maximum of them is $lambda_k$

2. Krylov Subspace: $K(A,q_1,k)=[q_1,A q_1,A^2 q_1,...,A^(k-1) q_1]$

$Q^T A Q=T, Q Q^T=I_n$

$Q=[q_1, q_2, ..., q_n]$

$T=mat(alpha_1, beta_1, ..., 0; beta_1, alpha_2,...,... ;
..., ...,..., beta_(n-1);0,... ,beta_(n-1),alpha_n)$

since $A Q=Q T, A q_k=beta_(k-1) q_(k-1)+alpha_k q_k+beta_k q_(k+1)=>q_k^T A q_k=alpha_k, $

$r_k=(A-alpha_k I)q_k-beta_(k-1) q_(k-1)=beta_k q_(k+1) =>q_(k+1)=r_k/beta_k$

$beta_k=||r_k||=>q_(k+1)=r_k/(||r_k||)$

$q_0=0, beta_0=1$, $r_0=q_1$ is randomly chosen

= TEBD+MPS

MPS, N sites

1. Trotter decomposition

$H=sum_i h_i$, 
$h_i=S_i^x S_(i+1)^x+S_i^y S_(i+1)^y+S_i^z S_(i+1)^z$

discretize time as $t=N tau(N-> infinity, tau->0)$, $tau$ is time step.

- one-order Trotter decomposition: 
 $e^(-i H tau)=e^(-i h_1 tau)e^(-i h_2 tau)...e^(-i h_(N-1) tau)+O(tau^2)$

Since $[h_i,h_(i+1)]!=0$, $[h_i,h_(i+2)]=0$, we decompose it as

$e^(-i H tau)=e^(-i H_(o d d) tau)e^(-i H_(e v e n) tau)$

- second-order Trotter decomposition:

$e^(-i H tau)=e^(-i H_(o d d) tau/2)e^(-i H_(e v e n) tau)e^(-i H_(o d d) tau/2)+O(tau^3)$

2. operate on MPS
#figure(
  image("images/TEBD1.png", width: 60%),
  caption: [Thin and Fat lines correspond to dimension 1 and > 1 on MPO bonds.],
) 

= Tensor decompositions

*using LinearAlgebra*
== SVD

(1)F=svd(A) 

F.U 

F.S is a vector, need to be: diagm(0 => F.S) to form a matrix

F.Vt return $V^+$

(2)can be used to generate unitary and isometric tensors
- unitary: U = svd(rand(d1,d1)).U
- isometric: W = svd(rand(d1,d2)).U

== spectral decomposition

for  Hermitian matrices or tensors, very useful for calculating eigen

H=0.5(A+A')

F=eigen(H)

F.vectors: return eigenvector

F.values: return eigenvalue(vector form,  need to be: diagm(0 => F.S) to form a matrix)

F.vectors': transpose eigenvector

== QR decomposition

$A=r a n d(d_1,d_2)$ 

F=qr(A)

F.Q: a $d_1 times d_2$ isometric matrix

F.R: a $d_2 times d_2$ upper-triangular matrix


= Frobenius norm for tensors

== norm for matrix

$||A||=sqrt(sum_(i j)|A_(i j)|^2)=tr(A^+ A)$

$A_(i j)$ is elements of A

== norm for tensor

$||A||=T t r(A^+ A)$ equal to contraction of A and $A^+$

== relationship of norm and svd

svd(A)=$U S V^+$, then $||A||=sqrt(sum_k|s_k|^2)$ ($s_k$: elements of S)







= Question 

differences between $M P S^2$ and $P E P S$

Morses Move and zig-zag


