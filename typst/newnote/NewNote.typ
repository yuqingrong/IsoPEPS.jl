#import "@preview/clean-math-paper:0.1.0": *

#let date = datetime.today().display("[month repr:long] [day], [year]")
#show: template.with(
  title: "Note for IsoPEPS",
  affiliations: (
    (id: 1, name: "Yuqing Rong"),
  ),
  date: date,
  heading-color: rgb("#2e44a6"),
  link-color: rgb("#12472b"),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
)

#import "@preview/codly:0.1.0": codly-init, codly, disable-codly
#show: codly-init.with()

#codly(languages: (
        julia: (name: "Julia", color: rgb("#41A241"), icon: none),
    ),
    breakable: false
)

= Introduction

This is a note document for the project: IsoPEPS of Rydberg atom arays. It includes basic knowledge (tensor network, matrix computation, Lie Algebra and Lie group, quantum mechanics, quantum computation and quantum information), coding technique of Julia and related paper review.

= Basic knowledge

== Tensor network

== Automatic Differentiation (AD)
Use the chain rule, seperate the gradients as a flow.
=== basis

control flow: 

examble: $y=f(x_1,x_2)=ln(x_1)+x_1 x_2-sin(x_2)$
- input variables: $v_(i-n), i=1,2,...,n$

#h(3.25cm)$v_(-1)=x_1$

#h(3.25cm)   $v_0=x_2$

- intermediate variables: $v_i,i=1,2,...,l$

#h(4.5cm)$v_1=ln(x_1)=ln(v_(-1))$

#h(4.5cm)$v_2=x_1 x_2=v_(-1)v_0$

#h(4.5cm)$v_3=sin(x_2)=sin(v_0)$

#h(4.5cm)$v_4=ln(x_1)+x_1 x_2=v_1+v_2$

#h(4.5cm)$v_5=v_4-v_3$

- output variables: $y_(m-i)=v_(l-i), i=m-1,...,0$

#h(3.5cm)$y=v_5$

#figure(
  image("images/control_flow.png", width: 80%),
  caption: [Computational graph],
) 
=== forward mode
$dot(y_j)=(partial y_j)/(partial x_i)|_(x=a), j=1,...,m;i=1,...,n$

First, we should set at which point we want to calculate the gradients. Then we calcaulate the gradient to $x_1, x_2, ..., x_n$ in order. When we calculate the gradient to $x_i$, we set $dot(x_i)=1$, the others $dot(x_j)=0,j!=i$(they are independent variables).

#figure(
  image("images/forward trace.png", width: 80%),
  caption: [forward mode],
) 

After n evaluations, all gradients to $x_i$ has been calculate. We put them into Jabian matrix.
#figure(
  image("images/Jacobian.png", width: 50%),
  caption: [Jacobian matrix],
) 
=== reverse mode
Suppose the ajiont: $overline(v)_i = (partial y)/(partial v_i)$

As forward mode, we should set the point we want to calculate, and calculate $y_1,y_2,..,y_m$ in order. We set $overline(V_5) = overline(y)=1$. 

#figure(
  image("images/reverse mode.png", width: 80%),
  caption: [Jacobian matrix],
) 

As forward mode, reveerse mode also has Jacobian after m evaluations.
#figure(
  image("images/rJacobian.png", width: 50%),
  caption: [Jacobian matrix],
) 

=== conclusion

$f: RR^n -> RR^m$. If $n>m$, the reverse mode is more efficient, elseif $m>n$, the forward mode is more efficient.

=== code


== Matrix computation
=== Matrix/Vector operations
+ pointwise multiplication: $C=A. *B =>c_(i j)=a_(i j)*b_(i j)$

  pointwise division: $C=A. "/" B =>c_(i j)=a_(i j)"/"b_(i j)$

+ saxpy: (scalar*vector) $y=alpha*x+y =>y_i=alpha*x_i+y_i$, update vector $y$ as $alpha*x+y$. "saxpy" is used in LAPACK.

+ gaxpy: (matrix*vector, matrix is sum of scalars) $y=y+A x =>y_i=y_i+sum_j a_(i j)x_j$, $y in R^m$, $x in R^n$, $A in R^(m*n)$.
 ```julia #Row-oriented gaxpy
for i in 1:m
  for j in 1:n
    y[i] = y[i] + A[i, j] * x[j]
  end
end

#or
for i in 1:m
  y[i] = y[i] + A[i, :] * x
end
```
 ```julia #Column-oriented gaxpy
for j in 1:n
  for i in 1:m
    y[i] = y[i] + A[i, j] * x[j]
  end
end

#or
for j in 1:n
  y = y + A[:, j] * x[j]
end
```
+ outer product: $A=A+x y^T =>a_(i j)=a_(i j)+x_i y_j$, $x in R^m$, $y in R^n$, $A in R^(m*n)$.
 ```julia
for i in 1:m
  A[i, :] = A[i, :] + x[i] * y^T
end
#or
for j in 1:n
  A[:, j] = A[:, j] + x * y[j]
end
```
+ matrix-matrix multiplication: $C=C + A B $. $ A in R^(m*r)$, $B in R^(r*n)$, $C in R^(m*n)$.
 
  - saxpy formulation

   each row of $C$: $c_i = c_i + a_i^T B$

    $A = vec(a_1^T, a_2^T, ..., a_m^T)$ ,  $C = vec(c_1^T, c_2^T, ..., c_m^T)$
  ```julia
  for i in 1:m
    C[i, :] = C[i, :] + A[i, :] * B
  end
  ```
    each column of $C$: $c_j = c_j + A b_j$

    $B = vec(b_1, b_2, ..., b_n)$ , $C = [c_1| c_2| ... |c_n]$
  ```julia
  for j in 1:n
    C[:, j] = C[:, j] + A * B[:, j]
  end
  ```
  #v(0.5cm)
 - outer product formulation

  $A = [a_1| a_2| ...| a_r]$ , $B = vec(b_1^T, b_2^T, ..., b_r^T)$ 
  ```julia
  for i in 1:r
    C = C + A[:,k] * B[k,:]
  end
  ```

=== band matrices: can improve efficiency of matrix-matrix multiplication.
  $A in R^(m*n)$ has lower bandwidth $p$ if $a_(i j)=0$ for $i>j+p$ and upper bandwidth $q$ if $a_(i j)=0$ for $j>i+q$.

  band matrices includes: diagonal, tridiagonal, bidiagonal, Hessenberg, and tridiagonal.
  
+ band storage: $a_(i j)=A.b a n d(i-j+q+1,j)$
  
  eg. : $y=y+A x$, we can replace $A$ with $A.b a n d$ to improve efficiency.
  ```julia
  for j in 1:n
    a1 = max(1, j-q); a2 = min(n, j+p);
    b1 = max(1, q+2-j); b2 = b1 + a2-a1;
    y(a1:a2) = y(a1:a2) + A.band(b1:b2,j) * x(j)
  end
  ```

+ vec storage of symmetric matrices: $a_(i j)=A.v e c((n-j/2)(j-1)+i)), 1<=j<=i<=n$,

  store the lower triangular part of the matrix in vector.

  eg.: $A = mat(1,2,3;2,4,5;3,5,6) <=> A.vec = [1,2,3,4,5,6]$
 
 #v(0.5cm)
=== permutation matrices: reordered rows $I_n$
  eg.: $P = mat(0,1,0,0;0,0,0,1;0,0,1,0;1,0,0,0) $, it can be rewrited as $v=[2,4,3,1]$, which tell us the position of "1" in each row. $P=I_n [v,:]$

+ get submatrix using $v$

 eg.: $v=1:2:n => v=[1 3 5 7]$

 eg,: $A[i,j] = A[1:2:n, 2:2:n]$, the elements would be reordered.

+ 3 examples

 - exchange permutation: 

  $v=[n:-1:1] =>   P=mat(0,0,0,1;0,0,1,0;0,1,0,0;1,0,0,0)$

 -  downshift permutation:

  $v=[4,1,2,3] => P=mat(0,0,0,1;1,0,0,0;0,1,0,0;0,0,1,0)$

 -  upshift permutation:

  $v=[2,3,4,1] => P=mat(0,1,0,0;0,0,1,0;0,0,0,1;1,0,0,0)$

 - mod-p perfect shuffle: 

  $P_(p,r) = I_n ([(1:r:n) (2:r:n) ... (r:r:n)],:)$

#v(0.5cm)
=== Kronecker product
+ properties:

  - $(B times.circle C)^T = B^T times.circle C^T$
  - $(B times.circle C)(D times.circle F) = B D times.circle C F$
  - $(B times.circle C)^(-1) = B^(-1) times.circle C^(-1)$
  - $B times.circle(C times.circle D) = (B times.circle C) times.circle D$
  - perfect shufffle permutation: $P(B times.circle C)Q^T = C times.circle B$

+ kronecker peoduct $=>$ matrix product (improve efficiency)

 $Y = (B times.circle C) X^T = (C X B^T)^T = B X^T C^T$
#v(0.5cm)
=== Hamiltonian and Symplectic matrix

+ Hamiltonian matrix: 

 $M = mat(A,G; F, -A^T)$, $F^T=G$  or

 $J M J^T = -M^T$, $J= mat(0,I_n; -I_n, 0)$

+ symplectic matrix:

 $S^T J S = J$, $J= mat(0,I_n; -I_n, 0)$

=== Strassen matrix multiplication

=== Fast Fourier Transform (FFT)
 For a polynomial $P(x)=p_0+p_1x+p_2x^2+...+p_n x^(n-1)$, we have 2 representations:
 + coefficient representation: $[p_0, p_1, p_2, ..., p_(n-1)]$
 + point-value representation: $[(x_0, P(x_0)), (x_1, P(x_1)), (x_2, P(x_2)), ..., (x_(n-1), P(x_(n-1))]$
If we use value representation,

$vec(P(x_0), P(x_1), P(x_2), dots.v, P(x_(n-1))) = mat(1, x_0, x_0^2, ..., x_0^(n-1); 1, x_1, x_1^2, ..., x_1^(n-1); 1, x_2, x_2^2, ..., x_2^(n-1); dots.v,dots.v,dots.v, dots.v;1, x_(n-1), x_(n-1)^2, ..., x_(n-1)^(n-1)) vec(p_0, p_1, p_2, dots.v, p_(n-1)) $

needs $n$ points?

$arrow.b$

if $P(x)=x^2 =>P(-x)=P(x),  n "points"=>2n "points"$

if $P(x)=x^3 =>P(-x)=-P(x),  n "points"=>2n "points"$

$P(x)$ can be cut into even and odd parts: 

$P(x)=p_0+p_1x+p_2x^2+...+p_n x^(n-1)=p_e (x^2) + x p_o (x^2)$

$n "points" [x_1, x_2, ..., x_n]$ 

let $x_2=-x_1, x_4=-x_3, ..., x_n=-x_(n-1)$

$arrow.b$

$n/2 "points" [x_1^2, x_2^2, ..., x_(n/2)^2]$

find the next pairs, let $x_2^2=-x_1^2, x_4^2=-x_3^2, ..., x_(n/2)^2=-x_(n/2-1)^2$

$arrow.b$

$n/4 "points" [P(x_1^2), P(x_2^2), ..., P(x_(n/4)^2)]$

$dots.v$

finally, we need only 1 point.
#v(0.7cm)
So, how to find these points?

$N$ roots of unity (on a unit circle): 

$z^n=1 => z=e^(2 pi i /n)$ 

then we set the points as: $omega^0, omega^1, omega^2, ..., omega^n$, we can easily find their pairs and cut half of them.
#v(0.7cm)

=== vector norms

 $p-n o r m s: ||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p), p>=1$

+ Holder inequality: $|x^T y| <= ||x||_p ||y||_q, 1/p+1/q=1$
  
  Cauchy-Schwarz inequality: $|x^T y| <= ||x||_2 ||y||_2$
  
+ convergence: if $lim_(k->infinity)||x^(k)-x||_p->0, p>=1$, then $x^(k)$ converges to $x$.

=== matrix norms

 + $p-n o r m s$: $||A||_p = max_(x != 0) (||A x||_p)/(||x||_p)$
  - $||A||_2 = sqrt(lambda_(max)(A^T A))$, equal to the largest singular value of $A$.
  - $||A||_1 = max_j sum_i |a_(i j)|$, largest sum over column.
  - $||A||_infinity = max_i sum_j |a_(i j)|$, largest sum over row.
 + Frobenius norms: $||A||_F = sqrt(sum_i sum_j |a_(i j|^2)) = sqrt(tr(A^H A))$

 === SVD
 Intuitive understanding: A matrix is an operation. An arbitrary object transformation can be seen as rotation, dimension adding(erasing), stretching, rotation.
 $A = U Sigma V^T$ $V^T$ for rotation, $Sigma$ for dimension adding(erasing)and stretching, $U$ for rotation.

=== LU decomposition
  $A = L U$, $L$ is a lower triangular matrix, $U$ is a upper triangular matrix.

  eg.: $A = mat(3,5;6,7)$

    $A = I A = mat(1,0; 0,1) mat(3,5;6,7) = mat(1,0;2,1) mat(3,5;0,-3)$

    $r_2 - 2*r_1$, so that it should add $2*r_1$ to become $A$.
== Lie Algebra and Lie group
=== general concepts
+ Lie group:

  a  Lie group is a smooth manifold that is also a group.
  
  - $S^1$: a circle, geometric object
  - $G L(n,CC)$: general linear group, ${n times n "invertible matrices" | CC}$
+ Lie Algebra:

  algebra with "Lie bracket" $[ , ]$, $[X,Y] = X Y - Y X$
  
  - $g l(n;CC)$: general linear algebra. ${n times n "matrices" | CC}$
== Quantum mechanics
=== transfer matrix
 $psi_(n+1) = T psi_ n$, $T$ is transfer matrix.

== Topology


= Coding technique

== Module
+ differences betweeb *using* and *import*:

  *using*: can use the fields directly which were exported by module.

  *import*: should be used with module name, Module.cat   
+ 
```julia
using .NiceStuff: nice, DOG
# will import the name nice and DOG
```

== Function 

+ objectid(x): return a unique identifier for x, something like what it was store in the storage system in computer.

+ haskey(collection, key) --> Bool: judge wether collection has the key.
 (has key)

+ Closures: The return of the function is a function.
 ```julia
function make_multiplier(factor)
    return x -> x * factor  
end

multiplier = make_multiplier(2)
multiplier(3)
6
```

+ promote(x,y): convert x and y to the same type.
 ```julia
 promote(1, 2.5)
 (1.0, 2.5)
 ```
== Type

+ Ref: If a function argument type is Ref, it can be modified in-place.
  $a = "Ref"(10)-->a[]=10, "we can modify" a[]=20$.

+ We can define an abstract type
 ```julia
abstract type «name» end
abstract type «name» <: «supertype» end
```

+ parametric type:
 ```julia
struct Point{T} # T is a type parameter
  field1::T
  field2::T
end

# We can use the parametric type to create a concrete type
Point{Float64}
```
 ```julia
 Point{Float64}<:Point{Real}
 false

 Point{Float64}<:Point{<:Real}
 true
 ```
+ tuple type:
 ```julia
 struct Tuple2{A,B}
    a::A
    b::B
 end
```
+ operations on type
 + isa: check if a object is a given type.
  ```julia
  isa(1, Int)
  true
  ```
 + supertype: get the supertype of a type.
  ```julia
  supertype(Point{Float64})
  Point{Real}
  ``` 

== Constructors
+ outer constructor: define function out of the struct, more flexible.
 ```julia
 struct Foo
   x::T
   y::T
 end
 #function can be defined like this
 Foo(x) = Foo(x, x)
 Foo() = Foo(x)
 ```
+ inner constructor: define function inside the struct, more restrictive.
 ```julia
 struct OrderedPair
  x::Real
  y::Real
  OrderedPair(x,y) = x > y ? error("out of order") : new(x,y) 
  # constraints and function definition.
end
 ```


== Operation

+ $x div y$: integer divide
+ updating operators: += , -= ,  /= , \=,  ÷= , %= , ^=  ,&= , |= , ⊻=  ,>>>=,  >>= , <<=.
+ vectorized "dot" operator: $[1,2,3]. "^" 3=[1, 8, 27]$
+ "pipe" operator	|>: x |> f pass x to f$->$f(x)
+ cbrt(x): ∛x

== Key words

+ global: a=1; global a+=1 change the value of a.
+ break: break the iteration. eg:

 ```julia
if j >= 3
  break
end
```
 a similar use is continue

+ baremodule: a lightweight module, module contains base functions when construct, but baremodule doesn't. It has to imports manually.

+ outer: change value of the variables out of the block.
  ```julia
function f()
  i = 0
  for outer i = 1:3
    # empty
  end
  return i
end;

f()
3
```
+ const: give specific, unchanging value 
 ```julia
 const a=1
 ```
== blocks
+ for, zip
 var1 = value1
 ```julia
for (j, k) in zip([1 2 3], [4 5 6 7])
  println((j,k))
end

(1, 4)
(2, 5)
(3, 6)
```
+ let: create local variables without pollute the global ones.
 ```julia
let 
  var1 = value1
  var2 = value2
  operations on var1, var2...
end
```
= Paper review

== Simulating 2D topological quantum phase transitions on a digital quantum computer @Liu_2024
+ sign function: 

  $s i g n = +1(g>0); 0(g=0); -1(g<0)$

+ $ZZ_2$ symmetry:

  $ZZ_2$ group: {0,1}, a system has only two distinct states. $ZZ_2$ operates twice on a state will return the oroginal state.

  eg.: bit flip: $X|0 angle.r = |1 angle.r, X X|0 angle.r = |0 angle.r$

  #h(0.75cm)     phrase flip: $Z|1 angle.r = -|1 angle.r, Z Z|1 angle.r = |1 angle.r$
+ time reversal symmetry:

  $T|psi(t) angle.r = |psi(-t) angle.r$

+ $ZZ_2^T$ symmetry (Time-Reversal Symmetry in $ZZ_2$ Systems):

  $T X T^(-1) = X$, $T Y T^(-1) = Y$, $T Z T^(-1) = Z$

== Efficient Simulation of Dynamics in Two-Dimensional Quantum Spin Systems with Isometric Tensor Networks @PhysRevB.106.245102

+ *isometry is a linear map*:  

  $W: V_s ->V_l$

  $W^dagger W = II, W W^dagger = P_(V_s)$, $P_(V_s)$ is projection operator of $V_s "to" V_l$.

+ *$LL_2$ norm*: Euclidean length of the vector

 $v = (v_1, v_2, ..., v_n)$ in $RR^n "or" CC^n$, 

 $||v||_2 = sqrt(|v_1|^2 + |v_2|^2 + ... + |v_n|^2)$
 
+ *norm tensor $N_l$*:

  The contraction $angle.l psi|psi angle.r$ but leave out tensor at site $l$. Because of the isometric condition, $N_l = II_(partial V)$.

#bibliography("bibliography.bib")


#show: appendices 


= Appendix section

#lorem(100)

$E(theta)=(angle.l psi(theta)|h_1|psi(theta) angle.r) / (angle.l psi(theta)|psi(theta) angle.r) + (angle.l psi(theta)|h_2|psi(theta) angle.r) / (angle.l psi(theta)|psi(theta) angle.r) $
let $A=angle.l psi(theta)|H|psi(theta) angle.r$
let $B=angle.l psi(theta)|psi(theta) angle.r$
how to calculate $(partial E)/(partial theta)$?

$theta=[theta_1, theta_2, theta_3]$