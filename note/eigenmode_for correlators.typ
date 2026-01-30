== eigen decomposition of the transfer matrix:

$E = |r_1 angle.r angle.l l_1| + sum_(alpha >=2) lambda_alpha|r_alpha angle.r angle.l l_alpha| $

$E_O:$ insert an operator $O$ between the two physical legs of the transfer matrix.

== for connected correlation:

$angle.l O_i O_(i+r)angle.r_c = angle.l O_i O_(i+r)angle.r- angle.l O_i angle.r angle.l O_i angle.r= angle.l l_1|E_O E^(r-1) E_O|r_1 angle.r - angle.l l_1|E_O|r_1 angle.r angle.l l_1|E_O|r_1 angle.r = sum_(alpha >=2)  angle.l l_1|E_O|r_alpha angle.r angle.l l_alpha|E_O|r_1 angle.r dot lambda_alpha^(r-1) $

so the dominant term is the largest $lambda_alpha$ with nonzero coefficient $c_alpha = angle.l l_1|E_O|r_alpha angle.r angle.l l_alpha|E_O|r_1 angle.r$

$angle.l O_i O_(i+r)angle.r_c ~ c_alpha lambda_alpha^(r-1)$ decay exponentially with $r$.


== fitting
if $lambda$ is complex, say $lambda = |lambda| e^(i theta)$

$angle.l O_i O_(i+r)angle.r_c ~ |lambda|^(r-1) cos(theta r+phi) ~ A^(r-1) cos(k r +b)$

fit and check if $|lambda|==A, k==theta$

== DOTO
1. function: get $E_O$ by contracting the indices (similar to getting transfer matrix)

2. function: calculate the $c_alpha$

3. change the fitting of acf as the fitting above, also draw the real decay to refer.