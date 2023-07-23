global Y counters
global X contrain
global D class3
set seed 42
ddml init partial, kfolds(5)
ddml E[Y|X]: pystacked $D $X, type(reg) method(rf)
ddml E[D|X]: pystacked $Y $X, type(reg) method(rf)
ddml desc
ddml crossfit
ddml estimate, robust