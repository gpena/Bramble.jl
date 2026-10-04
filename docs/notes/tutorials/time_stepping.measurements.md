# Measurements moved out of `docs/src/tutorials/time_stepping.md`

## Which to use: direct reuse against iterative solves

correct choice, with no iteration count and no preconditioner to manage. The
[form tutorial](@ref tutorial_form)'s solver note measured direct reuse an order of
magnitude faster per step than iterative AMG-CG or GMRES on repeated solves.

