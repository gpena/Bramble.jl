```@meta
CurrentModule = Bramble
```

# Bramble.jl

This documentation is for `Bramble.jl`, a Julia library implementing discretization methods to solve partial differential equations using finite differences on nonuniform grids.

```@raw html
<!-- Real Markdown links below, not raw HTML <a> tags: Documenter rewrites a `.md`-target
     link to the right URL for whichever build mode is active (pretty "page/" on the
     deployed site, plain "page.html" locally) automatically. A raw HTML href would have
     to guess that itself, and had been guessing wrong for the deployed site. These
     <div>s only wrap the styling; the interspersed plain Markdown still goes through
     Documenter's normal link resolution. -->
<div class="bramble-home-cards">
<div class="bramble-card">
```

**[Getting started](getting_started.md)**

A Poisson problem end to end in twenty lines: domain, mesh, grid space, form, solve.

```@raw html
</div>
<div class="bramble-card">
```

**[Discrete foundations](tutorials/geometry.md)**

Domains, meshes and their metric, grid spaces, and the discrete calculus the operators are built from.

```@raw html
</div>
<div class="bramble-card">
```

**[Gallery of PDEs](examples/heat_equation.md)**

Linear and nonlinear Poisson, convection–diffusion, reaction–diffusion, elasticity, a heat equation.

```@raw html
</div>
</div>
```

For more information on the types of discretizations encompassed by `Bramble.jl`, please consult the papers
* J. A. Ferreira and R. D. Grigorieff, [On the supraconvergence of elliptic finite difference schemes](https://doi.org/10.1016/S0168-9274(98)00048-8), Applied Numerical Mathematics 28 (1998), pp. 275-292

* S. Barbeiro, J. A. Ferreira and R. D. Grigorieff, [Supraconvergence of a finite difference scheme for solutions in ``H^s(0,L)``](https://doi.org/10.1093/imanum/dri018), IMA Journal of Numerical Analysis 25.4 (2005), pp. 797–811

* J. A. Ferreira and R. D. Grigorieff, [Supraconvergence and Supercloseness of a Scheme for Elliptic Equations on Nonuniform Grids](https://doi.org/10.1080/01630560600796485), Numerical Functional Analysis and Optimization 27.5-6 (2006), pp. 539–564
