# Observed orders of the level-set scheme

The scheme of `levelset_spaces` on the disk `r - 1` and the five-petal star
`r - (0.6 + 0.2 sin 5ϑ)` in the box ``[-1.25, 1.25]^2``, solved on a fresh jittered grid at
each of five refinement levels (20 to 320 points per axis, one seed), against the smooth
exact solution `sin 2x cos 2y + xy`. The plot shows the ``H^1_h`` error (solid) and the
maximum error (dashed) against the largest grid step, for `-Δₕ` on the disk and on the star
and for the disk with variable coefficients, a convection term and a reaction term. The
dotted lines have slopes 2 and 3.

```@raw html
<figure>
<svg viewBox="0 0 640 360" width="100%" style="max-width:640px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Log-log plot of the H1 and maximum errors against the grid step for the disk, the star and the disk with variable coefficients, with reference lines of slope 2 and 3.">
  <rect x="70.0" y="30.0" width="360.0" height="270.0" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="70.0" y1="30.0" x2="70.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="70.0" y="316.0" font-size="11" fill="currentColor" text-anchor="middle">10^-2</text>
  <line x1="310.0" y1="30.0" x2="310.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="310.0" y="316.0" font-size="11" fill="currentColor" text-anchor="middle">10^-1</text>
  <line x1="70.0" y1="300.0" x2="430.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="304.0" font-size="11" fill="currentColor" text-anchor="end">10^-7</text>
  <line x1="70.0" y1="246.0" x2="430.0" y2="246.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="250.0" font-size="11" fill="currentColor" text-anchor="end">10^-6</text>
  <line x1="70.0" y1="192.0" x2="430.0" y2="192.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="196.0" font-size="11" fill="currentColor" text-anchor="end">10^-5</text>
  <line x1="70.0" y1="138.0" x2="430.0" y2="138.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="142.0" font-size="11" fill="currentColor" text-anchor="end">10^-4</text>
  <line x1="70.0" y1="84.0" x2="430.0" y2="84.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="88.0" font-size="11" fill="currentColor" text-anchor="end">10^-3</text>
  <line x1="70.0" y1="30.0" x2="430.0" y2="30.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="34.0" font-size="11" fill="currentColor" text-anchor="end">10^-2</text>
  <text x="250.0" y="338.0" font-size="12" fill="currentColor" text-anchor="middle">grid step h</text>
  <text x="18" y="165.0" font-size="12" fill="currentColor" text-anchor="middle" transform="rotate(-90 18 165.0)">error</text>
  <polyline points="387.8,36.6 316.4,73.4 241.4,108.0 167.0,136.7 96.8,168.9" fill="none" stroke="#2563eb" stroke-width="2"/>
  <circle cx="387.8" cy="36.6" r="3" fill="#2563eb"/>
  <circle cx="316.4" cy="73.4" r="3" fill="#2563eb"/>
  <circle cx="241.4" cy="108.0" r="3" fill="#2563eb"/>
  <circle cx="167.0" cy="136.7" r="3" fill="#2563eb"/>
  <circle cx="96.8" cy="168.9" r="3" fill="#2563eb"/>
  <polyline points="387.8,69.1 316.4,113.7 241.4,162.7 167.0,209.4 96.8,246.9" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="387.8" cy="69.1" r="3" fill="#2563eb"/>
  <circle cx="316.4" cy="113.7" r="3" fill="#2563eb"/>
  <circle cx="241.4" cy="162.7" r="3" fill="#2563eb"/>
  <circle cx="167.0" cy="209.4" r="3" fill="#2563eb"/>
  <circle cx="96.8" cy="246.9" r="3" fill="#2563eb"/>
  <polyline points="387.8,46.9 316.4,86.6 241.4,118.5 167.0,144.9 96.8,178.2" fill="none" stroke="#ea580c" stroke-width="2"/>
  <circle cx="387.8" cy="46.9" r="3" fill="#ea580c"/>
  <circle cx="316.4" cy="86.6" r="3" fill="#ea580c"/>
  <circle cx="241.4" cy="118.5" r="3" fill="#ea580c"/>
  <circle cx="167.0" cy="144.9" r="3" fill="#ea580c"/>
  <circle cx="96.8" cy="178.2" r="3" fill="#ea580c"/>
  <polyline points="387.8,75.0 316.4,118.9 241.4,161.7 167.0,210.7 96.8,257.6" fill="none" stroke="#ea580c" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="387.8" cy="75.0" r="3" fill="#ea580c"/>
  <circle cx="316.4" cy="118.9" r="3" fill="#ea580c"/>
  <circle cx="241.4" cy="161.7" r="3" fill="#ea580c"/>
  <circle cx="167.0" cy="210.7" r="3" fill="#ea580c"/>
  <circle cx="96.8" cy="257.6" r="3" fill="#ea580c"/>
  <polyline points="387.8,31.7 316.4,69.2 241.4,102.5 167.0,130.0 96.8,162.5" fill="none" stroke="#9333ea" stroke-width="2"/>
  <circle cx="387.8" cy="31.7" r="3" fill="#9333ea"/>
  <circle cx="316.4" cy="69.2" r="3" fill="#9333ea"/>
  <circle cx="241.4" cy="102.5" r="3" fill="#9333ea"/>
  <circle cx="167.0" cy="130.0" r="3" fill="#9333ea"/>
  <circle cx="96.8" cy="162.5" r="3" fill="#9333ea"/>
  <polyline points="387.8,65.1 316.4,111.8 241.4,156.3 167.0,183.1 96.8,226.4" fill="none" stroke="#9333ea" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="387.8" cy="65.1" r="3" fill="#9333ea"/>
  <circle cx="316.4" cy="111.8" r="3" fill="#9333ea"/>
  <circle cx="241.4" cy="156.3" r="3" fill="#9333ea"/>
  <circle cx="167.0" cy="183.1" r="3" fill="#9333ea"/>
  <circle cx="96.8" cy="226.4" r="3" fill="#9333ea"/>
  <polyline points="96.8,201.4 387.8,70.4" fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="2,3"/>
  <text x="387.8" y="70.4" font-size="11" fill="currentColor" text-anchor="end" dy="-4">slope 2</text>
  <polyline points="96.8,201.4 387.8,4.9" fill="none" stroke="#16a34a" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="2,3"/>
  <text x="387.8" y="4.9" font-size="11" fill="#16a34a" text-anchor="end" dy="-4">slope 3</text>
  <line x1="444.0" y1="40.0" x2="470.0" y2="40.0" stroke="#2563eb" stroke-width="2"/>
  <text x="476.0" y="44.0" font-size="11" fill="currentColor">disk, -Δₕ, H¹ₕ</text>
  <line x1="444.0" y1="56.0" x2="470.0" y2="56.0" stroke="#2563eb" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="476.0" y="60.0" font-size="11" fill="currentColor">disk, -Δₕ, max</text>
  <line x1="444.0" y1="72.0" x2="470.0" y2="72.0" stroke="#ea580c" stroke-width="2"/>
  <text x="476.0" y="76.0" font-size="11" fill="currentColor">star, -Δₕ, H¹ₕ</text>
  <line x1="444.0" y1="88.0" x2="470.0" y2="88.0" stroke="#ea580c" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="476.0" y="92.0" font-size="11" fill="currentColor">star, -Δₕ, max</text>
  <line x1="444.0" y1="104.0" x2="470.0" y2="104.0" stroke="#9333ea" stroke-width="2"/>
  <text x="476.0" y="108.0" font-size="11" fill="currentColor">disk, variable, H¹ₕ</text>
  <line x1="444.0" y1="120.0" x2="470.0" y2="120.0" stroke="#9333ea" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="476.0" y="124.0" font-size="11" fill="currentColor">disk, variable, max</text>
</svg>
</figure>
```

The fitted slopes, in the order ``H^1_h``, maximum: disk 1.99 and 2.74, star
1.95 and 2.78, disk with variable coefficients 1.96 and 2.39. The page is
written by `poisson_levelset_convergence.jl` next to it; the test gating these orders, with
three seeds pooled, is `test/convergence/poisson_levelset.jl`.
