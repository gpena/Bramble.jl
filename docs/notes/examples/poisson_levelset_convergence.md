# Observed orders of the scheme on level-set domains

The scheme of `polygon_spaces` on the mesh of a level-set domain, solved at four or five
refinement levels (20 or 40 to 320 seeds per axis), against a smooth exact solution with
the datum ``g = u + \varphi/2`` on the vertices, which equals ``u`` on the curve and not at
a vertex off it. The curves are the unit disk and the egg at 0° (the closure path: the lines
grow to close the grid, the seeds are kept), and the ellipse at 30°, the egg at 35° and a
dented egg (the orbit path: the lines are fitted anew). The plot shows the ``H^1_h`` error
(solid) and the maximum error (dashed) against the largest step between the lines that hold
a node of the domain. The dotted lines are ``0.03h^2`` and ``h^2``.

```@raw html
<figure>
<svg viewBox="0 0 600 360" width="100%" style="max-width:600px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Log-log plot of the H1 and maximum errors against the grid step for the disk, the egg at 0 degrees, the ellipse at 30 degrees, the egg at 35 degrees and the dented egg, with reference lines of slope 2.">
  <rect x="70.0" y="30.0" width="360.0" height="270.0" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="160.0" y1="30.0" x2="160.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="160.0" y="316.0" font-size="11" fill="currentColor" text-anchor="middle">10^-2</text>
  <line x1="340.0" y1="30.0" x2="340.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="340.0" y="316.0" font-size="11" fill="currentColor" text-anchor="middle">10^-1</text>
  <line x1="70.0" y1="300.0" x2="430.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="304.0" font-size="11" fill="currentColor" text-anchor="end">10^-6</text>
  <line x1="70.0" y1="232.5" x2="430.0" y2="232.5" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="236.5" font-size="11" fill="currentColor" text-anchor="end">10^-5</text>
  <line x1="70.0" y1="165.0" x2="430.0" y2="165.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="169.0" font-size="11" fill="currentColor" text-anchor="end">10^-4</text>
  <line x1="70.0" y1="97.5" x2="430.0" y2="97.5" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="101.5" font-size="11" fill="currentColor" text-anchor="end">10^-3</text>
  <line x1="70.0" y1="30.0" x2="430.0" y2="30.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="34.0" font-size="11" fill="currentColor" text-anchor="end">10^-2</text>
  <text x="250.0" y="338.0" font-size="12" fill="currentColor" text-anchor="middle">grid step h</text>
  <text x="18" y="165.0" font-size="12" fill="currentColor" text-anchor="middle" transform="rotate(-90 18 165.0)">error</text>
  <polyline points="130.7,289.8 422.7,70.7" fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="2,3"/>
  <text x="422.7" y="70.7" font-size="11" fill="currentColor" text-anchor="end" dy="-4">slope 2</text>
  <polyline points="130.7,187.0 422.7,-32.1" fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="2,3"/>
  <text x="422.7" y="-32.1" font-size="11" fill="currentColor" text-anchor="end" dy="-4">slope 2</text>
  <polyline points="389.4,44.0 341.7,91.6 313.6,117.0 272.6,154.5 229.3,189.1" fill="none" stroke="#2563eb" stroke-width="2"/>
  <circle cx="389.4" cy="44.0" r="3" fill="#2563eb"/>
  <circle cx="341.7" cy="91.6" r="3" fill="#2563eb"/>
  <circle cx="313.6" cy="117.0" r="3" fill="#2563eb"/>
  <circle cx="272.6" cy="154.5" r="3" fill="#2563eb"/>
  <circle cx="229.3" cy="189.1" r="3" fill="#2563eb"/>
  <polyline points="389.4,87.7 341.7,129.7 313.6,170.7 272.6,195.0 229.3,251.0" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="389.4" cy="87.7" r="3" fill="#2563eb"/>
  <circle cx="341.7" cy="129.7" r="3" fill="#2563eb"/>
  <circle cx="313.6" cy="170.7" r="3" fill="#2563eb"/>
  <circle cx="272.6" cy="195.0" r="3" fill="#2563eb"/>
  <circle cx="229.3" cy="251.0" r="3" fill="#2563eb"/>
  <polyline points="393.4,69.1 351.1,98.1 309.0,140.9 276.0,161.1 229.1,209.9" fill="none" stroke="#ea580c" stroke-width="2"/>
  <circle cx="393.4" cy="69.1" r="3" fill="#ea580c"/>
  <circle cx="351.1" cy="98.1" r="3" fill="#ea580c"/>
  <circle cx="309.0" cy="140.9" r="3" fill="#ea580c"/>
  <circle cx="276.0" cy="161.1" r="3" fill="#ea580c"/>
  <circle cx="229.1" cy="209.9" r="3" fill="#ea580c"/>
  <polyline points="393.4,101.9 351.1,132.9 309.0,179.3 276.0,200.2 229.1,256.6" fill="none" stroke="#ea580c" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="393.4" cy="101.9" r="3" fill="#ea580c"/>
  <circle cx="351.1" cy="132.9" r="3" fill="#ea580c"/>
  <circle cx="309.0" cy="179.3" r="3" fill="#ea580c"/>
  <circle cx="276.0" cy="200.2" r="3" fill="#ea580c"/>
  <circle cx="229.1" cy="256.6" r="3" fill="#ea580c"/>
  <polyline points="389.7,31.6 274.5,135.6 236.0,137.4 171.5,195.0" fill="none" stroke="#16a34a" stroke-width="2"/>
  <circle cx="389.7" cy="31.6" r="3" fill="#16a34a"/>
  <circle cx="274.5" cy="135.6" r="3" fill="#16a34a"/>
  <circle cx="236.0" cy="137.4" r="3" fill="#16a34a"/>
  <circle cx="171.5" cy="195.0" r="3" fill="#16a34a"/>
  <polyline points="389.7,78.1 274.5,172.9 236.0,176.1 171.5,229.0" fill="none" stroke="#16a34a" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="389.7" cy="78.1" r="3" fill="#16a34a"/>
  <circle cx="274.5" cy="172.9" r="3" fill="#16a34a"/>
  <circle cx="236.0" cy="176.1" r="3" fill="#16a34a"/>
  <circle cx="171.5" cy="229.0" r="3" fill="#16a34a"/>
  <polyline points="328.3,115.0 299.7,121.5 246.0,126.7 130.7,216.0" fill="none" stroke="#9333ea" stroke-width="2"/>
  <circle cx="328.3" cy="115.0" r="3" fill="#9333ea"/>
  <circle cx="299.7" cy="121.5" r="3" fill="#9333ea"/>
  <circle cx="246.0" cy="126.7" r="3" fill="#9333ea"/>
  <circle cx="130.7" cy="216.0" r="3" fill="#9333ea"/>
  <polyline points="328.3,155.9 299.7,150.2 246.0,149.7 130.7,259.3" fill="none" stroke="#9333ea" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="328.3" cy="155.9" r="3" fill="#9333ea"/>
  <circle cx="299.7" cy="150.2" r="3" fill="#9333ea"/>
  <circle cx="246.0" cy="149.7" r="3" fill="#9333ea"/>
  <circle cx="130.7" cy="259.3" r="3" fill="#9333ea"/>
  <polyline points="422.7,33.9 294.8,116.9 258.9,149.4 207.3,187.8" fill="none" stroke="#dc2626" stroke-width="2"/>
  <circle cx="422.7" cy="33.9" r="3" fill="#dc2626"/>
  <circle cx="294.8" cy="116.9" r="3" fill="#dc2626"/>
  <circle cx="258.9" cy="149.4" r="3" fill="#dc2626"/>
  <circle cx="207.3" cy="187.8" r="3" fill="#dc2626"/>
  <polyline points="422.7,57.0 294.8,161.2 258.9,185.0 207.3,225.6" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="422.7" cy="57.0" r="3" fill="#dc2626"/>
  <circle cx="294.8" cy="161.2" r="3" fill="#dc2626"/>
  <circle cx="258.9" cy="185.0" r="3" fill="#dc2626"/>
  <circle cx="207.3" cy="225.6" r="3" fill="#dc2626"/>
  <line x1="444.0" y1="40.0" x2="470.0" y2="40.0" stroke="#2563eb" stroke-width="2"/>
  <text x="476.0" y="44.0" font-size="11" fill="currentColor">disk</text>
  <line x1="444.0" y1="56.0" x2="470.0" y2="56.0" stroke="#ea580c" stroke-width="2"/>
  <text x="476.0" y="60.0" font-size="11" fill="currentColor">egg 0°</text>
  <line x1="444.0" y1="72.0" x2="470.0" y2="72.0" stroke="#16a34a" stroke-width="2"/>
  <text x="476.0" y="76.0" font-size="11" fill="currentColor">ellipse 30°</text>
  <line x1="444.0" y1="88.0" x2="470.0" y2="88.0" stroke="#9333ea" stroke-width="2"/>
  <text x="476.0" y="92.0" font-size="11" fill="currentColor">egg 35°</text>
  <line x1="444.0" y1="104.0" x2="470.0" y2="104.0" stroke="#dc2626" stroke-width="2"/>
  <text x="476.0" y="108.0" font-size="11" fill="currentColor">dented egg</text>
</svg>
</figure>
```

The fitted slopes, in the order ``H^1_h``, maximum:
disk 2.42 and 2.68; egg 0° 2.28 and 2.49; ellipse 30° 1.95 and 1.81; egg 35° 1.39 and 1.48; dented egg 1.9 and 2.09.
The closure path is second order. On the orbit path the vertices lie within ``h^2/4`` of the curve, so the error stays
below a multiple of ``h^2`` at every level, but the number of lines follows a cost rule and
``h`` does not fall by a fixed factor from one level to the next: the fitted slope depends
on the curve (the egg at 35° has the largest jumps), and the gate asserts the bound on
``e/h^2`` and a floor on the slope of the ellipse. The page is written by
`poisson_levelset_convergence.jl` next to it; the test gating these orders
is `test/convergence/levelset.jl`.
