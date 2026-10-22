# Observed orders of the polygon scheme

The scheme of `polygon_spaces` on Figure 1's polygon, solved on a fresh grid closure
at each of six refinement levels (6 to 192 jittered grid lines per axis), against a smooth
exact solution. The plot shows the ``H^1_h`` error (solid) and the maximum error (dashed)
against the largest grid step, for the Poisson problem and for a problem with variable
coefficients and a mixed term. The dotted lines have slopes 2 and 1.5.

```@raw html
<figure>
<svg viewBox="0 0 600 360" width="100%" style="max-width:600px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Log-log plot of the H1 and maximum errors against the grid step for the Poisson and mixed-term problems, with reference lines of slope 2 and 1.5.">
  <rect x="70.0" y="30.0" width="360.0" height="270.0" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <line x1="190.0" y1="30.0" x2="190.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="190.0" y="316.0" font-size="11" fill="currentColor" text-anchor="middle">10^-1</text>
  <line x1="430.0" y1="30.0" x2="430.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="430.0" y="316.0" font-size="11" fill="currentColor" text-anchor="middle">10^0</text>
  <line x1="70.0" y1="300.0" x2="430.0" y2="300.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="304.0" font-size="11" fill="currentColor" text-anchor="end">10^-5</text>
  <line x1="70.0" y1="232.5" x2="430.0" y2="232.5" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="236.5" font-size="11" fill="currentColor" text-anchor="end">10^-4</text>
  <line x1="70.0" y1="165.0" x2="430.0" y2="165.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="169.0" font-size="11" fill="currentColor" text-anchor="end">10^-3</text>
  <line x1="70.0" y1="97.5" x2="430.0" y2="97.5" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="101.5" font-size="11" fill="currentColor" text-anchor="end">10^-2</text>
  <line x1="70.0" y1="30.0" x2="430.0" y2="30.0" stroke="currentColor" stroke-opacity="0.15"/>
  <text x="64.0" y="34.0" font-size="11" fill="currentColor" text-anchor="end">10^-1</text>
  <text x="250.0" y="338.0" font-size="12" fill="currentColor" text-anchor="middle">grid step h</text>
  <text x="18" y="165.0" font-size="12" fill="currentColor" text-anchor="middle" transform="rotate(-90 18 165.0)">error</text>
  <polyline points="426.8,72.8 364.3,89.9 315.9,126.7 252.6,158.1 191.7,197.3 119.2,239.3" fill="none" stroke="#2563eb" stroke-width="2"/>
  <circle cx="426.8" cy="72.8" r="3" fill="#2563eb"/>
  <circle cx="364.3" cy="89.9" r="3" fill="#2563eb"/>
  <circle cx="315.9" cy="126.7" r="3" fill="#2563eb"/>
  <circle cx="252.6" cy="158.1" r="3" fill="#2563eb"/>
  <circle cx="191.7" cy="197.3" r="3" fill="#2563eb"/>
  <circle cx="119.2" cy="239.3" r="3" fill="#2563eb"/>
  <polyline points="426.8,120.9 364.3,130.5 315.9,168.8 252.6,201.1 191.7,242.9 119.2,283.2" fill="none" stroke="#2563eb" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="426.8" cy="120.9" r="3" fill="#2563eb"/>
  <circle cx="364.3" cy="130.5" r="3" fill="#2563eb"/>
  <circle cx="315.9" cy="168.8" r="3" fill="#2563eb"/>
  <circle cx="252.6" cy="201.1" r="3" fill="#2563eb"/>
  <circle cx="191.7" cy="242.9" r="3" fill="#2563eb"/>
  <circle cx="119.2" cy="283.2" r="3" fill="#2563eb"/>
  <polyline points="426.8,32.5 364.3,62.9 315.9,92.8 252.6,128.5 191.7,166.4 119.2,206.0" fill="none" stroke="#ea580c" stroke-width="2"/>
  <circle cx="426.8" cy="32.5" r="3" fill="#ea580c"/>
  <circle cx="364.3" cy="62.9" r="3" fill="#ea580c"/>
  <circle cx="315.9" cy="92.8" r="3" fill="#ea580c"/>
  <circle cx="252.6" cy="128.5" r="3" fill="#ea580c"/>
  <circle cx="191.7" cy="166.4" r="3" fill="#ea580c"/>
  <circle cx="119.2" cy="206.0" r="3" fill="#ea580c"/>
  <polyline points="426.8,75.0 364.3,107.7 315.9,138.5 252.6,177.2 191.7,215.5 119.2,257.6" fill="none" stroke="#ea580c" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="426.8" cy="75.0" r="3" fill="#ea580c"/>
  <circle cx="364.3" cy="107.7" r="3" fill="#ea580c"/>
  <circle cx="315.9" cy="138.5" r="3" fill="#ea580c"/>
  <circle cx="252.6" cy="177.2" r="3" fill="#ea580c"/>
  <circle cx="191.7" cy="215.5" r="3" fill="#ea580c"/>
  <circle cx="119.2" cy="257.6" r="3" fill="#ea580c"/>
  <polyline points="119.2,279.9 426.8,106.9" fill="none" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="2,3"/>
  <text x="426.8" y="106.9" font-size="11" fill="currentColor" text-anchor="end" dy="-4">slope 2</text>
  <polyline points="119.2,279.9 426.8,150.2" fill="none" stroke="#16a34a" stroke-opacity="0.6" stroke-width="1.2" stroke-dasharray="2,3"/>
  <text x="426.8" y="150.2" font-size="11" fill="#16a34a" text-anchor="end" dy="-4">slope 1.5</text>
  <line x1="444.0" y1="40.0" x2="470.0" y2="40.0" stroke="#2563eb" stroke-width="2"/>
  <text x="476.0" y="44.0" font-size="11" fill="currentColor">Poisson, H¹ₕ</text>
  <line x1="444.0" y1="56.0" x2="470.0" y2="56.0" stroke="#2563eb" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="476.0" y="60.0" font-size="11" fill="currentColor">Poisson, max</text>
  <line x1="444.0" y1="72.0" x2="470.0" y2="72.0" stroke="#ea580c" stroke-width="2"/>
  <text x="476.0" y="76.0" font-size="11" fill="currentColor">mixed, H¹ₕ</text>
  <line x1="444.0" y1="88.0" x2="470.0" y2="88.0" stroke="#ea580c" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="476.0" y="92.0" font-size="11" fill="currentColor">mixed, max</text>
</svg>
</figure>
```

The fitted slopes, in the order ``H^1_h``, maximum: Poisson 1.99 and 1.98, mixed term
2.04 and 2.14. The page is written by `poisson_polygon_convergence.jl` next to it;
the test gating these orders is `test/convergence/poisson_polygon.jl`.
