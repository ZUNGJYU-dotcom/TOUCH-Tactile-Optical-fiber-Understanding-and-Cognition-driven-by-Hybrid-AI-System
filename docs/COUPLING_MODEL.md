# Wavelength-Shift Surface Coupling Model

The wavelength-shift edition does not reuse the optical-intensity edition's
directional downstream attenuation cascade.

Its simulated 3x3 surface uses only two qualitative mechanisms:

1. A fingertip contact patch can cover several nearby sensing locations.
2. The shared elastomer can mechanically transfer strain to neighboring FBGs.

The display matrix is fixed as:

```text
P11  P21  P31
P12  P22  P32
P13  P23  P33
```

Each simulated channel receives a signed peak displacement and a normalized
`abs(delta lambda)` visualization value. The coefficients are visual proxies,
not a measured transfer matrix. Real local-response inversion remains disabled
until controlled 3x3 experiments identify the coupling matrix and temperature
effects.
