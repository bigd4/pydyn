class Constants:
    """Physical constants used in simulations."""

    pi = 3.141592653589793  # The value of Pi
    mu_B = 5.7883817555e-5  # The Bohr Magneton [eV/T]
    mu_0 = 2.0133545 * 1e-25  # The vacuum permeability [T^2 m^3 / eV]
    gamma = 0.1760859644  # Gyromagnetic ratio of electron [rad/(ps*T)]

    kB = 8.617333262145e-5  # Boltzmann constant [eV/K]
    hplanck = 4.13566733e-3  # [eV*ps]
    hbar = hplanck / (2 * pi)  # [eV*ps]
    amu = 1.6605390689e-27  # [kg]
    mv2_to_e = 1.0364269e-4  # [amu*A^2/ps^2] -> [eV]
    e_to_mv2 = 1 / 1.0364269e-4  # [eV] -> [amu*A^2/ps^2]
    e_to_pV = 1.6021766208e6  # [eV -> bar*Angstrom^3]
    pV_to_e = 1 / 1.6021766208e6  # [bar*Angstrom^3] -> [eV]
