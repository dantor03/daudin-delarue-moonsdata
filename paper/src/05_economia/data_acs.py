"""ACSIncome (Folktables, Ding et al. 2021) — California 2018, binarizado.

Tarea: clasificar si un individuo gana > 50k USD/anyo a partir de su perfil
demografico/laboral.  Es la version moderna y mas grande del clasico Adult/Census
Income, sin sus problemas de filtrado y sesgo de muestreo.

Construccion (Ding et al. 2021):
  - Fuente: ACS PUMS (American Community Survey Public Use Microdata Sample)
  - Filtros: edad >= 16, ingresos > 100, horas trabajadas >= 1
  - Target binario: PINCP > 50000 (PINCP = total person's income)
  - California 2018 ~ 195k registros adultos

Para nuestros experimentos:
  - d_1=2  (solver explicito): AGEP, SCHL  (edad, anyos de educacion).
            Estas dos variables son las mas predictivas individualmente y
            permiten visualizacion 2D de la dinamica X_t.
  - d_1=4  (parametricos):     AGEP, SCHL, WKHP, SEX  (anyade horas trabajadas
            y sexo, ambas con peso conocido en la literatura de income gap).

  - N=500 muestras por experimento (uniforme en y).  Suficiente para que
    el solver explicito sea estable; al mismo tiempo manejable dado que
    pSGLD/MMD/Sinkhorn cuestan O(N^2) por epoca.

  - Estandarizacion: StandardScaler en cada feature (media 0, std 1) antes
    de la red.  Esto encaja con el regimen de "datos acotados" del paper.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

ROOT_SEED = 42


def get_acs_income(n_samples=500, year=2018, state="CA",
                    feature_set="d2", seed=ROOT_SEED, balance=True):
    """Carga ACSIncome y devuelve X (N, d), y (N,) en formatos numpy y torch.

    Parameters
    ----------
    feature_set : {"d2", "d4"}
        - "d2" -> [AGEP, SCHL]                 (para solver explicito)
        - "d4" -> [AGEP, SCHL, WKHP, SEX]      (para metodos parametricos)
    n_samples : int
        Numero de muestras totales.  Se sub-muestrea aleatoriamente con seed fijo.
    balance : bool
        Si True, sub-muestrea para tener 50/50 de la clase.  Esto es importante
        para el embedding y -> {-1, +1} del solver explicito (BCE vs MSE).

    Returns
    -------
    X      : (N, d) torch tensor   (estandarizado)
    y      : (N,)   torch tensor   (0/1 float)
    X_np   : (N, d) numpy
    y_np   : (N,)   numpy
    feat_names : list[str]
    """
    from folktables import ACSDataSource, ACSIncome

    rng = np.random.default_rng(seed)

    if feature_set == "d2":
        keep = ["AGEP", "SCHL"]
    elif feature_set == "d4":
        keep = ["AGEP", "SCHL", "WKHP", "SEX"]
    else:
        raise ValueError(f"feature_set desconocido: {feature_set}")

    src = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person")
    data = src.get_data(states=[state], download=True)
    X_full, y_full, _ = ACSIncome.df_to_numpy(data)
    feat_all = ACSIncome.features
    idx_keep = [feat_all.index(k) for k in keep]
    X_all = X_full[:, idx_keep].astype(np.float32)
    y_all = y_full.astype(np.float32)

    if balance:
        i0 = np.where(y_all == 0)[0]
        i1 = np.where(y_all == 1)[0]
        n_each = n_samples // 2
        i0 = rng.choice(i0, size=n_each, replace=False)
        i1 = rng.choice(i1, size=n_samples - n_each, replace=False)
        idx = np.concatenate([i0, i1])
        rng.shuffle(idx)
    else:
        idx = rng.choice(len(y_all), size=n_samples, replace=False)

    X_np = X_all[idx]
    y_np = y_all[idx]
    X_np = StandardScaler().fit_transform(X_np).astype(np.float32)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    X = torch.tensor(X_np, device=device)
    y = torch.tensor(y_np, device=device)
    return X, y, X_np, y_np, keep
