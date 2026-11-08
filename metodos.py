from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

TOL = 1e-9
NOMBRE_FICTICIO = "Ficticio"

Celda = Tuple[int, int]


# =============================================================================
# 1. MODELO Y VALIDACIÓN
# =============================================================================
class Problema:
    """Datos de un problema de transporte (minimización)."""

    def __init__(self, costos, oferta, demanda,
                 origenes: Optional[Sequence[str]] = None,
                 destinos: Optional[Sequence[str]] = None):
        self.costos = np.asarray(costos, dtype=float)
        self.oferta = np.asarray(oferta, dtype=float)
        self.demanda = np.asarray(demanda, dtype=float)
        m, n = self.costos.shape
        self.origenes = list(origenes) if origenes else [f"Origen {i + 1}" for i in range(m)]
        self.destinos = list(destinos) if destinos else [f"Destino {j + 1}" for j in range(n)]

    @property
    def m(self) -> int:
        return self.costos.shape[0]

    @property
    def n(self) -> int:
        return self.costos.shape[1]


def validar_datos(costos, oferta, demanda) -> None:
    """Lanza ValueError con un mensaje claro si los datos no son utilizables."""
    c = np.asarray(costos, dtype=float)
    s = np.asarray(oferta, dtype=float)
    d = np.asarray(demanda, dtype=float)
    if c.ndim != 2 or c.shape != (len(s), len(d)):
        raise ValueError("Las dimensiones de costos, oferta y demanda no coinciden.")
    if np.isnan(c).any():
        raise ValueError("Hay celdas de COSTO vacías o no numéricas.")
    if np.isnan(s).any() or np.isnan(d).any():
        raise ValueError("Hay valores de OFERTA o DEMANDA vacíos o no numéricos.")
    if (s < 0).any() or (d < 0).any():
        raise ValueError("La oferta y la demanda no pueden ser negativas.")
    if s.sum() <= 0 or d.sum() <= 0:
        raise ValueError("La oferta total y la demanda total deben ser mayores que 0.")


# =============================================================================
# 2. BALANCEO (NODO FICTICIO)
# =============================================================================
def balancear(p: Problema, costo_ficticio: float = 0.0) -> Tuple[Problema, dict]:
    """
    Devuelve (problema_balanceado, info).
    info = {oferta_total, demanda_total, balanceado, ficticio: None|'destino'|'origen',
            cantidad}
    """
    S, D = float(p.oferta.sum()), float(p.demanda.sum())
    info = {"oferta_total": S, "demanda_total": D, "balanceado": abs(S - D) < TOL,
            "ficticio": None, "cantidad": 0.0}
    if info["balanceado"]:
        return p, info
    if S > D:
        info.update(ficticio="destino", cantidad=S - D)
        return Problema(np.hstack([p.costos, np.full((p.m, 1), costo_ficticio)]),
                        p.oferta, np.append(p.demanda, S - D),
                        p.origenes, p.destinos + [NOMBRE_FICTICIO]), info
    info.update(ficticio="origen", cantidad=D - S)
    return Problema(np.vstack([p.costos, np.full((1, p.n), costo_ficticio)]),
                    np.append(p.oferta, D - S), p.demanda,
                    p.origenes + [NOMBRE_FICTICIO], p.destinos), info


# =============================================================================
# 3. MÉTODOS DE SOLUCIÓN INICIAL   ->  (x, basicas, pasos)
# =============================================================================
def _asignar(x, basicas, sup, dem, i, j, pasos, extra=""):
    q = min(sup[i], dem[j])
    if q > TOL:
        x[i, j] = q
        basicas.add((i, j))
    sup[i] -= q
    dem[j] -= q
    pasos.append(f"X{i + 1}{j + 1} = {q:.10g}{extra}")


def _tachar(sup, dem, i, j, filas: Set[int], cols: Set[int]):
    """Tacha la fila o columna agotada. Si se agotan ambas, tacha sólo una."""
    fila_agotada, col_agotada = sup[i] <= TOL, dem[j] <= TOL
    if fila_agotada and col_agotada:
        if len(filas) > 1:
            filas.discard(i)
        else:
            cols.discard(j)
    elif fila_agotada:
        filas.discard(i)
    else:
        cols.discard(j)


def esquina_noroeste(p: Problema):
    x = np.zeros((p.m, p.n))
    sup, dem = p.oferta.copy(), p.demanda.copy()
    filas, cols, basicas, pasos = set(range(p.m)), set(range(p.n)), set(), []
    while filas and cols:
        i, j = min(filas), min(cols)
        _asignar(x, basicas, sup, dem, i, j, pasos, "  (esquina noroeste de lo que queda)")
        _tachar(sup, dem, i, j, filas, cols)
    return x, basicas, pasos


def costo_minimo(p: Problema):
    x = np.zeros((p.m, p.n))
    sup, dem = p.oferta.copy(), p.demanda.copy()
    filas, cols, basicas, pasos = set(range(p.m)), set(range(p.n)), set(), []
    while filas and cols:
        _, _, i, j = min((p.costos[i, j], -min(sup[i], dem[j]), i, j)
                         for i in filas for j in cols)
        _asignar(x, basicas, sup, dem, i, j, pasos,
                 f"  (menor costo disponible: C{i + 1}{j + 1} = {p.costos[i, j]:.10g})")
        _tachar(sup, dem, i, j, filas, cols)
    return x, basicas, pasos


def vogel(p: Problema):
    x = np.zeros((p.m, p.n))
    sup, dem = p.oferta.copy(), p.demanda.copy()
    filas, cols, basicas, pasos = set(range(p.m)), set(range(p.n)), set(), []

    def penal(vals):
        v = sorted(vals)
        return v[1] - v[0] if len(v) > 1 else v[0]

    ronda = 0
    while filas and cols:
        ronda += 1
        pf = {i: penal([p.costos[i, j] for j in cols]) for i in filas}
        pc = {j: penal([p.costos[i, j] for i in filas]) for j in cols}
        mf = max(pf.items(), key=lambda t: (t[1], -t[0]))
        mc = max(pc.items(), key=lambda t: (t[1], -t[0]))
        pasos.append(f"[Ronda {ronda}] Penalización filas: "
                     + ", ".join(f"F{i + 1}={v:.10g}" for i, v in sorted(pf.items()))
                     + " | columnas: "
                     + ", ".join(f"C{j + 1}={v:.10g}" for j, v in sorted(pc.items())))
        if mf[1] >= mc[1]:
            i = mf[0]
            j = min(cols, key=lambda jj: (p.costos[i, jj], -min(sup[i], dem[jj]), jj))
            donde = f"fila {i + 1}, penalización {mf[1]:.10g}"
        else:
            j = mc[0]
            i = min(filas, key=lambda ii: (p.costos[ii, j], -min(sup[ii], dem[j]), ii))
            donde = f"columna {j + 1}, penalización {mc[1]:.10g}"
        _asignar(x, basicas, sup, dem, i, j, pasos, f"  (mayor penalización: {donde})")
        _tachar(sup, dem, i, j, filas, cols)
    return x, basicas, pasos


METODOS_INICIALES: Dict[str, Callable] = {
    "Esquina Noroeste": esquina_noroeste,
    "Costo Mínimo": costo_minimo,
    "Aproximación de Vogel": vogel,
}


# =============================================================================
# 4. COSTO, DEGENERACIÓN, MULTIPLICADORES, CICLOS
# =============================================================================
def costo_total(p: Problema, x: np.ndarray) -> float:
    """C = Σ Cij · Xij"""
    return float(np.sum(p.costos * x))


def detalle_costo(p: Problema, x: np.ndarray) -> str:
    """Texto tipo '40(1000) + 35(1000) = 75000'."""
    t = " + ".join(f"{p.costos[i, j]:.10g}({x[i, j]:.10g})"
                   for i in range(p.m) for j in range(p.n) if x[i, j] > TOL)
    return f"{t} = {costo_total(p, x):.10g}"


def completar_base(p: Problema, basicas: Set[Celda]) -> List[Celda]:
    """Agrega comodines ε (casillas básicas con 0) hasta tener m+n−1 básicas,
    sin formar ciclos. Modifica `basicas` y devuelve las celdas agregadas."""
    m, n = p.m, p.n
    padre = list(range(m + n))

    def find(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    for (i, j) in basicas:
        padre[find(i)] = find(m + j)
    agregadas: List[Celda] = []
    if len(basicas) >= m + n - 1:
        return agregadas
    for _, i, j in sorted((p.costos[i, j], i, j) for i in range(m) for j in range(n)
                          if (i, j) not in basicas):
        if find(i) != find(m + j):
            padre[find(i)] = find(m + j)
            basicas.add((i, j))
            agregadas.append((i, j))
            if len(basicas) == m + n - 1:
                break
    return agregadas


def calcular_multiplicadores(p: Problema, basicas: Set[Celda]):
    """Resuelve Ui + Vj = Cij (básicas) con U1 = 0. Devuelve (u, v, ecuaciones)."""
    u: List[Optional[float]] = [None] * p.m
    v: List[Optional[float]] = [None] * p.n
    u[0] = 0.0
    ecs = [f"X{i + 1}{j + 1}:  U{i + 1} + V{j + 1} = {p.costos[i, j]:.10g}"
           for (i, j) in sorted(basicas)]
    pend = set(basicas)
    while pend:
        avance = False
        for (i, j) in sorted(pend):
            if u[i] is not None and v[j] is None:
                v[j] = p.costos[i, j] - u[i]
            elif v[j] is not None and u[i] is None:
                u[i] = p.costos[i, j] - v[j]
            elif u[i] is None and v[j] is None:
                continue
            pend.discard((i, j))
            avance = True
        if not avance:
            raise ValueError("Base no conexa: faltan casillas básicas.")
    return np.array(u, float), np.array(v, float), ecs


def encontrar_ciclo(basicas: Set[Celda], entrante: Celda) -> Optional[List[Celda]]:
    """Ciclo cerrado (movimientos H/V alternados) sobre casillas básicas,
    empezando en `entrante` con signo +."""
    lista = sorted(basicas)

    def dfs(camino, horizontal):
        i, j = camino[-1]
        for c in lista:
            if c in camino:
                continue
            if (horizontal and c[0] == i) or (not horizontal and c[1] == j):
                nuevo = camino + [c]
                s = camino[0]
                cierra = (s[1] == c[1]) if horizontal else (s[0] == c[0])
                if len(nuevo) >= 4 and len(nuevo) % 2 == 0 and cierra:
                    return nuevo
                r = dfs(nuevo, not horizontal)
                if r:
                    return r
        return None

    return dfs([entrante], True) or dfs([entrante], False)


def reasignar(x: np.ndarray, basicas: Set[Celda], ciclo: List[Celda]):
    """Mueve θ = mín(x en casillas −) por el ciclo. Devuelve (θ, celda_que_sale)."""
    menos = ciclo[1::2]
    theta = min(x[c] for c in menos)
    sale = next(c for c in menos if abs(x[c] - theta) <= TOL)
    for k, c in enumerate(ciclo):
        x[c] += theta if k % 2 == 0 else -theta
    x[np.abs(x) < TOL] = 0.0
    basicas.add(ciclo[0])
    basicas.discard(sale)
    return theta, sale


def texto_ciclo(ciclo: List[Celda]) -> str:
    return " → ".join(f"X{i + 1}{j + 1}({'+' if k % 2 == 0 else '−'})"
                      for k, (i, j) in enumerate(ciclo))


# =============================================================================
# 5. PRUEBAS DE OPTIMALIDAD  (MODI y CRUCE DEL ARROYO)
#    Devuelven un dict con la información de la iteración.
# =============================================================================
def evaluar_modi(p: Problema, x: np.ndarray, basicas: Set[Celda]) -> dict:
    u, v, ecs = calcular_multiplicadores(p, basicas)
    ind = u[:, None] + v[None, :] - p.costos          # Iij = Ui + Vj − Cij
    detalle = []
    for i in range(p.m):
        for j in range(p.n):
            if (i, j) in basicas:
                ind[i, j] = np.nan
            else:
                detalle.append(f"X{i + 1}{j + 1}:  {u[i]:.10g} + {v[j]:.10g} − "
                               f"{p.costos[i, j]:.10g} = {ind[i, j]:.10g}")
    res = {"metodo": "MODI", "u": u, "v": v, "ecuaciones": ecs, "detalle": detalle,
           "matriz": ind, "matriz_nombre": "Indicadores Iij = Ui + Vj − Cij",
           "entrante": None, "ciclo": None, "alternativas": False}
    validos = ind[~np.isnan(ind)]
    if validos.size == 0 or not validos.max() > TOL:
        res["alternativas"] = bool(validos.size and np.any(np.abs(validos) <= TOL))
        return res
    k = int(np.nanargmax(ind))
    res["entrante"] = (k // p.n, k % p.n)
    res["valor_entrante"] = float(np.nanmax(ind))
    res["ciclo"] = encontrar_ciclo(basicas, res["entrante"])
    return res


def evaluar_stepping_stone(p: Problema, x: np.ndarray, basicas: Set[Celda]) -> dict:
    delta = np.full((p.m, p.n), np.nan)
    ciclos: Dict[Celda, List[Celda]] = {}
    detalle = []
    for i in range(p.m):
        for j in range(p.n):
            if (i, j) in basicas:
                continue
            ciclo = encontrar_ciclo(basicas, (i, j))
            if ciclo is None:
                raise RuntimeError(f"No se encontró ciclo para X{i + 1}{j + 1}.")
            ciclos[(i, j)] = ciclo
            delta[i, j] = sum((1 if k % 2 == 0 else -1) * p.costos[c]
                              for k, c in enumerate(ciclo))
            expr = " ".join(f"{'+' if k % 2 == 0 else '−'}{p.costos[c]:.10g}"
                            for k, c in enumerate(ciclo))
            detalle.append(f"X{i + 1}{j + 1}:  {texto_ciclo(ciclo)}   →   "
                           f"Δ = {expr} = {delta[i, j]:.10g}")
    res = {"metodo": "Cruce del Arroyo", "detalle": detalle, "matriz": delta,
           "matriz_nombre": "Costos marginales Δij (Σ+C − Σ−C del ciclo)",
           "entrante": None, "ciclo": None, "alternativas": False}
    validos = delta[~np.isnan(delta)]
    if validos.size == 0 or not validos.min() < -TOL:
        res["alternativas"] = bool(validos.size and np.any(np.abs(validos) <= TOL))
        return res
    k = int(np.nanargmin(delta))
    res["entrante"] = (k // p.n, k % p.n)
    res["valor_entrante"] = float(np.nanmin(delta))
    res["ciclo"] = ciclos[res["entrante"]]
    return res


METODOS_OPTIMIZACION: Dict[str, Optional[Callable]] = {
    "MODI": evaluar_modi,
    "Cruce del Arroyo": evaluar_stepping_stone,
    "Solo inicial": None,
}


def optimizar(p: Problema, x: np.ndarray, basicas: Set[Celda],
              evaluador: Callable, max_iter: int = 200) -> dict:
    """Itera hasta alcanzar el óptimo. Modifica x y basicas in situ."""
    iteraciones: List[dict] = []
    alternativas = False
    for k in range(1, max_iter + 1):
        it = evaluador(p, x, basicas)
        it["numero"] = k
        it["x"] = x.copy()                     # asignación al INICIO de la iteración
        it["basicas"] = set(basicas)
        it["costo"] = costo_total(p, x)
        if it["entrante"] is None:
            it["es_optima"] = True
            alternativas = it["alternativas"]
            iteraciones.append(it)
            return {"iteraciones": iteraciones, "alternativas": alternativas}
        it["es_optima"] = False
        it["theta"], it["sale"] = reasignar(x, basicas, it["ciclo"])
        it["costo_siguiente"] = costo_total(p, x)
        iteraciones.append(it)
    raise RuntimeError("Se alcanzó el máximo de iteraciones sin converger.")


# =============================================================================
# 6. FUNCIÓN PRINCIPAL PARA LA INTERFAZ
# =============================================================================
def resolver_transporte(costos, oferta, demanda,
                        origenes: Optional[Sequence[str]] = None,
                        destinos: Optional[Sequence[str]] = None,
                        metodo_inicial: str = "Costo Mínimo",
                        metodo_optimizacion: str = "MODI",
                        costo_ficticio: float = 0.0) -> dict:
    """
    Resuelve un problema completo y devuelve un diccionario con:
      problema        Problema balanceado (con nodo ficticio si aplica)
      info_balance    dict del balanceo
      pasos_iniciales lista de textos del método inicial
      x_inicial, basicas_inicial, comodines, costo_inicial
      optimizacion    None | {'iteraciones': [...], 'alternativas': bool}
      x_final, basicas_final, costo_final
      plan            lista de dicts {origen, destino, unidades, costo_unit, subtotal}
      historial_costos
    """
    if metodo_inicial not in METODOS_INICIALES:
        raise ValueError(f"Método inicial desconocido: {metodo_inicial}")
    if metodo_optimizacion not in METODOS_OPTIMIZACION:
        raise ValueError(f"Método de optimización desconocido: {metodo_optimizacion}")

    validar_datos(costos, oferta, demanda)
    p0 = Problema(costos, oferta, demanda, origenes, destinos)
    p, info = balancear(p0, costo_ficticio)

    x, basicas, pasos = METODOS_INICIALES[metodo_inicial](p)
    comodines = completar_base(p, basicas)
    costo_ini = costo_total(p, x)

    res = {
        "problema": p, "info_balance": info, "metodo_inicial": metodo_inicial,
        "pasos_iniciales": pasos, "x_inicial": x.copy(),
        "basicas_inicial": set(basicas), "comodines": comodines,
        "costo_inicial": costo_ini, "optimizacion": None,
        "metodo_optimizacion": metodo_optimizacion,
    }

    evaluador = METODOS_OPTIMIZACION[metodo_optimizacion]
    historial = [costo_ini]
    if evaluador is not None:
        opt = optimizar(p, x, basicas, evaluador)
        res["optimizacion"] = opt
        for it in opt["iteraciones"]:
            if not it["es_optima"]:
                historial.append(it["costo_siguiente"])

    res.update(x_final=x, basicas_final=set(basicas), costo_final=costo_total(p, x),
               historial_costos=historial)
    res["plan"] = [{"origen": p.origenes[i], "destino": p.destinos[j],
                    "unidades": float(x[i, j]), "costo_unit": float(p.costos[i, j]),
                    "subtotal": float(x[i, j] * p.costos[i, j])}
                   for i in range(p.m) for j in range(p.n) if x[i, j] > TOL]
    return res