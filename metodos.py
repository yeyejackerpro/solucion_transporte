# -*- coding: utf-8 -*-
"""
metodos.py — Lógica matemática: Transporte (métodos 1-5) y Métodos Numéricos (6-10)
===================================================================================
  Transporte (Clase 5.pdf):  1 Costo Mínimo · 2 Esquina Noroeste · 3 Vogel ·
                             4 Cruce del Arroyo · 5 MODI
  Métodos numéricos:         6 Sección Dorada · 7 Fibonacci ·
                             8 Interpolación Cuadrática · 9 Newton 1D ·
                             10 Máxima Inclinación (gradiente N-dimensional)

Este módulo NO imprime ni usa Streamlit: recibe datos y devuelve estructuras
(dict / listas de dict / arreglos NumPy) que `app.py` se encarga de mostrar.

Algoritmo de transporte
-----------------------
1. Balanceo: si oferta ≠ demanda se agrega un nodo ficticio (costos = 0).
2. Solución inicial: Esquina Noroeste, Costo Mínimo o Vogel.
3. Degeneración: se completan m + n − 1 casillas básicas con comodines ε.
4. Optimalidad:
     * MODI:             Ui + Vj = Cij (básicas),  Iij = Ui + Vj − Cij ≤ 0 ⇒ óptimo.
     * Cruce del Arroyo: Δij = Σ(+C) − Σ(−C) del ciclo de cada no básica ≥ 0 ⇒ óptimo.
5. Reasignación por el ciclo (+, −, +, −, …) con θ = mín(casillas −).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

TOL = 1e-9
NOMBRE_FICTICIO = "Ficticio"

Celda = Tuple[int, int]


# =============================================================================
# PARTE A — TRANSPORTE
# =============================================================================
# 1. MODELO Y VALIDACIÓN -------------------------------------------------------
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


# 2. BALANCEO (NODO FICTICIO) --------------------------------------------------
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


# 3. MÉTODOS DE SOLUCIÓN INICIAL   ->  (x, basicas, pasos) ---------------------
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


# 4. COSTO, DEGENERACIÓN, MULTIPLICADORES, CICLOS ------------------------------
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


# 5. PRUEBAS DE OPTIMALIDAD (MODI y CRUCE DEL ARROYO) --------------------------
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
    for k in range(1, max_iter + 1):
        it = evaluador(p, x, basicas)
        it["numero"] = k
        it["x"] = x.copy()                     # asignación al INICIO de la iteración
        it["basicas"] = set(basicas)
        it["costo"] = costo_total(p, x)
        if it["entrante"] is None:
            it["es_optima"] = True
            iteraciones.append(it)
            return {"iteraciones": iteraciones, "alternativas": it["alternativas"]}
        it["es_optima"] = False
        it["theta"], it["sale"] = reasignar(x, basicas, it["ciclo"])
        it["costo_siguiente"] = costo_total(p, x)
        iteraciones.append(it)
    raise RuntimeError("Se alcanzó el máximo de iteraciones sin converger.")


# 6. FUNCIÓN PRINCIPAL DE TRANSPORTE -------------------------------------------
def resolver_transporte(costos, oferta, demanda,
                        origenes: Optional[Sequence[str]] = None,
                        destinos: Optional[Sequence[str]] = None,
                        metodo_inicial: str = "Costo Mínimo",
                        metodo_optimizacion: str = "MODI",
                        costo_ficticio: float = 0.0) -> dict:
    """
    Resuelve un problema completo y devuelve un diccionario con:
      problema, info_balance, pasos_iniciales, x_inicial, basicas_inicial,
      comodines, costo_inicial, optimizacion (None | {'iteraciones', 'alternativas'}),
      x_final, basicas_final, costo_final, plan, historial_costos
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


# =============================================================================
# CATÁLOGO DE LOS 10 MÉTODOS (lo usa el menú lateral de app.py)
# =============================================================================
CAT_TRANSPORTE = "Transporte"
CAT_NUMERICOS = "Métodos Numéricos"

CATALOGO: Dict[str, dict] = {
    "costo_minimo":     {"n": 1,  "cat": CAT_TRANSPORTE, "nombre": "Método del Costo Mínimo"},
    "esquina_noroeste": {"n": 2,  "cat": CAT_TRANSPORTE, "nombre": "Método de la Esquina Noroeste"},
    "vogel":            {"n": 3,  "cat": CAT_TRANSPORTE, "nombre": "Aproximación de Vogel (VAM)"},
    "cruce_arroyo":     {"n": 4,  "cat": CAT_TRANSPORTE, "nombre": "Cruce del Arroyo (Stepping Stone)"},
    "modi":             {"n": 5,  "cat": CAT_TRANSPORTE, "nombre": "Algoritmo MODI"},
    "dorada":           {"n": 6,  "cat": CAT_NUMERICOS, "nombre": "Búsqueda de la Sección Dorada"},
    "fibonacci":        {"n": 7,  "cat": CAT_NUMERICOS, "nombre": "Búsqueda de Fibonacci"},
    "cuadratica":       {"n": 8,  "cat": CAT_NUMERICOS, "nombre": "Interpolación Cuadrática"},
    "newton":           {"n": 9,  "cat": CAT_NUMERICOS, "nombre": "Método de Newton (Unidimensional)"},
    "gradiente":        {"n": 10, "cat": CAT_NUMERICOS, "nombre": "Máxima Inclinación (Gradiente N-dim)"},
}
ORDEN: List[str] = sorted(CATALOGO, key=lambda k: CATALOGO[k]["n"])

# Métodos de transporte -> (solución inicial, optimización). None = la elige el usuario.
CONFIG_TRANSPORTE: Dict[str, Tuple[Optional[str], str]] = {
    "costo_minimo": ("Costo Mínimo", "Solo inicial"),
    "esquina_noroeste": ("Esquina Noroeste", "Solo inicial"),
    "vogel": ("Aproximación de Vogel", "Solo inicial"),
    "cruce_arroyo": (None, "Cruce del Arroyo"),
    "modi": (None, "MODI"),
}


# =============================================================================
# PARTE B — MÉTODOS NUMÉRICOS
# Utilidades: lectura segura de funciones con sympy
# =============================================================================
_TRANSFORMACIONES = standard_transformations + (implicit_multiplication_application,
                                                convert_xor)
_PERMITIDO = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]+$")
_PROHIBIDO = re.compile(r"__|[A-Za-z_]\s*\.\s*[A-Za-z_]|\b(?:import|eval|exec|open|compile|"
                        r"lambda|getattr|setattr|globals|locals|input|breakpoint)\b")
_FUNCIONES = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    "pi": sp.pi, "e": sp.E,
}


def parsear_expresion(texto: str, variables: Sequence[str],
                      alias: Optional[Dict[str, str]] = None):
    """
    Convierte texto (p. ej. '2x^2 - 3*sin(x)') en una expresión sympy.
    Acepta ^ como potencia y multiplicación implícita ('2x'). Sólo permite las
    variables dadas (y alias opcionales, p. ej. {'x': 'x1'}).
    Devuelve (expr, [símbolos en el orden de `variables`]).
    """
    t = (texto or "").strip()
    if not t:
        raise ValueError("Escribe la función f.")
    if len(t) > 300 or not _PERMITIDO.match(t) or _PROHIBIDO.search(t):
        raise ValueError("La función contiene caracteres o palabras no permitidos.")
    simbolos = {v: sp.Symbol(v, real=True) for v in variables}
    local = dict(_FUNCIONES)
    local.update(simbolos)
    for a, v in (alias or {}).items():
        local[a] = simbolos[v]
    try:
        expr = parse_expr(t, local_dict=local, transformations=_TRANSFORMACIONES)
    except Exception as err:                       # noqa: BLE001
        raise ValueError(f"No se pudo interpretar la función: {err}") from None
    if not isinstance(expr, sp.Expr):
        raise ValueError("La entrada no es una expresión matemática.")
    sobrantes = expr.free_symbols - set(simbolos.values())
    if sobrantes:
        raise ValueError("Variables no reconocidas: " + ", ".join(sorted(map(str, sobrantes)))
                         + f". Usa: {', '.join(variables)}.")
    return expr, [simbolos[v] for v in variables]


class FuncionEscalar:
    """Envuelve una expresión sympy como función numérica f(*args) -> float."""

    def __init__(self, expr, simbolos):
        self.expr, self.simbolos = expr, list(simbolos)
        self._f = sp.lambdify(self.simbolos, expr, "numpy")
        self.evaluaciones = 0

    def __call__(self, *args) -> float:
        self.evaluaciones += 1
        with np.errstate(all="ignore"):
            v = complex(self._f(*args))
        if abs(v.imag) > 1e-12 or not math.isfinite(v.real):
            raise ValueError("f no está definida (o no es real) en el punto "
                             + ", ".join(f"{a:.6g}" for a in args) + ".")
        return float(v.real)

    def muestrear(self, xs: np.ndarray) -> np.ndarray:
        """Evalúa en un arreglo (NaN donde f no esté definida)."""
        with np.errstate(all="ignore"):
            v = np.broadcast_to(np.asarray(self._f(xs), dtype=complex), xs.shape)
        ys = np.where(np.abs(v.imag) < 1e-12, v.real, np.nan).astype(float)
        ys[~np.isfinite(ys)] = np.nan
        return ys


def _f1d(texto: str):
    expr, sims = parsear_expresion(texto, ["x"])
    return expr, sims[0], FuncionEscalar(expr, sims)


def _signo(objetivo: str) -> int:
    if objetivo not in ("min", "max"):
        raise ValueError("El objetivo debe ser 'min' o 'max'.")
    return 1 if objetivo == "min" else -1


def _grafica_funcion(F: FuncionEscalar, lo: float, hi: float, n: int = 300) -> dict:
    margen = 0.1 * (hi - lo) if hi > lo else 1.0
    xs = np.linspace(lo - margen, hi + margen, n)
    return {"titulo": "Gráfica de f(x)", "x_label": "x", "x": xs.tolist(),
            "series": {"f(x)": F.muestrear(xs).tolist()}}


def _advertir_multimodal(F: FuncionEscalar, a: float, b: float, s: int) -> List[str]:
    """Aviso si en [a, b] hay más de un óptimo local (el método requiere unimodalidad)."""
    g = s * F.muestrear(np.linspace(a, b, 400))
    g = g[np.isfinite(g)]
    if g.size < 3:
        return []
    interiores = np.sum((g[1:-1] < g[:-2]) & (g[1:-1] <= g[2:]))
    if interiores > 1:
        return ["En [a, b] la función parece tener varios óptimos locales: el método "
                "sólo garantiza uno de ellos (requiere función unimodal)."]
    return []


def _validar_intervalo(a: float, b: float, tol: float) -> None:
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError("Se requiere a < b (intervalo de búsqueda válido).")
    if not tol > 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")


def _vec(x) -> str:
    return "(" + ", ".join(f"{float(v):.8g}" for v in x) + ")"


# =============================================================================
# MÉTODO 6 — BÚSQUEDA DE LA SECCIÓN DORADA
# =============================================================================
def seccion_dorada(texto_f: str, a: float, b: float, tol: float = 1e-4,
                   max_iter: int = 200, objetivo: str = "min") -> dict:
    """
    Reduce [a, b] con la razón áurea ρ = (√5 − 1)/2 ≈ 0.618:
        x1 = b − ρ(b−a),   x2 = a + ρ(b−a)
    Minimizar: si f(x1) ≤ f(x2) el óptimo está en [a, x2]; si no, en [x1, b].
    Reutiliza un punto por iteración (1 sola evaluación nueva).
    """
    expr, _, F = _f1d(texto_f)
    _validar_intervalo(a, b, tol)
    s = _signo(objetivo)
    rho = (math.sqrt(5) - 1) / 2
    n_teorico = max(0, math.ceil(math.log(tol / (b - a)) / math.log(rho)))
    a0, b0 = a, b
    x1, x2 = b - rho * (b - a), a + rho * (b - a)
    f1, f2 = F(x1), F(x2)
    filas, k = [], 0
    while (b - a) > tol and k < max_iter:
        k += 1
        filas.append({"k": k, "a": a, "b": b, "x1": x1, "x2": x2,
                      "f(x1)": f1, "f(x2)": f2, "Longitud": b - a})
        if s * f1 <= s * f2:                 # óptimo en [a, x2]
            b, x2, f2 = x2, x1, f1
            x1 = b - rho * (b - a)
            f1 = F(x1)
        else:                                # óptimo en [x1, b]
            a, x1, f1 = x1, x2, f2
            x2 = a + rho * (b - a)
            f2 = F(x2)
    x_opt = (a + b) / 2
    f_opt = F(x_opt)
    convergio = (b - a) <= tol
    return {
        "metodo": "Sección Dorada", "expr": expr, "latex": [sp.latex(expr)],
        "latex_etiquetas": ["f(x)"],
        "tabla": filas, "x_opt": x_opt, "f_opt": f_opt, "convergio": convergio,
        "resumen": [("x*", x_opt), ("f(x*)", f_opt), ("Intervalo final", f"[{a:.8g}, {b:.8g}]"),
                    ("Longitud final", b - a), ("Iteraciones", k),
                    ("Iteraciones teóricas", n_teorico), ("Evaluaciones de f", F.evaluaciones)],
        "advertencias": _advertir_multimodal(F, a0, b0, s)
        + ([] if convergio else ["Se alcanzó max_iter antes de llegar a la tolerancia."]),
        "grafica": _grafica_funcion(F, a0, b0),
    }


# =============================================================================
# MÉTODO 7 — BÚSQUEDA DE FIBONACCI
# =============================================================================
def busqueda_fibonacci(texto_f: str, a: float, b: float, tol: float = 1e-4,
                       objetivo: str = "min") -> dict:
    """
    Con F1 = F2 = 1, se elige n mínimo tal que F(n+1) ≥ (b−a)/tol. En la etapa m:
        x1 = a + [F(m−1)/F(m+1)](b−a),   x2 = b − [F(m−1)/F(m+1)](b−a)
    En la última etapa los dos puntos coinciden (punto medio), por lo que se
    separa uno una distancia ε (ε = 5 % de la tolerancia).
    """
    expr, _, F = _f1d(texto_f)
    _validar_intervalo(a, b, tol)
    s = _signo(objetivo)
    a0, b0 = a, b
    L0 = b - a
    tol_ef = 0.9 * tol                         # margen para el desplazamiento ε
    eps = 0.05 * tol
    fibs = [0, 1, 1]
    while len(fibs) < 5 or fibs[-1] < L0 / tol_ef:
        fibs.append(fibs[-1] + fibs[-2])
    n = len(fibs) - 2                          # F(n+1) = fibs[-1]

    def r(m: int) -> float:
        return fibs[m - 1] / fibs[m + 1]

    m = n
    x1, x2 = a + r(m) * (b - a), b - r(m) * (b - a)
    f1, f2 = F(x1), F(x2)
    filas, k = [], 0
    while True:
        k += 1
        filas.append({"k": k, "Etapa m": m, "a": a, "b": b, "x1": x1, "x2": x2,
                      "f(x1)": f1, "f(x2)": f2, "Longitud": b - a,
                      "F(m−1)/F(m+1)": r(m)})
        if s * f1 <= s * f2:
            b, x2, f2, retenido_x2 = x2, x1, f1, True
        else:
            a, x1, f1, retenido_x2 = x1, x2, f2, False
        m -= 1
        if m < 2:
            break
        L = b - a
        if m >= 3:
            if retenido_x2:
                x1 = a + r(m) * L
                f1 = F(x1)
            else:
                x2 = b - r(m) * L
                f2 = F(x2)
        elif retenido_x2:                      # m == 2: punto medio retenido
            x1 = x2 - eps
            f1 = F(x1)
        else:
            x2 = x1 + eps
            f2 = F(x2)
    x_opt = (a + b) / 2
    f_opt = F(x_opt)
    return {
        "metodo": "Fibonacci", "expr": expr, "latex": [sp.latex(expr)],
        "latex_etiquetas": ["f(x)"],
        "tabla": filas, "x_opt": x_opt, "f_opt": f_opt, "convergio": (b - a) <= tol,
        "resumen": [("x*", x_opt), ("f(x*)", f_opt), ("Intervalo final", f"[{a:.8g}, {b:.8g}]"),
                    ("Longitud final", b - a), ("n (nº de Fibonacci)", n),
                    ("F(n+1)", fibs[-1]), ("Evaluaciones de f", F.evaluaciones)],
        "advertencias": _advertir_multimodal(F, a0, b0, s),
        "grafica": _grafica_funcion(F, a0, b0),
    }


# =============================================================================
# MÉTODO 8 — INTERPOLACIÓN CUADRÁTICA (tres puntos)
# =============================================================================
def interpolacion_cuadratica(texto_f: str, x1: float, x2: float, x3: float,
                             tol: float = 1e-6, max_iter: int = 50,
                             objetivo: str = "min") -> dict:
    """
    Ajusta una parábola por (x1,f1), (x2,f2), (x3,f3) con x1<x2<x3 y f2 el mejor
    valor; su vértice es
        x4 = x2 − ½ · [(x2−x1)²(f2−f3) − (x2−x3)²(f2−f1)]
                    / [(x2−x1)(f2−f3) − (x2−x3)(f2−f1)]
    Se conservan los tres puntos que sigan encerrando el óptimo y se repite
    hasta que |x4 − x2| < tol.
    """
    expr, _, F = _f1d(texto_f)
    s = _signo(objetivo)
    if not (x1 < x2 < x3):
        raise ValueError("Se requiere x1 < x2 < x3.")
    if not tol > 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")
    f1, f2, f3 = F(x1), F(x2), F(x3)
    if not (s * f2 < s * f1 and s * f2 < s * f3):
        raise ValueError("Los tres puntos no encierran un óptimo: f(x2) debe ser "
                         f"{'menor' if s == 1 else 'mayor'} que f(x1) y f(x3). "
                         "Cambia los puntos o el objetivo.")
    lo0, hi0 = x1, x3
    filas, mensajes, convergio = [], [], False
    for k in range(1, max_iter + 1):
        num = (x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)
        den = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(den) < 1e-14:
            mensajes.append("Los tres puntos son (casi) colineales: la parábola degenera.")
            break
        x4 = x2 - 0.5 * num / den
        if not (x1 < x4 < x3):
            mensajes.append("El vértice cayó fuera de [x1, x3]; se detiene la búsqueda.")
            break
        f4 = F(x4)
        filas.append({"k": k, "x1": x1, "x2": x2, "x3": x3, "f(x1)": f1, "f(x2)": f2,
                      "f(x3)": f3, "x4 (vértice)": x4, "f(x4)": f4, "|x4−x2|": abs(x4 - x2)})
        if abs(x4 - x2) < tol:
            convergio = True
            if s * f4 < s * f2:
                x2, f2 = x4, f4
            break
        if x4 > x2:
            if s * f4 < s * f2:
                x1, f1, x2, f2 = x2, f2, x4, f4
            else:
                x3, f3 = x4, f4
        else:
            if s * f4 < s * f2:
                x3, f3, x2, f2 = x2, f2, x4, f4
            else:
                x1, f1 = x4, f4
    if not convergio and not mensajes:
        mensajes.append("Se alcanzó max_iter antes de llegar a la tolerancia.")
    return {
        "metodo": "Interpolación Cuadrática", "expr": expr, "latex": [sp.latex(expr)],
        "latex_etiquetas": ["f(x)"],
        "tabla": filas, "x_opt": x2, "f_opt": f2, "convergio": convergio,
        "resumen": [("x*", x2), ("f(x*)", f2), ("Iteraciones", len(filas)),
                    ("Evaluaciones de f", F.evaluaciones)],
        "advertencias": mensajes, "grafica": _grafica_funcion(F, lo0, hi0),
    }


# =============================================================================
# MÉTODO 9 — NEWTON UNIDIMENSIONAL
# =============================================================================
def newton_1d(texto_f: str, x0: float, tol: float = 1e-6, max_iter: int = 50,
              modo: str = "optimizacion") -> dict:
    """
    modo='optimizacion': busca puntos estacionarios resolviendo f'(x) = 0
                         x_{k+1} = x_k − f'(x_k) / f''(x_k)
    modo='raiz':         Newton-Raphson para f(x) = 0
                         x_{k+1} = x_k − f(x_k) / f'(x_k)
    Se detiene cuando |x_{k+1} − x_k| < tol.
    """
    if modo not in ("optimizacion", "raiz"):
        raise ValueError("modo debe ser 'optimizacion' o 'raiz'.")
    if not tol > 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")
    expr, x_sym, F = _f1d(texto_f)
    d1_expr, d2_expr = sp.diff(expr, x_sym), sp.diff(expr, x_sym, 2)
    d1 = FuncionEscalar(d1_expr, [x_sym])
    d2 = FuncionEscalar(d2_expr, [x_sym])
    opt = modo == "optimizacion"
    h, dh = (d1, d2) if opt else (F, d1)
    et_h, et_dh = ("f'(xk)", "f''(xk)") if opt else ("f(xk)", "f'(xk)")

    x, filas, mensajes, convergio = float(x0), [], [], False
    visitados = [x]
    for k in range(1, max_iter + 1):
        hx, dhx = h(x), dh(x)
        if abs(dhx) < 1e-14:
            mensajes.append(f"{et_dh} ≈ 0 en x = {x:.8g}: el paso de Newton no está definido.")
            break
        x_nuevo = x - hx / dhx
        fila = {"k": k, "xk": x}
        if opt:
            fila["f(xk)"] = F(x)
        fila.update({et_h: hx, et_dh: dhx, "x(k+1)": x_nuevo, "|Δx|": abs(x_nuevo - x)})
        filas.append(fila)
        visitados.append(x_nuevo)
        if abs(x_nuevo) > 1e12:
            mensajes.append("El método divergió (|x| > 1e12). Prueba otro punto inicial.")
            x = x_nuevo
            break
        converge = abs(x_nuevo - x) < tol
        x = x_nuevo
        if converge:
            convergio = True
            break
    if not convergio and not mensajes:
        mensajes.append("Se alcanzó max_iter antes de llegar a la tolerancia.")

    resumen = [("x*", x)]
    if opt:
        f_opt = F(x)
        curv = d2(x)
        clase = ("mínimo local" if curv > 1e-9 else "máximo local" if curv < -1e-9
                 else "no concluyente (f'' ≈ 0)")
        resumen += [("f(x*)", f_opt), ("f'(x*)", d1(x)), ("f''(x*)", curv),
                    ("Clasificación", clase)]
    else:
        f_opt = F(x)
        resumen += [("f(x*) (residuo)", f_opt)]
    resumen += [("Iteraciones", len(filas))]
    lo, hi = min(visitados), max(visitados)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    latex = [sp.latex(expr), sp.latex(d1_expr)] + ([sp.latex(d2_expr)] if opt else [])
    return {
        "metodo": "Newton 1D", "expr": expr, "latex": latex,
        "latex_etiquetas": ["f(x)", "f'(x)"] + (["f''(x)"] if opt else []),
        "tabla": filas, "x_opt": x, "f_opt": f_opt, "convergio": convergio,
        "resumen": resumen, "advertencias": mensajes,
        "grafica": _grafica_funcion(F, lo, hi),
    }


# =============================================================================
# MÉTODO 10 — MÁXIMA INCLINACIÓN (GRADIENTE N-DIMENSIONAL)
# =============================================================================
def _dorada_simple(phi: Callable[[float], float], a: float, b: float,
                   tol: float, max_iter: int = 200) -> float:
    rho = (math.sqrt(5) - 1) / 2
    x1, x2 = b - rho * (b - a), a + rho * (b - a)
    f1, f2 = phi(x1), phi(x2)
    k = 0
    while (b - a) > tol and k < max_iter:
        k += 1
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - rho * (b - a)
            f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + rho * (b - a)
            f2 = phi(x2)
    return (a + b) / 2


def _busqueda_lineal(phi: Callable[[float], float]) -> float:
    """t* ≈ argmin_{t≥0} φ(t): encierra el mínimo (reduce/expande) y aplica
    sección dorada. Devuelve 0 si no hay descenso."""
    phi0 = phi(0.0)
    t, ft = 1.0, phi(1.0)
    for _ in range(60):                         # reducir hasta tener descenso
        if ft < phi0:
            break
        t /= 2
        ft = phi(t)
    else:
        return 0.0
    lo = 0.0
    for _ in range(80):                         # expandir hasta que φ suba
        t2 = 2 * t
        f2 = phi(t2)
        if f2 > ft:
            return _dorada_simple(phi, lo, t2, 1e-12 * max(1.0, t2))
        lo, t, ft = t, t2, f2
    raise ValueError("f no está acotada en la dirección de búsqueda (no hay óptimo).")


def maxima_inclinacion(texto_f: str, x0: Sequence[float], tol: float = 1e-6,
                       max_iter: int = 100, objetivo: str = "min") -> dict:
    """
    x_{k+1} = x_k − t*·∇f(x_k),   t* = argmin_{t≥0} f(x_k − t∇f(x_k))
    (búsqueda lineal por sección dorada). Para MAXIMIZAR se sigue +∇f.
    Variables: x1..xn (y también x, y, z si n ≤ 3). Para cuando ‖∇f‖ < tol.
    """
    n = len(x0)
    if n < 1:
        raise ValueError("Indica al menos una variable.")
    if not tol > 0:
        raise ValueError("La tolerancia debe ser mayor que 0.")
    s = _signo(objetivo)
    nombres = [f"x{i + 1}" for i in range(n)]
    alias = {a: nombres[i] for i, a in enumerate(("x", "y", "z")) if i < n}
    expr, sims = parsear_expresion(texto_f, nombres, alias)
    F = FuncionEscalar(expr, sims)
    grad_expr = [sp.diff(expr, v) for v in sims]
    G = [FuncionEscalar(ge, sims) for ge in grad_expr]

    def grad(x):
        return np.array([g(*x) for g in G], dtype=float)

    x = np.array(x0, dtype=float)
    filas, mensajes, convergio = [], [], False
    historial = []
    for k in range(1, max_iter + 1):
        fx, gx = F(*x), grad(x)
        norma = float(np.linalg.norm(gx))
        historial.append(fx)
        if norma < tol:
            filas.append({"k": k, "x_k": _vec(x), "f(x_k)": fx, "∇f(x_k)": _vec(gx),
                          "‖∇f‖": norma, "t*": "—", "x_(k+1)": "—"})
            convergio = True
            break
        d = -s * gx                              # dirección de máximo descenso (o ascenso)
        t = _busqueda_lineal(lambda tt: s * F(*(x + tt * d)))
        x_nuevo = x + t * d
        filas.append({"k": k, "x_k": _vec(x), "f(x_k)": fx, "∇f(x_k)": _vec(gx),
                      "‖∇f‖": norma, "t*": t, "x_(k+1)": _vec(x_nuevo)})
        if t == 0.0:
            mensajes.append("No se encontró un paso que mejore f; se detiene.")
            break
        x = x_nuevo
    if not convergio and not mensajes:
        mensajes.append("Se alcanzó max_iter antes de que ‖∇f‖ < tol.")
    f_fin, g_fin = F(*x), grad(x)
    historial_final = historial if convergio else historial + [f_fin]
    return {
        "metodo": "Máxima Inclinación", "expr": expr,
        "latex": [sp.latex(expr)] + [sp.latex(g) for g in grad_expr],
        "latex_etiquetas": ["f"] + [f"∂f/∂x{i + 1}" for i in range(n)],
        "tabla": filas, "x_opt": x.tolist(), "f_opt": f_fin, "convergio": convergio,
        "resumen": [("x*", _vec(x)), ("f(x*)", f_fin),
                    ("‖∇f(x*)‖", float(np.linalg.norm(g_fin))),
                    ("Iteraciones", len(filas)), ("Evaluaciones de f", F.evaluaciones)],
        "advertencias": mensajes,
        "grafica": {"titulo": "Convergencia: f(x_k) por iteración", "x_label": "k",
                    "x": list(range(1, len(historial_final) + 1)),
                    "series": {"f(x_k)": historial_final}},
    }