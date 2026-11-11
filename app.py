# -*- coding: utf-8 -*-
"""
app.py — Interfaz Streamlit unificada: Transporte + Métodos Numéricos
=====================================================================
Un ÚNICO menú en la barra lateral (st.sidebar.selectbox) da acceso a los 10
métodos; la pantalla principal carga dinámicamente los campos de entrada y los
resultados del método elegido.

    1 Costo Mínimo · 2 Esquina Noroeste · 3 Vogel · 4 Cruce del Arroyo · 5 MODI
    6 Sección Dorada · 7 Fibonacci · 8 Interpolación Cuadrática · 9 Newton 1D
    10 Máxima Inclinación (gradiente N-dimensional)

Ejecutar (junto a metodos.py):

    pip install streamlit numpy pandas sympy
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from metodos import (
    CAT_NUMERICOS,
    CAT_TRANSPORTE,
    CATALOGO,
    CONFIG_TRANSPORTE,
    METODOS_INICIALES,
    NOMBRE_FICTICIO,
    ORDEN,
    TOL,
    busqueda_fibonacci,
    detalle_costo,
    interpolacion_cuadratica,
    maxima_inclinacion,
    newton_1d,
    parsear_expresion,
    resolver_transporte,
    seccion_dorada,
    texto_ciclo,
)

st.set_page_config(page_title="Optimización: Transporte y Métodos Numéricos",
                   page_icon="📐", layout="wide")

ICONOS = {CAT_TRANSPORTE: "🚚", CAT_NUMERICOS: "📈"}

DESCRIPCIONES = {
    "costo_minimo": "Asigna siempre a la casilla **más barata** disponible. "
                    "Entrega la solución básica inicial.",
    "esquina_noroeste": "Asigna desde la casilla **superior izquierda**, sin mirar los costos. "
                        "Entrega la solución básica inicial.",
    "vogel": "Usa **penalizaciones** (diferencia entre los dos menores costos de cada "
             "fila/columna). Suele dar la mejor solución inicial.",
    "cruce_arroyo": "Elige una solución inicial y la **optimiza** calculando el costo marginal Δ "
                    "de cada casilla no básica con su ciclo (Stepping Stone).",
    "modi": "Elige una solución inicial y la **optimiza** con multiplicadores "
            "Ui + Vj = Cij e indicadores Iij = Ui + Vj − Cij.",
    "dorada": "Reduce el intervalo [a, b] con la razón áurea (0.618) para hallar el "
              "mínimo/máximo de una función **unimodal**.",
    "fibonacci": "Reduce el intervalo [a, b] con razones de números de Fibonacci: el número de "
                 "evaluaciones se fija de antemano según la tolerancia.",
    "cuadratica": "Ajusta una **parábola** por tres puntos x1 < x2 < x3 que encierran el óptimo "
                  "y salta a su vértice, repitiendo hasta converger.",
    "newton": "Itera x ← x − f'(x)/f''(x) para hallar puntos estacionarios "
              "(o x ← x − f(x)/f'(x) para raíces).",
    "gradiente": "Método del **máximo descenso** en n variables: avanza por −∇f con paso óptimo "
                 "(búsqueda lineal) hasta que ‖∇f‖ < tolerancia.",
}


# =============================================================================
# UTILIDADES GENERALES
# =============================================================================
def fmt(v) -> str:
    """Número sin ceros sobrantes ni notación científica extrema."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.10g}"
    return str(v)


def _compat(funcion, *args, **kwargs):
    """Widget a ancho completo, compatible con Streamlit nuevo (width='stretch')
    y antiguo (use_container_width)."""
    try:
        return funcion(*args, width="stretch", **kwargs)
    except Exception:
        return funcion(*args, use_container_width=True, **kwargs)


def mostrar_df(obj, **kwargs) -> None:
    _compat(st.dataframe, obj, **kwargs)


def persistente(widget, etiqueta, clave, defecto, *args, **kwargs):
    """
    Crea un widget cuyo valor sobrevive al cambiar de método en el menú lateral
    (Streamlit descarta el estado de los widgets que dejan de dibujarse).
    """
    ss = st.session_state
    if clave not in ss:
        ss[clave] = ss.get("_p_" + clave, defecto)
    valor = widget(etiqueta, *args, key=clave, **kwargs)
    ss["_p_" + clave] = valor
    return valor


def leer_float(texto: str, nombre: str) -> float:
    try:
        return float(str(texto).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"«{nombre}» no es un número válido: «{texto}».") from None


def ejecutar(clave: str, resolver: bool, huella, calcular, mostrar) -> None:
    """Ejecuta (si se pulsó el botón), guarda el resultado por método y lo muestra."""
    ss = st.session_state
    if resolver:
        try:
            ss.resultados[clave] = (huella, calcular())
        except Exception as err:                         # noqa: BLE001
            ss.resultados.pop(clave, None)
            st.error(f"❌ {err}")
    st.markdown("---")
    registro = ss.resultados.get(clave)
    if registro is None:
        st.info("👈 Completa los datos y presiona **🚀 Resolver / Ejecutar** en el panel lateral.")
        return
    if registro[0] != huella:
        st.warning("Los datos cambiaron desde la última ejecución. "
                   "Presiona **Resolver / Ejecutar** para actualizar.")
    mostrar(registro[1])


# =============================================================================
# TRANSPORTE — presentación
# =============================================================================
def df_asignacion(p, x: np.ndarray, basicas) -> pd.DataFrame:
    """Cuadro de transporte: cada celda 'cantidad [costo unitario]'."""
    filas = []
    for i in range(p.m):
        fila = []
        for j in range(p.n):
            if (i, j) in basicas:
                q = "ε" if x[i, j] <= TOL else fmt(x[i, j])
            else:
                q = "·"
            fila.append(f"{q}  [{fmt(p.costos[i, j])}]")
        fila.append(fmt(p.oferta[i]))
        filas.append(fila)
    filas.append([fmt(d) for d in p.demanda] + [fmt(p.oferta.sum())])
    return pd.DataFrame(filas, index=p.origenes + ["Demanda"], columns=p.destinos + ["Oferta"])


def estilo_asignacion(df: pd.DataFrame, basicas):
    """Resalta en verde las casillas básicas."""
    def _css(_):
        css = pd.DataFrame("", index=df.index, columns=df.columns)
        for (i, j) in basicas:
            css.iloc[i, j] = "background-color: rgba(46,160,67,0.30); font-weight: 600;"
        return css
    return df.style.apply(_css, axis=None)


def tabla_asignacion(p, x, basicas) -> None:
    mostrar_df(estilo_asignacion(df_asignacion(p, x, basicas), basicas))
    st.caption("Formato: cantidad [costo unitario] · verde = casilla básica · "
               "· = no básica · ε = comodín de degeneración")


def df_matriz(p, matriz: np.ndarray, basicas, u=None, v=None) -> pd.DataFrame:
    """Matriz de indicadores / costos marginales ('—' en las básicas)."""
    filas = [["—" if (i, j) in basicas or np.isnan(matriz[i, j]) else fmt(matriz[i, j])
              for j in range(p.n)] for i in range(p.m)]
    cols, idx = list(p.destinos), list(p.origenes)
    if u is not None:
        for i in range(p.m):
            filas[i].append(fmt(u[i]))
        filas.append([fmt(val) for val in v] + [""])
        cols.append("Ui")
        idx.append("Vj")
    return pd.DataFrame(filas, index=idx, columns=cols)


def mostrar_balance(res) -> None:
    info = res["info_balance"]
    S, D = info["oferta_total"], info["demanda_total"]
    if info["balanceado"]:
        st.success(f"✅ Problema balanceado: oferta total = demanda total = {fmt(S)}.")
        return
    lugar = "un **destino ficticio**" if info["ficticio"] == "destino" else "un **origen ficticio**"
    st.warning(
        f"⚠️ **Problema desbalanceado**: oferta total = {fmt(S)} y demanda total = {fmt(D)}. "
        f"Se agregó automáticamente {lugar} con "
        f"{'demanda' if info['ficticio'] == 'destino' else 'oferta'} de "
        f"**{fmt(info['cantidad'])}** unidades y costos unitarios de 0.")
    with st.expander("Ver el problema balanceado (costos con el nodo ficticio)"):
        p = res["problema"]
        filas = [[fmt(c) for c in p.costos[i]] + [fmt(p.oferta[i])] for i in range(p.m)]
        filas.append([fmt(d) for d in p.demanda] + [fmt(p.oferta.sum())])
        mostrar_df(pd.DataFrame(filas, index=p.origenes + ["Demanda"],
                                columns=p.destinos + ["Oferta"]))


def mostrar_inicial(res) -> None:
    p = res["problema"]
    st.subheader(f"1️⃣ Solución inicial — {res['metodo_inicial']}")
    with st.expander("Ver paso a paso de la asignación inicial"):
        for k, s in enumerate(res["pasos_iniciales"], 1):
            st.markdown(f"**{k}.** `{s}`")
    tabla_asignacion(p, res["x_inicial"], res["basicas_inicial"])
    if res["comodines"]:
        st.info("Solución **degenerada**: faltaban casillas básicas (m + n − 1). "
                "Se agregaron comodines ε en "
                + ", ".join(f"X{i + 1}{j + 1}" for i, j in res["comodines"]) + ".")
    c1, c2 = st.columns(2)
    c1.metric("💲 Costo inicial", fmt(res["costo_inicial"]))
    c2.metric("Casillas básicas", f"{len(res['basicas_inicial'])}  (m+n−1 = {p.m + p.n - 1})")
    with st.expander("Ver el cálculo del costo inicial"):
        st.code(detalle_costo(p, res["x_inicial"]).replace(
            f" = {fmt(res['costo_inicial'])}", f"\n= {fmt(res['costo_inicial'])}"))


def resumen_iteraciones(its) -> pd.DataFrame:
    filas = []
    for it in its:
        es_modi = it["metodo"] == "MODI"
        if it["es_optima"]:
            filas.append({"Iter": it["numero"], "Costo": fmt(it["costo"]), "Entra": "—",
                          "Valor": "—", "Ciclo": "—", "θ": "—", "Sale": "—",
                          "Resultado": "✔ Óptima"})
        else:
            i, j = it["entrante"]
            si, sj = it["sale"]
            filas.append({"Iter": it["numero"], "Costo": fmt(it["costo"]),
                          "Entra": f"X{i + 1}{j + 1}",
                          "Valor": f"{'I' if es_modi else 'Δ'} = {fmt(it['valor_entrante'])}",
                          "Ciclo": texto_ciclo(it["ciclo"]), "θ": fmt(it["theta"]),
                          "Sale": f"X{si + 1}{sj + 1}",
                          "Resultado": f"→ costo {fmt(it['costo_siguiente'])}"})
    return pd.DataFrame(filas)


def mostrar_iteracion(p, it) -> None:
    titulo = f"Iteración {it['numero']} — costo {fmt(it['costo'])}"
    titulo += "  ✔ óptima" if it["es_optima"] else f"  →  {fmt(it['costo_siguiente'])}"
    with st.expander(titulo, expanded=False):
        if it["numero"] > 1:
            st.markdown("**Cuadro de transporte al inicio de la iteración**")
            tabla_asignacion(p, it["x"], it["basicas"])
        if it["metodo"] == "MODI":
            st.markdown("**Sistema Ui + Vj = Cij (casillas básicas, con U1 = 0)**")
            st.code("\n".join(it["ecuaciones"]))
            st.markdown("**Solución:** " + ",  ".join(f"U{i + 1} = {fmt(u)}"
                                                      for i, u in enumerate(it["u"]))
                        + "  |  " + ",  ".join(f"V{j + 1} = {fmt(v)}"
                                               for j, v in enumerate(it["v"])))
            st.markdown("**Indicadores de las no básicas:**  Iij = Ui + Vj − Cij")
        else:
            st.markdown("**Costo marginal Δ de cada casilla no básica (ciclo de Cruce del Arroyo)**")
        st.code("\n".join(it["detalle"]) if it["detalle"] else "(no hay casillas no básicas)")
        if it["metodo"] == "MODI":
            mostrar_df(df_matriz(p, it["matriz"], it["basicas"], it["u"], it["v"]))
        else:
            mostrar_df(df_matriz(p, it["matriz"], it["basicas"]))
        if it["es_optima"]:
            regla = "todos los indicadores son ≤ 0" if it["metodo"] == "MODI" \
                else "todos los costos marginales Δ son ≥ 0"
            st.success(f"✔ Solución óptima: {regla}.")
            if it["alternativas"]:
                st.caption("Hay valores = 0 en no básicas: existen soluciones óptimas alternativas.")
        else:
            i, j = it["entrante"]
            st.warning(f"Entra X{i + 1}{j + 1} (valor {fmt(it['valor_entrante'])}). "
                       f"Ciclo: {texto_ciclo(it['ciclo'])}. θ = {fmt(it['theta'])}. "
                       f"Sale X{it['sale'][0] + 1}{it['sale'][1] + 1}.")


def mostrar_plan(res) -> None:
    filas = []
    for r in res["plan"]:
        nota = "ficticio: no se envía / no se recibe" \
            if NOMBRE_FICTICIO in (r["origen"], r["destino"]) else ""
        filas.append({"Origen": r["origen"], "Destino": r["destino"],
                      "Unidades": r["unidades"], "Costo unitario": r["costo_unit"],
                      "Subtotal": r["subtotal"], "Nota": nota})
    mostrar_df(pd.DataFrame(filas), hide_index=True)


def mostrar_optimizacion(res) -> None:
    p = res["problema"]
    opt = res["optimizacion"]
    its = opt["iteraciones"]
    n_mejoras = sum(1 for it in its if not it["es_optima"])
    st.subheader(f"2️⃣ Optimización — {res['metodo_optimizacion']}")
    st.markdown(f"Se necesitaron **{n_mejoras}** reasignación(es) para llegar al óptimo "
                f"({len(its)} prueba(s) de optimalidad).")
    mostrar_df(resumen_iteraciones(its), hide_index=True)
    st.caption("Historial de costos: " + " → ".join(fmt(c) for c in res["historial_costos"]))
    for it in its:
        mostrar_iteracion(p, it)

    st.subheader("🏁 Matriz final óptima")
    tabla_asignacion(p, res["x_final"], res["basicas_final"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Costo inicial", fmt(res["costo_inicial"]))
    c2.metric("💲 Costo total minimizado", fmt(res["costo_final"]),
              delta=fmt(res["costo_final"] - res["costo_inicial"]), delta_color="inverse")
    c3.metric("Reasignaciones", n_mejoras)
    with st.expander("Ver el cálculo del costo mínimo"):
        st.code(detalle_costo(p, res["x_final"]))
    if opt["alternativas"]:
        st.caption("ℹ️ Existen soluciones óptimas alternativas con el mismo costo.")
    st.markdown("**Plan de envíos óptimo**")
    mostrar_plan(res)


def mostrar_solo_inicial(res) -> None:
    st.subheader("Plan de envíos de la solución inicial")
    st.info("Esta es una solución básica **factible**, pero no se verificó si es óptima. "
            "Para comprobarlo usa **MODI** o **Cruce del Arroyo** en el menú lateral.")
    mostrar_plan(res)


def mostrar_resultado_transporte(res) -> None:
    mostrar_balance(res)
    mostrar_inicial(res)
    if res["optimizacion"] is not None:
        mostrar_optimizacion(res)
    else:
        mostrar_solo_inicial(res)


# =============================================================================
# TRANSPORTE — captura de datos (tabla editable)
# =============================================================================
def tabla_base(m: int, n: int, previa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla (m+1) x (n+1): costos, oferta (última columna) y demanda (última fila).
    Si hay una tabla previa conserva los valores que quepan."""
    cols = [f"Destino {j + 1}" for j in range(n)] + ["Oferta"]
    idx = [f"Origen {i + 1}" for i in range(m)] + ["Demanda"]
    df = pd.DataFrame(np.nan, index=idx, columns=cols, dtype=float)
    if previa is not None:
        pm, pn = previa.shape[0] - 1, previa.shape[1] - 1
        for i in range(min(m, pm)):
            for j in range(min(n, pn)):
                df.iloc[i, j] = previa.iloc[i, j]
            df.iloc[i, n] = previa.iloc[i, pn]                 # oferta
        for j in range(min(n, pn)):
            df.iloc[m, j] = previa.iloc[pm, j]                 # demanda
    return df


def extraer_datos(df: pd.DataFrame, m: int, n: int):
    return (df.iloc[:m, :n].to_numpy(dtype=float), df.iloc[:m, n].to_numpy(dtype=float),
            df.iloc[m, :n].to_numpy(dtype=float))


def capturar_datos_transporte():
    """Dimensiones + tabla editable + nombres. Devuelve (m, n, tabla, origenes, destinos)."""
    ss = st.session_state

    # Al volver de un método numérico, reiniciar el editor desde lo ya digitado.
    if not ss.get("_vino_de_transporte", False) and ss.get("editada") is not None:
        ss.base = ss.editada.copy()
        ss.version += 1

    st.subheader("📝 Ingreso de datos")
    c1, c2, c3 = st.columns([1, 1, 1])
    m = int(persistente(c1.number_input, "Número de Orígenes (Oferta)", "tr_m", 3,
                        min_value=1, max_value=12, step=1))
    n = int(persistente(c2.number_input, "Número de Destinos (Demanda)", "tr_n", 3,
                        min_value=1, max_value=12, step=1))
    if c3.button("🗑️ Limpiar tabla"):
        ss.base = tabla_base(m, n)
        ss.editada = ss.base
        ss.version += 1
        ss.dims = (m, n)
        st.rerun()

    if ss.get("dims") != (m, n):                 # sólo se reconstruye si cambian m o n
        ss.base = tabla_base(m, n, ss.get("editada"))
        ss.dims = (m, n)
        ss.version += 1

    st.markdown("Escribe los **costos unitarios** en el cuerpo de la tabla, la **oferta** en "
                "la última columna y la **demanda** en la última fila "
                "(la celda inferior derecha se ignora).")
    editada = _compat(st.data_editor, ss.base, key=f"editor_{ss.version}", num_rows="fixed",
                      column_config={"Oferta": st.column_config.NumberColumn("Oferta")})
    ss.editada = editada

    with st.expander("🏷️ Nombres personalizados (opcional)"):
        k1, k2 = st.columns(2)
        origenes, destinos = [], []
        for i in range(m):
            nombre = persistente(k1.text_input, f"Nombre del origen {i + 1}", f"tr_no_{i}",
                                 f"Origen {i + 1}")
            origenes.append(nombre.strip() or f"Origen {i + 1}")
        for j in range(n):
            nombre = persistente(k2.text_input, f"Nombre del destino {j + 1}", f"tr_nd_{j}",
                                 f"Destino {j + 1}")
            destinos.append(nombre.strip() or f"Destino {j + 1}")
        if len(set(origenes)) < m or len(set(destinos)) < n:
            st.warning("Hay nombres repetidos; usa nombres distintos para evitar confusiones.")

    oferta_tot = float(np.nansum(editada.iloc[:m, n].to_numpy(dtype=float)))
    demanda_tot = float(np.nansum(editada.iloc[m, :n].to_numpy(dtype=float)))
    t1, t2, t3 = st.columns(3)
    t1.metric("Oferta total", fmt(oferta_tot))
    t2.metric("Demanda total", fmt(demanda_tot))
    t3.metric("Estado", "Balanceado ✅" if abs(oferta_tot - demanda_tot) < TOL
              else "Desbalanceado ⚠️ (se agregará nodo ficticio)")
    return m, n, editada, origenes, destinos


def pagina_transporte(clave: str, resolver: bool) -> None:
    inicial, optimizacion = CONFIG_TRANSPORTE[clave]
    if inicial is None:                           # métodos 4 y 5: elegir la solución inicial
        inicial = persistente(st.selectbox, "Método para la solución inicial", "tr_ini",
                              "Costo Mínimo", list(METODOS_INICIALES))
    m, n, tabla, origenes, destinos = capturar_datos_transporte()
    huella = (clave, inicial, tabla.shape,
              tuple(np.nan_to_num(tabla.to_numpy(dtype=float), nan=-1e300).ravel()),
              tuple(origenes), tuple(destinos))

    def calcular():
        costos, oferta, demanda = extraer_datos(tabla, m, n)
        return resolver_transporte(costos, oferta, demanda, origenes, destinos,
                                   metodo_inicial=inicial, metodo_optimizacion=optimizacion)

    ejecutar(clave, resolver, huella, calcular, mostrar_resultado_transporte)


# =============================================================================
# MÉTODOS NUMÉRICOS — presentación genérica
# =============================================================================
def mostrar_numerico(res: dict) -> None:
    st.subheader("📌 Resultado")
    for adv in res["advertencias"]:
        st.warning(adv)
    if res["convergio"]:
        st.success("✔ El método alcanzó la tolerancia pedida.")
    elif not res["advertencias"]:
        st.info("El método terminó sin verificar la tolerancia.")

    items = res["resumen"]
    for ini in range(0, len(items), 4):
        fila = items[ini:ini + 4]
        for col, (etiqueta, valor) in zip(st.columns(len(fila)), fila):
            col.metric(etiqueta, fmt(valor))

    with st.expander("Ver la función (y sus derivadas) en forma simbólica"):
        for etiqueta, lx in zip(res["latex_etiquetas"], res["latex"]):
            st.latex(f"{etiqueta.replace('∂', chr(92) + 'partial ')} = {lx}")

    st.subheader("📋 Tabla de iteraciones")
    if res["tabla"]:
        df = pd.DataFrame(res["tabla"])
        df = df.set_index(df.columns[0])
        mostrar_df(df.style.format(lambda v: fmt(v) if isinstance(v, (float, np.floating)) else v,
                                   na_rep="—"))
    else:
        st.caption("No hubo iteraciones.")

    g = res["grafica"]
    st.markdown(f"**{g['titulo']}**")
    datos = pd.DataFrame({g["x_label"]: g["x"], **g["series"]}).set_index(g["x_label"])
    _compat(st.line_chart, datos)


def vista_previa(texto: str, variables, alias=None) -> None:
    """Muestra la función interpretada en LaTeX (o el error de lectura)."""
    if not texto.strip():
        st.caption("Escribe la función. Admite +, −, *, /, ^ (o **), sin, cos, tan, exp, log, "
                   "sqrt, abs, pi, e; y multiplicación implícita (2x).")
        return
    try:
        expr, _ = parsear_expresion(texto, variables, alias)
        st.latex("f = " + sp.latex(expr))
    except ValueError as err:
        st.caption(f"⚠️ {err}")


def _objetivo(etiqueta: str) -> str:
    return "min" if etiqueta == "Minimizar" else "max"


# =============================================================================
# MÉTODOS NUMÉRICOS — páginas (campos de entrada dinámicos por método)
# =============================================================================
def pagina_busqueda_intervalo(clave: str, resolver: bool) -> None:
    """Sección Dorada, Fibonacci e Interpolación Cuadrática."""
    st.subheader("📝 Ingreso de datos")
    texto = persistente(st.text_input, "Función f(x)", f"{clave}_f", "",
                        placeholder="Ej.: x^2 - 4x + 3   |   exp(x) - 2x   |   x^2 + sin(3x)")
    vista_previa(texto, ["x"])

    c1, c2 = st.columns(2)
    objetivo = persistente(c1.selectbox, "Objetivo", f"{clave}_obj", "Minimizar",
                           ["Minimizar", "Maximizar"])
    tol_txt = persistente(c2.text_input, "Tolerancia (ε)", f"{clave}_tol",
                          "0.000001" if clave == "cuadratica" else "0.0001")

    if clave == "cuadratica":
        k1, k2, k3, k4 = st.columns(4)
        x1 = persistente(k1.text_input, "x1 (izquierda)", f"{clave}_x1", "0")
        x2 = persistente(k2.text_input, "x2 (centro)", f"{clave}_x2", "1")
        x3 = persistente(k3.text_input, "x3 (derecha)", f"{clave}_x3", "3")
        mi = int(persistente(k4.number_input, "Máx. iteraciones", f"{clave}_it", 50,
                             min_value=1, max_value=500, step=1))
        st.caption("Se requiere x1 < x2 < x3 con f(x2) mejor que f(x1) y f(x3) "
                   "(los tres puntos deben encerrar el óptimo).")
        campos = (x1, x2, x3, mi)
    else:
        k1, k2, k3 = st.columns(3)
        a = persistente(k1.text_input, "Límite inferior a", f"{clave}_a", "0")
        b = persistente(k2.text_input, "Límite superior b", f"{clave}_b", "1")
        if clave == "dorada":
            mi = int(persistente(k3.number_input, "Máx. iteraciones", f"{clave}_it", 200,
                                 min_value=1, max_value=1000, step=1))
        else:
            mi = None
        campos = (a, b, mi)

    huella = (clave, texto, objetivo, tol_txt, campos)

    def calcular():
        obj, tol = _objetivo(objetivo), leer_float(tol_txt, "Tolerancia")
        if clave == "cuadratica":
            return interpolacion_cuadratica(texto, leer_float(x1, "x1"), leer_float(x2, "x2"),
                                            leer_float(x3, "x3"), tol, mi, obj)
        lo, hi = leer_float(a, "a"), leer_float(b, "b")
        if clave == "dorada":
            return seccion_dorada(texto, lo, hi, tol, mi, obj)
        return busqueda_fibonacci(texto, lo, hi, tol, obj)

    ejecutar(clave, resolver, huella, calcular, mostrar_numerico)


def pagina_newton(clave: str, resolver: bool) -> None:
    st.subheader("📝 Ingreso de datos")
    texto = persistente(st.text_input, "Función f(x)", f"{clave}_f", "",
                        placeholder="Ej.: x^3 - 3x + 1   |   exp(x) - 2x   |   x^2 - 2")
    vista_previa(texto, ["x"])
    c1, c2 = st.columns([2, 1])
    modo = persistente(c1.selectbox, "Modo", f"{clave}_modo", "Optimización: resolver f'(x) = 0",
                       ["Optimización: resolver f'(x) = 0", "Raíz: resolver f(x) = 0"])
    mi = int(persistente(c2.number_input, "Máx. iteraciones", f"{clave}_it", 50,
                         min_value=1, max_value=500, step=1))
    k1, k2 = st.columns(2)
    x0 = persistente(k1.text_input, "Punto inicial x0", f"{clave}_x0", "1")
    tol_txt = persistente(k2.text_input, "Tolerancia (ε)", f"{clave}_tol", "0.000001")
    huella = (clave, texto, modo, x0, tol_txt, mi)

    def calcular():
        return newton_1d(texto, leer_float(x0, "x0"), leer_float(tol_txt, "Tolerancia"), mi,
                         "optimizacion" if modo.startswith("Optim") else "raiz")

    ejecutar(clave, resolver, huella, calcular, mostrar_numerico)


def pagina_gradiente(clave: str, resolver: bool) -> None:
    st.subheader("📝 Ingreso de datos")
    c1, c2, c3 = st.columns(3)
    n = int(persistente(c1.number_input, "Número de variables n", f"{clave}_n", 2,
                        min_value=1, max_value=8, step=1))
    objetivo = persistente(c2.selectbox, "Objetivo", f"{clave}_obj", "Minimizar",
                           ["Minimizar", "Maximizar"])
    mi = int(persistente(c3.number_input, "Máx. iteraciones", f"{clave}_it", 100,
                         min_value=1, max_value=1000, step=1))

    nombres = [f"x{i + 1}" for i in range(n)]
    alias = {a: nombres[i] for i, a in enumerate(("x", "y", "z")) if i < n}
    texto = persistente(st.text_input, f"Función f({', '.join(nombres)})", f"{clave}_f", "",
                        placeholder="Ej.: (x1-1)^2 + 2*(x2+2)^2   |   x1^2 + x1*x2 + x2^2 - 3x1")
    vista_previa(texto, nombres, alias)
    if n <= 3:
        st.caption("Variables: " + ", ".join(nombres) + "  (también puedes escribir x, y, z).")
    else:
        st.caption("Variables: " + ", ".join(nombres) + ".")

    st.markdown("**Punto inicial x⁰**")
    x0 = [persistente(col.text_input, f"x{i + 1}", f"{clave}_x0_{i}", "0")
          for i, col in enumerate(st.columns(n))]
    tol_txt = persistente(st.text_input, "Tolerancia sobre ‖∇f‖", f"{clave}_tol", "0.000001")
    huella = (clave, n, texto, objetivo, tuple(x0), tol_txt, mi)

    def calcular():
        punto = [leer_float(v, f"x{i + 1}") for i, v in enumerate(x0)]
        return maxima_inclinacion(texto, punto, leer_float(tol_txt, "Tolerancia"), mi,
                                  _objetivo(objetivo))

    ejecutar(clave, resolver, huella, calcular, mostrar_numerico)


PAGINAS_NUMERICAS = {
    "dorada": pagina_busqueda_intervalo,
    "fibonacci": pagina_busqueda_intervalo,
    "cuadratica": pagina_busqueda_intervalo,
    "newton": pagina_newton,
    "gradiente": pagina_gradiente,
}


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def etiqueta_menu(clave: str) -> str:
    info = CATALOGO[clave]
    return f"{ICONOS[info['cat']]} {info['cat']} · {info['n']}. {info['nombre']}"


def main() -> None:
    ss = st.session_state
    ss.setdefault("version", 0)
    ss.setdefault("resultados", {})

    # ---------------------- SIDEBAR: UN ÚNICO MENÚ --------------------------
    st.sidebar.title("📐 Optimización")
    clave = st.sidebar.selectbox("Selecciona el método", ORDEN, format_func=etiqueta_menu,
                                 key="metodo_activo")
    resolver = _compat(st.sidebar.button, "🚀 Resolver / Ejecutar", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{ICONOS[CAT_TRANSPORTE]} **{CAT_TRANSPORTE}**: métodos 1 a 5")
    st.sidebar.caption(f"{ICONOS[CAT_NUMERICOS]} **{CAT_NUMERICOS}**: métodos 6 a 10")

    # ---------------------- PANTALLA PRINCIPAL ------------------------------
    info = CATALOGO[clave]
    st.title(f"{ICONOS[info['cat']]} {info['nombre']}")
    st.caption(f"Categoría: {info['cat']} · Método {info['n']} de 10")
    st.markdown(DESCRIPCIONES[clave])

    if info["cat"] == CAT_TRANSPORTE:
        pagina_transporte(clave, resolver)
    else:
        PAGINAS_NUMERICAS[clave](clave, resolver)

    ss["_vino_de_transporte"] = info["cat"] == CAT_TRANSPORTE


if __name__ == "__main__":
    main()