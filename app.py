from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from metodos import (
    METODOS_INICIALES,
    METODOS_OPTIMIZACION,
    NOMBRE_FICTICIO,
    TOL,
    detalle_costo,
    resolver_transporte,
    texto_ciclo,
)

st.set_page_config(page_title="Modelo de Transporte", page_icon="🚚", layout="wide")


# =============================================================================
# UTILIDADES DE PRESENTACIÓN
# =============================================================================
def fmt(v: float) -> str:
    """Número sin ceros sobrantes ni notación científica."""
    return f"{v:.10g}"


def _compat(funcion, *args, **kwargs):
    """Llama a un widget con ancho completo, compatible con versiones de
    Streamlit nuevas (width='stretch') y antiguas (use_container_width)."""
    try:
        return funcion(*args, width="stretch", **kwargs)
    except Exception:
        return funcion(*args, use_container_width=True, **kwargs)


def mostrar_df(obj, **kwargs) -> None:
    _compat(st.dataframe, obj, **kwargs)


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
    return pd.DataFrame(filas, index=p.origenes + ["Demanda"],
                        columns=p.destinos + ["Oferta"])


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
        filas.append([fmt(x) for x in v] + [""])
        cols.append("Ui")
        idx.append("Vj")
    return pd.DataFrame(filas, index=idx, columns=cols)


# =============================================================================
# CAPTURA DE DATOS (HUD)
# =============================================================================
def tabla_base(m: int, n: int, previa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla editable (m+1) x (n+1): costos, oferta (última columna) y demanda
    (última fila). Si hay una tabla previa, conserva los valores que quepan."""
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
    costos = df.iloc[:m, :n].to_numpy(dtype=float)
    oferta = df.iloc[:m, n].to_numpy(dtype=float)
    demanda = df.iloc[m, :n].to_numpy(dtype=float)
    return costos, oferta, demanda


def firma(df, origenes, destinos, mi, mo):
    """Huella de los datos para detectar resultados desactualizados."""
    return (df.shape, tuple(np.nan_to_num(df.to_numpy(dtype=float), nan=-1e300).ravel()),
            tuple(origenes), tuple(destinos), mi, mo)


# =============================================================================
# RESULTADOS
# =============================================================================
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


def mostrar_iteracion(p, it, ultima: bool) -> None:
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
    st.caption("Historial de costos: "
               + " → ".join(fmt(c) for c in res["historial_costos"]))
    for it in its:
        mostrar_iteracion(p, it, it is its[-1])

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
    st.info("Elegiste «Solo inicial»: esta solución es factible, pero no se verificó "
            "si es óptima.")
    mostrar_plan(res)


def mostrar_resultados(res) -> None:
    mostrar_balance(res)
    mostrar_inicial(res)
    if res["optimizacion"] is not None:
        mostrar_optimizacion(res)
    else:
        mostrar_solo_inicial(res)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main() -> None:
    ss = st.session_state
    ss.setdefault("version", 0)

    # ------------------------------ SIDEBAR --------------------------------
    st.sidebar.title("⚙️ Configuración")
    metodo_inicial = st.sidebar.selectbox("Método de Solución Inicial",
                                          list(METODOS_INICIALES))
    metodo_opt = st.sidebar.selectbox("Método de Optimización",
                                      list(METODOS_OPTIMIZACION))
    resolver = _compat(st.sidebar.button, "🚀 Resolver Ejercicio", type="primary")
    st.sidebar.markdown("---")
    st.sidebar.caption("Llena la tabla de la pantalla principal y presiona "
                       "**Resolver Ejercicio**.")

    # ----------------------------- HUD: DATOS ------------------------------
    st.title("🚚 Solucionador del Modelo de Transporte")
    st.caption("Esquina Noroeste · Costo Mínimo · Vogel · MODI · Cruce del Arroyo")

    st.subheader("📝 Ingreso de datos")
    c1, c2, c3 = st.columns([1, 1, 1])
    m = int(c1.number_input("Número de Orígenes (Oferta)", min_value=1, max_value=12,
                            value=3, step=1))
    n = int(c2.number_input("Número de Destinos (Demanda)", min_value=1, max_value=12,
                            value=3, step=1))
    if c3.button("🗑️ Limpiar tabla"):
        ss.base = tabla_base(m, n)
        ss.editada = ss.base
        ss.version += 1
        ss.dims = (m, n)
        st.rerun()

    # La tabla se reconstruye (conservando lo digitado) sólo si cambian m o n.
    if ss.get("dims") != (m, n):
        ss.base = tabla_base(m, n, ss.get("editada"))
        ss.dims = (m, n)
        ss.version += 1

    st.markdown("Escribe los **costos unitarios** en el cuerpo de la tabla, la **oferta** en "
                "la última columna y la **demanda** en la última fila "
                "(la celda inferior derecha se ignora).")
    editada = _compat(
        st.data_editor, ss.base, key=f"editor_{ss.version}", num_rows="fixed",
        column_config={"Oferta": st.column_config.NumberColumn("Oferta")},
    )
    ss.editada = editada

    with st.expander("🏷️ Nombres personalizados (opcional)"):
        k1, k2 = st.columns(2)
        origenes = []
        destinos = []
        for i in range(m):
            nombre = k1.text_input(f"Nombre del origen {i + 1}", value=f"Origen {i + 1}",
                                   key=f"nom_o_{i}")
            origenes.append(nombre.strip() or f"Origen {i + 1}")
        for j in range(n):
            nombre = k2.text_input(f"Nombre del destino {j + 1}", value=f"Destino {j + 1}",
                                   key=f"nom_d_{j}")
            destinos.append(nombre.strip() or f"Destino {j + 1}")
        if len(set(origenes)) < m or len(set(destinos)) < n:
            st.warning("Hay nombres repetidos; usa nombres distintos para evitar confusiones.")

    # Totales en vivo
    oferta_tot = float(np.nansum(editada.iloc[:m, n].to_numpy(dtype=float)))
    demanda_tot = float(np.nansum(editada.iloc[m, :n].to_numpy(dtype=float)))
    t1, t2, t3 = st.columns(3)
    t1.metric("Oferta total", fmt(oferta_tot))
    t2.metric("Demanda total", fmt(demanda_tot))
    t3.metric("Estado", "Balanceado ✅" if abs(oferta_tot - demanda_tot) < TOL
              else "Desbalanceado ⚠️ (se agregará nodo ficticio)")

    # ------------------------------ RESOLVER -------------------------------
    huella = firma(editada, origenes, destinos, metodo_inicial, metodo_opt)
    if resolver:
        try:
            costos, oferta, demanda = extraer_datos(editada, m, n)
            ss.resultado = resolver_transporte(
                costos, oferta, demanda, origenes, destinos,
                metodo_inicial=metodo_inicial, metodo_optimizacion=metodo_opt)
            ss.huella_resultado = huella
        except (ValueError, RuntimeError) as err:
            ss.resultado = None
            st.error(f"❌ {err}")

    # ----------------------------- RESULTADOS ------------------------------
    st.markdown("---")
    if ss.get("resultado") is not None:
        if ss.get("huella_resultado") != huella:
            st.warning("Los datos o los métodos cambiaron desde la última resolución. "
                       "Presiona **Resolver Ejercicio** para actualizar.")
        mostrar_resultados(ss.resultado)
    else:
        st.info("👈 Elige los métodos en el panel lateral, completa la tabla y presiona "
                "**Resolver Ejercicio**.")


if __name__ == "__main__":
    main()