
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# MAPPA Dashboard + AED
# ------------------------------------------------------------
# Painel Streamlit para o projeto MAPPA
# Consome a saída do etl_mappa_v3.py
# Entrada esperada: pasta "saida_dashboard" com os CSVs gerados
# ============================================================

st.set_page_config(
    page_title="MAPPA — Dashboard de Acidentes",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_DIR = "saida_dashboard"


# -----------------------------
# Utilidades
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_dataset(base_dir: str) -> dict[str, pd.DataFrame]:
    base = Path(base_dir)
    files = {
        "fato": "fato_ocorrencias_dashboard.csv",
        "mapa": "fato_ocorrencias_mapa.csv",
        "agg_br": "agg_por_br.csv",
        "agg_mes": "agg_por_mes.csv",
        "agg_causa": "agg_por_causa.csv",
        "agg_faixa": "agg_por_faixa_horaria.csv",
        "agg_municipio": "agg_por_municipio.csv",
        "dim_causa": "dim_causa.csv",
        "dim_tipo": "dim_tipo_acidente.csv",
        "dim_rodovia": "dim_rodovia.csv",
        "dim_municipio": "dim_municipio.csv",
    }
    data = {}
    for key, name in files.items():
        path = base / name
        if path.exists():
            data[key] = load_csv(path)
    return data


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "data_inversa" in out.columns:
        out["data_inversa"] = pd.to_datetime(out["data_inversa"], errors="coerce")
    return out


def normalize_text_filters(s: pd.Series) -> pd.Series:
    return s.fillna("NAO INFORMADO").astype(str)


def safe_sum(df: pd.DataFrame, col: str) -> float:
    return float(df[col].fillna(0).sum()) if col in df.columns else 0.0


def safe_nunique(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique(dropna=True)) if col in df.columns else 0


def build_filtered(df: pd.DataFrame, anos, brs, municipios, severidades, causas):
    out = df.copy()
    if anos and "ano" in out.columns:
        out = out[out["ano"].isin(anos)]
    if brs and "br_formatada" in out.columns:
        out = out[out["br_formatada"].isin(brs)]
    if municipios and "municipio" in out.columns:
        out = out[out["municipio"].isin(municipios)]
    if severidades and "nivel_severidade" in out.columns:
        out = out[out["nivel_severidade"].isin(severidades)]
    if causas and "causa_acidente" in out.columns:
        out = out[out["causa_acidente"].isin(causas)]
    return out


def agg_filtered(df: pd.DataFrame):
    aggs = {}
    if "br_formatada" in df.columns:
        aggs["br"] = (
            df.groupby("br_formatada", dropna=False)
            .agg(acidentes=("br_formatada", "size"),
                 mortos_total=("mortos", "sum"),
                 feridos_total=("total_feridos", "sum"))
            .reset_index()
            .sort_values(["acidentes", "mortos_total"], ascending=[False, False])
        )
    if {"ano", "mes", "mes_nome"}.issubset(df.columns):
        temp = df.copy()
        temp["ano_mes_ord"] = temp["ano"].astype("Int64").astype(str) + "-" + temp["mes"].astype("Int64").astype(str).str.zfill(2)
        aggs["mes"] = (
            temp.groupby(["ano_mes_ord", "ano", "mes", "mes_nome"], dropna=False)
            .agg(acidentes=("ano_mes_ord", "size"),
                 mortos_total=("mortos", "sum"),
                 feridos_total=("total_feridos", "sum"))
            .reset_index()
            .sort_values(["ano", "mes"])
        )
    if "causa_acidente" in df.columns:
        aggs["causa"] = (
            df.groupby("causa_acidente", dropna=False)
            .agg(acidentes=("causa_acidente", "size"),
                 mortos_total=("mortos", "sum"),
                 feridos_total=("total_feridos", "sum"))
            .reset_index()
            .sort_values(["acidentes", "mortos_total"], ascending=[False, False])
        )
    if "faixa_horaria" in df.columns:
        ordem = ["00-05", "06-11", "12-17", "18-23", "NAO INFORMADO"]
        out = (
            df.groupby("faixa_horaria", dropna=False)
            .agg(acidentes=("faixa_horaria", "size"),
                 mortos_total=("mortos", "sum"),
                 feridos_total=("total_feridos", "sum"))
            .reset_index()
        )
        out["ord"] = out["faixa_horaria"].apply(lambda x: ordem.index(x) if x in ordem else 999)
        aggs["faixa"] = out.sort_values("ord").drop(columns="ord")
    if "municipio" in df.columns:
        key_cols = ["municipio"]
        if "uf" in df.columns:
            key_cols = ["uf", "municipio"]
        aggs["municipio"] = (
            df.groupby(key_cols, dropna=False)
            .agg(acidentes=("municipio", "size"),
                 mortos_total=("mortos", "sum"),
                 feridos_total=("total_feridos", "sum"))
            .reset_index()
            .sort_values(["acidentes", "mortos_total"], ascending=[False, False])
        )
    return aggs


def generate_insights(df: pd.DataFrame, aggs: dict[str, pd.DataFrame]) -> list[str]:
    insights = []
    if len(df) == 0:
        return ["Não há dados para o recorte selecionado."]
    if "br" in aggs and not aggs["br"].empty:
        row = aggs["br"].iloc[0]
        insights.append(f"A rodovia com mais ocorrências no recorte atual é **{row['br_formatada']}**, com **{int(row['acidentes'])} acidentes**.")
    if "causa" in aggs and not aggs["causa"].empty:
        row = aggs["causa"].iloc[0]
        insights.append(f"A principal causa registrada no recorte atual é **{row['causa_acidente']}**, com **{int(row['acidentes'])} ocorrências**.")
    if "faixa" in aggs and not aggs["faixa"].empty:
        row = aggs["faixa"].sort_values("acidentes", ascending=False).iloc[0]
        insights.append(f"A faixa horária com maior concentração de acidentes é **{row['faixa_horaria']}**, com **{int(row['acidentes'])} registros**.")
    if "mes" in aggs and not aggs["mes"].empty:
        row = aggs["mes"].sort_values("acidentes", ascending=False).iloc[0]
        insights.append(f"O pico mensal do recorte ocorreu em **{int(row['mes']):02d}/{int(row['ano'])}**, com **{int(row['acidentes'])} acidentes**.")
    if "nivel_severidade" in df.columns:
        sev = df["nivel_severidade"].value_counts(dropna=False)
        if "COM MORTOS" in sev.index:
            insights.append(f"Foram identificadas **{int(sev.get('COM MORTOS', 0))} ocorrências com mortos** no recorte selecionado.")
    return insights


def to_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("MAPPA")
st.sidebar.caption("Dashboard + AED em Streamlit")

data_dir = st.sidebar.text_input("Pasta de dados", value=DEFAULT_DATA_DIR)
data = load_dataset(data_dir)

if "fato" not in data:
    st.error(
        "Não encontrei `fato_ocorrencias_dashboard.csv` na pasta informada. "
        "Rode primeiro o `etl_mappa_v3.py` e aponte para a pasta `saida_dashboard`."
    )
    st.stop()

fato = ensure_datetime(data["fato"]).copy()

# Garante tipos esperados
for col in ["ano", "mes", "mortos", "total_feridos", "feridos_leves", "feridos_graves"]:
    if col in fato.columns:
        fato[col] = pd.to_numeric(fato[col], errors="coerce")

for col in ["br_formatada", "municipio", "nivel_severidade", "causa_acidente"]:
    if col in fato.columns:
        fato[col] = normalize_text_filters(fato[col])

anos_opts = sorted([int(a) for a in fato["ano"].dropna().unique()]) if "ano" in fato.columns else []
brs_opts = sorted([x for x in fato["br_formatada"].dropna().unique().tolist()]) if "br_formatada" in fato.columns else []
mun_opts = sorted([x for x in fato["municipio"].dropna().unique().tolist()]) if "municipio" in fato.columns else []
sev_opts = sorted([x for x in fato["nivel_severidade"].dropna().unique().tolist()]) if "nivel_severidade" in fato.columns else []
causa_opts = sorted([x for x in fato["causa_acidente"].dropna().unique().tolist()]) if "causa_acidente" in fato.columns else []

st.sidebar.subheader("Filtros")
anos_sel = st.sidebar.multiselect("Ano", anos_opts, default=anos_opts)
brs_sel = st.sidebar.multiselect("Rodovia", brs_opts, default=brs_opts)
mun_sel = st.sidebar.multiselect("Município", mun_opts, default=[])
sev_sel = st.sidebar.multiselect("Severidade", sev_opts, default=sev_opts)
causa_sel = st.sidebar.multiselect("Causa", causa_opts, default=[])

filtered = build_filtered(fato, anos_sel, brs_sel, mun_sel, sev_sel, causa_sel)
aggs = agg_filtered(filtered)

st.sidebar.markdown("---")
st.sidebar.download_button(
    "Baixar recorte filtrado (CSV)",
    data=to_download_bytes(filtered),
    file_name="mappa_recorte_filtrado.csv",
    mime="text/csv",
)

# -----------------------------
# Cabeçalho
# -----------------------------
st.title("MAPPA — Dashboard Interativo de Acidentes")
st.caption(
    "Modelador Analítico de Painéis para Prevenção de Acidentes em Rodovias Federais do DF e Entorno"
)

tabs = st.tabs(["Visão Geral", "AED", "Dashboard", "Mapa", "Insights", "Sobre a Base"])

# -----------------------------
# Tab 1 - Visão Geral
# -----------------------------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Acidentes", f"{len(filtered):,}".replace(",", "."))
    c2.metric("Feridos", f"{int(safe_sum(filtered, 'total_feridos')):,}".replace(",", "."))
    c3.metric("Mortos", f"{int(safe_sum(filtered, 'mortos')):,}".replace(",", "."))
    c4.metric("Rodovias", safe_nunique(filtered, "br_formatada"))
    c5.metric("Municípios", safe_nunique(filtered, "municipio"))

    st.markdown("### Resumo do recorte")
    resumo_cols = st.columns([1.3, 1, 1])
    with resumo_cols[0]:
        st.write(
            f"O recorte atual contém **{len(filtered):,}** ocorrências, distribuídas em "
            f"**{safe_nunique(filtered, 'br_formatada')}** rodovias e "
            f"**{safe_nunique(filtered, 'municipio')}** municípios."
            .replace(",", ".")
        )
        st.write(
            "O objetivo deste painel é apoiar a leitura de padrões por rodovia, período, "
            "causa, faixa horária e severidade, alinhado à proposta do MAPPA."
        )

    with resumo_cols[1]:
        if "br" in aggs and not aggs["br"].empty:
            fig = px.bar(
                aggs["br"],
                x="br_formatada",
                y="acidentes",
                title="Acidentes por BR",
                text_auto=True,
            )
            fig.update_layout(height=360, xaxis_title="", yaxis_title="Acidentes")
            st.plotly_chart(fig, use_container_width=True)

    with resumo_cols[2]:
        if "faixa" in aggs and not aggs["faixa"].empty:
            fig = px.bar(
                aggs["faixa"],
                x="faixa_horaria",
                y="acidentes",
                title="Faixa horária",
                text_auto=True,
            )
            fig.update_layout(height=360, xaxis_title="", yaxis_title="Acidentes")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 2 - AED
# -----------------------------
with tabs[1]:
    st.subheader("Análise Exploratória de Dados (AED)")
    st.write(
        "Esta seção resume a qualidade e a distribuição do recorte filtrado, apoiando a etapa "
        "de entendimento dos dados antes da interpretação visual."
    )

    st.markdown("#### 1. Visão estrutural")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Dimensões da base filtrada**")
        st.dataframe(pd.DataFrame({
            "Métrica": ["Linhas", "Colunas", "Rodovias únicas", "Municípios únicos", "Causas únicas"],
            "Valor": [
                len(filtered),
                filtered.shape[1],
                safe_nunique(filtered, "br_formatada"),
                safe_nunique(filtered, "municipio"),
                safe_nunique(filtered, "causa_acidente"),
            ]
        }), hide_index=True, use_container_width=True)
    with c2:
        st.write("**Valores ausentes por coluna (top 12)**")
        nulos = filtered.isna().sum().sort_values(ascending=False).head(12).reset_index()
        nulos.columns = ["coluna", "nulos"]
        st.dataframe(nulos, hide_index=True, use_container_width=True)

    st.markdown("#### 2. Distribuições principais")
    c3, c4 = st.columns(2)
    with c3:
        if "nivel_severidade" in filtered.columns:
            sev = filtered["nivel_severidade"].value_counts(dropna=False).reset_index()
            sev.columns = ["nivel_severidade", "acidentes"]
            fig = px.pie(sev, names="nivel_severidade", values="acidentes", title="Distribuição por severidade")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "tipo_acidente" in filtered.columns:
            top_tipo = (
                filtered["tipo_acidente"]
                .value_counts(dropna=False)
                .head(10)
                .reset_index()
            )
            top_tipo.columns = ["tipo_acidente", "acidentes"]
            fig = px.bar(
                top_tipo.sort_values("acidentes"),
                x="acidentes", y="tipo_acidente", orientation="h",
                title="Top 10 tipos de acidente"
            )
            fig.update_layout(height=380, xaxis_title="Acidentes", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 3. Amostra da base final")
    st.dataframe(filtered.head(50), use_container_width=True)

# -----------------------------
# Tab 3 - Dashboard
# -----------------------------
with tabs[2]:
    st.subheader("Painel analítico")
    row1 = st.columns(2)
    with row1[0]:
        if "mes" in aggs and not aggs["mes"].empty:
            mes = aggs["mes"].copy()
            mes["label"] = mes["ano"].astype(int).astype(str) + "-" + mes["mes"].astype(int).astype(str).str.zfill(2)
            fig = px.line(
                mes,
                x="label",
                y="acidentes",
                markers=True,
                title="Evolução mensal de acidentes",
            )
            fig.update_layout(height=380, xaxis_title="Ano-Mês", yaxis_title="Acidentes")
            st.plotly_chart(fig, use_container_width=True)
    with row1[1]:
        if "causa" in aggs and not aggs["causa"].empty:
            top_causa = aggs["causa"].head(10).sort_values("acidentes")
            fig = px.bar(
                top_causa,
                x="acidentes",
                y="causa_acidente",
                orientation="h",
                title="Top 10 causas de acidente",
                text_auto=True,
            )
            fig.update_layout(height=380, xaxis_title="Acidentes", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    row2 = st.columns(2)
    with row2[0]:
        if "br" in aggs and not aggs["br"].empty:
            fig = px.bar(
                aggs["br"],
                x="br_formatada",
                y=["acidentes", "mortos_total"],
                barmode="group",
                title="Acidentes e mortos por BR",
            )
            fig.update_layout(height=380, xaxis_title="", yaxis_title="Quantidade")
            st.plotly_chart(fig, use_container_width=True)
    with row2[1]:
        if "municipio" in aggs and not aggs["municipio"].empty:
            top_mun = aggs["municipio"].head(10).copy()
            if "uf" in top_mun.columns:
                top_mun["municipio_label"] = top_mun["municipio"] + " / " + top_mun["uf"]
                ycol = "municipio_label"
            else:
                ycol = "municipio"
            fig = px.bar(
                top_mun.sort_values("acidentes"),
                x="acidentes",
                y=ycol,
                orientation="h",
                title="Top 10 municípios com mais ocorrências",
                text_auto=True,
            )
            fig.update_layout(height=380, xaxis_title="Acidentes", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Tab 4 - Mapa
# -----------------------------
with tabs[3]:
    st.subheader("Mapa de ocorrências")
    mapa_df = filtered.copy()
    if {"latitude", "longitude"}.issubset(mapa_df.columns):
        mapa_df = mapa_df.dropna(subset=["latitude", "longitude"]).copy()
        mapa_df["latitude"] = pd.to_numeric(mapa_df["latitude"], errors="coerce")
        mapa_df["longitude"] = pd.to_numeric(mapa_df["longitude"], errors="coerce")
        mapa_df = mapa_df.dropna(subset=["latitude", "longitude"])

        # Limpeza de coordenadas inválidas para evitar mapa em branco
        mapa_df = mapa_df[
            mapa_df["latitude"].between(-90, 90) & mapa_df["longitude"].between(-180, 180)
        ].copy()

        if len(mapa_df) == 0:
            st.warning("Não há coordenadas válidas disponíveis no recorte atual.")
        else:
            st.write(f"Registros georreferenciados no recorte: **{len(mapa_df):,}**".replace(",", "."))

            center_lat = float(mapa_df["latitude"].median())
            center_lon = float(mapa_df["longitude"].median())

            # Amostra para manter boa performance no browser
            mapa_plot = mapa_df.copy()
            max_points = 8000
            if len(mapa_plot) > max_points:
                mapa_plot = mapa_plot.sample(max_points, random_state=42)
                st.caption(f"Exibindo uma amostra de {max_points:,} pontos no mapa para melhorar a performance.".replace(",", "."))

            hover_cols = [c for c in ["data_inversa", "br_formatada", "municipio", "causa_acidente", "tipo_acidente", "nivel_severidade", "mortos", "total_feridos"] if c in mapa_plot.columns]

            fig_map = px.scatter_map(
                mapa_plot,
                lat="latitude",
                lon="longitude",
                color="nivel_severidade" if "nivel_severidade" in mapa_plot.columns else None,
                hover_data=hover_cols,
                zoom=6,
                height=620,
                center={"lat": center_lat, "lon": center_lon},
                title="Distribuição geográfica das ocorrências"
            )
            fig_map.update_layout(
                map_style="open-street-map",
                margin={"l": 0, "r": 0, "t": 50, "b": 0},
                legend_title_text="Severidade"
            )
            st.plotly_chart(fig_map, use_container_width=True)

            with st.expander("Ver amostra dos pontos geográficos"):
                cols = [c for c in ["data_inversa", "br_formatada", "municipio", "causa_acidente", "nivel_severidade", "latitude", "longitude"] if c in mapa_df.columns]
                st.dataframe(mapa_df[cols].head(100), use_container_width=True)
    else:
        st.info("A base atual não possui latitude/longitude suficientes para mapa.")

# -----------------------------
# Tab 5 - Insights
# -----------------------------
with tabs[4]:
    st.subheader("Insights automáticos do recorte")
    insights = generate_insights(filtered, aggs)
    for item in insights:
        st.markdown(f"- {item}")

    st.markdown("#### Recomendações de leitura")
    st.write(
        "Use esses insights como ponto de partida. A interpretação final deve considerar "
        "limitações do recorte, parcialidade do ano em curso e qualidade das informações de origem."
    )

# -----------------------------
# Tab 6 - Sobre a base
# -----------------------------
with tabs[5]:
    st.subheader("Sobre a base")
    st.write(
        "Este painel consome a saída do `etl_mappa_v3.py`, que organiza dados da PRF em CSV "
        "para AED, visualização e dashboard, sem depender de autenticação ou banco online."
    )
    st.markdown("**Arquivos detectados**")
    st.write(sorted(list(data.keys())))

    st.markdown("**Dicionário resumido usado no painel**")
    dicionario = pd.DataFrame({
        "Campo": [
            "data_inversa", "ano", "mes", "br_formatada", "municipio",
            "causa_acidente", "tipo_acidente", "faixa_horaria",
            "total_feridos", "mortos", "nivel_severidade"
        ],
        "Descrição": [
            "Data da ocorrência",
            "Ano da ocorrência",
            "Mês da ocorrência",
            "Rodovia no formato BR-XXX",
            "Município associado ao registro",
            "Causa principal do acidente",
            "Tipo de acidente/sinistro",
            "Faixa horária derivada do horário",
            "Soma de feridos leves e graves",
            "Total de mortos na ocorrência",
            "Classificação derivada: sem vítimas, com feridos ou com mortos",
        ]
    })
    st.dataframe(dicionario, hide_index=True, use_container_width=True)

    st.markdown("**Limitações atuais**")
    st.write(
        "- A base de pessoas envolvidas só deve ser usada quando houver fonte real por pessoa.\n"
        "- O ano de 2026 pode estar parcial.\n"
        "- As análises dependem da completude e padronização dos CSVs publicados pela PRF."
    )
