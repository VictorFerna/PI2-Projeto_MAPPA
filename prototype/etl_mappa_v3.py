from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# ============================================================
# MAPPA ETL v3
# Base final em CSV para AED, visualização e dashboard
# ------------------------------------------------------------
# - Lê múltiplos CSVs da PRF (anos diferentes)
# - Padroniza colunas e valores
# - Filtra o recorte do projeto (DF/Entorno, BRs alvo)
# - Gera fato principal, base de mapa, dimensões e agregados
# - Não depende de login, senha ou banco de dados
# ============================================================

# BRs definidas no escopo atual do projeto.
DEFAULT_BRS = {"020", "040", "060", "070", "080"}
# Para evitar perder o entorno, por padrão considera DF, GO e MG.
# Se quiser restringir, passe: --ufs DF
DEFAULT_UFS = {"DF", "GO", "MG"}


def log(msg: str) -> None:
    print(msg, flush=True)


def normalizar_texto(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode("utf-8")
    txt = txt.lower().strip()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    return txt.strip("_")


def detectar_sep(path: Path) -> str:
    with path.open("r", encoding="latin1", errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[";", ",", "\t"])
        return dialect.delimiter
    except Exception:
        return ";" if sample.count(";") >= sample.count(",") else ","


def ler_csv(path: Path) -> pd.DataFrame:
    sep = detectar_sep(path)
    try:
        return pd.read_csv(path, sep=sep, encoding="latin1", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalizar_texto(c) for c in df.columns]

    aliases = {
        "id_ocorrencia": "id",
        "data": "data_inversa",
        "brasil": "br",
        "latitude_decimal": "latitude",
        "longitude_decimal": "longitude",
        "causa_principal": "causa_acidente",
        "tipo_sinistro": "tipo_acidente",
        "meteorologia": "condicao_metereologica",
        "condicao_meteorologica": "condicao_metereologica",
        "tipo_pessoa": "tipo_envolvido",
    }
    rename_map = {c: aliases[c] for c in df.columns if c in aliases}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def extrair_ano_do_arquivo(nome: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", nome)
    return int(m.group(1)) if m else None


def extrair_br(valor: object) -> Optional[str]:
    if pd.isna(valor):
        return None
    s = str(valor)
    m = re.search(r"(\d{2,3})", s)
    if not m:
        return None
    return m.group(1).zfill(3)


def limpar_string_serie(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .str.strip()
        .str.upper()
        .fillna(pd.NA)
    )


def to_num(serie: pd.Series) -> pd.Series:
    s = serie.astype("string").str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_data_robusta(serie: pd.Series) -> pd.Series:
    s = serie.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")

    formatos = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
    ]

    restantes = s.copy()
    for fmt in formatos:
        parsed = pd.to_datetime(restantes, format=fmt, errors="coerce")
        mask = parsed.notna() & out.isna()
        out.loc[mask] = parsed.loc[mask]
        restantes = restantes.loc[out.isna()]
        if restantes.empty:
            break

    if out.isna().any():
        fallback = pd.to_datetime(s.loc[out.isna()], errors="coerce", dayfirst=True)
        out.loc[out.isna()] = fallback

    return out


def faixa_horaria(horario: object) -> str:
    if pd.isna(horario):
        return "NAO INFORMADO"
    s = str(horario).strip()
    m = re.match(r"(\d{1,2})", s)
    if not m:
        return "NAO INFORMADO"
    hora = int(m.group(1))
    if 0 <= hora < 6:
        return "00-05"
    if 6 <= hora < 12:
        return "06-11"
    if 12 <= hora < 18:
        return "12-17"
    if 18 <= hora <= 23:
        return "18-23"
    return "NAO INFORMADO"


def mes_nome(m: float) -> Optional[str]:
    nomes = {
        1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR", 5: "MAI", 6: "JUN",
        7: "JUL", 8: "AGO", 9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
    }
    try:
        return nomes.get(int(m))
    except Exception:
        return None


def normalizar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strings principais
    for col in [
        "uf",
        "municipio",
        "causa_acidente",
        "tipo_acidente",
        "fase_dia",
        "condicao_metereologica",
        "tipo_pista",
        "dia_semana",
        "tracado_via",
        "uso_solo",
        "tipo_envolvido",
        "estado_fisico",
        "sexo",
        "tipo_veiculo",
    ]:
        if col in df.columns:
            df[col] = limpar_string_serie(df[col])

    # Nulos de causa
    if "causa_acidente" in df.columns:
        df["causa_acidente"] = df["causa_acidente"].replace({
            "": pd.NA,
            "NAN": pd.NA,
            "NONE": pd.NA,
        }).fillna("NAO INFORMADO")

    # Numéricos
    for col in [
        "latitude", "longitude", "km", "ilesos", "feridos_leves", "feridos_graves", "mortos", "idade"
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])

    # BR e UF
    if "br" in df.columns:
        df["br_num"] = df["br"].apply(extrair_br)
        df["br_formatada"] = df["br_num"].apply(lambda x: f"BR-{x}" if pd.notna(x) else pd.NA)
    else:
        df["br_num"] = pd.NA
        df["br_formatada"] = pd.NA

    # Datas
    if "data_inversa" in df.columns:
        df["data_inversa"] = parse_data_robusta(df["data_inversa"])
        df["ano"] = df["data_inversa"].dt.year
        df["mes"] = df["data_inversa"].dt.month
        df["mes_nome"] = df["mes"].apply(mes_nome)
        df["trimestre"] = df["data_inversa"].dt.quarter
        df["ano_mes"] = df["data_inversa"].dt.to_period("M").astype("string")
    else:
        df["ano"] = pd.NA
        df["mes"] = pd.NA
        df["mes_nome"] = pd.NA
        df["trimestre"] = pd.NA
        df["ano_mes"] = pd.NA

    # Horário
    if "horario" in df.columns:
        df["faixa_horaria"] = df["horario"].apply(faixa_horaria)
    else:
        df["faixa_horaria"] = "NAO INFORMADO"

    # Severidade
    for col in ["ilesos", "feridos_leves", "feridos_graves", "mortos"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df["total_feridos"] = df["feridos_leves"] + df["feridos_graves"]
    df["houve_feridos"] = (df["total_feridos"] > 0).astype(int)
    df["houve_mortos"] = (df["mortos"] > 0).astype(int)

    def classificar_severidade(row: pd.Series) -> str:
        if row["houve_mortos"] == 1:
            return "COM MORTOS"
        if row["houve_feridos"] == 1:
            return "COM FERIDOS"
        return "SEM VITIMAS"

    df["nivel_severidade"] = df.apply(classificar_severidade, axis=1)

    return df


def recortar_projeto(df: pd.DataFrame, ufs_alvo: Iterable[str], brs_alvo: Iterable[str]) -> pd.DataFrame:
    df = df.copy()

    if "uf" in df.columns:
        ufs_norm = {str(u).strip().upper() for u in ufs_alvo}
        df = df[df["uf"].isin(ufs_norm)]

    if "br_num" in df.columns:
        brs_norm = {str(b).zfill(3) for b in brs_alvo}
        df = df[df["br_num"].isin(brs_norm)]

    # Remove duplicatas por ocorrência quando houver id
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    else:
        campos = [c for c in ["data_inversa", "horario", "uf", "br_num", "municipio", "km"] if c in df.columns]
        if campos:
            df = df.drop_duplicates(subset=campos, keep="first")

    return df.reset_index(drop=True)


def exportar_csv(df: pd.DataFrame, path: Path, nome: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[OK] {nome}: {path} ({len(df)} linhas)")


def agregar_ocorrencias(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aggs: dict[str, pd.DataFrame] = {}

    aggs["agg_por_br"] = (
        df.groupby(["br_formatada"], dropna=False)
        .agg(
            acidentes=("br_formatada", "size"),
            mortos_total=("mortos", "sum"),
            feridos_total=("total_feridos", "sum"),
        )
        .reset_index()
        .sort_values(["acidentes", "mortos_total"], ascending=[False, False])
    )

    aggs["agg_por_mes"] = (
        df.groupby(["ano_mes", "ano", "mes", "mes_nome"], dropna=False)
        .agg(
            acidentes=("ano_mes", "size"),
            mortos_total=("mortos", "sum"),
            feridos_total=("total_feridos", "sum"),
        )
        .reset_index()
        .sort_values(["ano", "mes"])
    )

    aggs["agg_por_causa"] = (
        df.groupby(["causa_acidente"], dropna=False)
        .agg(
            acidentes=("causa_acidente", "size"),
            mortos_total=("mortos", "sum"),
            feridos_total=("total_feridos", "sum"),
        )
        .reset_index()
        .sort_values(["acidentes", "mortos_total"], ascending=[False, False])
    )

    aggs["agg_por_faixa_horaria"] = (
        df.groupby(["faixa_horaria"], dropna=False)
        .agg(
            acidentes=("faixa_horaria", "size"),
            mortos_total=("mortos", "sum"),
            feridos_total=("total_feridos", "sum"),
        )
        .reset_index()
        .sort_values("faixa_horaria")
    )

    aggs["agg_por_municipio"] = (
        df.groupby(["uf", "municipio"], dropna=False)
        .agg(
            acidentes=("municipio", "size"),
            mortos_total=("mortos", "sum"),
            feridos_total=("total_feridos", "sum"),
        )
        .reset_index()
        .sort_values(["acidentes", "mortos_total"], ascending=[False, False])
    )

    return aggs


def gerar_dimensoes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dims: dict[str, pd.DataFrame] = {}

    dims["dim_causa"] = (
        df[["causa_acidente"]]
        .dropna()
        .drop_duplicates()
        .sort_values("causa_acidente")
        .reset_index(drop=True)
    )

    dims["dim_tipo_acidente"] = (
        df[["tipo_acidente"]]
        .dropna()
        .drop_duplicates()
        .sort_values("tipo_acidente")
        .reset_index(drop=True)
    )

    dims["dim_rodovia"] = (
        df[["uf", "br_num", "br_formatada"]]
        .dropna(subset=["br_num"])
        .drop_duplicates()
        .sort_values(["uf", "br_num"])
        .reset_index(drop=True)
    )

    dims["dim_municipio"] = (
        df[["uf", "municipio"]]
        .dropna(subset=["municipio"])
        .drop_duplicates()
        .sort_values(["uf", "municipio"])
        .reset_index(drop=True)
    )

    return dims


def gerar_fato_pessoas(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    colunas_pessoa = ["tipo_envolvido", "estado_fisico", "sexo", "idade", "tipo_veiculo"]
    disponiveis = [c for c in colunas_pessoa if c in df.columns]
    if not disponiveis:
        return None

    base_cols = [c for c in ["id", "data_inversa", "ano", "mes", "uf", "br_formatada", "municipio", "tipo_acidente"] if c in df.columns]
    cols = base_cols + disponiveis
    pessoas = df[cols].copy()

    # Se vier de base de ocorrência, isso tende a não representar pessoas reais.
    # Exportamos apenas quando existe alguma informação de pessoa preenchida.
    mask_util = pd.Series(False, index=pessoas.index)
    for c in disponiveis:
        mask_util = mask_util | pessoas[c].notna()
    pessoas = pessoas[mask_util].drop_duplicates().reset_index(drop=True)

    return pessoas if not pessoas.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL do MAPPA para AED e dashboard")
    parser.add_argument("--input", default="dados_prf", help="Pasta com CSVs da PRF")
    parser.add_argument("--output", default="saida_dashboard", help="Pasta de saída")
    parser.add_argument("--ufs", nargs="*", default=sorted(DEFAULT_UFS), help="UFs do recorte (ex.: DF GO MG)")
    parser.add_argument("--brs", nargs="*", default=sorted(DEFAULT_BRS), help="BRs do recorte (ex.: 020 040 060 070 080)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    log("=== MAPPA ETL v3 ===")
    log(f"Lendo arquivos em: {input_dir.resolve()}")

    arquivos = sorted(input_dir.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {input_dir.resolve()}")

    log(f"Arquivos encontrados: {len(arquivos)}")
    for arq in arquivos:
        log(f" - {arq.name}")

    dfs: List[pd.DataFrame] = []
    for arq in arquivos:
        df = ler_csv(arq)
        df = padronizar_colunas(df)
        df["arquivo_origem"] = arq.name
        df["ano_origem"] = extrair_ano_do_arquivo(arq.name)
        dfs.append(df)

    bruto = pd.concat(dfs, ignore_index=True, sort=False)
    log(f"\nBase bruta concatenada: {len(bruto)} linhas x {bruto.shape[1]} colunas")

    bruto = normalizar_dataset(bruto)
    df = recortar_projeto(bruto, args.ufs, args.brs)

    log(f"Base tratada (recorte do projeto): {len(df)} linhas x {df.shape[1]} colunas")
    if "data_inversa" in df.columns:
        log(f"Datas inválidas (NaT): {int(df['data_inversa'].isna().sum())}")
    if "municipio" in df.columns:
        log(f"Municípios únicos: {df['municipio'].nunique(dropna=True)}")
        try:
            log("Top municípios:")
            top_mun = df["municipio"].value_counts(dropna=False).head(10)
            for idx, val in top_mun.items():
                log(f"   - {idx}: {val}")
        except Exception:
            pass
    if "ano" in df.columns:
        anos = sorted([int(a) for a in df["ano"].dropna().unique()])
        log(f"Anos presentes após recorte: {anos}")

    # Exporta fato principal
    exportar_csv(df, output_dir / "fato_ocorrencias_dashboard.csv", "fato_ocorrencias_dashboard")

    # Exporta fato de mapa (somente registros com coordenadas)
    if "latitude" in df.columns and "longitude" in df.columns:
        mapa = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        exportar_csv(mapa, output_dir / "fato_ocorrencias_mapa.csv", "fato_ocorrencias_mapa")

    # Exporta fato de pessoas só se houver base real/complementar
    pessoas = gerar_fato_pessoas(df)
    if pessoas is not None and len(pessoas) > 0:
        exportar_csv(pessoas, output_dir / "fato_pessoas_dashboard.csv", "fato_pessoas_dashboard")
    else:
        log("[INFO] fato_pessoas_dashboard não foi gerado: faltam colunas reais de pessoa ou não há dados úteis.")

    # Dimensões
    dims = gerar_dimensoes(df)
    for nome, dim in dims.items():
        exportar_csv(dim, output_dir / f"{nome}.csv", nome)

    # Agregados
    aggs = agregar_ocorrencias(df)
    for nome, agg in aggs.items():
        exportar_csv(agg, output_dir / f"{nome}.csv", nome)

    log("\nConcluído. Base pronta para AED, visualização e dashboard.")


if __name__ == "__main__":
    main()
