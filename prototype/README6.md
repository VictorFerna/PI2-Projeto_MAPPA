# MAPPA — Dashboard Streamlit

## Como executar

1. Gere os CSVs com o `etl_mappa_v3.py`
2. Garanta a pasta `saida_dashboard/` ao lado do arquivo `app_mappa_streamlit.py`
3. Instale os requisitos:

```bash
pip install -r requerimentos_dashboard.txt
```

4. Rode:

```bash
streamlit run app_mappa_streamlit_v2.py
```

## Arquivos esperados na pasta `saida_dashboard`

- fato_ocorrencias_dashboard.csv
- fato_ocorrencias_mapa.csv
- agg_por_br.csv
- agg_por_mes.csv
- agg_por_causa.csv
- agg_por_faixa_horaria.csv
- agg_por_municipio.csv
- dim_causa.csv
- dim_tipo_acidente.csv
- dim_rodovia.csv
- dim_municipio.csv
