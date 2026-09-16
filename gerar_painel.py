
"""
Gera o painel HTML do projeto a partir de dados_clusters.json.

O painel é autocontido: os gráficos são SVG gerados em Python e não dependem
 de bibliotecas JavaScript externas.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

PASTA_DADOS = Path("DB/dados_tratados")
CORES = ["#1F3A5F", "#B23A2E", "#2E7D6B", "#C98A2C", "#6B4C9A", "#3D8FB5"]
INK = "#16232E"
MUTED = "#5C6B7A"
LINE = "#D9E0E5"
SOFT = "#F1DCD8"


def fmt(v, casas=1):
    return f"{float(v):.{casas}f}".replace(".", ",")


def fmt_inteiro(v):
    return f"{int(v):,}".replace(",", ".")


def fmt_compacto(v):
    v = float(v)
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.1f}".replace(".", ",") + " mi"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.0f}" + " mil"
    return f"{v:.0f}"


def fmt_numero(v, casas=2):
    return f"{float(v):.{casas}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# SVG


def barras_metricas_svg(categorias, valores, cor, largura=520, altura=None, formatador=fmt_compacto):
    """Barras horizontais para comparar uma métrica discreta por valor de k.

    Usado na seleção do número de clusters. Não há linhas de tendência: cada
    valor de k é representado por uma barra independente.
    """
    n = max(1, len(categorias))
    altura = altura or max(250, 46 * n + 55)
    m = {"l": 72, "r": 70, "t": 18, "b": 30}
    pw = largura - m["l"] - m["r"]
    ph = altura - m["t"] - m["b"]
    passo = ph / n
    maxv = max(max((float(v) for v in valores), default=0.0), 0.0)
    vmax = maxv * 1.08 if maxv else 1.0

    partes = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = m["l"] + pw * frac
        valor = vmax * frac
        partes.append(
            f'<line x1="{x:.1f}" y1="{m["t"]}" x2="{x:.1f}" y2="{altura-m["b"]}" '
            f'stroke="{LINE}"/>'
        )
        partes.append(
            f'<text x="{x:.1f}" y="{altura-9}" font-size="10" fill="{MUTED}" '
            f'text-anchor="middle">{html.escape(formatador(valor))}</text>'
        )

    for i, (cat, val) in enumerate(zip(categorias, valores)):
        y = m["t"] + i * passo + passo * 0.20
        bh = passo * 0.50
        bw = pw * (float(val) / vmax) if vmax else 0
        partes.append(
            f'<text x="{m["l"]-12}" y="{y+bh*0.76:.1f}" font-size="11" '
            f'fill="{INK}" text-anchor="end">k = {html.escape(str(cat))}</text>'
        )
        partes.append(
            f'<rect x="{m["l"]}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'rx="3" fill="{cor}"/>'
        )
        partes.append(
            f'<text x="{m["l"]+bw+8:.1f}" y="{y+bh*0.76:.1f}" font-size="11" '
            f'fill="{INK}" font-weight="600">{html.escape(formatador(float(val)))}</text>'
        )

    partes.append("</svg>")
    return "".join(partes)



def barras_horizontais_svg(categorias, valores, cores, largura=760, altura=None, titulo_eixo="Percentual"):
    """Gráfico categórico horizontal; cada categoria tem sua própria linha."""
    n = max(1, len(categorias))
    altura = altura or max(220, 58 * n + 50)
    m = {"l": 190, "r": 70, "t": 20, "b": 28}
    pw, ph = largura - m["l"] - m["r"], altura - m["t"] - m["b"]
    passo = ph / n
    maxv = max(max(valores, default=0), 1)
    vmax = 100 if maxv <= 100 else maxv * 1.1

    partes = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = m["l"] + pw * frac
        valor = vmax * frac
        partes.append(f'<line x1="{x:.1f}" y1="{m["t"]}" x2="{x:.1f}" y2="{altura-m["b"]}" stroke="{LINE}"/>')
        partes.append(f'<text x="{x:.1f}" y="{altura-8}" font-size="10" fill="{MUTED}" text-anchor="middle">{fmt(valor, 0)}%</text>')

    for i, (cat, val) in enumerate(zip(categorias, valores)):
        y = m["t"] + i * passo + passo * 0.22
        bh = passo * 0.48
        bw = pw * (float(val) / vmax)
        cor = cores[i % len(cores)]
        partes.append(f'<text x="{m["l"]-12}" y="{y+bh*0.78:.1f}" font-size="11" fill="{INK}" text-anchor="end">{html.escape(str(cat))}</text>')
        partes.append(f'<rect x="{m["l"]}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{cor}"/>')
        partes.append(f'<text x="{m["l"]+bw+8:.1f}" y="{y+bh*0.78:.1f}" font-size="11" fill="{INK}" font-weight="600">{fmt(val)}%</text>')

    partes.append(f'<text x="{m["l"]+pw/2:.1f}" y="{altura-8}" font-size="0" fill="{MUTED}">{html.escape(titulo_eixo)}</text>')
    partes.append("</svg>")
    return "".join(partes)


def prevalencia_cluster_svg(perfis, media, largura=760, altura=300):
    categorias = [f"Cluster {p['cluster']}" for p in perfis]
    valores = [p["prevalencia_doenca_cardiaca_pct"] for p in perfis]
    cores = [CORES[p["cluster"] % len(CORES)] for p in perfis]
    maxv = max(max(valores, default=0), media) * 1.25 or 1
    m = {"l": 150, "r": 90, "t": 20, "b": 40}
    pw = largura - m["l"] - m["r"]
    passo = (altura - m["t"] - m["b"]) / max(1, len(categorias))
    partes = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    x_media = m["l"] + pw * media / maxv
    partes.append(f'<line x1="{x_media:.1f}" y1="{m["t"]}" x2="{x_media:.1f}" y2="{altura-m["b"]}" stroke="{MUTED}" stroke-dasharray="5,4" stroke-width="1.4"/>')
    partes.append(f'<text x="{x_media:.1f}" y="{m["t"]-5}" font-size="10" fill="{MUTED}" text-anchor="middle">média {fmt(media)}%</text>')
    for i, (cat, val, cor) in enumerate(zip(categorias, valores, cores)):
        y = m["t"] + i * passo + passo * 0.18
        bh = passo * 0.58
        bw = pw * val / maxv
        partes.append(f'<text x="{m["l"]-12}" y="{y+bh*0.72:.1f}" font-size="11" fill="{INK}" text-anchor="end">{cat}</text>')
        partes.append(f'<rect x="{m["l"]}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{cor}"/>')
        partes.append(f'<text x="{m["l"]+bw+8:.1f}" y="{y+bh*0.72:.1f}" font-size="11" fill="{INK}" font-weight="600">{fmt(val)}%</text>')
    partes.append("</svg>")
    return "".join(partes)


def dispersao_svg(pontos, cores, largura=760, altura=380):
    m = {"l": 44, "r": 20, "t": 16, "b": 34}
    pw, ph = largura - m["l"] - m["r"], altura - m["t"] - m["b"]
    xs = [p["x"] for p in pontos] or [0]
    ys = [p["y"] for p in pontos] or [0]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    fx, fy = (xmax - xmin) * 0.06 or 1, (ymax - ymin) * 0.06 or 1
    xmin, xmax, ymin, ymax = xmin - fx, xmax + fx, ymin - fy, ymax + fy

    def px(v): return m["l"] + (v - xmin) / (xmax - xmin) * pw
    def py(v): return m["t"] + ph - (v - ymin) / (ymax - ymin) * ph

    partes = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    partes.append(f'<line x1="{m["l"]}" y1="{m["t"]}" x2="{m["l"]}" y2="{m["t"]+ph}" stroke="{LINE}"/>')
    partes.append(f'<line x1="{m["l"]}" y1="{m["t"]+ph}" x2="{m["l"]+pw}" y2="{m["t"]+ph}" stroke="{LINE}"/>')
    partes.append(f'<text x="{m["l"]+pw/2}" y="{altura-6}" font-size="11" fill="{MUTED}" text-anchor="middle">Componente 1</text>')
    partes.append(f'<text x="14" y="{m["t"]+ph/2}" font-size="11" fill="{MUTED}" text-anchor="middle" transform="rotate(-90 14 {m["t"]+ph/2})">Componente 2</text>')
    for p in pontos:
        cor = cores[p["cluster"] % len(cores)]
        partes.append(f'<circle cx="{px(p["x"]):.1f}" cy="{py(p["y"]):.1f}" r="2.6" fill="{cor}" fill-opacity="0.55"/>')
    partes.append("</svg>")
    return "".join(partes)


# ---------------------------------------------------------------------------
# HTML helpers


def card_perfil(p: dict) -> str:
    cor = CORES[p["cluster"] % len(CORES)]
    return f"""
    <article class="profile-card" style="border-left-color:{cor}">
      <div>
        <div class="profile-name">Cluster {p['cluster']}</div>
        <div class="muted">{fmt_inteiro(p['tamanho'])} registros · {fmt(p['percentual_do_total'])}% da base</div>
      </div>
      <div class="trait-grid">
        <div><strong>{html.escape(p['diabetes_predominante'])}</strong><span>diabetes predominante</span></div>
        <div><strong>{html.escape(p['faixa_etaria_predominante'])}</strong><span>faixa etária predominante</span></div>
        <div><strong>{fmt(p['bmi_medio'])}</strong><span>IMC médio</span></div>
        <div><strong>{fmt(p['pct_pressao_alta'])}%</strong><span>pressão alta</span></div>
        <div><strong>{fmt(p['pct_colesterol_alto'])}%</strong><span>colesterol alto</span></div>
        <div><strong>{fmt(p['pct_fumante'])}%</strong><span>fumantes</span></div>
        <div><strong>{fmt(p['pct_dificuldade_caminhar'])}%</strong><span>dificuldade para caminhar</span></div>
        <div><strong>{fmt(p['pct_atividade_fisica'])}%</strong><span>atividade física</span></div>
        <div><strong>{html.escape(p['saude_geral_predominante'])}</strong><span>saúde geral predominante</span></div>
      </div>
      <div class="risk"><strong>{fmt(p['prevalencia_doenca_cardiaca_pct'])}%</strong><span>doença cardíaca observada</span></div>
    </article>"""


def tabela_associacoes(registros: list[dict]) -> str:
    linhas = []
    for r in registros:
        sinal = "+" if r["diferenca_pontos_percentuais"] >= 0 else ""
        linhas.append(
            f"<tr><td>{html.escape(r['nome'])}</td>"
            f"<td>{fmt(r['prevalencia_com_indicador_pct'])}%</td>"
            f"<td>{fmt(r['prevalencia_sem_indicador_pct'])}%</td>"
            f"<td>{sinal}{fmt(r['diferenca_pontos_percentuais'])} p.p.</td>"
            f"<td>{fmt_inteiro(r['n_com_indicador'])}</td><td>{fmt_inteiro(r['n_sem_indicador'])}</td></tr>"
        )
    return "".join(linhas)


def tabela_variaveis(variaveis: dict) -> str:
    linhas = []
    for coluna, info in variaveis.items():
        linhas.append(
            f"<tr><td><code>{html.escape(coluna)}</code></td>"
            f"<td>{html.escape(str(info.get('nome', coluna)))}</td>"
            f"<td>{html.escape(str(info.get('tipo', '—')))}</td>"
            f"<td>{html.escape(str(info.get('papel', '—')))}</td>"
            f"<td>{html.escape(str(info.get('descricao', '—')))}</td></tr>"
        )
    return "".join(linhas)


def card_indicador(nome: str, valor: str, legenda: str) -> str:
    return f'<div class="kpi"><div class="kpi-value">{valor}</div><div class="kpi-label">{html.escape(nome)}</div><div class="kpi-note">{html.escape(legenda)}</div></div>'


# ---------------------------------------------------------------------------
# Construção


def main() -> None:
    caminho_json = PASTA_DADOS / "dados_clusters.json"
    if not caminho_json.exists():
        raise FileNotFoundError(
            f"{caminho_json} não encontrado. Execute analise_clusters.py primeiro."
        )

    dados = json.loads(caminho_json.read_text(encoding="utf-8"))

    # Dados do painel
    perfis = sorted(dados["perfis"], key=lambda p: p["prevalencia_doenca_cardiaca_pct"])
    resumo = dados.get("resumo_base", {})
    fonte = dados.get("fonte", {})
    assoc = dados.get("associacoes_descritivas", [])
    interpretacao = dados.get("interpretacao", {})

    diabetes = resumo.get("diabetes", [])
    saude = resumo.get("saude_geral", [])
    idade = resumo.get("faixa_etaria", [])
    sexo = resumo.get("sexo", [])
    indicadores = resumo.get("indicadores_binarios", [])

    ks = [r["k"] for r in dados["varredura_k"]]
    inercias = [r["inercia"] for r in dados["varredura_k"]]
    silhuetas = [r["silhueta"] for r in dados["varredura_k"]]

    svg_cotovelo = barras_metricas_svg(ks, inercias, INK, formatador=fmt_compacto)
    svg_silhueta = barras_metricas_svg(ks, silhuetas, "#B23A2E", formatador=lambda v: fmt_numero(v, 2))
    svg_dispersao = dispersao_svg(dados["pontos_dispersao"], CORES)
    svg_prev = prevalencia_cluster_svg(perfis, dados["prevalencia_geral_doenca_cardiaca_pct"])

    svg_diabetes = barras_horizontais_svg(
        [x["categoria"] for x in diabetes], [x["percentual"] for x in diabetes], CORES,
        titulo_eixo="Distribuição dos respondentes",
    ) if diabetes else ""
    svg_saude = barras_horizontais_svg(
        [x["categoria"] for x in saude], [x["percentual"] for x in saude], CORES,
        titulo_eixo="Distribuição dos respondentes",
    ) if saude else ""
    svg_idade = barras_horizontais_svg(
        [x["categoria"] for x in idade], [x["percentual"] for x in idade], CORES,
        titulo_eixo="Distribuição dos respondentes",
    ) if idade else ""
    svg_sexo = barras_horizontais_svg(
        [x["categoria"] for x in sexo], [x["percentual"] for x in sexo], CORES,
        titulo_eixo="Distribuição dos respondentes",
    ) if sexo else ""

    indicadores_cat = [x["nome"] for x in indicadores]
    indicadores_val = [x["percentual_1"] for x in indicadores]
    svg_indicadores = barras_horizontais_svg(
        indicadores_cat, indicadores_val, CORES, altura=max(280, 55 * len(indicadores) + 55),
        titulo_eixo="Percentual com indicador = 1",
    ) if indicadores else ""

    cards = "".join(card_perfil(p) for p in perfis)
    linhas_assoc = tabela_associacoes(assoc)
    linhas_variaveis = tabela_variaveis({
        "HeartDiseaseorAttack": {"nome": "Doença cardíaca ou ataque cardíaco", "tipo": "alvo binário", "papel": "alvo; não entra no clustering", "descricao": "Desfecho usado somente depois da formação dos clusters para interpretação."},
        "HighBP": {"nome": "Pressão arterial alta", "tipo": "binária", "papel": "fator clínico associado", "descricao": "Indicador de pressão arterial alta."},
        "HighChol": {"nome": "Colesterol alto", "tipo": "binária", "papel": "fator clínico associado", "descricao": "Indicador de colesterol alto."},
        "CholCheck": {"nome": "Verificação de colesterol", "tipo": "binária", "papel": "acompanhamento preventivo", "descricao": "Indicador de verificação de colesterol."},
        "BMI": {"nome": "Índice de Massa Corporal", "tipo": "contínua", "papel": "característica antropométrica", "descricao": "Medida numérica de IMC."},
        "Diabetes_012": {"nome": "Diabetes", "tipo": "categórica ordinal", "papel": "condição clínica associada", "descricao": "0 = sem diabetes; 1 = pré-diabetes; 2 = diabetes."},
        "Smoker": {"nome": "Tabagismo", "tipo": "binária", "papel": "fator comportamental associado", "descricao": "Indicador de tabagismo conforme a codificação da base."},
        "HvyAlcoholConsump": {"nome": "Consumo elevado de álcool", "tipo": "binária", "papel": "fator comportamental associado", "descricao": "Indicador de consumo elevado de álcool."},
        "PhysActivity": {"nome": "Atividade física", "tipo": "binária", "papel": "fator comportamental associado", "descricao": "Indicador de prática de atividade física."},
        "Age": {"nome": "Faixa etária", "tipo": "ordinal", "papel": "característica demográfica", "descricao": "Categorias de idade codificadas de 1 a 13."},
        "Sex": {"nome": "Sexo", "tipo": "binária", "papel": "característica demográfica", "descricao": "Categoria sexual codificada numericamente."},
        "Stroke": {"nome": "Histórico de AVC", "tipo": "binária", "papel": "condição clínica associada", "descricao": "Indicador de histórico de AVC."},
        "GenHlth": {"nome": "Saúde geral percebida", "tipo": "ordinal", "papel": "estado de saúde", "descricao": "Autoavaliação de saúde, de excelente a ruim."},
        "DiffWalk": {"nome": "Dificuldade para caminhar", "tipo": "binária", "papel": "indicador funcional", "descricao": "Indicador de dificuldade para caminhar ou subir escadas."},
    })

    total = dados["total_registros"]
    total_cols = dados.get("total_colunas_tratadas", len(dados.get("features_usadas", [])))
    pca_pct = round(sum(dados.get("variancia_explicada_pca", [0, 0])) * 100)
    bmi = resumo.get("bmi", {})

    html_final = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Análise Preditiva e Segmentação de Perfis de Saúde</title>
<style>
:root{{--bg:#EFF2F5;--surface:#fff;--ink:{INK};--muted:{MUTED};--line:{LINE};--accent:#B23A2E;--soft:{SOFT};}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,Helvetica,sans-serif;line-height:1.55}}
.wrap{{max-width:1100px;margin:0 auto;padding:0 28px}} header{{padding:62px 0 38px;border-bottom:1px solid var(--line)}}
h1,h2,h3{{font-family:Georgia,'Times New Roman',serif;font-weight:600;letter-spacing:-.01em}} h1{{font-size:2.55rem;max-width:18ch;margin:0 0 14px}} h2{{font-size:1.6rem;margin:0 0 8px}} h3{{font-size:1.05rem;margin:0 0 6px}}
.eyebrow{{color:var(--accent);font-weight:700;font-size:.92rem;margin:0 0 12px}} .lede,.section-intro,.muted{{color:var(--muted)}} .lede{{max-width:76ch;font-size:1.06rem}}
section{{padding:44px 0;border-bottom:1px solid var(--line)}} .section-intro{{max-width:82ch;margin:0 0 24px}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}}
.kpi{{background:var(--surface);padding:20px}} .kpi-value{{font-size:1.55rem;font-weight:700}} .kpi-label{{font-weight:600;margin-top:2px}} .kpi-note{{font-size:.84rem;color:var(--muted);margin-top:5px}}
.info-grid,.chart-grid,.two-grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .three-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}
.box{{background:var(--surface);border:1px solid var(--line);padding:22px}} .box h3{{margin-bottom:12px}} .tag{{display:inline-block;background:var(--soft);padding:4px 8px;border-radius:12px;font-size:.78rem;margin:2px 4px 2px 0}}
.pipeline{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}} .step{{background:var(--surface);border:1px solid var(--line);padding:16px}} .step b{{display:block;margin-bottom:5px}} .step span{{font-size:.88rem;color:var(--muted)}}
.note{{margin-top:18px;padding:16px 18px;background:var(--soft);border-left:3px solid var(--accent)}}
.chart-box svg{{width:100%;height:auto;display:block}} table{{width:100%;border-collapse:collapse;background:var(--surface);font-size:.88rem}} th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}} th{{background:#F7F8FA}} code{{font-family:Consolas,monospace}}
.profile-list{{display:flex;flex-direction:column;gap:12px}} .profile-card{{background:var(--surface);border:1px solid var(--line);border-left:5px solid;padding:20px;display:grid;grid-template-columns:190px 1fr 150px;gap:20px;align-items:center}} .profile-name{{font:600 1.25rem Georgia,serif}}
.trait-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .trait-grid strong,.risk strong{{display:block;font:600 1.02rem Consolas,monospace}} .trait-grid span,.risk span{{display:block;color:var(--muted);font-size:.78rem}} .risk{{text-align:right}} .risk strong{{font-size:1.7rem;color:var(--accent)}}
.legend{{font-size:.82rem;color:var(--muted)}} footer{{padding:36px 0 60px;color:var(--muted);font-size:.86rem}} footer a{{color:var(--ink)}}
@media(max-width:850px){{.kpi-grid,.info-grid,.chart-grid,.two-grid,.three-grid,.pipeline{{grid-template-columns:1fr 1fr}}.profile-card{{grid-template-columns:1fr}}.risk{{text-align:left}}.trait-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:560px){{.wrap{{padding:0 16px}}h1{{font-size:2rem}}.kpi-grid,.info-grid,.chart-grid,.two-grid,.three-grid,.pipeline,.trait-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><div class="wrap">
<header>
  <p class="eyebrow">ANÁLISE PREDITIVA · SEGMENTAÇÃO DE PERFIS DE SAÚDE</p>
  <h1>O que existe na base e quais padrões cardiovasculares aparecem nela</h1>
  <p class="lede">O painel documenta a base usada no projeto, explica o tratamento dos dados, mostra as distribuições categóricas e apresenta os perfis encontrados pelo K-Means. As relações com doença cardíaca são descritivas: o painel não estabelece causalidade.</p>
  <div class="note"><strong>Fonte:</strong> {html.escape(fonte.get('dataset','Diabetes Health Indicators Dataset'))}, arquivo {html.escape(fonte.get('arquivo',''))}, disponibilizado via Kaggle a partir do BRFSS 2015. <a href="{html.escape(fonte.get('url','https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset'))}" target="_blank" rel="noopener">Abrir fonte</a>.</div>
</header>

<section>
  <h2>1. Visão geral da base</h2>
  <p class="section-intro">Esta versão do conjunto contém respostas de 253.680 participantes no arquivo utilizado pelo projeto. A base original do Kaggle possui 21 variáveis de entrada; aqui selecionamos um subconjunto focado no perfil clínico, demográfico, funcional e comportamental para a análise.</p>
  <div class="kpi-grid">
    {card_indicador('Registros analisados', fmt_inteiro(total), 'linhas após a seleção/tratamento')}
    {card_indicador('Colunas no arquivo tratado', fmt_inteiro(total_cols), 'inclui a variável-alvo')}
    {card_indicador('Features no clustering', fmt_inteiro(len(dados.get('features_usadas', []))), 'após One-Hot Encoding')}
    {card_indicador('Prevalência observada', fmt(dados['prevalencia_geral_doenca_cardiaca_pct'])+'%', 'HeartDiseaseorAttack na base')}
  </div>
  <div class="info-grid" style="margin-top:18px">
    <div class="box"><h3>O que o conjunto representa?</h3><p class="muted">É um conjunto derivado do BRFSS 2015, um inquérito de saúde. As variáveis representam respostas/indicadores de condições crônicas, comportamentos, saúde percebida, função e características demográficas.</p></div>
    <div class="box"><h3>Por que esta base foi escolhida?</h3><p class="muted">Ela reúne indicadores que podem ser combinados para estudar perfis de saúde e verificar se grupos com características semelhantes exibem prevalências observadas distintas de doença cardíaca.</p></div>
  </div>
</section>

<section>
  <h2>2. Dicionário das variáveis usadas</h2>
  <p class="section-intro">Esta tabela deixa explícito o significado de cada variável e o motivo de sua presença no pipeline.</p>
  <div style="overflow:auto"><table><thead><tr><th>Variável</th><th>Nome</th><th>Tipo</th><th>Papel</th><th>Descrição</th></tr></thead><tbody>{linhas_variaveis}</tbody></table></div>
</section>

<section>
  <h2>3. Limpeza e preparo: causas das decisões</h2>
  <p class="section-intro">Cada tratamento existe por uma razão técnica. O objetivo não é apenas “limpar”, mas preparar as variáveis para os algoritmos sem descartar informação desnecessariamente.</p>
  <div class="pipeline">
    <div class="step"><b>01 · Seleção</b><span>Reduzir a base às variáveis coerentes com o problema e evitar atributos que não participam da análise escolhida.</span></div>
    <div class="step"><b>02 · Validação</b><span>Checar valores ausentes e não numéricos antes de gerar as matrizes usadas pelo Machine Learning.</span></div>
    <div class="step"><b>03 · BMI extremo</b><span>Aplicar winsorização em vez de apagar linhas, preservando a quantidade de respondentes.</span></div>
    <div class="step"><b>04 · Pré-processamento</b><span>Fazer imputação, One-Hot Encoding e escalonamento para tornar as variáveis comparáveis no K-Means.</span></div>
  </div>
  <div class="note"><strong>Motivo principal:</strong> o K-Means trabalha com distância. Sem escalonamento/representação adequada, variáveis em escalas diferentes podem dominar o cálculo de similaridade. Já o One-Hot Encoding transforma a variável categórica de diabetes em colunas explícitas.</div>
  <div class="two-grid" style="margin-top:18px">
    <div class="box"><h3>BMI observado após tratamento</h3><p><strong>Média:</strong> {fmt_numero(bmi.get('media',0))} · <strong>Mediana:</strong> {fmt_numero(bmi.get('mediana',0))}</p><p class="muted">Faixa observada: {fmt_numero(bmi.get('min',0))} a {fmt_numero(bmi.get('max',0))}.</p></div>
    <div class="box"><h3>Cuidados de interpretação</h3><p class="muted">“Fator associado” não significa “causa”. A base é observacional e o clustering não é um estudo causal.</p></div>
  </div>
</section>

<section>
  <h2>4. Distribuições da base — categorias em barras horizontais</h2>
  <p class="section-intro">As variáveis categóricas/ordinais são apresentadas em barras horizontais para facilitar comparação de proporções, especialmente quando existem muitas categorias.</p>
  <div class="two-grid">
    <div class="box"><h3>Classificação de diabetes</h3>{svg_diabetes}</div>
    <div class="box"><h3>Saúde geral percebida</h3>{svg_saude}</div>
  </div>
  <div class="two-grid" style="margin-top:18px">
    <div class="box"><h3>Faixa etária</h3>{svg_idade}</div>
    <div class="box"><h3>Sexo</h3>{svg_sexo}</div>
  </div>
</section>

<section>
  <h2>5. Indicadores de saúde presentes na população</h2>
  <p class="section-intro">Aqui vemos quanto da base apresenta cada indicador. Essa leitura ajuda a entender o contexto antes de interpretar os clusters.</p>
  <div class="box">{svg_indicadores}</div>
</section>

<section>
  <h2>6. Associação descritiva com doença cardíaca</h2>
  <p class="section-intro">A tabela compara a prevalência observada de doença cardíaca entre pessoas com e sem cada indicador binário. A diferença está em pontos percentuais. Isso mostra associação na amostra, não prova causa.</p>
  <div style="overflow:auto"><table><thead><tr><th>Indicador</th><th>Com indicador</th><th>Sem indicador</th><th>Diferença</th><th>N com</th><th>N sem</th></tr></thead><tbody>{linhas_assoc}</tbody></table></div>
  <div class="note"><strong>Como usar:</strong> esses números ajudam a explicar por que certos fatores aparecem destacados na análise, mas não devem ser apresentados como “causas comprovadas” pela base.</div>
</section>

<section>
  <h2>7. Formação dos clusters</h2>
  <p class="section-intro">Foram testados diferentes valores de k. Para facilitar a leitura, cada valor de k aparece como uma barra horizontal. A inércia apoia o método do cotovelo; o silhouette mede a separação média dos grupos.</p>
  <div class="chart-grid">
    <div class="box"><h3>Inércia por k</h3><div class="muted">Método do cotovelo</div>{svg_cotovelo}</div>
    <div class="box"><h3>Silhouette por k</h3><div class="muted">Separação média dos grupos</div>{svg_silhueta}</div>
  </div>
  <div class="note"><strong>k escolhido: {dados['k_escolhido']}</strong>. O código seleciona o melhor silhouette entre os valores testados, usando a inércia como apoio visual.</div>
</section>

<section>
  <h2>8. Mapa dos perfis</h2>
  <p class="section-intro">PCA reduz as dimensões usadas no clustering para duas componentes apenas para visualização. As duas primeiras componentes explicam aproximadamente {pca_pct}% da variância.</p>
  <div class="box">{svg_dispersao}</div>
</section>

<section>
  <h2>9. O que cada cluster representa</h2>
  <p class="section-intro">Os cartões usam os valores originais da base para tornar os grupos interpretáveis. A prevalência de doença cardíaca é calculada depois da clusterização.</p>
  <div class="profile-list">{cards}</div>
</section>

<section>
  <h2>10. Prevalência observada por cluster</h2>
  <p class="section-intro">A comparação mostra como o desfecho se distribui nos grupos encontrados. Como o alvo não participou do clustering, essa etapa funciona como uma leitura posterior dos perfis.</p>
  <div class="box">{svg_prev}</div>
</section>

<section>
  <h2>11. Por que cada etapa existe?</h2>
  <div class="three-grid">
    <div class="box"><h3>Segmentação</h3><p class="muted">{html.escape(interpretacao.get('objetivo',''))}</p></div>
    <div class="box"><h3>K-Means</h3><p class="muted">{html.escape(interpretacao.get('motivo_kmeans',''))}</p></div>
    <div class="box"><h3>PCA e validação</h3><p class="muted">{html.escape(interpretacao.get('motivo_pca',''))}<br><br>{html.escape(interpretacao.get('motivo_validacao',''))}</p></div>
  </div>
</section>

<footer>
  <div><strong>Fonte:</strong> <a href="{html.escape(fonte.get('url','https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset'))}" target="_blank" rel="noopener">Diabetes Health Indicators Dataset — Kaggle</a>.</div>
  <div>Pipeline: tratamento.py → pre_processamento.py → analise_clusters.py → gerar_painel.py</div>
  <div>Gerado em {html.escape(str(dados.get('gerado_em','')))}</div>
</footer>
</div></body></html>'''

    saida = PASTA_DADOS / "painel.html"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(html_final, encoding="utf-8")
    print(f"Painel gerado em {saida.resolve()}")


if __name__ == "__main__":
    main()
