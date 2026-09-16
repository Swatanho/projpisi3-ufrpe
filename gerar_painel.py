"""
Gera o painel HTML a partir de dados_graficos.json.

O HTML é autocontido e desenha os gráficos em SVG, sem bibliotecas JS externas.

Dependências:
    apenas Python padrão nesta etapa.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CORES = ["#1F3A5F", "#B23A2E", "#2E7D6B", "#C98A2C", "#6B4C9A", "#3D8FB5"]
INK = "#16232E"
MUTED = "#5C6B7A"
LINE = "#D9E0E5"
SOFT = "#F1DCD8"
BG = "#EFF2F5"
SURFACE = "#FFFFFF"


def fmt(v, casas=1):
    return f"{float(v):.{casas}f}".replace(".", ",")


def fmt_inteiro(v):
    return f"{int(v):,}".replace(",", ".")


def barras_grupo_svg(dados: list[dict], largura: int = 800, altura: int = 380, maxv: float = 100.0) -> str:
    """Barras horizontais agrupadas: duas barras (mulher/homem) por métrica.

    A legenda no topo identifica as cores. Não repetimos “Mulher/Homem”
    ao lado de cada barra, evitando sobreposição visual com os nomes dos
    indicadores, especialmente nos rótulos mais longos.
    """
    n = len(dados)
    m = {"l": 245, "r": 65, "t": 28, "b": 34}
    pw = largura - m["l"] - m["r"]
    ph = altura - m["t"] - m["b"]
    passo = ph / max(1, n)
    bar_h = min(14, passo * 0.22)
    gap = max(4, bar_h * 0.35)
    cores = {0: "#B23A2E", 1: "#1F3A5F"}
    out = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    for frac in (0, .25, .5, .75, 1):
        x = m["l"] + pw * frac
        out.append(f'<line x1="{x:.1f}" y1="{m["t"]}" x2="{x:.1f}" y2="{altura-m["b"]}" stroke="{LINE}"/>')
        out.append(f'<text x="{x:.1f}" y="{altura-10}" font-size="10" fill="{MUTED}" text-anchor="middle">{fmt(maxv*frac,0)}%</text>')

    # Legenda única do gráfico: vermelho = Mulher; azul = Homem.
    out.append(f'<rect x="{largura-180}" y="4" width="10" height="10" rx="2" fill="{cores[0]}"/><text x="{largura-164}" y="13" font-size="11" fill="{MUTED}">Mulher</text>')
    out.append(f'<rect x="{largura-96}" y="4" width="10" height="10" rx="2" fill="{cores[1]}"/><text x="{largura-80}" y="13" font-size="11" fill="{MUTED}">Homem</text>')

    for i, item in enumerate(dados):
        y0 = m["t"] + i * passo + passo * .16
        grupo_h = 2 * bar_h + gap
        # Rótulo do indicador centralizado verticalmente entre as duas barras.
        label_y = y0 + grupo_h / 2 + 4
        out.append(f'<text x="{m["l"]-14}" y="{label_y:.1f}" font-size="11" fill="{INK}" text-anchor="end">{html.escape(item["nome"])}</text>')
        for j, val in enumerate(item["valores"]):
            y = y0 + j * (bar_h + gap)
            bw = pw * float(val["percentual"]) / maxv
            out.append(f'<rect x="{m["l"]}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" rx="3" fill="{cores[val["sex"]]}"/>')
            out.append(f'<text x="{m["l"]+bw+7:.1f}" y="{y+bar_h*0.82:.1f}" font-size="10.5" fill="{INK}" font-weight="600">{fmt(val["percentual"])}%</text>')
    out.append("</svg>")
    return "".join(out)


def heatmap_svg(heat: dict, minimo: float, maximo: float, formato: str, largura: int = 820, altura: int = 360) -> str:
    linhas, colunas, valores = heat["linhas"], heat["colunas"], heat["valores"]
    m = {"l": 150, "r": 30, "t": 48, "b": 58}
    pw = largura - m["l"] - m["r"]
    ph = altura - m["t"] - m["b"]
    cw = pw / len(colunas)
    ch = ph / len(linhas)
    span = maximo - minimo if maximo != minimo else 1

    def interp(v: float) -> str:
        # Escala neutra azul -> vermelho construída manualmente para SVG.
        t = max(0.0, min(1.0, (v - minimo) / span))
        r = int(232 * t + 45 * (1-t))
        g = int(102 * (1-t) + 73 * t)
        b = int(173 * (1-t) + 117 * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    out = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    for j, col in enumerate(colunas):
        x = m["l"] + j * cw + cw / 2
        out.append(f'<text x="{x:.1f}" y="{m["t"]-18}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{html.escape(col["label"])}</text>')
    for i, row in enumerate(linhas):
        y = m["t"] + i * ch + ch / 2
        out.append(f'<text x="{m["l"]-10}" y="{y+4:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{html.escape(row["label"])}</text>')
        for j, _col in enumerate(colunas):
            valor = valores[i][j]
            x = m["l"] + j * cw
            yy = m["t"] + i * ch
            cor = "#ECEFF1" if valor is None else interp(valor)
            out.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{cw-2:.1f}" height="{ch-2:.1f}" rx="3" fill="{cor}"/>')
            if valor is not None:
                texto = f"{valor:.1f}%" if formato == "pct" else f"{valor:.2f}"
                text_color = "#FFFFFF" if (valor-minimo)/span > 0.58 else INK
                out.append(f'<text x="{x+cw/2:.1f}" y="{yy+ch/2+4:.1f}" font-size="9.5" fill="{text_color}" text-anchor="middle" font-weight="600">{texto}</text>')
    out.append(f'<text x="{m["l"]+pw/2:.1f}" y="{altura-8}" font-size="10" fill="{MUTED}" text-anchor="middle">Faixas de renda (1 = menor · 8 = maior)</text>')
    out.append("</svg>")
    return "".join(out)


def narrativa_grafico1(res: dict) -> str:
    d = res
    return (
        f"Na amostra, a taxa observada de doença cardíaca/ataque cardíaco é {fmt(d['taxa_cardiaca_mulher'])}% entre mulheres e {fmt(d['taxa_cardiaca_homem'])}% entre homens. "
        f"O gráfico coloca esse desfecho lado a lado com tabagismo ({fmt(d['tabagismo_mulher'])}% vs. {fmt(d['tabagismo_homem'])}%), consumo elevado de álcool ({fmt(d['alcool_mulher'])}% vs. {fmt(d['alcool_homem'])}%) e ausência de atividade física ({fmt(d['sedentarismo_mulher'])}% vs. {fmt(d['sedentarismo_homem'])}%). "
        "O objetivo é verificar se as diferenças de comportamento acompanham as diferenças observadas no evento cardíaco; o gráfico, isoladamente, não identifica a causa da diferença."
    )


def narrativa_grafico2(res: dict) -> str:
    return (
        f"A verificação de colesterol ocorreu em {fmt(res['cholcheck_mulher'])}% das mulheres e {fmt(res['cholcheck_homem'])}% dos homens. "
        f"A cobertura de saúde é de {fmt(res['cobertura_mulher'])}% e {fmt(res['cobertura_homem'])}%, respectivamente. "
        f"Já a proporção que relatou deixar de ir ao médico por causa do custo é de {fmt(res['barreira_custo_mulher'])}% entre mulheres e {fmt(res['barreira_custo_homem'])}% entre homens. "
        "Essas diferenças permitem discutir prevenção e acesso sem transformar uma associação observacional em julgamento de comportamento individual."
    )


def narrativa_grafico3(res: dict) -> str:
    return (
        f"Entre a menor e a maior faixa de renda, a média de GenHlth varia de {fmt(res['genhlth_renda_menor'],2)} para {fmt(res['genhlth_renda_maior'],2)} e a prevalência observada de diabetes varia de {fmt(res['diabetes_renda_menor_pct'],1)}% para {fmt(res['diabetes_renda_maior_pct'],1)}%. "
        f"Por escolaridade, GenHlth varia de {fmt(res['genhlth_educacao_menor'],2)} para {fmt(res['genhlth_educacao_maior'],2)} e o diabetes de {fmt(res['diabetes_educacao_menor_pct'],1)}% para {fmt(res['diabetes_educacao_maior_pct'],1)}%. "
        "Como GenHlth é codificada de 1 (excelente) a 5 (ruim), valores maiores representam pior saúde percebida. As células do heatmap mostram onde esses padrões se concentram, mas não provam causalidade."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=Path("DB/dados_tratados/dados_graficos.json"))
    parser.add_argument("--saida", type=Path, default=Path("DB/dados_tratados/painel.html"))
    args = parser.parse_args()
    dados = json.loads(args.dados.read_text(encoding="utf-8"))
    g1 = dados["graficos"]["genero_riscos"]
    g2 = dados["graficos"]["prevencao_barreira"]
    g3 = dados["graficos"]["socioeconomico"]

    # limites para os heatmaps
    vals1 = [v for row in g3["heatmap_genhlth"]["valores"] for v in row if v is not None]
    vals2 = [v for row in g3["heatmap_diabetes"]["valores"] for v in row if v is not None]
    heat1 = heatmap_svg(g3["heatmap_genhlth"], min(vals1), max(vals1), "num")
    heat2 = heatmap_svg(g3["heatmap_diabetes"], min(vals2), max(vals2), "pct")
    svg1 = barras_grupo_svg(g1["dados"])
    svg2 = barras_grupo_svg(g2["dados"], altura=320)

    html_final = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Perfil de Saúde e Indicadores Cardiovasculares</title>
<style>
:root{{--bg:{BG};--surface:{SURFACE};--ink:{INK};--muted:{MUTED};--line:{LINE};--accent:#B23A2E;--soft:{SOFT};}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,Helvetica,sans-serif;line-height:1.55}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 28px}} header{{padding:64px 0 42px;border-bottom:1px solid var(--line)}}
.eyebrow{{margin:0 0 12px;color:var(--accent);font-weight:700;font-size:.9rem;letter-spacing:.03em}} h1,h2,h3{{font-family:Georgia,'Times New Roman',serif;font-weight:600;letter-spacing:-.01em}} h1{{font-size:2.7rem;max-width:20ch;margin:0 0 14px}} h2{{font-size:1.7rem;margin:0 0 8px}} h3{{font-size:1.05rem;margin:0 0 5px}} .lede,.intro,.muted{{color:var(--muted)}} .lede{{font-size:1.06rem;max-width:82ch}}
section{{padding:46px 0;border-bottom:1px solid var(--line)}} .box{{background:var(--surface);border:1px solid var(--line);padding:22px}} .chart-wrap{{background:var(--surface);border:1px solid var(--line);padding:18px 20px}} .note{{margin-top:18px;background:var(--soft);border-left:3px solid var(--accent);padding:15px 17px}} .kpi-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-top:22px}} .kpi{{background:var(--surface);padding:18px}} .kpi b{{display:block;font-size:1.5rem}} .kpi span{{color:var(--muted);font-size:.86rem}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .svg-title{{font-weight:700;margin-bottom:5px}} .legend{{font-size:.82rem;color:var(--muted)}} footer{{padding:34px 0 60px;color:var(--muted);font-size:.84rem}} a{{color:var(--ink)}}
@media(max-width:860px){{.two,.kpi-grid{{grid-template-columns:1fr}}h1{{font-size:2.2rem}}}}
</style>
</head>
<body><div class="wrap">
<header>
<p class="eyebrow">ANÁLISE EXPLORATÓRIA DE SAÚDE</p>
<h1>Como gênero, prevenção e condição socioeconômica se relacionam aos indicadores de saúde</h1>
<p class="lede">Esta página foi reduzida a três cruzamentos diretamente interpretáveis da base. Os percentuais mostram associações observadas na amostra; eles não permitem afirmar que um fator cause sozinho um desfecho.</p>
<div class="kpi-grid">
  <div class="kpi"><b>{fmt_inteiro(dados['total_registros'])}</b><span>registros analisados</span></div>
  <div class="kpi"><b>3</b><span>análises principais</span></div>
  <div class="kpi"><b>2015</b><span>BRFSS / dataset disponibilizado via Kaggle</span></div>
</div>
</header>

<section>
<h2>1. Gênero vs. fatores de risco e eventos cardíacos</h2>
<p class="intro">Cada indicador é comparado separadamente entre mulheres e homens. “Sem atividade física” é calculado como o complemento de PhysActivity para tornar visualmente explícita a ideia de sedentarismo.</p>
<div class="chart-wrap"><div class="svg-title">Percentual de cada grupo que apresenta o indicador</div>{svg1}<div class="legend">Doença cardíaca = desfecho observado · tabagismo/álcool = comportamentos de risco · sem atividade física = comportamento associado ao sedentarismo.</div></div>
<div class="note"><strong>Leitura:</strong> {html.escape(narrativa_grafico1(g1['resumo']))}</div>
</section>

<section>
<h2>2. Comportamento preventivo e barreira financeira</h2>
<p class="intro">Aqui a comparação muda de foco: a pergunta é quanto cada grupo relata acompanhamento preventivo, cobertura de saúde e dificuldade financeira para buscar atendimento.</p>
<div class="chart-wrap"><div class="svg-title">Percentual de cada grupo que respondeu “sim” ao indicador</div>{svg2}</div>
<div class="note"><strong>Leitura:</strong> {html.escape(narrativa_grafico2(g2['resumo']))}</div>
</section>

<section>
<h2>3. Determinantes sociais da saúde</h2>
<p class="intro">O heatmap cruza duas dimensões socioeconômicas: renda (colunas) e escolaridade (linhas). À esquerda vemos a média de saúde geral percebida (GenHlth); à direita, a prevalência observada de diabetes binário.</p>
<div class="two">
  <div class="box"><h3>Saúde geral percebida — média de GenHlth</h3><p class="muted">1 = excelente · 5 = ruim · valores maiores representam pior saúde percebida.</p>{heat1}</div>
  <div class="box"><h3>Diabetes — prevalência observada</h3><p class="muted">Percentual de respondentes com Diabetes_binary = 1.</p>{heat2}</div>
</div>
<div class="note"><strong>Leitura:</strong> {html.escape(narrativa_grafico3(g3['resumo']))}</div>
</section>

<section>
<h2>O que esses três gráficos permitem dizer?</h2>
<div class="two">
  <div class="box"><h3>O que a análise mede</h3><p class="muted">Diferenças de percentuais entre grupos definidos por sexo e padrões de saúde associados a renda e escolaridade. O foco é identificar onde as taxas observadas são diferentes e quais combinações merecem investigação.</p></div>
  <div class="box"><h3>O que a análise não mede</h3><p class="muted">Ela não isola efeitos causais, não controla simultaneamente todas as variáveis e não permite concluir que gênero, renda, escolaridade ou um comportamento, isoladamente, causou o desfecho.</p></div>
</div>
</section>

<footer>
<div><strong>Fonte:</strong> <a href="{html.escape(dados['fonte']['url'])}" target="_blank" rel="noopener">Diabetes Health Indicators Dataset — Kaggle</a>.</div>
<div>Pipeline: tratamento.py → analise_clusters.py → gerar_painel.py</div>
<div>Gerado em {html.escape(str(dados.get('gerado_em','')))}</div>
</footer>
</div></body></html>'''

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(html_final, encoding="utf-8")
    print(f"Painel gerado em {args.saida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
