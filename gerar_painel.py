#!/usr/bin/env python3
"""
Gera o painel HTML (DB/dados_tratados/painel.html) a partir de
dados_clusters.json. Todos os gráficos são SVG desenhados aqui mesmo — o
arquivo final não depende de internet nem de bibliotecas JS externas,
então abre em qualquer navegador com um duplo clique.
"""

from __future__ import annotations

import json
from pathlib import Path

PASTA_DADOS = Path("DB/dados_tratados")
CORES = ["#1F3A5F", "#B23A2E", "#2E7D6B", "#C98A2C", "#6B4C9A", "#3D8FB5"]
INK = "#16232E"
MUTED = "#5C6B7A"
LINE = "#D9E0E5"


def fmt(v, casas=1):
    return f"{v:.{casas}f}".replace(".", ",")


# ---------------------------------------------------------------- SVG utils

def fmt_ptbr_inteiro(v):
    return f"{v:,.0f}".replace(",", ".")


def fmt_compacto(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}".replace(".", ",") + " mi"
    if v >= 1_000:
        return f"{v/1_000:.0f} mil"
    return f"{v:.0f}"


def fmt_ptbr_decimal(v):
    return f"{v:.2f}".replace(".", ",")


def linha_chart(valores_x, valores_y, cor, largura=380, altura=220, formatador=fmt_compacto):
    m = {"l": 50, "r": 14, "t": 16, "b": 30}
    pw, ph = largura - m["l"] - m["r"], altura - m["t"] - m["b"]
    ymin, ymax = min(valores_y), max(valores_y)
    folga = (ymax - ymin) * 0.12 or 1
    ymin, ymax = ymin - folga, ymax + folga

    def px(i):
        return m["l"] + (i / (len(valores_x) - 1)) * pw

    def py(v):
        return m["t"] + ph - (v - ymin) / (ymax - ymin) * ph

    pontos = [(px(i), py(v)) for i, v in enumerate(valores_y)]
    caminho = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pontos)

    partes = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Sans, Arial, sans-serif">']
    # grid horizontal (4 linhas)
    for f in (0, 0.33, 0.66, 1.0):
        y = m["t"] + ph * f
        val = ymax - (ymax - ymin) * f
        partes.append(f'<line x1="{m["l"]}" y1="{y:.1f}" x2="{largura-m["r"]}" y2="{y:.1f}" stroke="{LINE}" stroke-width="1"/>')
        partes.append(f'<text x="{m["l"]-8}" y="{y+3:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{formatador(val)}</text>')
    partes.append(f'<path d="{caminho}" fill="none" stroke="{cor}" stroke-width="2.2"/>')
    for i, (x, y) in enumerate(pontos):
        partes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{cor}"/>')
        partes.append(f'<text x="{x:.1f}" y="{altura-8}" font-size="10" fill="{MUTED}" text-anchor="middle">{valores_x[i]}</text>')
    partes.append("</svg>")
    return "".join(partes)


def dispersao_svg(pontos, cores, largura=760, altura=380):
    m = {"l": 44, "r": 20, "t": 16, "b": 34}
    pw, ph = largura - m["l"] - m["r"], altura - m["t"] - m["b"]
    xs = [p["x"] for p in pontos]
    ys = [p["y"] for p in pontos]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    fx, fy = (xmax - xmin) * 0.06 or 1, (ymax - ymin) * 0.06 or 1
    xmin, xmax, ymin, ymax = xmin - fx, xmax + fx, ymin - fy, ymax + fy

    def px(v):
        return m["l"] + (v - xmin) / (xmax - xmin) * pw

    def py(v):
        return m["t"] + ph - (v - ymin) / (ymax - ymin) * ph

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


def barras_svg(categorias, valores, cores, media, largura=760, altura=280):
    m = {"l": 46, "r": 20, "t": 16, "b": 34}
    pw, ph = largura - m["l"] - m["r"], altura - m["t"] - m["b"]
    vmax = max(max(valores), media) * 1.18
    n = len(categorias)
    gap = pw / n * 0.32
    bw = pw / n - gap

    def py(v):
        return m["t"] + ph - (v / vmax) * ph

    partes = [f'<svg viewBox="0 0 {largura} {altura}" xmlns="http://www.w3.org/2000/svg">']
    for f in (0, 0.5, 1.0):
        y = m["t"] + ph * (1 - f)
        partes.append(f'<line x1="{m["l"]}" y1="{y:.1f}" x2="{largura-m["r"]}" y2="{y:.1f}" stroke="{LINE}"/>')
        partes.append(f'<text x="{m["l"]-8}" y="{y+3:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{vmax*f:.0f}%</text>')
    for i, (cat, val) in enumerate(zip(categorias, valores)):
        x = m["l"] + i * (pw / n) + gap / 2
        y = py(val)
        h = m["t"] + ph - y
        partes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{cores[i % len(cores)]}" rx="2"/>')
        partes.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" font-size="11" fill="{INK}" text-anchor="middle" font-weight="600">{fmt(val)}%</text>')
        partes.append(f'<text x="{x+bw/2:.1f}" y="{altura-10}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{cat}</text>')
    ym = py(media)
    partes.append(f'<line x1="{m["l"]}" y1="{ym:.1f}" x2="{largura-m["r"]}" y2="{ym:.1f}" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="5,4"/>')
    partes.append(f'<text x="{largura-m["r"]}" y="{ym-6:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">média geral: {fmt(media)}%</text>')
    partes.append("</svg>")
    return "".join(partes)


# ---------------------------------------------------------------- montagem

def cartao_perfil(p: dict) -> str:
    cor = CORES[p["cluster"] % len(CORES)]
    tamanho_fmt = f"{p['tamanho']:,}".replace(",", ".")
    return f"""
    <div class="profile-card" style="border-left-color:{cor}">
      <div class="id">
        <div class="name">Cluster {p['cluster']}</div>
        <div class="size">{tamanho_fmt} pessoas · {fmt(p['percentual_do_total'])}% da base</div>
      </div>
      <div class="traits">
        <div><b>{p['diabetes_predominante']}</b><span>diabetes</span></div>
        <div><b>{p['faixa_etaria_predominante']}</b><span>faixa etária</span></div>
        <div><b>{fmt(p['bmi_medio'])}</b><span>IMC médio</span></div>
        <div><b>{fmt(p['pct_pressao_alta'])}%</b><span>pressão alta</span></div>
        <div><b>{fmt(p['pct_colesterol_alto'])}%</b><span>colesterol alto</span></div>
        <div><b>{fmt(p['pct_fumante'])}%</b><span>fumantes</span></div>
      </div>
      <div class="risk">
        <div class="pct" style="color:{cor}">{fmt(p['prevalencia_doenca_cardiaca_pct'])}%</div>
        <div class="lbl">doença cardíaca</div>
      </div>
    </div>"""


def main():
    dados = json.loads((PASTA_DADOS / "dados_clusters.json").read_text(encoding="utf-8"))

    ks = [r["k"] for r in dados["varredura_k"]]
    inercias = [r["inercia"] for r in dados["varredura_k"]]
    silhuetas = [r["silhueta"] for r in dados["varredura_k"]]

    svg_cotovelo = linha_chart(ks, inercias, INK, formatador=fmt_compacto)
    svg_silhueta = linha_chart(ks, silhuetas, "#B23A2E", formatador=fmt_ptbr_decimal)
    svg_dispersao = dispersao_svg(dados["pontos_dispersao"], CORES)

    perfis_por_cluster_id = sorted(dados["perfis"], key=lambda p: p["cluster"])
    svg_barras = barras_svg(
        [f"Cluster {p['cluster']}" for p in perfis_por_cluster_id],
        [p["prevalencia_doenca_cardiaca_pct"] for p in perfis_por_cluster_id],
        [CORES[p["cluster"] % len(CORES)] for p in perfis_por_cluster_id],
        dados["prevalencia_geral_doenca_cardiaca_pct"],
    )

    legenda = "".join(
        f'<span><span class="dot" style="background:{CORES[p["cluster"] % len(CORES)]}"></span>Cluster {p["cluster"]}</span>'
        for p in perfis_por_cluster_id
    )

    cartoes = "".join(cartao_perfil(p) for p in dados["perfis"])
    pca_var_pct = round((dados["variancia_explicada_pca"][0] + dados["variancia_explicada_pca"][1]) * 100)

    k_note = (
        f'k escolhido: <b>{dados["k_escolhido"]}</b> — maior silhouette entre os valores testados, '
        f'em uma região onde a curva de inércia também perde inclinação (o "cotovelo").'
    )

    template = Path("../painel_template2.html").read_text(encoding="utf-8")
    html = (
        template
        .replace("__TOTAL__", f"{dados['total_registros']:,}".replace(",", "."))
        .replace("__FEATURES__", str(len(dados["features_usadas"])))
        .replace("__PREV_GERAL__", fmt(dados["prevalencia_geral_doenca_cardiaca_pct"]) + "%")
        .replace("__PCA_VAR__", str(pca_var_pct) + "%")
        .replace("__K_NOTE__", k_note)
        .replace("__SVG_COTOVELO__", svg_cotovelo)
        .replace("__SVG_SILHUETA__", svg_silhueta)
        .replace("__SVG_DISPERSAO__", svg_dispersao)
        .replace("__LEGENDA__", legenda)
        .replace("__CARTOES_PERFIS__", cartoes)
        .replace("__SVG_BARRAS__", svg_barras)
    )

    saida = PASTA_DADOS / "painel.html"
    saida.write_text(html, encoding="utf-8")
    print(f"Painel gerado em {saida.resolve()}")


if __name__ == "__main__":
    main()
