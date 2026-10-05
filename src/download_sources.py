"""Obtém fontes oficiais adicionais, mantendo respostas e hashes para auditoria."""
import hashlib
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "data/raw/sidra_populacao_2022.json": "https://apisidra.ibge.gov.br/values/t/4714/n6/all/v/93/p/2022?formato=json",
    "data/raw/sidra_populacao_2024.json": "https://apisidra.ibge.gov.br/values/t/6579/n6/all/v/9324/p/2024?formato=json",
    "data/geo/municipios_2022.geojson": "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR?formato=application/vnd.geo%2Bjson&qualidade=minima&intrarregiao=municipio&periodo=2022",
}


def main():
    manifest = {}
    for name, url in SOURCES.items():
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Baixando {name}", flush=True)
            with urlopen(Request(url, headers={"User-Agent": "AcademicHomicideStudy/1.0"}), timeout=60) as response:
                content = response.read()
            if content.startswith(b"\x1f\x8b"):
                content = gzip.decompress(content)
            json.loads(content)
            path.write_bytes(content)
        content = path.read_bytes()
        manifest[name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest(),
                          "verificado_em_utc": datetime.now(timezone.utc).isoformat()}
    (ROOT / "data/raw/fontes_adicionais.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Fontes obtidas; manifesto registrado.")


if __name__ == "__main__":
    main()
