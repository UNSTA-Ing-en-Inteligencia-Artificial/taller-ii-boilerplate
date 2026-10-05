import os
import re
import sys

import numpy as np

MODELOS = {
    "en": {
        "word2vec": "word2vec-google-news-300",
        "glove": "glove-wiki-gigaword-100",
        "fasttext": "fasttext-wiki-news-subwords-300",
        "bert": "bert-base-uncased",
    },
    "es": {
        "word2vec": None,
        "glove": None,
        "fasttext": "bin/cc.es.bin.gz",
        "bert": "dccuchile/bert-base-spanish-wwm-cased",
    },
}

_CACHE = {}


def cargar_una_vez(clave, fn):
    if clave not in _CACHE:
        _CACHE[clave] = fn()
    return _CACHE[clave]


def coseno(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cargar_estatico(origen, es_fasttext=False):
    from gensim import downloader
    from gensim.models import KeyedVectors

    if os.path.exists(origen):
        if es_fasttext and origen.endswith(".bin"):
            from gensim.models.fasttext import load_facebook_vectors
            return load_facebook_vectors(origen)
        return KeyedVectors.load_word2vec_format(origen, binary=origen.endswith(".bin"))
    return downloader.load(origen)


def vector_estatico(kv, token):
    for cand in (token, token.lower(), token.capitalize()):
        if cand in kv.key_to_index:
            return np.array(kv[cand]), "en el vocabulario" + ("" if cand == token else f" (como '{cand}')")
    try:
        return np.array(kv[token]), "FUERA del vocabulario: reconstruido con subpalabras"
    except KeyError:
        return None, "FUERA del vocabulario (este modelo no tiene vector para el token)"


def vector_bert(token, oracion, modelo, capa=-1):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = cargar_una_vez(("bert_tok", modelo), lambda: AutoTokenizer.from_pretrained(modelo))
    mod = cargar_una_vez(("bert_mod", modelo),
                         lambda: AutoModel.from_pretrained(modelo, output_hidden_states=True).eval())

    m = re.search(r"(?<!\w)%s(?!\w)" % re.escape(token), oracion, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"El token '{token}' no aparece en la oración: {oracion!r}")

    enc = tok(oracion, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        salida = mod(**enc)
    h = salida.hidden_states[capa][0]  # (n_tokens, 768)

    idx = [i for i, (a, b) in enumerate(offsets) if b > a and a < m.end() and b > m.start()]
    if not idx:
        raise ValueError("No se pudo alinear el token con las subpalabras del modelo.")
    piezas = tok.convert_ids_to_tokens(enc["input_ids"][0][idx])
    return h[idx].mean(dim=0).numpy(), f"subpalabras: {piezas}, capa {capa}"


def vector_elmo(token, oracion, ruta_modelo):
    from simple_elmo import ElmoModel

    modelo = cargar_una_vez(("elmo", ruta_modelo), lambda: _cargar_elmo(ElmoModel, ruta_modelo))
    palabras = oracion.split()
    norm = lambda w: re.sub(r"\W+", "", w.lower())
    pos = [i for i, w in enumerate(palabras) if norm(w) == norm(token)]
    if not pos:
        raise ValueError(f"El token '{token}' no aparece en la oración: {oracion!r}")
    vecs = modelo.get_elmo_vectors([palabras], layers="average")  # (1, n_palabras, dim)
    return np.array(vecs[0, pos[0]]), "promedio de las capas de ELMo"


def _cargar_elmo(ElmoModel, ruta):
    m = ElmoModel()
    m.load(ruta)
    return m


def vectorizar(token, oracion, cfg, modelos, capa=-1):
    resultados = {}
    for nombre in modelos:
        origen = cfg.get(nombre)
        try:
            if nombre in ("word2vec", "glove", "fasttext"):
                print(f"  cargando {nombre} ({origen})...", file=sys.stderr)
                kv = cargar_una_vez((nombre, origen), lambda: cargar_estatico(origen, nombre == "fasttext"))
                resultados[nombre] = vector_estatico(kv, token)
            elif nombre == "bert":
                resultados[nombre] = vector_bert(token, oracion or token, origen, capa)
            elif nombre == "elmo":
                resultados[nombre] = vector_elmo(token, oracion or token, origen)
        except Exception as e:
            resultados[nombre] = (None, f"error: {e}")
    return resultados


def mostrar(titulo, resultados):
    print(f"\n=== {titulo} ===")
    for nombre, (vec, nota) in resultados.items():
        if vec is None:
            print(f"{nombre:9s} -- {nota}")
        else:
            primeros = np.array2string(vec[:6], precision=3, suppress_small=True)
            print(f"{nombre:9s} dim={len(vec):4d}  norma={np.linalg.norm(vec):6.2f}  {primeros} ...")
            print(f"{'':9s} {nota}")


token = "movie"
contexto = "I did not like the movie"
r1 = vectorizar(token, contexto, MODELOS["en"], ["bert", "word2vec", "fasttext"])
mostrar(f"'{token}'", r1)





