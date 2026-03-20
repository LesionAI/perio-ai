import streamlit as st
import json
from pathlib import Path
import pandas as pd
import tempfile
import os
import re
import unicodedata
import difflib
import hashlib
from faster_whisper import WhisperModel

st.set_page_config(page_title="PerioAI", layout="wide")

# -------------------------
# MODELO WHISPER (LOCAL)
# -------------------------

@st.cache_resource
def cargar_modelo_whisper():
    return WhisperModel("base", compute_type="int8")


def transcribir_audio_desde_ruta(ruta_audio):
    modelo = cargar_modelo_whisper()

    segments, _info = modelo.transcribe(
        ruta_audio,
        language="es",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
        temperature=0.0,
    )

    texto = ""
    for segment in segments:
        if segment.text:
            texto += segment.text + " "

    return texto.strip()


def transcribir_audio_bytes(audio_bytes, suffix=".wav"):
    ruta = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            ruta = tmp.name

        return transcribir_audio_desde_ruta(ruta)
    finally:
        if ruta and os.path.exists(ruta):
            try:
                os.remove(ruta)
            except Exception:
                pass


# -------------------------
# CONFIGURACIÓN
# -------------------------

BASE_DIR = Path(__file__).parent.resolve()
ARCHIVO = BASE_DIR / "pacientes.json"

ORDRE_SUP = ["18", "17", "16", "15", "14", "13", "12", "11", "21", "22", "23", "24", "25", "26", "27", "28"]
ORDRE_INF = ["48", "47", "46", "45", "44", "43", "42", "41", "31", "32", "33", "34", "35", "36", "37", "38"]

TODOS_LOS_DIENTES = set(ORDRE_SUP + ORDRE_INF)

# -------------------------
# NORMALIZACIÓN / OCR
# -------------------------

NUMEROS_TEXTO = {
    "cero": 0,
    "uno": 1,
    "un": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
}

VOCABULARIO_CANONICO = [
    "vestibular", "vesti", "vest", "bucal", "buccal", "labial",
    "palatino", "palatinno", "palatin", "palatinoo", "pal", "lingual", "ling", "linguall",
    "movilidad", "mov", "mobilidad",
    "sangrado", "sangra", "bop",
    "mesial", "mes", "mesiale",
    "central", "centro", "medio",
    "distal", "dist", "dis",
    "diente", "pieza", "numero", "num",
    "con", "y", "en", "cara", "superficie"
]


def quitar_acentos(texto):
    if not texto:
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def hash_bytes(data):
    return hashlib.md5(data).hexdigest()


def normalizar_frase_ocr(cmd: str) -> str:
    if not cmd:
        return ""

    cmd = quitar_acentos(cmd.lower())

    reemplazos_regex = [
        (r"\boencial\b", " mesial "),
        (r"\bocencial\b", " mesial "),
        (r"\boncencial\b", " mesial "),
        (r"\bmencial\b", " mesial "),
        (r"\bmesiai\b", " mesial "),
        (r"\blistal\b", " distal "),
        (r"\blistai\b", " distal "),
        (r"\blisdal\b", " distal "),
        (r"\bdistai\b", " distal "),
        (r"\bdlstal\b", " distal "),
        (r"\bcenral\b", " central "),
        (r"\bcentrai\b", " central "),
        (r"\bmobilitad\b", " movilidad "),
        (r"\bmoviiidad\b", " movilidad "),
        (r"\bsangradoo\b", " sangrado "),
    ]

    for patron, reemplazo in reemplazos_regex:
        cmd = re.sub(patron, reemplazo, cmd)

    return re.sub(r"\s+", " ", cmd).strip()


def corregir_token_ocr(token):
    if not token:
        return token

    t = quitar_acentos(token.lower().strip())

    correcciones_directas = {
        "vestí": "vesti",
        "vesti.": "vesti",
        "listal": "distal",
        "listai": "distal",
        "lisdal": "distal",
        "distai": "distal",
        "dlstal": "distal",
        "oencial": "mesial",
        "ocencial": "mesial",
        "oncencial": "mesial",
        "mencial": "mesial",
        "mesiai": "mesial",
        "cenral": "central",
        "centrai": "central",
        "mobilitad": "movilidad",
        "moviiidad": "movilidad",
        "sangradoo": "sangrado",
    }

    if t in correcciones_directas:
        return correcciones_directas[t]

    if t.isdigit():
        return t

    if t in NUMEROS_TEXTO:
        return t

    if t in VOCABULARIO_CANONICO:
        return t

    candidato = difflib.get_close_matches(t, VOCABULARIO_CANONICO, n=1, cutoff=0.70)
    if candidato:
        return candidato[0]

    return t


def reemplazar_numeros_escritos(texto):
    tokens = texto.split()
    normalizados = []

    for tok in tokens:
        tok_corr = corregir_token_ocr(tok)
        if tok_corr in NUMEROS_TEXTO:
            normalizados.append(str(NUMEROS_TEXTO[tok_corr]))
        else:
            normalizados.append(tok_corr)

    return " ".join(normalizados)


def normalizar_texto_comando(cmd):
    if not cmd:
        return ""

    cmd = normalizar_frase_ocr(cmd)
    cmd = cmd.lower().strip()
    cmd = quitar_acentos(cmd)

    cmd = re.sub(r"(?<=\d)\s*[-_/]\s*(?=\d)", " ", cmd)
    cmd = re.sub(r"(?<=\d)\s*[.,]\s*(?=\d)", " ", cmd)

    cmd = cmd.replace(",", " ")
    cmd = cmd.replace(";", " ")
    cmd = cmd.replace(":", " ")
    cmd = cmd.replace(".", " ")

    cmd = re.sub(r"\bdel\b", " ", cmd)
    cmd = re.sub(r"\bde\b", " ", cmd)
    cmd = re.sub(r"\bla\b", " ", cmd)
    cmd = re.sub(r"\bel\b", " ", cmd)
    cmd = re.sub(r"\blas\b", " ", cmd)
    cmd = re.sub(r"\blos\b", " ", cmd)

    cmd = re.sub(r"\b(diente|pieza)\s+numero\b", " ", cmd)
    cmd = re.sub(r"\b(diente|pieza)\b", " ", cmd)
    cmd = re.sub(r"\bsuperficie\b", " ", cmd)
    cmd = re.sub(r"\bcara\b", " ", cmd)

    cmd = re.sub(r"\s+", " ", cmd).strip()

    tokens = [corregir_token_ocr(tok) for tok in cmd.split()]
    cmd = " ".join(tokens)

    cmd = reemplazar_numeros_escritos(cmd)
    cmd = re.sub(r"\s+", " ", cmd).strip()

    return cmd


# -------------------------
# CARGA / GUARDADO
# -------------------------

def normalizar_paciente(datos):
    if not isinstance(datos, dict):
        return {
            "nombre": "",
            "apellido": "",
            "periodontograma": {},
            "comandos_no_comprendidos": []
        }

    nombre = datos.get("nombre", "")
    apellido = datos.get("apellido", "")
    periodontograma = datos.get("periodontograma", {})
    comandos_no_comprendidos = datos.get("comandos_no_comprendidos", [])

    if not isinstance(periodontograma, dict):
        periodontograma = {}

    if not isinstance(comandos_no_comprendidos, list):
        comandos_no_comprendidos = []

    return {
        "nombre": nombre,
        "apellido": apellido,
        "periodontograma": periodontograma,
        "comandos_no_comprendidos": comandos_no_comprendidos
    }


def cargar_pacientes():
    if not ARCHIVO.exists():
        return {}

    try:
        with open(ARCHIVO, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "pacientes" in data:
            data = data["pacientes"]

        if not isinstance(data, dict):
            return {}

        return {pid: normalizar_paciente(datos) for pid, datos in data.items()}
    except Exception:
        return {}


def guardar():
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(st.session_state.pacientes, f, ensure_ascii=False, indent=4)


# -------------------------
# SESIÓN
# -------------------------

if "pacientes" not in st.session_state:
    st.session_state.pacientes = cargar_pacientes()

if "paciente_activo" not in st.session_state:
    st.session_state.paciente_activo = None

if "view" not in st.session_state:
    st.session_state.view = "inicio"

if "ultimo_texto_transcrito" not in st.session_state:
    st.session_state.ultimo_texto_transcrito = ""

if "ultimo_texto_normalizado" not in st.session_state:
    st.session_state.ultimo_texto_normalizado = ""

if "ultimo_resultado_ok" not in st.session_state:
    st.session_state.ultimo_resultado_ok = None

if "ultimo_resultado_msg" not in st.session_state:
    st.session_state.ultimo_resultado_msg = ""

if "ultimo_origen_audio" not in st.session_state:
    st.session_state.ultimo_origen_audio = ""

if "ultimo_audio_hash_micro" not in st.session_state:
    st.session_state.ultimo_audio_hash_micro = ""

if "ultimo_audio_hash_archivo" not in st.session_state:
    st.session_state.ultimo_audio_hash_archivo = ""


# -------------------------
# PACIENTES
# -------------------------

def crear_paciente(pid, nombre, apellido):
    pid = pid.strip()
    nombre = nombre.strip()
    apellido = apellido.strip()

    if not pid or not nombre or not apellido:
        return False, "Completa todos los campos."

    if pid in st.session_state.pacientes:
        return False, "El ID ya existe."

    st.session_state.pacientes[pid] = {
        "nombre": nombre,
        "apellido": apellido,
        "periodontograma": {},
        "comandos_no_comprendidos": []
    }
    guardar()
    return True, "Paciente creado correctamente."


def eliminar_paciente(pid):
    if pid in st.session_state.pacientes:
        del st.session_state.pacientes[pid]
        guardar()


def seleccionar_paciente(pid):
    st.session_state.paciente_activo = pid
    st.session_state.view = "paciente"


def paciente():
    pid = st.session_state.paciente_activo
    if pid and pid in st.session_state.pacientes:
        return st.session_state.pacientes[pid]
    return None


# -------------------------
# HELPERS PARSER
# -------------------------

SINONIMOS_VESTIBULAR = {
    "vestibular", "vesti", "vest", "bucal", "buccal", "labial"
}

SINONIMOS_PALATINO = {
    "palatino", "palatinno", "palatin", "palatinoo", "pal", "lingual", "ling", "linguall"
}

SINONIMOS_MOVILIDAD = {
    "movilidad", "mov", "mobilidad"
}

SINONIMOS_SANGRADO = {
    "sangrado", "sangra", "bop"
}

SINONIMOS_SITIO_MESIAL = {"mesial", "mes", "mesiale"}
SINONIMOS_SITIO_CENTRAL = {"central", "centro", "medio"}
SINONIMOS_SITIO_DISTAL = {"distal", "dist", "dis"}


def inicializar_diente_si_no_existe(periodontograma, diente):
    if diente not in periodontograma:
        periodontograma[diente] = {
            "vestibular": [None, None, None],
            "palatino": [None, None, None],
            "sangrado": [],
            "movilidad": None
        }


def es_diente(token):
    return token.isdigit() and len(token) == 2 and token in TODOS_LOS_DIENTES


def normalizar_cara(token):
    t = corregir_token_ocr(token)
    if t in SINONIMOS_VESTIBULAR:
        return "vestibular"
    if t in SINONIMOS_PALATINO:
        return "palatino"
    return None


def normalizar_tipo(token):
    t = corregir_token_ocr(token)
    if t in SINONIMOS_MOVILIDAD:
        return "movilidad"
    if t in SINONIMOS_SANGRADO:
        return "sangrado"
    return None


def normalizar_sitio(token):
    t = quitar_acentos(token.lower().strip())

    if t in SINONIMOS_SITIO_MESIAL:
        return "mesial"
    if t in SINONIMOS_SITIO_CENTRAL:
        return "central"
    if t in SINONIMOS_SITIO_DISTAL:
        return "distal"

    correcciones_seguras = {
        "oencial": "mesial",
        "ocencial": "mesial",
        "oncencial": "mesial",
        "mencial": "mesial",
        "mesiai": "mesial",
        "cenral": "central",
        "centrai": "central",
        "listal": "distal",
        "listai": "distal",
        "lisdal": "distal",
        "distai": "distal",
        "dlstal": "distal",
    }

    if t in correcciones_seguras:
        return correcciones_seguras[t]

    return None


def convertir_numero_simple(token):
    t = corregir_token_ocr(token)
    if t.isdigit():
        return int(t)
    return NUMEROS_TEXTO.get(t)


def normalizar_tripleta_compacta(token):
    t = re.sub(r"\D", "", corregir_token_ocr(token))

    if len(t) == 3:
        return [int(t[0]), int(t[1]), int(t[2])]

    if len(t) == 4 and len(set(t)) == 1:
        return [int(t[0]), int(t[1]), int(t[2])]

    return None


def extraer_tres_valores(tokens, start_idx):
    if start_idx >= len(tokens):
        return None, 0

    compacta = normalizar_tripleta_compacta(tokens[start_idx])
    if compacta is not None:
        return compacta, 1

    if start_idx + 2 < len(tokens):
        v1 = convertir_numero_simple(tokens[start_idx])
        v2 = convertir_numero_simple(tokens[start_idx + 1])
        v3 = convertir_numero_simple(tokens[start_idx + 2])

        if v1 is not None and v2 is not None and v3 is not None:
            return [v1, v2, v3], 3

    return None, 0


def es_token_estructural(token):
    if es_diente(token):
        return True
    if normalizar_cara(token) is not None:
        return True
    if normalizar_tipo(token) is not None:
        return True
    if token in {"con", "y", "en"}:
        return True
    return False


def extraer_sitios_sangrado(tokens, start_idx):
    sitios = []
    consumidos = 0
    i = start_idx

    while i < len(tokens):
        token = tokens[i]

        if es_token_estructural(token):
            break

        sitio = normalizar_sitio(token)
        if sitio is None:
            break

        if sitio not in sitios:
            sitios.append(sitio)

        consumidos += 1
        i += 1

    return sitios, consumidos


def registrar_comando_no_comprendido(texto):
    p = paciente()
    if not p:
        return
    texto = texto.strip()
    if not texto:
        return
    p.setdefault("comandos_no_comprendidos", [])
    p["comandos_no_comprendidos"].append(texto)
    guardar()


def borrar_comandos_no_comprendidos():
    p = paciente()
    if not p:
        return
    p["comandos_no_comprendidos"] = []
    guardar()


# -------------------------
# PARSER ROBUSTO
# -------------------------

def comando_periodontal(cmd):
    p = paciente()
    if not p:
        return False, "Selecciona un paciente primero."

    cmd_normalizado = normalizar_texto_comando(cmd)
    st.session_state.ultimo_texto_normalizado = cmd_normalizado

    tokens = cmd_normalizado.split()

    if not tokens:
        return False, "Comando vacío."

    diente_actual = None
    hubo_cambio = False
    mensajes_ok = []
    mensajes_error = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if es_diente(token):
            diente_actual = token
            inicializar_diente_si_no_existe(p["periodontograma"], diente_actual)
            i += 1
            continue

        if token in {"con", "y", "en"}:
            i += 1
            continue

        if diente_actual is None:
            mensajes_error.append(f"Se ignoró '{token}' porque falta el número de diente.")
            i += 1
            continue

        datos = p["periodontograma"][diente_actual]

        cara = normalizar_cara(token)
        if cara is not None:
            valores, consumidos = extraer_tres_valores(tokens, i + 1)

            if valores is None:
                mensajes_error.append(f"Faltan 3 valores para {cara} en diente {diente_actual}.")
                i += 1
                continue

            datos[cara] = valores
            hubo_cambio = True
            mensajes_ok.append(f"{diente_actual} {cara}={'-'.join(map(str, valores))}")
            i += 1 + consumidos
            continue

        tipo = normalizar_tipo(token)

        if tipo == "movilidad":
            if i + 1 >= len(tokens):
                mensajes_error.append(f"Falta el grado de movilidad del diente {diente_actual}.")
                i += 1
                continue

            siguiente = tokens[i + 1]
            valor = convertir_numero_simple(siguiente)

            if valor is None or valor not in [0, 1, 2, 3]:
                mensajes_error.append(f"Movilidad inválida en diente {diente_actual}.")
                if not es_token_estructural(siguiente):
                    i += 2
                else:
                    i += 1
                continue

            datos["movilidad"] = valor
            hubo_cambio = True
            mensajes_ok.append(f"{diente_actual} movilidad={valor}")
            i += 2
            continue

        if tipo == "sangrado":
            sitios, consumidos = extraer_sitios_sangrado(tokens, i + 1)

            if consumidos == 0:
                mensajes_error.append(f"Falta el sitio de sangrado del diente {diente_actual}.")
                if i + 1 < len(tokens):
                    siguiente = tokens[i + 1]
                    if not es_token_estructural(siguiente):
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
                continue

            for sitio in sitios:
                if sitio not in datos["sangrado"]:
                    datos["sangrado"].append(sitio)

            hubo_cambio = True
            mensajes_ok.append(f"{diente_actual} sangrado={','.join(sitios)}")
            i += 1 + consumidos
            continue

        mensajes_error.append(f"No se entendió '{token}' en diente {diente_actual}.")
        i += 1

    if hubo_cambio:
        guardar()

    bloques = []
    if mensajes_ok:
        bloques.append("Datos guardados: " + " | ".join(mensajes_ok))
    if mensajes_error:
        bloques.append("Avisos: " + " | ".join(mensajes_error))

    if hubo_cambio:
        return True, " ".join(bloques)

    return False, " ".join(bloques) if bloques else "No se detectó ningún cambio."


# -------------------------
# ANÁLISIS CLÍNICO
# -------------------------

def obtener_max_bolsas(periodontograma):
    maximo = 0
    for _diente, datos in periodontograma.items():
        for cara in ["vestibular", "palatino"]:
            valores = datos.get(cara, [])
            if isinstance(valores, list):
                nums = [v for v in valores if isinstance(v, int)]
                if nums:
                    maximo = max(maximo, max(nums))
    return maximo


def contar_dientes_con_bolsas(periodontograma, umbral=4):
    conteo = 0
    for _diente, datos in periodontograma.items():
        tiene = False
        for cara in ["vestibular", "palatino"]:
            valores = datos.get(cara, [])
            if isinstance(valores, list) and any(isinstance(v, int) and v >= umbral for v in valores):
                tiene = True
        if tiene:
            conteo += 1
    return conteo


def contar_sitios_sangrado(periodontograma):
    total = 0
    for _diente, datos in periodontograma.items():
        total += len(datos.get("sangrado", []))
    return total


def contar_dientes_movilidad(periodontograma):
    total = 0
    grados = []
    for _diente, datos in periodontograma.items():
        mov = datos.get("movilidad")
        if mov is not None:
            total += 1
            grados.append(mov)
    return total, max(grados) if grados else 0


def dientes_con_movilidad(periodontograma):
    resultado = []
    for diente, datos in periodontograma.items():
        mov = datos.get("movilidad")
        if mov is not None and mov > 0:
            resultado.append((diente, mov))
    return sorted(resultado, key=lambda x: x[0])


def score_periodontal(periodontograma):
    max_bolsa = obtener_max_bolsas(periodontograma)
    dientes_bolsa_4 = contar_dientes_con_bolsas(periodontograma, umbral=4)
    sitios_sangrado = contar_sitios_sangrado(periodontograma)
    dientes_mov, max_mov = contar_dientes_movilidad(periodontograma)

    score = 0
    score += max_bolsa * 8
    score += dientes_bolsa_4 * 5
    score += sitios_sangrado * 2
    score += dientes_mov * 6
    score += max_mov * 10
    return min(score, 100)


def es_periodonto_saludable(periodontograma):
    if not periodontograma:
        return False

    max_bolsa = obtener_max_bolsas(periodontograma)
    sitios_sangrado = contar_sitios_sangrado(periodontograma)
    dientes_mov, _ = contar_dientes_movilidad(periodontograma)
    dientes_bolsa_4 = contar_dientes_con_bolsas(periodontograma, umbral=4)

    return (
        max_bolsa <= 3
        and sitios_sangrado == 0
        and dientes_mov == 0
        and dientes_bolsa_4 == 0
    )


def es_gingivitis_probable(periodontograma):
    if not periodontograma:
        return False

    max_bolsa = obtener_max_bolsas(periodontograma)
    sitios_sangrado = contar_sitios_sangrado(periodontograma)
    dientes_mov, _ = contar_dientes_movilidad(periodontograma)
    dientes_bolsa_4 = contar_dientes_con_bolsas(periodontograma, umbral=4)

    return (
        max_bolsa <= 3
        and sitios_sangrado > 0
        and dientes_mov == 0
        and dientes_bolsa_4 == 0
    )


def determinar_estadio(periodontograma):
    if not periodontograma:
        return "No concluyente"

    max_bolsa = obtener_max_bolsas(periodontograma)
    dientes_mov, max_mov = contar_dientes_movilidad(periodontograma)

    if max_bolsa <= 3 and max_mov == 0:
        return "Sin periodontitis"

    if max_bolsa >= 6 or max_mov >= 2:
        return "III"
    if max_bolsa >= 4:
        return "II"

    return "I"


def determinar_grado(periodontograma):
    if not periodontograma:
        return "No concluyente"

    if es_periodonto_saludable(periodontograma) or es_gingivitis_probable(periodontograma):
        return "No aplica"

    sitios_sangrado = contar_sitios_sangrado(periodontograma)
    dientes_bolsa_4 = contar_dientes_con_bolsas(periodontograma, umbral=4)
    max_bolsa = obtener_max_bolsas(periodontograma)

    if max_bolsa >= 6 or dientes_bolsa_4 >= 6 or sitios_sangrado >= 8:
        return "C"
    if max_bolsa >= 4 or dientes_bolsa_4 >= 3 or sitios_sangrado >= 4:
        return "B"
    return "A"


def describir_estadio(estadio):
    descripciones = {
        "I": "afectación leve",
        "II": "afectación moderada",
        "III": "afectación severa",
        "Sin periodontitis": "sin signos compatibles con periodontitis",
        "No concluyente": "datos insuficientes",
    }
    return descripciones.get(estadio, "")


def describir_grado(grado):
    descripciones = {
        "A": "riesgo de evolución bajo",
        "B": "riesgo de evolución moderado",
        "C": "riesgo de evolución alto",
        "No aplica": "no aplica",
        "No concluyente": "datos insuficientes",
    }
    return descripciones.get(grado, "")


def determinar_extension(periodontograma):
    if not periodontograma:
        return "No concluyente"

    if es_periodonto_saludable(periodontograma) or es_gingivitis_probable(periodontograma):
        return "No aplica"

    dientes_registrados = len(periodontograma)
    dientes_afectados = contar_dientes_con_bolsas(periodontograma, umbral=4)

    if dientes_registrados == 0:
        return "No concluyente"

    proporcion = dientes_afectados / dientes_registrados

    if dientes_afectados == 0:
        return "Sin extensión significativa"
    if proporcion < 0.30:
        return "Localizada"
    return "Generalizada"


def generar_indicaciones(periodontograma):
    max_bolsa = obtener_max_bolsas(periodontograma)
    sitios_sangrado = contar_sitios_sangrado(periodontograma)
    dientes_mov, max_mov = contar_dientes_movilidad(periodontograma)

    indicaciones = []

    if es_periodonto_saludable(periodontograma):
        indicaciones.append("Tejidos compatibles con salud periodontal según los datos registrados.")
        indicaciones.append("Mantenimiento periodontal y control clínico periódico.")
        indicaciones.append("Refuerzo de la higiene oral domiciliaria.")
        return indicaciones

    if es_gingivitis_probable(periodontograma):
        indicaciones.append("Control de placa e instrucciones de higiene oral personalizadas.")
        indicaciones.append("Profilaxis o tartrectomía supragingival según la evaluación clínica.")
        indicaciones.append("Reevaluación del sangrado al sondaje tras la fase inicial.")
        return indicaciones

    indicaciones.append("Instrucciones de higiene oral personalizadas.")

    if max_bolsa >= 4:
        indicaciones.append("Raspado y alisado radicular según la evaluación clínica.")
        indicaciones.append("Reevaluación periodontal en 4-6 semanas.")
    else:
        indicaciones.append("Profilaxis y control periodontal no quirúrgico según la evaluación clínica.")

    if sitios_sangrado > 0:
        indicaciones.append("Reevaluación del sangrado al sondaje tras la fase inicial.")
    if max_bolsa >= 6:
        indicaciones.append("Valorar tratamiento periodontal avanzado y estudio radiográfico.")
    if dientes_mov > 0:
        indicaciones.append("Vigilar la movilidad dental y ajustar el pronóstico por diente.")
    if max_mov >= 2:
        indicaciones.append("Considerar ferulización y/o ajuste oclusal según el contexto clínico.")

    return indicaciones


def generar_conclusion_clinica(periodontograma):
    if not periodontograma:
        return "Sin datos periodontales registrados."

    dientes_registrados = len(periodontograma)
    max_bolsa = obtener_max_bolsas(periodontograma)
    dientes_bolsa_4 = contar_dientes_con_bolsas(periodontograma, umbral=4)
    sitios_sangrado = contar_sitios_sangrado(periodontograma)
    dientes_mov, max_mov = contar_dientes_movilidad(periodontograma)

    if es_periodonto_saludable(periodontograma):
        return (
            f"Dientes registrados: {dientes_registrados}. "
            f"Profundidad máxima: {max_bolsa} mm. "
            f"Sitios con sangrado: {sitios_sangrado}. "
            f"Dientes con movilidad: {dientes_mov}. "
            f"Conclusión orientativa: hallazgos compatibles con salud periodontal."
        )

    if es_gingivitis_probable(periodontograma):
        return (
            f"Dientes registrados: {dientes_registrados}. "
            f"Profundidad máxima: {max_bolsa} mm. "
            f"Sitios con sangrado: {sitios_sangrado}. "
            f"Dientes con movilidad: {dientes_mov}. "
            f"Conclusión orientativa: hallazgos compatibles con inflamación gingival sin bolsas periodontales patológicas."
        )

    estadio = determinar_estadio(periodontograma)
    grado = determinar_grado(periodontograma)
    extension = determinar_extension(periodontograma)

    return (
        f"Dientes registrados: {dientes_registrados}. "
        f"Profundidad máxima: {max_bolsa} mm. "
        f"Dientes con bolsas ≥4 mm: {dientes_bolsa_4}. "
        f"Sitios con sangrado: {sitios_sangrado}. "
        f"Dientes con movilidad: {dientes_mov} (grado máximo {max_mov}). "
        f"Conclusión orientativa: periodontitis {extension.lower()}, estadio {estadio}, grado {grado}."
    )


# -------------------------
# TABLAS
# -------------------------

def limpiar_valor_para_tabla(v):
    if v is None or v == "":
        return ""
    try:
        return str(int(v))
    except Exception:
        return str(v)


def formater_triplet(valores):
    if not isinstance(valores, list) or len(valores) != 3:
        return ""
    return " ".join(limpiar_valor_para_tabla(v) for v in valores)


def formater_sangrado(valores):
    if not valores:
        return ""
    abreviaciones = {"mesial": "M", "central": "C", "distal": "D"}
    return " ".join(abreviaciones.get(v, v) for v in valores)


def generar_fila_arcada(periodontograma, dientes, tipo):
    fila = {}
    for diente in dientes:
        datos = periodontograma.get(diente, {})
        if tipo == "vestibular":
            fila[diente] = formater_triplet(datos.get("vestibular", [None, None, None]))
        elif tipo == "palatino":
            fila[diente] = formater_triplet(datos.get("palatino", [None, None, None]))
        elif tipo == "movilidad":
            fila[diente] = limpiar_valor_para_tabla(datos.get("movilidad", ""))
        elif tipo == "sangrado":
            fila[diente] = formater_sangrado(datos.get("sangrado", []))
    return fila


def generar_tableau_arcada(periodontograma, dientes, es_superior=True):
    filas = []
    nombre_palatino = "Palatino" if es_superior else "Lingual"

    filas.append({"Medida": nombre_palatino, **generar_fila_arcada(periodontograma, dientes, "palatino")})
    filas.append({"Medida": "Vestibular", **generar_fila_arcada(periodontograma, dientes, "vestibular")})
    filas.append({"Medida": "Sangrado", **generar_fila_arcada(periodontograma, dientes, "sangrado")})
    filas.append({"Medida": "Movilidad", **generar_fila_arcada(periodontograma, dientes, "movilidad")})

    return pd.DataFrame(filas)


def style_cell_arcada(val):
    if pd.isna(val) or val == "":
        return ""

    texto = str(val).strip()

    if texto in {"0", "1", "2", "3"}:
        v = int(texto)
        if v == 0:
            return "background-color: #f3f4f6; color: #374151; text-align: center;"
        if v == 1:
            return "background-color: #fde68a; color: #92400e; text-align: center;"
        if v == 2:
            return "background-color: #fdba74; color: #9a3412; text-align: center;"
        if v == 3:
            return "background-color: #fecaca; color: #991b1b; text-align: center;"

    partes = texto.split()
    if len(partes) == 3 and all(p.isdigit() for p in partes):
        nums = [int(p) for p in partes]
        max_val = max(nums)
        if max_val <= 3:
            return "background-color: #dcfce7; color: #166534; text-align: center;"
        if max_val == 4:
            return "background-color: #fed7aa; color: #9a3412; text-align: center;"
        return "background-color: #fecaca; color: #991b1b; text-align: center;"

    if any(x in texto for x in ["M", "C", "D"]):
        return "background-color: #fee2e2; color: #991b1b; font-weight: bold; text-align: center;"

    return "text-align: center;"


def styler_tableau_arcada(df):
    styler = df.style
    subset_cols = [col for col in df.columns if col != "Medida"]

    for col in subset_cols:
        styler = styler.map(style_cell_arcada, subset=[col])

    styler = styler.set_properties(**{"text-align": "center", "font-size": "15px"})
    styler = styler.set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#f3f4f6"),
                ("color", "#111827"),
                ("font-weight", "bold"),
                ("text-align", "center"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
                ("vertical-align", "middle"),
            ],
        },
    ])
    return styler


# -------------------------
# INTERFAZ
# -------------------------

st.title("🦷 PerioAI")
st.subheader("Asistente inteligente para periodontograma")
st.divider()

if st.session_state.view == "inicio":
    st.header("Crear paciente")

    col1, col2, col3 = st.columns(3)
    with col1:
        pid = st.text_input("ID")
    with col2:
        nombre = st.text_input("Nombre")
    with col3:
        apellido = st.text_input("Apellido")

    if st.button("Crear paciente"):
        ok, msg = crear_paciente(pid, nombre, apellido)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.divider()
    st.header("Buscar paciente")
    buscar = st.text_input("Buscar por nombre o ID")
    st.divider()

    pacientes = st.session_state.pacientes

    if not pacientes:
        st.info("No hay pacientes todavía.")
    else:
        for pid, datos in pacientes.items():
            nombre = datos.get("nombre", "")
            apellido = datos.get("apellido", "")
            nombre_completo = f"{nombre} {apellido}".strip()

            texto_busqueda = buscar.lower().strip()
            if texto_busqueda:
                if texto_busqueda not in nombre_completo.lower() and texto_busqueda not in pid.lower():
                    continue

            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"### {nombre_completo if nombre_completo else 'Paciente sin nombre'}")
                    st.write(f"ID: {pid}")
                with col2:
                    if st.button("Abrir", key=f"abrir_{pid}"):
                        seleccionar_paciente(pid)
                        st.rerun()

elif st.session_state.view == "paciente":
    p = paciente()

    if not p:
        st.warning("No hay paciente seleccionado.")
        if st.button("⬅ Volver al inicio"):
            st.session_state.view = "inicio"
            st.rerun()
        st.stop()

    pid = st.session_state.paciente_activo

    st.header(f"👤 {p.get('nombre', '')} {p.get('apellido', '')}")
    st.write(f"ID paciente: **{pid}**")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✏️ Editar datos"):
            st.session_state.view = "editar"
            st.rerun()

    with col2:
        if st.button("🦷 Periodontograma"):
            st.session_state.view = "periodontograma"
            st.rerun()

    with col3:
        if st.button("🗑 Eliminar paciente"):
            eliminar_paciente(pid)
            st.session_state.paciente_activo = None
            st.session_state.view = "inicio"
            st.rerun()

    st.divider()

    if st.button("⬅ Volver al inicio"):
        st.session_state.view = "inicio"
        st.rerun()

elif st.session_state.view == "editar":
    p = paciente()

    if not p:
        st.warning("No hay paciente seleccionado.")
        if st.button("⬅ Volver al inicio"):
            st.session_state.view = "inicio"
            st.rerun()
        st.stop()

    st.header("Editar paciente")

    nombre = st.text_input("Nombre", p.get("nombre", ""))
    apellido = st.text_input("Apellido", p.get("apellido", ""))

    if st.button("Guardar cambios"):
        p["nombre"] = nombre.strip()
        p["apellido"] = apellido.strip()
        guardar()
        st.success("Paciente actualizado.")

    if st.button("⬅ Volver"):
        st.session_state.view = "paciente"
        st.rerun()

elif st.session_state.view == "periodontograma":
    p = paciente()

    if not p:
        st.warning("No hay paciente seleccionado.")
        if st.button("⬅ Volver al inicio"):
            st.session_state.view = "inicio"
            st.rerun()
        st.stop()

    st.header(f"Periodontograma — {p.get('nombre', '')} {p.get('apellido', '')}")

    if st.button("⬅ Volver paciente"):
        st.session_state.view = "paciente"
        st.rerun()

    st.divider()

    st.subheader("🎤 Dictado por voz")

    col_reset_mic_1, col_reset_mic_2 = st.columns([1, 3])
    with col_reset_mic_1:
        if st.button("Reprocesar micro"):
            st.session_state.ultimo_audio_hash_micro = ""

    audio = st.audio_input("Grabar voz")

    if audio is not None:
        audio_bytes = audio.getvalue()
        audio_hash = hash_bytes(audio_bytes)

        if audio_hash != st.session_state.ultimo_audio_hash_micro:
            st.session_state.ultimo_audio_hash_micro = audio_hash
            try:
                texto = transcribir_audio_bytes(audio_bytes, suffix=".wav")
                st.session_state.ultimo_texto_transcrito = texto
                st.session_state.ultimo_origen_audio = "micrófono"

                if texto:
                    ok, msg = comando_periodontal(texto)
                    st.session_state.ultimo_resultado_ok = ok
                    st.session_state.ultimo_resultado_msg = msg

                    if not ok:
                        registrar_comando_no_comprendido(texto)
                else:
                    st.session_state.ultimo_resultado_ok = False
                    st.session_state.ultimo_resultado_msg = "No se detectó texto en el audio."
            except Exception as e:
                st.session_state.ultimo_resultado_ok = False
                st.session_state.ultimo_resultado_msg = f"Error durante la transcripción: {e}"

    if st.session_state.ultimo_origen_audio == "micrófono" and st.session_state.ultimo_texto_transcrito:
        st.info(f"Texto reconocido: {st.session_state.ultimo_texto_transcrito}")

    st.divider()

    st.subheader("📁 Subir audio")

    archivo_audio = st.file_uploader(
        "Sube un archivo de audio para transcribir",
        type=["wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm"],
    )

    if archivo_audio is not None:
        st.audio(archivo_audio)

        if st.button("Transcribir archivo"):
            try:
                audio_bytes = archivo_audio.getvalue()
                audio_hash = hash_bytes(audio_bytes)

                extension = Path(archivo_audio.name).suffix.lower()

                if audio_hash != st.session_state.ultimo_audio_hash_archivo:
                    st.session_state.ultimo_audio_hash_archivo = audio_hash

                    texto = transcribir_audio_bytes(
                        audio_bytes,
                        suffix=extension if extension else ".wav"
                    )

                    st.session_state.ultimo_texto_transcrito = texto
                    st.session_state.ultimo_origen_audio = "archivo"

                    if texto:
                        ok, msg = comando_periodontal(texto)
                        st.session_state.ultimo_resultado_ok = ok
                        st.session_state.ultimo_resultado_msg = msg

                        if not ok:
                            registrar_comando_no_comprendido(texto)
                    else:
                        st.session_state.ultimo_resultado_ok = False
                        st.session_state.ultimo_resultado_msg = "No se detectó texto en el archivo de audio."
                else:
                    st.session_state.ultimo_resultado_ok = True
                    st.session_state.ultimo_resultado_msg = "Ese archivo ya fue transcrito."
            except Exception as e:
                st.session_state.ultimo_resultado_ok = False
                st.session_state.ultimo_resultado_msg = f"Error durante la transcripción del archivo: {e}"

    if st.session_state.ultimo_origen_audio == "archivo" and st.session_state.ultimo_texto_transcrito:
        st.info(f"Texto reconocido: {st.session_state.ultimo_texto_transcrito}")

    if st.session_state.ultimo_resultado_msg:
        if st.session_state.ultimo_resultado_ok is True:
            st.success(st.session_state.ultimo_resultado_msg)
        elif st.session_state.ultimo_resultado_ok is False:
            st.error(st.session_state.ultimo_resultado_msg)

    st.divider()

    perio = p.get("periodontograma", {})

    if not perio:
        st.info("Sin datos todavía.")
    else:
        st.subheader("Arcada superior")
        df_sup = generar_tableau_arcada(perio, ORDRE_SUP, es_superior=True)
        styled_sup = styler_tableau_arcada(df_sup)
        st.dataframe(styled_sup, width="stretch", hide_index=True)

        st.divider()

        st.subheader("Arcada inferior")
        df_inf = generar_tableau_arcada(perio, ORDRE_INF, es_superior=False)
        styled_inf = styler_tableau_arcada(df_inf)
        st.dataframe(styled_inf, width="stretch", hide_index=True)

        st.divider()

        st.subheader("📊 Análisis automático")

        estadio = determinar_estadio(perio)
        grado = determinar_grado(perio)
        extension = determinar_extension(perio)
        conclusion = generar_conclusion_clinica(perio)
        indicaciones = generar_indicaciones(perio)
        moviles = dientes_con_movilidad(perio)

        texto_estadio = describir_estadio(estadio)
        texto_grado = describir_grado(grado)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Estadio", f"{estadio} — {texto_estadio}")
        with c2:
            st.metric("Grado", f"{grado} — {texto_grado}")
        with c3:
            st.metric("Extensión", extension)

        st.info(conclusion)

        with st.expander("Interpretación del estadio y del grado"):
            st.write("**Estadio I**: afectación leve")
            st.write("**Estadio II**: afectación moderada")
            st.write("**Estadio III**: afectación severa")
            st.write("**Grado A**: riesgo de evolución bajo")
            st.write("**Grado B**: riesgo de evolución moderado")
            st.write("**Grado C**: riesgo de evolución alto")

        if moviles:
            texto_mov = ", ".join([f"{d} (grado {g})" for d, g in moviles])
            st.warning(f"Dientes con movilidad: {texto_mov}")

        st.subheader("Indicaciones orientativas")
        for ind in indicaciones:
            st.write(f"- {ind}")

    st.divider()

    st.subheader("Comandos no comprendidos")

    comandos_no = p.get("comandos_no_comprendidos", [])

    if not comandos_no:
        st.info("No hay comandos no comprendidos.")
    else:
        for idx, texto in enumerate(comandos_no, start=1):
            st.write(f"{idx}. {texto}")

        if st.button("Borrar comandos no comprendidos"):
            borrar_comandos_no_comprendidos()
            st.rerun()
