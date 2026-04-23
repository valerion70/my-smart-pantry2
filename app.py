import streamlit as st
import google.generativeai as genai
from PIL import Image
from supabase import create_client, Client
import pandas as pd
import json
import re
import time
from datetime import date, timedelta

# --- CONFIGURAZIONE E CONNESSIONE ---
st.set_page_config(page_title="Smart Pantry Cloud", page_icon="🍎")

# Caricamento segreti
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GEMINI_KEY = st.secrets["GEMINI_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)

# --- HELPER AI CON FALLBACK E RATE-LIMIT ---
# Gerarchie di modelli per tipologia di compito. La rete di sicurezza finale
# è sempre gemini-2.5-flash-lite, il modello con la quota RPM più ampia.
# NOTA: se un model ID non esiste ancora nell'SDK, l'helper scarta quel
# modello e passa al successivo senza disturbare l'utente.
AI_MODELS_TESTO = (
    "gemini-3-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
AI_MODELS_IMMAGINI = (
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-3-flash-lite",
    "gemini-2.5-flash-lite",
)
_AI_MIN_INTERVAL_SEC = 1.2  # anti doppio-click: ignora chiamate troppo ravvicinate


def _is_quota_error(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "429" in s
        or "quota" in s
        or "rate" in s
        or "resourceexhausted" in s
        or "resource_exhausted" in s
    )


def _is_model_unavailable_error(err: Exception) -> bool:
    s = str(err).lower()
    return (
        "404" in s
        or "not found" in s
        or "notfound" in s
        or "is not supported" in s
        or "is not found" in s
        or "unsupported" in s
        or "permission" in s  # alcuni modelli richiedono tier a pagamento
    )


def _ai_generate(contents, models=AI_MODELS_TESTO, max_retries_transient=1):
    """
    Chiama Gemini provando in sequenza i modelli indicati.
    - Su errore di quota (429) passa SUBITO al modello successivo.
    - Su modello inesistente/non accessibile passa al successivo (no retry).
    - Su errore transiente fa un breve retry con backoff.
    - Solleva l'ultima eccezione se tutti i modelli falliscono.
    """
    # Anti doppio-click: throttle minimo fra chiamate consecutive.
    now = time.monotonic()
    last = st.session_state.get("_ai_last_call_ts", 0.0)
    if now - last < _AI_MIN_INTERVAL_SEC:
        time.sleep(_AI_MIN_INTERVAL_SEC - (now - last))
    st.session_state["_ai_last_call_ts"] = time.monotonic()

    last_err = None
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            last_err = e
            continue
        for attempt in range(max_retries_transient + 1):
            try:
                return model.generate_content(contents)
            except Exception as e:
                last_err = e
                if _is_quota_error(e) or _is_model_unavailable_error(e):
                    break  # prova prossimo modello
                if attempt < max_retries_transient:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    raise last_err if last_err else RuntimeError("AI generation failed")


# --- COSTANTI PASTI ---
ORDINE_PASTI = ["colazione", "spuntino", "pranzo", "merenda", "cena"]
ETICHETTE_PASTI = {
    "colazione": "Colazione",
    "spuntino": "Spuntino",
    "pranzo": "Pranzo",
    "merenda": "Merenda",
    "cena": "Cena",
}

# --- COSTANTI PROFILO NUTRIZIONALE ---
ATTIVITA_QUOTIDIANA_OPZIONI = [
    ("sedentario", "Sedentario — ufficio/divano"),
    ("leggero", "Leggero — in piedi buona parte del giorno"),
    ("attivo", "Attivo — cammini molto / lavoro in movimento"),
    ("pesante", "Pesante — lavoro fisico (cantiere, magazzino)"),
]
INTENSITA_ALLENAMENTO_OPZIONI = [
    ("leggera", "Leggera"),
    ("moderata", "Moderata"),
    ("intensa", "Intensa"),
]
OBIETTIVO_OPZIONI = [
    ("deficit", "Perdere grasso (deficit)"),
    ("mantenimento", "Mantenere il peso"),
    ("surplus", "Aumentare massa (surplus)"),
]
ESPERIENZA_OPZIONI = [
    ("principiante", "Principiante"),
    ("intermedio", "Intermedio"),
    ("avanzato", "Avanzato"),
]
APPROCCIO_OPZIONI = [
    ("equilibrato", "Equilibrato"),
    ("lowcarb", "Low Carb"),
    ("cheto", "Chetogenico"),
    ("vegano", "Vegano"),
]
CUCINA_OPZIONI = [
    ("principiante", "Principiante — ricette semplici e veloci"),
    ("intermedio", "Intermedio — mi districo in cucina"),
    ("avanzato", "Avanzato — anche ricette elaborate"),
]
ATTREZZATURA_OPZIONI = [
    ("fornelli", "Fornelli / piano cottura"),
    ("forno", "Forno"),
    ("microonde", "Microonde"),
    ("friggitrice_aria", "Friggitrice ad aria"),
    ("pentola_pressione", "Pentola a pressione"),
    ("frullatore", "Frullatore / mixer"),
    ("planetaria", "Robot da cucina / planetaria"),
    ("bilancia", "Bilancia da cucina"),
    ("griglia", "Griglia / barbecue"),
]

# --- GESTIONE SESSIONE UTENTE ---
if "user" not in st.session_state:
    st.session_state.user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None
if "conferma_eliminazione" not in st.session_state:
    # lista di id da eliminare in attesa di conferma, oppure None
    st.session_state.conferma_eliminazione = None
if "preview_dati" not in st.session_state:
    st.session_state.preview_dati = None
if "preview_stimati" not in st.session_state:
    st.session_state.preview_stimati = set()
if "uploader_key" not in st.session_state:
    # incrementato dopo ogni salvataggio per resettare il file_uploader
    st.session_state.uploader_key = 0
if "pasti_abilitati" not in st.session_state:
    st.session_state.pasti_abilitati = list(ORDINE_PASTI)
if "profilo" not in st.session_state:
    st.session_state.profilo = None

# Streamlit rie-esegue create_client a ogni rerun → la sessione auth viene persa
# e le policy RLS che usano auth.uid() falliscono. Ripristiniamola dai token salvati.
if st.session_state.access_token and st.session_state.refresh_token:
    try:
        supabase.auth.set_session(
            st.session_state.access_token,
            st.session_state.refresh_token,
        )
    except Exception:
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        
def sign_up(email, password):
    try:
        supabase.auth.sign_up({"email": email, "password": password})
        st.success("Registrazione completata! Controlla la mail per confermare (se richiesto) o prova il login.")
    except Exception as e:
        st.error(f"Errore registrazione: {e}")

def login(email, password):
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user = res.user
        st.session_state.access_token = res.session.access_token
        st.session_state.refresh_token = res.session.refresh_token
        st.rerun()
    except Exception:
        st.error("Credenziali non valide.")

def _to_float(value):
    """Converte in float se possibile, altrimenti None. Estrae il primo numero da stringhe tipo '25,3 g'."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        m = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
        if m:
            try:
                return float(m.group(0).replace(",", "."))
            except ValueError:
                return None
        return None

def _stessi_valori(a, b):
    """Confronto che tratta NaN == NaN come True (per non scatenare falsi update)."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return a == b

_MESI_IT = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]

def _data_it(d):
    """Formatta una data come '3 aprile' (senza anno)."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return f"{d.day} {_MESI_IT[d.month - 1]}"

def _aggiungi_a_lista_spesa(nome, user_id):
    """Inserisce il prodotto nella lista della spesa se non è già presente."""
    if not nome:
        return
    try:
        existing = (
            supabase.table("lista_spesa").select("id")
            .eq("user_id", user_id).eq("nome", nome).limit(1).execute()
        )
        if existing.data:
            return
        supabase.table("lista_spesa").insert(
            {"user_id": user_id, "nome": nome}
        ).execute()
    except Exception:
        pass

def _applica_delta_dispensa(nome, delta_grammi, user_id):
    """Scala la quantità del prodotto in dispensa di `delta_grammi`.
    Positivo = consumato (sottrae), negativo = ripristino (aggiunge).
    Se la quantità scende a 0, il prodotto viene eliminato dalla dispensa
    e aggiunto alla lista della spesa. Silenzioso se non è più in dispensa.
    """
    if not nome or not delta_grammi:
        return
    try:
        res = (
            supabase.table("dispensa").select("id, quantita")
            .eq("user_id", user_id).eq("nome", nome).limit(1).execute()
        )
        if not res.data:
            return
        riga = res.data[0]
        q_attuale = float(riga.get("quantita") or 0)
        q_nuova = max(0.0, q_attuale - float(delta_grammi))
        if q_nuova <= 0:
            (
                supabase.table("dispensa").delete()
                .eq("id", riga["id"]).eq("user_id", user_id).execute()
            )
            _aggiungi_a_lista_spesa(nome, user_id)
        else:
            (
                supabase.table("dispensa").update({"quantita": q_nuova})
                .eq("id", riga["id"]).eq("user_id", user_id).execute()
            )
    except Exception:
        pass

def _render_pasti(df_giorno):
    """Mostra i consumi raggruppati per pasto; ogni pasto è un expander con gli ingredienti."""
    df_g = df_giorno.copy()
    if "pasto" not in df_g.columns:
        df_g["pasto"] = None

    def _mostra(gruppo, label):
        tot_kcal = gruppo["kcal"].sum()
        titolo = ""
        if "ricetta" in gruppo.columns:
            ricette = {
                str(r).strip() for r in gruppo["ricetta"].dropna().tolist()
                if str(r).strip()
            }
            if len(ricette) == 1:
                titolo = f": {next(iter(ricette))}"
        with st.expander(f"{label}{titolo} — {tot_kcal:.0f} kcal"):
            tab = gruppo[["nome", "quantita", "kcal", "carbi", "pro", "grassi"]].copy()
            tab.columns = ["Nome", "g", "kcal", "Carbi", "Pro", "Grassi"]
            st.dataframe(tab, hide_index=True, use_container_width=True)

    for pasto in ORDINE_PASTI:
        gruppo = df_g[df_g["pasto"] == pasto]
        if not gruppo.empty:
            _mostra(gruppo, ETICHETTE_PASTI[pasto])

    altri = df_g[~df_g["pasto"].isin(ORDINE_PASTI)]
    if not altri.empty:
        _mostra(altri, "Altro")

def _form_profilo_e_salva(prof_attuale, user_id, key_prefix, mostra_target=True, testo_bottone="🤖 Calcola obiettivi con IA"):
    """Render del form profilo+obiettivi+cucina. Gestisce calcolo AI e upsert.
    `key_prefix` serve a non duplicare le chiavi dei widget tra sidebar e onboarding."""
    st.caption(
        "Il tuo target giornaliero di kcal e macro viene calcolato dall'IA a partire "
        "dai dati qui sotto. Aggiornalo solo se cambiano peso, obiettivo o stile di vita."
    )

    st.markdown("**1. Profilo biometrico**")
    sesso = st.radio(
        "Sesso",
        ["M", "F"],
        format_func=lambda x: "Maschio" if x == "M" else "Femmina",
        horizontal=True,
        index=1 if prof_attuale.get("sesso") == "F" else 0,
        key=f"{key_prefix}_sesso",
    )
    eta = st.number_input(
        "Età", min_value=10, max_value=120, step=1,
        value=int(prof_attuale.get("eta") or 30),
        key=f"{key_prefix}_eta",
    )
    altezza = st.number_input(
        "Altezza (cm)", min_value=100.0, max_value=230.0, step=0.5,
        value=float(prof_attuale.get("altezza_cm") or 170.0),
        key=f"{key_prefix}_altezza",
    )
    peso = st.number_input(
        "Peso (kg)", min_value=30.0, max_value=300.0, step=0.1,
        value=float(prof_attuale.get("peso_kg") or 70.0),
        key=f"{key_prefix}_peso",
    )
    aggiungi_circ = st.checkbox(
        "Aggiungi misure di collo e vita *(facoltativo)*",
        value=bool(prof_attuale.get("collo_cm") or prof_attuale.get("vita_cm")),
        key=f"{key_prefix}_aggiungi_circ",
    )
    st.caption(
        "🛈 Le circonferenze sono opzionali: servono solo a migliorare la precisione "
        "della stima del grasso corporeo."
    )
    if aggiungi_circ:
        collo = st.number_input(
            "Misura Circonferenza del Collo (in cm) — facoltativo",
            min_value=20.0, max_value=80.0, step=0.5,
            value=float(prof_attuale.get("collo_cm") or 35.0),
            key=f"{key_prefix}_collo",
        )
        vita = st.number_input(
            "Misura Circonferenza della Vita (in cm) — facoltativo",
            min_value=40.0, max_value=200.0, step=0.5,
            value=float(prof_attuale.get("vita_cm") or 80.0),
            key=f"{key_prefix}_vita",
        )
    else:
        collo = None
        vita = None

    st.markdown("**2. Stile di vita**")
    attivita_keys = [k for k, _ in ATTIVITA_QUOTIDIANA_OPZIONI]
    idx_att = (
        attivita_keys.index(prof_attuale["attivita_quotidiana"])
        if prof_attuale.get("attivita_quotidiana") in attivita_keys else 0
    )
    attivita = st.selectbox(
        "Attività quotidiana",
        attivita_keys,
        format_func=lambda k: dict(ATTIVITA_QUOTIDIANA_OPZIONI)[k],
        index=idx_att,
        key=f"{key_prefix}_attivita",
    )
    freq_all = st.slider(
        "Allenamenti a settimana", min_value=0, max_value=7,
        value=int(prof_attuale.get("freq_allenamento") or 0),
        key=f"{key_prefix}_freq_all",
    )
    tipologia = st.text_input(
        "Tipologia di allenamento",
        value=(prof_attuale.get("intensita_allenamento") or "") if freq_all > 0 else "",
        placeholder="es. Bodybuilding, Corsa, Nuoto, Crossfit, Calcio…",
        disabled=(freq_all == 0),
        help=(
            "Scrivi la disciplina (anche più di una) — l'IA la userà per stimare "
            "il dispendio energetico."
            if freq_all > 0 else "Disabilitato se non ti alleni."
        ),
        key=f"{key_prefix}_tipologia",
    )

    st.markdown("**3. Obiettivi e preferenze**")
    obiettivo_keys = [k for k, _ in OBIETTIVO_OPZIONI]
    idx_obj = (
        obiettivo_keys.index(prof_attuale["obiettivo"])
        if prof_attuale.get("obiettivo") in obiettivo_keys else 1
    )
    obiettivo = st.selectbox(
        "Obiettivo",
        obiettivo_keys,
        format_func=lambda k: dict(OBIETTIVO_OPZIONI)[k],
        index=idx_obj,
        key=f"{key_prefix}_obiettivo",
    )
    esperienza_keys = [k for k, _ in ESPERIENZA_OPZIONI]
    idx_esp = (
        esperienza_keys.index(prof_attuale["esperienza"])
        if prof_attuale.get("esperienza") in esperienza_keys else 0
    )
    esperienza = st.selectbox(
        "Esperienza di allenamento",
        esperienza_keys,
        format_func=lambda k: dict(ESPERIENZA_OPZIONI)[k],
        index=idx_esp,
        key=f"{key_prefix}_esperienza",
    )
    approccio_keys = [k for k, _ in APPROCCIO_OPZIONI]
    idx_appr = (
        approccio_keys.index(prof_attuale["approccio_alimentare"])
        if prof_attuale.get("approccio_alimentare") in approccio_keys else 0
    )
    approccio = st.selectbox(
        "Approccio alimentare",
        approccio_keys,
        format_func=lambda k: dict(APPROCCIO_OPZIONI)[k],
        index=idx_appr,
        key=f"{key_prefix}_approccio",
    )

    st.markdown("**4. Cucina e attrezzatura**")
    st.caption("L'IA userà queste info per suggerirti ricette compatibili con quello che puoi e ami fare.")
    cucina_keys = [k for k, _ in CUCINA_OPZIONI]
    idx_cuc = (
        cucina_keys.index(prof_attuale["abilita_cucina"])
        if prof_attuale.get("abilita_cucina") in cucina_keys else 0
    )
    abilita_cucina = st.selectbox(
        "Abilità in cucina",
        cucina_keys,
        format_func=lambda k: dict(CUCINA_OPZIONI)[k],
        index=idx_cuc,
        key=f"{key_prefix}_cucina",
    )

    attr_esistenti = prof_attuale.get("attrezzatura") or []
    if isinstance(attr_esistenti, str):
        try:
            attr_esistenti = json.loads(attr_esistenti)
        except Exception:
            attr_esistenti = []
    attr_keys = [k for k, _ in ATTREZZATURA_OPZIONI]
    attrezzatura = st.multiselect(
        "Attrezzatura disponibile",
        attr_keys,
        default=[a for a in attr_esistenti if a in attr_keys],
        format_func=lambda k: dict(ATTREZZATURA_OPZIONI)[k],
        key=f"{key_prefix}_attrezzatura",
    )

    calcola = st.button(
        testo_bottone,
        type="primary",
        use_container_width=True,
        key=f"{key_prefix}_btn_calcola",
    )

    if calcola:
        tipologia_str = (tipologia or "").strip()
        prompt_prof = (
            "Agisci come nutrizionista esperto. Dato questo profilo, calcola i target "
            "nutrizionali giornalieri.\n"
            f"- Sesso: {'Maschio' if sesso == 'M' else 'Femmina'}\n"
            f"- Età: {int(eta)}\n"
            f"- Altezza: {altezza} cm\n"
            f"- Peso: {peso} kg\n"
        )
        if aggiungi_circ:
            prompt_prof += f"- Collo: {collo} cm\n- Vita: {vita} cm\n"
        prompt_prof += (
            f"- Attività quotidiana: {dict(ATTIVITA_QUOTIDIANA_OPZIONI)[attivita]}\n"
            f"- Allenamenti: {int(freq_all)} volte a settimana\n"
        )
        if freq_all > 0 and tipologia_str:
            prompt_prof += (
                f"- Tipologia allenamento (testo libero dell'utente): "
                f"{tipologia_str}\n"
            )
        prompt_prof += (
            f"- Obiettivo: {dict(OBIETTIVO_OPZIONI)[obiettivo]}\n"
            f"- Esperienza: {dict(ESPERIENZA_OPZIONI)[esperienza]}\n"
            f"- Approccio alimentare: {dict(APPROCCIO_OPZIONI)[approccio]}\n\n"
            "Procedura:\n"
            "1. BMR con Mifflin-St Jeor.\n"
            "2. TDEE con moltiplicatore attività quotidiana "
            "(sedentario 1.2, leggero 1.4, attivo 1.55, pesante 1.725) "
            "+ aggiustamento per allenamento: valuta la tipologia indicata "
            "dall'utente (es. bodybuilding/forza ~+0.04 per sessione; corsa, "
            "nuoto, crossfit, ciclismo intenso ~+0.06 per sessione; discipline "
            "leggere come yoga o camminata ~+0.02 per sessione). Se la tipologia "
            "non è indicata o non è riconoscibile, assumi +0.04 per sessione.\n"
            "3. Calorie obiettivo: deficit -15/-20%, mantenimento invariato, surplus +10/+15%.\n"
            "4. Proteine g/kg di peso: principiante 1.6, intermedio 1.8, avanzato 2.0.\n"
            "5. Distribuzione macro residua:\n"
            "   - equilibrato: grassi ~30% kcal totali, resto carbi\n"
            "   - lowcarb: carbi ~20%, grassi ~45%\n"
            "   - cheto: carbi <10% (max 50 g), grassi ~70%\n"
            "   - vegano: stessa distribuzione di equilibrato\n\n"
            "Output ESCLUSIVO (solo JSON valido, senza code fence, senza commenti):\n"
            "{\n"
            '  "target_kcal": number,\n'
            '  "target_pro": number,\n'
            '  "target_carbi": number,\n'
            '  "target_grassi": number,\n'
            '  "note": "string breve (1-2 righe), anche vuota"\n'
            "}"
        )
        try:
            with st.spinner("🤖 Calcolo in corso..."):
                res_ai = _ai_generate(prompt_prof, models=AI_MODELS_TESTO)
            raw = res_ai.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            elif raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            targets = json.loads(raw.strip())

            payload_prof = {
                "user_id": user_id,
                "sesso": sesso,
                "eta": int(eta),
                "altezza_cm": float(altezza),
                "peso_kg": float(peso),
                "collo_cm": float(collo) if collo is not None else None,
                "vita_cm": float(vita) if vita is not None else None,
                "attivita_quotidiana": attivita,
                "freq_allenamento": int(freq_all),
                "intensita_allenamento": (tipologia_str or None) if freq_all > 0 else None,
                "obiettivo": obiettivo,
                "esperienza": esperienza,
                "approccio_alimentare": approccio,
                "abilita_cucina": abilita_cucina,
                "attrezzatura": attrezzatura,
                "target_kcal": _to_float(targets.get("target_kcal")),
                "target_pro": _to_float(targets.get("target_pro")),
                "target_carbi": _to_float(targets.get("target_carbi")),
                "target_grassi": _to_float(targets.get("target_grassi")),
                "note": targets.get("note") or None,
            }
            (
                supabase.table("profilo_utente")
                .upsert(payload_prof, on_conflict="user_id").execute()
            )
            st.session_state.profilo = payload_prof
            st.success("Obiettivi aggiornati.")
            st.rerun()
        except json.JSONDecodeError:
            st.error("L'IA ha risposto in un formato non valido. Riprova.")
        except Exception as e:
            st.error(f"Errore: {e}")

    if mostra_target and prof_attuale.get("target_kcal"):
        st.markdown("---")
        st.caption("🎯 Target giornaliero attuale")
        st.metric("kcal", f"{prof_attuale['target_kcal']:.0f}")
        st.caption(
            f"Carbi {prof_attuale.get('target_carbi') or 0:.0f} g · "
            f"Pro {prof_attuale.get('target_pro') or 0:.0f} g · "
            f"Grassi {prof_attuale.get('target_grassi') or 0:.0f} g"
        )
        if prof_attuale.get("note"):
            st.caption(prof_attuale["note"])

# --- INTERFACCIA DI ACCESSO ---
if st.session_state.user is None:
    st.title("🔐 Accesso Smart Pantry")
    menu = ["Login", "Registrati"]
    choice = st.sidebar.selectbox("Menu", menu)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Entra"):
            login(email, password)
    else:
        if st.button("Crea Account"):
            sign_up(email, password)
    st.stop()  # Blocca il resto dell'app se non loggato

# --- SEZIONE APP (UTENTE LOGGATO) ---
if st.session_state.user is None:
    st.warning("Devi prima effettuare il login per usare la tua dispensa.")
    st.stop()
else:
    user_id = st.session_state.user.id
    st.sidebar.write(f"Connesso come: {st.session_state.user.email}")
    if st.sidebar.button("Logout"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.conferma_eliminazione = None
        st.session_state.profilo = None
        st.rerun()

    if st.session_state.profilo is None:
        try:
            res_prof = (
                supabase.table("profilo_utente").select("*")
                .eq("user_id", user_id).limit(1).execute()
            )
            st.session_state.profilo = res_prof.data[0] if res_prof.data else {}
        except Exception:
            st.session_state.profilo = {}

    prof_attuale = st.session_state.profilo or {}
    profilo_presente = bool(prof_attuale.get("target_kcal"))

    # --- ONBOARDING: se il profilo non è ancora configurato, blocca l'app
    # e mostra un wizard a tutta pagina che raccoglie profilo + obiettivi
    # + abilità in cucina + attrezzatura.
    if not profilo_presente:
        st.title("👋 Benvenuto in Smart Pantry!")
        st.markdown(
            "Prima di iniziare, completa questo breve profilo. "
            "Servirà a calcolare il tuo fabbisogno giornaliero e a suggerirti "
            "ricette adatte alla tua cucina e alla tua attrezzatura."
        )
        st.divider()
        _form_profilo_e_salva(
            prof_attuale, user_id,
            key_prefix="onb",
            mostra_target=False,
            testo_bottone="✅ Salva profilo e inizia",
        )
        st.stop()

    with st.sidebar.expander("Impostazioni pasti"):
        st.caption("Scegli quali pasti mostrare nel menu del diario.")
        abilitati_new = [
            p for p in ORDINE_PASTI
            if st.checkbox(
                ETICHETTE_PASTI[p],
                value=(p in st.session_state.pasti_abilitati),
                key=f"chk_pasto_{p}",
            )
        ]
        st.session_state.pasti_abilitati = abilitati_new
        if not abilitati_new:
            st.warning("Abilitane almeno uno per poter registrare pasti.")

    with st.sidebar.expander(
        "Profilo e obiettivi" + (" ✓" if profilo_presente else ""),
        expanded=False,
    ):
        _form_profilo_e_salva(prof_attuale, user_id, key_prefix="prof")

    tab1, tab2, tab_spesa, tab3, tab4, tab_prog, tab_port = st.tabs(
        [
            "📸 Aggiungi", "📦 Dispensa", "🛒 Spesa", "🍳 Ricette AI",
            "🍽️ Diario", "📈 Progressi", "💰 Portafoglio",
        ]
    )

    with tab1:
        st.header("Aggiungi la spesa")
        st.caption(
            "Carica la foto dello scontrino e le foto dei prodotti acquistati. "
            "L'AI userà lo scontrino come guida e le foto per completare nomi, "
            "quantità e valori nutrizionali."
        )
        st.markdown(
            """
            <style>
            [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
            section[data-testid="stFileUploaderDropzone"] button { font-size: 0 !important; }
            section[data-testid="stFileUploaderDropzone"] button::before {
                content: "Foto";
                font-size: 0.875rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        up_scontrino = st.file_uploader(
            "Foto dello scontrino",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key=f"uploader_sc_{st.session_state.uploader_key}",
        )
        up_prodotti = st.file_uploader(
            "Foto dei prodotti acquistati",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_prod_{st.session_state.uploader_key}",
        )

        if up_scontrino is not None:
            img_scontrino = Image.open(up_scontrino)
            imgs_prodotti = [Image.open(f) for f in (up_prodotti or [])]

            cols_prev = st.columns(min(1 + len(imgs_prodotti), 4) or 1)
            with cols_prev[0]:
                st.image(img_scontrino, width=180, caption="Scontrino")
            for i, im in enumerate(imgs_prodotti):
                with cols_prev[(i + 1) % len(cols_prev)]:
                    st.image(im, width=160, caption=f"Prodotto {i + 1}")

            pronti = up_scontrino is not None and len(imgs_prodotti) > 0
            if not pronti:
                st.info("Carica almeno una foto dei prodotti per abilitare l'analisi.")

            if st.button("🤖 Analizza spesa", disabled=not pronti):
                prompt = (
                    "Ti vengono fornite più immagini della stessa spesa. "
                    "LA PRIMA IMMAGINE è lo SCONTRINO ed è la fonte AUTORITATIVA per "
                    "lista voci, prezzi, data e totale. LE IMMAGINI SUCCESSIVE mostrano "
                    "i prodotti acquistati: usale per leggere nomi completi, pesi netti, "
                    "valori nutrizionali e categorie. "
                    "NON aggiungere voci che non compaiono sullo scontrino e NON duplicarle.\n\n"
                    "Restituisci ESCLUSIVAMENTE un JSON valido con queste chiavi:\n"
                    "- data (string, formato YYYY-MM-DD): data dello scontrino; null se non leggibile.\n"
                    "- negozio (string|null): nome del supermercato/negozio.\n"
                    "- totale (number|null): totale pagato in euro.\n"
                    "- voci (array): ciascuna voce con:\n"
                    "    - nome (string): nome SEMPLIFICATO e generico (senza marca/codici). "
                    "Esempi: 'Olive verdi denocciolate', 'Spaghetti', 'Pan bauletto integrale'.\n"
                    "    - categoria (string): UNA fra "
                    "['Frutta e verdura','Carne e pesce','Latticini e uova','Pane e cereali',"
                    "'Scatolame e conserve','Surgelati','Bevande','Dolci e snack','Casa e cura','Altro'].\n"
                    "    - prezzo (number): prezzo pagato in euro (dopo sconti riga).\n"
                    "    - quantita_g (number|null): peso netto della confezione in grammi. "
                    "Prima cercalo sullo scontrino, poi sulla confezione nelle foto. "
                    "Converti kg→g (0.5 kg → 500) e ml→g (1 ml = 1 g). Null se non ricavabile.\n"
                    "    - kcal (number|null): energia per 100 g (dalla tabella nutrizionale nelle foto).\n"
                    "    - carbi (number|null): carboidrati totali per 100 g.\n"
                    "    - pro (number|null): proteine per 100 g.\n"
                    "    - grassi (number|null): grassi totali per 100 g.\n"
                    "IGNORA sottovoci 'di cui saturi', 'di cui zuccheri' ecc.: leggi solo i totali. "
                    "Ignora righe di subtotale, sconti totali, totale, IVA, resto, pagamento. "
                    "Se un campo non è leggibile/ricavabile, usa null (non inventare, non usare 0 come fallback). "
                    "Non aggiungere spiegazioni o code fence: solo JSON puro."
                )
                def _parse_json_ai(testo):
                    t = (testo or "").strip()
                    if t.startswith("```json"):
                        t = t[7:]
                    elif t.startswith("```"):
                        t = t[3:]
                    if t.endswith("```"):
                        t = t[:-3]
                    return json.loads(t.strip())

                try:
                    with st.spinner("🤖 Analizzo scontrino e prodotti..."):
                        res = _ai_generate(
                            [prompt, img_scontrino] + imgs_prodotti,
                            models=AI_MODELS_IMMAGINI,
                        )
                    raw = _parse_json_ai(res.text)
                    voci = raw.get("voci") or []
                    righe = []
                    for v in voci:
                        nome_v = (v.get("nome") or "").strip()
                        if not nome_v:
                            continue
                        prezzo_v = _to_float(v.get("prezzo"))
                        if prezzo_v is None:
                            continue
                        qg = _to_float(v.get("quantita_g"))
                        prezzo_kg = (prezzo_v / (qg / 1000.0)) if qg and qg > 0 else None
                        righe.append({
                            "nome": nome_v,
                            "categoria": v.get("categoria") or "Altro",
                            "prezzo": float(prezzo_v),
                            "quantita_g": float(qg) if qg is not None else None,
                            "prezzo_kg": round(prezzo_kg, 2) if prezzo_kg is not None else None,
                            "kcal": _to_float(v.get("kcal")),
                            "carbi": _to_float(v.get("carbi")),
                            "pro": _to_float(v.get("pro")),
                            "grassi": _to_float(v.get("grassi")),
                        })
                    st.session_state.spesa_preview = {
                        "data": raw.get("data") or date.today().isoformat(),
                        "negozio": raw.get("negozio"),
                        "totale": _to_float(raw.get("totale")),
                        "righe": righe,
                    }
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Risposta AI non in JSON valido. Riprova con foto più nitide.")
                except Exception as e:
                    if "API_KEY" in str(e) or "invalid" in str(e).lower():
                        st.error("Chiave API Gemini non valida. Aggiorna la chiave in .streamlit/secrets.toml")
                    elif _is_quota_error(e):
                        st.error("Limite di richieste Gemini raggiunto. Attendi qualche minuto e riprova.")
                    else:
                        st.error(f"Errore AI: {e}")

        preview_sp = st.session_state.get("spesa_preview")
        if preview_sp:
            st.divider()
            st.subheader("✏️ Controlla le voci prima di salvare")

            col_d, col_n = st.columns(2)
            with col_d:
                try:
                    data_default_sp = date.fromisoformat(preview_sp["data"])
                except Exception:
                    data_default_sp = date.today()
                data_sp = st.date_input(
                    "Data scontrino", value=data_default_sp,
                    key="sp_data_edit", format="DD/MM/YYYY",
                )
            with col_n:
                negozio_sp = st.text_input(
                    "Negozio", value=preview_sp.get("negozio") or "",
                    key="sp_negozio_edit",
                )
            if preview_sp.get("totale") is not None:
                st.caption(f"Totale rilevato: € {preview_sp['totale']:.2f}")

            df_sp = pd.DataFrame(preview_sp["righe"])
            categorie_opts = [
                "Frutta e verdura", "Carne e pesce", "Latticini e uova",
                "Pane e cereali", "Scatolame e conserve", "Surgelati",
                "Bevande", "Dolci e snack", "Casa e cura", "Altro",
            ]
            edited_sp = st.data_editor(
                df_sp, hide_index=True, num_rows="dynamic",
                use_container_width=True, key="sp_editor",
                column_config={
                    "nome": st.column_config.TextColumn("Prodotto"),
                    "categoria": st.column_config.SelectboxColumn(
                        "Categoria", options=categorie_opts,
                    ),
                    "prezzo": st.column_config.NumberColumn(
                        "Prezzo (€)", min_value=0.0, step=0.01, format="%.2f",
                    ),
                    "quantita_g": st.column_config.NumberColumn(
                        "Quantità (g)", min_value=0.0, step=1.0,
                    ),
                    "prezzo_kg": st.column_config.NumberColumn(
                        "€ / kg", format="%.2f", disabled=True,
                    ),
                    "kcal": st.column_config.NumberColumn("kcal/100g", min_value=0.0, step=1.0),
                    "carbi": st.column_config.NumberColumn("Carbi/100g", min_value=0.0, step=0.1),
                    "pro": st.column_config.NumberColumn("Pro/100g", min_value=0.0, step=0.1),
                    "grassi": st.column_config.NumberColumn("Grassi/100g", min_value=0.0, step=0.1),
                },
            )
            st.caption(
                "Le voci con quantità in grammi verranno aggiunte alla dispensa "
                "(sommando alla quantità esistente se il prodotto è già presente). "
                "Tutte le voci vengono registrate nello storico prezzi."
            )

            c1, c2, _ = st.columns([2, 2, 4])
            with c1:
                if st.button("💾 Salva spesa", type="primary", use_container_width=True):
                    errori = []
                    n_prezzi = 0
                    n_disp = 0
                    for _, r in edited_sp.iterrows():
                        nome_r = str(r.get("nome") or "").strip()
                        prezzo_r = _to_float(r.get("prezzo"))
                        if not nome_r or prezzo_r is None or prezzo_r <= 0:
                            continue
                        qg_r = _to_float(r.get("quantita_g"))
                        p_kg = (prezzo_r / (qg_r / 1000.0)) if qg_r and qg_r > 0 else None

                        try:
                            supabase.table("storico_prezzi").insert({
                                "user_id": user_id,
                                "data": data_sp.isoformat(),
                                "nome": nome_r,
                                "categoria": r.get("categoria") or "Altro",
                                "prezzo": float(prezzo_r),
                                "quantita_g": float(qg_r) if qg_r is not None else None,
                                "prezzo_kg": round(p_kg, 2) if p_kg is not None else None,
                                "negozio": negozio_sp or None,
                            }).execute()
                            n_prezzi += 1
                        except Exception as e:
                            errori.append(f"Storico {nome_r}: {e}")

                        if qg_r and qg_r > 0:
                            try:
                                existing = (
                                    supabase.table("dispensa").select("id, quantita")
                                    .eq("user_id", user_id).eq("nome", nome_r)
                                    .limit(1).execute()
                                )
                                if existing.data:
                                    riga_e = existing.data[0]
                                    q_totale = float(riga_e.get("quantita") or 0) + float(qg_r)
                                    (
                                        supabase.table("dispensa").update({"quantita": q_totale})
                                        .eq("id", riga_e["id"]).eq("user_id", user_id).execute()
                                    )
                                else:
                                    supabase.table("dispensa").insert({
                                        "user_id": user_id,
                                        "nome": nome_r,
                                        "quantita": float(qg_r),
                                        "kcal": _to_float(r.get("kcal")),
                                        "carbi": _to_float(r.get("carbi")),
                                        "pro": _to_float(r.get("pro")),
                                        "grassi": _to_float(r.get("grassi")),
                                    }).execute()
                                n_disp += 1
                            except Exception as e:
                                errori.append(f"Dispensa {nome_r}: {e}")

                    if errori:
                        st.error("Alcune voci non salvate:\n- " + "\n- ".join(errori))
                    else:
                        st.success(
                            f"Salvate {n_prezzi} voci nello storico prezzi · "
                            f"{n_disp} aggiunte/aggiornate in dispensa."
                        )
                        st.session_state.spesa_preview = None
                        st.session_state.uploader_key += 1
                        st.rerun()
            with c2:
                if st.button("Annulla", use_container_width=True):
                    st.session_state.spesa_preview = None
                    st.rerun()

    with tab2:
        st.header("La tua Dispensa")

        res = supabase.table("dispensa").select("*").eq("user_id", user_id).execute()

        if not res.data:
            st.info("La tua dispensa è vuota. Aggiungi il primo prodotto dalla tab '📸 Aggiungi'.")
        else:
            prodotti_disp = sorted(res.data, key=lambda p: (p.get("nome") or "").lower())
            st.caption("Clicca su un prodotto per modificarne i valori o eliminarlo.")

            for p in prodotti_disp:
                pid = p["id"]
                nome_p = p.get("nome") or "(senza nome)"
                qta_p = _to_float(p.get("quantita"))
                kcal_p = _to_float(p.get("kcal"))
                parti = [f"**{nome_p}**"]
                if qta_p is not None:
                    parti.append(f"{qta_p:g} g")
                if kcal_p is not None:
                    parti.append(f"{kcal_p:g} kcal/100g")
                header = " · ".join(parti)

                with st.expander(header):
                    col1, col2 = st.columns(2)
                    with col1:
                        nome_v = st.text_input(
                            "Nome", value=nome_p, key=f"disp_nome_{pid}"
                        )
                        quantita_v = st.number_input(
                            "Quantità (g)",
                            value=float(qta_p) if qta_p is not None else 0.0,
                            min_value=0.0, step=1.0,
                            key=f"disp_qta_{pid}",
                        )
                        kcal_v = st.number_input(
                            "kcal / 100 g",
                            value=float(kcal_p) if kcal_p is not None else 0.0,
                            min_value=0.0, step=1.0,
                            key=f"disp_kcal_{pid}",
                        )
                    with col2:
                        carbi_p = _to_float(p.get("carbi"))
                        pro_p = _to_float(p.get("pro"))
                        grassi_p = _to_float(p.get("grassi"))
                        carbi_v = st.number_input(
                            "Carboidrati / 100 g",
                            value=float(carbi_p) if carbi_p is not None else 0.0,
                            min_value=0.0, step=0.1,
                            key=f"disp_carbi_{pid}",
                        )
                        pro_v = st.number_input(
                            "Proteine / 100 g",
                            value=float(pro_p) if pro_p is not None else 0.0,
                            min_value=0.0, step=0.1,
                            key=f"disp_pro_{pid}",
                        )
                        grassi_v = st.number_input(
                            "Grassi / 100 g",
                            value=float(grassi_p) if grassi_p is not None else 0.0,
                            min_value=0.0, step=0.1,
                            key=f"disp_grassi_{pid}",
                        )

                    cb1, cb2, _ = st.columns([2, 2, 4])
                    with cb1:
                        if st.button("💾 Salva", type="primary",
                                     key=f"disp_save_{pid}", use_container_width=True):
                            if not (nome_v or "").strip():
                                st.error("Il nome è obbligatorio.")
                            else:
                                payload = {
                                    "nome": nome_v.strip(),
                                    "quantita": float(quantita_v),
                                    "kcal": float(kcal_v),
                                    "carbi": float(carbi_v),
                                    "pro": float(pro_v),
                                    "grassi": float(grassi_v),
                                }
                                try:
                                    if payload["quantita"] <= 0:
                                        (
                                            supabase.table("dispensa").delete()
                                            .eq("id", pid).eq("user_id", user_id).execute()
                                        )
                                        _aggiungi_a_lista_spesa(payload["nome"], user_id)
                                        st.success(f"'{payload['nome']}' spostato nella lista della spesa.")
                                    else:
                                        (
                                            supabase.table("dispensa").update(payload)
                                            .eq("id", pid).eq("user_id", user_id).execute()
                                        )
                                        st.success(f"'{payload['nome']}' aggiornato.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Errore salvataggio: {e}")
                    with cb2:
                        if st.button("🗑️ Elimina", key=f"disp_del_{pid}",
                                     use_container_width=True):
                            st.session_state.conferma_eliminazione = [pid]
                            st.rerun()

            # --- Pannello di conferma eliminazione ---
            if st.session_state.conferma_eliminazione:
                ids_to_del = st.session_state.conferma_eliminazione
                nomi_to_del = [
                    p.get("nome") for p in prodotti_disp if p["id"] in ids_to_del
                ]
                st.warning(
                    f"Stai per eliminare **{len(ids_to_del)}** prodott"
                    f"{'o' if len(ids_to_del) == 1 else 'i'}: "
                    f"{', '.join(n for n in nomi_to_del if n)}. L'azione è irreversibile."
                )
                c1, c2, _ = st.columns([2, 2, 4])
                with c1:
                    if st.button("✅ Conferma", type="primary", use_container_width=True):
                        errori = []
                        for row_id in ids_to_del:
                            try:
                                (
                                    supabase.table("dispensa")
                                    .delete()
                                    .eq("id", row_id)
                                    .eq("user_id", user_id)
                                    .execute()
                                )
                            except Exception as e:
                                errori.append(f"Riga {row_id}: {e}")
                        st.session_state.conferma_eliminazione = None
                        if errori:
                            st.error("Alcune eliminazioni non sono andate a buon fine:\n" + "\n".join(errori))
                        else:
                            st.success(f"Eliminat{'o' if len(ids_to_del) == 1 else 'i'} {len(ids_to_del)} prodott{'o' if len(ids_to_del) == 1 else 'i'}.")
                        st.rerun()
                with c2:
                    if st.button("Annulla", use_container_width=True):
                        st.session_state.conferma_eliminazione = None
                        st.rerun()

    with tab_spesa:
        st.header("Lista della spesa")
        st.caption(
            "I prodotti finiti (quantità a 0) vengono spostati qui automaticamente. "
            "Seleziona le voci da rimuovere quando le hai ricomprate."
        )
        res_sp = (
            supabase.table("lista_spesa").select("*")
            .eq("user_id", user_id).order("created_at", desc=True).execute()
        )
        items_sp = res_sp.data or []
        if not items_sp:
            st.info("La lista della spesa è vuota.")
        else:
            df_sp = pd.DataFrame(items_sp)
            df_sp_edit = df_sp[["id", "nome"]].copy()
            df_sp_edit.insert(0, "seleziona", False)
            edited_sp = st.data_editor(
                df_sp_edit,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "id": None,
                    "seleziona": st.column_config.CheckboxColumn(
                        "✓", default=False, width="small",
                    ),
                    "nome": st.column_config.TextColumn("Nome", disabled=True),
                },
                disabled=["nome"],
                use_container_width=True,
                key="spesa_editor",
            )
            ids_sp_sel = [int(i) for i in edited_sp[edited_sp["seleziona"] == True]["id"].tolist()]
            if ids_sp_sel:
                if st.button(
                    f"🗑️ Rimuovi dalla lista ({len(ids_sp_sel)})",
                    type="primary",
                    key="spesa_rimuovi",
                ):
                    errori = []
                    for row_id in ids_sp_sel:
                        try:
                            (
                                supabase.table("lista_spesa").delete()
                                .eq("id", row_id).eq("user_id", user_id).execute()
                            )
                        except Exception as e:
                            errori.append(f"Riga {row_id}: {e}")
                    if errori:
                        st.error("Alcune rimozioni non sono andate a buon fine:\n" + "\n".join(errori))
                    else:
                        st.success(
                            f"Rimoss{'o' if len(ids_sp_sel) == 1 else 'i'} {len(ids_sp_sel)} voc{'e' if len(ids_sp_sel) == 1 else 'i'}."
                        )
                        st.rerun()

    with tab3:
        st.header("Consiglio dello Chef AI")
        res_db_chef = (
            supabase.table("dispensa")
            .select("nome, quantita, kcal, carbi, pro, grassi")
            .eq("user_id", user_id).execute()
        )
        ingredienti_chef = res_db_chef.data or []
        mappa_disp_chef = {i["nome"]: i for i in ingredienti_chef}

        def _chef_store_response(testo_ai_):
            m_json_ = re.search(r"```json\s*(.+?)\s*```", testo_ai_, re.S)
            ingr_ai_ = []
            if m_json_:
                try:
                    raw_ = json.loads(m_json_.group(1))
                    for r_ in raw_ if isinstance(raw_, list) else []:
                        nome_r_ = (r_ or {}).get("nome")
                        qta_r_ = _to_float((r_ or {}).get("quantita_g"))
                        if nome_r_ and qta_r_ and nome_r_ in mappa_disp_chef:
                            ingr_ai_.append({"nome": nome_r_, "quantita_g": float(qta_r_)})
                except Exception:
                    ingr_ai_ = []
            testo_vis_ = re.sub(
                r"```json\s*.+?\s*```", "", testo_ai_, flags=re.S
            ).strip()
            titolo_ = ""
            for _l in testo_vis_.splitlines():
                s_ = _l.strip().lstrip("#").strip().strip("*_").strip()
                if s_:
                    titolo_ = s_[:120]
                    break
            st.session_state.chef_ricetta_text = testo_vis_
            st.session_state.chef_ricetta_titolo = titolo_
            st.session_state.chef_ricetta_ingredienti = ingr_ai_
            st.session_state.chef_confirm_mode = False
            for it_ in ingr_ai_:
                qta_disp_ = _to_float(
                    mappa_disp_chef.get(it_["nome"], {}).get("quantita")
                ) or 0.0
                proposta_ = min(it_["quantita_g"], qta_disp_) if qta_disp_ > 0 else it_["quantita_g"]
                st.session_state[f"chef_qty_{it_['nome']}"] = float(proposta_)

        if st.button("Cosa cucino oggi?"):
            ingredienti = ingredienti_chef
            if ingredienti:
                def _fmt_num(v, suffisso=""):
                    n = _to_float(v)
                    return f"{n:g}{suffisso}" if n is not None else "n/d"

                righe_ing = []
                for i in ingredienti:
                    righe_ing.append(
                        f"- {i.get('nome')}: "
                        f"disponibili {_fmt_num(i.get('quantita'), ' g')}, "
                        f"valori per 100 g → "
                        f"{_fmt_num(i.get('kcal'))} kcal, "
                        f"carbi {_fmt_num(i.get('carbi'), ' g')}, "
                        f"pro {_fmt_num(i.get('pro'), ' g')}, "
                        f"grassi {_fmt_num(i.get('grassi'), ' g')}"
                    )
                lista_ing = "\n".join(righe_ing)

                cucina_label = dict(CUCINA_OPZIONI).get(prof_attuale.get("abilita_cucina"))
                attr_user = prof_attuale.get("attrezzatura") or []
                if isinstance(attr_user, str):
                    try:
                        attr_user = json.loads(attr_user)
                    except Exception:
                        attr_user = []
                attr_labels = [dict(ATTREZZATURA_OPZIONI).get(a) for a in attr_user]
                attr_labels = [a for a in attr_labels if a]

                vincoli = []
                if cucina_label:
                    vincoli.append(f"- Abilità in cucina dell'utente: {cucina_label}")
                if attr_labels:
                    vincoli.append(
                        "- Attrezzatura disponibile (usa SOLO questi strumenti): "
                        + ", ".join(attr_labels)
                    )
                else:
                    vincoli.append(
                        "- Attrezzatura non specificata: proponi una ricetta che richieda solo "
                        "strumenti di base (pentola, padella, fornello)."
                    )
                blocco_vincoli = "\n".join(vincoli)

                p = (
                    "Suggerisci una ricetta usando gli ingredienti sotto elencati. "
                    "Per ciascun ingrediente hai la quantità disponibile in dispensa (in grammi) "
                    "e i valori nutrizionali per 100 g. Rispetta i limiti di disponibilità "
                    "(non usare più di quanto indicato). Adatta la complessità della ricetta al "
                    "livello dell'utente (principiante = passaggi semplici e pochi strumenti; "
                    "avanzato = puoi osare di più) e proponi SOLO tecniche realizzabili con "
                    "l'attrezzatura elencata. Alla fine della ricetta, riporta una stima dei "
                    "macro totali del piatto (kcal, carbi, proteine, grassi). "
                    "Mantieni la risposta molto sintetica: riduci della metà la lunghezza "
                    "tipica, usa frasi brevi ed elenchi essenziali, senza introduzioni o "
                    "commenti superflui.\n"
                    "ALLA FINE, su una nuova riga, includi SOLO un blocco ```json ... ``` "
                    "con la lista degli ingredienti usati, nel formato: "
                    "[{\"nome\": \"<nome esatto come nella dispensa>\", \"quantita_g\": <numero>}]. "
                    "I nomi DEVONO corrispondere esattamente a quelli della dispensa.\n\n"
                    f"Vincoli dell'utente:\n{blocco_vincoli}\n\n"
                    f"Ingredienti disponibili:\n{lista_ing}"
                )
                try:
                    with st.spinner("🤖 Lo chef AI sta pensando a una ricetta..."):
                        response = _ai_generate(p)
                    _chef_store_response(response.text or "")
                    st.session_state.chef_prompt_base = p
                    st.session_state.chef_chat_history = []
                except Exception as e:
                    if "API_KEY" in str(e) or "invalid" in str(e).lower():
                        st.error("Chiave API Gemini non valida. Aggiorna la chiave in .streamlit/secrets.toml")
                    elif _is_quota_error(e):
                        st.error("Limite di richieste Gemini raggiunto. Attendi qualche minuto e riprova.")
                    else:
                        st.error(f"Errore AI: {e}")
            else:
                st.warning("Aggiungi prima qualcosa in dispensa!")

        ricetta_text = st.session_state.get("chef_ricetta_text")
        if ricetta_text:
            st.markdown(ricetta_text)

            if not st.session_state.get("chef_confirm_mode") and st.session_state.get("chef_prompt_base"):
                with st.expander("💬 Chiedi una variante allo chef", expanded=False):
                    storico = st.session_state.get("chef_chat_history") or []
                    variante_txt = st.text_area(
                        "Cosa vuoi cambiare?",
                        placeholder="Es: Troppo complicato, non ho voglia di usare il forno, proponimi una variante in padella",
                        key="chef_chat_input",
                        height=80,
                    )
                    if st.button("🔁 Rigenera ricetta", key="chef_chat_send"):
                        req = (variante_txt or "").strip()
                        if not req:
                            st.warning("Scrivi una richiesta prima di rigenerare.")
                        else:
                            nuovo_storico = storico + [req]
                            prompt_r = (
                                st.session_state.chef_prompt_base
                                + "\n\n---\nRicetta precedente:\n"
                                + st.session_state.chef_ricetta_text
                                + "\n\nRichieste successive dell'utente (dalla più vecchia alla più recente):\n"
                                + "\n".join(f"- {q}" for q in nuovo_storico)
                                + "\n\nProponi una NUOVA ricetta che soddisfi l'ultima richiesta, "
                                  "rispettando i vincoli iniziali e mantenendo lo stesso formato "
                                  "di output (incluso il blocco ```json``` finale con gli ingredienti usati)."
                            )
                            try:
                                with st.spinner("🤖 Lo chef AI sta rivedendo la ricetta..."):
                                    resp_r = _ai_generate(prompt_r)
                                _chef_store_response(resp_r.text or "")
                                st.session_state.chef_chat_history = nuovo_storico
                                st.rerun()
                            except Exception as e:
                                if _is_quota_error(e):
                                    st.error("Limite di richieste Gemini raggiunto. Riprova tra poco.")
                                else:
                                    st.error(f"Errore AI: {e}")

            ingr_ricetta = st.session_state.get("chef_ricetta_ingredienti") or []

            if not ingr_ricetta:
                st.info(
                    "Non è stato possibile estrarre la lista ingredienti dalla ricetta: "
                    "registrala manualmente dal Diario."
                )
            elif not st.session_state.get("chef_confirm_mode"):
                if st.button("✅ Seleziona questa ricetta", key="chef_select_btn"):
                    st.session_state.chef_confirm_mode = True
                    st.rerun()
            else:
                st.subheader("Conferma quantità")
                st.caption("Regola con ➖ / ➕ (step = 10% della quantità in dispensa).")
                for it in ingr_ricetta:
                    nome_i = it["nome"]
                    key_q = f"chef_qty_{nome_i}"
                    max_g = _to_float(
                        mappa_disp_chef.get(nome_i, {}).get("quantita")
                    ) or 0.0
                    step = round(max_g * 0.1, 2) if max_g > 0 else 10.0
                    if step <= 0:
                        step = 1.0
                    cur = float(st.session_state.get(key_q, it["quantita_g"]))
                    c1, c2, c3, c4 = st.columns([4, 1, 1, 2])
                    c1.markdown(f"**{nome_i}** — disponibile: {max_g:g} g")
                    if c2.button("➖", key=f"chef_minus_{nome_i}"):
                        st.session_state[key_q] = max(0.0, round(cur - step, 2))
                        st.rerun()
                    if c3.button("➕", key=f"chef_plus_{nome_i}"):
                        nuovo = round(cur + step, 2)
                        if max_g > 0:
                            nuovo = min(max_g, nuovo)
                        st.session_state[key_q] = nuovo
                        st.rerun()
                    c4.markdown(f"**{st.session_state[key_q]:g} g**")

                abilitati_chef = st.session_state.pasti_abilitati or list(ORDINE_PASTI)
                col_pc, col_dc = st.columns(2)
                with col_pc:
                    pasto_chef = st.selectbox(
                        "Pasto",
                        abilitati_chef,
                        format_func=lambda p: ETICHETTE_PASTI.get(p, p),
                        key="chef_pasto_sel",
                    )
                with col_dc:
                    oggi_chef = date.today()
                    opzioni_date_chef = [
                        oggi_chef + timedelta(days=i) for i in range(-7, 3)
                    ]
                    data_chef = st.selectbox(
                        "Data",
                        opzioni_date_chef,
                        index=opzioni_date_chef.index(oggi_chef),
                        format_func=lambda d: _data_it(d).capitalize(),
                        key="chef_data_sel",
                    )

                col_ok, col_ko = st.columns(2)
                with col_ok:
                    if st.button("✔️ Conferma e registra", type="primary", key="chef_confirm_btn"):
                        errori = []
                        registrati = 0
                        for it in ingr_ricetta:
                            nome_i = it["nome"]
                            grammi = float(
                                st.session_state.get(f"chef_qty_{nome_i}", 0) or 0
                            )
                            if grammi <= 0:
                                continue
                            prod = mappa_disp_chef.get(nome_i, {})
                            fattore = grammi / 100.0
                            payload = {
                                "user_id": user_id,
                                "data": data_chef.isoformat(),
                                "pasto": pasto_chef,
                                "nome": nome_i,
                                "quantita": grammi,
                                "kcal": (prod.get("kcal") or 0) * fattore,
                                "carbi": (prod.get("carbi") or 0) * fattore,
                                "pro": (prod.get("pro") or 0) * fattore,
                                "grassi": (prod.get("grassi") or 0) * fattore,
                                "ricetta": st.session_state.get("chef_ricetta_titolo") or None,
                            }
                            try:
                                supabase.table("consumi").insert(payload).execute()
                                _applica_delta_dispensa(nome_i, grammi, user_id)
                                registrati += 1
                            except Exception as e:
                                errori.append(f"{nome_i}: {e}")
                        if errori:
                            st.error("Errori:\n- " + "\n- ".join(errori))
                        else:
                            st.success(
                                f"Registrati {registrati} ingredienti nel diario; dispensa aggiornata."
                            )
                            for it in ingr_ricetta:
                                st.session_state.pop(f"chef_qty_{it['nome']}", None)
                            st.session_state.chef_ricetta_text = None
                            st.session_state.chef_ricetta_ingredienti = None
                            st.session_state.chef_confirm_mode = False
                            st.rerun()
                with col_ko:
                    if st.button("✖️ Annulla", key="chef_cancel_btn"):
                        st.session_state.chef_confirm_mode = False
                        st.rerun()

    with tab4:
        st.header("Diario Alimentare")

        # --- Form per registrare un consumo ---
        st.subheader("➕ Registra un pasto")
        res_disp = supabase.table("dispensa").select("*").eq("user_id", user_id).execute()
        prodotti = res_disp.data or []

        if not prodotti:
            st.info("La dispensa è vuota: aggiungi prodotti prima di registrare consumi.")
        elif not st.session_state.pasti_abilitati:
            st.warning(
                "Nessun pasto abilitato. Apri 'Impostazioni pasti' nella sidebar e "
                "attivane almeno uno."
            )
        else:
            oggi_iso = date.today().isoformat()
            try:
                res_oggi = (
                    supabase.table("consumi").select("pasto")
                    .eq("user_id", user_id).eq("data", oggi_iso).execute()
                )
                pasti_oggi = {r.get("pasto") for r in (res_oggi.data or []) if r.get("pasto")}
            except Exception:
                pasti_oggi = set()

            abilitati = st.session_state.pasti_abilitati
            def_idx = 0
            for i, p in enumerate(abilitati):
                if p not in pasti_oggi:
                    def_idx = i
                    break

            pasto_sel = st.selectbox(
                "Pasto",
                abilitati,
                format_func=lambda p: ETICHETTE_PASTI[p],
                index=def_idx,
                key="diario_pasto",
            )

            mappa_prodotti = {p["nome"]: p for p in prodotti}
            col_p, col_q, col_d = st.columns([3, 2, 2])
            with col_p:
                nome_sel = st.selectbox("Prodotto", list(mappa_prodotti.keys()), key="diario_prod")

            prodotto_sel = mappa_prodotti[nome_sel]
            qta_conf = _to_float(prodotto_sel.get("quantita")) or 0.0

            opzioni_quantita = [
                "Tutta la confezione",
                "1/2 confezione",
                "1/3 confezione",
                "1/4 confezione",
                "Personalizzato (g)",
            ]
            frazioni = {
                "Tutta la confezione": 1.0,
                "1/2 confezione": 0.5,
                "1/3 confezione": 1/3,
                "1/4 confezione": 0.25,
            }

            with col_q:
                scelta_q = st.selectbox("Quantità utilizzata", opzioni_quantita, key="diario_q_opt")
            with col_d:
                data_sel = st.date_input("Data", value=date.today(), key="diario_data")

            if scelta_q == "Personalizzato (g)":
                grammi = st.number_input(
                    "Grammi", min_value=0.0, step=10.0, value=100.0, key="diario_g_custom"
                )
            else:
                grammi = qta_conf * frazioni[scelta_q]
                if qta_conf > 0:
                    st.caption(f"≈ {grammi:g} g")
                else:
                    st.warning("Quantità della confezione non impostata per questo prodotto: imposta una quantità personalizzata.")

            if st.button("💾 Registra consumo", type="primary"):
                if grammi <= 0:
                    st.error("La quantità deve essere maggiore di 0.")
                else:
                    p = mappa_prodotti[nome_sel]
                    fattore = grammi / 100.0
                    payload = {
                        "user_id": user_id,
                        "data": data_sel.isoformat(),
                        "pasto": pasto_sel,
                        "nome": nome_sel,
                        "quantita": grammi,
                        "kcal": (p.get("kcal") or 0) * fattore,
                        "carbi": (p.get("carbi") or 0) * fattore,
                        "pro": (p.get("pro") or 0) * fattore,
                        "grassi": (p.get("grassi") or 0) * fattore,
                    }
                    try:
                        supabase.table("consumi").insert(payload).execute()
                        _applica_delta_dispensa(nome_sel, grammi, user_id)
                        st.success(f"Registrato: {grammi:g} g di {nome_sel}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore salvataggio: {e}")

        # --- Preset pasti ricorrenti ---
        with st.expander("🧩 Preset pasti ricorrenti"):
            try:
                res_preset = (
                    supabase.table("preset_pasti").select("*")
                    .eq("user_id", user_id).order("nome").execute()
                )
                presets = res_preset.data or []
            except Exception as e:
                presets = []
                st.error(f"Errore caricamento preset: {e}")

            def _parse_ingredienti(raw):
                if isinstance(raw, list):
                    return raw
                if isinstance(raw, str):
                    try:
                        return json.loads(raw)
                    except Exception:
                        return []
                return []

            # --- Usa preset ---
            st.markdown("**Usa un preset**")
            if not presets:
                st.caption("Nessun preset salvato. Creane uno dal blocco qui sotto.")
            else:
                mappa_preset = {p["nome"]: p for p in presets}
                preset_sel_nome = st.selectbox(
                    "Preset", list(mappa_preset.keys()), key="preset_usa_sel"
                )
                preset_sel = mappa_preset[preset_sel_nome]
                ingr = _parse_ingredienti(preset_sel.get("ingredienti"))
                if ingr:
                    riepilogo = ", ".join(
                        f"{i.get('nome')} ({_to_float(i.get('quantita')) or 0:g} g)"
                        for i in ingr
                    )
                    st.caption(f"Contenuto: {riepilogo}")

                abilitati_presets = st.session_state.pasti_abilitati or list(ORDINE_PASTI)
                pasto_default = preset_sel.get("pasto")
                idx_def = (
                    abilitati_presets.index(pasto_default)
                    if pasto_default in abilitati_presets else 0
                )
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    preset_pasto = st.selectbox(
                        "Pasto",
                        abilitati_presets,
                        format_func=lambda p: ETICHETTE_PASTI.get(p, p),
                        index=idx_def,
                        key="preset_usa_pasto",
                    )
                with col_p2:
                    preset_data = st.date_input(
                        "Data", value=date.today(), key="preset_usa_data"
                    )

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("📥 Registra preset", type="primary", key="preset_usa_btn"):
                        if not ingr:
                            st.error("Preset vuoto.")
                        else:
                            errori = []
                            for i in ingr:
                                nome_i = i.get("nome")
                                qta_i = _to_float(i.get("quantita")) or 0
                                if not nome_i or qta_i <= 0:
                                    continue
                                payload = {
                                    "user_id": user_id,
                                    "data": preset_data.isoformat(),
                                    "pasto": preset_pasto,
                                    "nome": nome_i,
                                    "quantita": qta_i,
                                    "kcal": _to_float(i.get("kcal")) or 0,
                                    "carbi": _to_float(i.get("carbi")) or 0,
                                    "pro": _to_float(i.get("pro")) or 0,
                                    "grassi": _to_float(i.get("grassi")) or 0,
                                }
                                try:
                                    supabase.table("consumi").insert(payload).execute()
                                    _applica_delta_dispensa(nome_i, qta_i, user_id)
                                except Exception as e:
                                    errori.append(f"{nome_i}: {e}")
                            if errori:
                                st.error("Alcuni ingredienti non registrati:\n- " + "\n- ".join(errori))
                            else:
                                st.success(
                                    f"Registrati {len(ingr)} ingredienti da '{preset_sel_nome}'."
                                )
                                st.rerun()
                with col_b2:
                    if st.button("🗑️ Elimina preset", key="preset_del_btn"):
                        try:
                            (
                                supabase.table("preset_pasti").delete()
                                .eq("id", preset_sel["id"]).eq("user_id", user_id).execute()
                            )
                            st.success(f"Preset '{preset_sel_nome}' eliminato.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Errore eliminazione: {e}")

            st.divider()

            # --- Salva preset dai pasti di oggi ---
            st.markdown("**Salva un preset dai pasti di oggi**")
            oggi_iso_pre = date.today().isoformat()
            try:
                res_oggi_full = (
                    supabase.table("consumi")
                    .select("nome, quantita, kcal, carbi, pro, grassi, pasto")
                    .eq("user_id", user_id).eq("data", oggi_iso_pre).execute()
                )
                consumi_oggi = res_oggi_full.data or []
            except Exception:
                consumi_oggi = []

            if not consumi_oggi:
                st.caption("Nessun pasto registrato oggi da cui creare un preset.")
            else:
                pasti_con_consumi = sorted(
                    {c.get("pasto") for c in consumi_oggi if c.get("pasto")},
                    key=lambda p: ORDINE_PASTI.index(p) if p in ORDINE_PASTI else 99,
                )
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    pasto_src = st.selectbox(
                        "Pasto di oggi",
                        pasti_con_consumi,
                        format_func=lambda p: ETICHETTE_PASTI.get(p, p),
                        key="preset_save_pasto",
                    )
                with col_s2:
                    nome_preset = st.text_input(
                        "Nome del preset",
                        placeholder="es. Colazione standard",
                        key="preset_save_nome",
                    )

                ingr_src = [c for c in consumi_oggi if c.get("pasto") == pasto_src]
                if ingr_src:
                    riepilogo_src = ", ".join(
                        f"{c.get('nome')} ({_to_float(c.get('quantita')) or 0:g} g)"
                        for c in ingr_src
                    )
                    st.caption(f"Ingredienti: {riepilogo_src}")

                if st.button("💾 Salva preset", key="preset_save_btn"):
                    nome_clean = (nome_preset or "").strip()
                    if not nome_clean:
                        st.error("Inserisci un nome per il preset.")
                    elif not ingr_src:
                        st.error("Nessun ingrediente da salvare.")
                    else:
                        ingredienti_payload = [
                            {
                                "nome": c.get("nome"),
                                "quantita": _to_float(c.get("quantita")) or 0,
                                "kcal": _to_float(c.get("kcal")) or 0,
                                "carbi": _to_float(c.get("carbi")) or 0,
                                "pro": _to_float(c.get("pro")) or 0,
                                "grassi": _to_float(c.get("grassi")) or 0,
                            }
                            for c in ingr_src
                        ]
                        try:
                            supabase.table("preset_pasti").insert({
                                "user_id": user_id,
                                "nome": nome_clean,
                                "pasto": pasto_src,
                                "ingredienti": ingredienti_payload,
                            }).execute()
                            st.success(f"Preset '{nome_clean}' salvato.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Errore salvataggio preset: {e}")

        st.divider()

        # --- Visualizzazione ---
        vista = st.radio("Visualizzazione", ["Oggi", "Ultimi 7 giorni"], horizontal=True)

        oggi = date.today()
        if vista == "Oggi":
            res_c = (
                supabase.table("consumi").select("*")
                .eq("user_id", user_id).eq("data", oggi.isoformat())
                .order("created_at", desc=True).execute()
            )
            dati_c = res_c.data or []
            if not dati_c:
                st.info("Nessun consumo registrato oggi.")
            else:
                df_c = pd.DataFrame(dati_c)
                tot_kcal = df_c["kcal"].sum()
                tot_carbi = df_c["carbi"].sum()
                tot_pro = df_c["pro"].sum()
                tot_grassi = df_c["grassi"].sum()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("kcal", f"{tot_kcal:.0f}")
                m2.metric("Carbo (g)", f"{tot_carbi:.1f}")
                m3.metric("Proteine (g)", f"{tot_pro:.1f}")
                m4.metric("Grassi (g)", f"{tot_grassi:.1f}")
                if prof_attuale.get("target_kcal"):
                    t_kcal = float(prof_attuale.get("target_kcal") or 0)
                    t_carbi = float(prof_attuale.get("target_carbi") or 0)
                    t_pro = float(prof_attuale.get("target_pro") or 0)
                    t_grassi = float(prof_attuale.get("target_grassi") or 0)

                    def _frac(attuale, target):
                        if target <= 0:
                            return 0.0
                        return min(1.0, max(0.0, attuale / target))

                    st.caption(f"🎯 kcal: {tot_kcal:.0f} / {t_kcal:.0f}")
                    pct_kcal = int(round(_frac(tot_kcal, t_kcal) * 100))
                    st.markdown(
                        f"""
                        <div style="background:#eceff1;border-radius:8px;
                                    height:22px;width:100%;overflow:hidden;
                                    margin-bottom:8px;">
                          <div style="background:#ff6b35;height:100%;
                                      width:{pct_kcal}%;border-radius:8px;
                                      transition:width .3s ease;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"🎯 Carbo: {tot_carbi:.1f} / {t_carbi:.0f} g")
                    st.progress(_frac(tot_carbi, t_carbi))
                    st.caption(f"🎯 Proteine: {tot_pro:.1f} / {t_pro:.0f} g")
                    st.progress(_frac(tot_pro, t_pro))
                    st.caption(f"🎯 Grassi: {tot_grassi:.1f} / {t_grassi:.0f} g")
                    st.progress(_frac(tot_grassi, t_grassi))

                st.subheader("Pasti di oggi")
                _render_pasti(df_c)
        else:
            # Ultimi 7 giorni ESCLUSO oggi → da oggi-7 a ieri
            inizio = oggi - timedelta(days=7)
            fine = oggi - timedelta(days=1)
            res_c = (
                supabase.table("consumi").select("*")
                .eq("user_id", user_id)
                .gte("data", inizio.isoformat())
                .lte("data", fine.isoformat())
                .execute()
            )
            dati_c = res_c.data or []
            if not dati_c:
                st.info("Nessun consumo registrato negli ultimi 7 giorni.")
            else:
                df_c = pd.DataFrame(dati_c)
                # Somma per giorno, media solo sui giorni con almeno un pasto.
                agg = df_c.groupby("data")[["kcal", "carbi", "pro", "grassi"]].sum().reset_index()
                n_giorni = len(agg)
                media_kcal = agg["kcal"].sum() / n_giorni
                media_carbi = agg["carbi"].sum() / n_giorni
                media_pro = agg["pro"].sum() / n_giorni
                media_grassi = agg["grassi"].sum() / n_giorni

                st.caption(
                    f"Media giornaliera dal {_data_it(inizio)} al {_data_it(fine)} (oggi escluso)"
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("kcal/giorno", f"{media_kcal:.0f}")
                m2.metric("Carbo/giorno (g)", f"{media_carbi:.1f}")
                m3.metric("Proteine/giorno (g)", f"{media_pro:.1f}")
                m4.metric("Grassi/giorno (g)", f"{media_grassi:.1f}")
                if prof_attuale.get("target_kcal"):
                    st.caption(
                        f"🎯 Target: {prof_attuale['target_kcal']:.0f} kcal · "
                        f"{prof_attuale.get('target_carbi') or 0:.0f} g carbi · "
                        f"{prof_attuale.get('target_pro') or 0:.0f} g pro · "
                        f"{prof_attuale.get('target_grassi') or 0:.0f} g grassi"
                    )

                st.subheader("Dettaglio per giorno")
                for data_str in sorted(df_c["data"].unique(), reverse=True):
                    df_giorno = df_c[df_c["data"] == data_str]
                    tot_kcal_g = df_giorno["kcal"].sum()
                    tot_carbi_g = df_giorno["carbi"].sum()
                    tot_pro_g = df_giorno["pro"].sum()
                    tot_grassi_g = df_giorno["grassi"].sum()
                    with st.container(border=True):
                        st.markdown(f"#### {_data_it(data_str)}")
                        t_kcal_d = float(prof_attuale.get("target_kcal") or 0)
                        t_carbi_d = float(prof_attuale.get("target_carbi") or 0)
                        t_pro_d = float(prof_attuale.get("target_pro") or 0)
                        t_grassi_d = float(prof_attuale.get("target_grassi") or 0)

                        def _delta(attuale, target, digits=0):
                            if target <= 0:
                                return None
                            return f"{attuale - target:+.{digits}f} vs target"

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(
                            "kcal", f"{tot_kcal_g:.0f}",
                            delta=_delta(tot_kcal_g, t_kcal_d, 0),
                        )
                        c2.metric(
                            "Carbo (g)", f"{tot_carbi_g:.1f}",
                            delta=_delta(tot_carbi_g, t_carbi_d, 1),
                        )
                        c3.metric(
                            "Proteine (g)", f"{tot_pro_g:.1f}",
                            delta=_delta(tot_pro_g, t_pro_d, 1),
                        )
                        c4.metric(
                            "Grassi (g)", f"{tot_grassi_g:.1f}",
                            delta=_delta(tot_grassi_g, t_grassi_d, 1),
                        )
                        _render_pasti(df_giorno)

        st.divider()

        with st.expander("✏️ Modifica o elimina pasti registrati"):
            giorni_indietro = st.number_input(
                "Mostra pasti degli ultimi (giorni)",
                min_value=1, max_value=365, value=30, step=1,
                key="modifica_giorni",
            )
            inizio_mod = oggi - timedelta(days=int(giorni_indietro) - 1)
            res_mod = (
                supabase.table("consumi").select("*")
                .eq("user_id", user_id)
                .gte("data", inizio_mod.isoformat())
                .order("data", desc=True)
                .execute()
            )
            dati_mod = res_mod.data or []
            if not dati_mod:
                st.info(f"Nessun pasto negli ultimi {int(giorni_indietro)} giorni.")
            else:
                df_mod_orig = pd.DataFrame(dati_mod)
                if "pasto" not in df_mod_orig.columns:
                    df_mod_orig["pasto"] = None

                st.caption(
                    "Apri il pasto e modifica i grammi (i nutrizionali si riscalano). "
                    "Seleziona con ✓ gli ingredienti da eliminare."
                )

                for data_str in sorted(df_mod_orig["data"].unique(), reverse=True):
                    df_day = df_mod_orig[df_mod_orig["data"] == data_str]
                    with st.container(border=True):
                        st.markdown(f"#### {_data_it(data_str)}")

                        gruppi = []
                        for p in ORDINE_PASTI:
                            df_p = df_day[df_day["pasto"] == p]
                            if not df_p.empty:
                                gruppi.append((ETICHETTE_PASTI[p], df_p, f"{data_str}_{p}", p))
                        altri = df_day[~df_day["pasto"].isin(ORDINE_PASTI)]
                        if not altri.empty:
                            gruppi.append(("Altro", altri, f"{data_str}_altro", None))

                        for label, df_pasto, ks, pasto_val in gruppi:
                            tot_kcal = df_pasto["kcal"].sum()
                            with st.expander(f"{label} — {tot_kcal:.0f} kcal"):
                                df_edit = df_pasto[["id", "nome", "quantita"]].copy()
                                df_edit.insert(0, "seleziona", False)

                                edited = st.data_editor(
                                    df_edit,
                                    hide_index=True,
                                    num_rows="fixed",
                                    column_config={
                                        "id": None,
                                        "seleziona": st.column_config.CheckboxColumn(
                                            "✓", default=False, width="small",
                                        ),
                                        "nome": st.column_config.TextColumn("Nome", disabled=True),
                                        "quantita": st.column_config.NumberColumn(
                                            "g", min_value=0.0, step=1.0,
                                        ),
                                    },
                                    disabled=["nome"],
                                    use_container_width=True,
                                    key=f"mod_editor_{ks}",
                                )

                                modifiche = {}
                                for _, riga_new in edited.iterrows():
                                    row_id = int(riga_new["id"])
                                    riga_old = df_pasto[df_pasto["id"] == row_id].iloc[0]
                                    q_new = float(riga_new["quantita"]) if pd.notna(riga_new["quantita"]) else 0.0
                                    q_old = float(riga_old["quantita"]) if pd.notna(riga_old["quantita"]) else 0.0
                                    if not _stessi_valori(q_new, q_old):
                                        diff = {"quantita": q_new}
                                        if q_old > 0:
                                            fattore = q_new / q_old
                                            for c in ("kcal", "carbi", "pro", "grassi"):
                                                val_old = riga_old.get(c)
                                                if pd.notna(val_old):
                                                    diff[c] = float(val_old) * fattore
                                        modifiche[row_id] = diff

                                ha_mod = bool(modifiche)
                                ids_sel = [int(i) for i in edited[edited["seleziona"] == True]["id"].tolist()]

                                cs, cd, _ = st.columns([2, 2, 4])
                                with cs:
                                    salva = st.button(
                                        "💾 Salva",
                                        type="primary" if ha_mod else "secondary",
                                        disabled=not ha_mod,
                                        use_container_width=True,
                                        key=f"mod_salva_{ks}",
                                    )
                                with cd:
                                    if ids_sel:
                                        elimina = st.button(
                                            f"🗑️ Elimina ({len(ids_sel)})",
                                            use_container_width=True,
                                            key=f"mod_elim_{ks}",
                                        )
                                    else:
                                        elimina = False

                                if salva and ha_mod:
                                    errori = []
                                    for row_id, diff in modifiche.items():
                                        try:
                                            riga_vecchia = df_pasto[df_pasto["id"] == row_id].iloc[0]
                                            (
                                                supabase.table("consumi").update(diff)
                                                .eq("id", row_id).eq("user_id", user_id).execute()
                                            )
                                            if "quantita" in diff:
                                                q_vecchia = (
                                                    float(riga_vecchia["quantita"])
                                                    if pd.notna(riga_vecchia["quantita"]) else 0.0
                                                )
                                                delta = float(diff["quantita"]) - q_vecchia
                                                _applica_delta_dispensa(
                                                    riga_vecchia["nome"], delta, user_id,
                                                )
                                        except Exception as e:
                                            errori.append(f"Riga {row_id}: {e}")
                                    if errori:
                                        st.error("Aggiornamenti falliti:\n" + "\n".join(errori))
                                    else:
                                        st.success(
                                            f"Aggiornat{'o' if len(modifiche) == 1 else 'i'} {len(modifiche)}."
                                        )
                                        st.rerun()

                                if elimina and ids_sel:
                                    errori = []
                                    for row_id in ids_sel:
                                        try:
                                            riga_vecchia = df_pasto[df_pasto["id"] == row_id].iloc[0]
                                            (
                                                supabase.table("consumi").delete()
                                                .eq("id", row_id).eq("user_id", user_id).execute()
                                            )
                                            q_vecchia = (
                                                float(riga_vecchia["quantita"])
                                                if pd.notna(riga_vecchia["quantita"]) else 0.0
                                            )
                                            _applica_delta_dispensa(
                                                riga_vecchia["nome"], -q_vecchia, user_id,
                                            )
                                        except Exception as e:
                                            errori.append(f"Riga {row_id}: {e}")
                                    if errori:
                                        st.error("Eliminazioni fallite:\n" + "\n".join(errori))
                                    else:
                                        st.success(
                                            f"Eliminat{'o' if len(ids_sel) == 1 else 'i'} {len(ids_sel)}."
                                        )
                                        st.rerun()

                                if pasto_val is not None and prodotti:
                                    st.divider()
                                    st.caption("Aggiungi un ingrediente a questo pasto")
                                    mappa_add = {pr["nome"]: pr for pr in prodotti}
                                    ca_p, ca_g = st.columns([3, 2])
                                    with ca_p:
                                        nome_add = st.selectbox(
                                            "Prodotto",
                                            list(mappa_add.keys()),
                                            key=f"mod_add_prod_{ks}",
                                        )
                                    with ca_g:
                                        grammi_add = st.number_input(
                                            "Grammi",
                                            min_value=0.0, step=10.0, value=100.0,
                                            key=f"mod_add_g_{ks}",
                                        )
                                    if st.button(
                                        "➕ Aggiungi ingrediente",
                                        key=f"mod_add_btn_{ks}",
                                        disabled=(grammi_add <= 0),
                                    ):
                                        prod_add = mappa_add[nome_add]
                                        fattore_add = grammi_add / 100.0
                                        payload_add = {
                                            "user_id": user_id,
                                            "data": data_str,
                                            "pasto": pasto_val,
                                            "nome": nome_add,
                                            "quantita": grammi_add,
                                            "kcal": (prod_add.get("kcal") or 0) * fattore_add,
                                            "carbi": (prod_add.get("carbi") or 0) * fattore_add,
                                            "pro": (prod_add.get("pro") or 0) * fattore_add,
                                            "grassi": (prod_add.get("grassi") or 0) * fattore_add,
                                        }
                                        try:
                                            supabase.table("consumi").insert(payload_add).execute()
                                            _applica_delta_dispensa(nome_add, grammi_add, user_id)
                                            st.success(f"Aggiunto {grammi_add:g} g di {nome_add}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Errore: {e}")

    with tab_prog:
        st.header("Progressi")
        st.caption(
            "Registra peso e circonferenze per monitorare i progressi nel tempo."
        )

        with st.form("misura_form", clear_on_submit=True):
            col_d, col_p = st.columns(2)
            with col_d:
                m_data = st.date_input(
                    "Data", value=date.today(), key="misura_data",
                    format="DD/MM/YYYY",
                )
            with col_p:
                m_peso = st.number_input(
                    "Peso (kg)", min_value=0.0, max_value=400.0,
                    step=0.1, value=0.0, key="misura_peso",
                )
            col_v, col_c = st.columns(2)
            with col_v:
                m_vita = st.number_input(
                    "Vita (cm)", min_value=0.0, max_value=300.0,
                    step=0.1, value=0.0, key="misura_vita",
                )
            with col_c:
                m_collo = st.number_input(
                    "Collo (cm)", min_value=0.0, max_value=300.0,
                    step=0.1, value=0.0, key="misura_collo",
                )
            m_note = st.text_input("Note (opzionale)", key="misura_note")
            submitted = st.form_submit_button("💾 Salva misurazione", type="primary")
            if submitted:
                if m_peso <= 0 and m_vita <= 0 and m_collo <= 0:
                    st.error("Inserisci almeno un valore (peso, vita o collo).")
                else:
                    payload_m = {
                        "user_id": user_id,
                        "data": m_data.isoformat(),
                        "peso_kg": m_peso if m_peso > 0 else None,
                        "vita_cm": m_vita if m_vita > 0 else None,
                        "collo_cm": m_collo if m_collo > 0 else None,
                        "note": m_note or None,
                    }
                    try:
                        supabase.table("misurazioni").insert(payload_m).execute()
                        st.success("Misurazione salvata.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore salvataggio: {e}")

        st.divider()
        st.subheader("Storico")
        try:
            res_m = (
                supabase.table("misurazioni").select("*")
                .eq("user_id", user_id).order("data", desc=False).execute()
            )
            dati_m = res_m.data or []
        except Exception as e:
            dati_m = []
            st.error(f"Errore caricamento storico: {e}")

        if not dati_m:
            st.info("Nessuna misurazione registrata.")
        else:
            df_m = pd.DataFrame(dati_m)
            df_m["data"] = pd.to_datetime(df_m["data"])
            df_m = df_m.sort_values("data")

            col_ch = [c for c in ("peso_kg", "vita_cm", "collo_cm") if c in df_m.columns]
            if col_ch:
                st.line_chart(df_m.set_index("data")[col_ch])

            df_show = df_m.sort_values("data", ascending=False).copy()
            df_show["data"] = df_show["data"].dt.strftime("%d/%m/%Y")
            cols_show = ["data"] + [
                c for c in ("peso_kg", "vita_cm", "collo_cm", "note") if c in df_show.columns
            ]
            st.dataframe(
                df_show[cols_show].rename(columns={
                    "data": "Data", "peso_kg": "Peso (kg)",
                    "vita_cm": "Vita (cm)", "collo_cm": "Collo (cm)", "note": "Note",
                }),
                hide_index=True, use_container_width=True,
            )

            with st.expander("🗑️ Elimina una misurazione"):
                opzioni_m = {
                    f"{r['data'].strftime('%d/%m/%Y') if hasattr(r['data'], 'strftime') else r['data']} — "
                    f"{(r.get('peso_kg') or '—')} kg": r["id"]
                    for _, r in df_m.iterrows()
                }
                scelta_m = st.selectbox(
                    "Misurazione", list(opzioni_m.keys()), key="misura_del_sel"
                )
                if st.button("Elimina", key="misura_del_btn"):
                    try:
                        (
                            supabase.table("misurazioni").delete()
                            .eq("id", opzioni_m[scelta_m]).eq("user_id", user_id).execute()
                        )
                        st.success("Misurazione eliminata.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Errore: {e}")

    with tab_port:
        st.header("Portafoglio")
        try:
            res_port = (
                supabase.table("storico_prezzi").select("*")
                .eq("user_id", user_id).execute()
            )
            dati_port = res_port.data or []
        except Exception as e:
            dati_port = []
            st.error(f"Errore caricamento storico: {e}")

        if not dati_port:
            st.info("Ancora nessuna scansione. Carica uno scontrino dalla tab '🧾 Scontrino'.")
        else:
            df_port = pd.DataFrame(dati_port)
            df_port["data"] = pd.to_datetime(df_port["data"])
            df_port["prezzo"] = pd.to_numeric(df_port["prezzo"], errors="coerce").fillna(0.0)
            if "prezzo_kg" in df_port.columns:
                df_port["prezzo_kg"] = pd.to_numeric(df_port["prezzo_kg"], errors="coerce")
            df_port["mese"] = df_port["data"].dt.to_period("M")

            oggi_ts = pd.Timestamp.today()
            mese_cur = pd.Period(oggi_ts, "M")
            mese_prec = mese_cur - 1

            spesa_cur = float(df_port[df_port["mese"] == mese_cur]["prezzo"].sum())
            spesa_prec = float(df_port[df_port["mese"] == mese_prec]["prezzo"].sum())
            if spesa_prec > 0:
                delta_pct = (spesa_cur - spesa_prec) / spesa_prec * 100
                delta_str = f"{delta_pct:+.1f}% vs mese prec."
            else:
                delta_str = None

            c1, c2 = st.columns(2)
            c1.metric(
                f"Spesa {_MESI_IT[mese_cur.month - 1].capitalize()} {mese_cur.year}",
                f"€ {spesa_cur:.2f}", delta=delta_str, delta_color="inverse",
            )
            c2.metric(
                f"Spesa {_MESI_IT[mese_prec.month - 1].capitalize()} {mese_prec.year}",
                f"€ {spesa_prec:.2f}",
            )

            st.divider()
            st.subheader("Distribuzione per categoria (mese corrente)")
            df_cat = (
                df_port[df_port["mese"] == mese_cur]
                .groupby("categoria")["prezzo"].sum().reset_index()
                .sort_values("prezzo", ascending=False)
            )
            if df_cat.empty:
                st.info("Nessuna spesa nel mese corrente.")
            else:
                try:
                    import plotly.express as _px
                    fig = _px.pie(df_cat, values="prezzo", names="categoria", hole=0.4)
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.bar_chart(df_cat.set_index("categoria")["prezzo"])

            st.divider()
            st.subheader("Andamento prezzo al kg")
            df_kg = df_port[df_port["prezzo_kg"].notna()] if "prezzo_kg" in df_port.columns else df_port.iloc[0:0]
            prodotti_unici = sorted(df_kg["nome"].unique().tolist())
            if not prodotti_unici:
                st.info("Nessun prodotto con prezzo al kg calcolato.")
            else:
                col_ps, col_mes = st.columns([3, 2])
                with col_ps:
                    prod_sel = st.selectbox(
                        "Prodotto", prodotti_unici, key="port_prod_sel"
                    )
                with col_mes:
                    mesi_sel = st.slider(
                        "Ultimi N mesi", min_value=1, max_value=24, value=6,
                        key="port_mesi",
                    )
                inizio_p = oggi_ts - pd.DateOffset(months=mesi_sel)
                df_prod = (
                    df_kg[(df_kg["nome"] == prod_sel) & (df_kg["data"] >= inizio_p)]
                    .sort_values("data")
                )
                if df_prod.empty:
                    st.info("Nessun dato nel periodo selezionato.")
                else:
                    st.line_chart(df_prod.set_index("data")["prezzo_kg"])
                    media = float(df_prod["prezzo_kg"].mean())
                    mn = float(df_prod["prezzo_kg"].min())
                    mx = float(df_prod["prezzo_kg"].max())
                    st.caption(
                        f"{len(df_prod)} rilevazioni · media € {media:.2f}/kg · "
                        f"min € {mn:.2f} · max € {mx:.2f}"
                    )

