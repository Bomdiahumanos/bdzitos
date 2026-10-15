from __future__ import annotations

import io
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import chess
import chess.pgn
import chess.svg
import chess.engine
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

APP_VERSION = "0.1.0"
CHESSCOM_API = "https://api.chess.com/pub/player"
USER_AGENT = f"AberturaLab/{APP_VERSION} (personal chess-study tool)"
OPENING_MAX_MOVE = 12

st.set_page_config(
    page_title="AberturaLab — treinador de repertório",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px;}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.35); border-radius: 16px; margin-bottom: 1rem;}
      .hero h1 {margin: 0 0 .25rem 0; font-size: 2.1rem;}
      .muted {opacity: .78;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_username(username: str) -> str:
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", username):
        raise ValueError("O nome de usuário contém caracteres inesperados.")
    return username


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chesscom_entries(username: str, months: int, max_games: int, time_class: str) -> tuple[list[dict[str, Any]], str]:
    """Fetch public Chess.com PGNs serially; no login or password is used."""
    username = clean_username(username)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    archive_url = f"{CHESSCOM_API}/{username.lower()}/games/archives"
    response = requests.get(archive_url, headers=headers, timeout=25)
    if response.status_code == 404:
        raise ValueError(
            f"Não encontrei o usuário público **{username}** na API do Chess.com. "
            "Confira a grafia ou importe um arquivo PGN."
        )
    response.raise_for_status()
    archives = response.json().get("archives", [])
    if not archives:
        return [], "A conta existe, mas a API não retornou arquivos públicos de partidas."

    selected_archives = archives if months == 0 else archives[-months:]
    entries: list[dict[str, Any]] = []
    checked = 0
    # Latest months first. Requests are deliberately sequential to respect API guidance.
    for month_url in reversed(selected_archives):
        month_response = requests.get(month_url, headers=headers, timeout=30)
        if month_response.status_code == 429:
            raise RuntimeError("O Chess.com limitou temporariamente as consultas (HTTP 429). Aguarde um pouco e tente de novo.")
        month_response.raise_for_status()
        month_games = month_response.json().get("games", [])
        for item in month_games:
            rules = str(item.get("rules", "chess")).lower()
            if rules != "chess":
                continue
            tc = str(item.get("time_class", "unknown")).lower()
            if time_class != "all" and tc != time_class:
                continue
            pgn = item.get("pgn")
            if not pgn:
                continue
            entries.append({
                "pgn": pgn,
                "time_class": tc,
                "url": item.get("url", ""),
                "end_time": item.get("end_time", 0),
                "is_rated": bool(item.get("rated", False)),
            })
            checked += 1
        # Once enough games have been found, do not fetch older months unnecessarily.
        if checked >= max_games:
            break

    entries.sort(key=lambda x: int(x.get("end_time") or 0), reverse=True)
    entries = entries[:max_games]
    if not entries:
        return [], "Nenhuma partida padrão encontrada com esses filtros."
    return entries, f"Partidas carregadas da API pública do Chess.com: {len(entries)}."


def pgn_entries_from_upload(uploaded_file) -> list[dict[str, Any]]:
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")
    entries: list[dict[str, Any]] = []
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        buf = io.StringIO()
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        buf.write(game.accept(exporter))
        entries.append({"pgn": buf.getvalue(), "time_class": "unknown", "url": "", "end_time": 0, "is_rated": False})
    return entries


def normalise_result(result: str, user_is_white: bool) -> tuple[str, float | None]:
    if result == "1-0":
        return ("Vitória", 1.0) if user_is_white else ("Derrota", 0.0)
    if result == "0-1":
        return ("Derrota", 0.0) if user_is_white else ("Vitória", 1.0)
    if result == "1/2-1/2":
        return "Empate", 0.5
    return "Desconhecido", None


def to_pt_san(san: str) -> str:
    """Convert the leading English piece letter in SAN to Portuguese notation."""
    if not san:
        return san
    mapping = {"K": "R", "Q": "D", "R": "T", "B": "B", "N": "C"}
    # Do not alter castling, promotions after '=', or a pawn move.
    if san[0] in mapping:
        return mapping[san[0]] + san[1:]
    return san


def side_label(is_white: bool) -> str:
    return "Brancas" if is_white else "Negras"


def move_label(fullmove: int, is_white: bool, san: str) -> str:
    prefix = f"{fullmove}." if is_white else f"{fullmove}..."
    return f"{prefix}{to_pt_san(san)}"


def fen_position_key(fen: str) -> str:
    # Exclude move counters; retain board, side-to-move, castling rights and en-passant.
    return " ".join(fen.split()[:4])


def parse_entries(entries: list[dict[str, Any]], username: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    games: list[dict[str, Any]] = []
    reply_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    username_cf = username.strip().casefold()

    for entry_index, entry in enumerate(entries):
        try:
            game = chess.pgn.read_game(io.StringIO(entry.get("pgn", "")))
        except Exception:
            game = None
        if game is None:
            warnings.append(f"Uma entrada PGN ({entry_index + 1}) não pôde ser lida.")
            continue
        headers = game.headers
        white_name = headers.get("White", "?")
        black_name = headers.get("Black", "?")
        white_is_user = white_name.casefold() == username_cf
        black_is_user = black_name.casefold() == username_cf

        # If an exported PGN uses generic names, the account perspective is ambiguous.
        if white_is_user == black_is_user:
            warnings.append(
                f"Partida ignorada: não consegui identificar {username} com certeza "
                f"(White='{white_name}', Black='{black_name}'). Confira os nomes no PGN."
            )
            continue
        user_is_white = white_is_user
        outcome, score = normalise_result(headers.get("Result", "*"), user_is_white)
        opponent = black_name if user_is_white else white_name
        try:
            date_label = headers.get("Date", "????.??.??")
            if date_label != "????.??.??":
                date_label = date_label.replace(".", "-")
        except Exception:
            date_label = "—"
        opening = headers.get("Opening", "") or "Abertura não identificada"
        eco = headers.get("ECO", "")
        if opening in ("?", "Unknown", "Abertura não identificada") and eco:
            opening = f"Abertura não identificada ({eco})"
        if opening == "?":
            opening = "Abertura não identificada"

        board = game.board()
        move_rows: list[dict[str, Any]] = []
        for ply, move in enumerate(game.mainline_moves()):
            before_fen = board.fen()
            san = board.san(move)
            mover_is_white = board.turn == chess.WHITE
            fullmove = board.fullmove_number
            row = {
                "ply": ply,
                "fullmove": fullmove,
                "mover_is_white": mover_is_white,
                "san": san,
                "uci": move.uci(),
                "before_fen": before_fen,
                "position_key": fen_position_key(before_fen),
            }
            board.push(move)
            row["after_fen"] = board.fen()
            move_rows.append(row)

        game_id = headers.get("Link") or entry.get("url") or f"{date_label}|{white_name}|{black_name}|{entry_index}"
        parsed = {
            "game_id": game_id,
            "pgn": entry.get("pgn", ""),
            "time_class": entry.get("time_class") or "unknown",
            "date": date_label,
            "white": white_name,
            "black": black_name,
            "opponent": opponent,
            "user_is_white": user_is_white,
            "color": side_label(user_is_white),
            "result": outcome,
            "score": score,
            "eco": eco,
            "opening": opening,
            "white_elo": headers.get("WhiteElo", ""),
            "black_elo": headers.get("BlackElo", ""),
            "url": entry.get("url") or headers.get("Link", ""),
            "moves": move_rows,
        }
        games.append(parsed)

        # Build patterns for a position where the user chose a move, then the opponent replied.
        for idx, row in enumerate(move_rows):
            if row["mover_is_white"] != user_is_white or row["fullmove"] > OPENING_MAX_MOVE:
                continue
            if idx + 1 >= len(move_rows):
                continue
            reply = move_rows[idx + 1]
            # The next half-move must belong to the opponent.
            if reply["mover_is_white"] == user_is_white:
                continue
            reply_rows.append({
                "position_key": row["position_key"],
                "fullmove": row["fullmove"],
                "user_is_white": user_is_white,
                "played_san": row["san"],
                "reply_san": reply["san"],
                "score": score,
                "result": outcome,
                "opening": opening,
                "opponent": opponent,
                "game_id": game_id,
            })

    games.sort(key=lambda g: (g["date"] != "????-??-??", g["date"]), reverse=True)
    return games, reply_rows, warnings


def wdl_counts(scores: list[float | None]) -> tuple[int, int, int, int]:
    usable = [s for s in scores if s is not None]
    wins = sum(1 for s in usable if s == 1.0)
    draws = sum(1 for s in usable if s == 0.5)
    losses = sum(1 for s in usable if s == 0.0)
    return len(usable), wins, draws, losses


def make_opening_stats(games: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        groups[(game["opening"], game["color"])].append(game)
    for (opening, color), items in groups.items():
        n, wins, draws, losses = wdl_counts([g["score"] for g in items])
        rows.append({
            "Abertura": opening,
            "Cor": color,
            "Partidas": len(items),
            "Vitórias": wins,
            "Empates": draws,
            "Derrotas": losses,
            "Taxa de vitória (%)": round(100 * wins / n, 1) if n else None,
            "Pontuação (%)": round(100 * sum(s for s in [g["score"] for g in items] if s is not None) / n, 1) if n else None,
            "ECO": next((g["eco"] for g in items if g["eco"]), ""),
        })
    return pd.DataFrame(rows)


def make_color_stats(games: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for color in ("Brancas", "Negras"):
        items = [g for g in games if g["color"] == color]
        n, wins, draws, losses = wdl_counts([g["score"] for g in items])
        rows.append({
            "Cor": color, "Partidas": len(items), "Vitórias": wins,
            "Empates": draws, "Derrotas": losses,
            "Taxa de vitória (%)": round(100 * wins / n, 1) if n else 0,
            "Pontuação (%)": round(100 * sum(s for s in [g["score"] for g in items] if s is not None) / n, 1) if n else 0,
        })
    return pd.DataFrame(rows)


def make_reply_stats(reply_rows: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[tuple[str, int, bool, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reply_rows:
        key = (row["position_key"], row["fullmove"], row["user_is_white"], row["played_san"], row["reply_san"])
        groups[key].append(row)
    out = []
    for (position_key, fullmove, user_is_white, played, reply), items in groups.items():
        scores = [x["score"] for x in items if x["score"] is not None]
        n = len(scores)
        wins, draws, losses = [sum(1 for s in scores if s == target) for target in (1.0, 0.5, 0.0)]
        position_side = "Brancas" if user_is_white else "Negras"
        out.append({
            "Posição": position_key,
            "Lance do jogador": move_label(fullmove, user_is_white, played),
            "Resposta do adversário": move_label(fullmove, not user_is_white, reply),
            "Cor": position_side,
            "Amostra": n,
            "Vitórias": wins,
            "Empates": draws,
            "Derrotas": losses,
            "Derrota (%)": round(100 * losses / n, 1) if n else 0,
            "Pontuação (%)": round(100 * sum(scores) / n, 1) if n else 0,
            "Jogada (SAN inglês)": played,
            "Resposta (SAN inglês)": reply,
        })
    if not out:
        return pd.DataFrame(columns=["Posição", "Lance do jogador", "Resposta do adversário", "Cor", "Amostra", "Vitórias", "Empates", "Derrotas", "Derrota (%)", "Pontuação (%)", "Jogada (SAN inglês)", "Resposta (SAN inglês)"])
    return pd.DataFrame(out).sort_values(["Amostra", "Derrota (%)"], ascending=[False, False]).reset_index(drop=True)


def eval_cp(info: dict[str, Any], pov: chess.Color) -> int:
    score = info["score"].pov(pov).score(mate_score=10000)
    return int(score if score is not None else 0)


def pv_to_san(board: chess.Board, pv: list[chess.Move], max_moves: int = 5) -> str:
    temp = board.copy()
    san_list = []
    for move in pv[:max_moves]:
        if move not in temp.legal_moves:
            break
        fullmove = temp.fullmove_number
        mover_is_white = temp.turn == chess.WHITE
        san = temp.san(move)
        san_list.append(move_label(fullmove, mover_is_white, san))
        temp.push(move)
    return " ".join(san_list)


def classify_loss(loss_cp: int) -> str:
    if loss_cp >= 300:
        return "Erro grave"
    if loss_cp >= 150:
        return "Erro"
    if loss_cp >= 80:
        return "Imprecisão"
    return "—"


def run_engine_analysis(
    games: list[dict[str, Any]],
    username: str,
    engine_path: str,
    game_limit: int,
    seconds_per_position: float,
    max_fullmove: int,
) -> tuple[list[dict[str, Any]], str]:
    """Analyze the user's moves in the opening phase with a local UCI engine."""
    target_games = [g for g in games if g["score"] is not None][:game_limit]
    total_candidate_moves = sum(
        1 for g in target_games for m in g["moves"]
        if m["mover_is_white"] == g["user_is_white"] and m["fullmove"] <= max_fullmove
    )
    if not target_games or total_candidate_moves == 0:
        return [], "Não há lances suficientes para analisar com a configuração atual."

    reports: list[dict[str, Any]] = []
    progress = st.progress(0, text="Preparando Stockfish…")
    status = st.empty()
    completed = 0
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=15)
    except Exception as exc:
        progress.empty()
        status.empty()
        raise RuntimeError(
            "Não consegui abrir a engine. Instale o Stockfish e informe o caminho do executável "
            "(no Windows, geralmente stockfish.exe)."
        ) from exc

    try:
        try:
            engine.configure({"Threads": 1, "Hash": 128})
        except Exception:
            pass
        for g_index, g in enumerate(target_games):
            game = chess.pgn.read_game(io.StringIO(g["pgn"]))
            if game is None:
                continue
            board = game.board()
            try:
                target_user_is_white = g["user_is_white"]
                for move in game.mainline_moves():
                    current_is_user_turn = board.turn == target_user_is_white
                    if current_is_user_turn and board.fullmove_number <= max_fullmove:
                        before = board.copy()
                        played_san = before.san(move)
                        fen_before = before.fen()
                        try:
                            info_before = engine.analyse(before, chess.engine.Limit(time=seconds_per_position))
                            pv = info_before.get("pv", [])
                            best_move = pv[0] if pv else move
                            best_san = before.san(best_move)
                            best_cp = eval_cp(info_before, target_user_is_white)
                            after = before.copy()
                            after.push(move)
                            info_after = engine.analyse(after, chess.engine.Limit(time=seconds_per_position))
                            after_cp = eval_cp(info_after, target_user_is_white)
                            loss_cp = max(0, best_cp - after_cp)
                            label = classify_loss(loss_cp)
                            if loss_cp >= 80 and best_move != move:
                                reports.append({
                                    "Data": g["date"],
                                    "Abertura": g["opening"],
                                    "Cor": g["color"],
                                    "Lance": move_label(before.fullmove_number, target_user_is_white, played_san),
                                    "Jogada feita": to_pt_san(played_san),
                                    "Melhor jogada": to_pt_san(best_san),
                                    "Tipo": label,
                                    "Perda estimada (cp)": loss_cp,
                                    "Avaliação antes (cp)": best_cp,
                                    "Avaliação depois (cp)": after_cp,
                                    "FEN": fen_before,
                                    "FEN chave": fen_position_key(fen_before),
                                    "melhor_uci": best_move.uci(),
                                    "melhor_san_en": best_san,
                                    "pgn": g["pgn"],
                                    "jogo": g["game_id"],
                                    "oponente": g["opponent"],
                                    "pv": pv_to_san(before, pv),
                                })
                        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, ValueError):
                            pass
                        completed += 1
                        progress.progress(min(completed / max(total_candidate_moves, 1), 1.0), text=f"Analisando abertura: {completed}/{total_candidate_moves} decisões")
                    board.push(move)
                    # Update status without spamming the UI too much.
                    if completed and completed % 10 == 0:
                        status.caption(f"Partida {g_index + 1}/{len(target_games)} · {len(reports)} posições para revisar até agora")
            except Exception:
                # Keep results from already analyzed positions if one malformed game causes an issue.
                continue
    finally:
        engine.quit()
        progress.empty()
        status.empty()
    reports.sort(key=lambda r: r["Perda estimada (cp)"], reverse=True)
    msg = f"Análise concluída: {len(reports)} posições com imprecisão/erro detetável em {completed} decisões analisadas."
    return reports, msg


def game_rows_df(games: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for g in games:
        rows.append({
            "Data": g["date"], "Abertura": g["opening"], "ECO": g["eco"],
            "Cor": g["color"], "Resultado": g["result"], "Oponente": g["opponent"],
            "Ritmo": g["time_class"], "URL": g["url"],
        })
    return pd.DataFrame(rows)


def render_chessboard(fen: str, size: int = 380) -> None:
    try:
        board = chess.Board(fen)
        svg = chess.svg.board(board=board, size=size, coordinates=True)
        components.html(svg, height=size + 28, scrolling=False)
    except Exception:
        st.code(fen)


def engine_report_csv(reports: list[dict[str, Any]]) -> bytes:
    if not reports:
        return b""
    cols = ["Data", "Abertura", "Cor", "Lance", "Jogada feita", "Melhor jogada", "Tipo", "Perda estimada (cp)", "Avaliação antes (cp)", "Avaliação depois (cp)", "FEN", "pv"]
    return pd.DataFrame(reports)[cols].to_csv(index=False).encode("utf-8-sig")


# ------------------------------ SIDEBAR ----------------------------------
st.sidebar.title("♟️ AberturaLab")
st.sidebar.caption("Seu repertório, seus padrões, seus próximos exercícios.")
source = st.sidebar.radio("Fonte das partidas", ["Chess.com (API pública)", "Arquivo PGN"], index=0)
username = st.sidebar.text_input("Usuário do Chess.com", value="Machadodea6", help="Usado para identificar sua cor e seus resultados. Não pedimos sua senha.").strip()

entries: list[dict[str, Any]] = []
load_error: str | None = None
load_message: str | None = None

if source == "Chess.com (API pública)":
    period_label = st.sidebar.selectbox("Histórico", ["Últimos 3 meses", "Últimos 6 meses", "Últimos 12 meses", "Todo o histórico"], index=2)
    months = {"Últimos 3 meses": 3, "Últimos 6 meses": 6, "Últimos 12 meses": 12, "Todo o histórico": 0}[period_label]
    max_games = st.sidebar.selectbox("Máximo de partidas a carregar", [100, 250, 500, 1000, 2500], index=2)
    time_class_label = st.sidebar.selectbox("Ritmo", ["Todos", "Blitz", "Rápidas", "Bullet", "Diárias"], index=0)
    time_class = {"Todos": "all", "Blitz": "blitz", "Rápidas": "rapid", "Bullet": "bullet", "Diárias": "daily"}[time_class_label]
    if st.sidebar.button("Buscar minhas partidas", type="primary", use_container_width=True):
        try:
            with st.spinner("Consultando os arquivos públicos do Chess.com…"):
                fetched, message = fetch_chesscom_entries(username, months, max_games, time_class)
            st.session_state["aberturalab_entries"] = fetched
            st.session_state["aberturalab_username"] = username
            st.session_state["aberturalab_source"] = "chesscom"
            st.session_state["aberturalab_load_message"] = message
            st.session_state.pop("aberturalab_engine_reports", None)
        except Exception as exc:
            load_error = str(exc)
    if st.sidebar.button("Limpar cache da API", use_container_width=True):
        fetch_chesscom_entries.clear()
        st.sidebar.success("Cache limpo. Busque de novo para atualizar.")
else:
    uploaded = st.sidebar.file_uploader("Envie seu PGN", type=["pgn", "txt"], help="O PGN deve conter os nomes dos jogadores, para identificarmos suas partidas e sua cor.")
    if uploaded is not None:
        try:
            entries = pgn_entries_from_upload(uploaded)
            st.session_state["aberturalab_entries"] = entries
            st.session_state["aberturalab_username"] = username
            st.session_state["aberturalab_source"] = "pgn"
            st.session_state["aberturalab_load_message"] = f"Arquivo lido: {len(entries)} partidas PGN encontradas."
            if st.session_state.get("aberturalab_upload_name") != uploaded.name:
                st.session_state.pop("aberturalab_engine_reports", None)
            st.session_state["aberturalab_upload_name"] = uploaded.name
        except Exception as exc:
            load_error = f"Não consegui ler o arquivo PGN: {exc}"

st.markdown(
    """
    <div class="hero">
      <h1>♟️ AberturaLab</h1>
      <div class="muted">Treinador de repertório baseado nas suas próprias partidas — estatísticas primeiro, engine depois, exercícios no final.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(load_error)

expected_source = "chesscom" if source == "Chess.com (API pública)" else "pgn"
saved_entries = st.session_state.get("aberturalab_entries", []) if st.session_state.get("aberturalab_source") == expected_source else []
saved_username = st.session_state.get("aberturalab_username", username) or username
if not saved_entries:
    st.info("**Comece carregando suas partidas.** Use o botão na lateral para buscar o histórico público de `Machadodea6`, ou selecione **Arquivo PGN** para importar partidas exportadas do Chess.com.")
    st.markdown("""
    **O que esta primeira versão entrega**

    - aproveitamento por cor e por abertura (vitórias, empates, derrotas e pontuação);
    - posições repetidas em que uma resposta específica do adversário aparece associada a maus resultados;
    - análise dos seus lances de abertura com Stockfish, se instalado;
    - exercícios retirados das suas próprias imprecisões, com tabuleiro e resposta escondida.

    **Limite importante:** estatísticas de 1–3 partidas são pistas, não leis teóricas. O painel mostra sempre o tamanho da amostra.
    """)
    st.stop()

if saved_username != username:
    st.warning(f"As partidas carregadas são de `{saved_username}`, mas o campo atual está como `{username}`. Altere o campo para o usuário carregado ou busque/importar de novo.")
    st.stop()

with st.spinner("Organizando resultados, aberturas e respostas recorrentes…"):
    games, reply_rows, parse_warnings = parse_entries(saved_entries, saved_username)

if not games:
    st.error("Não consegui associar as partidas ao usuário informado. Confirme se o PGN contém esse nome exatamente nas tags White/Black.")
    if parse_warnings:
        with st.expander("Detalhes dos avisos"):
            for warning in parse_warnings[:12]:
                st.write(f"• {warning}")
    st.stop()

st.caption(st.session_state.get("aberturalab_load_message", f"{len(games)} partidas prontas para analisar."))
if parse_warnings:
    with st.expander(f"Avisos de importação ({len(parse_warnings)})"):
        for warning in parse_warnings[:20]:
            st.write(f"• {warning}")
        if len(parse_warnings) > 20:
            st.write(f"… e mais {len(parse_warnings) - 20} aviso(s).")

color_df = make_color_stats(games)
opening_df = make_opening_stats(games)
reply_df = make_reply_stats(reply_rows)

wins = sum(1 for g in games if g["score"] == 1.0)
draws = sum(1 for g in games if g["score"] == 0.5)
losses = sum(1 for g in games if g["score"] == 0.0)
known_results = wins + draws + losses
score_pct = (100 * (wins + 0.5 * draws) / known_results) if known_results else 0
win_pct = (100 * wins / known_results) if known_results else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Partidas analisadas", f"{len(games):,}".replace(",", "."))
m2.metric("Taxa de vitória", f"{win_pct:.1f}%")
m3.metric("Pontuação", f"{score_pct:.1f}%")
m4.metric("W / D / L", f"{wins} / {draws} / {losses}")

# Engine controls are separate from the statistical view so Stockfish is optional.
with st.expander("⚙️ Configurar análise de engine (opcional)", expanded=False):
    st.write("O Stockfish roda **no seu próprio computador**. Isso evita enviar seus PGNs para um serviço de análise externo.")
    engine_path = st.text_input("Caminho do executável Stockfish", value="stockfish", help="Se estiver no PATH, basta 'stockfish'. No Windows, use o caminho completo até stockfish.exe.")
    c1, c2, c3 = st.columns(3)
    engine_game_limit = c1.slider("Partidas mais recentes para a engine", min_value=5, max_value=min(100, max(5, len(games))), value=min(20, max(5, len(games))), step=5)
    seconds_per_position = c2.slider("Tempo por posição (segundos)", min_value=0.10, max_value=1.00, value=0.20, step=0.05)
    max_fullmove = c3.slider("Até qual lance analisar", min_value=6, max_value=20, value=12, step=2)
    st.caption("Mais tempo por posição melhora a análise, mas deixa o processo mais lento. A análise foca nos lances do usuário durante a abertura.")
    start_engine = st.button("Analisar aberturas com Stockfish", type="primary")
    if start_engine:
        try:
            with st.spinner("Stockfish está avaliando as posições…"):
                reports, message = run_engine_analysis(games, saved_username, engine_path.strip() or "stockfish", engine_game_limit, seconds_per_position, max_fullmove)
            st.session_state["aberturalab_engine_reports"] = reports
            st.session_state["aberturalab_engine_message"] = message
            st.success(message)
        except Exception as exc:
            st.error(str(exc))

reports = st.session_state.get("aberturalab_engine_reports", [])
engine_message = st.session_state.get("aberturalab_engine_message", "")

tabs = st.tabs(["Visão geral", "Repertório", "Respostas perigosas", "Exercícios Stockfish", "Partidas"])

with tabs[0]:
    st.subheader("Desempenho por cor")
    st.dataframe(color_df, use_container_width=True, hide_index=True)
    if not opening_df.empty:
        st.subheader("Aberturas mais jogadas")
        top_openings = (opening_df.groupby("Abertura", as_index=False)
                        .agg(Partidas=("Partidas", "sum"), Vitórias=("Vitórias", "sum"), Empates=("Empates", "sum"), Derrotas=("Derrotas", "sum")))
        top_openings["Taxa de vitória (%)"] = (100 * top_openings["Vitórias"] / top_openings["Partidas"]).round(1)
        top_openings = top_openings.sort_values("Partidas", ascending=False).head(10)
        st.bar_chart(top_openings.set_index("Abertura")["Partidas"])
        st.dataframe(top_openings, use_container_width=True, hide_index=True)
    st.caption("Taxa de vitória = vitórias ÷ partidas com resultado conhecido. Pontuação = (vitórias + ½ empates) ÷ partidas. Não são a mesma métrica.")

with tabs[1]:
    st.subheader("Seu repertório por abertura e cor")
    st.write("Separei a mesma abertura por brancas e negras para evitar misturar repertórios diferentes.")
    if opening_df.empty:
        st.info("Não há tags de abertura utilizáveis nas partidas carregadas.")
    else:
        min_sample = st.slider("Mostrar apenas aberturas com pelo menos N partidas", min_value=1, max_value=max(1, min(20, int(opening_df["Partidas"].max()))), value=2)
        filtered_openings = opening_df[opening_df["Partidas"] >= min_sample].sort_values(["Partidas", "Pontuação (%)"], ascending=[False, False])
        st.dataframe(filtered_openings, use_container_width=True, hide_index=True)
        st.download_button("Baixar estatísticas por abertura (CSV)", data=filtered_openings.to_csv(index=False).encode("utf-8-sig"), file_name="aberturalab_aberturas.csv", mime="text/csv")
        st.caption("Cuidado com rankings de 1–2 partidas: uma única vitória ou derrota muda muito a porcentagem. Compare sempre abertura, cor e tamanho da amostra.")

with tabs[2]:
    st.subheader("Respostas que aparecem nos seus resultados ruins")
    st.write("Aqui o sistema agrupa posições iguais, o lance que você jogou e a resposta imediata do adversário. Isso ajuda a encontrar padrões concretos — não prova, sozinho, que uma resposta é teoricamente vencedora.")
    if reply_df.empty:
        st.info("Ainda não encontrei sequências suficientes para agrupar.")
    else:
        c1, c2 = st.columns(2)
        min_n = c1.slider("Amostra mínima", min_value=1, max_value=max(1, min(20, int(reply_df["Amostra"].max()))), value=min(3, max(1, int(reply_df["Amostra"].max()))))
        only_bad = c2.checkbox("Priorizar padrões em que perdi com frequência", value=True)
        patterns = reply_df[reply_df["Amostra"] >= min_n].copy()
        if only_bad:
            patterns = patterns.sort_values(["Derrota (%)", "Amostra"], ascending=[False, False])
        else:
            patterns = patterns.sort_values(["Amostra", "Derrota (%)"], ascending=[False, False])
        if patterns.empty:
            st.info("Nenhum padrão com essa amostra mínima. Reduza o filtro ou carregue mais partidas.")
        else:
            st.dataframe(patterns.drop(columns=["Posição", "Jogada (SAN inglês)", "Resposta (SAN inglês)"]), use_container_width=True, hide_index=True)
            selected = st.selectbox("Ver posição de um padrão", options=list(range(len(patterns))), format_func=lambda i: f"{patterns.iloc[i]['Lance do jogador']} · resposta {patterns.iloc[i]['Resposta do adversário']} · {int(patterns.iloc[i]['Amostra'])} partida(s)")
            chosen = patterns.iloc[selected]
            render_chessboard(" ".join([chosen["Posição"], "0 1"]), size=320)
            st.caption("Tabuleiro reconstruído a partir da posição antes do seu lance. Se a posição envolver regras especiais, o FEN completo original é preservado no CSV geral; a chave mostrada aqui omite contadores de lances.")
        st.download_button("Baixar padrões de resposta (CSV)", data=reply_df.to_csv(index=False).encode("utf-8-sig"), file_name="aberturalab_respostas.csv", mime="text/csv")

with tabs[3]:
    st.subheader("Exercícios tirados das suas próprias partidas")
    if not reports:
        st.info("Ainda não há exercícios. Abra **Configurar análise de engine**, indique onde está o Stockfish e clique em **Analisar aberturas com Stockfish**.")
        st.markdown("O exercício aparece antes do seu lance original. Tente achar uma jogada melhor no tabuleiro; depois revele a principal sugestão da engine.")
    else:
        st.success(engine_message or f"{len(reports)} posições prontas para revisar.")
        st.caption("A resposta é a primeira escolha do Stockfish nessa análise — pode haver outras jogadas boas. Avaliações e classificação dependem do tempo de cálculo configurado.")
        labels = [f"#{i+1} · {r['Lance']} · {r['Tipo']} · perda ~{r['Perda estimada (cp)']} cp" for i, r in enumerate(reports)]
        selected_ex = st.selectbox("Escolha um exercício", options=list(range(len(reports))), format_func=lambda i: labels[i], key="exercise_choice")
        ex = reports[selected_ex]
        left, right = st.columns([1, 1])
        with left:
            render_chessboard(ex["FEN"], size=390)
        with right:
            st.markdown(f"**Abertura:** {ex['Abertura']}")
            st.markdown(f"**Cor:** {ex['Cor']} · **Lance:** {ex['Lance']}")
            st.markdown(f"**Na partida você jogou:** {ex['Jogada feita']}")
            guess = st.text_input("Qual lance você jogaria? (ex.: Cf3, Nf3 ou coordenadas e2e4)", key=f"guess_{selected_ex}")
            if st.button("Conferir minha resposta", key=f"check_{selected_ex}"):
                proposed = guess.strip().replace("+", "").replace("#", "")
                expected_values = {ex["melhor_uci"].lower(), ex["melhor_san_en"].replace("+", "").replace("#", ""), to_pt_san(ex["melhor_san_en"]).replace("+", "").replace("#", "")}
                proposed_clean = proposed.replace("+", "").replace("#", "")
                if proposed_clean.lower() in {x.lower() for x in expected_values}:
                    st.success("Boa! Você encontrou a principal escolha desta análise.")
                else:
                    st.info("Não é a primeira escolha encontrada pelo Stockfish nesta análise. Revele a resposta e compare a ideia tática/posicional.")
            with st.expander("Revelar resposta e ideia da engine"):
                st.markdown(f"**Principal sugestão:** {ex['Melhor jogada']}")
                st.markdown(f"**Linha principal:** `{ex['pv']}`")
                st.markdown(f"**Perda estimada na partida:** {ex['Perda estimada (cp)']} centipawns")
                st.markdown(f"**Avaliação antes/depois:** {ex['Avaliação antes (cp)']} / {ex['Avaliação depois (cp)']} cp, do ponto de vista do jogador analisado.")
        st.divider()
        st.markdown("**Resumo dos pontos para revisar**")
        report_df = pd.DataFrame(reports)
        view_cols = ["Data", "Abertura", "Cor", "Lance", "Jogada feita", "Melhor jogada", "Tipo", "Perda estimada (cp)"]
        st.dataframe(report_df[view_cols], use_container_width=True, hide_index=True)
        st.download_button("Baixar relatório de engine (CSV)", data=engine_report_csv(reports), file_name="aberturalab_stockfish.csv", mime="text/csv")

with tabs[4]:
    st.subheader("Partidas incluídas no relatório")
    games_df = game_rows_df(games)
    f1, f2 = st.columns(2)
    color_filter = f1.selectbox("Filtrar por cor", ["Todas", "Brancas", "Negras"])
    result_filter = f2.selectbox("Filtrar por resultado", ["Todos", "Vitória", "Empate", "Derrota", "Desconhecido"])
    filtered_games_df = games_df.copy()
    if color_filter != "Todas":
        filtered_games_df = filtered_games_df[filtered_games_df["Cor"] == color_filter]
    if result_filter != "Todos":
        filtered_games_df = filtered_games_df[filtered_games_df["Resultado"] == result_filter]
    st.dataframe(filtered_games_df, use_container_width=True, hide_index=True, column_config={"URL": st.column_config.LinkColumn("Partida", display_text="Abrir")})
    st.download_button("Baixar partidas listadas (CSV)", data=filtered_games_df.to_csv(index=False).encode("utf-8-sig"), file_name="aberturalab_partidas.csv", mime="text/csv")

st.divider()
st.caption("AberturaLab é um protótipo local de estudo, não afiliado ao Chess.com ou ao Stockfish. As conclusões estatísticas descrevem somente as partidas carregadas. A engine é uma referência forte, mas a análise varia com tempo, versão e configuração.")
