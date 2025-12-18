import os
import re
import time
import random
import datetime
from typing import List, Optional, Dict, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

import streamlit.components.v1 as components


# =========================
# API / Ads keys (直書き禁止)
# =========================
def get_secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def load_api_key() -> Optional[str]:
    return get_secret("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


API_KEY = load_api_key()
if not API_KEY:
    st.error("APIキーがありません。.streamlit/secrets.toml か 環境変数 GEMINI_API_KEY を設定してください。")
    st.stop()


ADSENSE_CLIENT = get_secret("ADSENSE_CLIENT") or os.environ.get("ADSENSE_CLIENT")  # 例: ca-pub-xxxxxxxx
ADSENSE_SLOT_TOP = get_secret("ADSENSE_SLOT_TOP") or os.environ.get("ADSENSE_SLOT_TOP")  # 例: 1234567890
ADSENSE_SLOT_BOTTOM = get_secret("ADSENSE_SLOT_BOTTOM") or os.environ.get("ADSENSE_SLOT_BOTTOM")


@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)


MODEL_NAME = "gemini-2.5-flash"  # ユーザーに選ばせない（固定）


# =========================
# 日付デフォルト（次の土曜）
# =========================
def default_race_date() -> datetime.date:
    today = datetime.date.today()
    wd = today.weekday()  # Mon=0 ... Sun=6
    if wd in (5, 6):  # Sat/Sun
        return today
    days_ahead = (5 - wd) % 7
    if days_ahead == 0:
        days_ahead = 7
    return today + datetime.timedelta(days=days_ahead)


# =========================
# Structured Output schema
# =========================
class Pick(BaseModel):
    枠番: int = Field(..., ge=1, le=8)
    馬番: int = Field(..., ge=1, le=18)
    馬名: str = Field(..., min_length=1)

    人気想定: Optional[str] = None
    評価: str = Field(description="S/A/B/C のいずれか")

    能力: int = Field(ge=0, le=100)
    適性: int = Field(ge=0, le=100)
    展開: int = Field(ge=0, le=100)
    調子: int = Field(ge=0, le=100)
    不確実性: int = Field(ge=0, le=100, description="高いほど情報不足")

    思考による分析結果: str
    懸念点: str


class PredictionResult(BaseModel):
    picks: List[Pick]


# =========================
# Retry / utils
# =========================
def parse_retry_seconds(e: Exception) -> Optional[float]:
    s = str(e)
    m = re.search(r"Please retry in ([0-9.]+)s", s)
    if m:
        return float(m.group(1))
    m = re.search(r"'retryDelay': '([0-9.]+)s'", s)
    if m:
        return float(m.group(1))
    return None


def is_quota_zero_message(e: Exception) -> bool:
    s = str(e)
    return ("limit: 0" in s) and ("Quota exceeded" in s)


def call_with_retry(fn, max_attempts=6, base_sleep=0.8, max_sleep=20.0):
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            last = e
            msg = str(e)
            retryable = (
                "503" in msg or "UNAVAILABLE" in msg
                or "429" in msg or "RESOURCE_EXHAUSTED" in msg
            )
            if (not retryable) or attempt == max_attempts:
                raise

            delay = parse_retry_seconds(e)
            if delay is not None:
                sleep = min(max_sleep, delay + random.random())
            else:
                sleep = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                sleep *= (0.7 + 0.6 * random.random())
            time.sleep(sleep)
    raise last


def extract_sources_from_response(response) -> List[dict]:
    sources = []
    try:
        cand = response.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        if not gm:
            return sources
        chunks = getattr(gm, "grounding_chunks", None) or []
        for ch in chunks:
            web = getattr(ch, "web", None)
            if web and getattr(web, "uri", None):
                title = getattr(web, "title", None) or web.uri
                sources.append({"title": title, "uri": web.uri})
    except Exception:
        pass
    return sources


def uniq_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values())
    if s <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: v / s for k, v in w.items()}


def compute_total_score(p: Pick, w: Dict[str, float]) -> float:
    return (
        w["能力"] * p.能力
        + w["適性"] * p.適性
        + w["展開"] * p.展開
        + w["調子"] * p.調子
        + w["確度"] * (100 - p.不確実性)
    )


def grade_rank(g: str) -> int:
    # ソート用：Sが最強
    return {"S": 0, "A": 1, "B": 2, "C": 3}.get(g, 9)


# =========================
# Ads (AdSense想定)
# =========================
def render_adsense(slot: Optional[str], height: int = 120):
    """
    AdSenseを入れる枠（client/slotが未設定なら何も出さない）
    NOTE: 審査通過後に本番コードで動く想定。審査前は出さない方が無難。
    """
    if not ADSENSE_CLIENT or not slot:
        return

    html = f"""
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_CLIENT}"
     crossorigin="anonymous"></script>
<ins class="adsbygoogle"
     style="display:block"
     data-ad-client="{ADSENSE_CLIENT}"
     data-ad-slot="{slot}"
     data-ad-format="auto"
     data-full-width-responsive="true"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({{}});
</script>
"""
    components.html(html, height=height)


# =========================
# Entry (出馬表) extraction
# =========================
ENTRY_BLOCK_RE = re.compile(r"<ENTRY>\s*(.*?)\s*</ENTRY>", re.S)


def parse_entry_block(text: str) -> List[Dict[str, object]]:
    m = ENTRY_BLOCK_RE.search(text)
    if not m:
        return []

    body = m.group(1).strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

    entries = []
    for ln in lines:
        if ln.startswith("枠番"):
            continue

        parts = re.split(r"[,\uFF0C\t]+|\s{2,}", ln)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 3:
            m2 = re.match(r"^\s*(\d+)\s+(\d+)\s+(.+?)\s*$", ln)
            if not m2:
                continue
            parts = [m2.group(1), m2.group(2), m2.group(3)]

        try:
            waku = int(parts[0])
            uma = int(parts[1])
        except Exception:
            continue

        name = parts[2].strip()
        if not name:
            continue

        entries.append({"枠番": waku, "馬番": uma, "馬名": name})

    return entries


def search_candidate_urls(query: str) -> Tuple[List[str], List[dict]]:
    client = get_client()

    prompt = f"""
次のレースの「出馬表（枠番・馬番・馬名が載っているページ）」のURL候補を探してください。
優先: netkeiba の shutuba（race.netkeiba.com/race/shutuba.html?race_id=...）など。

対象レース: {query}

出力は文章でOK（URLを含める）。ただし正確なURLを含めてください。
"""
    cfg = types.GenerateContentConfig(
        tools=[{"google_search": {}}],
        temperature=0.0,
    )

    resp = call_with_retry(lambda: client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=cfg,
    ))

    sources = extract_sources_from_response(resp)

    urls = [s["uri"] for s in sources if s.get("uri")]
    urls += re.findall(r"https?://[^\s)】]+", resp.text or "")
    urls = uniq_keep_order(urls)
    return urls, sources


def rank_urls(urls: List[str]) -> List[str]:
    def score(u: str) -> int:
        u2 = u.lower()
        sc = 0
        if "race.netkeiba.com" in u2 and "shutuba" in u2:
            sc += 100
        if "race_id=" in u2:
            sc += 30
        if "shutuba" in u2:
            sc += 20
        return sc

    return sorted(urls, key=score, reverse=True)


def extract_entry_from_url(url: str) -> List[Dict[str, object]]:
    client = get_client()

    cfg = types.GenerateContentConfig(
        tools=[{"url_context": {}}],
        temperature=0.0,
    )

    prompt = f"""
次のURLの内容を読み取り、出馬表（枠番・馬番・馬名）だけを抽出してください。

【厳守】
- 馬名はページの表記を「そのまま」コピー。キャッチコピー/説明文/装飾は禁止。
- 推測で埋めない。不明ならその行は出さない。
- 出力は必ず次の形式のみ（これ以外の文字は禁止）:

<ENTRY>
枠番,馬番,馬名
1,1,馬名
...
</ENTRY>

URL: {url}
"""
    resp = call_with_retry(lambda: client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=cfg,
    ))

    return parse_entry_block(resp.text or "")


def get_entry_list(query: str, entry_url_override: str = "") -> Tuple[List[Dict[str, object]], str, List[dict]]:
    if entry_url_override.strip():
        url = entry_url_override.strip()
        entries = extract_entry_from_url(url)
        if len(entries) >= 5:
            return entries, url, []
        raise ValueError("貼り付けた出馬表URLから枠番/馬番/馬名を取得できませんでした。別の出馬表URLを貼ってください。")

    urls, sources = search_candidate_urls(query)
    urls_ranked = rank_urls(urls)

    for url in urls_ranked[:8]:
        entries = extract_entry_from_url(url)
        if len(entries) >= 5:
            return entries, url, sources

    raise ValueError("出馬表(ENTRYブロック)が取得できませんでした。日付/競馬場/Rを確認し、出馬表URLを貼るか、日付を変更して再実行してください。")


# =========================
# Prediction flow
# =========================
def build_research_memo(query: str, entry_url: str, entries: List[Dict[str, object]]) -> Tuple[str, List[dict]]:
    client = get_client()

    horses_compact = " / ".join([f"{e['馬名']}({e['枠番']}-{e['馬番']})" for e in entries])

    cfg = types.GenerateContentConfig(
        tools=[{"url_context": {}}, {"google_search": {}}],
        temperature=0.3,
    )

    prompt = f"""
あなたは論理的な競馬分析官です。
次のレースについて、最新情報を調査して「分析メモ」を作ってください。

【対象レース】
{query}

【出馬表URL（枠番・馬番・馬名の根拠）】
{entry_url}

【出走馬（短縮）】
{horses_compact}

【出力（テキスト）】
- レース条件（天気/馬場/コース傾向/枠の有利不利）
- ペースと展開想定（脚質分布）
- 有力馬候補（根拠・不安要素）
- 情報不足（取れない情報があれば明示）
"""
    resp = call_with_retry(lambda: client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=cfg,
    ))
    sources = extract_sources_from_response(resp)
    return (resp.text or ""), sources


def validate_picks_against_entries(picks: List[Pick], entries: List[Dict[str, object]]) -> Tuple[bool, str]:
    name_to_nums = {e["馬名"]: (e["枠番"], e["馬番"]) for e in entries}
    for p in picks:
        if p.馬名 not in name_to_nums:
            return False, f"馬名が出馬表に存在しません: {p.馬名}"
        w, n = name_to_nums[p.馬名]
        if p.枠番 != w or p.馬番 != n:
            return False, f"枠番/馬番が出馬表と一致しません: {p.馬名} (出力={p.枠番}-{p.馬番}, 正={w}-{n})"
        if p.評価 not in ["S", "A", "B", "C"]:
            return False, f"評価が不正です: {p.馬名} 評価={p.評価}"
    return True, ""


def format_picks_to_json(entries: List[Dict[str, object]], memo: str, weights: Dict[str, float]) -> PredictionResult:
    client = get_client()

    allowed = [{"枠番": e["枠番"], "馬番": e["馬番"], "馬名": e["馬名"]} for e in entries]

    prompt = f"""
あなたは論理的な競馬分析官です。
次の「出馬表リスト」に載っている馬だけから、有力馬を5〜6頭選び、指定のJSON形式のみで出力してください。

【絶対ルール】
- 馬名は「出馬表リスト」の文字列を完全一致で使用（キャッチコピー禁止）
- 枠番/馬番も出馬表リストの値をそのまま使用
- 出力はJSONのみ（前後に文章禁止）
- 評価は S/A/B/C
- スコアは 0〜100 の整数

【重視ポイント（正規化済）】
能力={weights["能力"]:.3f}, 適性={weights["適性"]:.3f}, 展開={weights["展開"]:.3f}, 調子={weights["調子"]:.3f}, 確度={weights["確度"]:.3f}

【出馬表リスト】
{allowed}

【分析メモ】
{memo}

【出力JSONフォーマット】
{{"picks":[{{"枠番":1,"馬番":1,"馬名":"...","人気想定":"1番人気など","評価":"S","能力":80,"適性":70,"展開":60,"調子":65,"不確実性":20,"思考による分析結果":"...","懸念点":"..."}}]}}
"""
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=PredictionResult.model_json_schema(),
        temperature=0.2,
    )

    resp = call_with_retry(lambda: client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=cfg,
    ))
    return PredictionResult.model_validate_json(resp.text)


def get_prediction(entries: List[Dict[str, object]], memo: str, weights: Dict[str, float]) -> List[Pick]:
    last_err = None
    for _ in range(3):
        result = format_picks_to_json(entries, memo, weights)
        ok, reason = validate_picks_against_entries(result.picks, entries)
        if ok:
            return result.picks
        last_err = reason
        memo = memo + f"\n\n[重要] 前回の出力は不正でした。理由: {reason}\n出馬表リストの馬名・枠番・馬番を必ず完全一致で使ってください。"
    raise ValueError(f"出力の整合性チェックに失敗しました: {last_err}")


def run_prediction(query: str, weights: Dict[str, float], entry_url_override: str = "") -> Tuple[pd.DataFrame, List[dict], str]:
    entries, entry_url, sources1 = get_entry_list(query, entry_url_override=entry_url_override)
    memo, sources2 = build_research_memo(query, entry_url, entries)
    picks = get_prediction(entries, memo, weights)

    df = pd.DataFrame([p.model_dump() for p in picks])
    df["総合点"] = [round(compute_total_score(p, weights), 1) for p in picks]

    # 表示名（買い目用）
    df["馬番付き"] = df.apply(lambda r: f"{int(r['馬番'])}({r['馬名']})", axis=1)

    # 安定ソート：総合点 desc → 評価(S>A>B>C) → 馬番 asc
    df["評価順位"] = df["評価"].apply(grade_rank)
    df = df.sort_values(by=["総合点", "評価順位", "馬番"], ascending=[False, True, True]).reset_index(drop=True)
    df = df.drop(columns=["評価順位"])

    sources = sources1 + sources2
    seen = set()
    uniq_sources = []
    for s in sources:
        u = s.get("uri")
        if u and u not in seen:
            seen.add(u)
            uniq_sources.append(s)

    return df, uniq_sources, entry_url


# =========================
# Cache（API無駄撃ち防止）
# =========================
def weights_to_key(weights: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    # 小数誤差で別物扱いにならないよう丸める
    return (
        round(weights["能力"], 4),
        round(weights["適性"], 4),
        round(weights["展開"], 4),
        round(weights["調子"], 4),
        round(weights["確度"], 4),
    )


@st.cache_data(ttl=600, show_spinner=False)  # 10分キャッシュ
def run_prediction_cached(query: str, weights_key: Tuple[float, float, float, float, float], entry_url_override: str):
    weights = {"能力": weights_key[0], "適性": weights_key[1], "展開": weights_key[2], "調子": weights_key[3], "確度": weights_key[4]}
    df, sources, entry_url = run_prediction(query=query, weights=weights, entry_url_override=entry_url_override)
    return df, sources, entry_url


# =========================
# Betting helper（馬番付き）
# =========================
def make_bets(df_ranked: pd.DataFrame, ticket_type: str, budget_yen: int) -> pd.DataFrame:
    if budget_yen <= 0 or df_ranked.empty:
        return pd.DataFrame(columns=["券種", "買い目", "金額(円)"])

    top = df_ranked["馬番付き"].tolist()
    unit = 100

    def split_amount(total: int, k: int) -> List[int]:
        total = (total // unit) * unit
        if k <= 0 or total <= 0:
            return []
        base = (total // k // unit) * unit
        amounts = [base] * k
        rem = total - base * k
        i = 0
        while rem >= unit and i < k:
            amounts[i] += unit
            rem -= unit
            i += 1
        return amounts

    bets = []

    if ticket_type in ["馬連", "ワイド", "馬単"]:
        if len(top) >= 2:
            axis = top[0]
            opp = top[1:4]
            combos = [f"{axis} → {o}" for o in opp] if ticket_type == "馬単" else [f"{axis} - {o}" for o in opp]
            amts = split_amount(budget_yen, len(combos))
            for c, a in zip(combos, amts):
                bets.append({"券種": ticket_type, "買い目": c, "金額(円)": a})

    elif ticket_type == "単勝":
        amts = split_amount(budget_yen, 1)
        bets.append({"券種": "単勝", "買い目": top[0], "金額(円)": amts[0] if amts else 0})

    elif ticket_type == "複勝":
        k = min(2, len(top))
        combos = top[:k]
        amts = split_amount(budget_yen, k)
        for c, a in zip(combos, amts):
            bets.append({"券種": "複勝", "買い目": c, "金額(円)": a})

    elif ticket_type == "三連複":
        if len(top) >= 3:
            a, b = top[0], top[1]
            combos = [f"{a} - {b} - {c}" for c in top[2:5]]
            amts = split_amount(budget_yen, len(combos))
            for c, aamt in zip(combos, amts):
                bets.append({"券種": "三連複", "買い目": c, "金額(円)": aamt})

    elif ticket_type == "三連単":
        if len(top) >= 3:
            a, b = top[0], top[1]
            combos = [f"{a} → {b} → {c}" for c in top[2:5]]
            amts = split_amount(budget_yen, len(combos))
            for c, aamt in zip(combos, amts):
                bets.append({"券種": "三連単", "買い目": c, "金額(円)": aamt})

    return pd.DataFrame(bets)


# =========================
# UI
# =========================
JRA_PLACES = ["中山", "阪神", "東京", "京都", "中京", "小倉", "新潟", "福島", "札幌", "函館"]
NAR_PLACES = ["大井", "川崎"]

st.set_page_config(page_title="AI競馬予想", layout="wide")
st.title("🐎 AI競馬予想")
st.caption("日付・競馬場・R番号を選んで押すだけ。重視ポイントだけ調整できます。")

# 上部広告（設定した場合のみ表示）
render_adsense(ADSENSE_SLOT_TOP, height=110)

with st.sidebar:
    st.header("レース入力")
    race_date = st.date_input("日付", value=default_race_date())

    places = JRA_PLACES + NAR_PLACES
    place = st.selectbox("競馬場", places)

    race_num = st.number_input("レース番号 (R)", min_value=1, max_value=12, value=11)

    with st.expander("うまくいかない時（任意）"):
        entry_url_override = st.text_input("出馬表URLを貼る（任意）", value="")

    st.divider()
    st.header("重視ポイント（合計は自動調整）")
    w_ability = st.slider("能力（実績・クラス）", 0, 100, 40, 1)
    w_fit = st.slider("適性（距離・馬場・コース）", 0, 100, 25, 1)
    w_pace = st.slider("展開（脚質・ペース）", 0, 100, 20, 1)
    w_form = st.slider("調子（調教・気配）", 0, 100, 15, 1)
    w_cert = st.slider("確度（情報が揃っているか）", 0, 100, 20, 1)

    st.divider()
    st.header("買い目（任意）")
    ticket_type = st.selectbox("券種", ["馬連", "ワイド", "馬単", "三連複", "三連単", "単勝", "複勝"], index=0)
    budget_yen = st.number_input("予算（円）", min_value=0, step=100, value=1000)

    invalid_jra_day = (place in JRA_PLACES) and (race_date.weekday() not in (5, 6))
    if invalid_jra_day:
        st.warning("JRA（中央）は基本土日開催です。土日に変更してください。")

    run_btn = st.button("AI予想を作成", type="primary", disabled=invalid_jra_day)

if run_btn:
    date_str = race_date.strftime("%Y年%m月%d日")
    rn = f"{race_num}R"
    query = f"{date_str} {place} {rn} 出馬表 枠順"

    st.subheader(f"🎯 対象: {date_str} {place} {rn}")

    weights = normalize_weights({
        "能力": float(w_ability),
        "適性": float(w_fit),
        "展開": float(w_pace),
        "調子": float(w_form),
        "確度": float(w_cert),
    })

    # リアルタイム経過時間（整数のみ）
    time_box = st.empty()
    status_box = st.empty()
    start = time.perf_counter()

    def set_elapsed(prefix: str = "⏱ 経過時間"):
        elapsed = int(time.perf_counter() - start)
        time_box.info(f"{prefix}: {elapsed} 秒")

    import concurrent.futures as cf

    weights_key = weights_to_key(weights)

    def job():
        return run_prediction_cached(query=query, weights_key=weights_key, entry_url_override=entry_url_override)

    status_box.write("AIが情報収集・分析しています…")
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(job)
        while not fut.done():
            set_elapsed()
            time.sleep(0.2)

        try:
            df, sources, entry_url = fut.result()
        except Exception as e:
            set_elapsed("✖ 失敗（経過時間）")
            if "ENTRYブロック" in str(e):
                st.error("出馬表が取れませんでした。まず日付（開催日）を確認。ダメなら「出馬表URLを貼る」を使うのが最短です。")
            elif isinstance(e, genai_errors.ClientError) and is_quota_zero_message(e):
                st.error("Gemini API の利用枠が 0 の可能性があります（Billing/プラン設定）。Google側の使用状況を確認してください。")
            else:
                st.error(str(e))
            st.exception(e)
            st.stop()

    set_elapsed("✅ 完了（経過時間）")
    status_box.empty()

    # 見た目改善：本命を上に出す
    if not df.empty:
        top = df.iloc[0]
        st.metric("本命（暫定）", f"{int(top['馬番'])}番 {top['馬名']}", f"総合点 {top['総合点']} / 評価 {top['評価']}")

    st.success("予想を作成しました！")

    st.markdown("### 📌 出馬表の参照URL")
    if entry_url:
        st.markdown(f"➡️ [出馬表を開く]({entry_url})")
    else:
        st.caption("参照URLが取れない場合もあります。")

    st.markdown("### 📊 有力馬（5〜6頭）")
    show_cols = [
        "枠番", "馬番", "馬名",
        "評価", "総合点", "人気想定",
        "能力", "適性", "展開", "調子", "不確実性",
        "思考による分析結果", "懸念点"
    ]
    st.dataframe(df[show_cols], use_container_width=True)

    # CSV
    csv = df[show_cols].to_csv(index=False).encode("utf-8_sig")
    st.download_button("CSVをダウンロード", data=csv, file_name=f"{place}_{race_num}R_{race_date:%Y%m%d}_ai.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("### 💰 買い目（参考）")
    bets_df = make_bets(df_ranked=df, ticket_type=ticket_type, budget_yen=int(budget_yen))
    if bets_df.empty:
        st.info("買い目は未作成です（予算0円 or データ不足）。")
    else:
        st.dataframe(bets_df, use_container_width=True)

    with st.expander("🔎 参考ソース（取得できた場合）"):
        if sources:
            for s in sources[:10]:
                st.write(f"- {s.get('title','')}: {s.get('uri','')}")
        else:
            st.caption("ソースが取得できない場合もあります。")

    # 下部広告（設定した場合のみ表示）
    render_adsense(ADSENSE_SLOT_BOTTOM, height=140)

else:
    st.info("左の入力を埋めて「AI予想を作成」を押してください。")
