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


# =========================
# API KEY（直書き禁止）
# =========================
def load_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        key = None
    return key or os.environ.get("GEMINI_API_KEY")


API_KEY = load_api_key()
if not API_KEY:
    st.error("APIキーがありません。.streamlit/secrets.toml か 環境変数 GEMINI_API_KEY を設定してください。")
    st.stop()


@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)


# =========================
# Structured Output schema
# =========================
class HorsePick(BaseModel):
    馬名: str
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
    picks: List[HorsePick]


# =========================
# Helpers（ログは出さない）
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


def sleep_with_timer(total: float, update_time_cb):
    """待機中も経過時間表示だけは更新する（ログは出さない）"""
    remaining = max(0.0, float(total))
    while remaining > 0:
        update_time_cb()
        step = 1.0 if remaining > 1.0 else remaining
        time.sleep(step)
        remaining -= step


def call_with_retry(fn, update_time_cb, max_attempts=6, base_sleep=0.8, max_sleep=20.0):
    """
    503/429など一時エラーに強いリトライ（retryDelay優先）
    ※ログは表示しない
    """
    for attempt in range(1, max_attempts + 1):
        update_time_cb()
        try:
            return fn()

        except (genai_errors.ServerError, genai_errors.ClientError) as e:
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

            sleep_with_timer(sleep, update_time_cb)


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


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values())
    if s <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: v / s for k, v in w.items()}


def compute_total_score(p: HorsePick, w: Dict[str, float]) -> float:
    return (
        w["能力"] * p.能力
        + w["適性"] * p.適性
        + w["展開"] * p.展開
        + w["調子"] * p.調子
        + w["確度"] * (100 - p.不確実性)
    )


def make_bets(df_ranked: pd.DataFrame, ticket_type: str, budget_yen: int) -> pd.DataFrame:
    if budget_yen <= 0 or df_ranked.empty:
        return pd.DataFrame(columns=["券種", "買い目", "金額(円)"])

    top = df_ranked["馬名"].tolist()
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
            opp = top[1:4]  # 最大3点
            if ticket_type == "馬単":
                combos = [f"{axis} → {o}" for o in opp]
            else:
                combos = [f"{axis} - {o}" for o in opp]
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
            combos = [f"{a} - {b} - {c}" for c in top[2:5]]  # 最大3点
            amts = split_amount(budget_yen, len(combos))
            for c, aamt in zip(combos, amts):
                bets.append({"券種": "三連複", "買い目": c, "金額(円)": aamt})

    elif ticket_type == "三連単":
        if len(top) >= 3:
            a, b = top[0], top[1]
            combos = [f"{a} → {b} → {c}" for c in top[2:5]]  # 最大3点
            amts = split_amount(budget_yen, len(combos))
            for c, aamt in zip(combos, amts):
                bets.append({"券種": "三連単", "買い目": c, "金額(円)": aamt})

    return pd.DataFrame(bets)


# =========================
# Core: Search -> JSON (2-step)
# =========================
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

def run_prediction(query: str, weights: Dict[str, float], force_refresh: bool, update_time_cb) -> Tuple[pd.DataFrame, List[dict]]:
    client = get_client()

    # 同一クエリの検索メモをセッションに保持（安定運用UIは出さないが、内部で無駄打ちを減らす）
    st.session_state.setdefault("memo_cache", {})
    memo_cache: Dict[str, dict] = st.session_state["memo_cache"]

    memo = None
    sources: List[dict] = []

    if (not force_refresh) and (query in memo_cache):
        memo = memo_cache[query]["memo"]
        sources = memo_cache[query]["sources"]

    # Step1: 検索（toolsあり）
    if memo is None:
        prompt_research = f"""
あなたは論理的な競馬分析官です。
Google検索（最新情報）を使って、次のレースを分析してください。

【対象レース】
{query}

注意：
- レース名が不明でも、日付・競馬場・R番号が一致するレースを対象にしてください。

次を必ず含めて、箇条書き中心で整理してください：
- レース条件（天気・馬場・コース傾向・枠の傾向）
- 出走馬（有力馬候補）と根拠
- 展開想定（ペース、脚質分布）
- 状態面（調教/馬体重/コメント等が取れれば）
- 情報不足・不確実性（取れない情報は明示）
"""

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        cfg1 = types.GenerateContentConfig(tools=[grounding_tool], temperature=0.3)

        last_err = None
        for mn in MODEL_CANDIDATES:
            try:
                resp1 = call_with_retry(
                    lambda: client.models.generate_content(
                        model=mn,
                        contents=prompt_research,
                        config=cfg1,
                    ),
                    update_time_cb=update_time_cb,
                )
                memo = resp1.text
                sources = extract_sources_from_response(resp1)
                memo_cache[query] = {"memo": memo, "sources": sources}
                break
            except Exception as e:
                last_err = e
                continue

        if memo is None:
            raise last_err

    # Step2: JSON整形（toolsなし）
    prompt_format = f"""
次の分析メモをもとに、有力馬を5〜6頭に絞って結論を出してください。

【出力要件】
- 出力は必ず JSON のみ（JSON以外の文字は禁止）
- トップレベルは次の形：
  {{ "picks": [ ... ] }}
- 評価は S/A/B/C のいずれか
- 各スコアは整数で 0〜100
- 不確実性は高いほど情報不足
- コメントは2〜5文で簡潔に

【分析メモ】
{memo}
"""

    cfg2 = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=PredictionResult.model_json_schema(),
        temperature=0.2,
    )

    last_err2 = None
    result: Optional[PredictionResult] = None

    for mn in MODEL_CANDIDATES:
        try:
            resp2 = call_with_retry(
                lambda: client.models.generate_content(
                    model=mn,
                    contents=prompt_format,
                    config=cfg2,
                ),
                update_time_cb=update_time_cb,
            )
            result = PredictionResult.model_validate_json(resp2.text)
            break
        except Exception as e:
            last_err2 = e
            continue

    if result is None:
        raise last_err2

    df = pd.DataFrame([p.model_dump() for p in result.picks])
    df["総合点"] = [round(compute_total_score(p, weights), 1) for p in result.picks]
    df = df.sort_values(by="総合点", ascending=False).reset_index(drop=True)

    return df, sources


# =========================
# UI
# =========================
st.set_page_config(page_title="AI競馬予想", layout="wide")
st.title("🐎 AI競馬予想（かんたん）")
st.caption("日付・競馬場・R番号を選んで押すだけ。重視ポイントだけ調整できます。ログは表示しません。")

with st.sidebar:
    st.header("レース入力")
    race_date = st.date_input("日付", value=datetime.date.today())

    places = ["中山", "阪神", "東京", "京都", "中京", "小倉", "新潟", "福島", "札幌", "函館", "大井", "川崎"]
    place = st.selectbox("競馬場", places)

    race_num = st.number_input("レース番号 (R)", min_value=1, max_value=12, value=11)

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

    # 初心者向け：必要なときだけ更新
    force_refresh = st.checkbox("最新情報で再検索する（必要なときだけ）", value=False)

    run_btn = st.button("AI予想を作成", type="primary")


if run_btn:
    date_str = race_date.strftime("%Y年%m月%d日")
    rn = f"{race_num}R"
    query = f"{date_str} {place}競馬場 {rn} 出馬表 予想".strip()

    st.subheader(f"🎯 対象: {date_str} {place} {rn}")

    weights = normalize_weights({
        "能力": float(w_ability),
        "適性": float(w_fit),
        "展開": float(w_pace),
        "調子": float(w_form),
        "確度": float(w_cert),
    })

    # 経過時間表示（ログの代わり）
    time_box = st.empty()
    start = time.perf_counter()

    def update_time():
        elapsed = time.perf_counter() - start
        time_box.info(f"⏱ 経過時間: {elapsed:.1f} 秒")

    update_time()

    try:
        with st.spinner("AIが情報収集・分析しています…"):
            df, sources = run_prediction(
                query=query,
                weights=weights,
                force_refresh=force_refresh,
                update_time_cb=update_time,
            )

        elapsed = time.perf_counter() - start
        time_box.success(f"✅ 完了！ かかった時間: {elapsed:.1f} 秒")

        st.success("予想を作成しました！")

        st.markdown("### 📊 有力馬（5〜6頭）")
        show_cols = [
            "馬名", "評価", "総合点", "人気想定",
            "能力", "適性", "展開", "調子", "不確実性",
            "思考による分析結果", "懸念点"
        ]
        st.dataframe(df[show_cols], use_container_width=True)

        st.markdown("---")
        st.markdown("### 💰 買い目（参考）")
        bets_df = make_bets(df_ranked=df, ticket_type=ticket_type, budget_yen=int(budget_yen))
        if bets_df.empty:
            st.info("買い目は未作成です（予算0円 or データ不足）。")
        else:
            st.dataframe(bets_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔎 参考ソース（取得できた場合）")
        if sources:
            seen = set()
            uniq = []
            for s in sources:
                u = s.get("uri")
                if u and u not in seen:
                    seen.add(u)
                    uniq.append(s)
            for s in uniq[:10]:
                st.write(f"- {s.get('title','')}: {s.get('uri','')}")
        else:
            st.caption("ソースが取得できない場合もあります。")

    except genai_errors.ClientError as e:
        elapsed = time.perf_counter() - start
        time_box.error(f"✖ 失敗（{elapsed:.1f} 秒）")

        if is_quota_zero_message(e):
            st.error("Gemini API の利用枠が 0 の可能性があります（Billing/プラン設定）。Google側の使用状況を確認してください。")
        else:
            wait = parse_retry_seconds(e)
            if wait is not None:
                st.error(f"利用制限に達しました。{int(wait)}秒ほど待ってから再実行してください。")
            else:
                st.error(f"APIエラー: {e}")

        st.exception(e)

    except Exception as e:
        elapsed = time.perf_counter() - start
        time_box.error(f"✖ 失敗（{elapsed:.1f} 秒）")
        st.error("予想の作成に失敗しました。時間をおいて再実行してください。")
        st.exception(e)

else:
    st.info("左の入力を埋めて「AI予想を作成」を押してください。")
