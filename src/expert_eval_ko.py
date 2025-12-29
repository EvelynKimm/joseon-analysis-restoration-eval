import os
import re
import time
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

RESTORED_CSV = "src/expert_eval_candidates.csv"

# ✅ 이전 응답(annotator별 선택값) CSV
PREV_LOG_CSV = "src/joseon_expert_eval_post.csv"

DN_PATTERN = re.compile(r"\[D\d+\]")

LOG_HEADER = [
    "timestamp",
    "annotator",
    "data_id",
    "q1_selected_labels",
    "q1_selected_indices",
    "q1_selected_model_rank",
    "q1_no_answer",
    "q1_comment",
    "system_rank_1_label",
    "system_rank_1_model",
    "global_comment",
]


@st.cache_resource
def _get_sheets_service():
    if "GCP_SERVICE_ACCOUNT" not in st.secrets:
        raise RuntimeError("Streamlit Secrets에 GCP_SERVICE_ACCOUNT가 없습니다.")

    sa_info = dict(st.secrets["GCP_SERVICE_ACCOUNT"])

    if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
        sa_info["private_key"] = sa_info["private_key"].replace("\r\n", "\n")

    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def append_log_row_to_sheet(row: dict) -> None:
    """
    Google Spreadsheet에 row 한 줄을 append 한다.
    - 헤더가 없으면 1행에 헤더를 먼저 쓴다.
    - 429 등 일시 오류는 백오프 재시도한다.
    """
    if "SHEET_ID" not in st.secrets:
        raise RuntimeError("Streamlit Secrets에 SHEET_ID가 없습니다.")
    sheet_id = str(st.secrets["SHEET_ID"]).strip()
    tab = str(st.secrets.get("SHEET_TAB", "log")).strip()

    service = _get_sheets_service()

    header_range = f"{tab}!A1:O1"
    existing = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=sheet_id, range=header_range)
        .execute()
        .get("values", [])
    )
    if not existing:
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=header_range,
            valueInputOption="RAW",
            body={"values": [LOG_HEADER]},
        ).execute()

    values = [str(row.get(k, "")) for k in LOG_HEADER]
    last_err = None
    for attempt in range(5):
        try:
            (
                service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=sheet_id,
                    range=f"{tab}!A:O",
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": [values]},
                )
                .execute()
            )
            return
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Google Sheet 저장 실패: {last_err}")


def format_restored_sentence(masked_document: str, restored_sentence: str) -> str:
    """
    masked_document에서 [Dn] 위치를 기준으로,
    restored_sentence에서 해당 위치에 들어간 복원된 토큰을 <span class="restored-token">로 감싸서
    HTML 문자열로 반환한다.
    """
    if not masked_document or not restored_sentence:
        return escape(str(restored_sentence))

    segments = DN_PATTERN.split(masked_document)
    tokens = DN_PATTERN.findall(masked_document)
    num_tokens = len(tokens)

    result_html = ""
    r_pos = 0

    for i, seg in enumerate(segments):
        if seg:
            idx = restored_sentence.find(seg, r_pos)
            if idx == -1:
                if r_pos < len(restored_sentence):
                    span = restored_sentence[r_pos:]
                    result_html += f'<span class="restored-token">{escape(span)}</span>'
                return result_html

            if i > 0 and idx > r_pos:
                span = restored_sentence[r_pos:idx]
                result_html += f'<span class="restored-token">{escape(span)}</span>'
                num_tokens = max(num_tokens - 1, 0)

            result_html += escape(seg)
            r_pos = idx + len(seg)
        else:
            continue

    if r_pos < len(restored_sentence):
        tail = restored_sentence[r_pos:]
        if num_tokens > 0:
            result_html += f'<span class="restored-token">{escape(tail)}</span>'
        else:
            result_html += escape(tail)

    return result_html


@st.cache_data
def load_restored_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["data_id", "model", "candidate_rank"]).reset_index(drop=True)
    return df


@st.cache_data
def load_prev_log(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)

    # 최소 컬럼 검사
    if "annotator" not in df.columns or "data_id" not in df.columns:
        return pd.DataFrame()

    # timestamp 있으면 timestamp 기준 마지막 것을 유지
    if "timestamp" in df.columns:
        df = df.sort_values(["annotator", "data_id", "timestamp"])
    else:
        df = df.sort_values(["annotator", "data_id"])

    df = df.drop_duplicates(subset=["annotator", "data_id"], keep="last").reset_index(
        drop=True
    )
    return df


def _norm_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _parse_int_list(val) -> list[int]:
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return out


def render_final_page():
    annotator = st.session_state.get("annotator_name", "")

    st.title("평가 종료")
    st.markdown(
        """
        참여해 주셔서 감사합니다.  
        평가 전반에 대한 총평이나 인상, 시스템에 대한 총평을 자유롭게 남겨주세요.
        
        총평 내용 예시
        - 전체적인 난이도 
        - 각 시스템(시스템 1, 2, 3)에 대한 전반적인 인상  
        - 실제 복원 연구·업무에서 이러한 도구를 사용할 때 기대되는 장점  
        - 시스템 측면에서의 개선점 제안 등  
        """
    )

    st.subheader("총평 작성 필수")
    global_comment = st.text_area(
        "평가 전반에 대한 인상 및 개선 제안을 작성해 주세요.",
        key="global_comment_final",
        height=200,
    )

    if st.button("설문 제출"):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "annotator": annotator,
            "data_id": "__GLOBAL__",
            "q1_selected_labels": "",
            "q1_selected_indices": "",
            "q1_selected_model_rank": "",
            "q1_no_answer": "",
            "q1_comment": "",
            "system_rank_1_label": "",
            "system_rank_1_model": "",
            "global_comment": global_comment,
        }

        try:
            append_log_row_to_sheet(row)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.success("응답이 제출되었습니다. 감사합니다.")


def scroll_to_top() -> None:
    st.components.v1.html(
        """
        <script>
        (function() {
          try {
            const doc = window.parent.document;
            const topEl = doc.getElementById("page_top");
            if (topEl) {
              topEl.scrollIntoView({behavior: "auto", block: "start"});
            } else {
              window.parent.scrollTo(0, 0);
            }
          } catch (e) {
            window.scrollTo(0, 0);
          }
        })();
        </script>
        """,
        height=0,
    )


def _nan_to_empty(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    if s.strip().lower() == "nan":
        return ""
    return s


def main():
    st.set_page_config(page_title="고문서 복원 결과 전문가 평가", layout="wide")
    st.markdown('<div id="page_top"></div>', unsafe_allow_html=True)

    st.markdown(
        """
    <style>
    .sticky-masked-wrapper {
        position: fixed;
        top: 3.5rem;
        left: 0;
        right: 0;
        z-index: 100;
        background-color: #ffffff;
        border-bottom: 1px solid #d0d0d0;
    }
    .sticky-inner {
        max-width: 70rem;
        margin: 0 auto;
        padding: 0.75rem 1.25rem 1rem 1.25rem;
    }
    .masked-box {
        padding: 0.9rem 1.1rem;
        border-radius: 0.7rem;
        border: 1px solid #999999;
        background-color: #f0f0f0;
        font-size: 1.55rem;
        line-height: 2.15;
    }

    .restored-token {
        color: #d9534f;
        font-weight: 900;
    }

    .option-label {
        font-weight: 900;
        margin-bottom: 0.55rem;
        font-size: 1.25rem;
    }
    .option-sentence {
        font-size: 1.45rem;
        line-height: 2.05;
    }

    /* Q1 카드 */
    .q1-card {
        padding: 1.25rem 1.35rem;
        border-radius: 0.95rem;
        border: 2px solid #d0d0d0;
        background-color: #f5f5f5;
        margin-bottom: 0.4rem;
        min-height: 7.2rem;
        transition: 0.12s ease-in-out;
    }
    .q1-card.selected {
        border-color: #2f80ed;
        background-color: #eaf3ff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    }

    .q1-toggle-wrapper button {
        width: 100%;
        border-radius: 0.6rem;
        background-color: #e0e0e0;
        color: #333333;
        border: 1px solid #c0c0c0;
        font-size: 0.95rem;
        padding: 0.25rem 0.5rem;
        margin-top: 0.25rem;
    }
    .q1-toggle-wrapper button:hover {
        background-color: #d5d5d5;
        border-color: #b0b0b0;
    }

    /* Q2 카드 */
    .model-label {
        font-size: 1.45rem;
        font-weight: 900;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .model-check {
        width: 1.25rem;
        height: 1.25rem;
        border: 2px solid #7a7a7a;
        border-radius: 0.25rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
        line-height: 1;
        background: #ffffff;
    }
    .model-check.checked {
        border-color: #2f80ed;
        background: #eaf3ff;
        font-weight: 900;
    }

    .q2-card {
        padding: 1.10rem 1.15rem;
        border-radius: 0.95rem;
        border: 2px solid #d0d0d0;
        background-color: #f5f5f5;
        margin-bottom: 0.4rem;
        transition: 0.12s ease-in-out;
    }
    .q2-card.selected {
        border-color: #2f80ed;
        background-color: #eaf3ff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.10);
    }

    .q2-toggle-wrapper button {
        width: 100%;
        border-radius: 0.6rem;
        background-color: #e0e0e0;
        color: #333333;
        border: 1px solid #c0c0c0;
        font-size: 0.95rem;
        padding: 0.25rem 0.5rem;
        margin-top: 0.25rem;
    }
    .q2-toggle-wrapper button:hover {
        background-color: #d5d5d5;
        border-color: #b0b0b0;
    }

    .model-sentence {
        font-size: 1.45rem;
        line-height: 2.05;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if "annotator_name" not in st.session_state:
        st.session_state["annotator_name"] = ""
    if "intro_done" not in st.session_state:
        st.session_state["intro_done"] = False
    if "data_idx" not in st.session_state:
        st.session_state["data_idx"] = 0
    if "finished" not in st.session_state:
        st.session_state["finished"] = False
    if "need_scroll_top" not in st.session_state:
        st.session_state["need_scroll_top"] = False

    # 소개 페이지
    if not st.session_state["intro_done"]:
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.title("고문서 복원 결과 전문가 평가")
            st.markdown(
                """
#### 1. 평가의 개요 및 목적

이 평가는 실제로 훼손된 『조선왕조실록』, 『승정원일기』 원문을 여러 시스템이 복원한 문장으로 재구성했을 때, 전문가의 시각에서 **얼마나 역사적으로 타당하고 자연스러운 복원인가**를 정성적으로 평가하기 위한 것입니다.

이 평가는 **어떤 시스템이 조선 시대 문서 복원에 더 적합한지를 파악하고, 향후 고문서 복원 시스템이 실제 복원 전문가가 활용하는 도구로 사용될 때의 개선 방향을 설정하는 것**을 최종 목표로 합니다.

##### 좋은 복원의 기준

평가 시에는 다음과 같은 기준을 종합적으로 고려해 주시면 감사하겠습니다.
 
- 해당 연대·사건·왕대에 실제로 가능한 인물·관직·지명·사건 구성이며, 이미 알려진 사실과 명백히 충돌하지 않는지  
- 인명, 지명 등 각 손상 위치에 자연스럽게 어울리도록 복원되었는지
- 같은 위치의 다른 후보와 비교했을 때, 정보 왜곡이 적고, 역사적·언어적 측면에서 더 설득력 있는지  

Q1, Q2는 이러한 기준을 바탕으로, 개별 문장 수준과 시스템 수준에서 각각 복원 품질을 평가하기 위한 문항입니다.

---

#### 2. 문항 설명

전체 문항은 Q1과 Q2, 두 가지로 구성되어 있으며, **반드시 Q1 → Q2 순서로 평가**해 주시기를 부탁드립니다.  

##### Q1

- 제시된 각 보기는, 동일한 훼손 문장을 서로 다른 시스템이 복원한 결과입니다.  
- 각 보기를 서로 독립적인 문장으로 보시고, 앞에서 제시한 **좋은 복원의 기준**에 따라 정답이라고 볼 수 있는 보기를 **복수 선택**해 주시면 됩니다.  
- 정답이라고 판단되는 후보가 전혀 없다고 보실 경우에는, ‘정답이 없다’ 옵션을 선택해 주시면 됩니다.

##### Q2

- 두 번째 문항에서는 개별 문장 단위가 아니라, **각 시스템이 제시한 전체 후보 집합**을 단위로 평가합니다.  
- 화면에는 실제 시스템 이름이 노출되지 않고, 블라인드 처리된 상태로 표시됩니다.  
- Q1에서 이미 접하신 문장들을, 이번에는 "이 시스템이 전반적으로 얼마나 합리적인 복원을 많이 제시하는가"라는 관점에서 보신다고 생각해 주시면 됩니다.  
- 전반적인 복원 품질이 가장 우수하다고 판단되는 시스템을 선택해주시면 됩니다.

---

#### 3. 코멘트 작성 안내

##### 각 문제별 코멘트
  각 문항마다 다음과 같은 내용을 자유롭게 남겨 주실 수 있습니다.  
  - 고유명사 복원과 관련하여 인상적인 사례  
  - 역사적·문체적으로 부자연스럽거나 명백히 잘못된 복원에 대한 설명  
  - 특히 좋다고 느낀 보기, 특히 문제가 있다고 판단하신 보기에 대한 의견 등  

  각 문제별 코멘트는 선택 사항이지만, 모델 개선과 복원 전략 설계에 매우 중요한 정성 자료로 활용되므로 가능하다면 적극적으로 작성해 주시면 감사하겠습니다.

##### 전체 평가 종료 후 코멘트 (최종 페이지)
  모든 문항 평가를 마치면, 전체 평가를 마무리하는 코멘트 페이지가 한 번 더 나타납니다.  

  이 최종 코멘트에는 다음과 같은 내용을 포함해 주시면 좋습니다.  
    - 전체적인 난이도 
    - 각 시스템(시스템 1, 2, 3)에 대한 전반적인 인상  
    - 실제 복원 연구·업무에서 이러한 도구를 사용할 때 기대되는 장점  
    - 시스템 측면에서의 개선점 제안 등  

  이 최종 코멘트는 **필수 작성**을 부탁드립니다.  
  연구 방향 설정 및 시스템 개선 시 핵심 참고 자료로 활용될 예정입니다.

---

#### 4. 평가 기간 안내

본 전문가 평가는 다음 기간 동안 진행됩니다.  
##### 25년 12월 21일 18시 00분 ~ 25년 12월 27일 23시 59분 
해당 기간 내에 모든 문항(Q1, Q2)과 최종 코멘트 작성을 완료해 주시기를 부탁드립니다.

본 평가는 문항별로 응답이 저장됩니다. 평가를 잠시 중단하셨다가 다시 진행하실 경우, 왼쪽 사이드 바에서 마지막으로 평가하신 문항의 data_id를 확인하신 뒤 해당 항목부터 이어서 진행해주시면 됩니다.

평가 중 문제가 발생하면 아래 연락처로 문의해 주시기 바랍니다.
##### 010-5024-9304                 
"""
            )

            st.markdown("---")
            name_input = st.text_input(
                "평가자 이름을 입력해 주세요.",
                key="annotator_input",
                value=st.session_state["annotator_name"],
            )
            st.markdown("- 입력하신 이름은 로그 파일에만 저장됩니다.")
            if st.button("평가 시작하기"):
                if not name_input.strip():
                    st.error("이름을 입력해 주세요.")
                    st.stop()
                st.session_state["annotator_name"] = name_input.strip()
                st.session_state["intro_done"] = True
                st.rerun()
        return

    # 최종 페이지
    if st.session_state.get("finished", False):
        render_final_page()
        return

    annotator = st.session_state["annotator_name"]
    st.title("복원 문장 평가 도구")
    if st.session_state.get("need_scroll_top", False):
        scroll_to_top()
        st.session_state["need_scroll_top"] = False

    if not os.path.exists(RESTORED_CSV):
        st.error(f"복원 결과 CSV 파일을 찾을 수 없습니다: {RESTORED_CSV}")
        st.stop()

    df = load_restored_csv(RESTORED_CSV)
    if df.empty:
        st.error("CSV 파일에 데이터가 없습니다.")
        st.stop()

    data_ids = sorted(df["data_id"].unique().tolist())
    if st.session_state["data_idx"] >= len(data_ids):
        st.session_state["data_idx"] = len(data_ids) - 1

    st.sidebar.header("평가자")
    st.sidebar.markdown(f"이름: {annotator}")

    st.sidebar.subheader("문제 선택")
    selected_data_id = st.sidebar.selectbox(
        "data_id 선택",
        options=data_ids,
        index=st.session_state["data_idx"],
    )
    current_idx = data_ids.index(selected_data_id)
    st.session_state["data_idx"] = current_idx
    current_data_id = data_ids[current_idx]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"현재 항목: `{current_data_id}`")
    st.sidebar.markdown(f"진행 상황: {current_idx + 1} / {len(data_ids)}")
    st.sidebar.markdown("---")

    cur_df = df[df["data_id"] == current_data_id].copy()
    cur_df = cur_df.sort_values(["model", "candidate_rank"]).reset_index(drop=True)
    option_rows = cur_df.to_dict("records")
    models = sorted(cur_df["model"].unique().tolist())

    if len(option_rows) == 0:
        st.error("해당 data_id에 대한 후보 문장이 없습니다.")
        st.stop()

    masked_document = ""
    if "masked_document" in cur_df.columns and len(cur_df) > 0:
        masked_document = str(cur_df["masked_document"].iloc[0])

    king_val = ""
    if "king" in cur_df.columns and len(cur_df) > 0:
        king_val = str(cur_df["king"].iloc[0] or "").strip()

    def _format_source_line(data_id: str, king: str) -> str:
        try:
            prefix, rest = data_id.split("_", 1)
        except ValueError:
            prefix, rest = "", data_id

        p = prefix.lower().strip()
        if p == "jrs":
            corpus = "승정원일기"
        elif p == "ajd":
            corpus = "조선왕조실록"
        else:
            corpus = prefix if prefix else "출처 미상"

        date_part = rest.split("_", 1)[0]
        parts = date_part.split("-")
        if len(parts) == 3 and all(x.isdigit() for x in parts):
            y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
            month_name = datetime(y, m, d).strftime("%B")
            if king:
                return f"{corpus}, {d} {month_name} {y}, {king} era"
            return f"{corpus}, {d} {month_name} {y}"
        return f"{corpus}, {king} era" if king else corpus

    source_line = _format_source_line(current_data_id, king_val)

    if masked_document:
        st.markdown(
            f"""
            <div class="sticky-masked-wrapper">
                <div class="sticky-inner">
                    <h3>손상 문장</h3>
                    <div style="margin: 0.35rem 0 0.65rem 0; font-size:1.25rem; font-weight:800; color:#333333;">
                    {escape(source_line)}
                    </div>
                    <div class="masked-box">
                        {escape(masked_document)}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height: 9.0rem;"></div>', unsafe_allow_html=True)

    # Q2 라벨(블라인드)
    anon_labels = [f"시스템 {i + 1}" for i in range(len(models))]
    real2anon = {m: anon for m, anon in zip(models, anon_labels)}
    anon2real = {anon: m for m, anon in real2anon.items()}

    prev_df = load_prev_log(PREV_LOG_CSV)
    prev_row = {}
    if not prev_df.empty:
        a_df = prev_df[
            prev_df["annotator"].astype(str).str.strip() == str(annotator).strip()
        ]
        if not a_df.empty:
            hit = a_df[a_df["data_id"].astype(str) == str(current_data_id)]
            if not hit.empty:
                prev_row = hit.iloc[0].to_dict()

    prefill_key = f"prefilled_{annotator}_{current_data_id}"
    if prev_row and not st.session_state.get(prefill_key, False):
        # Q1 선택 indices
        prev_indices = set(_parse_int_list(prev_row.get("q1_selected_indices", "")))
        for i in range(len(option_rows)):
            sel_key = f"q1_option_{current_data_id}_{i}"
            st.session_state[sel_key] = (i + 1) in prev_indices

        # Q1 정답 없음
        st.session_state[f"q1_no_answer_{current_data_id}"] = _norm_bool(
            prev_row.get("q1_no_answer", False)
        )

        # Q1 코멘트
        st.session_state[f"q1_comment_{current_data_id}"] = _nan_to_empty(
            prev_row.get("q1_comment", "")
        )

        # Q2 best system
        best_key = f"q2_best_model_{current_data_id}"
        best_label_saved = str(prev_row.get("system_rank_1_label", "") or "").strip()
        best_model_saved = str(prev_row.get("system_rank_1_model", "") or "").strip()

        if best_label_saved.startswith("시스템"):
            st.session_state[best_key] = best_label_saved
        elif best_model_saved in real2anon:
            st.session_state[best_key] = real2anon[best_model_saved]
        else:
            st.session_state[best_key] = ""

        st.session_state[prefill_key] = True

    # Q1
    st.subheader("문항 1")
    st.markdown(
        "##### 각 보기 아래 버튼을 사용해 정답 후보를 선택하세요. 복수 선택 가능합니다."
    )
    st.markdown(
        "- 제시된 각 보기는 위 훼손 문장을 서로 다른 시스템이 복원한 결과입니다."
    )
    st.markdown(
        "- 전체 보기 중 앞에서 제시한 좋은 복원의 기준에 따라 정답이라고 볼 수 있는 보기를 복수 선택해주세요."
    )
    st.markdown('- 정답이라고 판단되는 보기가 없다면, "정답 없음"을 선택해 주세요.')

    cols = st.columns(3)
    for i, row in enumerate(option_rows):
        col = cols[i % 3]
        with col:
            sel_key = f"q1_option_{current_data_id}_{i}"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = False

            selected = bool(st.session_state.get(sel_key, False))
            selected_cls = "selected" if selected else ""

            label = f"보기 {i + 1}"
            html_sentence = format_restored_sentence(
                masked_document=masked_document,
                restored_sentence=str(row["restored_sentence"]),
            )

            st.markdown(
                f"""
                <div class="q1-card {selected_cls}">
                    <div class="option-label">{escape(label)}</div>
                    <div class="option-sentence">{html_sentence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="q1-toggle-wrapper">', unsafe_allow_html=True)
            if st.button(
                "위 보기 선택 또는 해제",
                key=f"q1_btn_{current_data_id}_{i}",
                use_container_width=True,
            ):
                st.session_state[sel_key] = not selected
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:1px solid #d0d0d0; margin: 0.6rem 0 1.0rem 0;">',
        unsafe_allow_html=True,
    )
    no_answer_val = st.checkbox(
        "정답 없음",
        key=f"q1_no_answer_{current_data_id}",
    )

    # Q2
    st.markdown("---")
    st.subheader("문항 2")
    st.markdown(
        "##### 시스템 1, 2, 3 중 전반적으로 가장 좋은 시스템 하나를 선택하세요."
    )
    st.markdown(
        "- 전반적으로 가장 합리적이고 자연스러운 후보를 제시한 시스템 하나를 선택해 주세요."
    )

    best_key = f"q2_best_model_{current_data_id}"
    if best_key not in st.session_state:
        st.session_state[best_key] = ""

    best_selected = st.session_state.get(best_key, "")

    cols_models = st.columns(len(models)) if len(models) <= 3 else st.columns(3)
    for idx_m, m in enumerate(models):
        col = cols_models[idx_m % len(cols_models)]
        with col:
            anon = real2anon[m]
            is_best = best_selected == anon
            check_cls = "checked" if is_best else ""
            mark = "✓" if is_best else ""
            selected_cls = "selected" if is_best else ""

            m_df = cur_df[cur_df["model"] == m].sort_values("candidate_rank")
            sentences_html = ""
            for _, r in m_df.iterrows():
                html_sentence = format_restored_sentence(
                    masked_document=masked_document,
                    restored_sentence=str(r["restored_sentence"]),
                )
                sentences_html += f'<div class="model-sentence">- {html_sentence}</div>'

            st.markdown(
                f"""
                <div class="q2-card {selected_cls}">
                    <div class="model-label">
                        <span class="model-check {check_cls}">{mark}</span>
                        {escape(anon)}
                    </div>
                    {sentences_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="q2-toggle-wrapper">', unsafe_allow_html=True)
            if st.button(
                "위 시스템 선택 또는 해제",
                key=f"q2_btn_{current_data_id}_{anon}",
                use_container_width=True,
            ):
                if st.session_state.get(best_key, "") == anon:
                    st.session_state[best_key] = ""
                else:
                    st.session_state[best_key] = anon
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # 코멘트
    st.markdown(
        '<hr style="border:1px solid #b0b0b0; margin: 1.2rem 0 1.0rem 0;">',
        unsafe_allow_html=True,
    )
    q1_comment = st.text_area(
        "이 항목에 대한 코멘트(선택)",
        key=f"q1_comment_{current_data_id}",
        height=120,
    )

    st.markdown("---")

    # 좌우 여백을 크게 두고, 가운데 영역에 버튼 2개 배치
    outer_l, center_area, outer_r = st.columns([2, 3, 2])
    with center_area:
        btn_l, gap, btn_r = st.columns([1, 0.2, 1])

        with btn_l:
            if st.button("이전 항목으로 이동", use_container_width=True):
                if current_idx > 0:
                    st.session_state["data_idx"] = current_idx - 1
                    st.session_state["need_scroll_top"] = True
                    st.rerun()
                else:
                    st.info("첫 번째 항목입니다.")

        with btn_r:
            if st.button("응답 저장 후 다음 항목으로 이동", use_container_width=True):
                if not annotator.strip():
                    st.error("먼저 이름을 입력해 주세요.")
                    st.stop()

                selected_labels = []
                selected_indices = []
                selected_model_ranks = []

                for i, row in enumerate(option_rows):
                    sel_key = f"q1_option_{current_data_id}_{i}"
                    if bool(st.session_state.get(sel_key, False)):
                        selected_labels.append(f"보기 {i + 1}")
                        selected_indices.append(i + 1)
                        selected_model_ranks.append(
                            f"{row['model']}:{row['candidate_rank']}"
                        )

                if (not selected_labels) and (not bool(no_answer_val)):
                    st.error('최소 1개 보기를 선택하거나 "정답 없음"을 선택해 주세요.')
                    st.stop()

                best_label = st.session_state.get(best_key, "")
                if not best_label:
                    st.error("문항 2에서 가장 좋은 시스템 1개를 선택해 주세요.")
                    st.stop()

                best_real = anon2real.get(best_label, "")

                log_row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "annotator": annotator,
                    "data_id": current_data_id,
                    "q1_selected_labels": ";".join(selected_labels),
                    "q1_selected_indices": ",".join(str(i) for i in selected_indices),
                    "q1_selected_model_rank": ";".join(selected_model_ranks),
                    "q1_no_answer": bool(no_answer_val),
                    "q1_comment": q1_comment,
                    "system_rank_1_label": best_label,
                    "system_rank_1_model": best_real,
                    "global_comment": "",
                }

                try:
                    append_log_row_to_sheet(log_row)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                if current_idx < len(data_ids) - 1:
                    st.session_state["data_idx"] = current_idx + 1
                    st.session_state["need_scroll_top"] = True
                    st.rerun()
                else:
                    st.success(
                        "마지막 항목이 저장되었습니다. 최종 코멘트 페이지로 이동합니다."
                    )
                    st.session_state["finished"] = True
                    st.session_state["need_scroll_top"] = True
                    st.rerun()


if __name__ == "__main__":
    main()
