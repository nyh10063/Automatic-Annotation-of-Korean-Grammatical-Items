# 한국어 문법 항목 자동 주석 공개 실행 코드

이 저장소는 논문에 나온 A/B 추론 파이프라인을 심사위원이 직접 실행해 볼 수 있도록 정리한 최소 공개 버전입니다.

사용 방법은 단순합니다. 한국어 문장을 CSV 파일에 넣고, 실행 스크립트를 돌린 뒤, `predictions.csv` 파일을 확인하면 됩니다.

## 포함된 기능

- A 파이프라인 추론: 다면(가정/조건 제시), ㄴ/은 적 있/없(경험 유무 서술)
- B 파이프라인 추론: ㄴ/은/는데1(상황/배경 제시), ㄴ/은/는데2(대립/대조), 고 말1(안타까움), 고 말2(의지), 까지1(범위의 끝), 까지2(더함), 까지3(지나침)
- `sentence` 한 열만 사용하는 간단한 CSV 입력
- 심사위원이 보기 쉬운 CSV 결과 파일

학습 데이터, 내부 주석 파일, 전체 실험 로그는 이 저장소에 포함하지 않았습니다.

## 입력 파일 형식

입력 CSV에는 `sentence` 열 하나만 있으면 됩니다.

```csv
sentence
"나도 언니처럼 예쁘다면 참 좋을 텐데."
"저는 제주도에 간 적이 있습니다."
```

`id` 열은 넣지 않아도 됩니다. 실행 코드가 결과 파일에 `1, 2, 3, ...` 순서로 자동 번호를 붙입니다.

문장 안에 쉼표가 있으면 문장 전체를 큰따옴표로 감싸는 것이 가장 안전합니다. 큰따옴표가 없어도 가능한 한 자동 복구하도록 만들었지만, 표준 CSV 형식은 큰따옴표를 쓰는 방식입니다.

## 결과 파일

실행 결과는 `outputs/` 폴더에 저장됩니다.

심사위원이 주로 보면 되는 파일은 아래 두 파일입니다.

```text
outputs/a_run/predictions.csv
outputs/b_run/predictions.csv
```

추가 확인용 파일도 함께 생성됩니다.

```text
predictions.jsonl
debug_detection.jsonl
summary.json
```

일반적으로는 `predictions.csv`만 확인하면 됩니다.

## 모델 체크포인트

A 파이프라인 모델은 Hugging Face에 공개되어 있습니다.

```text
nyh1006/kmwe-a-pipeline-encoder
```

내려받은 뒤에는 아래 구조가 되도록 두면 됩니다.

```text
checkpoints/a_best/
├─ encoder/
├─ tokenizer/
└─ head.pt
```

B 파이프라인 모델은 Qwen 기반 LLM 체크포인트입니다. Hugging Face에서 내려받거나, 로컬/Google Drive 경로를 직접 지정해서 사용할 수 있습니다.

B 실행 시에는 아래 두 경로를 지정해야 합니다.

```text
B_MODEL_DIR
B_TOKENIZER_DIR
```

## 빠른 실행 방법

먼저 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

A 파이프라인 실행:

```bash
bash scripts/run_a_infer.sh \
  reviewer_inputs/a_input.csv \
  outputs/a_run \
  checkpoints/a_best \
  /path/to/expredict.xlsx
```

B 파이프라인 실행:

```bash
bash scripts/run_b_infer.sh \
  reviewer_inputs/b_input.csv \
  outputs/b_run \
  /path/to/b_model \
  /path/to/b_tokenizer \
  /path/to/expredict.xlsx
```

실행 후에는 아래 파일을 확인하면 됩니다.

```text
outputs/a_run/predictions.csv
outputs/b_run/predictions.csv
```

## Colab에서 실행하는 경우

Colab에서는 Google Drive를 마운트한 뒤, 이 저장소를 불러오고, 필요한 패키지를 설치한 다음 위 실행 명령을 사용하면 됩니다.

A 모델을 Hugging Face에서 내려받는 예시는 아래와 같습니다.

```bash
hf download nyh1006/kmwe-a-pipeline-encoder \
  --local-dir checkpoints/a_best
```

그 다음에는 위의 A/B 실행 명령을 그대로 사용하면 됩니다.

## 폴더 구성

```text
kmwe/               공개 실행용 핵심 코드
scripts/            A/B 실행 스크립트
reviewer_inputs/    심사위원 입력 CSV 예시
examples/           입력 예시와 예상 출력 예시
docs/               입력 형식과 재현 안내 문서
outputs/            실행 결과 저장 위치
debug_detection.jsonl 등은 상세 확인용 파일
checkpoints/        모델 체크포인트를 둘 위치
```

## 라이선스와 사용 범위

코드 라이선스와 모델 사용 조건은 별도로 정리합니다.

공개 체크포인트는 기반 모델의 라이선스 조건을 함께 따릅니다. 특히 B 파이프라인 LLM은 Qwen/Qwen3-8B 기반 모델을 사용하므로, 사용자는 해당 기반 모델의 라이선스 조건도 함께 준수해야 합니다.

이 저장소는 논문 검증, 연구, 교육 목적의 실행을 돕기 위해 공개합니다.
