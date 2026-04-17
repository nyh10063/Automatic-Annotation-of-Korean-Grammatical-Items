# <언어 모델과 규칙을 활용한 한국어 문법 항목 자동 주석 연구> 공개 실행 코드

이 저장소는 논문에 나온 A/B 추론 파이프라인을 심사위원 선생님께서 직접 실행해 볼 수 있도록 정리한 최소 공개 버전입니다.
사용 방법은 단순합니다. 주석을 달고자 하는 한국어 문장을 CSV 파일에 넣고, 실행 스크립트를 돌린 뒤, `predictions.csv` 파일을 확인하면 됩니다.
학습 데이터, 내부 주석 파일, 전체 실험 로그는 이 저장소에 포함하지 않았습니다.

## 이 깃 허브 저장소의 모든 폴더와 파일 구조를 코랩으로 다운하는 방법

선생님께서 쓰시는 코랩에서 마운트를 먼저 하시고 아래 명령어를 코랩에서 실행하시면 이 깃의 모든 폴더와 파일들이 선생님의 구글드라이브의 "/kmwe" 폴더로 복사됩니다. 

```text
%cd /content/drive/MyDrive
!rm -rf kmwe
!git clone https://github.com/nyh10063/Automatic-Annotation-of-Korean-Grammatical-Items.git kmwe

%cd /content/drive/MyDrive/kmwe
!git branch --show-current
!ls
!pip -q install -r requirements.txt
```

그 후에 선생님께서 주석을 하고 싶은 문장들을 아래와 같이 입력하시면 되십니다.

## 주석 대상 문장 입력 방법

A그룹 문법 항목의 주석을 달고자 하는 경우 문장을 이 파일에 넣으시면 됩니다: reviewer_inputs/a_input.csv
A그룹 문법 항목: 다면(가정/조건 제시), ㄴ/은 적 있/없(경험 유무 서술)

B그룹 문법 항목의 주석을 달고자 하는 경우 문장을 이 파일에 넣으시면 됩니다: reviewer_inputs/b_input.csv
B그룹 문법 항목: ㄴ/은/는데1(상황/배경 제시), ㄴ/은/는데2(대립/대조), 고 말1(안타까움), 고 말2(의지), 까지1(범위의 끝), 까지2(더함), 까지3(지나침)

위 csv파일의 `sentence` 열에 주석을 달고자 하는 문장을 아래로 나열합니다.
참고로 현재 선생님께서 주석기를 바로 실행할 수 있도록 예문이 들어가 있습니다. 
선생님께서 따로 실험을 해보기를 원하시면 예문을 지우고 원하는 문장을 넣으시면 되십니다.

```csv
(예시)
sentence
나도 언니처럼 예쁘다면 참 좋을 텐데.
저는 제주도에 간 적이 있습니다.
```

이후 추론 라인A 인코더 모델이나 추론 라인B의 디코더 모델을 실행하시려면 아래 명령어를 코랩에서 실행하시면 되십니다.

## 추론 라인A 인코더 모델 실행 명령어 및 주석 결과 파일 위치

코랩에서 구글 드라이브를 마운트 하신 후 아래 명령어를 실행해 주시면 되십니다.
추론 라인A 학습 인코더(도메인 적응 학습+미세 조정 DeBERTa-v3-base-korean) 가중치 파일이 자동으로 선생님의 구글드라이브로 받아진 후 실행됩니다.
코랩의 런타임 종류는 T4를 선택하시면 무리 없이 동작 가능합니다.

```text
%cd /content/drive/MyDrive/kmwe

!bash scripts/run_a_infer.sh \
  reviewer_inputs/a_input.csv \
  outputs/a_run \
  auto
```

주석 결과는 아래 폴더와 파일을 열어 보시면 됩니다.

```text
outputs/a_run/predictions.csv
```

선생님께서 예문들 그대로 실행하셨다면 아래 결과를 보실 수 있으십니다. 
2~4행: ㄴ/은 적 있/없(경험 유무 서술) 주석 결과
5~7행: 다면(가정/조건 제시) 주석 결과
8~10행: 주석 대상 문법 항목 없음


## 추론 라인B 디코더 모델 실행 명령어 및 주석 결과 파일 위치

논문에 기록한 것과 같이 미세 조정된 GPT4.1을 사용한 주석 결과가 가장 좋습니다.
그러나 미세 조정된 GPT4.1의 공개 가능 여부가 법적으로 불확실하여 미세 조정된 Llama 3.1 8b를 사용한 추론 라인을 올려드립니다. 

코랩에서 구글 드라이브를 마운트 하신 후 아래 명령어를 실행해 주시면 되십니다.
추론 라인B 학습 디코더(미세 조정된 Llama 모델)에 필요한 파일이 자동으로 선생님의 구글드라이브로 받아지고 실행됩니다.
코랩 런타임은 12gb의 GPU 메모리를 필요로 하며 따라서 T4로 설정하는 것을 추천드립니다. 
처음 실행할 때는 모델 다운로드(16gb)와 로딩에 시간이 걸릴 수 있습니다.

```text
%cd /content/drive/MyDrive/kmwe
!bash scripts/run_b_infer.sh \
  reviewer_inputs/b_input.csv \
  outputs/b_run \
  auto \
  auto
```

주석 결과는 아래 폴더와 파일을 열어 보시면 됩니다.

```text
outputs/b_run/predictions.csv
```

선생님께서 예문들 그대로 실행하셨다면 아래 결과를 보실 수 있으십니다. 
2~3행: 고 말1(안타까움) 주석 결과
4~5행: 고 말2(의지) 주석 결과
6~7행: 까지1(범위의 끝) 주석 결과
8~9행: 까지2(더함) 주석 결과
10~11행: 까지3(지나침) 주석 결과
12~13행: ㄴ/은/는데1(상황/배경 제시) 주석 결과
14~15행: ㄴ/은/는데2(대립/대조) 주석 결과 
15~16행: 주석 대상 문법 항목 없음. 16행에서 의존 명사 "~는데"를 규칙 라인이 찾지만 디코더가 오탐으로 판단합니다.

실행 결과를 보시면 Llama 3.1 8b 모델이 14행의 판단이 틀린 것을 발견할 수 있습니다.
참고로 미세 조정된 GPT4.1은 모두 정답을 맞혔습니다.


## 모델 체크포인트

추론 라인A,B 모델은 Hugging Face에 공개되어 있습니다.

```text
https://huggingface.co/nyh1006/kmwe-a-pipeline-encoder
```

필요하시면 내려 받으시면 되며내려 아래 구조가 되도록 두면 됩니다.

```text
checkpoints/a_best/
├─ encoder/
├─ tokenizer/
└─ head.pt
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

## 선택 실행: GPT-4.1 사용

기본 B 파이프라인은 공개된 Qwen 기반 체크포인트를 사용합니다.

추가로, OpenAI API key가 있는 경우에는 GPT-4.1 또는 fine-tuned GPT-4.1 모델로도 같은 입력 파일을 실행해 볼 수 있습니다. 이 기능은 선택 사항입니다.

GPT-4.1 실행은 모델 파일을 내려받지 않고 OpenAI API를 호출합니다. 따라서 사용자의 `OPENAI_API_KEY`가 필요하며, API 사용 비용이 발생할 수 있습니다. API key는 절대 GitHub에 올리면 안 됩니다.

입력 CSV 형식과 출력 CSV 형식은 Qwen B 파이프라인과 같습니다.

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4.1"

bash scripts/run_b_openai_infer.sh \
  reviewer_inputs/b_input.csv \
  outputs/b_openai_run \
  /path/to/expredict.xlsx
```

만약 fine-tuned GPT-4.1 모델을 사용하려면 `OPENAI_MODEL`에 해당 fine-tuned model id를 넣으면 됩니다.

```bash
export OPENAI_MODEL="ft:your_fine_tuned_model_id"
```

논문 공개용 기본 재현 경로는 Qwen 기반 runner입니다. GPT-4.1 runner는 같은 규칙 탐지와 같은 프롬프트를 OpenAI API로 실행해 보는 선택 기능입니다.

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
