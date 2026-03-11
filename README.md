# VODAI PoC - 動画解析 CLI ツール

動画ファイルから**字幕生成・誤字補正・シーン要約・感情推定**を一括で行う CLI ツールの Proof of Concept。

## 目的

配信 VOD などの動画コンテンツに対して、以下の解析をパイプラインで実行する:

- **字幕生成**: WhisperX による日本語タイムスタンプ付き文字起こし
- **誤字補正**: LLM でASR誤認識を文脈ベースで補正（差分ログ付き）
- **シーン分割・要約**: PySceneDetect でシーン境界を検出し、代表フレーム + 字幕を LLM で要約
- **感情推定**: 音声次元感情（wav2vec2）+ テキスト感情（LUKE WRIME）+ prosody 特徴量を融合し、感情タイムラインを生成

## 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (パッケージマネージャ)
- ffmpeg (システムにインストール済みであること)
- LLM プロバイダー: [Ollama](https://ollama.com/) (ローカル開発) または OpenAI API

## セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/T4ko0522/speech-processing-poc.git
cd speech-processing-poc

# 依存関係のインストール
uv sync

# 環境変数を設定
cp .env.example .env
# .env を編集して API キーを設定
```

### 環境変数

| 変数名 | 必須 | 説明 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI 使用時 | OpenAI API キー |
| `HF_TOKEN` | WhisperX アライメント時 | HuggingFace トークン（pyannote.audio のゲートモデル用） |

### Ollama を使う場合（デフォルト）

```bash
# Ollama をインストール後、モデルを取得
ollama pull gemma3:4b
```

## 使い方

```bash
# 基本的な使い方
uv run vodai <入力動画ファイル>

# オプション付き
uv run vodai input.mp4 --output-dir ./results --device cuda --config my_config.yaml
```

### CLI オプション

| オプション | 短縮 | デフォルト | 説明 |
|---|---|---|---|
| `--output-dir` | `-o` | `poc/output` | 出力ディレクトリ |
| `--device` | `-d` | `cpu` | 推論デバイス (`cpu` / `cuda`) |
| `--config` | `-c` | `poc/configs/default.yaml` | 設定ファイルパス |

### 対応入力形式

- 動画: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`
- 音声: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `transcript_raw.json` | WhisperX の生文字起こし結果 |
| `transcript_fixed.json` | LLM による誤字補正済み字幕 + 補正差分ログ |
| `output.srt` / `output.vtt` | 字幕ファイル（SRT / WebVTT） |
| `scenes.json` | シーン境界 + LLM 要約 + キーワード |
| `frames/` | シーン代表フレーム画像（PNG） |
| `emotions.json` | 感情タイムライン（融合済み valence/arousal + ラベル） |
| `report.json` | パイプライン実行レポート（各ステップの所要時間・成否） |

## 設定

`poc/configs/default.yaml` で全パラメータを管理する。主要な設定項目:

```yaml
# LLM プロバイダー切り替え
llm:
  provider: ollama          # "ollama" or "openai"

# ASR モデルサイズ（CPU: medium 以下推奨）
asr:
  model_name: medium        # tiny / base / small / medium / large-v3
  compute_type: int8        # int8 / float16 / float32

# 感情融合の重み
emotion:
  fusion:
    text_weight: 0.6        # テキスト感情の重み
    dimensional_weight: 0.2 # 音声次元感情の重み
```

## パイプライン処理フロー

```
入力動画
  │
  ├─ Step 1: 音声抽出 (ffmpeg → 16kHz mono WAV)          [Fatal]
  ├─ Step 2: 文字起こし (WhisperX + word-level alignment) [Fatal]
  ├─ Step 3: 誤字補正 (LLM チャンク単位補正)              [スキップ可]
  ├─ Step 4: シーン検出 (PySceneDetect + フレーム抽出)    [スキップ可]
  ├─ Step 5: シーン要約 (LLM + Vision)                    [スキップ可]
  ├─ Step 6: 感情推定 (次元感情 + テキスト感情 + prosody)  [スキップ可]
  └─ Step 7: 出力書き出し (JSON / SRT / VTT / report)
```

Step 1・2 は失敗時にパイプライン全体を停止する。Step 3〜6 は失敗してもスキップして後続を継続する。

## 技術スタック

| 領域 | 技術 |
|---|---|
| 音声抽出 | ffmpeg |
| 文字起こし | WhisperX |
| 誤字補正・シーン要約 | OpenAI API / Ollama (OpenAI 互換) |
| シーン分割 | PySceneDetect (ContentDetector) |
| 音声感情 | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim |
| テキスト感情 | Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime |
| prosody | librosa (F0, RMS energy, speech rate) |
| データモデル | Pydantic v2 |
| CLI | Click + Rich |

## 開発

```bash
# リンター
uv run ruff check poc/

# フォーマッター
uv run ruff format poc/
```
