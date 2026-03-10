# VODAI向け 動画解析PoC 設計書 v0.1

## 1. 目的
- 動画から字幕を生成し誤字補正する
- 感情の起伏を時系列で推定する
- シーン単位で場面要約する

## 2. PoC範囲
- CLI実行のみ
- 入力はローカル動画ファイル
- 出力は JSON / SRT / VTT
- API受け取りや本番UI連携は対象外

## 3. 採用技術
- 言語: Python 3.12
- 前処理: ffmpeg
- 字幕起こし: WhisperX
- 誤字補正: OpenAI GPT-4.1 API
- 場面要約: OpenAI GPT-4o API
- シーン分割: PySceneDetect
- 感情推定: Speech系モデル + テキスト感情モデルの融合

## 4. 処理フロー
1. `input.mp4` を受け取る
2. ffmpegで音声抽出と正規化
3. WhisperXでタイムスタンプ付き字幕生成
4. GPT-4.1で文脈つき誤字補正
5. PySceneDetectでシーン分割
6. 各シーン代表フレームをGPT-4oで要約
7. 音声特徴と字幕感情を融合して感情タイムライン生成
8. 結果を統合して書き出し

## 5. 想定ディレクトリ
```
poc/
  input/
  output/
  src/
    pipeline/
    asr/
    correction/
    scene/
    emotion/
    io/
  prompts/
  configs/
  scripts/
```

## 6. 出力物
- `transcript_raw.json`
- `transcript_fixed.json`
- `scenes.json`
- `emotions.json`
- `output.srt`
- `output.vtt`
- `report.json`

## 7. 最低限の品質基準
- 字幕補正後の可読性が目視で改善
- 処理失敗率が低いこと
- 1本あたりの処理時間が運用可能範囲
- 低信頼区間を抽出できること

## 8. 運用ルール
- 誤字補正は差分ログを必ず保存
- 低信頼区間のみ人手レビュー
- API失敗時はリトライ後に対象区間を未補正で通す
