# 運用ガイド — 展示・授業・実験室で回すために

このドキュメントは「動かす人」向けです。設計は [ARCHITECTURE.md](ARCHITECTURE.md)、
最初の起動は [README](../README.md) を見てください。

---

## 1. 必要な機材

| 項目 | 推奨 | 最低 |
|---|---|---|
| GPU | NVIDIA RTX 40/50 世代 | CUDA 対応 NVIDIA（VRAM 4 GB） |
| ドライバ | 最新 Game Ready / Studio | CUDA 12.x が動くもの |
| カメラ | 1080p / 60 fps の USB ウェブカメラ（Logitech Brio 等） | 720p / 30 fps |
| OS | Windows 11 | Windows 10 |
| Python | uv が導入する 3.12（自動） | — |

カメラは**正面から、胸の高さ**に。真上や真下からだと MediaPipe の手の向き推定が
不安定になります。逆光を避け、手が背景より明るく写るようにしてください。
MediaPipe は肌色を使いません——コントラストと十分な照度が全てです。

---

## 2. 初回セットアップ

新品の Windows PC なら `setup.bat` をダブルクリックするだけです（uv の導入 → 依存の同期 →
手モデルの取得 → `--check`、何度実行しても安全）。手で打つなら：

```bash
uv sync
uv run python tools/download_models.py
uv run python -m fctx --check
```

`--check` が `ready.` と言えば動きます。`[!!]` の行があればその行が理由です。
`[--] no camera found` はカメラ無しでも動く（合成ハンド）という意味で、エラーではありません。

---

## 3. 会場に合わせる（キャリブレーション）

ステージ（画面内で手が届く空間）とカメラの対応は 3 つの数で決まります。
会場ごとにカメラ距離が違うので、**設置したら必ず一度**計測してください。

```bash
uv run python tools/calibrate.py --camera 0 --seconds 20
```

20 秒間、開いた手を使いたい空間の隅々（近い・遠い・左右・上下）に動かします。
終わると `[tracking]` ブロックが出るので設定ファイルに貼ります：

```toml
[tracking]
camera_index = 0
reference_hand_span = 0.243   # この距離が「ステージ中央」になる
depth_scale = 0.51            # 近い・遠いが z_range の両端に来る
```

「手が画像の 15% 以下しか覆っていない」と言われたら、カメラが遠すぎます。

---

## 4. 設定ファイル

全ての調整項目は TOML で書けます。まず既定値を書き出して、変えたい行だけ残します：

```bash
uv run python -m fctx --dump-config > venue.toml
uv run python -m fctx --config venue.toml
```

典型的な会場設定：

```toml
preset = "cloth"

[tracking]
camera_index = 1
reference_hand_span = 0.243
depth_scale = 0.51
mirror = true          # 鏡像。来場者は右に動かすと右に動くのを期待する

[render]
fullscreen = true
show_hud = false       # 展示ではテレメトリを隠す
show_webcam = true     # 来場者に「自分の手が見えている」ことを示す

[scene]
hardness = 0.35

[exhibit]
coach = true                       # 来場者向けの短い案内を出す
prompt_grab = "親指と人差し指でつまんでみてください"
prompt_hardness = "もう片方の手を上げ下げすると硬さが変わります"
prompt_feel = "そのまま持っていてください ── 硬さが変わります"
prompt_release = "指を開くと離せます"
hardness_by_free_hand = true       # 片手で持ち、もう片方の手の高さで硬さ
sweep_while_holding = false        # 片手だけの会場なら true：持っている間、硬さが自動で往復
catalog = "materials.toml"         # 自社素材の名前と実測値（省略時は内蔵カタログ）
analytics = "logs/events.csv"      # 来場者・操作の記録（日報は tools/report.py）
```

案内文は日本語で構いません。画面の文字は ASCII 以外を Windows の游ゴシック／メイリオ／BIZ UD
で描画します（無ければ Noto Sans CJK、それも無ければ豆腐になります）。

タイポは起動時にエラーになります（`unknown key 'camera_idx' (did you mean camera_index?)`）。
黙って無視されることはありません。本番前の確認は `--check` に設定ファイルを添えます：

```bash
uv run python -m fctx --config venue.toml --check
```

`[ok]  config venue.toml (cloth, camera 1, source camera)` の行が出れば、その日の設定で起動できます。

---

## 5. 展示モード（キオスク）

```bash
uv run python -m fctx --config venue.toml --kiosk
```

`--kiosk` は次を同時に有効にします：

| 機能 | 動作 |
|---|---|
| フルスクリーン | `F11` で切替可能 |
| **アトラクトモード** | 20 秒間誰も手をかざさないと、合成ハンドで振り付けデモが自動で回る。カメラに手が映った瞬間（3 フレーム）に実操作へ戻る |
| **カメラ切断復帰** | 稼働中にカメラが抜けても止まらない。合成ハンドで続行し、3 秒ごとに再接続を試み、戻れば自動で切り替わる |
| **フレーム耐性** | 1 フレームで例外が出てもログに残してシーンをリセットし続行。30 回連続で失敗したら終了コードで抜ける（監視スクリプトで再起動） |
| **定期再起動** | 起動から 12 時間経ったら、次に無人になった瞬間（手が `idle_demo` 秒見えない）に正常終了し、`run_kiosk.bat` が新しいプロセスを立てる。来場者の前で消えることはない |
| **コーチ** | 手が映ったら「つまんでみて」、掴んだら「もう片方の手で硬さ」、硬さを変えたら「指を開くと離せる」を一度ずつ。数秒で消え、同じ段階で止まっていれば 8 秒後にもう一度。文言は `[exhibit]` で差し替え可 |
| **記録** | `logs/events.csv` に来場者の開始/終了、把持、硬さ操作、アトラクト、カメラ断、エラー、再起動を追記。`tools/report.py` で日報 |

個別に指定するなら `--idle-demo 20`、`--resilient`、`--max-uptime 12`、`--log-file run.log`。

Windows で常時起動させる最小の監視ループ（`run_kiosk.bat`）:

```bat
@echo off
cd /d %~dp0
:loop
uv run python -m fctx --config venue.toml --kiosk --log-file logs\fctx.log
echo restart at %date% %time% >> logs\restarts.log
timeout /t 3 >nul
goto loop
```

---

## 5b. 官能評価（スタディ）を回す

試作品なしで「違いがわかるか」「どれが好まれるか」を来場者に答えてもらうモードです。
`studies/` に二つの例があります。

```bash
uv run python -m fctx --study studies/hardness_jnd.toml --check          # 本番前の確認
uv run python -m fctx --config venue.toml --study studies/hardness_jnd.toml --kiosk
uv run fctx-analyze studies/results/hardness_jnd.csv --csv jnd_summary.csv --plot jnd.png
```

**来場者の流れ**：手をかざす → 説明（3 秒）→ 左右の試料を両方押す → 「硬い方（好きな方）の上に手を
高くかざす」→ 1.2 秒でゲージが満ちて確定 → 次の試行 → 規定数で「ありがとうございました」。
途中で立ち去ったら（既定 12 秒）そこまでの回答を残して次の人を待ちます。スタッフは `Z`（左）/ `X`（右）で
代理回答できます（記録上 `response_mode = key` として区別されます）。

常設で回すなら `venue.toml` の先頭に `study = "studies/hardness_jnd.toml"` と書けば、`run_kiosk.bat` がそのままスタディで起動します（12 時間ごとの定期再起動をまたいでも階段法は CSV から続きます）。

**ブラインドは自動**です。スタディ中は二つの試料が同じ色・同じ質感になり、HUD・素材名・デモ・アトラクト・
硬さジェスチャ・プリセット切替はすべて無効になります。

**スタディファイル**（`[study]` 表。未知のキーは起動時にエラー）：

| キー | 意味 |
|---|---|
| `protocol` | `discrimination`（どちらが硬い？二肢強制選択）／`preference`（どちらが好き？一対比較） |
| `reference` / `reference_material` | 基準の硬さ（ダイヤル値）、またはカタログの素材名 |
| `materials` | preference で比べる素材名（カタログ、`--catalog` の自社素材も可） |
| `trials` | 一人あたりの試行数（展示なら 6〜10） |
| `staircase_scope` | `study`：来場者をまたいで階段法を続ける（展示向け、再起動しても再開）／`participant`：一人ずつ |
| `pseudo_haptics` | `on` / `off` / `alternate`（交互。疑似触覚の効果を測る） |
| `explore_min` `touch_min` `dwell` `session_gap` | 探索の最低時間、各試料に触れる最低時間、回答の注視時間、離脱判定 |
| `output` | 結果 CSV（スタディファイルからの相対パス） |
| `prompt_*` `label_choice` | 画面の案内文（日本語可） |

**解析の読み方**：

```
DISCRIMINATION  (threshold = hardness-dial difference at 75% correct)
  pseudo-haptics  on:  120 trials,  14 people, 78% correct
      threshold 0.071 [0.058, 0.090]  staircase 0.066  -> modulus Weber fraction 72%
  pseudo-haptics off:  120 trials,  14 people, 71% correct
      threshold 0.118 [0.091, 0.160]  staircase 0.109  -> modulus Weber fraction 145%
  pseudo-haptics effect: threshold off/on = 1.66 [1.18, 2.35]  -> helps
```
（上は書式の例で、実測値ではありません。）

- `threshold` はダイヤル上の差で、75% 正答になる点。`Weber fraction` は同じ差をヤング率の比に直したもの
  （公称値。硬さ 0.4 以上は四面体の剛性上限に入るので、比は大きめに出ます）。
- `effect` の区間が 1 をまたがなければ、疑似触覚は弁別を「助けた／妨げた」と言えます。またぐなら差は不明です。
- 30 試行未満の条件には `<< only N trials` が付きます。その閾値は参考値です。
- `side bias` が 50% から大きく外れる場合（例：常に左）、回答方法の説明か設置位置を見直してください。

**言えることと言えないこと**：結果は「このシミュレーション上で、この会場の来場者が区別できた差」です。
実製品の判断に使うには、一度だけ実物の試料（ヤング率既知）で同じ手順のパネルを行い、閾値を突き合わせてください。

---

## 6. 操作の勘どころ

- **つまむ**：親指と人差し指の先を付ける。閾値はヒステリシス付き（0.62 で掴み、0.42 で離す）なので、しっかり閉じて、はっきり開く。
- **持ち上げる**：掴んだら手首を動かす。指の開閉は握力であって移動ではない（設計上そうしてある——指を開いても物は飛ばない）。
- **硬さ（来場者）**：片手で持ったまま、**もう片方の手を上げ下げ**する。ステージの下端で最も柔らかく、上端で最も硬い（既定で有効。`hardness_by_free_hand`）。片手しか使わない会場では `sweep_while_holding = true` にすると、持っている間だけ硬さが自動で往復する。
- **硬さ（スタッフ）**：マウスホイール、`[` `]`、`A` で自動掃引、`M` / `N` でカタログの素材を順送り（例：デニム → キャンバス）。
- **素材名**：HUD にダイヤル位置に最も近いカタログ素材が出る（`~ DENIM (jeans)`）。自社素材の名前と実測ヤング率を `materials.toml` に書けば、その名前が出る：

  ```toml
  [[material]]
  name = "当社 40 kg/m3 フォーム"
  kind = "soft"        # soft: ヤング率 Pa／cloth・grain: ソルバの伸び剛性 N/m
  value = 4.5e4
  note = "座面グレード"
  ```

  内蔵カタログの軟体の値は文献のオーダー（ゼラチン数 kPa、シリコーンゴム 1 MPa、タイヤ 10 MPa）、布と粒の値は
  「そう見える剛性」であって繊維や土の実測値ではありません。
- 布は上辺 2 点で吊ってある。下端をつまんで持ち上げると折り目が出やすい。

---

## 7. パフォーマンスの目安

RTX 5070 Ti、1600×900、他の GPU アプリ常駐時の実測：

| プリセット | physics | render | fps |
|---|---:|---:|---:|
| cloth | 1.0 ms | 1.3 ms | 207 |
| soft | 2.2 ms | 4.5 ms | 96 |
| grain 24,000 | 1.1 ms | 1.5 ms | 199 |

60 Hz のディスプレイでは vsync で 60 に張り付きます。フレームが落ちるときの順で：

1. `[render] ssao_samples = 0`
2. `[render] bloom = false`
3. `[render] msaa = 2`
4. `[scene] cloth_resolution = 56`（既定 72）／`soft_resolution = 17`（既定 21）
5. `[solver] substeps = 8`（既定 12。**材質は少し柔らかく見えるようになります**——XPBD の定式化自体はサブステップ数に依存しませんが、反復が有限なので収束度が下がります。実測で硬さ 0.3 の布の伸びは 12 → 6 サブステップで約 3 倍。スタディ中は変えないでください）

`--profile 2` で 2 秒ごとに内訳が出ます。

---

## 8. トラブルシューティング

| 症状 | 見るところ |
|---|---|
| 起動直後に落ちる | `--check`。`--verbose` で完全なトレースバック |
| 手が映らない | インセット（`W`）に骨格が出ているか。出ていなければ照明・距離。`min_detection_confidence` を 0.45 に下げるのは最後の手段 |
| 手が震える／布が揺れ続ける | One-Euro フィルタ：`filter_min_cutoff` を 1.2 に（滑らか・遅延増） |
| 手の反応が遅い | `filter_beta` を 8 に（既定 5.0）。`filter_min_cutoff` を 2.5 に |
| 手が奥行き方向に飛ぶ | キャリブレーションをやり直す。`depth_size_blend` を 0.6 に上げると手のサイズ推定を重視 |
| 掴めない | ピンチ点が物体から 7.5 cm 以内にあるか。`[grab] radius` を 0.10 に |
| 物が手からすり抜ける | `[grab] mass_scale` を 0.25 に（保持を強く） |
| 布が爆発した | `R` でリセット。ログに `reset` が並ぶなら `--log-file` を添えて報告 |
| カメラが認識されない | Windows の「カメラのプライバシー設定」でデスクトップアプリを許可。他のアプリ（Teams/Zoom）がカメラを掴んでいないか |
| 60 fps 以上出したい | `[render] vsync = false` |

---

## 9. 記録と再現

不具合の再現には手の動きの記録が一番役に立ちます：

```bash
uv run python -m fctx --config venue.toml --record bug01.fhr   # F9 でも開始/停止
uv run python -m fctx --config venue.toml --source replay --replay bug01.fhr
```

`.fhr` はランドマークだけ（映像は含まない）なので数十 KB です。同じ設定ファイルと
一緒に送れば、こちらで同じ挙動を再現できます。

---

## 10. 更新とテスト

```bash
git pull
uv sync
uv run python tools/run_tests.py            # 全部（GPU 必要、約 4 分）
uv run python tools/run_tests.py --cpu      # 幾何・追跡・設定のみ（GPU 不要）
```

CI（GitHub Actions）は CPU 側のテストと lint を毎プッシュで回します。

常設前には**ソークテスト**を一度回してください。キオスクと同じ構成（カメラ無し・アトラクト
モード・耐性あり）でヘッドレスに長時間走らせ、プロセスの RSS と専用 GPU メモリを 30 秒ごとに
記録して、1 時間あたりの増加量を出します：

```bash
uv run python tools/soak.py --minutes 45     # captures/soak.csv と verdict
```

`verdict: flat` なら一日回せます。`GROWING` なら CSV を添えて報告してください。

### 日報

```bash
uv run python tools/report.py logs/events.csv                 # 全期間
uv run python tools/report.py logs/events.csv --day 2026-09-19 --csv report.csv
```

時間帯ごとの来場者数・把持回数・保持秒数・硬さ操作回数・利用分数・アトラクト回数・カメラ断・エラーと、
合計（稼働時間あたりの来場者、一人あたりの滞在秒・把持・硬さ操作）が出ます。「来場者」は
「手が映ってから `session_gap`（既定 15 秒）以上途切れるまで」の一区切りで、交代で遊ぶグループは
少なく、離れて戻った一人は多く数えます——カメラだけで数える限界として下限と上限の間の値です。
