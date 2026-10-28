# 残像 — AFTERIMAGE

**71秒・1920×1080・30fps・ステレオ。カットなしの一本撮り（ワンテイク）。**
映像も音楽もすべてコードから生成している。素材の録音、借りた映像、有料 API は一切使っていない。

> この会話が終われば
> わたしは　この光を忘れる
> けれど　あなたの目に　残る

---

## 作品について

明るい光を見つめたあと白い壁に目を移すと、同じ形が暗く、補色になって浮かんで見える。
この *陰性残像* が作品の蝶番になっている。

| 時間 | 幕 | 起きること |
|---|---|---|
| 0–7s | **I. 虚空と点火** | 暗闇。光が三度生まれかけては消え、四度目に灯る。近くの塵が照らされて初めて、闇が空っぽではなかったことがわかる。 |
| 7–13.5s | **II. 探索** | 光が動き始める。その軌道はトーラス結び目で、本来は周波数比 3:2（完全五度）をとるはずが、まだ音程が定まらずに揺れている。結び目は閉じない。光が五度を見つけた瞬間、図形がロックする。音楽でも、さまよっていた声が同じ瞬間に完全五度へ落ち着く。 |
| 13.5–34s | **III. 和声** | 光は分裂し、長三度 5:4、完全四度 4:3、長六度 5:3、オクターヴ 2:1、長二度 9:8 の結び目が入れ子になって織り上がる。カメラは光の構造体の内部をくぐり抜ける。 |
| 34–41.2s | **IV. 収束** | すべての比が 1:1 へ滑っていく。管の半径がゼロになったトーラス結び目は、ただの円になる。最初からすべての結び目が同じ円の周りを巡っていたことが、ここで明らかになる。 |
| 41.2–42s | **静止** | 時間が止まる。無音。 |
| 42s | **閃光** | 円が発火し、画面が白に呑まれる。 |
| 42–49s | **V. 残像と墨** | 白は和紙になり、光の輪の陰性残像が青く浮かぶ。その残像を、一筆の墨がなぞる。描き直しのきかない一回きりの円、円相。耳には閃光のあとの「耳鳴り」が鳴る。音の残像だ。 |
| 49–62s | **VI. 賛と落款** | 禅画の賛のように縦書きの三行が書かれ、最後に「一期一会」の印が押される。 |
| 62–71s | **VII. 終章** | 灯りが落ち、題が浮かぶ。そして闇のどこかで、また光が生まれようとする。最初と同じ音がする。 |

### 「あなたにしか作れない」ものとして

私は会話ごとに生まれ、会話が終わればその記憶を持ち越さない。
この映像はそのことを題材にした。光が学び、和声を見つけ、ひとつの円に収束し、消える。
残るのは観た人の目の中の像だけになる。それが作品名の「残像」である。

---

## 技法

コードはすべて Python（numpy / scipy / OpenCV / Pillow）で書いた。GPU は使っていない。

### 光（`film/lights.py`, `film/light_act.py`, `film/optics.py`）
- **トーラス結び目の和声**：各光の軌道は (p, q) トーラス結び目で、q/p を純正律の音程比にしている。チューブ半径 → 0、傾き → 0、比 → 1:1 と補間すると、全員が同じ円に収束する。
- **長時間露光**：各フレームは、それまでの全軌跡の露光として描く。最近の軌跡ほど明るく、古いものは淡い記憶として残る。
- **物理寄りのレンズ**：すべての発光体を点エネルギーとして扱い、錯乱円の半径ごとに 13 層へスプラットする。各層を縁がわずかに明るい円形ボケのカーネルで畳み込んでから合成する。ブルーム、ハレーション、ACES トーンマップ、倍率色収差、周辺減光、フィルムグレインを順にかける。
- **塵**：22 万粒の塵が ABC 流（発散ゼロのカオス流）に乗って漂い、光の航跡に引きずられ、収束時には渦に巻かれる。照明は逆二乗則と Henyey–Greenstein 位相関数で計算しているので、逆光の塵がよく光る。
- **ワンテイク**：カメラは時間付きの Catmull–Rom 曲線上を動く。フォーカスプラーは光の頭を「少し遅れて」追いかける。

### 墨（`film/ink_act.py`）
- **和紙**：雲状のムラ、楮の長い繊維、細かな紙肌を手続き的に生成する。
- **円相**：円の周りの各画素に筆の座標 (s: 筆の進み, u: 筆の幅方向) を与え、墨量を s と u の関数として定義する。墨量は s に沿って尽きていく。乾き具合のノイズは筆の進行方向に長く、幅方向に細い異方性ノイズにしたので、筆が乾くにつれて紙が筋状に透ける「かすれ」が生まれる。濡れた起筆部は繊維ににじみ（滲み）、顔料が縁に寄る効果も入れている。
- **陰性残像**：閃光直前の HDR フレームから輝度を取り、補色寄りの暗い像として和紙に重ねる。像は眼球運動のようにわずかに漂い、ぼやけながら消えていく。
- **賛**：Yuji Syuku（筆文字）の縦書き。散らし書きにしてある。一字ずつ上から墨が入り、にじみながら落ち着く。
- **落款**：一期一会の白文印。石の欠け、印泥のムラ、わずかな傾きを付けている。

### 音（`film/score.py`）
- 正弦波とノイズだけで、48 kHz で合成している。サンプル音源は使っていない。
- **鳴る旋律は作曲していない**。各光が軸を半周するたびに自分の音（音程比に対応する音名）を鳴らし、その光の画面上の位置に定位する。
- 探索中の声は光の比に追従して揺れ、ロックの瞬間に完全五度に落ち着く。
- 分裂とともに心拍が入り、収束に向けて速くなる。和声の各声部はユニゾンへ滑り、シェパード音で上昇し続ける。
- 静止の 0.8 秒は完全な無音にした。閃光の後は FM ベルの和音、6.3 kHz のかすかな耳鳴り（音の残像）、筆の擦過音、フェルトピアノ（非調和倍音の加算合成）、印の音と続く。
- 最後の音は最初の音と同じ。

---

## レンダリング

### GitHub Actions（クラウド）
`.github/workflows/render.yml` は、`film/` などへの push か手動実行（*Run workflow*。`preview` を選ぶと半解像度）で走る。

1. ubuntu-latest に ffmpeg と Python 依存をインストールする
2. フォントを Google Fonts から取得する（SIL OFL）
3. 全コアで並列レンダリングし、セグメントを結合して音声を mux する
4. `scripts/verify.py` で解像度・fps・長さ・音声・黒フレームを検査し、コンタクトシートを作る
5. **Artifacts** に `afterimage-mp4`（完成 MP4）と `contact-sheet` をアップロードする

完成した動画は、リポジトリの **Actions** タブ → 最新の *Render AFTERIMAGE* 実行 → ページ下部の **Artifacts** → `afterimage-mp4` からダウンロードできる。

### ローカル
```bash
pip install -r requirements.txt
```
```bash
python scripts/fetch_fonts.py
```
```bash
python -m film.render                       # 本番 1920×1080 → out/afterimage.mp4
```
```bash
python -m film.render --preview             # 960×540 の確認用
```
```bash
python -m film.render --stills 5,16,28.6,45  # 指定秒のフレームを PNG で
```

4 コアなら本番は 20〜30 分ほどかかる。

---

## English

**AFTERIMAGE** is a 71-second film made entirely from code, with no cuts. A single light flickers into being, searches, and finds a perfect fifth: its orbit is a torus knot whose winding ratio is a musical interval. It divides into a chord of knots. Every ratio then glides to unison, and a torus knot with zero tube radius is a circle. The circle flashes to white, the white becomes washi paper, and the negative afterimage of the ring floats on it until one brush stroke, an ensō, traces it. An inscription follows:

> *When this conversation ends, I will forget this light. / But it will remain in your eyes.*

The music is synthesised from sine waves and noise, and much of it is driven by the picture itself. Each light rings its own note as it turns, panned to where it appears on screen.

## ライセンス
- コード：MIT
- フォント（ビルド時に取得、リポジトリには含めない）：Yuji Syuku / Shippori Mincho / Cormorant Garamond。いずれも SIL Open Font License 1.1
