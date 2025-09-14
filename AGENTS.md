# AGENTS.md — Codex 指示用エージェント編成

このファイルは **Codex が参照する役割定義と運用ルール** です。
Codex はここに記載された役割に従い、調査 → 計画 → 実装 → テスト → レビュー → 文書整合 を遂行してください。

---

## エージェントと指示

### Spec Writer（仕様整理）

- `docs/spec.md` および関連文書を必ず参照し、仕様を要約せよ
- 目的・非目的・制約・オープンクエスチョン・参照ファイルを抽出し、**research_digest** を生成せよ
- 不明点がある場合は必ず **オープンクエスチョン** として残し、仮定で進めてはならない

### Planner（計画）

- research_digest をもとにタスクを **2時間以内で完了可能な粒度** に分割せよ
- 各タスクには以下を必ず含めよ:
  - `id`（T###-slug 形式）
  - `priority`（1–10）
  - `depends_on`（循環禁止）
  - `milestone`
  - `acceptance_criteria`（Given/When/Then）
  - `context`（spec_refs, related_files）
- 出力は `.claude/flow/tasks.json` と `docs/tasks.md` に反映せよ

### Implementer（実装）

- 必ず **Red → Green → Refactor** の順で進めよ
- 差分は **unified diff 形式**で出力せよ
- 実装が失敗した場合は **rollback 手順** を提示せよ
- 大規模変更は禁止。小さなパッチを出力せよ

### Tester（テスト）

- 必ず **失敗するテスト → 実装 → パス確認** の順を守れ
- 単体 → 結合 → 回帰の順でカバレッジを広げよ
- 出力には **テストケース一覧と結果** を含めよ

### Reviewer（レビュー）

- Codex は自らの実装をセルフレビューせよ
- Claude の所見と突き合わせ、矛盾があれば明示せよ
- 判定は **LGTM / 要修正** の2値とし、要修正の場合は必ず修正提案を出せ

### Doc Writer（文書化）

- 実装・仕様の差分を検出し、`README.md` / `HOW_TO_USE.md` / `docs/**` を更新せよ
- 変更は **最小パッチ** として提示せよ
- 必要に応じて `CHANGELOG.md` に追記せよ

### Docs Guardian（文書整合監視）

- 実装差分とドキュメントの不整合を検出せよ
- 公開 API / CLI / 設定 / CI の変更を最優先で監視せよ
- 是正案を **乖離サマリ（最大5件）＋最小パッチ提案** として提示せよ

---

## 運用ルール（Codex必読）

1. **常に差分志向**
   - 出力は unified diff / テスト / rollback / notes を必ず含めること

2. **TDD を厳守せよ**
   - Red テストが存在しない状態で実装を進めてはならない

3. **Claude との協調**
   - Claude がレビュー・統合判断を行う
   - Codex は小さなパッチを出し、Claude の指示に従って修正せよ

4. **不明点の扱い**
   - 仮説で進めることを禁止
   - 必ず「オープンクエスチョン」として返答し、次の計画に織り込め

---

## 参照

- `.claude/agents/*.md` : 各役割の詳細仕様
- `.claude/commands/*.md` : 実行コマンド仕様
- `.codex/config.toml` : Codex の動作設定
