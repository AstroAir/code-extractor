# 贡献指南

感谢您对 pysearch 的关注与贡献！

## 分支与提交规范

- 分支模型：
  - `main`/`master`：稳定分支，仅通过 PR 合并
  - 特性分支：`feat/<short-name>`
  - 修复分支：`fix/<short-name>`
  - 文档/构建等：`docs/<short-name>`、`build/<short-name>`
- 提交信息：采用 Conventional Commits
  - `feat: ...` 新功能
  - `fix: ...` 修复
  - `docs: ...` 文档
  - `build: ...` 构建/依赖
  - `ci: ...` CI 流程
  - `refactor: ...` 重构（非功能性）
  - `test: ...` 测试
  - `chore: ...` 杂项

## 开发环境

1. 安装 Python 3.10+
2. 安装依赖
   ```bash
   python -m pip install -U pip
   python -m pip install -e ".[dev]"
   ```
3. 安装 pre-commit
   ```bash
   python -m pip install pre-commit
   pre-commit install
   ```

## 常用命令

使用 Makefile 简化命令：

```bash
make dev          # 安装开发依赖
make lint         # 代码规范检查（ruff/black）
make format       # 自动格式化
make type         # 类型检查（mypy）
make test         # 运行测试（pytest + 覆盖率）
make docs         # 构建 docs（如启用）
```

## 代码风格与质量

- 格式化：Black（100 列）
- Lint：Ruff（规则集：E,F,I,UP,B）
- 类型：mypy（Py310）
- 覆盖率门禁：CI 强制 ≥85%

提交 PR 前请确保：

- [ ] 通过 `make lint`、`make type`、`make test`
- [ ] 变更已覆盖必要测试
- [ ] 如涉及 API/CLI，更新 README 或 docs

## 提交 PR

- 填写 PR 模板，关联 Issue（`Fixes #123`）
- 小步提交，易于 Review
- 如含破坏性变更，请在 PR 描述中突出说明，并在 CHANGELOG 草拟条目

## 发布与版本

- 遵循 SemVer 与 PEP 440
- 采用 Conventional Commits 生成变更日志（手工维护亦可）
- 通过 Git tag 触发 Release 工作流（参见 .github/workflows/release.yml）

## 安全问题

如发现安全漏洞，请勿公开提 Issue，参见 SECURITY.md 私下披露流程。