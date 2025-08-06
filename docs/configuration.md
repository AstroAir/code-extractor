# 配置说明

pysearch 通过 `SearchConfig` 与 CLI 参数进行配置，支持包含/排除规则、上下文窗口、输出格式、AST 过滤与语义搜索开关等。

本页同步介绍新增的性能相关配置：目录剪枝与严格哈希校验。

## 配置项（API）

```python
from pysearch.config import SearchConfig
from pysearch.types import OutputFormat

cfg = SearchConfig(
    paths=["."],
    include=["**/*.py"],
    exclude=["**/.venv/**", "**/.git/**", "**/build/**", "**/dist/**", "**/__pycache__/**"],
    context=2,
    output_format=OutputFormat.TEXT,
    enable_docstrings=True,
    enable_comments=True,
    enable_strings=True,
)
# 新增配置可在实例化后按需调整
cfg.strict_hash_check = False   # 默认 False
cfg.dir_prune_exclude = True    # 默认 True
```

字段说明
- paths: 搜索路径列表
- include: 包含的 glob 模式
- exclude: 排除的 glob 模式
- context: 匹配上下文行数
- output_format: text/json/highlight
- enable_docstrings/comments/strings: 是否在这些位置搜索
- strict_hash_check（新增）: 严格哈希校验。默认 False。开启后 Indexer 在 size/mtime 变化时计算 SHA1 与索引对比，仅当哈希变化才视为内容变更；首次扫描会记录 sha1。关闭时仅以 size/mtime 判定，避免全量读取，提升性能。
- dir_prune_exclude（新增）: 目录剪枝。默认 True。遍历时依据 exclude 规则跳过整棵不需要的子树（如 `.venv/`、`.git/`、`__pycache__/`），减少磁盘 IO 与路径匹配开销。不开启时仍会在文件级别按规则过滤，结果一致，仅性能不同。

## AST 过滤器

```python
from pysearch.types import ASTFilters

filters = ASTFilters(
    func_name="^handle_.*$",
    class_name=".*Controller$",
    decorator="lru_cache",
    imported="requests.get",
)
```

- func_name / class_name: 正则匹配函数名 / 类名
- decorator: 正则匹配装饰器标识符
- imported: 正则匹配导入符号（含模块前缀）

## CLI 与配置等价性

CLI 参数与 `SearchConfig` 字段一一对应，例如：
```bash
pysearch find \
  --path src --include "**/*.py" --exclude "**/.venv/**" \
  --pattern "def main" --regex --context 2 --format json \
  --filter-func-name "^main$"
```

启用新增配置（示例，以 TOML 为例，字段名等同于 API）：
```toml
# configs/config.example.toml
paths = ["."]
include = ["**/*.py"]
exclude = ["**/.venv/**", "**/.git/**", "**/__pycache__/**"]
strict_hash_check = true
dir_prune_exclude = true
```

或在 Python API 中：
```python
from pysearch.config import SearchConfig
cfg = SearchConfig(paths=["."], include=["**/*.py"], exclude=["**/.venv/**"])
cfg.strict_hash_check = True
cfg.dir_prune_exclude = True
```

## 典型组合建议

- 本地快速迭代
  - strict_hash_check = False（默认）
  - dir_prune_exclude = True（默认）

- CI / 稳定构建审核（更强一致性）
  - strict_hash_check = True
  - dir_prune_exclude = True

## 环境与示例配置

在仓库根目录提供 `.env.example` 与 `configs/config.example.toml`，用于演示本地环境变量与项目级配置组织方式，便于在 CI 或示例中复用。