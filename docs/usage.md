# 快速上手

## 安装
需要 Python 3.10+。
```bash
pip install -U pip
pip install -e .
# 可选：开发依赖
pip install -e ".[dev]"
```

## CLI 使用
```bash
pysearch find \
  --path src \
  --include "**/*.py" \
  --pattern "def main" \
  --regex \
  --context 2 \
  --format json
```

常用参数
- --path: 搜索路径（可多次提供）
- --include/--exclude: 包含/排除的 glob
- --pattern: 模式（文本或正则）
- --regex: 启用正则
- --context: 上下文行数
- --format: text/json/highlight
- --filter-func-name / --filter-class-name / --filter-decorator / --filter-import
- --no-docstrings / --no-comments / --no-strings
- --stats: 打印统计

## Python API
```python
from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Query, OutputFormat

cfg = SearchConfig(paths=["."], include=["**/*.py"], context=2)
engine = PySearch(cfg)

q = Query(pattern="ClassName", use_regex=False, use_ast=False, output=OutputFormat.JSON)
res = engine.run(q)
print(res.stats)
```

## 输出格式
- text: 纯文本
- json: 机器可读 JSON（推荐与工具链集成）
- highlight: 交互终端彩色高亮（需 TTY）

## 故障排查
- 未命中：检查 include/exclude 与路径
- 慢：减少路径、增大 exclude、关闭不必要的解析（如 docstrings）
- 编码：确保文件为 UTF-8