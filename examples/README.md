# PySearch Examples

本目录包含 PySearch 各核心功能的示例脚本，帮助您快速上手。

## 前置条件

```bash
# 安装 pysearch（开发模式）
pip install -e .

# 可选依赖（按需安装）
pip install -e ".[watch]"      # 文件监控
pip install -e ".[graphrag]"   # GraphRAG
pip install -e ".[semantic]"   # 高级语义搜索
pip install -e ".[all]"        # 全部可选功能
```

## 示例列表

| 编号 | 文件 | 功能说明 |
|------|------|----------|
| 01 | [01_basic_search.py](01_basic_search.py) | 文本搜索、正则搜索、输出格式切换、搜索统计 |
| 02 | [02_ast_search.py](02_ast_search.py) | AST 结构化搜索、按函数/类/装饰器/导入过滤 |
| 03 | [03_fuzzy_search.py](03_fuzzy_search.py) | 模糊搜索（多算法）、语音搜索、拼写纠正 |
| 04 | [04_boolean_search.py](04_boolean_search.py) | 布尔逻辑查询 (AND/OR/NOT)、快速计数 |
| 05 | [05_semantic_search.py](05_semantic_search.py) | 语义搜索、概念级代码检索 |
| 06 | [06_dependency_analysis.py](06_dependency_analysis.py) | 依赖图、循环依赖检测、耦合度分析、重构建议 |
| 07 | [07_history_and_bookmarks.py](07_history_and_bookmarks.py) | 搜索历史、书签管理、搜索分析、导出与备份 |
| 08 | [08_caching_and_performance.py](08_caching_and_performance.py) | 缓存管理、排序策略、结果聚类 |
| 09 | [09_file_watching.py](09_file_watching.py) | 文件监控、实时索引更新 |
| 10 | [10_workspace_and_multi_repo.py](10_workspace_and_multi_repo.py) | 工作区管理、多仓库搜索 |
| 11 | [11_ide_integration.py](11_ide_integration.py) | IDE 集成（跳转定义、查找引用、补全、诊断） |
| 12 | [12_error_handling_and_logging.py](12_error_handling_and_logging.py) | 错误处理、日志配置 |

## 运行方式

```bash
# 在项目根目录下运行（确保 src/ 目录可被搜索）
python examples/01_basic_search.py

# 或指定搜索路径
python examples/01_basic_search.py --path /your/project
```

> **注意**: 大部分示例默认搜索当前项目的 `src/` 目录。请确保在项目根目录下运行。
