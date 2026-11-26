# Fault-Injector

这个仓库的阶段 0 重放一个真实 Python 回归。humanize 4.11.0 把 `naturalsize(1.123456789)` 显示为 `1 Bytes`，4.12.0 保留了小数。这里使用上游 4.12.0 的提交和其直接后继修复，调用 Workflow Environment Factory 物化 Case，再交给 RunCase CLI 做语义校验。[原工单](https://github.com/python-humanize/humanize/issues/239) · [合入的修复 PR](https://github.com/python-humanize/humanize/pull/240)

## 重放

需要 Git、可创建 venv 的 CPython 3.12 以及自带 npm 的 Node.js。CI 固定使用 Python 3.12.14 和 Node.js 24.22.3。首次准备从 GitHub、PyPI 和 npm 取得固定输入，执行目标断言时只读取本地文件。

```shell
git clone https://github.com/rrrrrredy/fault-injector.git
cd fault-injector
python replay.py
```

在新建的临时目录中，入口按提交拉取源码，用上游 Python 约束安装 Pydantic、JSON Schema 校验器和 pytest，按上游 npm 锁构建并链接真正的 `rci`。四个提交、三个 Schema 内容摘要、真实回归出处和经确认的变量值均在 [stage0.lock.json](stage0.lock.json)。本阶段执行两个上游，另外两个只登记版本。

## 核对证据

每次执行向终端打印证据目录。成功时，其中有 `receipt.json` 和 `cases/case-{0,1,2}.json`。Ubuntu 工作流也会上传这四份文件。[GitHub Actions](https://github.com/rrrrrredy/fault-injector/actions/workflows/replay.yml)

- 基础 Case 调用与工单相同的断言，预期输出是 `1 Bytes`。其 nodeid 为 `fixtures/test_humanize_239.py::test_fractional_bytes`。夹具从当前工作树加载公开 API `humanize.naturalsize` 所导出的函数实现，避免源代码检出缺少构建时生成的版本文件。两次执行共用同一份夹具。
- 基础和派生 Case 的默认精度依次为 `%.1f`、`%.0f`、`%.3f`。输入 3000 字节时三个格式产生可区分的输出。原工单的小数点问题在三个配置中共享一个来源 ID。派生 Case 另含 Factory 生成的 `parent_case_id`。
- `cases[0].faulty` 和 `cases[0].fixed` 保留 Factory 实际子进程的输出、退出码和耗时。`reset_fingerprint_1` 和 `reset_fingerprint_2` 记录上游在两次独立物化时实际返回的指纹，可与 `validation.baseline_manifest_ref` 比对。Factory 指纹包含文件的可执行权限位，因此在每个宿主内比较三值；两个操作系统间可另核对 Git 原始 `baseline_git_tree`。
- `rci validate --semantic cases` 在证据目录对三份导出文档一次完成校验。只有三个 Case 都通过 Factory 验收、同源计数等于一且 CLI 返回 0，入口才写出最终回执。

这是受信夹具的宿主重放。Blueprint 保留固定镜像摘要，实际执行器在回执中记录为 `local-test-only`；重放使用主机 Python。两处观察代码原样返回 Factory 的指纹和进程结果，让文件保存上游默认结果对象没有保留的证据。[镜像摘要记录](https://github.com/docker-library/repo-info/blob/abe9d0133c7e10ada4cb0eff33aad5d1019bfd1e/repos/python/remote/3.12-slim.md)

## 许可

项目与随附的 RunCase Schema 使用 Apache-2.0，署名见 [NOTICE](NOTICE)。重放按固定提交获取的 humanize 使用上游 MIT 许可证。[项目许可](LICENSE) · [humanize 许可](https://github.com/python-humanize/humanize/blob/a0602c70bb6e11322107b51114333bab7adab021/LICENCE)
