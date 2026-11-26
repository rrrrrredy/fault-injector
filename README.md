# Fault-Injector

## 阶段 2 的 CPU 接口

在仓库根目录可直接导入以下模块。训练数据锁定到 Hugging Face 提交 `29b9338d1742ec1013bbc72e533eba3c0be8c274` 的 `train/actor.zip` 和 `train/grader.zip`。两份归档逐字节校验后，沙盒只接受 `blinker`、`ordered`、`termcolor`、`time` 四个种子 ID。评分端保管 grader 和完整验收回执，模型输入使用 actor。

```python
from stage2.sandbox import Pool
from stage2.rewards import evaluate, export_case, extract_diff, reward
from stage2.state_store import open_state, record_step, snapshot, restore, cursor_state

with Pool(train_dir, runtime=wasi_dir) as pool:
    observation = evaluate(pool, seed_id, completion)
    # observation["reward"] is the scalar. Raw diffs can also be passed to pool.verify().
with open_state(db_path) as con:
    record_step(con, run_id, step, [
        {"seed": seed_id, "completion": completion, "result": observation}
    ], metadata={"next_seed": next_seed_id})
    archive = snapshot(con, archive_path)
db, checkpoint = restore(archive, new_empty_directory)
```

`Pool(train, *, runtime, workers=WORKERS, timeout=12, drive_paths=(), exempt_drive_paths=())` 支持并发调用 `verify(seed_id, diff, *, timeout=None)`。默认并发数为 `min(4, max(1, (os.cpu_count() or 2) // 2))`。宿主 Git 只操作冻结切片中的现有 Python 源文件。候选代码在无网络的 WASI CPython 内执行，源文件、测试及解释器目录只读。`drive_paths` 用来补充自定义挂载点。可读的 Drive 挂载会阻止启动。只有显式列入 `exempt_drive_paths` 的路径可豁免宿主侧检查，来宾的只读目录范围始终固定。退出上下文会等候在途验证结束并清理工作树。

`extract_diff(completion)` 提取补丁。`evaluate(pool, seed_id, completion, *, timeout=None)` 运行验证，`reward(observation)` 是可单独调用的纯函数。冻结断言在故障态失败、反向补丁还原全绿且指纹一致得 `+1`，补丁应用后原测试仍全绿得 `0`，无效补丁、超时或其他验证失败得 `-1`。结果包含 `reason`、`files_changed`、`syntax_ok`、`import_ok`、`failed_nodes`、两态退出码和重置指纹。

`export_case(train, seed_id, completion, *, runtime, tools, out)` 独立复验正例，生成三个同源 Case，并返回带有两次重置指纹和 RunCase 语义校验结果的回执。`out` 须为空，输出包含 `cases/`、`candidate.diff`、`fixture.bundle` 和 `receipt.json`。三个解释器哈希种子为 `1729`、`2027`、`4093`。`source_count` 按候选补丁来源计数。

`record_step(con, run, step, rollouts, *, metadata=None)` 在同一事务中提交一个 step 的全部评估记录和断点元数据。`cursor_state(con)` 返回各 run 的末次 step、元数据及总记录数。`snapshot(con, archive)` 用 SQLite 在线备份生成 ZIP 和同名 `.sha256`。`restore(archive, new_directory)` 校验归档、数据库摘要和 SQLite 完整性，返回 `(数据库路径, 断点摘要)`。活跃数据库始终放在本地磁盘。

### 从干净环境复算

使用 CPython 3.12、Git 和带 npm 的 Node.js，在已激活的空 Python 虚拟环境中执行。前四条命令准备受信依赖和冻结输入。最后一条在本地离线运行，生成 `cpu-receipt.json`。已有上述两份 HF 归档时，`--train` 可直接指向存放它们的目录，跳过 `prepare-train`。Ubuntu CI 在独立网络命名空间中执行相同验收。

```shell
python -m pip install -r stage2/requirements-cpu.txt
python -m stage2.sandbox prepare --runtime evidence/setup/wasi
python -m stage2.check prepare-train --train evidence/setup/train
python -m stage2.rewards prepare-export --tools evidence/setup/export
python -m stage2.check verify --train evidence/setup/train --runtime evidence/setup/wasi --tools evidence/setup/export --out evidence/stage2-check
```


## 阶段 1 数据集重放

在冻结提交上运行 `python stage1/build.py`。需要 CPython 3.12、Git 和带 npm 的 Node.js。命令从指定 Git 对象重建切片，使用固定版本的 Qwen3-14B 分词器，分别运行公开与隐藏测试，并用阶段 0 的 Factory 与 RunCase 验收两道真实回归。输出位于本次独立的 `evidence/stage1-*/` 目录。也可以用 `--output <空目录>` 指定位置。

训练、未见种子和历史回归按仓库家族两两隔离。输入清单见 [manifest.json](stage1/manifest.json)，复算目标见 [fingerprints.json](stage1/fingerprints.json)。完整审计记录在 `audit.json`，两套可分发包在 `release/`。只向训练端提供 train 的 actor 数据；评测时只向参与者提供单题 actor 目录。评分端持有 grader 归档与出处，三份同源 Case 计一个真实回归。数据集卡片分别说明署名、原始许可与适用范围。

## 阶段 0 的历史参照题

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
