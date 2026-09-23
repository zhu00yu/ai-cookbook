# llama.cpp 官方 CUDA Docker 镜像在 WSL2 (Docker Desktop) 下不加载 CUDA 后端的报告与已验证修复方案

> 本报告基于 2026-09-22 ~ 2026-09-23 的完整实测排障，包含问题复现、证据链、根因分析和一套
> **已验证可用**
> 的自建镜像方案。
> 目标读者：llama.cpp 维护者 / 受 "官方镜像看得到 GPU 但实际跑 CPU" 问题困扰的用户。



***

## 1. 一句话摘要

官方 `ghcr.io/ggml-org/llama.cpp:server-cuda` / `:full-cuda` 镜像在 **WSL2 + Docker Desktop** 环境下，容器内能看到 GPU（`nvidia-smi`、`/dev/dxg` 均正常），但 llama.cpp **从不激活 CUDA 后端**，实测为纯 CPU 速度；根因是官方镜像依赖运行时插件（dlopen `libggml-cuda.so`）的加载机制在此环境**静默失败**，且失败无任何日志。**已验证的修复**：用 `-DGGML_BACKEND_DL=OFF` 自建镜像，将 CUDA 静态集成进核心库依赖链，配合 `--load-mode none`（绕开 WSL2 挂载盘 mmap 阻塞），GPU 实测生效（生成速度～62-65 tok/s，CPU 对照 23.6 tok/s，提速约 2.7 倍）。



***

## 2. 环境



| 项              | 值                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 操作系统           | Windows（Docker Desktop Linux 引擎 = WSL2 后端）                                                                                                 |
| Docker Desktop | 29.7.2                                                                                                                                     |
| GPU            | NVIDIA GeForce MX450（2GB 显存，Turing 架构 = **sm\_75**）                                                                                        |
| NVIDIA 驱动      | 581.95（CUDA 13.0，容器 CUDA 12.8 向前兼容）                                                                                                        |
| 容器架构           | x86\_64（legacy CPU）                                                                                                                        |
| 镜像             | `ghcr.io/ggml-org/llama.cpp:server-cuda`（digest sha256:0192ab25… / b98cf7ad…）、`ghcr.io/ggml-org/llama.cpp:full-cuda`（10.3GB，2026-09-22 构建） |
| 模型             | MiniCPM5-1B-Q4\_K\_M.gguf（656MB）                                                                                                           |



***

## 3. 问题现象



* 容器能正常启动、`/health` 正常、模型能加载能出字 ——**一切 "看起来正常"**。

* 容器内 GPU 可见：`nvidia-smi -L` 列出 MX450、`/dev/dxg` 存在、驱动已挂载（`ldd` 显示 `libcuda.so.1 => /usr/lib/x86_64-linux-gnu/libcuda.so.1`）。

* 但 llama.cpp 加载模型后**零 CUDA 日志、零 offload 日志**，`kv_unified=true`。

* 实测速度（同模型、`-ngl 99`）：


  * 官方 `server-cuda`：`prompt_eval ≈ 0.43 tok/s`、`gen ≈ 3.0 tok/s`（纯 CPU）

  * 官方 `full-cuda`（即使手动注入环境变量）：`gen ≈ 3.17 tok/s`（纯 CPU）

  * 自建镜像：`gen ≈ 62-65 tok/s`、prompt 处理 243 tok/s（18 tokens / 74ms）

## 4. 复现步骤



```
\\# 拉取官方 CUDA 镜像

docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

\\# 运行（GPU 透传）

docker run --rm --gpus all -v /path/to/models:/models \\\\

\&#x20; \\--entrypoint llama-server ghcr.io/ggml-org/llama.cpp:server-cuda \\\\

\&#x20; -m /models/MiniCPM5-1B-Q4\\\_K\\\_M.gguf --host 0.0.0.0 --port 8080 -ngl 99
```

观察：启动日志无任何 `CUDA` / `load_backend` / `offload` 字样；`/completion` 接口实测为 CPU 速度。



***

## 5. 证据链（为什么确定是镜像问题而非环境问题）



| # | 检查项                                              | 结果                                                              |
| - | ------------------------------------------------ | --------------------------------------------------------------- |
| 1 | 容器内 `nvidia-smi -L`                              | ✅ 列出 `GPU 0: NVIDIA GeForce MX450`（GPU 透传正常）                    |
| 2 | 容器内 `/dev/dxg`                                   | ✅ 存在                                                            |
| 3 | `ldd /app/llama-server`                          | ❌ **无任何 CUDA 库**（只有 libggml.so/libggml-base.so）；launcher 仅 17KB |
| 4 | `libllama-server-impl.so` 符号统计                   | ❌ `ggml_backend_cuda` 出现 **0 次**（主程序不含 CUDA 代码）                 |
| 5 | 手动指定插件 `GGML_BACKEND_PATH=/app/libggml-cuda.so`  | ❌ 无效                                                            |
| 6 | 再叠加 `LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64` | ❌ 仍无效，实测 `gen ≈ 3.17 tok/s`（纯 CPU）                              |
| 7 | `ldd /app/libggml-cuda.so`（插件本体）                 | 依赖基本齐全；唯一 `libcuda.so.1 => not found`，挂 `--gpus` 后存在（非阻塞点）      |

结论：**镜像内 165MB 的&#x20;**`libggml-cuda.so`**&#x20;插件存在且依赖完整，但主程序运行时的插件加载机制未生效，且失败被静默吞掉**。



***

## 6. 根因分析

### 6.1 官方镜像采用 "运行时插件（dlopen）" 架构，而该路径在此环境失效

新版 llama.cpp 默认 `GGML_BACKEND_DL=ON`：主程序不静态链接 CUDA，而是在启动时通过 `ggml_backend_load_all()` 扫描并 dlopen 后端插件（源码：`ggml/src/ggml-backend-reg.cpp`）。

关键代码路径：



```
// ggml/src/ggml-backend-reg.cpp

void ggml\\\_backend\\\_load\\\_all() {

\&#x20;   ...

\&#x20;   ggml\\\_backend\\\_load\\\_best("cuda", silent, dir\\\_path);   // 尝试加载 libggml-cuda-\\\*.so

\&#x20;   ...

\&#x20;   const char \\\* backend\\\_path = std::getenv("GGML\\\_BACKEND\\\_PATH");

\&#x20;   if (backend\\\_path) {

\&#x20;       ggml\\\_backend\\\_load(backend\\\_path);                // 手动指定插件

\&#x20;   }

}
```



```
\\#ifdef NDEBUG

\&#x20;   bool silent = true;          // release 构建：加载失败不打印任何日志！

\\#else

\&#x20;   bool silent = false;

\\#endif
```

两个关键点：



1. `silent=true`**（release 构建）**：dlopen 失败、符号缺失、注册失败 ——**全部静默**，用户和 CI 都看不到任何错误。这正是 "看起来一切正常却跑 CPU" 的原因。

2. 手动指定 `GGML_BACKEND_PATH` 后仍失败（证据 #5/#6），说明失败点在插件 dlopen / 注册的更深处（很可能是插件与主程序的 backend API 版本不匹配，或注册接口未正确对接），**不是单纯 "路径没配置"**。

### 6.2 自建时还会遇到的第二个坑：NVIDIA 容器镜像不含 `libcuda.so`

`nvidia/cuda:*` 镜像（devel/runtime）**故意不包含驱动库&#x20;**`libcuda.so`（驱动属于宿主机）。因此在容器内编译 llama.cpp 时，链接阶段会报：



```
/usr/bin/ld: libggml-cuda.so: undefined reference to \\\`cuMemAddressFree'

/usr/bin/ld: libggml-cuda.so: undefined reference to \\\`cuMemMap'

...（一堆 CUDA Driver API 符号）

collect2: error: ld returned 1 exit status
```

需要给链接器加 `-Wl,--allow-shlib-undefined`（运行时由 `--gpus` 把宿主驱动挂载进容器解析）。

### 6.3 自建时还会遇到的第三个坑：WSL2 挂载盘上 mmap 模型文件会无限阻塞

容器挂载 Windows 目录（如 `C:/models:/models`）走 WSL2 的 9p/virtiofs 协议，llama.cpp 默认用 mmap 读 GGUF 模型文件，**在挂载盘上 mmap 会无限阻塞**：进程 0% CPU、日志永远停在 `load_model` 之前，看起来像死机（该现象在官方 CPU-only 镜像上不会暴露，因为它从不真正加载模型）。

修复：运行时加 `--load-mode none`（禁用 mmap，改普通读取）。

### 6.4 设备选择建议

WSL2 容器内 CUDA 枚举中 MX450 为 `device 0`，建议 `--gpus device=0` 精确透传（避免 `--gpus all` 把核显 / D3D 等一并传入）。



***

## 7. 已验证的解决方案：自建静态集成镜像

### 7.1 Dockerfile（完整，可直接使用）



```
\\# 自建 llama.cpp (官方 master) CUDA server 镜像

\\# 关键修复1: GGML\\\_BACKEND\\\_DL=OFF -> CUDA 静态编进依赖链, 绕开运行时插件加载失效的坑

\\# 关键修复2: CMAKE\\\_EXE\\\_LINKER\\\_FLAGS=-Wl,--allow-shlib-undefined -> 镜像无 libcuda.so 的链接问题

\\# 目标 GPU: NVIDIA MX450 (Turing, sm\\\_75) -> CMAKE\\\_CUDA\\\_ARCHITECTURES=75 (按你的 GPU 架构调整)

\\# 基础镜像可按需换官方源: nvidia/cuda:12.8.1-devel-ubuntu24.04 (本示例用 DaoCloud 国内镜像源)

ARG CUDA\\\_VERSION=12.8.1

ARG UBUNTU\\\_VERSION=24.04

ARG REGISTRY=docker.m.daocloud.io

FROM \\\${REGISTRY}/nvidia/cuda:\\\${CUDA\\\_VERSION}-devel-ubuntu\\\${UBUNTU\\\_VERSION} AS build

RUN (sed -i 's@//\\\[^ ]\\\*archive.ubuntu.com@//mirrors.aliyun.com@g; s@//security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; sed -i 's@//\\\[^ ]\\\*archive.ubuntu.com@//mirrors.aliyun.com@g; s@//security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list 2>/dev/null; true)

RUN apt-get update && \\\\

\&#x20;   apt-get install -y --no-install-recommends \\\\

\&#x20;       gcc-14 g++-14 build-essential cmake python3 python3-pip git libssl-dev libgomp1 \\\\

\&#x20;   && rm -rf /var/lib/apt/lists/\\\*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /app

COPY . .

RUN cmake -B build -DGGML\\\_NATIVE=OFF -DGGML\\\_CUDA=ON -DGGML\\\_BACKEND\\\_DL=OFF \\\\

\&#x20;       -DCMAKE\\\_CUDA\\\_ARCHITECTURES=75 -DGGML\\\_CPU\\\_ALL\\\_VARIANTS=OFF \\\\

\&#x20;       -DLLAMA\\\_BUILD\\\_TESTS=OFF -DLLAMA\\\_BUILD\\\_EXAMPLES=ON -DLLAMA\\\_BUILD\\\_SERVER=ON \\\\

\&#x20;       -DCMAKE\\\_EXE\\\_LINKER\\\_FLAGS=-Wl,--allow-shlib-undefined \\\\

\&#x20;   && cmake --build build --config Release -j\\\$(nproc)

RUN mkdir -p /app/lib && find build -name "\\\*.so\\\*" -exec cp -P {} /app/lib \\\\;

FROM \\\${REGISTRY}/nvidia/cuda:\\\${CUDA\\\_VERSION}-runtime-ubuntu\\\${UBUNTU\\\_VERSION} AS runtime

RUN apt-get update && \\\\

\&#x20;   apt-get install -y --no-install-recommends libgomp1 curl \\\\

\&#x20;   && rm -rf /var/lib/apt/lists/\\\*

WORKDIR /app

COPY --from=build /app/lib/ /app/

COPY --from=build /app/build/bin/llama-server /app/llama-server

COPY --from=build /app/build/bin/llama /app/llama

ENV NVIDIA\\\_VISIBLE\\\_DEVICES=all NVIDIA\\\_DRIVER\\\_CAPABILITIES=compute,utility

ENV LLAMA\\\_ARG\\\_HOST=0.0.0.0

HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT \\\["/app/llama-server"]
```

> 说明：
> `GGML_BACKEND_DL=OFF`
> 后，llama-server 的
> `ldd`
> 直接包含
> `libggml-cuda.so`
> （静态依赖），不再依赖运行时插件加载。

### 7.2 构建与运行



```
\\# 1. 下载官方源码

\\#    https://codeload.github.com/ggml-org/llama.cpp/zip/refs/heads/master

\\# 2. 把上面的 Dockerfile 放进源码根目录，构建（约 35-50 分钟，8 核）

docker build -f Dockerfile.cuda.local -t llama-server-cuda-local:latest .

\\# 3. 运行（关键参数：device=0 + load-mode none）

docker run -d --name llama-server \\\\

\&#x20; \\--gpus device=0 \\\\

\&#x20; -v C:/zhuyu/ai/models:/models \\\\

\&#x20; -p 127.0.0.1:8080:8080 \\\\

\&#x20; llama-server-cuda-local:latest \\\\

\&#x20; -m /models/MiniCPM5-1B-Q4\\\_K\\\_M.gguf \\\\

\&#x20; \\--host 0.0.0.0 --port 8080 \\\\

\&#x20; -ngl 99 -c 2048 --no-warmup --no-ui \\\\

\&#x20; \\--load-mode none          # ← 必须！绕开 WSL2 挂载盘 mmap 阻塞
```

### 7.3 验证结果（本机实测，MiniCPM5-1B-Q4\_K\_M）



| 场景                          | 生成速度              | 备注                              |
| --------------------------- | ----------------- | ------------------------------- |
| 官方 `server-cuda`（CPU-only）  | ≈ 3.0 tok/s       | 基线                              |
| 官方 `full-cuda` + 全部环境变量注入   | ≈ 3.17 tok/s      | 插件加载仍失败                         |
| 自建镜像（`-ngl 0`，纯 CPU 对照）     | ≈ 23.6 tok/s      | 同镜像同机器                          |
| **自建镜像（**`-ngl 99`**，GPU）** | **≈ 62-65 tok/s** | **提速～2.7x**，prompt 处理 243 tok/s |

验证方法：`POST /completion`，读取响应 `timings.predicted_per_second`。



***

## 8. 给 llama.cpp 维护者的建议（按优先级）



1. **插件加载失败不要静默**：`ggml/src/ggml-backend-reg.cpp` 中 release 构建 `silent=true` 吞掉了所有后端加载错误。建议至少 `GGML_LOG_WARN` 级别输出失败原因（dlopen error、score 函数缺失、注册失败），否则 "镜像跑 CPU" 这类问题完全不可诊断。

2. **官方 CUDA 镜像改用静态集成**：`GGML_BACKEND_DL=OFF` 方案经实测在本环境可靠（无需插件加载）；或至少提供 `*-cuda-static` 变体镜像。

3. **CI 增加 GPU 冒烟测试**：官方 CI 目前只验证容器能启动、模型能出字，无法发现 "GPU 未激活"。建议加一条 GPU 环境（哪怕 CPU 环境只验证 `ggml_backend_count()` / 设备枚举）断言 CUDA 后端真实可用，并打印 offload 层数。

4. **WSL2 mmap 问题**：`--load-mode auto` 在 9p/virtiofs 挂载盘上会无限阻塞，建议在 mmap 失败 / 超时后自动回退普通读取，或文档明确提示 WSL2 挂载盘需 `--load-mode none`。



***

## 9. 附录



* 相关源码：`ggml/src/ggml-backend-reg.cpp`（`ggml_backend_load_all` / `ggml_backend_load_best` / `silent` 逻辑）

* 本报告所有数字均为同一台机器、同一模型（MiniCPM5-1B-Q4\_K\_M.gguf）实测；速度取 `/completion` 返回的 `timings`。

* 环境差异注意：本机 GPU 为 Turing sm\_75（MX450），`CMAKE_CUDA_ARCHITECTURES` 需按目标 GPU 调整（可用 `nvidia-smi --query-gpu=compute_cap --format=csv` 查询）。
