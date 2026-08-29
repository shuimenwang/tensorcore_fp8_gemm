# 从 1.1 到 59.3 TFLOPS：一个数学系本科生的 FP8 Tensor Core GEMM 手写之路

> 本项目从零手写 CUDA FP8 GEMM 算子，最终达到 **59.3 TFLOPS**（RTX 5060，2048³ 矩阵），
> 为 **cuBLASLt FP8 官方算子的 53.5%**，相对朴素 CUDA 实现加速约 **55 倍**。
> 项目完整记录了五轮逐层优化（naive → cp.async 双缓冲 → Swizzle 消 bank conflict →
> mma 指令 FP8 在线反量化 → ldmatrix + 两相流水线）的动机、实现、数据与归因分析，
> 以及每一轮踩过的坑。所有结论均以 Nsight Compute 硬件计数器数据为依据。

---

## 目录

- [1. 项目目标](#1-项目目标)
- [2. 优化手段分层导览](#2-优化手段分层导览)
- [3. 探究心路：从看不懂 GPU 到手写 PTX](#3-探究心路从看不懂-gpu-到手写-ptx开发日志摘编)
  - [3.1 起点：先弄懂「数据住在哪」](#31-起点先弄懂数据住在哪731)
  - [3.2 双缓冲：在纸上把一个线程推演一遍](#32-双缓冲在纸上把一个线程推演一遍84)
  - [3.3 第一次 Profiling：让硬件自己说出瓶颈](#33-第一次-profiling让硬件自己说出瓶颈812)
  - [3.4 最难的一仗 Part I：L3 mma 在线反量化融合](#34-最难的一仗-part-il3-mma-在线反量化融合823-日志的-80-痛苦)
  - [3.5 最难的一仗 Part II：L4 ldmatrix + 两相流水线](#35-最难的一仗-part-iil4-ldmatrix-硬件加载--b-矩阵两相流水线)
  - [3.6 方法论沉淀（调试 + AI 工作流）](#36-方法论沉淀调试--ai-工作流)
- [4. 数据分析](#4-数据分析)
  - [4.1 纵向对比：五层逐级优化](#41-纵向对比五层逐级优化)
  - [4.2 横向对比：完整版 vs cuBLASLt vs Naive 全指标解剖](#42-横向对比完整版-vs-cublaslt-vs-naive-全指标解剖)
- [5. 下一步优化方向（面向大模型推理优化实习）](#5-下一步优化方向面向大模型推理优化实习)
- [6. 复现指南](#6-复现指南)
- [附：数据缺口清单（待补测/待贴图）](#附数据缺口清单待补测待贴图)

---

## 1. 项目目标

大模型推理中，矩阵乘法（GEMM）通常占据 60% 以上的算力开销；FP8 量化（E4M3/E5M2）是当前推理
加速的主流手段。本项目回答一个问题：

**在不使用任何模板库（CUTLASS 等）的前提下，一个非 CS 出身的学生，能靠对硬件的理解手写一个
多接近 cuBLAS 的 FP8 GEMM？**

为此设立了三条标准：

1. **正确性硬标准**：每种实现必须先通过 CPU 逐元素真值比对（MaxDiff 归零）才允许进入 benchmark；
2. **性能硬标准**：以 cuBLASLt FP8 为上限标尺，用 TFLOPS 和「达到 cuBLAS 的百分比」衡量每一轮进步；
3. **理解硬标准**：不允许出现"不知道为什么变快"的优化——每一轮必须给出 NCU 硬件计数器层面的归因。

**软硬件环境**：NVIDIA GeForce RTX 5060（Blackwell, sm_120，30 SM，L2 24 MB，显存总线 128-bit）、
CUDA 12.9、Inline PTX 手写 Tensor Core 指令。性能上限以 cuBLASLt FP8 实测为参考。

---

## 2. 优化手段分层导览

整个项目按「榨取硬件的深度」分为五层，每层对应一个可独立编译测试的算子：

| 层级 | 算子 | 核心技术 | 测试程序 | 精度 |
|---|---|---|---|---|
| L0 朴素 | `naive_fp8_gemm_kernel` | 每线程算一个 C 元素，无共享内存 | 各测试程序内置 | FP8→FP32 |
| L1 异步流水线 | `async_double_buffer_pipeline_kernel` | 分块 + 共享内存 + `cp.async` 硬件异步拷贝 + 双缓冲 | `test_async_copy` | FP32 |
| L2 消 bank 冲突 | `swizzle_bank_free_gemm_kernel` | XOR Swizzle 坐标变换，零 padding 消除 32-bank 冲突 | `test_bank_conflict` | FP16 |
| L3 Tensor Core 起步 | `fused_fp8_gemm0.cu` (v0) | Inline PTX `mma.sync.m16n8k32` + FP8 在线反量化融合 | `test_gemm_correctness` | FP8→FP32 |
| L4 完整版 | `fused_fp8_gemm.cu` | `ldmatrix(.trans)` 硬件加载 + B 矩阵两相流水线（寄存器预取）+ `__byte_perm` 打包 | `test_fp8_gemm_comparison` | FP8→FP32 |

**项目结构**：

```
tensorcore_fp8_gemm/
├── include/
│   ├── fused_fp8_gemm.cuh        # 完整版 PTX 设施 (mma / ldmatrix / cp.async 封装)
│   └── fused_fp8_gemm0.cuh       # v0 版 PTX 设施 (不含 ldmatrix)
├── src/
│   ├── async_pipeline.cu         # L1: FP32 分块 + cp.async 双缓冲
│   ├── tiled_gemm_bank.cu        # L2: FP16 HGEMM + XOR Swizzle
│   ├── fused_fp8_gemm0.cu        # L3: FP8 mma baseline
│   └── fused_fp8_gemm.cu         # L4: 完整版 (ldmatrix + 两相流水线)
├── tests/                        # 各层级三路对比测试 (naive / cuBLASLt / custom)
├── scripts/
│   ├── run_ncu_profile.sh        # NCU 批量采集 (shape × impl 笛卡尔积)
│   └── analyze_ncu.py            # 报告 CSV 汇总对比
└── docs/
    ├── 2026-07-31.md             # 开发日志 Day 1: CPU vs GPU 存储架构对比
    ├── 2026-08-04.md             # 开发日志 Day 2: 双缓冲流水线 + 坐标映射推演
    ├── 2026-08-12.md             # 开发日志 Day 3: NCU 指标解读 + Swizzle
    ├── 2026-08-23.md             # 开发日志 Day 4-7: mma 指令三天拉锯战
    ├── 四轮测试结果.md            # 四轮 benchmark 的完整终端输出存档
    └── images/                   # Nsight Compute 截图放置目录（空，按 §4.1/§4.2 清单贴图）
        ├── l1_async_ncu.png          # 截图①（对应 §4.1 L1）
        ├── l2_swizzle_ncu.png        # 截图②（对应 §4.1 L2）
        ├── l4_roofline_3way.png      # 截图③（对应 §4.1 L4）
        ├── l4_compute_workload.png   # 截图④（对应 §4.2）
        ├── l4_memory_workload.png    # 截图⑤（对应 §4.2）
        ├── l4_warp_state.png         # 截图⑥（对应 §4.2）
        └── l4_occupancy.png          # 截图⑦（对应 §4.2）
```

**关键实现细节**（以 L4 完整版为例）：

- **Tile 规模** 128×128×64，256 线程/block，8 warp 按 4×2 划分，每 warp 负责 32×64 输出；
- **MMA 指令** `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`：每条指令 16×8×32 块、
  8192 FLOP，A 占 4 个 .b32 寄存器、B 占 2 个、累加器 4 个 .f32；
- **FP8 在线反量化**：Tensor Core 按 FP8 乘累加、以 FP32 存累加器，最后一个 scale 乘法
  （`scale_a × scale_b`）在写回时融合完成，整个 GEMM 不引入单独的 dequant pass；
- **B 矩阵两相流水线**：`prefetch_b_tile`（发射 `ld.global`，不等待）→ 计算当前 tile 的 mma
  → `store_b_tile_to_smem`（`__byte_perm` 重组 + `st.shared.v2`），把全局内存延迟藏进计算；
- **双缓冲**：共享内存开 Stage 0/1 乒乓，`cp.async.wait_group` 控制水线深度。

---

## 3. 探究心路：从看不懂 GPU 到手写 PTX（开发日志摘编）

### 3.1 起点：先弄懂「数据住在哪」（7.31）

项目第一天没有写代码，而是画了两张图：CPU 与 GPU 的存储层次对比。理解了三件事：

- GPU 没有大缓存，而是把**程序员可调配的共享内存**直接放在每个 SM 内部——「越靠近计算核心，
  存储越小、越快」，整个优化史就是不断把数据往里搬的历史；
- ALU 以 32 个为一组，和 warp（32 线程）一一对应——这不是巧合，是 SIMT 的设计本意；
- SM 寄存器总量固定，单线程用得越多，可驻留 warp 越少——这就是 occupancy 权衡，
  后来在 L4 阶段成为实测瓶颈（128 寄存器/线程）。

### 3.2 双缓冲：在纸上把一个线程推演一遍（8.4）

L1 阶段最大的收获不是性能，而是一个方法论教训：**「懂分块矩阵的数学」不等于「写得出无 Bug 的
物理坐标映射」**。数学上分块只是 $C_{ij}=\sum_k A_{ik}B_{kj}$ 划几条虚线；但显存是一维长线、
warp 是一维平铺、共享内存是 32-bank 交叉阵列，三者数量对不上时，`tid/32` 写成 `tid/64`
这类错误 GPU 不报错、只会默默输出垃圾值。解决办法是把一个具体线程
（如 `blockIdx=(1,0), threadIdx=(2,3)`）从「去 A 矩阵第几字节拉数据 → 存到共享内存哪个下标 →
从哪个位置取数计算 → 结果写回 C 的哪个一维地址」在纸上代入数字推演一遍。
这个笨办法后来在 L3、L4 的调试中反复救了我。

### 3.3 第一次 Profiling：让硬件自己说出瓶颈（8.12）

L1 完成后第一次用 NCU 抓数据（见 4.1 节表格）：Long Scoreboard Stall 从 8.51 暴跌到 0.03，
证明 `cp.async` 双缓冲确实把访存延迟藏掉了；但 Achieved Occupancy 从 92.3% 掉到 66.5%——
双缓冲让共享内存翻倍、驻留 block 数减半。**这是一个经典的工程权衡**：延迟已被掩盖后，
高 occupancy 的作用（藏延迟）被流水线替代，牺牲它是值得的。
这一轮学会了「硬件计数器 → 物理归因」的推理方式，后面所有优化决策都基于它。

L2 的 Swizzle 则是纯数学的胜利：共享内存 32 个 bank，同一 warp 若访问同一 bank 的不同地址
会串行化。传统解法是 padding（浪费空间+破坏对齐），我用**按位异或**重排坐标——
把「取同一列」的地址在 bank 维度上打散，零空间开销换取零冲突。

### 3.4 最难的一仗 Part I：L3 mma 在线反量化融合（8.23 日志的 80% 痛苦）

L3 是本项目耗时最长、收获最大的阶段。目标是手写 PTX 调用 Tensor Core，实现
**FP8 输入 → m16n8k32 MMA → FP32 累加** 这一整套在线反量化融合，
省去单独的 dequant pass 以及其中间访存开销。第一天我"读懂"了 CUDA 代码骨架，
三天后才发现：我读的只是语法，根本没有读透**每一条指令所要求的、寄存器内数据的物理字节布局**。

排查方式是把整个数据流从头核对：从 block → tile → 每个 m16n8k32 片段，
逐阶段手写坐标，逐个确认 A、B、C 的数据结构。
最终的**重大突破**不在最后一行补丁，而在三条认知上的推进：

1. **A 片段寄存器上下交错搞反**——A 寄存器 a[1] 与 a[2] 的内容错位（对应 m16n8k32 的
   "右上组" 与 "左下组" 被跳跃式加载时交换了位置）。这类 bug GPU 从不报错、
   只默默输出 garbage，唯一能定位的方法就是把单个 warp 的 32 个线程在一次单独加载后的
   寄存器值、与 PTX 文档 Figure 逐格比对。
2. **16B 向量化缺失边界兼容机制**——从 gmem → smem 时我为了吃满总线，
   一律按 16B `cp.async`；但矩阵 K/N 若非 16 字节倍数时最后一段会读越界或漏零。
   补上了 "非 16B 对齐的边界自动回退至 4B 向量加载 + 谓词零填充" 的分支——
   这个看似小的兼容修复直接让非对齐 shape 的 MaxDiff 从噪声量级降到可控。
3. **明确了 AI 辅助编程的方法论**——此前我一直想靠自己硬啃官方文档和 GitHub 单文件实现，
   进展极慢；卡住一周后把 AI IDE 接入工作流，几天内就把
   「寄存器片段排布」「未对齐的回退分支」「探针 kernel 模板」三件核心脚手架搭出来了。

同时踩过的环境坑记录在此（后来反复出现）：

- **sm_120 架构适配**：曾回退 sm_89 编译，希望让 cuBLAS 用 mma 而非 wgmma 以公平对比，
  结果 `unsupported toolchain`——PTX 无法在更新硬件上 JIT 回旧架构，
  最终 CMake 必须锁定 `CMAKE_CUDA_ARCHITECTURES=120`；
- **NCU 权限**：`ERR_NVGPUCTRPERM` 需 `NVreg_RestrictProfilingToAdminUsers=0` 开放计数器；
- **kernel 名匹配**：NCU 过滤要用真实符号名（如 `fused_fp8_tensor_core_gemm_kernel`、
  cuBLAS 的 `nvjet`），而不是测试代码里打的中文标签。

### 3.5 最难的一仗 Part II：L4 ldmatrix 硬件加载 + B 矩阵两相流水线

L3 虽然正确性通过，但片段加载走的是 `ld.shared` + lambda 拼装字节——
一条 mma 需要的 A 片段要几十条 LD 指令，warp 的有用指令率极低，
喂料远跟不上 Tensor Core 的胃口。所以 L4 的核心方向只有一个：**用硬件加载器替代软件拼装**。

这一阶段引入了两个关键优化：

1. **用 `ldmatrix` 替代 `ld.shared` 从 smem → 寄存器的搬运**。`ldmatrix.x4`
   一条指令就把 A 的 16×32 片段（64 个 FP8）完整装入 warp 的 4×.b32 寄存器，
   数量级地减少了 LSU 发射槽占用。但 B 有两个麻烦：
   - mma 对 B 的输入要求是**按列方向的片段**，而 `ldmatrix` 默认按行加载；
   - `ldmatrix` 原本只为 fp16（16-bit 元素）设计，直接装 FP8（8-bit）会和
     MMA 期望的片段布局错位。
   解决路径：B 矩阵在 smem 内设计为 **K-pair 交错打包**（两个相邻 K 行的同列 FP8
   拼进一个 uint16），然后调用 `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`，
   通过硬件的 `.trans` 标志一次完成「按列取 + 转置 + 匹配 16-bit」三件事。
   代价是：从 gmem 读到 smem 的写入这一步，不能再直接 `cp.async` 直写，
   必须经寄存器走 `__byte_perm` 字节重组 + `st.shared`。
2. **两相流水线（Prefetch-Store Overlap）**。既然 B 的 smem 写入要占用额外的
   `__byte_perm` + `st.shared` 指令时间，那就把原本一步的 B 加载拆成两步：
   - Phase 1 `prefetch_b_tile`：**只发射**若干 `ld.global.b32`（不等返回），
     把 B 的 raw 字节拉进寄存器；
   - Phase 2 `store_b_tile_to_smem`：在**当前 tile 的 mma 已经在跑**、
     gmem 返回的 raw 已经就绪之后，执行 `__byte_perm` 打包 + `st.shared` 写入下阶段 smem。
   思路和第一板块的 `cp.async` 双缓冲同源——只是把 "异步" 从硬件单元接管变成
   软件手工管水线；作用都是把 gmem 延迟藏进计算。

这个阶段真正卡到最后一天的 bug 也刚好落在 `__byte_perm` 上：
K-pair 打包选择子写成了 `0x7632`（MSB-first nibble 序），导致 B 矩阵
n=2,3,6,7… 列的高字节全错位，MaxDiff≈0.003——**最后才用独立探针 kernel**
（只跑一段 `__byte_perm`，喂已知模式、打印字节）逐位验证，
把选择子从 `0x7632` 修正为 `0x7362`（LSB-first nibble 序），
一行代码修复，5 组 shape MaxDiff 全部归零。

### 3.6 方法论沉淀（调试 + AI 工作流）

**调试方法论三条**：

- **探针隔离验证**：怀疑哪条指令就单独写一个最小 kernel 验证它，绝不在大工程里盲猜；
- **控制变量对比**：不同测试程序的绝对 TFLOPS **不可直接比较**（计时、warmup、冷缓存策略不同）。
  我曾误以为 L4 比 L3 慢（62.75 vs 59.31），实际上两个程序里 cuBLAS 也同步下降了 6~7%——
  用 cuBLAS 归一化后，L4（53.5%）> L3（52.8%），是净提升。此后所有对比固定同一程序、
  同一 shape、同一轮；
- **数学先行**：算术强度、寄存器占用、bank 分布都可以先算后写。L4 的屋顶图分析
  （见 4.2 节）在写代码之前就预言了「这是 compute-bound 问题，优化方向是喂料而非搬料」。

**AI 辅助编程四条原则**（L3 卡住一周后才明确，极大提高了后续效率）：

1. **关键性知识不能外包**：底层框架知识、优化策略、数据流逻辑仍必须自己学会并理解。
   AI 帮你写的 `__byte_perm` 选择子如果自己不理解 nibble 序，bug 出现时根本识别不了。
2. **工作流分四层，逐层下放 AI**：
   - ① 顶层方向 / 大项目框架 / 解决策略：自己定（结合大模型推理优化岗的知识体系）；
   - ② 把方向细化为具体的实现步骤、模块流程、接口约定（自己先拉出骨架）；
   - ③ 用明确的框架和流程，让 AI 做**生成、补全、静态检查、测试用例书写**这类
     机械化但易错的部分；
   - ④ 对关键硬件指令（mma/ldmatrix/__byte_perm）的输出结果，
     必须回到 探针 kernel + 独立打印 做人工复核，不能信 AI 给的选择子。
3. **省钱策略**：对话缓存命中（同一话题不反复重开）能显著节省 token；
   方向型问题用轻量模型先跑一遍筛；硬核实现/代码审查再上重模型。
4. **模型搭配**：日常实现主力用已购买的 Mimo（Agent/对话分离 + 自定义提示词工程），
   知识补全和 PTX 文档交叉验证用 Gemini；后续流程里先让 Mimo 明确方向，
   再让它基于方向生成实现，比来回问 "该怎么做" 更省、更准。

---

## 4. 数据分析

> 口径说明：各轮使用不同测试程序与数据类型（L1 为 FP32、L2 为 FP16、L3/L4 为 FP8），
> TFLOPS 绝对值跨轮不可直接比；本文同时给出「vs 上一轮同程序加速比」与
> 「vs cuBLASLt 归一化百分比」两条线索。cuBLASLt 同为 FP8 时是唯一公平的上限标尺。

### 4.1 纵向对比：五层逐级优化

#### L0 → L1：cp.async 双缓冲（消除访存等待）

FP32 GEMM，`test_async_copy`：

| Shape | Naive (TFLOPS) | +双缓冲 (TFLOPS) | 加速比 | cuBLAS FP32 (TFLOPS) | 达 cuBLAS |
|---|---|---|---|---|---|
| 512³ | 1.223 | 1.518 | 1.24× | 6.806 | 22.3% |
| 1024³ | 1.383 | 1.614 | 1.17× | 10.806 | 14.9% |
| 2048³ | 1.604 | 1.843 | 1.15× | 12.909 | 14.3% |
| 4096³ | 1.458 | 1.859 | 1.27× | 11.851 | 15.7% |

**NCU 归因**（核心四指标）：

| 指标 | Naive | 双缓冲 | 变化 | 物理含义 |
|---|---|---|---|---|
| Long Scoreboard Stall | 8.51 | 0.03~0.04 | **-99.5%** | 等 global 内存的挂起几乎归零，延迟掩盖成功 |
| DRAM Throughput | 1.96% | 2.73% | +39.3% | 向量化对齐访存减少了请求碎片 |
| Achieved Occupancy | 92.3% | 66.5% | -25.8% | 双缓冲 smem 翻倍 → 驻留 block 减半（主动权衡） |
| SM Throughput | 91.3% | 85.3% | -6.6% | 指令结构变化，但有效吞吐更高 |


**结论**：瓶颈性质变了——从「等数据」变成「算不过来」，后续优化空间转向计算侧与共享内存侧。

---

#### L1 → L2：XOR Swizzle（消除 bank conflict）

FP16 HGEMM，`test_bank_conflict`，512³（本轮仅测了此 shape）：

| 实现 | 耗时 | TFLOPS | 加速比 | 达 cuBLAS(21.877) |
|---|---|---|---|---|
| Naive（有 bank conflict） | 0.248 ms | 1.084 | 1× | 4.96% |
| **Swizzle（bank-free）** | 0.057 ms | **4.680** | **4.32×** | 21.4% |

**NCU 归因**：Duration 245 µs → 55.1 µs；Swizzle 算子 Compute/Memory Throughput 61.37%，
而 Naive 的 SM 单元利用率高达 90.5% 却慢 4.45 倍——典型的「忙而无效」：bank conflict 把
共享内存流水线串行化，SM 看起来满负荷，实际全耗在重试的访存指令上。寄存器 34 → 125
（Swizzle 集成了异步流水线与展开）。

**结论**：同等算法下，仅改变共享内存坐标映射就拿到 4.3 倍——「硬件物理通道错位」是 CUDA
性能里最隐蔽也最值得系统学习的一类问题。

---

#### L2 → L3：Tensor Core + FP8 在线反量化（本质性跃迁）

FP8 GEMM，`test_gemm_correctness`（v0 实现，B 矩阵 CPU 预转置方案）：

| Shape | Naive (TFLOPS) | v0 mma FP8 (TFLOPS) | vs Naive | cuBLASLt FP8 (TFLOPS) | 达 cuBLAS |
|---|---|---|---|---|---|
| 512³ | 1.13 | 15.65 | 13.9× | 26.24 | **59.6%** |
| 2048³ | 1.22 | 62.75 | **51.4×** | 118.82 | **52.8%** |

这是全程最大的一次跃迁（+27 倍）。原因不在工程技巧而在**计算单元的代际差**：
一条 `mma.sync.m16n8k32` 在一个 warp 内完成 8192 FLOP，密度是 CUDA Core FMA 的约三个数量级；
同时 FP8 输入 + FP32 累加的设计（在线反量化）保证了精度无损——三路 MaxDiff 全部为 0。
此轮之后，「达 cuBLAS 五成」成为基准线，剩余差距从「算力代差」收敛为「喂料工程」。

---

#### L3 → L4：ldmatrix + 两相流水线（压榨指令效率）

完整版，`test_fp8_gemm_comparison`（naive / cuBLASLt / 完整版同程序同轮对比）：

| Shape | Naive (TFLOPS) | cuBLASLt FP8 (TFLOPS) | **完整版 (TFLOPS)** | 达 cuBLAS | vs Naive |
|---|---|---|---|---|---|
| 512³ | 1.064 | 25.799 | **15.439** | 59.8% | 14.5× |
| 1024³ | — | 78.568 | **35.697** | 45.4% | — |
| 2048×2048×1024 | — | 92.916 | **53.819** | 57.9% | — |
| 2048³ | — | 110.834 | **59.310** | 53.5% | — |

L3→L4 的改动与效果（同为 FP8，跨程序用 cuBLAS 归一化对比）：

| 改动 | 解决的问题 | 效果证据 |
|---|---|---|
| `ldmatrix.x4` 加载 A 片段 | 消除逐 `ld.shared` 拼装的 LSU 指令开销 | 一条指令替代数十条 |
| `ldmatrix.x2.trans` 加载 B | 硬件转置替代 CPU 预转置，去掉全局内存的额外 pass | 端到端管线简化 |
| B 两相流水线（寄存器预取） | 主循环中 global 延迟暴露 | 预取发射后紧接 mma，延迟被计算掩盖 |
| `__byte_perm` K-pair 打包 | 适配 mma 输入字节序（0x7362 修复后全 shape 精确） | MaxDiff 0.00000 |

**一个诚实的对比陷阱**（值得写进任何性能报告）：L4 在 2048³ 的绝对值 59.31 低于 L3 的 62.75，
曾让我误判为"优化负收益"。深挖后发现是**测试程序差异**：两个程序里 cuBLAS 也同步从 118.8
降到 110.8（-6.7%）。归一化后 L3=52.8%、L4=53.5%，L4 确实更快。另外，边际分析显示
K 从 1024→2048 时增量吞吐约 **74 TFLOPS**（Δ8.59 GFLOP / Δ116 µs），远高于平均值 59.3——
说明固定开销（launch、首个 tile 的同步等待）仍在稀释短 K 场景的平均值，这为后续
「加深流水线摊薄固定开销」提供了直接依据。

### 4.2 横向对比：完整版 vs cuBLASLt vs Naive 全指标解剖

采集条件：`ncu --set basic`，shape M=2048 N=2048 K=1024，各算子单独 kernel 采样。

| 指标 | Naive | 完整版 (custom) | cuBLASLt (nvjet) |
|---|---|---|---|
| TFLOPS（benchmark 同 shape） | ~1.1 | **53.8** | 92.9 |
| Compute (SM) Throughput | 99.07% | **72%** | 65.20% |
| Memory Throughput (SOL) | 9.07% | **51.7%**（瓶颈在 L1/SMEM 层） | 56.86% |
| DRAM Throughput | 0.43% | 19.6% | 32.81% |
| Achieved Occupancy | **97%** | **31.5%** | 16.10% |
| 寄存器 / 线程（ptxas 输出） | 34 | **128** | — |
| Warp stall 第一主因 | 【访存延迟类】 | **math_pipe_throttle**（等 Tensor Core 排队） | **math_pipe_throttle**（等 Tensor Core 排队）|
| Tensor 物理管线利用率 | ~0 | **~30%** | ~65% |
| 数据搬运路径 A | gmem 逐元素 | cp.async（gmem→smem 直写） | cp.async / TMA |
| 数据搬运路径 B | gmem 逐元素 | ld.global→寄存器→__byte_perm→st.shared | **TMA 直写 smem** |
| 流水线深度 | 无 | **2 级** | **~6 级** |
| 同步机制 | `__syncthreads` | 每 tile 一次 `__syncthreads` | 细粒度 mbarrier |

> 🔲 **贴图位置**：`docs/images/l4_compute_workload.png`
>  custom 与 cublas 的 Compute Workload Analysis 对比页

> 🔲 **贴图位置**：`docs/images/l4_memory_workload.png`
> Memory Workload Analysis，突出 Memory SOL 51.7%（L1/SMEM 层瓶颈）与 DRAM 19.6% 的分层差。

> 🔲 **贴图位置**：`docs/images/l4_warp_state.png`
> Warp State Statistics / Warp Stall Reasons 的条形对比，突出 `stall_math_pipe_throttle` 占比。

> 🔲 **贴图位置**：`docs/images/l4_occupancy.png`
> Occupancy 页：理论 vs 实际 occupancy 对比
**具体差距归因**（53.8 vs 92.9，差 42%）：

1. **B 矩阵加载路径差一个量级的指令效率**（最大单项）。我的 B 走
   `ld.global → 寄存器 → __byte_perm → st.shared` 四步，每字节都消耗 LSU 发射槽；
   cuBLAS 用 **TMA**（`cp.async.bulk.tensor`）一条指令完成 gmem→smem 直写，零寄存器中转。
   这直接体现在 Memory SOL 51.7%——我的瓶颈卡在 L1/共享内存层而非 DRAM 层（19.6%）。
2. **流水线深度 2 vs ~6 级**。等待 Tensor Core 的 warp（stall_math_pipe_throttle 主导）说明
   「算不过来」，但 Tensor 管线利用率仅 ~30%——两者并存的原因是：mma 指令之间依赖链长、
   喂料指令（ldmatrix/perm/st）与 mma 争抢发射槽，队列有 warp 在排但排进的是别人的指令。
   cuBLAS 用 6 级流水把喂料完全前置。
3. **同步粒度**：每 tile 一次 `__syncthreads()` 全 block 栅栏 vs mbarrier 按生产者/消费者
   细粒度通知，后者消除 tile 边界的空泡周期。
4. **Occupancy 31.5% vs 更高**：128 寄存器/线程 × 256 线程 + 37376 B smem 使每 SM 只能驻留
   2 个 block，warp 调度器可用来填气泡的候选 warp 少。

**屋顶图解读**（自我验证的正确性）：算术强度 = 2MNK / (A+B+C 流量) ≈ 8.6 GFLOP / 20 MB ≈ 430
FLOP/Byte，机器平衡点 ≈ 97 TFLOPS ÷ 450 GB/s ≈ 215 FLOP/Byte，430 ≫ 215 → **天生
compute-bound**。所以 DRAM 19.6% 不是问题（分块复用把显存流量压得很低，这正是 GEMM 分块的
意义）；真正要修的是 L1/SMEM 层的 51.7%。naive 是反面对照：occupancy 97% 却只有 ~1.1
TFLOPS——高 occupancy 只代表「有很多 warp 在排队」，不代表排的是有用的指令。

---

## 5. 下一步优化方向（面向大模型推理优化实习）

按「预期收益 × 与推理业务的相关性」排序：

1. **TMA 替代 B 矩阵两相流水线**（最大单项收益）
   用 `cuTensorMapEncodeTiled` + `cp.async.bulk.tensor` 直写 smem，砍掉
   `ld.global→__byte_perm→st.shared` 路径，直击 Memory SOL 51.7% 的 L1 层瓶颈。
   这也是 cuBLAS（nvjet）的核心做法——复刻它是理解官方算子设计哲学的最短路径。
2. **流水线 2 → 4/6 级 + mbarrier 细粒度同步**
   消除 tile 边界 `__syncthreads` 空泡，进一步摊薄固定开销（边际分析已证明短 K 场景固定开销
   占比高，而 LLM 推理 decode 阶段恰恰是小 batch、短 K 场景）。
3. **Warp Specialization（生产者/消费者分离）**
   部分 warp 专职搬运、部分专职计算，模仿 nvjet 的 warp 角色划分；配合 persistent kernel
   减少 launch 开销。
4. **对接 block-scaled FP8（mxfp8）**
   sm_120a 支持 `kind::mxf8f6f4.block_scale` MMA（DeepSeek 等训练/推理使用的微缩放格式）。
   把当前 per-tensor scale 升级为 per-block scale，直接对接当前大模型 FP8 量化的前沿实践。
5. **工程化与融合**
   PyTorch C++ Extension 封装 → 接入推理 benchmark（与 vLLM 的 kernel 对比）；
   epilogue 融合 bias/GELU；延伸到 FlashAttention 式的 GEMM+Softmax 融合——
   这是从「写 kernel」到「做推理优化」的关键一跳。

**认知总结**：L4 阶段我曾系统调研过 MMA 指令形状——FP8 在经典 mma 下只有 m16n8k16/k32 两种
选择，k32 每条指令 8192 FLOP 已是最优（k16 是负优化；更大的形状属于 int4/b1 或数据中心
tcgen05，消费级硬件不存在）。**指令形状这条线已经到头，剩余差距全部在数据搬运与调度**——
这个「先证明当前选择最优、再定位真正瓶颈」的过程，比单纯刷高 TFLOPS 更接近推理优化岗的
日常工作方式。

---

## 6. 复现指南

```bash
# 编译 (RTX 5060 需 sm_120，见 CMakeLists.txt CMAKE_CUDA_ARCHITECTURES=120)
cd tensorcore_fp8_gemm && cmake -B build && cmake --build build -j$(nproc)

# 三路对比 benchmark（naive / cuBLASLt / custom）
./build/test_fp8_gemm_comparison --shape 2048,2048,2048

# NCU 采集（需先开放计数器权限）
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
CUDA_VISIBLE_DEVICES=0 ncu --set basic \
  -k regex:fused_fp8_tensor_core_gemm_kernel \
  -o custom_report ./build/test_fp8_gemm_comparison --profile --impl custom --shape 2048,2048,1024

# 批量采集 + CSV 汇总（shape × impl 笛卡尔积）
./scripts/run_ncu_profile.sh
```

---