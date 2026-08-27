/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#ifdef COMPILE_SIMD_RISCV_RVV

#include <riscv_vector.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {

// ===========================================================================
// 函数：compute_PQ_dis_tables_dsub2 的 RVV 实现（PQ 距离表计算）
// ---------------------------------------------------------------------------
// 功能：为每个查询向量 x 预先计算它与每个子段(dsub=2)所有 ksub 个码本中心
//   的距离(或内积)，存成查找表 dis_tables，供 PQ 扫描时直接查表复用。
// 输出布局：dis_tables[i*M*ksub + m*ksub + k] = 第 i 个查询向量第 m 段
//   与第 k 号中心的距离。
// 优化手段(三点)：
//   1) SOA 数据重排：把交错布局 [c0,c1,c0,c1,...] 的中心拆成 c0_all/c1_all
//      两个连续数组，使每段第 0 维/第 1 维分别连续，向量 load 无需跨步。
//   2) RVV 向量化：e32m1(VLMAX=8@VLEN128)，一次算 8 个中心的距离。
//   3) M 维 unroll x2：同时处理相邻两段 m、m+1，交错发射独立计算提升 ILP、
//      掩盖访存延迟。
// 代价：每次调用多一次 M*ksub 数组拆分拷贝 + 两份临时内存；nx 较大时摊销。
// ===========================================================================
template <>
void compute_PQ_dis_tables_dsub2<SIMDLevel::RISCV_RVV>(
        size_t d,                    // 原始向量维度
        size_t ksub,                 // 每个子段的码本中心数
        const float* all_centroids,  // 所有段中心，交错布局(每中心占 2 个连续 float)
        size_t nx,                   // 查询向量个数
        const float* x,              // 查询向量，行优先 x[i*d + dim]
        bool is_inner_product,       // true=算内积，false=算 L2 平方距离
        float* dis_tables) {         // 输出距离表
    size_t M = d / 2;                // 子段数 = 总维度 / 子段维度(2)
    FAISS_THROW_IF_NOT(ksub % 8 == 0); // e32m1 VLMAX=8，要求 ksub 是 8 的倍数，避免尾巴

        // --- 质心解交错(deinterleave) ---
        // 原始质心两维交错存放 [c0,c1,c0,c1,...]，不利于连续 load。
        // 拆成两个分离数组：cd0_buf 存所有 dim0，cd1_buf 存所有 dim1，
        // 之后就能用连续 vle32 一次取 8 个 dim0 / 8 个 dim1。
    float* c0_all = new float[M * ksub];  // 所有段中心的第 0 维
    float* c1_all = new float[M * ksub];  // 所有段中心的第 1 维
    for (size_t m = 0; m < M; m++) {
        const float* cm = all_centroids + m * ksub * 2;  // 第 m 段的中心起始
        float* c0m = c0_all + m * ksub;                  // 第 m 段第 0 维输出
        float* c1m = c1_all + m * ksub;                  // 第 m 段第 1 维输出
        for (size_t k = 0; k < ksub; k++) {
            c0m[k] = cm[2 * k];       // 第 k 号中心的第 0 维
            c1m[k] = cm[2 * k + 1];   // 第 k 号中心的第 1 维
        }
    }

    size_t vl = __riscv_vsetvl_e32m1(ksub); // 设 VL(SEW=32,LMUL=1)，VLEN=128 时=8

    if (is_inner_product) {
        // --- 内积分支：表项 = x0*c0 + x1*c1（x 与中心的点积） ---  unroll x2
        for (size_t i = 0; i < nx; i++) {
            const float* xi = x + i * d;                    // 第 i 个查询向量
            float* oi = dis_tables + i * M * ksub;          // 第 i 个查询的输出表
            for (size_t m = 0; m + 1 < M; m += 2) {         // M 维 unroll x2，处理相邻两段
                float x0a = xi[2 * m], x1a = xi[2 * m + 1];          // 段 m 的二维
                float x0b = xi[2 * (m + 1)], x1b = xi[2 * (m + 1) + 1]; // 段 m+1 的二维
                const float* c0a = c0_all + m * ksub;        // 段 m 第 0 维数组
                const float* c1a = c1_all + m * ksub;        // 段 m 第 1 维数组
                const float* c0b = c0_all + (m + 1) * ksub;  // 段 m+1 第 0 维数组
                const float* c1b = c1_all + (m + 1) * ksub;  // 段 m+1 第 1 维数组
                float* oa = oi + m * ksub;                   // 段 m 的输出
                float* ob = oi + (m + 1) * ksub;             // 段 m+1 的输出
                for (size_t k = 0; k < ksub; k += vl) {      // 每次处理 vl 个中心
                    // 段 m：ra = vc0*x0a + x1a*vc1（8 个中心的点积）
                    vfloat32m1_t vc0 = __riscv_vle32_v_f32m1(c0a + k, vl); // 载入 8 个 c0
                    vfloat32m1_t vc1 = __riscv_vle32_v_f32m1(c1a + k, vl); // 载入 8 个 c1
                    vfloat32m1_t ra = __riscv_vfmacc_vf_f32m1(
                            __riscv_vfmul_vf_f32m1(vc0, x0a, vl), x1a, vc1, vl);
                    //   └─ vfmul：vc0*x0a；vfmacc：acc + x1a*vc1 → x0a*c0+x1a*c1
                    __riscv_vse32_v_f32m1(oa + k, ra, vl);   // 写回段 m

                    // 段 m+1：同上，复用 x0b/x1b
                    vc0 = __riscv_vle32_v_f32m1(c0b + k, vl);
                    vc1 = __riscv_vle32_v_f32m1(c1b + k, vl);
                    vfloat32m1_t rb = __riscv_vfmacc_vf_f32m1(
                            __riscv_vfmul_vf_f32m1(vc0, x0b, vl), x1b, vc1, vl);
                    __riscv_vse32_v_f32m1(ob + k, rb, vl);   // 写回段 m+1
                }
            }
        }
    } else {
        // --- L2 平方距离分支：表项 = (x0-c0)² + (x1-c1)² ---
        for (size_t i = 0; i < nx; i++) {
            const float* xi = x + i * d;
            float* oi = dis_tables + i * M * ksub;
            for (size_t m = 0; m + 1 < M; m += 2) {
                float x0a = xi[2 * m], x1a = xi[2 * m + 1];
                float x0b = xi[2 * (m + 1)], x1b = xi[2 * (m + 1) + 1];
                const float* c0a = c0_all + m * ksub;
                const float* c1a = c1_all + m * ksub;
                const float* c0b = c0_all + (m + 1) * ksub;
                const float* c1b = c1_all + (m + 1) * ksub;
                float* oa = oi + m * ksub;
                float* ob = oi + (m + 1) * ksub;
                for (size_t k = 0; k < ksub; k += vl) {
                    // 段 m：r = (c0-x0a)² + (c1-x1a)²
                    vfloat32m1_t vc0 = __riscv_vle32_v_f32m1(c0a + k, vl);
                    vfloat32m1_t vc1 = __riscv_vle32_v_f32m1(c1a + k, vl);
                    vfloat32m1_t d0 = __riscv_vfsub_vf_f32m1(vc0, x0a, vl); // d0 = c0 - x0a
                    vfloat32m1_t d1 = __riscv_vfsub_vf_f32m1(vc1, x1a, vl); // d1 = c1 - x1a
                    d0 = __riscv_vfmul_vv_f32m1(d0, d0, vl);                // d0 = d0²
                    d1 = __riscv_vfmul_vv_f32m1(d1, d1, vl);                // d1 = d1²
                    vfloat32m1_t r = __riscv_vfadd_vv_f32m1(d0, d1, vl);    // r = d0 + d1
                    __riscv_vse32_v_f32m1(oa + k, r, vl);   // 写回段 m
                    
                    // 段 m+1：同上，复用 x0b/x1b
                    vc0 = __riscv_vle32_v_f32m1(c0b + k, vl);
                    vc1 = __riscv_vle32_v_f32m1(c1b + k, vl);
                    d0 = __riscv_vfsub_vf_f32m1(vc0, x0b, vl);
                    d1 = __riscv_vfsub_vf_f32m1(vc1, x1b, vl);
                    d0 = __riscv_vfmul_vv_f32m1(d0, d0, vl);
                    d1 = __riscv_vfmul_vv_f32m1(d1, d1, vl);
                    r = __riscv_vfadd_vv_f32m1(d0, d1, vl);
                    __riscv_vse32_v_f32m1(ob + k, r, vl);   // 写回段 m+1
                }
            }
        }
    }

    delete[] c0_all;   // 释放临时数组
    delete[] c1_all;
}


template <>
float fvec_norm_L2sqr<SIMDLevel::RISCV_RVV>(const float* x, size_t d) {
    float res = 0.0f;
    size_t i = 0;
    for (; i < d;) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vxx = __riscv_vfmul_vv_f32m8(vx, vx, vl);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m8_f32m1(
                vxx, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        res += __riscv_vfmv_f_s_f32m1_f32(vred);
        i += vl;
    }
    return res;
}

// ===========================================================================
// fvec_L2sqr_ny — non-transposed (row-major y) squared-L2 distance.
//   dis[i] = sum_{j=0}^{d-1} (x[j] - y[i*d + j])^2,  for i in [0, ny)
// y is row-major: the i-th vector occupies y[i*d .. i*d+d-1].
// RVV strategy: vectorize over ny with strided loads. For each dim j, gather
// the j-th component of `chunk` consecutive y vectors via vlse32 (stride =
// d*sizeof(float)), then diff = y_vec - x[j], acc += diff*diff. A single
// accumulator and one strided load + two vector ops per dim keeps register
// pressure low, allowing e32m4 (VLMAX=16 on VLEN=128) as the starting LMUL.
// This is fully general in d and ny (no experiment-parameter special-casing).
// ===========================================================================
template <>
void fvec_L2sqr_ny<SIMDLevel::RISCV_RVV>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    const size_t chunk = 32; // e32m8: VLMAX = LMUL*VLEN/SEW = 8*128/32 = 32
    const ptrdiff_t stride_bytes = (ptrdiff_t)d * (ptrdiff_t)sizeof(float);

    if (ny >= chunk) {
        (void)__riscv_vsetvl_e32m8(chunk);
        for (; i + chunk <= ny; i += chunk) {
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, chunk);
            const float* yb = y + i * d; // base of this chunk
            for (size_t j = 0; j < d; j++) {
                vfloat32m8_t y_vec =
                        __riscv_vlse32_v_f32m8(yb + j, stride_bytes, chunk);
                vfloat32m8_t diff = __riscv_vfsub_vf_f32m8(y_vec, x[j], chunk);
                acc = __riscv_vfmacc_vv_f32m8(acc, diff, diff, chunk);
            }
            __riscv_vse32_v_f32m8(dis + i, acc, chunk);
        }
    }

    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        const float* yb = y + i * d;
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vlse32_v_f32m8(yb + j, stride_bytes, vl);
            vfloat32m8_t diff = __riscv_vfsub_vf_f32m8(y_vec, x[j], vl);
            acc = __riscv_vfmacc_vv_f32m8(acc, diff, diff, vl);
        }
        __riscv_vse32_v_f32m8(dis + i, acc, vl);
    }
}


template <>
float fvec_L2sqr<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_L2sqr<SIMDLevel::NONE>(x, y, d);
}

template <>
float fvec_inner_product<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_inner_product<SIMDLevel::NONE>(x, y, d);
}

template <>
float fvec_L1<SIMDLevel::RISCV_RVV>(const float* x, const float* y, size_t d) {
    return fvec_L1<SIMDLevel::NONE>(x, y, d);
}

template <>
float fvec_Linf<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_Linf<SIMDLevel::NONE>(x, y, d);
}

template <>
void fvec_inner_product_batch_4<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_inner_product_batch_4<SIMDLevel::NONE>(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

template <>
void fvec_L2sqr_batch_4<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_L2sqr_batch_4<SIMDLevel::NONE>(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

// ===========================================================================
// 函数二：fvec_L2sqr_ny_transposed 的 RVV 实现
// ---------------------------------------------------------------------------
// 功能：计算 1 个查询 x 与 ny 个"转置存储"的向量 y 的平方 L2 距离。
//   转置布局：y[i + j*d_offset] = 第 i 个 y 向量的第 j 维(按维度优先存放)。
//   y_sqlen[i] 已预先存好第 i 个 y 向量的平方长度 ‖yᵢ‖²。
// 数学展开式(避免逐维相减)：
//   ‖x − yᵢ‖² = ‖x‖² + ‖yᵢ‖² − 2·(x·yᵢ)
//   → dis[i]   = x_sqlen + y_sqlen[i] − 2·Σ_j x[j]·y[i + j*d_offset]
// RVV 策略：选 e32m8(VLMAX=32)，一次处理 32 个 y 向量。
//   因 d_offset 通常=ny，取"32 个向量的同一维 j"实际是连续 32 个元素 →
//   用连续 vle32 即可，无需跨步 load；j 循环内仅 acc 一个活跃向量，
//   寄存器压力极小，可放心用最大 LMUL 换最高并行度。
// ===========================================================================
template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::RISCV_RVV>(
        float* dis,            // 输出：ny 个平方 L2 距离
        const float* x,        // 单个查询向量，d 维
        const float* y,        // ny 个数据库向量，转置布局
        const float* y_sqlen,  // 预存的 ny 个 y 向量平方长度
        size_t d,              // 维度
        size_t d_offset,       // y 在维度方向的步长(通常=ny)
        size_t ny) {           // y 向量个数
    // Compute squared length of query subvector
    // --- 第 1 步：标量循环算查询 x 的平方长度 ‖x‖²，只算一次 ---
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    // Strip-mine ny dimension with e32m8 (VLMAX=32 on VLEN=128)
    // --- 第 2 步：沿 ny 维度做 strip-mining，每块 32 个向量 ---
    size_t i = 0;
    const size_t chunk = 32; // e32m8: VLMAX = LMUL*VLEN/SEW = 8*128/32 = 32

    if (ny >= chunk) {
        (void)__riscv_vsetvl_e32m8(chunk); // 主体用固定 VL=32
        for (; i + chunk <= ny; i += chunk) {
            // acc = x_sqlen + y_sqlen[i..i+31]
            // 初值 = ‖yᵢ‖² + ‖x‖²，即 y_sqlen 这 32 个值各加 x_sqlen
            vfloat32m8_t acc = __riscv_vle32_v_f32m8(y_sqlen + i, chunk); // 取 32 个 ‖yᵢ‖²
            acc = __riscv_vfadd_vf_f32m8(acc, x_sqlen, chunk);            // 各加 ‖x‖²

            // acc += (-2 * x[j]) * y[j*d_offset + i..j*d_offset + i+31]
            // 维度循环：逐维累加 −2·x[j]·y_vec，等价于减去 2·(x·yᵢ)
            for (size_t j = 0; j < d; j++) {
                // 取 32 个 y 向量的第 j 维(d_offset=ny 时为连续 load)
                vfloat32m8_t y_vec =
                        __riscv_vle32_v_f32m8(y + j * d_offset + i, chunk);
                // acc += (−2·x[j]) · y_vec  → 融合乘加，符号已并入系数
                acc = __riscv_vfmacc_vf_f32m8(
                        acc, -2.0f * x[j], y_vec, chunk);
            }

            __riscv_vse32_v_f32m8(dis + i, acc, chunk); // 写回 32 个距离结果
        }
    }

    // Tail: process remaining ny % 32 elements
    // --- 第 3 步：尾部，处理 ny % 32 个剩余向量，用动态 VL ---
    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i); // 按剩余元素数设 VL(≤32)

        vfloat32m8_t acc = __riscv_vle32_v_f32m8(y_sqlen + i, vl); // 取剩余 ‖yᵢ‖²
        acc = __riscv_vfadd_vf_f32m8(acc, x_sqlen, vl);            // 各加 ‖x‖²

        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vle32_v_f32m8(y + j * d_offset + i, vl); // 取剩余向量的第 j 维
            acc = __riscv_vfmacc_vf_f32m8(
                    acc, -2.0f * x[j], y_vec, vl); // acc += (−2·x[j])·y_vec
        }

        __riscv_vse32_v_f32m8(dis + i, acc, vl); // 写回剩余距离
    }
}

// ===========================================================================
// 函数三：fvec_inner_products_ny 的 RVV 实现
// ---------------------------------------------------------------------------
// 功能：计算 1 个查询 x 与 ny 个"行优先存储"的向量 y 的内积。
//   行优先布局：第 i 个 y 向量占 y[i*d .. i*d+d-1]。
//   核心公式：ip[i] = Σ_{j=0..d−1} x[j] · y[i*d + j]
// RVV 策略：选 e32m4(VLMAX=16)，一次处理 16 个 y 向量。
//   因 y 行优先，要"同时取 16 个向量的第 j 维"，必须用跨步 load vlse32，
//   跨步 = 一个完整向量长度 = d 个 float = d*4 字节。
//   注：跨步 load 吞吐有限，LMUL sweep 后 e32m4 是 strided load 吞吐与
//   寄存器效率的最佳平衡(m8 收益递减、m2 并行度不足)。
// ===========================================================================
template <>
void fvec_inner_products_ny<SIMDLevel::RISCV_RVV>(
        float* ip,        // 输出：ny 个内积结果
        const float* x,   // 单个查询向量，d 维
        const float* y,   // ny 个数据库向量，行优先
        size_t d,         // 维度
        size_t ny) {      // y 向量个数
    // Strip-mine ny dimension with e32m4 (VLMAX=16 on VLEN=128)
    // e32m4 proved optimal in LMUL sweep: best balance of strided-load
    // throughput (vlse32 stride=d*4) and register file efficiency
    // Core formula: ip[i] = sum_{j=0}^{d-1} x[j] * y[i*d + j]
    size_t i = 0;
    const size_t chunk = 16; // e32m4: VLMAX = LMUL*VLEN/SEW = 4*128/32 = 16
    // 跨步字节步长 = 一个 y 向量占的字节数；vlse32 的 stride 以字节为单位
    const ptrdiff_t stride_bytes = d * sizeof(float);

    if (ny >= chunk) {
        (void)__riscv_vsetvl_e32m4(chunk); // 主体用固定 VL=16
        for (; i + chunk <= ny; i += chunk) {
            // 16 个累加器清零(广播标量 0.0f 到整个向量)
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, chunk);

            // 维度循环：逐维累加 x[j] * (16 个 y 向量的第 j 维)
            for (size_t j = 0; j < d; j++) {
                // 跨步 load：从 y+j 出发，每隔 stride_bytes 取一个 float，
                // 共取 16 个 → 正好是 16 个行优先向量的第 j 维
                vfloat32m4_t y_vec = __riscv_vlse32_v_f32m4(
                        y + j, stride_bytes, chunk);
                // acc += x[j] · y_vec  → 16 个向量各自的内积分量累加
                acc = __riscv_vfmacc_vf_f32m4(
                        acc, x[j], y_vec, chunk);
            }

            __riscv_vse32_v_f32m4(ip + i, acc, chunk); // 写回 16 个内积结果
        }
    }

    // Tail: process remaining ny % chunk elements
    // --- 尾部：处理 ny % 16 个剩余向量，用动态 VL ---
    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m4(ny - i); // 按剩余元素数设 VL(≤16)

        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl); // 累加器清零

        for (size_t j = 0; j < d; j++) {
            // 跨步 load 剩余向量的第 j 维
            vfloat32m4_t y_vec = __riscv_vlse32_v_f32m4(
                    y + j, stride_bytes, vl);
            acc = __riscv_vfmacc_vf_f32m4(
                    acc, x[j], y_vec, vl); // acc += x[j] · y_vec
        }

        __riscv_vse32_v_f32m4(ip + i, acc, vl); // 写回剩余内积
    }
}



template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

template <>
void fvec_madd<SIMDLevel::RISCV_RVV>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    fvec_madd<SIMDLevel::NONE>(n, a, bf, b, c);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::RISCV_RVV>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin<SIMDLevel::NONE>(n, a, bf, b, c);
}

#define DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(metric)                 \
    template <>                                                     \
    float VectorDistance<metric, SIMDLevel::RISCV_RVV>::operator()( \
            const float* x, const float* y) const {                 \
        return VectorDistance<metric, SIMDLevel::NONE>(             \
                this->d, this->metric_arg)(x, y);                   \
    }

DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_L2)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_INNER_PRODUCT)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_L1)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Linf)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Lp)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Canberra)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_BrayCurtis)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_JensenShannon)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Jaccard)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_NaNEuclidean)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_GOWER)

#undef DEFINE_VECTOR_DISTANCE_RVV_FALLBACK

} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::RISCV_RVV
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#endif // COMPILE_SIMD_RISCV_RVV
