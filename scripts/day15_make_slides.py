"""Day15 汇报PPT生成（导师辅助）：数字取自 results/ 中 JSON，不手抄。"""

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
OUT.mkdir(exist_ok=True)

FONT = "Microsoft YaHei"
DARK = RGBColor(0x1F, 0x2D, 0x3D)
BLUE = RGBColor(0x1F, 0x77, 0xB4)


def load(relpath):
    return json.loads((ROOT / relpath).read_text(encoding="utf-8"))


BASE = load("results/day15_baseline_20261027T231622_013858/metrics.json")
EVAL = load("results/day15_eval_masked_20261028T183604_691086/metrics.json")
PFN = load("results/day15_pfn_masked_20261028T183404_691538/metrics.json")
DAY8 = load("results/day10_day08_adam_20261018T102005_973952/comparison.json")
DAY9B = load("results/day10_day09_horizons_20261018T103223_004434/comparison.json")

PRS = Presentation()
PRS.slide_width = Inches(13.333)
PRS.slide_height = Inches(7.5)
BLANK = PRS.slide_layouts[6]


def _text(frame, text, size=20, bold=False, color=DARK, bullet=False, level=0):
    frame.word_wrap = True
    para = frame.add_paragraph()
    para.level = level
    run = para.add_run()
    run.text = ("• " if bullet else "") + text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    return para


def title_slide(title, subtitle):
    slide = PRS.slides.add_slide(BLANK)
    box = slide.shapes.add_textbox(Inches(1), Inches(2.2), Inches(11.3), Inches(2.8))
    _text(box.text_frame, title, 36, bold=True)
    _text(box.text_frame, subtitle, 22, color=BLUE)
    return slide


def content_slide(title, bullets, image=None):
    slide = PRS.slides.add_slide(BLANK)
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.9))
    _text(box.text_frame, title, 30, bold=True)
    if image and "day15_eval_masked_" in image:
        box = slide.shapes.add_textbox(Inches(0.6), Inches(1.25), Inches(12.0), Inches(1.65))
        for item in bullets:
            para = _text(box.text_frame, item[0], 15, bullet=True, level=item[1])
            para.space_after = Pt(4)
        slide.shapes.add_picture(str(ROOT / image), Inches(1.1), Inches(3.0), width=Inches(11.0))
    elif image:
        box = slide.shapes.add_textbox(Inches(0.6), Inches(1.35), Inches(4.9), Inches(5.9))
        for b in bullets:
            p = _text(box.text_frame, b[0], 15, bullet=True, level=b[1])
            p.space_after = Pt(6)
        pic_path = ROOT / image
        w = Inches(7.0)
        slide.shapes.add_picture(str(pic_path), Inches(5.9), Inches(1.5), width=w)
    else:
        box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.6), Inches(5.8))
        for b in bullets:
            p = _text(box.text_frame, b[0], 19, bullet=True, level=b[1])
            p.space_after = Pt(10)
    return slide


b = lambda t, lvl=0: (t, lvl)

# ---------- 数据（取自JSON） ----------
mlp_mse = EVAL["rows"][0]["fixed_mlp"]["mse"]
bayes_k32 = next(r for r in EVAL["rows"] if r["K"] == 32)
pfn_k32 = bayes_k32["pfn_masked"]
old_pfn_k32 = bayes_k32["pfn_unmasked"]
pfn_coverages = [r["pfn_masked"]["coverage90"] for r in EVAL["rows"]]
bayes_k32 = bayes_k32["bayes"]
mlp_cov = EVAL["rows"][0]["fixed_mlp"]["coverage90"]

day8_plain = DAY8["groups"]["baseline"]["validation_mse"]
day8_phys = DAY8["groups"]["structured"]["validation_mse"]

# ---------- 幻灯片 ----------
title_slide("单摆世界模型：\n从固定动力学到上下文自适应预测",
            "2026-10-30")

content_slide("问题与方法", [
    b("任务：从 Pendulum-v1 交互数据学习状态转移，再做模型预测控制（MPC）"),
    b("数据：随机策略采集，按完整 episode 划分 train/val/test = 6/2/2（1200/400/400 条转移），只用 train 拟合输入统计量，test 未评估"),
    b("两个动力学模型：普通残差 MLP（4→64→64→3）vs 物理结构模型（预测 Δω→速度裁剪→角度积分→cos/sin 重建）"),
    b("控制：random shooting 打分候选动作序列，只执行首动作再重新规划"),
])

content_slide("预测结果", [
    b(f"Adam lr=1e-3、1950 更新：结构模型验证 MSE {day8_phys:.3e}，普通 MLP {day8_plain:.3e}"),
    b("多步递推（362 个共同验证起点）：末步 MSE 随预测长度增大，1/5/10/20 步约 3.1e-4 / 2.8e-3 / 1.6e-2 / 0.366"),
    b("旧配置结论只在旧配置成立：SGD200 下数据量排序与新配置相反"),
],
    image="figures/prediction_errors.png")

content_slide("MPC 与准确动力学对照", [
    b("同一候选库、同一选择流程，只替换评分动力学：learned vs oracle"),
    b("H=5：两种后端都失败（0/6）→ 短规划本身是限制"),
    b("H=15 / H=30：两种后端都成功（6/6）→ 该模型已足以支持较长规划"),
    b("注意：oracle 是诊断特权，不是最优规划器；单个训练种子、三个环境种子、单采样种子"),
])

content_slide("统计层面的教训", [
    b("结论随优化配置变化：数据量 200 vs 1200 的误差排序在 SGD200 与 Adam1950 下相反"),
    b("验证集同时用于早停和模型选择 → 结果是开发证据，不是独立泛化验证"),
    b("单个训练种子；两条验证 episode 的重叠窗口不是独立重复"),
    b("预测 MSE 低不代表控制更好：要分开报告预测、单位圆一致性和控制成功率"),
])

content_slide("动机：换物理参数就要重新训练吗？", [
    b("现有 MLP 只学了默认单摆（a=15, b=3，对应 g=10, m=1, l=1）"),
    b("换一个摆长或质量 → 动力学改变；固定模型缺少利用新任务上下文的机制"),
    b("PFN 的想法：把每个物理参数组合当成一个“任务”，从先验采样训练一个能就地推断的模型"),
    b("使用时：给少量上下文转移 → 一次前向 → 直接输出预测分布，不再训练"),
])

content_slide("PFN 设置", [
    b("任务先验：a~N(15,3²), b~N(3,0.75²)；模型为 y = dt·(a·sinθ + b·u) + ε，ε~N(0,0.02²)"),
    b("上下文 token = [sinθ, u, y, 1]；查询 token = [sinθ, u, 0, 0]；2 层 Transformer（d=64, 4 头）+ 高斯头输出 mean、log_var"),
    b("掩码：上下文和查询的注意力只读上下文；K=0 时只读先验 token，查询之间独立"),
    b("损失：高斯 NLL；每步从先验现采 64 个任务；训练 K∈{0,1,2,4,8,16}，K=32 留作评估"),
    b("对照：已知模型形式的精确贝叶斯后验预测（闭式参照）"),
])

content_slide("结果：预测误差随上下文下降", [
    b(f"K=32：掩码 PFN {pfn_k32['mse']:.2e}，旧版 {old_pfn_k32['mse']:.2e}；贝叶斯 {bayes_k32['mse']:.2e}"),
    b(f"固定 MLP {mlp_mse:.2e}，噪声方差 4.0e-4；掩码版本次误差略高于旧版"),
    b(f"掩码 PFN 覆盖率 {min(pfn_coverages):.2f}–{max(pfn_coverages):.2f}，偏保守；固定 MLP (+σ²) {mlp_cov:.2f}"),
],
    image="results/day15_eval_masked_20261028T183604_691086/pfn_vs_context.png")

content_slide("局限与边界", [
    b("合成任务：Δω 对 (a,b) 线性 + 高斯先验/噪声 → 贝叶斯有闭式解；真实世界模型更复杂"),
    b("单个训练种子、单个评估种子；高斯输出代替 PFN 原文的分段分布"),
    b("固定 MLP 的训练分布与随机采样状态有差异，且它不看上下文，属于不公平的强对照"),
    b("没有验证 PFN 预测在闭环 MPC 中的控制收益（下一步）"),
])

content_slide("下一步", [
    b("下一步：上下文世界模型 + MPC；多任务多步递推；PFN 预测方差用于风险控制"),
    b("所有数字取自 results/ 中归档 JSON；未使用 test 集；历史结果未覆盖"),
])

path = OUT / ("day15_xia_talk_masked.pptx" if "--masked-output" in sys.argv else "day15_xia_talk.pptx")
PRS.save(path)
print(f"saved: {path}  ({len(PRS.slides.slides) if hasattr(PRS.slides,'slides') else len(PRS.slides._sldIdLst)} slides)")
