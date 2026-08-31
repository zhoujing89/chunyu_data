import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 允许 tqdm 在 pandas 中使用
tqdm.pandas()

# --- 1. 初始化数据 ---
# 假设 record / data 已经定义好
data = pd.read_excel(r"D:\DeepLearning\chunyu_data\AAA最终版本数据\0823技术验证数据\Text-related derived variables.xlsx")  # 如果变量名是 data，请改成 data = data

# --- 2. 加载词典 (与原代码完全一致) ---
medical_words_set = set()
try:
    with open(r"C:\Users\28578\Desktop\THUOCL_medical.txt", encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip().split()[0]
            if len(word) > 1:
                medical_words_set.add(word)
    for w in medical_words_set:
        jieba.add_word(w)
    print(f"成功加载医学词典，共 {len(medical_words_set)} 个词")
except Exception as e:
    print(f"⚠️ 未找到医学词典，医学密度计算可能不准！错误信息: {e}")

stop_words_set = set()
try:
    with open('H:\\bishe\\doc_info\\dict\\stopwords-master\\cn_stopwords.txt', encoding='utf-8') as f:
        for line in f.readlines():
            stop_words_set.add(line.strip())
except:
    stop_words_set = {'的', '了', '和', '是', '就', '都', '而', '及', '与'}
    print("⚠️ 未找到停用词表，使用内置基础停用词。")

# 词典定义 (保持不变，本验证不需要用到)
certainty_words = {'肯定', '一定', '确诊', '绝对', '明显', '必须', '无疑', '明确', '百分之百', '确保', '显然', '根本',
                   '立刻', '马上', '不用怀疑', '保证', '正是'}
uncertainty_words = {'可能', '大概', '也许', '好像', '恐怕', '估计', '不排除', '或许', '看情况', '观察', '疑似', '貌似',
                     '大约', '可能吧', '不一定', '难说'}
politeness_words = {'您', '请', '祝', '麻烦', '谢谢', '感谢', '不用谢', '不客气', '没事', '没关系', '早日康复', '放心',
                    '安心', '不用担心', '抱歉', '不好意思', '好的', '收到', '亲', '家长', '朋友'}


# ================= 3. 核心计算函数（增加返回 medical_cnt） =================
def analyze_doctor_text_v2(text_content):
    """
    返回: medical_density, certainty_score, politeness_density, total_valid_words, medical_cnt
    如果无法计算，统一返回 (None, None, None, None, None)
    """
    if not isinstance(text_content, str) or len(text_content.strip()) == 0:
        return None, None, None, None, None

    try:
        words = jieba.lcut(text_content)

        total_valid_words = 0
        medical_cnt = 0
        certainty_cnt = 0
        uncertainty_cnt = 0
        politeness_cnt = 0

        for w in words:
            if w in stop_words_set or len(w.strip()) == 0:
                continue

            total_valid_words += 1

            if w in medical_words_set:
                medical_cnt += 1
            if w in certainty_words:
                certainty_cnt += 1
            elif w in uncertainty_words:
                uncertainty_cnt += 1
            if w in politeness_words:
                politeness_cnt += 1

        if total_valid_words == 0:
            return None, None, None, None, None

        medical_density = medical_cnt / total_valid_words
        certainty_score = (certainty_cnt - uncertainty_cnt) / total_valid_words
        politeness_density = politeness_cnt / total_valid_words

        return medical_density, certainty_score, politeness_density, total_valid_words, medical_cnt

    except Exception as e:
        return None, None, None, None, None


# ================= 4. 执行计算并保存 =================
print("正在计算文本指标（含医学术语计数）...")

results = data['医生对话'].progress_apply(
    lambda x: pd.Series(analyze_doctor_text_v2(x))
)

results.columns = [
    'medical_density',
    'certainty_score',
    'politeness_density',
    'word_count',
    'medical_cnt'  # 新增：匹配到的医学术语出现次数
]

# 合并回原表
data = pd.concat([data, results], axis=1)

# ================= 5. 医学术语二元指标技术验证（阈值敏感性） =================
print("\n" + "=" * 60)
print("医学术语指标阈值敏感性分析")
print("=" * 60)

# 定义“符合条件的咨询”：成功计算出 word_count 的样本（即有效文本）
eligible_mask = data['word_count'].notna()
eligible = data.loc[eligible_mask].copy()
n_eligible = len(eligible)

print(f"符合条件的咨询总数: {n_eligible}")

# 三种阈值的二元指标
eligible['has_med_1'] = (eligible['medical_cnt'] >= 1).astype(int)
eligible['has_med_2'] = (eligible['medical_cnt'] >= 2).astype(int)
eligible['has_med_3'] = (eligible['medical_cnt'] >= 3).astype(int)

# 各阈值下的占比
pct_1 = eligible['has_med_1'].mean() * 100
pct_2 = eligible['has_med_2'].mean() * 100
pct_3 = eligible['has_med_3'].mean() * 100

print(f"\n【占比结果】")
print(f"至少 1 个医学术语 (主定义): {pct_1:.2f}%")
print(f"至少 2 个医学术语:           {pct_2:.2f}%")
print(f"至少 3 个医学术语:           {pct_3:.2f}%")

# 与主定义（≥1）的一致性（agreement）
# 一致性 = 两个二元变量取值完全相同的比例
agree_2 = (eligible['has_med_1'] == eligible['has_med_2']).mean() * 100
agree_3 = (eligible['has_med_1'] == eligible['has_med_3']).mean() * 100

print(f"\n【与主定义（≥1）的一致性】")
print(f"与 ≥2 阈值的一致性: {agree_2:.2f}%")
print(f"与 ≥3 阈值的一致性: {agree_3:.2f}%")

# 额外诊断信息（可选，方便检查）
print(f"\n【补充分布】")
print(f"medical_cnt 描述统计:")
print(eligible['medical_cnt'].describe().round(2))
print(f"\nmedical_cnt 取值分布（前10）:")
print(eligible['medical_cnt'].value_counts().sort_index().head(10))

# 将二元指标也合并回原 data（方便后续使用）
data.loc[eligible_mask, 'has_med_1'] = eligible['has_med_1']
data.loc[eligible_mask, 'has_med_2'] = eligible['has_med_2']
data.loc[eligible_mask, 'has_med_3'] = eligible['has_med_3']

print("\n计算完成。可将 data 保存或继续后续分析。")



################################


df = pd.read_excel(r"D:\DeepLearning\chunyu_data\AAA最终版本数据\0823技术验证数据\Vocal_Warmth.xlsx")
df.columns

cols = ['warmth_score', 55, 64, '75,25']

corr_matrix = df[cols].corr(method='spearman')
print(corr_matrix)

print(corr_matrix['warmth_score'].drop('warmth_score'))


#################################
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr  # 若无 scipy，可改用 pandas .corr
import warnings

warnings.filterwarnings('ignore')

# 允许 tqdm 在 pandas 中使用
tqdm.pandas()

# =====================================================================
# 0. 假设前序代码已运行：data 已存在，且包含列 '医生对话'
#    同时已加载 medical_words_set、stop_words_set 等
# =====================================================================

# =====================================================================
# 1. 定义完整的 36 个礼貌相关表达式词典
#    ★★★ 请务必替换为你补充材料中发布的真实 36 项列表 ★★★
# =====================================================================
politeness_expressions = [
    # 原有核心项（约 21 项）
    '您', '请', '祝', '麻烦', '谢谢', '感谢', '不用谢', '不客气', '没事', '没关系',
    '早日康复', '放心', '安心', '不用担心', '抱歉', '不好意思', '好的', '收到', '亲', '家长',
    '朋友',
    # 扩展至 36 项的常见医患礼貌表达（多词短语已加入，便于精确匹配）
    '您好', '请问', '麻烦您', '谢谢您', '非常感谢', '祝您', '请放心', '早日好转',
    '辛苦了', '希望有帮助', '请描述', '马上为您', '不用客气', '没关系的', '感谢信任'
]

assert len(politeness_expressions) == 36, f"词典长度应为 36，当前为 {len(politeness_expressions)}"
print(f"完整礼貌词典已加载，共 {len(politeness_expressions)} 个表达式")

# 将所有表达式强制加入 jieba 词典，保证多词短语不被切分
for expr in politeness_expressions:
    jieba.add_word(expr)

# =====================================================================
# 2. 预分词缓存：只做一次 jieba，后续 36 次循环只做集合匹配
# =====================================================================
print("\n正在对所有文本进行预分词并缓存有效词列表（仅此一次）...")


def get_valid_words(text):
    """返回去除停用词后的有效词列表；无法计算时返回 None"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    try:
        words = jieba.lcut(text)
        valid = [w for w in words if w not in stop_words_set and len(w.strip()) > 0]
        return valid if len(valid) > 0 else None
    except Exception:
        return None


# 缓存
data['_valid_words'] = data['医生对话'].progress_apply(get_valid_words)

# 符合条件的咨询掩码（与前序 word_count 定义一致）
eligible_mask = data['_valid_words'].notna()
eligible = data.loc[eligible_mask].copy()
n_eligible = len(eligible)
print(f"符合条件的咨询总数: {n_eligible}")

# =====================================================================
# 3. 计算完整词典下的原始礼貌密度（作为基准）
# =====================================================================
print("\n计算完整词典下的原始礼貌密度...")

full_set = set(politeness_expressions)


def compute_politeness_density(valid_words, politeness_set):
    """给定有效词列表与礼貌集合，返回密度"""
    if valid_words is None or len(valid_words) == 0:
        return np.nan
    cnt = sum(1 for w in valid_words if w in politeness_set)
    return cnt / len(valid_words)


eligible['politeness_density_full'] = eligible['_valid_words'].apply(
    lambda vw: compute_politeness_density(vw, full_set)
)

# 可选：把完整密度写回 data（便于后续使用）
data.loc[eligible_mask, 'politeness_density_full'] = eligible['politeness_density_full']

print(f"原始礼貌密度描述统计:\n{eligible['politeness_density_full'].describe().round(4)}")

# =====================================================================
# 4. 留一表达式（leave-one-expression-out）敏感性分析
# =====================================================================
print("\n" + "=" * 70)
print("开始留一表达式敏感性分析（共 36 次重新计算）")
print("=" * 70)

spearman_corrs = []

for i, expr_to_remove in enumerate(tqdm(politeness_expressions, desc="Leave-one-out")):
    # 构建精简词典（剩余 35 个）
    reduced_set = full_set - {expr_to_remove}

    # 重新计算所有符合条件样本的密度
    reduced_densities = eligible['_valid_words'].apply(
        lambda vw: compute_politeness_density(vw, reduced_set)
    )

    # 计算与完整词典密度的 Spearman 等级相关系数
    # 只使用两者都非缺失的样本（理论上应全部有效）
    valid_pair_mask = reduced_densities.notna() & eligible['politeness_density_full'].notna()
    if valid_pair_mask.sum() < 2:
        rho = np.nan
    else:
        rho, _ = spearmanr(
            eligible.loc[valid_pair_mask, 'politeness_density_full'],
            reduced_densities[valid_pair_mask]
        )

    spearman_corrs.append(rho)

    # 可选：打印每一步的简要信息（调试用，正式跑可注释）
    # print(f"  移除「{expr_to_remove}」→ Spearman ρ = {rho:.4f}")

# 转为 Series 方便统计
corr_series = pd.Series(spearman_corrs, index=politeness_expressions)

# =====================================================================
# 5. 汇总结果（直接对应文中需要填入的 XX）
# =====================================================================
print("\n" + "=" * 70)
print("留一表达式敏感性分析结果")
print("=" * 70)

min_rho = corr_series.min()
max_rho = corr_series.max()
median_rho = corr_series.median()

print(f"斯皮尔曼相关系数范围: {min_rho:.4f} ~ {max_rho:.4f}")
print(f"中位数: {median_rho:.4f}")
print(f"\n详细分布:")
print(corr_series.describe().round(4))

print("\n各表达式被移除后的相关系数（按 ρ 升序排列，便于查看影响最大的表达式）:")
print(corr_series.sort_values().round(4))

# 可选：保存结果到 CSV 便于论文附录
corr_series.to_csv(r"D:\DeepLearning\chunyu_data\AAA最终版本数据\0823技术验证数据\politeness_leave_one_out_spearman.csv", encoding='utf-8-sig')
# "D:\DeepLearning\chunyu_data\AAA最终版本数据\0823技术验证数据\Text-related derived variables.xlsx"
print("\n分析完成。")









