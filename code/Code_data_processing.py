import torch
import pandas as pd
df = pd.read_csv("/07_04sheet1.csv")
print(df.head())
df["时间"] = pd.to_datetime(df["时间"])
start_date = pd.to_datetime("2022-01-03 00:00:00")
end_date   = pd.to_datetime("2022-06-27 23:59:59")
# 筛选时间
filtered_df = df[(df["时间"] >= start_date) & (df["时间"] <= end_date)]
output_file = r"/0704-2022_01_03-06_27.csv"
filtered_df.to_csv(output_file, index=False)





# ====一、 患者心意指标==============================

# =================
# 1.分别计算0502和0704表中医生收到患者心意次数和金额
# =================
df = pd.read_excel(r"D:\DeepLearning\doc_record\0704-2022_01_03-06_27.xlsx") # 对0502表进行同样操作

df["时间"] = pd.to_datetime(df["时间"])
start_date = pd.to_datetime("2022-01-03")
df["周次"] = ((df["时间"] - start_date).dt.days // 7) + 1
df = df[df["周次"].between(1, 26)]

id_col = "id"
name_col = "姓名"
amount_col = "金额"

# 3. 计算周度金额总和
weekly_amount_df = df.groupby([id_col, name_col, "周次"])[amount_col].sum().reset_index(name="金额")

# 4. 计算周度心意次数
weekly_count_df = df.groupby([id_col, name_col, "周次"]).size().reset_index(name="次数")

all_ids = df[[id_col, name_col]].drop_duplicates()
all_weeks = pd.DataFrame({"周次": range(1, 26 + 1)})
full = all_ids.merge(all_weeks, how="cross")

full = full.merge(weekly_amount_df, on=[id_col, name_col, "周次"], how="left")
full = full.merge(weekly_count_df, on=[id_col, name_col, "周次"], how="left")

full["金额"] = full["金额"].fillna(0)
full["次数"] = full["次数"].fillna(0)

amount_pivot = full.pivot(index=[id_col, name_col], columns="周次", values="金额")
count_pivot = full.pivot(index=[id_col, name_col], columns="周次", values="次数")
amount_pivot.columns = [f"第{week}周_金额" for week in amount_pivot.columns]
count_pivot.columns = [f"第{week}周_次数" for week in count_pivot.columns]
final = pd.concat([amount_pivot, count_pivot], axis=1)

output_file = r"D:\DeepLearning\process\0704患者心意统计含次数.xlsx"
final.to_excel(output_file, index=True)

file_0502 = r"D:\DeepLearning\process\0502后患者心意统计.xlsx"
file_0704 = r"D:\DeepLearning\process\0704患者心意统计含次数.xlsx"

df_0502 = pd.read_excel(file_0502)
df_0704 = pd.read_excel(file_0704)

# ======================
# 2.去重合并0502和0704
# ======================
df_merged = (
    pd.concat([df_0502, df_0704], ignore_index=True)
    .drop_duplicates(subset="id", keep="last")
)

df_merged = df_merged.sort_values("姓名").reset_index(drop=True)

save_path = r"D:\DeepLearning\process\0502_0704患者心意统计_并集.xlsx"

# ======================
# 3.整理成长表
# ======================
df = pd.read_excel(r"D:\DeepLearning\process\0502_0704患者心意统计_并集.xlsx")
df = df.drop('来源', axis=1)
static_cols = ['id', '姓名']


amount_cols = [col for col in df.columns if col.startswith('第') and col.endswith('_金额')]
count_cols  = [col for col in df.columns if col.startswith('第') and col.endswith('_次数')]

#  Melt 金额
df_amount = pd.melt(
    df,
    id_vars=static_cols,
    value_vars=amount_cols,
    var_name='week_var',
    value_name='金额'
)
df_amount['周数'] = df_amount['week_var'].str.extract(r'第(\d+)周_金额')[0].astype(int)
df_amount = df_amount.drop(columns=['week_var'])

# Melt 次数
df_count = pd.melt(
    df,
    id_vars=static_cols,
    value_vars=count_cols,
    var_name='week_var',
    value_name='次数'
)
df_count['周数'] = df_count['week_var'].str.extract(r'第(\d+)周_次数')[0].astype(int)
df_count = df_count.drop(columns=['week_var'])

# 6. 合并成最终长表（按 id + 周数 对齐）
long_df = pd.merge(
    df_amount,
    df_count,
    on=['id', '姓名', '周数'],
    how='outer'
)

long_df = long_df[['id', '姓名', '周数', '金额', '次数']]
long_df = long_df.sort_values(['id', '周数']).reset_index(drop=True)
long_df.to_excel(r"D:\DeepLearning\process\患者心意长表.xlsx", index=False)





# ==== 二、复诊情况指标 ===========
# ======================
# 1.为了避免重复操作，先将 0704和 0502 表进行合并
# ======================
file1_path = r"D:\DeepLearning\doc_record\0502问诊记录filtered_2022_01_03_2022_06_27.xlsx"
file2_path = r"D:\DeepLearning\doc_record\0704问诊_filtered_2022_01_03_2022_06_27.xlsx"
output_path = r"D:\DeepLearning\doc_record\05020704merged_records.xlsx"
df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

merged_df = pd.concat([df1, df2], ignore_index=True) # 将两个表合并

final_df = merged_df.drop_duplicates(keep='first') # 去重，保留第一次出现的记录

print(f"合并后总记录数: {len(final_df)}")
final_df.to_excel(output_path, index=False)

# ======================
# 2. 统计医生每周是否有复诊以及复诊次数
# ======================
df = pd.read_excel(r"D:\DeepLearning\process\05020704merged_records.xlsx")
df["评价时间"] = pd.to_datetime(df["评价时间"])

start_date = pd.to_datetime("2022-01-03")

df["week"] = ((df["评价时间"] - start_date).dt.days // 7 + 1)
df = df[(df["week"] >= 1) & (df["week"] <= 26)]


df = df.sort_values(["id", "用户id", "评价时间"])
df["visit_order"] = df.groupby(["id", "用户id"]).cumcount() + 1  # 在该医生下，该用户第几次问诊
df["is_return"] = (df["visit_order"] >= 2).astype(int) # 复诊：第2次及以后


# 每个医生-每周 复诊次数
weekly_return_times = (
    df[df["is_return"] == 1]
    .groupby(["id", "week"])
    .size()
    .reset_index(name="复诊次数")
)

weekly_return_wide = (
    weekly_return_times
    .pivot(index="id", columns="week", values="复诊次数")
    .fillna(0)
    .astype(int)
)

weekly_return_wide.columns = [f"复诊次数_{int(c)}" for c in weekly_return_wide.columns]
weekly_return_wide = weekly_return_wide.reset_index()


id_name = df[["id", "姓名"]].drop_duplicates()
final_df = id_name.merge(weekly_return_wide, on="id", how="left")


for i in range(1, 27):
    col = f"复诊次数_{i}"
    if col not in final_df.columns:
        final_df[col] = 0
final_df = final_df[["id", "姓名"] + [f"复诊次数_{i}" for i in range(1, 27)]]

final_df.to_excel(r"D:\DeepLearning\process\医生每周复诊次数.xlsx", index=False)

# 统计医生是否有复诊情况
df = pd.read_excel(r"D:\DeepLearning\process\医生每周复诊次数.xlsx")
df.fillna(0, inplace=True)
columns_to_process = [col for col in df.columns if col not in ['id', '姓名']]

for col in columns_to_process:
    df.loc[df[col] != 0, col] = 1
print(df)
df.to_excel(r"D:\DeepLearning\process\医生每周是否有复诊.xlsx", index=False)

# ======================
# 合并 & 转化成长表
# ======================
a = pd.read_excel(r"D:\DeepLearning\process\医生每周是否有复诊.xlsx")
b = pd.read_excel(r"D:\DeepLearning\process\医生每周复诊次数.xlsx")

merged_df = pd.merge(a, b, on='id')
output_path = r"D:\DeepLearning\process\是否有复诊和复诊次数.xlsx"
merged_df.to_excel(output_path, index=False)

file_path = r"D:\DeepLearning\process\是否有复诊和复诊次数.xlsx"
output_file = r'D:\DeepLearning\process\long_复诊.xlsx'

df = pd.read_excel(file_path)

static_cols = ['id', '姓名']
fuzhen_cols = [col for col in df.columns if col.startswith('是否有复诊_')]
fuzhen_count_cols = [col for col in df.columns if col.startswith('复诊次数_')]


# Melt “是否有复诊”
df_fuzhen = pd.melt(
    df,
    id_vars=static_cols,
    value_vars=fuzhen_cols,
    var_name='week_var',
    value_name='是否有复诊'
)
df_fuzhen['周数'] = df_fuzhen['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_fuzhen = df_fuzhen.drop(columns=['week_var'])

#  Melt “复诊次数”
df_count = pd.melt(
    df,
    id_vars=static_cols,
    value_vars=fuzhen_count_cols,
    var_name='week_var',
    value_name='复诊次数'
)
df_count['周数'] = df_count['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_count = df_count.drop(columns=['week_var'])

# 合并成最终长表（按 id + 姓名 + 周数 对齐）
long_df = pd.merge(
    df_fuzhen,
    df_count,
    on=['id', '姓名', '周数'],
    how='outer'
)
long_df = long_df[['id', '姓名', '周数', '是否有复诊', '复诊次数']]
long_df = long_df.sort_values(['id', '周数']).reset_index(drop=True)
long_df.to_excel(output_file, index=False)


# ======================
# 三、新增服务人次
# ======================
import pandas as pd
from functools import reduce

files = [
    r"D:\DeepLearning\doc_infor\doctor_info_2022_01_03.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_01_10.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_01_17.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_01_24.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_01_31.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_02-07.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_02-14.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_02-21.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_02-28.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_03-07.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_03-14.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_03-21.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_03-28.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_04-04.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_04-11.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_04-18.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_04-25.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_05-02.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_05-09.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_05-16.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_05-23.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_05-30.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_06-06.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_06-13.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_06-20.xls",
    r"D:\DeepLearning\doc_infor\doctor_info_2022_06-27.xls"
]

frames = []
for idx, path in enumerate(files, start=1):
    df = pd.read_excel(path)
    df = df[["id", "姓名", "服务人次(点击进入)"]].copy()
    df = df.rename(columns={"服务人次(点击进入)": f"服务人次_{idx}"})
    df = df.drop_duplicates(subset="id", keep="first")
    frames.append(df)


merged = reduce(lambda left, right: pd.merge(left, right, on=["id", "姓名"], how="outer"), frames)


num_weeks = len(frames)
for i in range(2, num_weeks + 1):
    merged[f"新增服务人次_{i}"] = merged[f"服务人次_{i}"] - merged[f"服务人次_{i - 1}"]

output_file = r"C:\Users\28578\Desktop\26服务人次(点击进入)_含新增.xlsx"
merged.to_excel(output_file, index=False)

# ======================
# 主表宽表转化成长表
# ======================
df = pd.read_excel(r"C:\Users\28578\Desktop\chunyu_data\merged_1.xlsx", sheet_name='Sheet1')
static_cols = [
    'id', '姓名', '总出现次数', '科室', '职称', '医院',
    '注册时间', '医生介绍', '图文资讯价格', '照片', '标签'
]

service_cols = [col for col in df.columns if col.startswith('服务人次_')]
new_service_cols = [col for col in df.columns if col.startswith('新增服务人次_')]
good_rate_cols = [col for col in df.columns if col.startswith('好评率_')]
peer_recog_cols = [col for col in df.columns if col.startswith('同行认可_')]
follow_cols = [col for col in df.columns if col.startswith('关注人数_')]
reg_duration_cols = [col for col in df.columns if col.startswith('第') and '周注册时长(天)' in col]

# 分别 melt（转为长格式）
# 服务人次
df_service = pd.melt(
    df,
    id_vars=static_cols,
    value_vars=service_cols,
    var_name='week_var',
    value_name='服务人次'
)
df_service['周数'] = df_service['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_service = df_service.drop(columns=['week_var'])

# 新增服务人次（从第2周开始，第1周自动为 NaN）
df_new_service = pd.melt(
    df, id_vars=['id'], value_vars=new_service_cols,
    var_name='week_var', value_name='新增服务人次'
)
df_new_service['周数'] = df_new_service['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_new_service = df_new_service.drop(columns=['week_var'])

# 好评率
df_good = pd.melt(
    df, id_vars=['id'], value_vars=good_rate_cols,
    var_name='week_var', value_name='好评率'
)
df_good['周数'] = df_good['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_good = df_good.drop(columns=['week_var'])

# 同行认可
df_peer = pd.melt(
    df, id_vars=['id'], value_vars=peer_recog_cols,
    var_name='week_var', value_name='同行认可'
)
df_peer['周数'] = df_peer['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_peer = df_peer.drop(columns=['week_var'])

# 关注人数
df_follow = pd.melt(
    df, id_vars=['id'], value_vars=follow_cols,
    var_name='week_var', value_name='关注人数'
)
df_follow['周数'] = df_follow['week_var'].str.extract(r'_(\d+)$')[0].astype(int)
df_follow = df_follow.drop(columns=['week_var'])

# 注册时长(天)
df_reg = pd.melt(
    df, id_vars=['id'], value_vars=reg_duration_cols,
    var_name='week_var', value_name='注册时长(天)'
)
df_reg['周数'] = df_reg['week_var'].str.extract(r'第(\d+)周')[0].astype(int)
df_reg = df_reg.drop(columns=['week_var'])

# 按 id + 周数 逐一合并
long_df = df_service
for temp_df in [df_new_service, df_good, df_peer, df_follow, df_reg]:
    long_df = pd.merge(long_df, temp_df, on=['id', '周数'], how='left')


final_cols = [
    'id', '姓名', '总出现次数', '周数', '服务人次', '新增服务人次',
    '好评率', '同行认可', '关注人数', '注册时长(天)',
    '科室', '职称', '医院', '注册时间', '医生介绍',
    '图文资讯价格', '照片', '标签'
]
long_df = long_df[final_cols]

long_df = long_df.sort_values(['id', '周数']).reset_index(drop=True)
long_df.to_excel(r"C:\Users\28578\Desktop\chunyu_data\long_1.xlsx", index=False)











# 匿名化处理代码：
import os
import pandas as pd
from pathlib import Path

# ====================== 配置区域 ======================
# 所有需要处理的文件路径（已加入 feedback metrics 文件）
file_paths = [
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Physician Profile Feature\Physician Profile Feature.csv",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (Weekly)\Text-related derived variables (weekly).xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (Weekly)\Audio-related derived variables (weekly).xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (Weekly)\Basic derived variables (weekly).xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (consultation as basic unit)\Text-related derived variables (without invalid rows).xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (consultation as basic unit)\Text-related derived variables.xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (consultation as basic unit)\Audio-related derived variables (without invalid variables).xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (consultation as basic unit)\Audio-related derived variables.xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (consultation as basic unit)\Basic derived variables (without invalid rows).xlsx",
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\Multimodal Derived Behavioral Variables (consultation as basic unit)\Basic derived variables.xlsx",
    # 新增文件
    r"D:\DeepLearning\chunyu_data\AAA最终版本数据\去除隐私变量版本\Derived Variables\feedback metrics (Weekly).csv",
]

# 目标列名
ID_COLUMN = "doc_id"

# 新ID的前缀和位数
NEW_ID_PREFIX = "doc_"
NEW_ID_DIGITS = 6          # 生成 doc_000001 这种格式

# 输出子文件夹名称
OUTPUT_SUBDIR = "anonymized"
# =====================================================


def read_file(path: str) -> pd.DataFrame:
    """根据扩展名读取文件"""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # 尝试常见编码
        for encoding in ["utf-8", "gbk", "gb18030", "latin1"]:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法解析编码: {path}")
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: {path}")


def save_file(df: pd.DataFrame, original_path: str):
    """保存处理后的文件，保持原扩展名"""
    original_path = Path(original_path)
    
    # 在原文件同级目录创建 anonymized 子文件夹
    out_dir = original_path.parent / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / original_path.name
    
    if original_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(out_path, index=False)
    
    print(f"  已保存 → {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("开始收集所有唯一的 doc_id ...")
    print("=" * 60)
    
    all_ids = set()
    file_dfs = {}  # 缓存读取的 DataFrame
    
    for path in file_paths:
        path = Path(path)
        if not path.exists():
            print(f"[警告] 文件不存在，跳过: {path}")
            continue
        
        print(f"读取: {path.name}")
        try:
            df = read_file(path)
            file_dfs[str(path)] = df
            
            if ID_COLUMN not in df.columns:
                print(f"  [警告] 列 '{ID_COLUMN}' 不存在，当前列名: {list(df.columns)}")
                continue
            
            ids = df[ID_COLUMN].dropna().unique()
            all_ids.update(ids)
            print(f"  找到 {len(ids)} 个唯一 doc_id")
        except Exception as e:
            print(f"  [错误] 读取失败: {e}")
    
    if not all_ids:
        print("未找到任何 doc_id，程序结束。")
        return
    
    # 排序后生成稳定映射（保证多次运行结果一致）
    sorted_ids = sorted(all_ids, key=lambda x: str(x))
    id_mapping = {
        old_id: f"{NEW_ID_PREFIX}{str(i+1).zfill(NEW_ID_DIGITS)}"
        for i, old_id in enumerate(sorted_ids)
    }
    
    print("\n" + "=" * 60)
    print(f"共收集到 {len(id_mapping)} 个唯一医生")
    print("映射示例（前5个）:")
    for i, (old, new) in enumerate(list(id_mapping.items())[:5]):
        print(f"  {old}  →  {new}")
    if len(id_mapping) > 5:
        print("  ...")
    print("=" * 60)
    
    # 保存映射表
    mapping_df = pd.DataFrame({
        "original_doc_id": list(id_mapping.keys()),
        "anonymized_doc_id": list(id_mapping.values())
    })
    
    # 映射表放在第一个有效文件所在目录的 anonymized 下
    first_valid = next(iter(file_dfs.keys()))
    mapping_dir = Path(first_valid).parent / OUTPUT_SUBDIR
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = mapping_dir / "doc_id_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    print(f"\n映射表已保存: {mapping_path}")
    
    # 开始替换并保存
    print("\n" + "=" * 60)
    print("开始替换并保存文件 ...")
    print("=" * 60)
    
    for path_str, df in file_dfs.items():
        path = Path(path_str)
        print(f"\n处理: {path.name}")
        
        if ID_COLUMN not in df.columns:
            print("  跳过（无 doc_id 列）")
            continue
        
        # 替换
        df[ID_COLUMN] = df[ID_COLUMN].map(id_mapping)
        
        # 检查是否有未映射的值
        unmapped = df[ID_COLUMN].isna().sum()
        if unmapped > 0:
            print(f"  [注意] 有 {unmapped} 行未能映射（原值为空或异常）")
        
        save_file(df, path)
    
    print("\n" + "=" * 60)
    print("全部处理完成！")
    print(f"新文件保存在各原文件目录下的 '{OUTPUT_SUBDIR}' 文件夹中")
    print(f"映射关系保存在: {mapping_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
