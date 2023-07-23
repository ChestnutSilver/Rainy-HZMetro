import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置警告过滤器，这里过滤了将来可能会发生的警告（FutureWarning）。
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# 绘图样式的设置，使用了 seaborn 库，将绘图样式设置为 'seaborn'，并将字体比例设置为1。
plt.style.use("seaborn")
sns.set(font_scale=1)

# 设置数据集目录
path = "E:\\1.同济大学各学科资料\\3.2 大三年级下学期\\11.专业实习\\专业实习\\code\\HangzhouMetro"
test = pd.read_csv(path + "/Metro_testA/testA_submit_2019-01-29.csv")
test_28 = pd.read_csv(path + "/Metro_testA/testA_record_2019-01-28.csv")
station_con = pd.read_csv("Metro_roadMap.csv")


# 计算真实值与预测值之间的绝对百分比误差的平均值
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# 对原始数据进行处理和计算，提取了一些基本特征，并返回结果数据
# df_：原始数据
# test：待处理的测试数据，与 df_ 具有相同的列。
# time_str：时间字符串，用于处理日期的替换操作。
def get_base_features(df_, test, time_str):
    df = df_.copy()
    df["startTime"] = df["time"].apply(
        lambda x: x[:15].replace(time_str, "-01-29") + "0:00"
    )
    df = (
        df.groupby(["startTime", "stationID"])
        .status.agg(["count", "sum"])
        .reset_index()
    )
    df = test.merge(df, "left", ["stationID", "startTime"])
    df["time"] = df["startTime"].apply(
        lambda x: x[:15].replace("-01-29", time_str) + "0:00"
    )
    del df["startTime"], df["endTime"]
    # base time
    df["day"] = df["time"].apply(lambda x: int(x[8:10]))
    df["week"] = pd.to_datetime(df["time"]).dt.dayofweek + 1
    df["hour"] = df["time"].apply(lambda x: int(x[11:13]))
    df["minute"] = df["time"].apply(lambda x: int(x[14:15] + "0"))
    result = df.copy()
    # in,out
    result["inNums"] = result["sum"]
    result["outNums"] = result["count"] - result["sum"]
    result["day_since_first"] = result["day"] - 1
    # 填充缺失值
    result.fillna(0, inplace=True)
    del result["sum"], result["count"]
    return result


time_str = "-01-28"
data1 = get_base_features(test_28, test, time_str)

###29号时间等信息是本身的，inNums和outNums用的28号的数据
###后面也就可以直接将29号作为测试集
time_str = "-01-29"
df = pd.read_csv(path + "/Metro_testA/testA_record_2019-01-28.csv")
df["time"] = df["time"].apply(lambda x: x[:15].replace("-01-28", time_str) + "0:00")
df = get_base_features(df, test, time_str)
data1 = pd.concat([data1, df], axis=0, ignore_index=True)

# 对指定路径下的每个 csv 文件进行处理和特征提取，并将提取的结果合并到 data1 中。最终得到的 data1 包含了所有文件的特征提取结果。
data_list = os.listdir(path + "/Metro_train/")
for i in range(0, len(data_list)):
    if data_list[i].split(".")[-1] == "csv":
        time_str = data_list[i].split(".")[0][11:17]
        print(data_list[i], i)
        df = pd.read_csv(path + "/Metro_train/" + data_list[i])
        df = get_base_features(df, test, time_str)
        data1 = pd.concat([data1, df], axis=0, ignore_index=True)
    else:
        continue


# 计算进站出站客流量的特征，
# 包括每个站口在每周每小时的最大、最小、平均和总出站客流量，
# 每个站口在每周的最大、最小、平均和总出站客流量，
# 每个站口的最大、最小、平均和总出站客流量，
# 每天所有站口的最大、最小、平均和总出站客流量
def more_feature(result):
    tmp = result.copy()
    tmp = tmp[["stationID", "week", "day", "hour"]]
    ###按week计算每个站口每小时客流量特征
    tmp = result.groupby(["stationID", "week", "hour"], as_index=False)["inNums"].agg(
        {
            "inNums_ID_dh_max": "max",  ###
            "inNums_ID_dh_min": "min",  ###
            "inNums_ID_dh_mean": "mean",  ###
            "inNums_ID_dh_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["stationID", "week", "hour"], how="left")
    ###按week计算每个站口客流量特征
    tmp = result.groupby(["stationID", "week"], as_index=False)["inNums"].agg(
        {
            "inNums_ID_d_max": "max",
            "inNums_ID_d_min": "min",  # 都为0
            "inNums_ID_d_mean": "mean",  ##
            "inNums_ID_d_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["stationID", "week"], how="left")
    ###每个站口所有天客流量特征
    tmp = result.groupby(["stationID"], as_index=False)["inNums"].agg(
        {
            "inNums_ID_max": "max",
            "inNums_ID_min": "min",
            "inNums_ID_mean": "mean",  ##
            "inNums_ID_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["stationID"], how="left")
    ###每天所有站口客流量特征
    tmp = result.groupby(["day"], as_index=False)["inNums"].agg(
        {
            "inNums_d_max": "max",
            "inNums_d_min": "min",  # 都为0
            "inNums_d_mean": "mean",  ##
            "inNums_d_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["day"], how="left")

    ###出站与进站类似
    tmp = result.groupby(["stationID", "week", "hour"], as_index=False)["outNums"].agg(
        {
            "outNums_ID_dh_max": "max",
            "outNums_ID_dh_min": "min",  ##
            "outNums_ID_dh_mean": "mean",  ##
            "outNums_ID_dh_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["stationID", "week", "hour"], how="left")
    tmp = result.groupby(["stationID", "week"], as_index=False)["outNums"].agg(
        {
            "outNums_ID_d_max": "max",
            "outNums_ID_d_min": "min",  # 都为0
            "outNums_ID_d_mean": "mean",  ##
            "outNums_ID_d_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["stationID", "week"], how="left")
    tmp = result.groupby(["stationID"], as_index=False)["outNums"].agg(
        {
            "outNums_ID_max": "max",
            "outNums_ID_min": "min",
            "outNums_ID_mean": "mean",
            "outNums_ID_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["stationID"], how="left")
    tmp = result.groupby(["day"], as_index=False)["outNums"].agg(
        {
            "outNums_d_max": "max",
            "outNumss_d_min": "min",  # 都为0
            "outNums_d_mean": "mean",
            "outNums_d_sum": "sum",
        }
    )
    result = result.merge(tmp, on=["day"], how="left")

    return result


data2 = more_feature(data1)

# 删除data2中那些缺失值比例超过 90% 的列
good_cols = list(data2.columns)
for col in data2.columns:
    rate = data2[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.90:
        good_cols.remove(col)
        print(col, rate)

data2 = data2[good_cols]

# 将28，29拼接到最后，整体有序了
data_28 = data2[data2.day == 28]
data_29 = data2[data2.day == 29]
data2 = data2[(data2.day != 28) & (data2.day != 29)]
data2 = pd.concat([data2, data_28, data_29], axis=0, ignore_index=True)
data = data2.copy()

# 剔除周末
data = data[(data.day != 5) & (data.day != 6) & (data.day != 1)]
data = data[(data.day != 12) & (data.day != 13)]
data = data[(data.day != 19) & (data.day != 20)]
data = data[(data.day != 26) & (data.day != 27)]


# shift时间，144个时间点是一天，选取的近三天的时间及其组合特征
# data_in_sta（输入数据框）
# data_in_shift_cols（要进行时间滞后处理的输入列）
# data_out_shift_cols（要进行时间滞后处理的输出列
def time_shift(data_in_sta, data_in_shfit_cols, data_out_shfit_cols):
    # 指定了时间滞后的起始位置和结束位置。时间滞后的单位是 144，并且最多滞后 3 个时间点。
    lag_start = 144
    lag_end = 144 * 3
    data_out_sta = data_in_sta.copy()
    # 循环迭代滞后范围内的所有时间点
    for i in range(lag_start, lag_end + 1, 144):
        # 将该列的值向后滞后 i 个时间点
        for col in data_in_shfit_cols:
            data_in_sta[col + "_lag_{}".format(i)] = data_in_sta[col].shift(i)
            if (col != "inNums") & (col != "outNums") & (i == lag_end):
                del data_in_sta[col]
        for col1 in data_out_shfit_cols:
            data_out_sta[col1 + "_lag_{}".format(i)] = data_out_sta[col1].shift(i)
            if (col1 != "inNums") & (col1 != "outNums") & (i == lag_end):
                del data_out_sta[col1]

    return data_in_sta, data_out_sta


# 由于只shift inNums和outNums，则先排除其余特征
data_in_shfit = pd.DataFrame()
data_out_shfit = pd.DataFrame()

data_in_shfit_cols = list(data)
data_in_shfit_cols.remove("stationID")
data_in_shfit_cols.remove("time")
data_in_shfit_cols.remove("day")
data_in_shfit_cols.remove("week")
data_in_shfit_cols.remove("hour")
data_in_shfit_cols.remove("minute")
data_in_shfit_cols.remove("day_since_first")

data_out_shfit_cols = list(data)
data_out_shfit_cols.remove("stationID")
data_out_shfit_cols.remove("time")
data_out_shfit_cols.remove("day")
data_out_shfit_cols.remove("week")
data_out_shfit_cols.remove("hour")
data_out_shfit_cols.remove("minute")
data_out_shfit_cols.remove("day_since_first")


###对每个站口进行shift操作
for i in range(81):
    data_temp = data[data["stationID"] == i]
    data_in_sta, data_out_sta = time_shift(
        data_temp, data_in_shfit_cols, data_out_shfit_cols
    )
    data_in_shfit = pd.concat([data_in_shfit, data_in_sta], axis=0, ignore_index=True)
    data_out_shfit = pd.concat(
        [data_out_shfit, data_out_sta], axis=0, ignore_index=True
    )


# inNums训练及预测
data_in_shfit_temp = data_in_shfit.copy()
del data_in_shfit_temp["time"]
data_in_shfit_temp.fillna(0, inplace=True)

# 使用 XGBoost 模型对指定日期的数据进行训练和预测，并计算预测结果的误差。
# 进行时间序列的交叉验证，比如选取23号作为验证集，则23号前的作为训练集，24号作为测试集，依次类
# test_day 的列表，其中包含要进行验证和预测的日期
# error_in 的空列表，用于保存每个验证集下一天作为测试集时的误差
test_day = [23, 24, 25, 28]
error_in = []

for i in test_day:
    # 设置测试集日期
    if (i != 28) & (i != 25):
        test = data_in_shfit_temp[data_in_shfit_temp.day == i + 1]
        y_test = test["inNums"]
        del test["inNums"]
        del test["outNums"]
    # 由于剔除周末，25要选择28
    if i == 25:
        test = data_in_shfit_temp[data_in_shfit_temp.day == i + 3]
        y_test = test["inNums"]
        del test["inNums"]
        del test["outNums"]

    print("###############################inNums验证集", i)
    # 设置训练集和验证集
    train = data_in_shfit_temp[data_in_shfit_temp.day < i]
    valid = data_in_shfit_temp[data_in_shfit_temp.day == i]
    y_train = train["inNums"]
    y_valid = valid["inNums"]

    del train["inNums"], valid["inNums"]
    del train["outNums"], valid["outNums"]

    # 创建三个数据集对应的特征矩阵
    dtrain = xgb.DMatrix(train, label=y_train)
    dtest = xgb.DMatrix(test)
    dval = xgb.DMatrix(valid, label=y_valid)

    # 用于监视 XGBoost 模型的训练进程
    watchlist = [(dtrain, "train"), (dval, "val")]

    # XGBoost 模型的参数设置
    # eta：学习率（Learning Rate），控制每个弱学习器的权重。较小的值可以使模型更加稳定，但需要更多的迭代次数才能收敛到最优解。默认值为0.3
    # max_depth：树的最大深度限制，用于控制模型的复杂度。较大的值可以增加模型的容量，但也容易导致过拟合。默认值为6
    # subsample：子样本比例，用于控制每棵树使用的训练样本的比例。默认值为1，表示使用全部样本。这里每棵树使用训练样本80 %。
    # colsample_bytree：列采样比例，用于控制每棵树使用的特征的比例。默认值为1，表示使用全部特征。这里每棵树使用特征的80 %。
    # objective：目标函数，指定XGBoost模型的优化目标。这里设置为'reg:linear'，表示使用线性回归作为目标函数。
    # eval_metric：评估指标，用于衡量模型在训练过程中的性能。这里设置为'mae'，表示使用平均绝对误差（Mean Absolute Error）作为评估指标。
    # silent：是否显示训练过程中的输出信息。默认值为False，这里设置为True，表示不显示输出信息。
    # nthread：并行线程数，用于加速训练过程。默认值为当前系统上可用的最大线程数，这里设置为4。
    xgb_params = {
        "eta": 0.004,
        "max_depth": 11,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:linear",
        "eval_metric": "mae",
        "silent": True,
        "nthread": 4,
    }
    # 使用 XGBoost 模型进行训练。
    # dtrain 是训练集的特征矩阵和标签，
    # num_boost_round 是迭代次数，
    # evals 是用于验证模型的特征矩阵和标签，
    # early_stopping_rounds 是早停轮数，
    # verbose_eval 是控制打印信息的频率，
    # params 是模型参数。
    clf = xgb.train(
        dtrain=dtrain,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=100,
        params=xgb_params,
    )
    # 则使用训练好的模型clf对测试集dtest进行预测，并将预测结果存储在prediction_in变量中。
    # 计算预测结果与真实值y_test之间的平均绝对百分比误差，并将其存储在error变量中。最后，将误差error添加到error_in列表中
    # 并打印验证集和对应的误差。
    if i != 28:
        prediction_in = clf.predict(dtest, ntree_limit=clf.best_iteration)
        error = mean_absolute_percentage_error(np.abs(np.round(prediction_in)), y_test)
        error_in.append(error)
        print("验证集：", i)
        print("验证集下一天作为测试集的误差为：", error)
# 验证分数为误差平均值
print("inNums的CV验证分数：", np.mean(error_in))

# 最终预测29号时要加上28号一共的数据集，过程与上述类似
X_data = data_in_shfit_temp[data_in_shfit_temp.day < 29]
test = data_in_shfit_temp[data_in_shfit_temp.day == 29]
valid = data_in_shfit_temp[data_in_shfit_temp.day == 28]
y_valid = valid["inNums"]
del valid["outNums"], valid["inNums"]
y_data = X_data["inNums"]
y_test = test["inNums"]
del X_data["inNums"], test["inNums"]
del X_data["outNums"], test["outNums"]
dtrain = xgb.DMatrix(X_data, label=y_data)
dtest = xgb.DMatrix(test)
dval = xgb.DMatrix(valid, label=y_valid)
watchlist = [(dtrain, "train")]
clf = xgb.train(
    dtrain=dtrain,
    num_boost_round=clf.best_iteration,
    early_stopping_rounds=100,
    evals=watchlist,
    verbose_eval=100,
    params=xgb_params,
)


# Get the evaluation results during training
evals_result = clf.evals_result()

# Extract the training and validation metrics
train_metrics = evals_result["train"]["mae"]
val_metrics = evals_result["val"]["mae"]

# Plot the iteration process
plt.figure(figsize=(10, 6))
plt.plot(train_metrics, label="Train")
plt.plot(val_metrics, label="Validation")
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.title("XGBoost Iteration Process")
plt.legend()
plt.show()


prediction_in = clf.predict(dtest, ntree_limit=clf.best_iteration)
prediction = pd.DataFrame()
prediction["inNums"] = prediction_in
prediction["inNums"] = abs(np.round(prediction["inNums"]))
error_test_in = mean_absolute_percentage_error(
    abs(np.round(prediction["inNums"])), y_test
)

# outNums训练及预测,过程与inNum相同
data_out_shfit_temp = data_out_shfit.copy()

del data_out_shfit_temp["time"]
data_out_shfit_temp.fillna(0, inplace=True)
# 训练验证测试23 24 25 28
test_day = [23, 24, 25, 28]
error_out = []
for i in test_day:
    if (i != 28) & (i != 25):
        test_out = data_out_shfit_temp[data_out_shfit_temp.day == i + 1]
        y_test_out = test_out["outNums"]
        del test_out["inNums"]
        del test_out["outNums"]

    if i == 25:
        test_out = data_out_shfit_temp[data_out_shfit_temp.day == i + 3]
        y_test_out = test_out["outNums"]
        del test_out["inNums"]
        del test_out["outNums"]

    print("###############################outNums验证集", i)
    train_out = data_out_shfit_temp[data_out_shfit_temp.day < i]

    valid_out = data_out_shfit_temp[data_out_shfit_temp.day == i]

    y_train_out = train_out["outNums"]
    y_valid_out = valid_out["outNums"]

    del train_out["inNums"], valid_out["inNums"]
    del train_out["outNums"], valid_out["outNums"]

    dtrain = xgb.DMatrix(train_out, label=y_train_out)
    dtest = xgb.DMatrix(test_out)
    dval = xgb.DMatrix(valid_out, label=y_valid_out)
    watchlist = [(dtrain, "train"), (dval, "val")]
    xgb_params = {
        "eta": 0.004,
        "max_depth": 11,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:linear",
        "eval_metric": "mae",
        "silent": True,
        "nthread": 4,
    }
    clf = xgb.train(
        dtrain=dtrain,
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=100,
        verbose_eval=100,
        params=xgb_params,
    )

    if i != 28:
        prediction_out = clf.predict(dtest, ntree_limit=clf.best_iteration)
        error = mean_absolute_percentage_error(
            np.abs(np.round(prediction_out)), y_test_out
        )
        error_out.append(error)
        print("验证集：", i)
        print("验证集下一天作为测试集的误差为：", error)

print("outNums的CV验证分数：", np.mean(error_out))

# 最终预测29号
X_data_out = data_out_shfit_temp[data_out_shfit_temp.day < 29]
test_out = data_out_shfit_temp[data_out_shfit_temp.day == 29]
valid_out = data_out_shfit_temp[data_out_shfit_temp.day == 28]
y_valid_out = valid_out["outNums"]

del valid_out["inNums"], valid_out["outNums"]
y_data_out = X_data_out["outNums"]
y_test_out = test_out["outNums"]
del X_data_out["inNums"], test_out["inNums"]
del X_data_out["outNums"], test_out["outNums"]

dtrain = xgb.DMatrix(X_data_out, label=y_data_out)
dtest = xgb.DMatrix(test_out)
dval = xgb.DMatrix(valid, label=y_valid)
watchlist = [(dtrain, "train")]
clf = xgb.train(
    dtrain=dtrain,
    num_boost_round=clf.best_iteration,
    early_stopping_rounds=100,
    evals=watchlist,
    verbose_eval=100,
    params=xgb_params,
)

prediction_out = clf.predict(dtest, ntree_limit=clf.best_iteration)
prediction["outNums"] = prediction_out

prediction["outNums"] = abs(np.round(prediction["outNums"]))
error_test_out = mean_absolute_percentage_error(
    abs(np.round(prediction["outNums"])), y_test_out
)

# 最终得分为总误差平均
print("最终inNums和outNums得分：", (np.mean(error_in) + np.mean(error_out)) / 2)

# 将预测结果存在指定文档里
sub = pd.read_csv(path + "/Metro_testA/testA_submit_2019-01-29.csv")
sub["inNums"] = prediction["inNums"].values
sub["outNums"] = prediction["outNums"].values
sub[["stationID", "startTime", "endTime", "inNums", "outNums"]].to_csv(
    "sub2.csv", index=False
)
