# 棉花冠层叶绿素含量（LCC）反演建模流水线

植被指数（VI）与 GLCM 纹理特征融合的对比建模，包含分组交叉验证、多模型超参搜索与 SHAP 可解释性分析。

## 目录结构

```
.
├── data/
│   └── LCC.xlsx              # 样本级特征表，540 行 × 118 列
├── main.py                   # 建模主程序
├── requirements.txt          # 直接依赖，锁定版本
├── requirements-lock.txt     # 完整环境快照（pip freeze）
├── LICENSE                   # MIT
└── README.md
```

## 环境

- Python **3.10.5**（pyenv-win）

```
python -m pip install -r requirements.txt
```

需要完全一致的复现环境时，改用全量锁定文件：

```
python -m pip install -r requirements-lock.txt
```

## 数据

`data/LCC.xlsx` 的 Sheet1：

- `id` — 样点编号
- `Reproductive period` — 生育期
- `Processing` — 水氮处理
- `叶绿素含量(mg/g)` — 目标变量
- 其余 115 列 — 植被指数（NDVI、GNDVI、NDRE、NDVIrededge、SR、RVI、MSR 等）与 GLCM 纹理统计量

## 运行

在仓库根目录执行（`main.py` 中 `DATA_PATH = os.path.join("data", "LCC.xlsx")`，为相对路径）：

```
python main.py
```

产物写入 `Q1/run_<tag>/`（已在 `.gitignore` 中排除）：

- `Fig_*.png` / `.pdf` — 特征重要性、热力图、训练/测试散点、SHAP 汇总与依赖图
- `model_results.xlsx` — 各模型与特征组合的精度对比
- `best_model_predictions.xlsx` — 最优模型预测值
- `best_model_*.joblib` / `.pkl` — 最优模型
- `SHAP_*/` — SHAP 明细

## 流水线要点

- **特征**：VI + 纹理，经 `SelectKBest` / `mutual_info_regression` 筛选
- **预处理**：`ColumnTransformer` + `SimpleImputer` + `PowerTransformer`，类别变量用 `OneHotEncoder`
- **模型**：`PLSRegression`、`ElasticNet`、`RandomForestRegressor`、`ExtraTreesRegressor`、LightGBM
- **调参**：`RandomizedSearchCV` 配 `GroupKFold`，按样点分组切分以避免信息泄漏
- **指标**：R²、RMSE、RPD
- **可解释性**：SHAP

## 许可

代码以 [MIT](LICENSE) 许可发布。
