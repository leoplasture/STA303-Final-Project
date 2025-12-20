import os
import subprocess
import itertools

# 强制 CUDA（如不需要可注释）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------
# 参数网格（清晰、结构化、易扩展）
# -----------------------------------------
param_grid = {
    "lr":        [1e-3, 5e-4, 1e-4],
    "batch":     [32, 64],
    "gamma":     [0.95, 0.99],
}

episodes = 200

results = []


# -----------------------------------------
# 生成所有组合（通用，最优做法）
# -----------------------------------------
def generate_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# -----------------------------------------
# 运行命令并抓取输出
# -----------------------------------------
def run(cmd):
    print("\n🟦 Running:", cmd)
    out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    text = stdout.decode()
    print(text)
    return text


# -----------------------------------------
# 遍历所有参数组合
# -----------------------------------------
for params in generate_combinations(param_grid):

    print("\n===============================")
    print(" Testing hyperparameters:", params)
    print("===============================")

    lr    = params["lr"]
    batch = params["batch"]
    gm    = params["gamma"]

    # 训练
    train_cmd = (
        f"python train_test.py --mode train --algo dqn "
        f"--episodes {episodes} --lr {lr} --batch_size {batch} --gamma {gm}"
    )
    run(train_cmd)

    # 评估
    eval_cmd = (
        f"python train_test.py --mode eval --algo dqn --episodes 10 --render False"
    )
    eval_output = run(eval_cmd)

    # 抓取平均分
    avg_score = None
    for line in eval_output.splitlines():
        if "Average over" in line:
            avg_score = float(line.split(":")[-1])
            break

    if avg_score is None:
        avg_score = 0

    results.append((avg_score, params))


# -----------------------------------------
# 输出最佳参数
# -----------------------------------------
results.sort(key=lambda x: -x[0])
best_score, best_params = results[0]

print("\n============================================")
print(" Optimal Hyperparameters Found")
print("============================================")
print("Score:", best_score)
print("Parameters:", best_params)
print("============================================")