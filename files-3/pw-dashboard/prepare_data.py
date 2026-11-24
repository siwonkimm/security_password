"""원본 CSV -> data/ 정제본. 비밀번호 원문은 저장소에 올리지 않기 위해 지표만 남긴다."""
import pandas as pd
U = {"seconds":1,"minutes":60,"hours":3600,"days":86400,"weeks":604800,"months":2629800,"years":31557600}
SRC = "/mnt/user-data/uploads/"
m = pd.read_csv(SRC+"dataset.csv").drop_duplicates("password")  # 100,000행 -> 고유 25,000개
m = m.drop(columns=["password","crack_time"])
m.to_csv("data/metrics_unique.csv", index=False)
p = pd.read_csv(SRC+"passwords.csv").dropna(subset=["rank"])
p["online_sec"] = p.value * p.time_unit.map(U)
p[["rank","password","category","online_sec","offline_crack_sec","strength"]].to_csv("data/common_passwords.csv", index=False)
pd.read_csv(SRC+"Cybersecurity_Password_Strength_Dataset_1000Rows.csv").to_csv("data/cyber_synth.csv", index=False)
