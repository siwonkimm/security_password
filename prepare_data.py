"""데이터 준비 안내
1) 흔한 비밀번호 목록(common_passwords.csv) 정리: 이 파일을 실행
2) 유출 표본 피처 테이블(features_unique.csv): python app.py --build-features dataset.csv
   (원본 dataset.csv의 중복 비밀번호를 제거하고 비밀번호 원문 대신 길이·문자 구성 등 피처만 저장)
"""
import pandas as pd
U = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400, "weeks": 604800, "months": 2629800, "years": 31557600}
p = pd.read_csv("passwords.csv").dropna(subset=["rank"])
p["online_sec"] = p.value * p.time_unit.map(U)
p[["rank", "password", "category", "online_sec", "offline_crack_sec", "strength"]].to_csv("data/common_passwords.csv", index=False)
